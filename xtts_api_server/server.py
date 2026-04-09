from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import wave
import threading
from pathlib import Path
from uuid import uuid4
from loguru import logger

import pyaudio

from xtts_api_server.tts_funcs import TTSWrapper, supported_languages, InvalidSettingsError

# Default Folders
DEVICE = os.getenv('DEVICE',"cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
MODEL_FOLDER = os.getenv('MODEL', 'models')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_VERSION = os.getenv("MODEL_VERSION", "k2-fsa/OmniVoice")  # Defaulted to OmniVoice
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
USE_CACHE = os.getenv("USE_CACHE") == 'true'
LOAD_ASR = os.getenv("LOAD_ASR", "true") == "true"

# STREAMING VARS
STREAM_MODE = os.getenv("STREAM_MODE") == 'true'
STREAM_MODE_IMPROVE = os.getenv("STREAM_MODE_IMPROVE") == 'true'
STREAM_PLAY_SYNC = os.getenv("STREAM_PLAY_SYNC") == 'true'

# Initialize OmniVoice API wrapper
app = FastAPI()
XTTS = TTSWrapper(OUTPUT_FOLDER, SPEAKER_FOLDER, MODEL_FOLDER, LOWVRAM_MODE, MODEL_SOURCE, MODEL_VERSION, DEVICE, DEEPSPEED, USE_CACHE, LOAD_ASR)

XTTS.load_model() 

if USE_CACHE:
    logger.info("Caching is enabled. Repeat requests will return pre-generated files.")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def play_audio_file(file_path):
    """Local playback mechanism replacing Coqui RealtimeTTS engine"""
    try:
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        logger.error(f"Failed to play audio locally: {e}")

class OutputFolderRequest(BaseModel):
    output_folder: str

class SpeakerFolderRequest(BaseModel):
    speaker_folder: str

class ModelNameRequest(BaseModel):
    model_name: str

class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str 
    language: str

class SynthesisFileRequest(BaseModel):
    text: str
    speaker_wav: str 
    language: str
    file_name_or_path: str  

@app.get("/speakers_list")
def get_speakers():
    return XTTS.get_speakers()

@app.get("/speakers")
def get_speakers_special():
    return XTTS.get_speakers_special()

@app.get("/languages")
def get_languages():
    return {"languages": XTTS.list_languages()}

@app.get("/get_folders")
def get_folders():
    return {
        "speaker_folder": XTTS.speaker_folder, 
        "output_folder": XTTS.output_folder,
        "model_folder": XTTS.model_folder
    }

@app.get("/get_models_list")
def get_models_list():
    return XTTS.get_models_list()

@app.get("/get_tts_settings")
def get_tts_settings():
    return {**XTTS.tts_settings, "stream_chunk_size": XTTS.stream_chunk_size}

@app.get("/sample/{file_name:path}")
def get_sample(file_name: str):
    if ".." in file_name:
        raise HTTPException(status_code=404, detail="Path traversal detected.") 
    file_path = os.path.join(XTTS.speaker_folder, file_name)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/set_output")
def set_output(output_req: OutputFolderRequest):
    XTTS.set_out_folder(output_req.output_folder)
    return {"message": f"Output folder set to {output_req.output_folder}"}

@app.post("/set_speaker_folder")
def set_speaker_folder(speaker_req: SpeakerFolderRequest):
    XTTS.set_speaker_folder(speaker_req.speaker_folder)
    return {"message": f"Speaker folder set to {speaker_req.speaker_folder}"}

@app.post("/switch_model")
def switch_model(modelReq: ModelNameRequest):
    XTTS.switch_model(modelReq.model_name)
    return {"message": f"Model switched to {modelReq.model_name}"}

@app.post("/set_tts_settings")
def set_tts_settings_endpoint(tts_settings_req: TTSSettingsRequest):
    XTTS.set_tts_settings(**tts_settings_req.dict())
    return {"message": "Settings successfully applied"}

@app.get('/tts_stream')
async def tts_stream(request: Request, text: str = Query(), speaker_wav: str = Query(), language: str = Query()):
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400, detail="Language unsupported.")
            
    async def generator():
        chunks = XTTS.process_tts_to_file(
            text=text,
            speaker_name_or_path=speaker_wav,
            language=language.lower(),
            stream=True,
        )
        yield XTTS.get_wav_header()
        async for chunk in await chunks: # Await the stream generator function returned by process_tts_to_file
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(generator(), media_type='audio/x-wav')

@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    try:
        if request.language.lower() not in supported_languages:
            raise HTTPException(status_code=400, detail="Language unsupported.")

        output_file_path = XTTS.process_tts_to_file(
            text=request.text,
            speaker_name_or_path=request.speaker_wav,
            language=request.language.lower(),
            file_name_or_path=f'{str(uuid4())}.wav'
        )

        if STREAM_MODE or STREAM_MODE_IMPROVE:
            if STREAM_PLAY_SYNC:
                play_audio_file(output_file_path)
            else:
                threading.Thread(target=play_audio_file, args=(output_file_path,)).start()

            # Create 1s of silence for clients like SillyTavern expecting immediate HTTP return
            this_dir = Path(__file__).parent.resolve()
            output = this_dir / "silence.wav"
            if not output.exists():
                with wave.open(str(output), 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(24000)
                    f.writeframes(b'\x00' * 24000)

            return FileResponse(path=output, media_type='audio/wav', filename="silence.wav")
        else:
            if not XTTS.enable_cache_results:
                background_tasks.add_task(os.unlink, output_file_path)
            return FileResponse(path=output_file_path, media_type='audio/wav', filename="output.wav")

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/tts_to_file")
async def tts_to_file(request: SynthesisFileRequest):
    try:
        if request.language.lower() not in supported_languages:
             raise HTTPException(status_code=400, detail="Language unsupported.")

        output_file = XTTS.process_tts_to_file(
            text=request.text,
            speaker_name_or_path=request.speaker_wav,
            language=request.language.lower(),
            file_name_or_path=request.file_name_or_path 
        )
        return {"message": "The audio was successfully made and stored.", "output_path": output_file}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")