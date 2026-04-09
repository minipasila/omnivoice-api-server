import torch
import torchaudio
import os
import time 
import re
import json
import socket
import io
import wave
import numpy as np
import gc

from loguru import logger
from datetime import datetime
from pathlib import Path

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# Class to check tts settings
class InvalidSettingsError(Exception):
    pass

# List of supported language codes (Retained for API compatibility)
supported_languages = {
    "ab":"Abkhazian", "acw":"Hijazi Arabic", "acx":"Omani Arabic", "adx":"Amdo Tibetan",
    "ady":"Adyghe", "aeb":"Tunisian Arabic", "afb":"Gulf Arabic", "arb":"Standard Arabic",
    "ars":"Najdi Arabic", "ary":"Moroccan Arabic", "arz":"Egyptian Arabic", "as":"Assamese",
    "ayl":"Libyan Arabic", "ba":"Bashkir", "be":"Belarusian", "bg":"Bulgarian",
    "bn":"Bengali", "bo":"Tibetan", "br":"Breton", "brx":"Bodo",
    "bs":"Bosnian", "ca":"Catalan", "ckb":"Central Kurdish", "cs":"Czech",
    "cv":"Chuvash", "cy":"Welsh", "da":"Danish", "de":"German",
    "dgo":"Dogri", "dv":"Dhivehi", "el":"Greek", "en":"English",
    "eo":"Esperanto", "es":"Spanish", "et":"Estonian", "eu":"Basque",
    "fa":"Persian", "fi":"Finnish", "fr":"French", "fue":"Borgu Fulfulde",
    "fy":"Western Frisian", "ga":"Irish", "gjk":"Kachi Koli", "gl":"Galician",
    "gu":"Gujarati", "gui":"Eastern Bolivian Guaraní", "hi":"Hindi", "hno":"Northern Hindko",
    "hr":"Croatian", "hu":"Hungarian", "hy":"Armenian", "id":"Indonesian",
    "is":"Icelandic", "it":"Italian", "ja":"Japanese", "ka":"Georgian",
    "kab":"Kabyle", "kbd":"Kabardian", "kk":"Kazakh", "kln":"Kalenjin",
    "kmr":"Northern Kurdish", "kn":"Kannada", "knn":"Konkani", "ko":"Korean",
    "ks":"Kashmiri", "kxp":"Wadiyara Koli", "ky":"Kirghiz", "lg":"Ganda",
    "lt":"Lithuanian", "ltg":"Latgalian", "luo":"Luo", "lus":"Lushai",
    "lv":"Latvian", "mai":"Maithili", "mhr":"Eastern Mari", "mk":"Macedonian",
    "ml":"Malayalam", "mn":"Mongolian", "mni":"Manipuri", "mr":"Marathi",
    "mrj":"Western Mari", "mt":"Maltese", "mvy":"Indus Kohistani", "nl":"Dutch",
    "no":"Norwegian", "npi":"Nepali", "odk":"Od", "orc":"Orma",
    "ory":"Odia", "pa":"Panjabi", "phl":"Phalura", "phr":"Pahari-Potwari",
    "pl":"Polish", "ps":"Pushto", "pt":"Portuguese", "ro":"Romanian",
    "ru":"Russian", "rw":"Kinyarwanda", "sa":"Sanskrit", "sat":"Santali",
    "sd":"Sindhi", "sk":"Slovak", "sl":"Slovenian", "sr":"Serbian",
    "sv":"Swedish", "sw":"Swahili", "ta":"Tamil", "te":"Telugu",
    "th":"Thai", "tr":"Turkish", "tt":"Tatar", "ug":"Uighur",
    "uk":"Ukrainian", "ur":"Urdu", "uz":"Uzbek", "vi":"Vietnamese",
    "yue":"Cantonese", "zh":"Chinese"
}

default_tts_settings = {
    "temperature" : 0.75,
    "length_penalty" : 1.0,
    "repetition_penalty": 5.0,
    "top_k" : 50,
    "top_p" : 0.85,
    "speed" : 1,
    "enable_text_splitting": True
}

reversed_supported_languages = {name: code for code, name in supported_languages.items()}

class TTSWrapper:
    def __init__(self, output_folder="./output", speaker_folder="./speakers", model_folder="./models", lowvram=False, model_source="local", model_version="k2-fsa/OmniVoice", device="cuda", deepspeed=False, enable_cache_results=True, load_asr=True):
        self.load_asr = load_asr
        self.cuda = device
        self.device = 'cpu' if lowvram else (self.cuda if torch.cuda.is_available() else "cpu")
        self.lowvram = lowvram

        self.model_source = model_source
        self.model_version = model_version
        # --- ADD THIS SAFEGUARD ---
        if self.model_version in ["v2.0.0", "v2.0.1", "v2.0.2", "v2.0.3", "main"]:
            logger.warning(f"Legacy XTTS model version '{self.model_version}' detected. Overriding to 'k2-fsa/OmniVoice'.")
            self.model_version = "k2-fsa/OmniVoice"
        # --------------------------
        self.tts_settings = default_tts_settings
        self.stream_chunk_size = 100

        self.deepspeed = deepspeed

        self.speaker_folder = speaker_folder
        self.output_folder = output_folder
        self.model_folder = model_folder

        self.create_directories()

        self.enable_cache_results = enable_cache_results
        self.cache_file_path = os.path.join(output_folder, "cache.json")

        if self.enable_cache_results:
            with open(self.cache_file_path, 'w') as cache_file:
                json.dump({}, cache_file)

    def check_model_version_old_format(self, model_version):
        return model_version # OmniVoice uses full huggingface names

    def get_models_list(self):
        entries = os.listdir(self.model_folder)
        models = [name for name in entries if os.path.isdir(os.path.join(self.model_folder, name))]
        if "k2-fsa/OmniVoice" not in models:
            models.append("k2-fsa/OmniVoice")
        return models

    def get_wav_header(self, channels:int=1, sample_rate:int=24000, width:int=2) -> bytes:
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(sample_rate)
            out.writeframes(b"")
        wav_buf.seek(0)
        return wav_buf.read()

    def check_cache(self, text_params):
        if not self.enable_cache_results:
            return None
        try:
            with open(self.cache_file_path) as cache_file:
                cache_data = json.load(cache_file)
            for entry in cache_data.values():
                if all(entry[key] == value for key, value in text_params.items()):
                    return entry['file_name']
            return None
        except FileNotFoundError:
            return None

    def update_cache(self, text_params, file_name):
        if not self.enable_cache_results:
            return None
        try:
            if os.path.exists(self.cache_file_path) and os.path.getsize(self.cache_file_path) > 0:
                with open(self.cache_file_path, 'r') as cache_file:
                    cache_data = json.load(cache_file)
            else:
                cache_data = {}

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            cache_data[timestamp] = {**text_params, 'file_name': file_name}

            with open(self.cache_file_path, 'w') as cache_file:
                json.dump(cache_data, cache_file)

        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    def load_model(self, load=True):
        logger.info(f"Loading OmniVoice model '{self.model_version}' to {self.device} (ASR: {self.load_asr})...")
        dtype = torch.float16 if "cuda" in self.device else torch.float32

        # If we really want to be sure Whisper doesn't touch anything:
        if not self.load_asr:
            os.environ["OMNIVOICE_SKIP_ASR"] = "1" # Some versions check this

        self.model = OmniVoice.from_pretrained(
            self.model_version,
            device_map=self.device,
            dtype=dtype,
            load_asr=self.load_asr
        )
        logger.info("OmniVoice model successfully loaded!")

    def load_local_model(self, load=True):
        self.load_model(load)

    def switch_model(self, model_name):
        logger.info(f"Switching model to {model_name}")
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model_version = model_name
        self.load_model()

    def switch_model_device(self):
        pass # OmniVoice handles its own memory/chunking thresholds natively

    def create_directories(self):
        for sanctuary in [self.output_folder, self.speaker_folder, self.model_folder]:
            absolute_path = os.path.abspath(os.path.normpath(sanctuary))
            if not os.path.exists(absolute_path):
                os.makedirs(absolute_path)

    def set_speaker_folder(self, folder):
        self.speaker_folder = folder
        self.create_directories()

    def set_out_folder(self, folder):
        self.output_folder = folder
        self.create_directories()

    def set_tts_settings(self, temperature, speed, length_penalty, repetition_penalty, top_p, top_k, enable_text_splitting, stream_chunk_size):
        self.tts_settings = {
            "temperature": temperature,
            "speed": speed,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
            "enable_text_splitting": enable_text_splitting,
        }
        self.stream_chunk_size = stream_chunk_size

    def get_wav_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.wav')]

    def _get_speakers(self):
        speakers = []
        for f in os.listdir(self.speaker_folder):
            full_path = os.path.join(self.speaker_folder,f)
            if os.path.isdir(full_path):
                subdir_files = self.get_wav_files(full_path) 
                if not subdir_files: continue
                speaker_wav = [os.path.join(full_path, s) for s in subdir_files]
                speakers.append({
                    'speaker_name': f,
                    'speaker_wav': speaker_wav,
                    'preview': os.path.join(f, subdir_files[0])
                })
            elif f.endswith('.wav'):
                speaker_name = os.path.splitext(f)[0]
                speakers.append({
                    'speaker_name': speaker_name,
                    'speaker_wav': full_path,
                    'preview': f
                })
        return speakers

    def get_speakers(self):
        return [ s['speaker_name'] for s in self._get_speakers() ] 

    def get_local_ip(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('10.255.255.255', 1))
                return s.getsockname()[0] 
        except Exception:
            return None

    def get_speakers_special(self):
        BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
        TUNNEL_URL = os.getenv('TUNNEL_URL', '')
        speakers_special = []

        for speaker in self._get_speakers():
            preview_url = f"{TUNNEL_URL if TUNNEL_URL else BASE_URL}/sample/{speaker['preview']}"
            speakers_special.append({
                    'name': speaker['speaker_name'],
                    'voice_id': speaker['speaker_name'],
                    'preview_url': preview_url
            })
        return speakers_special

    def list_languages(self):
        return reversed_supported_languages

    def clean_text(self, text):
        text = re.sub(r'[\*\r\n]', '', text)
        text = re.sub(r'"\s?(.*?)\s?"', r"'\1'", text)
        return text

    @torch.inference_mode()
    async def stream_generation(self, text, speaker_wav_path, language, output_file, transcript=None):
        generate_start_time = time.time()

        ref_text = transcript
        if ref_text is None and not self.load_asr:
            ref_text = ""

        try:
            pos_temp = self.tts_settings.get("temperature", 0.75) * 6.5
            gen_config = OmniVoiceGenerationConfig(
                position_temperature=pos_temp,
                preprocess_prompt=True,
                postprocess_output=True
            )

            audio = self.model.generate(
                text=text,
                ref_audio=speaker_wav_path,
                ref_text=ref_text,
                speed=self.tts_settings.get("speed", 1.0),
                generation_config=gen_config
            )

            waveform = audio[0].squeeze(0).cpu().numpy()
            waveform = np.clip(waveform, -1.0, 1.0)
            waveform = (waveform * 32767).astype(np.int16)

            torchaudio.save(output_file, audio[0].cpu(), 24000)

            wav_bytes = waveform.tobytes()
            chunk_size = 4096
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i:i+chunk_size]

        finally:
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(f"Stream generation processing time: {time.time() - generate_start_time:.2f} s")

    @torch.inference_mode()
    def local_generation(self, text, speaker_wav_path, language, output_file, transcript=None):
        generate_start_time = time.time()

        # Handle ASR skip logic: if no transcript and ASR is off, send empty string to trick it
        ref_text = transcript
        if ref_text is None and not self.load_asr:
            logger.warning("ASR disabled and no .txt found. Using empty ref_text to skip Whisper.")
            ref_text = ""

        try:
            pos_temp = self.tts_settings.get("temperature", 0.75) * 6.5
            gen_config = OmniVoiceGenerationConfig(
                position_temperature=pos_temp,
                preprocess_prompt=True,
                postprocess_output=True
            )

            # Pass ref_audio and ref_text directly into generate() as per README
            audio = self.model.generate(
                text=text,
                ref_audio=speaker_wav_path,
                ref_text=ref_text,
                speed=self.tts_settings.get("speed", 1.0),
                generation_config=gen_config
            )
            torchaudio.save(output_file, audio[0].cpu(), 24000)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(f"Generation processing time: {time.time() - generate_start_time:.2f} s")

    def api_generation(self, text, speaker_wav, language, output_file):
        self.local_generation(text, speaker_wav, language, output_file)

    def get_speaker_wav(self, speaker_name_or_path):
        if speaker_name_or_path.endswith('.wav'):
            speaker_wav_path = speaker_name_or_path if os.path.isabs(speaker_name_or_path) else os.path.join(self.speaker_folder, speaker_name_or_path)
        else:
            full_path = os.path.join(self.speaker_folder, speaker_name_or_path)
            wav_file = f"{full_path}.wav"
            if os.path.isdir(full_path):
                subdir_wavs = self.get_wav_files(full_path)
                if not subdir_wavs: raise ValueError(f"no wav files found in {full_path}")
                speaker_wav_path = os.path.join(full_path, subdir_wavs[0])
            elif os.path.isfile(wav_file):
                speaker_wav_path = wav_file
            else:
                raise ValueError(f"Speaker {speaker_name_or_path} not found.")

        # Look for a transcript file (.txt) with the same name as the .wav
        transcript = None
        transcript_path = speaker_wav_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                logger.info(f"Found transcript for speaker: {transcript_path}")
            except Exception as e:
                logger.error(f"Failed to read transcript: {e}")

        # Now we return a tuple: (path_to_wav, transcript_text_or_None)
        return speaker_wav_path, transcript

    def process_tts_to_file(self, text, speaker_name_or_path, language, file_name_or_path="out.wav", stream=False):
        # UNPACK BOTH HERE:
        speaker_wav_path, transcript = self.get_speaker_wav(speaker_name_or_path)

        output_file = file_name_or_path if os.path.isabs(file_name_or_path) else os.path.join(self.output_folder, file_name_or_path)

        if os.path.isfile(text) and text.lower().endswith('.txt'):
            with open(text, 'r', encoding='utf-8') as f: text = f.read()

        if self.enable_cache_results:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = os.path.join(self.output_folder, f"{timestamp}_cache_{file_name_or_path}")

        clear_text = self.clean_text(text)
        text_params = {'text': clear_text, 'speaker_name_or_path': speaker_name_or_path, 'language': language}
        cached_result = self.check_cache(text_params)

        if cached_result:
            logger.info("Using cached result.")
            return cached_result

        if stream:
            async def stream_fn():
                # PASS transcript HERE:
                async for chunk in self.stream_generation(clear_text, speaker_wav_path, language, output_file, transcript=transcript):
                    yield chunk
                self.update_cache(text_params, output_file)
            return stream_fn()
        else:
            # PASS transcript HERE:
            self.local_generation(clear_text, speaker_wav_path, language, output_file, transcript=transcript)

        self.update_cache(text_params, output_file)
        return output_file