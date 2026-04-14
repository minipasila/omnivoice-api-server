# OmniVoice API Server (XTTS-Compatible)

A simple FastAPI Server to run [OmniVoice](https://github.com/k2-fsa/OmniVoice) as a drop-in replacement for XTTSv2. 

This project is a fork of the original [xtts-api-server](https://github.com/daswer123/xtts-api-server), heavily modified to use the state-of-the-art **OmniVoice** engine while maintaining **100% API compatibility** with clients like [SillyTavern](https://github.com/SillyTavern/SillyTavern). 

By using this server, you get the superior zero-shot voice cloning and fast inference of OmniVoice, without needing to change any of your frontend client settings!

There's a **Google Colab version** included in the repository (`OmniVoice-api-server.ipynb`) you can use if your computer does not have a dedicated GPU. (Also there's one that specifically works for Kaggle `OmniVoice-api-server-Kaggle.ipynb`)

## Key Features vs Original XTTS
* **OmniVoice Backend:** Uses `k2-fsa/OmniVoice`, supporting 600+ languages and offering better zero-shot voice cloning. (Currently the code supports 118 languages that have a minimum of 20 hours of training data)
* **Speaker Transcripts:** Supports providing a `.txt` file with the same name as the speaker `.wav` to skip ASR and provide the ground truth text, ensuring even more accurate cloning.
* **API Compatible:** Your existing SillyTavern (or other frontend) settings will still work. XTTS generation parameters (like temperature and speed) are automatically translated into OmniVoice equivalents.
* **Optimized VRAM:** Includes automatic PyTorch garbage collection and inference-mode optimizations to prevent Out-Of-Memory (OOM) errors over long sessions.
* **Fast Downloading:** Uses `hf_transfer` for high-speed model weight downloads from HuggingFace.

## Installation

Since the package relies on the latest GitHub version of OmniVoice, it is recommended to install directly from the repository.

**Step 1: Install PyTorch**
We recommend installing the **GPU version** to ensure fast processing.
```bash
# Windows / Linux (CUDA 12.1 or 12.4 recommended)
pip install torch>=2.4 torchaudio>=2.4 --index-url https://download.pytorch.org/whl/cu121
```

**Step 2: Install the Server**
```bash
git clone https://github.com/minipasila/omnivoice-api-server.git
cd omnivoice-api-server
pip install .
```

## Use Docker with Docker Compose

A Dockerfile is provided to build a Docker image, and a `docker-compose.yml` file is provided to run the server.

```bash
git clone https://github.com/minipasila/omnivoice-api-server.git
cd omnivoice-api-server/docker
docker compose up -d
```

## Starting Server

To maintain compatibility with old launcher scripts and frontend clients, the execution module remains named `xtts_api_server`.

`python -m xtts_api_server` will run on default ip and port (localhost:8020)

```text
usage: xtts_api_server [-h] [-hs HOST] [-p PORT] [-d DEVICE] [-sf SPEAKER_FOLDER] [-o OUTPUT] [-mf MODEL_FOLDER] [-t TUNNEL_URL] [-ms MODEL_SOURCE] [-v MODEL_VERSION] [--listen] [--lowvram] [--deepspeed] [--use-cache] [--streaming-mode] [--streaming-mode-improve] [--stream-play-sync] [--no-asr]

Run OmniVoice within an XTTS-compatible FastAPI application

options:
  -h, --help            show this help message and exit
  -hs HOST, --host HOST
  -p PORT, --port PORT
  -d DEVICE, --device DEVICE
                        `cpu` or `cuda`, you can specify which video card to use, for example, `cuda:0`
  -sf SPEAKER_FOLDER, --speaker-folder SPEAKER_FOLDER
                        The folder where you get the samples for tts
  -o OUTPUT, --output OUTPUT
                        Output folder
  -mf MODEL_FOLDER, --model-folder MODEL_FOLDER
                        Folder where models will be stored
  -t TUNNEL_URL, --tunnel TUNNEL_URL
                        URL of tunnel used (e.g: ngrok, localtunnel)
  -ms MODEL_SOURCE, --model-source {api,apiManual,local}
                        Kept for compatibility
  -v MODEL_VERSION, --version MODEL_VERSION
                        Defaults to "k2-fsa/OmniVoice". Will download automatically from HuggingFace.
  --listen              Allows the server to be used outside the local computer, similar to -hs 0.0.0.0
  --lowvram             Force the model onto the CPU.
  --deepspeed           (Ignored: Kept so old launcher scripts don't break)
  --use-cache           Enables caching of results. If there is a repeated request, you will get a file instead of generating from scratch.
  --streaming-mode      Enables local audio playback stream simulation.
  --streaming-mode-improve
                        Includes an improved streaming mode.
  --stream-play-sync    Additional flag for streaming mode.
  --no-asr              Disable loading the ASR (Whisper) model to save VRAM/bandwidth. Recommended if you provide transcripts for all speakers.
```

If you want your host to listen on your local network, use `-hs 0.0.0.0` or use `--listen`.

The `-t` or `--tunnel` flag is needed so that when you fetch speakers via GET request, you get the correct link to hear the preview in your frontend.

The first time you run the server, it will automatically download the OmniVoice model weights (~4-5GB) from HuggingFace.

## API Docs

API Docs can be accessed from [http://localhost:8020/docs](http://localhost:8020/docs)

## How to add a speaker

By default, a `speakers` folder will appear in the directory. You need to put `.wav` files with your voice samples there. You can also create a subfolder and put several voice samples of the same speaker inside; the server will automatically use the first one as a preview.

### Using Ground Truth Transcripts (Recommended)
OmniVoice uses Whisper (ASR) by default to understand what the speaker is saying in the reference clip. For even better results, you can provide the exact text for each clip:
1. Put a `.txt` file with the **exact same name** as your `.wav` file in the same folder (e.g., `speaker1.wav` and `speaker1.txt`).
2. Put the text content inside the `.txt` file.
3. If you do this for all speakers, you can run the server with `--no-asr` to save about 2–3GB of VRAM and improve startup speed!

## Note on creating samples for quality voice cloning

OmniVoice is highly accurate, meaning it will clone background noise if your sample has it. 
* Keep clips about 3–10 seconds long.
* Ensure the audio is down-sampled to a Mono, 24000Hz 16 Bit wav file. 
* Ensure the clip doesn't have background noises, hissing, or music. Clean audio is key!
* Make sure the clip doesn't start or end with breathy sounds (breathing in/out etc).

## Credit

1. Massive thanks to the **[k2-fsa team](https://github.com/k2-fsa/OmniVoice)** for developing the incredible OmniVoice engine.
2. Thanks to **[daswer123](https://github.com/daswer123/xtts-api-server)** for the original `xtts-api-server` architecture. 
3. Thanks to the author **Kolja Beigel** for the repository [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS), whose code inspired much of the original streaming architecture.