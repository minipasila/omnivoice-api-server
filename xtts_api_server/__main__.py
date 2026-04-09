import uvicorn
from argparse import ArgumentParser
import os

parser = ArgumentParser(description="Run the Uvicorn server.")
parser.add_argument("-hs", "--host", default="localhost", help="Host to bind")
parser.add_argument("-p", "--port", default=8020, type=int, help="Port to bind")
parser.add_argument("-d", "--device", default="cuda", type=str, help="Device that will be used, you can choose cpu or cuda")
parser.add_argument("-sf", "--speaker-folder", default="speakers/", type=str, help="The folder where you get the samples for tts")
parser.add_argument("-o", "--output", default="output/", type=str, help="Output folder")
parser.add_argument("-t", "--tunnel", default="", type=str, help="URL of tunnel used (e.g: ngrok, localtunnel)")
parser.add_argument("-mf", "--model-folder", default="models/", type=str, help="The place where models for XTTS will be stored.")
parser.add_argument("-ms", "--model-source", default="local", choices=["api","apiManual", "local"],
                    help="Kept for backwards compatibility.")
parser.add_argument("-v", "--version", default="k2-fsa/OmniVoice", type=str, help="Model repository ID (defaults to k2-fsa/OmniVoice)")
parser.add_argument("--listen", action='store_true', help="Allows the server to be used outside the local computer, similar to -hs 0.0.0.0")
parser.add_argument("--lowvram", action='store_true', help="Enable low vram mode (Handled natively by OmniVoice).")
parser.add_argument("--deepspeed", action='store_true', help="Enables deepspeed mode (Kept for compatibility).")
parser.add_argument("--use-cache", action='store_true', help="Enables caching of results.")
parser.add_argument("--streaming-mode", action='store_true', help="Enables streaming mode.")
parser.add_argument("--streaming-mode-improve", action='store_true', help="Includes an improved streaming mode.")
parser.add_argument("--stream-play-sync", action='store_true', help="Additional flag for streaming mode.")
parser.add_argument("--no-asr", action='store_true', help="Disable loading the ASR (Whisper) model to save VRAM/bandwidth.")

args = parser.parse_args()

os.environ["LISTEN"] = str(args.listen).lower()
host_ip = "0.0.0.0" if args.listen else args.host

os.environ['DEVICE'] = args.device  
os.environ['OUTPUT'] = args.output  
os.environ['SPEAKER'] = args.speaker_folder 
os.environ['MODEL'] = args.model_folder 
os.environ['BASE_HOST'] = host_ip  
os.environ['BASE_PORT'] = str(args.port)  
os.environ['BASE_URL'] = "http://" + host_ip + ":" + str(args.port) 
os.environ['TUNNEL_URL'] = args.tunnel  
os.environ['MODEL_SOURCE'] = args.model_source  
os.environ["MODEL_VERSION"] = args.version 
os.environ["USE_CACHE"] = str(args.use_cache).lower() 
os.environ["DEEPSPEED"] = str(args.deepspeed).lower() 
os.environ["LOWVRAM_MODE"] = str(args.lowvram).lower() 
os.environ["STREAM_MODE"] = str(args.streaming_mode).lower() 
os.environ["STREAM_MODE_IMPROVE"] = str(args.streaming_mode_improve).lower() 
os.environ["STREAM_PLAY_SYNC"] = str(args.stream_play_sync).lower() 
os.environ["LOAD_ASR"] = str(not args.no_asr).lower()

from xtts_api_server.server import app

if __name__ == "__main__":
    uvicorn.run(app, host=host_ip, port=args.port)