import os

DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "downloaded_videos")
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

SQLALCHEMY_DATABASE_URI = "sqlite:///project.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False

MODEL_PATH_DEFAULT = os.getenv("MINICPMV_PATH", "./models/MiniCPM-V-4_5")
ASR_MODEL_DEFAULT = os.getenv("WHISPER_MODEL", "./models/faster-whisper-base")
ASR_COMPUTE_TYPE_DEFAULT = "int8_float16"

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5-7b-instruct")
