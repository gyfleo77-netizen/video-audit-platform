馃幀 Video Intelligent Audit Platform
A localized video content moderation system based on the MiniCPM-V 4.5 multimodal large model and Faster-Whisper speech recognition. It supports intelligent scene segmentation, sensitive content detection, speech transcription, and automated generation of graphic analysis reports.

鉁?Core Features
Intelligent Scene Segmentation: Automatically identifies video shot transitions, slicing long videos into independent scene fragments for refined analysis.
Multimodal Understanding: Leverages the powerful visual capabilities of MiniCPM-V 4.5 to identify people, actions, objects, and potential risks (violence, pornography, sensitive political content, etc.) in the footage.
Speech Transcription (ASR): Extracts spoken dialogue from the video to assist in content compliance judgment.
Online Video Download: Built-in yt-dlp support allows direct downloading and analysis via URL input (Bilibili/YouTube).
Visual Reports: Generates detailed reports containing keyframes, timestamps, risk descriptions, and audit recommendations.
鈿狅笍 Read Before Deployment (Essential for Beginners)
Before starting, please ensure your computer meets the following hard requirements, otherwise the program will not run correctly:

1. Hardware Requirements
Graphics Card (GPU): Must be an NVIDIA graphics card.
Recommended VRAM: 12GB or more (for smooth operation of MiniCPM-V 4.5).
Minimum VRAM: 8GB (You must check the "4bit Quantization" mode in the web interface).
Memory (RAM): 16GB or more recommended.
Storage: Reserve at least 20GB of space for model files.
2. System Software (Must Install)
Aside from Python dependencies, you must install the following tools at the system level:

FFmpeg: Used for video decoding and processing. If not installed, videos cannot be read!
Ubuntu/Debian: sudo apt install ffmpeg
MacOS: brew install ffmpeg
Windows: Download the FFmpeg compiled package, unzip it, and add the bin directory to your system environment variable PATH.
馃殌 Quick Deployment Guide
Step 1: Clone Project & Configure Environment
It is recommended to use Conda to create an isolated Python environment:

# 1. Clone the code (Assuming you have already downloaded the code package)
cd video-audit-platform

# 2. Create Python 3.10 environment
conda create -n video_audit python=3.10
conda activate video_audit

# 3. Install project dependencies
# Please ensure you use the latest requirements.txt
pip install -r requirements.txt
Step 2: Download Model Weights (Crucial Step)
The project loads models locally by default to avoid downloading them every time. You need to manually download the models and place them according to the following structure:

Create a models folder:

mkdir models
Download MiniCPM-V 4.5:

Download Source: Hugging Face (Note: Check for v4.5 specifically) or ModelScope (Recommended for mainland China).
Place all downloaded files into the models/MiniCPM-V-4_5 directory.
Download Faster-Whisper:

Download Source: Hugging Face.
Place the downloaded files into the models/faster-whisper-base directory.
Final Directory Structure:

video-audit-platform/
├── appp_api.py
├── index2.html
├── start_server2.sh
├── requirements.txt
├── RENAMED.md
└── backend/
    ├── __init__.py
    ├── config.py
    ├── runtime.py
    ├── video_utils.py
    └── asr_utils.py
Step 3: Start the Service
We provide a one-click startup script that automatically handles environment variables and background processes.

Linux / Mac Users:

# Add execution permission (required for the first run)
chmod +x start_server.sh

# Start the service
./start_server.sh
Windows Users:
Please manually run the following in two separate terminal windows:

Backend:
python appp_api.py
2.Frontend:

python -m http.server 8000
Then open your browser and visit: http://localhost:8000

馃摉 User Manual
Once started successfully, the browser should automatically open http://localhost:8000.

1. Video Upload & Analysis
Local Upload: Click "Video Input" -> "Local Video Upload", and drag in an MP4/MOV file.
URL Download: Enter the video URL (e.g., Bilibili) in the "Online Video Download" box and click download. Once finished, it will automatically populate the analysis list.
2. Parameter Configuration (Left Sidebar)
Scene Detection Threshold: Default 27.0. If scenes are too fragmented (too many scenes), increase this value (e.g., 30-35); if transitions are missed, decrease it.
Frame Extraction FPS: Default 1.5. Takes 1.5 screenshots per second for AI analysis. Setting this too high will significantly increase analysis time.
Use 4bit Quantization: Users with less than 12GB VRAM must check this, otherwise you may encounter Out Of Memory (OOM) errors.
3. View Reports
Click "Start Intelligent Analysis" and wait for the progress bar. Upon completion:

View the comprehensive summary and risk rating in the "Analysis Results" tab.
Click "Scene Detailed Analysis" to see the visual description and corresponding speech text for each segment.
Supports exporting reports as JSON or TXT.
鉂?FAQ
Q1: Error FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'

A: System FFmpeg is not installed. Please refer to the "Read Before Deployment" section to install it, and ensure you can run ffmpeg -version directly in your terminal.

Q2: Error CUDA out of memory

A: Video memory (VRAM) is insufficient.

Check "Use 4bit Quantization" on the left sidebar.

Lower "Frame Extraction FPS" to 0.5 or 1.0.

Ensure no other programs are using the GPU.

Q3: Why is the analysis speed slow?

A: Video analysis is a compute-intensive task. Speed depends on your GPU performance, video length, and frame extraction density. Using a stronger graphics card (e.g., RTX 3090/4090) will significantly increase speed.

Q4: Startup script error OSError: ... libcudnn_cnn_infer.so.8

A: This is a common issue in Linux environments. start_server.sh has built-in auto-repair logic. Please ensure you use this script to start the service rather than running the Python command directly.

馃摐 License
This project is licensed under the Apache 2.0 License. The model weights used follow their respective open-source agreements:

MiniCPM-V: OpenBMB License
Faster-Whisper: MIT License Happy Auditing! 馃帀
