# app_api.py

import os
import tempfile
import json
import subprocess
from typing import List, Tuple
# --- 新增：导入 datetime 用于记录时间戳 ---
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
# --- 新增：导入 SQLAlchemy ---
from flask_sqlalchemy import SQLAlchemy

# --- 你的原始脚本中的所有 import ---
import torch
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from faster_whisper import WhisperModel
import numpy as np
from PIL import Image
import yt_dlp

# --- 初始化 Flask 应用 ---
app = Flask(__name__)
CORS(app)  # 允许跨域请求，让前端可以访问
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'downloaded_videos')
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# ================== 数据库配置 (新增部分) ==================
# 设置数据库文件的路径。这个 'project.db' 文件就是您想要的那个静态、持久化的文件。
# 它会自动生成在项目目录下的一个名为 'instance' 的文件夹内。
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化 SQLAlchemy
db = SQLAlchemy(app)

# 定义数据库模型 (AnalysisLog 数据表)
class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 每条记录的唯一ID
    user_ip = db.Column(db.String(45), nullable=False)  # 记录用户IP
    video_filename = db.Column(db.String(255), nullable=False) # 记录原始视频文件名
    has_violations = db.Column(db.Boolean, default=False, nullable=False) # 记录是否有违规内容 (True/False)
    report_summary = db.Column(db.Text, nullable=True) # 存储风险部分的摘要，方便快速预览
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) # 记录调用的时间

    def to_dict(self):
        """辅助函数：将模型对象转换为字典，方便返回JSON"""
        return {
            "id": self.id,
            "user_ip": self.user_ip,
            "video_filename": self.video_filename,
            "has_violations": self.has_violations,
            "report_summary": self.report_summary,
            "timestamp": self.timestamp.isoformat() + "Z" # 使用ISO格式并加上Z表示UTC时间
        }


# ================== 全局默认参数 (无变化) ==================
MODEL_PATH_DEFAULT = os.getenv("MINICPMV_PATH", "./models/MiniCPM-V-4_5")
ASR_MODEL_DEFAULT = os.getenv("WHISPER_MODEL", "./models/faster-whisper-base")
ASR_COMPUTE_TYPE_DEFAULT = "int8_float16"

# ================== 全局模型加载 (无变化) ==================
print("服务器启动中... 正在加载模型，请稍候...")
# ================== 新增一个辅助函数 ==================
def analyze_scene_in_chunks(model, tokenizer, scene_data, base_prompt: str, max_frames_per_chunk: int = 30):
    """
    分析一个可能很长的场景，如果帧数超过阈值，则分块处理。
    
    :param scene_data: 单个场景的数据，包含 'frames' 和 'temporal_ids'
    :param base_prompt: 针对该场景的基础提示词
    :param max_frames_per_chunk: 每个分块的最大帧数
    :return: 该场景的完整分析文本
    """
    frames = scene_data["frames"]
    temporal_ids = scene_data["temporal_ids"][0] # temporal_ids 是一个嵌套列表
    
    if len(frames) <= max_frames_per_chunk:
        # 如果帧数没有超过阈值，直接调用原始的 call_chat
        return call_chat(model, tokenizer, frames, [temporal_ids], base_prompt)

    print(f"场景帧数 ({len(frames)}) 超过阈值 ({max_frames_per_chunk})，开始分块处理...")
    
    chunk_analyses = []
    num_chunks = (len(frames) + max_frames_per_chunk - 1) // max_frames_per_chunk
    
    for i in range(num_chunks):
        start_idx = i * max_frames_per_chunk
        end_idx = min((i + 1) * max_frames_per_chunk, len(frames))
        
        chunk_frames = frames[start_idx:end_idx]
        chunk_temporal_ids = temporal_ids[start_idx:end_idx]
        
        # 构造针对每个分块的提示词
        if i == 0:
            # 第一个分块，使用基础提示词
            chunk_prompt = base_prompt
        else:
            # 后续分块，提示模型这是场景的延续
            chunk_prompt = (
                f"这是同一个场景的第 {i+1}/{num_chunks} 部分的延续画面。请继续你的分析。\n"
                f"上一部分的分析摘要是：'{chunk_analyses[-1][:200]}...'\n\n"
                f"现在请基于新的画面进行分析: {base_prompt}"
            )
        
        print(f"  - 正在处理分块 {i+1}/{num_chunks} (帧: {start_idx} to {end_idx})...")
        
        # 对当前分块进行分析
        chunk_result = call_chat(model, tokenizer, chunk_frames, [chunk_temporal_ids], chunk_prompt)
        chunk_analyses.append(chunk_result)
        
        # 强制释放显存，确保每个分块处理前都有干净的状态
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 将所有分块的分析结果拼接成一个完整的报告
    full_analysis = "\n\n".join(chunk_analyses)
    return full_analysis
def load_model_global(model_path: str, use_quant: bool = False):
    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if use_quant:
        print("警告: 4-bit 量化加载在 API 模式下可能需要额外配置，此处将使用标准加载。")
    mdl = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        mdl = mdl.cuda()
    mdl.eval()
    return mdl, tok, dtype

def load_asr_model_global(model_name: str, compute_type: str = ASR_COMPUTE_TYPE_DEFAULT):
    from pathlib import Path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # ==================== 关键修改 ====================
    # 强制在 CPU 上运行 ASR，以彻底绕开 cuDNN 兼容性问题
    device = "cpu"
    # 在 CPU 模式下，compute_type 最好使用 'int8' 以获得最佳性能
    compute_type = "int8" 
    print(f"警告: ASR 模型被强制在 CPU ({compute_type}) 模式下运行以保证兼容性。")
    # ================================================

    model_dir = Path(model_name).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"错误: ASR 模型目录不存在: {model_dir}")
        return None
    try:
        return WhisperModel(str(model_dir), device=device, compute_type=compute_type, local_files_only=True)
    except TypeError:
        return WhisperModel(str(model_dir), device=device, compute_type=compute_type)

model, tokenizer, model_dtype = load_model_global(MODEL_PATH_DEFAULT)
asr_model = load_asr_model_global(ASR_MODEL_DEFAULT)
print("模型加载完毕！服务器已准备就绪。")


# ================== 所有工具函数 (无变化) ==================
# ... (您所有的工具函数都保持原样，这里为了简洁省略，请确保您的文件中保留了它们) ...
def map_to_nearest_scale(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    tree = cKDTree(scale[:, None])
    _, indices = tree.query(values[:, None])
    return scale[indices]

def resize_safe(pil_img: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    if s <= max_side:
        return pil_img
    r = max_side / float(s)
    return pil_img.resize((int(round(w * r)), int(round(h * r))), Image.BICUBIC)

def get_video_codec_info(video_path: str):
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'v:0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            if 'streams' in info and len(info['streams']) > 0:
                return info['streams'][0].get('codec_name', 'unknown')
    except Exception as e:
        print(f"获取视频编码信息失败: {e}")
    return 'unknown'

def convert_video_if_needed(video_path: str, codec: str) -> str:
    incompatible_codecs = ['av1', 'vp9', 'hevc']
    if codec.lower() not in incompatible_codecs:
        return video_path
    base, ext = os.path.splitext(video_path)
    converted_path = f"{base}_h264{ext}"
    try:
        print(f"检测到 {codec.upper()} 编码，正在转换为 H264...")
        cmd = ['ffmpeg', '-i', video_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'copy', '-y', converted_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(converted_path):
            print("视频转换完成。")
            return converted_path
        else:
            print(f"转换失败，将尝试直接读取原视频。错误: {result.stderr}")
            return video_path
    except Exception as e:
        print(f"转换过程出错: {e}，将尝试直接读取原视频。")
        return video_path

def read_video_safe(video_path: str):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        _ = vr[0]
        return vr, 'decord'
    except Exception as e:
        print(f"Decord 读取失败: {e}，尝试使用 OpenCV")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        return cap, 'opencv'
    except Exception as e:
        raise RuntimeError(f"所有视频读取方法都失败了: {e}")

def get_video_frames_opencv(cap, frame_indices):
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(Image.new('RGB', (640, 480), color='black'))
    return frames

def detect_scenes(video_path: str, threshold: float = 27.0, use_adaptive: bool = False):
    video = open_video(video_path)
    scene_manager = SceneManager()
    detector = AdaptiveDetector() if use_adaptive else ContentDetector(threshold=threshold)
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return [(scene[0].frame_num, scene[1].frame_num) for scene in scene_list]

def asr_transcribe_full(asr_model, video_path: str, language: str, beam_size: int, vad_filter: bool):
    segments_out = []
    try:
        segments, info = asr_model.transcribe(
            video_path,
            language=(None if not language else language),
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=False,
        )
        for seg in segments:
            segments_out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
        full_text = "".join(s["text"] for s in segments_out).strip()
        return segments_out, full_text, info
    except Exception as e:
        print(f"ASR 转写失败: {e}")
        return [], "", None

def asr_text_per_scene(asr_segments, scenes, margin: float = 0.15):
    scene_texts = []
    for (start_t, end_t) in scenes:
        pieces = [seg["text"] for seg in asr_segments if not (seg["end"] < start_t - margin or seg["start"] > end_t + margin)]
        scene_texts.append("".join(pieces).strip())
    return scene_texts

def clip_text(t: str, max_chars: int) -> str:
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + " ……（已截断）"

def encode_video_by_scenes(video_path: str, scenes: List[Tuple[int, int]], choose_fps: float, max_side: int, time_scale: float):
    vr, method = read_video_safe(video_path)
    fps = float(vr.get_avg_fps()) if method == 'decord' else float(vr.get(cv2.CAP_PROP_FPS))
    total = len(vr) if method == 'decord' else int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0
    scene_data = []
    for scene_idx, (start_frame, end_frame) in enumerate(scenes):
        scene_length = end_frame - start_frame
        scene_duration = scene_length / fps if fps > 0 else 0.0
        need_frames = max(1, int(round(choose_fps * scene_duration)))
        scene_frame_idx = np.array([start_frame + int(round(i * (scene_length - 1) / max(1, need_frames - 1))) for i in range(need_frames)], dtype=np.int32)
        if method == 'decord':
            batch = vr.get_batch(scene_frame_idx).asnumpy()
            frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in batch]
        else:
            frames = get_video_frames_opencv(vr, scene_frame_idx)
        frames = [resize_safe(im, max_side=max_side) for im in frames]
        ts = scene_frame_idx / fps if fps > 0 else np.linspace(start_frame / fps, end_frame / fps, num=len(scene_frame_idx), endpoint=False)
        grid = np.arange(0, max(duration, 1e-6), time_scale)
        ts_id = (map_to_nearest_scale(ts, grid) / time_scale).astype(np.int32)
        scene_meta = {"start_time": float(start_frame / fps), "end_time": float(end_frame / fps)}
        scene_data.append({"frames": frames, "temporal_ids": [ts_id.tolist()], "meta": scene_meta})
    if method == 'opencv':
        vr.release()
    overall_meta = {"fps": fps, "duration_sec": duration, "num_scenes": len(scenes), "total_picked_frames": sum(len(s["frames"]) for s in scene_data), "read_method": method}
    return scene_data, overall_meta

def call_chat(model, tokenizer, frames, temporal_ids, prompt: str):
    msgs = [{"role": "user", "content": frames + [prompt]}]
    with torch.inference_mode():
        out = model.chat(msgs=msgs, tokenizer=tokenizer, use_image_id=False, max_slice_nums=1, temporal_ids=temporal_ids)
    return out[0] if isinstance(out, (list, tuple)) else out

def get_overall_report_multi(model, tokenizer, frames, tids, partials, overall_transcript_text="", scene_texts=None):
    scene_texts = scene_texts or []
    prompt0 = (
        "你将获得：（1）全片的代表帧；（2）分场景摘要；（3）可用的语音转写文本。"
        "请做【完整内容高度概括】（500字以上），按时间顺序梳理主线、关键事件、人物关系与风格，必要时以转写文本为主、图像为辅做纠错。\n"
        "\n--- 分场景摘要 ---\n" + "\n".join(partials) +
        ("\n\n--- 全文转写（截断或为空则忽略） ---\n" + (overall_transcript_text[:6000] + (" ……（已截断）" if len(overall_transcript_text) > 6000 else "")) if overall_transcript_text else "")
    )
    summary = call_chat(model, tokenizer, frames, tids, prompt0)
    prompt1 = (
        "基于代表帧/分场景摘要/语音转写，提取视频【基本信息】（严格结构化输出）：\n"
        "   - 场景描述（可列出每个场景一句话）：\n"
        "   - 主要人物（含未成年人标注）：\n"
        "   - 总时长：\n"
        "   - 语种/口音（如可判断）：\n"
        "   - 声音与画面是否一致：\n"
        "\n--- 分场景摘要 ---\n" + "\n".join(partials)
    )
    info = call_chat(model, tokenizer, frames, tids, prompt1)
    prompt2 = (
        '结合代表帧与语音转写，逐项整理本视频所有敏感内容和风险证据（无则写"未发现"）：\n'
        '   - 【血腥暴力】：\n'
        '   - 【色情低俗】：\n'
        '   - 【涉政敏感】：\n'
        '   - 【宗教极端】：\n'
        '   - 【歧视侮辱】：\n'
        '   - 【恐怖惊悚】：\n'
        '   - 【吸毒酗酒赌博】：\n'
        '   - 【自杀自残引导】：\n'
        '   - 【未成年人不当行为】：\n'
        '   - 【涉枪涉爆危险物品】：\n'
        '   - 【敏感符号/旗帜/服饰】：\n'
        '\n--- 分场景摘要 ---\n' + "\n".join(partials)
    )
    risks = call_chat(model, tokenizer, frames, tids, prompt2)
    prompt3 = (
        "请给出本视频审核【结论与建议】（6条以内，必要时引用转写中的关键句片段作为证据）。\n"
        "\n--- 分场景摘要 ---\n" + "\n".join(partials)
    )
    conclusion = call_chat(model, tokenizer, frames, tids, prompt3)
    report = (
        "## 视频完整内容摘要：\n" + summary.strip() + "\n\n"
        "### 1. 视频基本信息：\n" + info.strip() + "\n\n"
        "### 2. 风险证据帧：\n" + risks.strip() + "\n\n"
        "### 3. 结论与建议：\n" + conclusion.strip()
    )
    return report

# ================== 核心分析逻辑 (无变化) ==================
def run_full_analysis(video_path, params):
    print("开始分析视频:", video_path)
    print("使用参数:", params)
    use_scene = params.get('useSceneDetection', True)
    scene_thresh = params.get('sceneThreshold', 27.0)
    use_adaptive = params.get('useAdaptive', False)
    fps = params.get('chooseFps', 1.5)
    side = params.get('maxSide', 512)
    t_scale = params.get('timeScale', 0.1)
    prompt = params.get('promptText', "请分析这个场景")
    enable_asr = params.get('enableAsr', True)
    asr_lang = params.get('asrLanguage', "")
    asr_beam = params.get('asrBeamSize', 5)
    asr_vad = params.get('asrVad', True)
    asr_max_chars = params.get('asrMaxChars', 1200)
    codec = get_video_codec_info(video_path)
    processed_video_path = convert_video_if_needed(video_path, codec)
    if use_scene:
        scenes = detect_scenes(processed_video_path, threshold=scene_thresh, use_adaptive=use_adaptive)
        if not scenes:
            print("未检测到场景切换，将视频作为单一场景处理。")
            vr, _ = read_video_safe(processed_video_path)
            total_frames = len(vr) if isinstance(vr, VideoReader) else int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
            scenes = [(0, total_frames)]
    else:
        vr, _ = read_video_safe(processed_video_path)
        total_frames = len(vr) if isinstance(vr, VideoReader) else int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
        scenes = [(0, total_frames)]
    scene_data, meta = encode_video_by_scenes(processed_video_path, scenes, fps, side, t_scale)
    full_transcript, scene_transcripts = "", []
    if enable_asr and asr_model:
        print("开始语音转写...")
        scene_times = [(s["meta"]["start_time"], s["meta"]["end_time"]) for s in scene_data]
        segments, full_transcript, _ = asr_transcribe_full(asr_model, processed_video_path, asr_lang, asr_beam, asr_vad)
        scene_transcripts = asr_text_per_scene(segments, scene_times)
    else:
        scene_transcripts = [""] * len(scene_data)
    print("开始分场景分析...")
    partials = []
    # 【新增】从参数中获取每块最大帧数，提供一个默认值
    max_frames_per_chunk = params.get('maxFramesPerChunk', 30) 

    for i, (scene, transcript) in enumerate(zip(scene_data, scene_transcripts)):
        short_transcript = clip_text(transcript, asr_max_chars)
        
        # 【修改】我们先构建基础的提示词
        scene_prompt = (
            f"这是视频的第 {i+1}/{len(scene_data)} 个场景 (时间: {scene['meta']['start_time']:.1f}s - {scene['meta']['end_time']:.1f}s)。\n"
            f"该场景语音转写: {short_transcript or '无'}\n\n"
            f"现在请基于画面和转写文本进行分析: {prompt}"
        )
        
        # 【核心修改】调用新的分块处理函数
        # part = call_chat(model, tokenizer, scene["frames"], scene["temporal_ids"], scene_prompt) # <- 注释掉旧的调用
        part = analyze_scene_in_chunks(model, tokenizer, scene, scene_prompt, max_frames_per_chunk)
        
        partials.append(part)
    print("生成总报告...")
    rep_frames = [s["frames"][0] for s in scene_data[:3] if s["frames"]]
    rep_tids = [[i*5 for i in range(len(rep_frames))]]
    overall_report = get_overall_report_multi(model, tokenizer, rep_frames, rep_tids, partials, full_transcript, scene_transcripts)
    scene_details = [{
        "title": f"场景 {i+1}",
        "time": f"{s['meta']['start_time']:.2f}s - {s['meta']['end_time']:.2f}s",
        "analysis": p,
        "transcript": t
    } for i, (s, p, t) in enumerate(zip(scene_data, partials, scene_transcripts))]
    final_result = {
        "stats": {
            "scenes": meta['num_scenes'],
            "frames": meta['total_picked_frames'],
            "duration": f"{meta['duration_sec']:.2f}s",
            "fps": f"{meta['fps']:.2f}",
        },
        "report": overall_report,
        "full_transcript": full_transcript,
        "scene_details": scene_details
    }
    if processed_video_path != video_path and os.path.exists(processed_video_path):
        os.remove(processed_video_path)
    return final_result


# --- 新增：辅助函数，用于从报告中判断是否违规 ---
def determine_violation_status(report_text: str) -> Tuple[bool, str]:
    """
    从完整的分析报告中解析出风险部分，并判断是否存在违规。
    返回: (是否有违规, 风险部分的文本)
    """
    try:
        risk_section_start = report_text.find("2. 风险证据帧：")
        if risk_section_start == -1:
            return False, "无法找到风险分析部分。"
        next_section_start = report_text.find("3. 结论与建议：", risk_section_start)
        if next_section_start != -1:
            risk_text = report_text[risk_section_start:next_section_start].strip()
        else:
            risk_text = report_text[risk_section_start:].strip()
        
        # 判断逻辑：如果风险文本中除了标题外，有效内容只包含“未发现”，则认为无违规。
        # 这样比单纯检查 "未发现" 更鲁棒，防止报告中提到 "未发现XX，但发现YY" 的情况。
        meaningful_text = risk_text.replace("2. 风险证据帧：", "").strip()
        # 创建一个列表，包含所有表示“无风险”的关键词
        no_violation_keywords = ["未发现", "无", "未见明显"]
        # 检查是否有任何风险类别后面跟了非“无风险”的描述
        lines = [line.strip() for line in meaningful_text.split('\n') if line.strip()]
        has_violations = False
        for line in lines:
            if '：' in line: # 检查是否是风险条目，如 "【血腥暴力】："
                content = line.split('：', 1)[1].strip()
                if content and not any(keyword in content for keyword in no_violation_keywords):
                    has_violations = True
                    break
        
        return has_violations, risk_text
    except Exception as e:
        print(f"解析报告时出错: {e}")
        return False, "报告解析失败。"


# --- 【新增】用于处理视频下载的接口 ---
@app.route('/download', methods=['POST'])
def download_video_endpoint():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "请求中未包含 URL"}), 400

    video_url = data['url']
    print(f"收到下载请求，URL: {video_url}")

    try:
        # 定义下载选项，文件名将包含视频标题
        output_template = os.path.join(DOWNLOAD_FOLDER, '%(title)s.%(ext)s')
        
        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # 优先下载 mp4
            'merge_output_format': 'mp4',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            # 获取下载后的实际文件名
            downloaded_file_path = ydl.prepare_filename(info_dict)

        print(f"视频下载成功: {downloaded_file_path}")
        
        return jsonify({
            "message": "下载成功", 
            "filename": os.path.basename(downloaded_file_path),
            "original_title": info_dict.get('title', 'N/A')
        })

    except Exception as e:
        print(f"yt-dlp 下载失败: {e}")
        return jsonify({"error": f"视频下载失败: {str(e)}"}), 500


# ================== Flask API 接口 (正确且唯一的版本) ==================
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_video_endpoint():
    # ---------------------------------------------
    # 第一部分：处理 GET 请求
    # ---------------------------------------------
    if request.method == 'GET':
        # 如果是 GET 请求，就返回一个提示页面，然后函数结束。
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>视频分析接口</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 40px; }
                h1 { color: #333; }
                p { color: #666; }
                code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>这是一个视频分析的后端接口</h1>
            <p>您不能通过浏览器直接访问这个地址来上传视频。</p>
            <p>请通过我们的前端页面来使用文件上传功能。</p>
            <p>这个地址 (<code>/analyze</code>) 只接受 <code>POST</code> 方法提交的视频文件。</p>
        </body>
        </html>
        """

    # ---------------------------------------------
    # 第二部分：处理 POST 请求
    # ---------------------------------------------
    # 如果代码能执行到这里，说明 request.method 一定是 'POST'
     # --- 【核心修改】处理 POST 请求的逻辑 ---
    user_ip = request.remote_addr
    is_temp_file = False # 标记是否是临时文件，分析完要删除

    # 判断是文件上传，还是基于已下载的文件名进行分析
    if 'video' in request.files and request.files['video'].filename != '':
        # 1. 处理文件上传 (原始逻辑)
        video_file = request.files['video']
        video_filename = video_file.filename
        print(f"收到来自 IP [{user_ip}] 的文件上传请求，处理文件: {video_filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name
        is_temp_file = True # 标记为临时文件

    elif 'filename' in request.form:
        # 2. 处理基于已下载文件的分析请求 (新逻辑)
        video_filename = request.form.get('filename')
        print(f"收到来自 IP [{user_ip}] 的分析请求，处理已下载文件: {video_filename}")
        
        video_path = os.path.join(DOWNLOAD_FOLDER, video_filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "服务器上找不到指定的已下载文件"}), 404
        
        # downloaded files are not temporary
        is_temp_file = False 

    else:
        return jsonify({"error": "请求无效，既没有上传文件也没有提供文件名"}), 400
    
    try:
        params = {
            'useSceneDetection': request.form.get('useSceneDetection', 'true').lower() == 'true',
            'sceneThreshold': float(request.form.get('sceneThreshold', 27.0)),
            'useAdaptive': request.form.get('useAdaptive', 'false').lower() == 'true',
            'chooseFps': float(request.form.get('chooseFps', 1.5)),
            'maxSide': int(request.form.get('maxSide', 512)),
            'timeScale': float(request.form.get('timeScale', 0.1)),
            'promptText': request.form.get('promptText', ''),
            'enableAsr': request.form.get('enableAsr', 'true').lower() == 'true',
            'asrLanguage': request.form.get('asrLanguage', ''),
            'asrBeamSize': int(request.form.get('asrBeamSize', 5)),
            'asrVad': request.form.get('asrVad', 'true').lower() == 'true',
            'asrMaxChars': int(request.form.get('asrMaxChars', 1200)),
            'maxFramesPerChunk': int(request.form.get('maxFramesPerChunk', 30))
        }

        results = run_full_analysis(video_path, params)
        
        with app.app_context():
            overall_report = results.get("report", "")
            has_violations, risk_summary = determine_violation_status(overall_report)

            new_log = AnalysisLog(
                user_ip=user_ip,
                video_filename=video_filename,
                has_violations=has_violations,
                report_summary=risk_summary
            )
            
            db.session.add(new_log)
            db.session.commit()
            print(f"新记录已存入数据库: ID={new_log.id}, IP={user_ip}, 文件名='{video_filename}', 违规={has_violations}")
        
        return jsonify(results)

    except Exception as e:
        print(f"处理请求时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# --- 新增：用于获取所有历史记录的接口 ---
@app.route('/logs', methods=['GET'])
def get_logs():
    """返回数据库中所有的分析日志，按时间倒序排列。"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    with app.app_context():
        # 使用分页查询，提高性能
        pagination = AnalysisLog.query.order_by(AnalysisLog.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        logs = pagination.items
        return jsonify({
            "total": pagination.total,
            "pages": pagination.pages,
            "current_page": page,
            "logs": [log.to_dict() for log in logs]
        })


# --- 启动 Flask 服务器 ---
if __name__ == '__main__':
    with app.app_context():
        # 这行代码会自动检查 AnalysisLog 模型，
        # 如果数据库文件 'project.db' 中没有对应的表，就会自动创建。
        # 如果表已存在，则不会做任何事。
        db.create_all()
        print("数据库已初始化。")

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)