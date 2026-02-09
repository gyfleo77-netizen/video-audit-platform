import os
from pathlib import Path

from faster_whisper import WhisperModel

from .config import ASR_COMPUTE_TYPE_DEFAULT

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
