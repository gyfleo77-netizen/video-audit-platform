import json
import os
import subprocess
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

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
