import os
from typing import Tuple

import cv2
import requests
import torch
from decord import VideoReader
from transformers import AutoModel, AutoTokenizer

from .asr_utils import asr_text_per_scene, asr_transcribe_full, clip_text, load_asr_model_global
from .config import ASR_MODEL_DEFAULT, LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME, MODEL_PATH_DEFAULT
from .video_utils import (
    convert_video_if_needed,
    detect_scenes,
    encode_video_by_scenes,
    get_video_codec_info,
    read_video_safe,
)

print("服务器启动中... 正在加载模型，请稍候...")

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

def call_chat(model, tokenizer, frames, temporal_ids, prompt: str):
    msgs = [{"role": "user", "content": frames + [prompt]}]
    with torch.inference_mode():
        out = model.chat(msgs=msgs, tokenizer=tokenizer, use_image_id=False, max_slice_nums=1, temporal_ids=temporal_ids)
    return out[0] if isinstance(out, (list, tuple)) else out

def call_llm_api(
    prompt: str,
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    api_base: str = None,
    api_key: str = None,
    model_name: str = None,
) -> str:
    """
    外部 LLM 调用（默认假设兼容 OpenAI /chat/completions），用于总摘要/基础信息/风险/结论。
    """
    base = api_base or LLM_API_BASE
    key = api_key or LLM_API_KEY
    model = model_name or LLM_MODEL_NAME
    if not base:
        raise RuntimeError("LLM_API_BASE 未配置，无法调用外部模型。")
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        if "output_text" in data:
            return data["output_text"]
        raise ValueError("LLM 响应缺少文本内容。")
    except Exception as e:
        raise RuntimeError(f"外部 LLM 调用失败: {e}")

def get_overall_report_multi(model, tokenizer, frames, tids, partials, overall_transcript_text="", scene_texts=None, llm_config=None):
    scene_texts = scene_texts or []
    cfg = llm_config or {}
    prompt0 = (
        "你将获得：（1）全片的代表帧；（2）分场景摘要；（3）可用的语音转写文本。"
        "请做【完整内容高度概括】（500字以上），按时间顺序梳理主线、关键事件、人物关系与风格，必要时以转写文本为主、图像为辅做纠错。\n"
        "\n--- 分场景摘要 ---\n" + "\n".join(partials) +
        ("\n\n--- 全文转写（截断或为空则忽略） ---\n" + (overall_transcript_text[:6000] + (" ……（已截断）" if len(overall_transcript_text) > 6000 else "")) if overall_transcript_text else "")
    )
    try:
        summary = call_llm_api(
            prompt0,
            max_tokens=1600,
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key"),
            model_name=cfg.get("model_name"),
        )
    except Exception as e:
        print(f"外部 LLM 生成总摘要失败，回退本地模型: {e}")
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
    try:
        info = call_llm_api(
            prompt1,
            max_tokens=800,
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key"),
            model_name=cfg.get("model_name"),
        )
    except Exception as e:
        print(f"外部 LLM 生成基础信息失败，回退本地模型: {e}")
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
    try:
        risks = call_llm_api(
            prompt2,
            max_tokens=800,
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key"),
            model_name=cfg.get("model_name"),
        )
    except Exception as e:
        print(f"外部 LLM 生成风险分析失败，回退本地模型: {e}")
        risks = call_chat(model, tokenizer, frames, tids, prompt2)
    prompt3 = (
        "请给出本视频审核【结论与建议】（6条以内，必要时引用转写中的关键句片段作为证据）。\n"
        "\n--- 分场景摘要 ---\n" + "\n".join(partials)
    )
    try:
        conclusion = call_llm_api(
            prompt3,
            max_tokens=600,
            api_base=cfg.get("api_base"),
            api_key=cfg.get("api_key"),
            model_name=cfg.get("model_name"),
        )
    except Exception as e:
        print(f"外部 LLM 生成结论失败，回退本地模型: {e}")
        conclusion = call_chat(model, tokenizer, frames, tids, prompt3)
    report = (
        "## 视频完整内容摘要：\n" + summary.strip() + "\n\n"
        "### 1. 视频基本信息：\n" + info.strip() + "\n\n"
        "### 2. 风险证据帧：\n" + risks.strip() + "\n\n"
        "### 3. 结论与建议：\n" + conclusion.strip()
    )
    return report

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
    llm_config = {
        "api_base": params.get("llmApiBase") or LLM_API_BASE,
        "api_key": params.get("llmApiKey") or LLM_API_KEY,
        "model_name": params.get("llmModelName") or LLM_MODEL_NAME,
    }
    overall_report = get_overall_report_multi(
        model,
        tokenizer,
        rep_frames,
        rep_tids,
        partials,
        full_transcript,
        scene_transcripts,
        llm_config=llm_config,
    )
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

model, tokenizer, model_dtype = load_model_global(MODEL_PATH_DEFAULT)
asr_model = load_asr_model_global(ASR_MODEL_DEFAULT)
print("模型加载完毕！服务器已准备就绪。")
