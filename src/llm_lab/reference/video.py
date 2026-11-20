"""How two vision-language models read a video (the Video tab of the Multimodal page).

    qwen3vl   Qwen3-VL-8B          2 frames per second, every 2 frames merged into one grid of tokens,
                                   a text timestamp before each grid; fewer pixels per frame as videos grow
    llava_ov  LLaVA-OneVision-7B   32 frames however long the video, each 384 × 384 → 729 patches,
                                   pooled 2 × 2 (bilinear) to 196 tokens

The rules follow transformers' Qwen3VLVideoProcessor (with the per-frame pixel cap of the reference
qwen-vl-utils) and LlavaOnevisionProcessor; tests/test_video_models.py checks them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .vision import QWEN, Decoder, Encoder

FPS = 2                       # Qwen3-VL samples 2 frames per second of video
MIN_FRAMES, MAX_FRAMES = 4, 768
TEMPORAL_PATCH = 2            # frames merged into one temporal patch
FACTOR = 32                   # patch 16 × spatial merge 2
MIN_PIXELS = 4096             # video_preprocessor_config.json: size.shortest_edge
MAX_PIXELS = 25_165_824       # size.longest_edge: pixels summed over all sampled frames
MAX_VIDEO_TOKENS = 768        # per-frame cap of the reference implementation, in merged tokens
OV_FRAMES = 32                # LLaVA-OneVision's video setting: 32 frames sampled uniformly
OV_SIDE = 384


@dataclass(frozen=True)
class VideoLanguageModel:
    key: str
    name: str
    source: str
    encoder: Encoder
    decoder: Decoder
    video_token_id: int


QWEN3VL = VideoLanguageModel('qwen3vl', 'Qwen3-VL-8B', QWEN.source, QWEN.encoder, QWEN.decoder, video_token_id=151656)
LLAVA_OV = VideoLanguageModel(
    'llava_ov', 'LLaVA-OneVision-7B', 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    Encoder('SigLIP So400m/14-384', layers=26, hidden=1152, heads=16, intermediate=4304, patch=14, image_size=OV_SIDE),
    Decoder('Qwen2-7B', layers=28, hidden=3584, heads=28, kv_heads=4, head_dim=128, intermediate=18944,
            vocab=152128, max_positions=32768),
    video_token_id=151647)

VIDEO_MODELS = {m.key: m for m in (QWEN3VL, LLAVA_OV)}

PARAMETERS = {   # (vision encoder, connector, language model incl. LM head), counted from transformers
    'qwen3vl': (415_891_440, 160_496_896, 8_190_735_360),
    'llava_ov': (397_747_744, 16_980_992, 7_616_078_848),   # the LM part includes the image_newline vector
}


def sample_frames(seconds: float, video_fps: float = 30.0) -> list[int]:
    """Indices of the frames Qwen3-VL reads: 2 per second, at least 4, at most 768, spread evenly."""
    total = max(1, round(seconds * video_fps))
    n = int(total / video_fps * FPS)
    n = min(max(n, MIN_FRAMES), MAX_FRAMES, total)
    if n == 1:
        return [0]
    step = (total - 1) / (n - 1)                       # as numpy.linspace, whose last value is exact
    return [round(i * step) for i in range(n - 1)] + [total - 1]   # round(): halves to even, as numpy


def video_smart_resize(frames: int, height: int, width: int, max_pixels: int, min_pixels: int = MIN_PIXELS, factor: int = FACTOR):
    """Qwen3-VL's video resize: keep the aspect ratio, multiples of 32, all frames together within the budget."""
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(frames / TEMPORAL_PATCH) * TEMPORAL_PATCH
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt(frames * height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class VideoPlan:
    frames: int                  # frames the encoder reads
    frame_size: tuple[int, int]  # (height, width) of each frame as encoded
    groups: int                  # Qwen3-VL: grids of 2 frames, each with its own timestamp
    tokens_per_group: int
    tokens: int                  # video tokens in the text sequence (timestamps not included)
    timestamps: tuple[float, ...]


def video_plan(model: str, seconds: float, height: int = 720, width: int = 1280, video_fps: float = 30.0) -> VideoPlan:
    if model == 'qwen3vl':
        idx = sample_frames(seconds, video_fps)
        n = len(idx)
        per_frame = max(min(MAX_VIDEO_TOKENS * FACTOR * FACTOR, MAX_PIXELS // n), int(MIN_PIXELS * 1.05))
        rh, rw = video_smart_resize(n, height, width, per_frame * n)
        groups = math.ceil(n / TEMPORAL_PATCH)     # an odd last frame is repeated to fill its pair
        per_group = (rh // FACTOR) * (rw // FACTOR)
        times = [i / video_fps for i in idx]
        stamps = tuple((times[i] + times[min(i + TEMPORAL_PATCH - 1, n - 1)]) / 2 for i in range(0, n, TEMPORAL_PATCH))
        return VideoPlan(n, (rh, rw), groups, per_group, groups * per_group, stamps)
    if model == 'llava_ov':
        side = math.ceil(OV_SIDE // LLAVA_OV.encoder.patch / 2)   # 27 patches → 14 after 2 × 2 pooling
        per_frame = side * side
        return VideoPlan(OV_FRAMES, (OV_SIDE, OV_SIDE), OV_FRAMES, per_frame, OV_FRAMES * per_frame + 1, ())
    raise KeyError(model)


def video_cache_values(model: str, seconds: float) -> int:
    m = VIDEO_MODELS[model]
    return video_plan(model, seconds).tokens * m.decoder.kv_values_per_token()


def onevision_pool(features, grid: int = 27):
    """LLaVA-OneVision's video pooling: each frame's 27 × 27 grid of vectors, resized bilinearly to 14 × 14."""
    import torch.nn.functional as F
    frames, positions, dim = features.shape
    x = features.view(frames, grid, grid, dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(math.ceil(grid / 2), math.ceil(grid / 2)), mode='bilinear')
    return x.permute(0, 2, 3, 1).reshape(frames, -1, dim)


def hf_config(model: str):
    import transformers as tf
    if model == 'qwen3vl':
        from . import vision
        return vision.hf_config('qwen')
    if model == 'llava_ov':
        e, d = LLAVA_OV.encoder, LLAVA_OV.decoder
        return tf.LlavaOnevisionConfig(
            vision_config=dict(model_type='siglip_vision_model', hidden_size=e.hidden, intermediate_size=e.intermediate,
                               num_hidden_layers=e.layers, num_attention_heads=e.heads, image_size=e.image_size, patch_size=e.patch,
                               vision_use_head=False),
            text_config=dict(model_type='qwen2', hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                             num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, vocab_size=d.vocab,
                             max_position_embeddings=d.max_positions, tie_word_embeddings=False),
            video_token_index=LLAVA_OV.video_token_id)
    raise KeyError(model)
