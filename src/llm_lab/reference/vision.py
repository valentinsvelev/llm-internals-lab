"""How four vision-language models turn an image into something their language model reads.

Each is one way of connecting a vision encoder to a decoder:

    llava   LLaVA-1.5-7B          projector    336 × 336 crop → 576 tokens in the text sequence
    qwen    Qwen3-VL-8B           native size  any size → one token per 32 × 32 pixels, 3D positions (M-RoPE)
    llama   Llama 3.2 11B Vision  cross-attn   up to 4 tiles of 560 → read by 8 cross-attention layers
    gemma   Gemma 3 4B            pooling      896 × 896 → 4,096 patches → averaged to 256 tokens

Numbers come from each model's config.json and preprocessor_config.json. The resizing rules, the
connectors, the position ids and the masks are written to match transformers, and
tests/test_vision_models.py checks that they do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------- configs


@dataclass(frozen=True)
class Encoder:
    name: str
    layers: int
    hidden: int
    heads: int
    intermediate: int
    patch: int
    image_size: int | None        # fixed input side; None: native resolution
    global_layers: int = 0        # Llama 3.2: 8 more layers that see all tiles at once


@dataclass(frozen=True)
class Decoder:
    name: str
    layers: int                   # all decoder layers, cross-attention layers included
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    max_positions: int
    cross_layers: tuple[int, ...] = ()
    sliding_window: int | None = None
    sliding_pattern: int = 0      # Gemma 3: every 6th layer is global, the other 5 use a 1,024-token window

    @property
    def self_layers(self) -> int:
        return self.layers - len(self.cross_layers)

    def kv_values_per_token(self) -> int:
        """Keys and values one text or image token stores, summed over the self-attention layers."""
        return 2 * self.kv_heads * self.head_dim * self.self_layers

    def cross_kv_values_per_position(self) -> int:
        """Keys and values one image position stores, summed over the cross-attention layers."""
        return 2 * self.kv_heads * self.head_dim * len(self.cross_layers)


@dataclass(frozen=True)
class VisionLanguageModel:
    key: str
    name: str
    source: str
    kind: str                     # 'projector' | 'native' | 'cross' | 'pooled'
    encoder: Encoder
    decoder: Decoder
    image_token_id: int


LLAVA = VisionLanguageModel(
    'llava', 'LLaVA-1.5-7B', 'llava-hf/llava-1.5-7b-hf', 'projector',
    Encoder('CLIP ViT-L/14-336', layers=24, hidden=1024, heads=16, intermediate=4096, patch=14, image_size=336),
    Decoder('Vicuna-7B v1.5', layers=32, hidden=4096, heads=32, kv_heads=32, head_dim=128, intermediate=11008,
            vocab=32064, max_positions=4096),
    image_token_id=32000)
QWEN = VisionLanguageModel(
    'qwen', 'Qwen3-VL-8B', 'Qwen/Qwen3-VL-8B-Instruct', 'native',
    Encoder('Qwen3-VL ViT', layers=27, hidden=1152, heads=16, intermediate=4304, patch=16, image_size=None),
    Decoder('Qwen3-8B', layers=36, hidden=4096, heads=32, kv_heads=8, head_dim=128, intermediate=12288,
            vocab=151936, max_positions=262144),
    image_token_id=151655)
LLAMA = VisionLanguageModel(
    'llama', 'Llama 3.2 11B Vision', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'cross',
    Encoder('Llama 3.2 ViT-H/14', layers=32, hidden=1280, heads=16, intermediate=5120, patch=14, image_size=560,
            global_layers=8),
    Decoder('Llama 3.1 8B + 8 cross-attention layers', layers=40, hidden=4096, heads=32, kv_heads=8, head_dim=128,
            intermediate=14336, vocab=128256, max_positions=131072, cross_layers=(3, 8, 13, 18, 23, 28, 33, 38)),
    image_token_id=128256)
GEMMA = VisionLanguageModel(
    'gemma', 'Gemma 3 4B', 'google/gemma-3-4b-it', 'pooled',
    Encoder('SigLIP So400m/14', layers=27, hidden=1152, heads=16, intermediate=4304, patch=14, image_size=896),
    Decoder('Gemma 3 4B', layers=34, hidden=2560, heads=8, kv_heads=4, head_dim=256, intermediate=10240,
            vocab=262208, max_positions=131072, sliding_window=1024, sliding_pattern=6),
    image_token_id=262144)

VLMS = {m.key: m for m in (LLAVA, QWEN, LLAMA, GEMMA)}

# Parameters (vision encoder, connector, language model incl. LM head), as counted from the transformers
# models built from hf_config(); Qwen's connector includes the three DeepStack mergers.
PARAMETERS = {
    'llava': (303_507_456, 20_979_712, 6_738_939_904),
    'qwen': (415_891_440, 160_496_896, 8_190_735_360),
    'llama': (863_567_379, 31_461_376, 9_775_192_080),
    'gemma': (416_866_032, 2_950_272, 3_880_263_168),
}

# preprocessing settings (preprocessor_config.json)
QWEN_MERGE, QWEN_MIN_PIXELS, QWEN_MAX_PIXELS = 2, 65536, 16777216
LLAMA_MAX_TILES = 4
GEMMA_TOKENS = 256                # mm_tokens_per_image
LLAVA_IMAGE_TOKENS = 576          # image_seq_length

# ------------------------------------------------------------------------------------- image → tokens


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = QWEN_MIN_PIXELS,
                 max_pixels: int = QWEN_MAX_PIXELS) -> tuple[int, int]:
    """Qwen's resize: both sides a multiple of `factor` (patch 16 × merge 2), pixel count kept in range."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError('aspect ratio above 200')
    h, w = round(height / factor) * factor, round(width / factor) * factor     # Python rounds half to even
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def tile_arrangements(max_tiles: int = LLAMA_MAX_TILES) -> list[tuple[int, int]]:
    """Llama 3.2: every grid of at most `max_tiles` tiles, as (columns, rows)."""
    return [(w, h) for w in range(1, max_tiles + 1) for h in range(1, max_tiles + 1) if w * h <= max_tiles]


def tiled_canvas(height: int, width: int, max_tiles: int = LLAMA_MAX_TILES, tile: int = 560) -> tuple[int, int]:
    """Llama 3.2: the canvas (height, width) the image is fitted into. The smallest upscale if one exists,
    else the mildest downscale; among equal scales, the smallest area."""
    canvases = [(a * tile, b * tile) for a, b in tile_arrangements(max_tiles)]   # read as (height, width), as transformers does
    scales = [min(ch / height, cw / width) for ch, cw in canvases]
    up = [s for s in scales if s >= 1]
    best = min(up) if up else max(s for s in scales if s < 1)
    return min((c for c, s in zip(canvases, scales) if s == best), key=lambda c: c[0] * c[1])


def fit_to_canvas(height: int, width: int, canvas_h: int, canvas_w: int, tile: int = 560) -> tuple[int, int]:
    """Llama 3.2: resize keeping the aspect ratio, no side below one tile or above the canvas."""
    th, tw = min(max(height, tile), canvas_h), min(max(width, tile), canvas_w)
    if tw / width < th / height:
        return min(math.floor(height * tw / width) or 1, th), tw
    return th, min(math.floor(width * th / height) or 1, tw)


@dataclass(frozen=True)
class ImagePlan:
    resized: tuple[int, int]      # (height, width) of the pixels the encoder sees; Llama: the image inside its canvas
    canvas: tuple[int, int]       # Llama: the tiled canvas; others: same as resized
    patches: tuple[int, int]      # patch grid: of the image (Qwen) or of one tile / the square input
    tiles: int                    # tiles holding the image (Llama), else 1
    encoder_positions: int        # vectors the vision encoder outputs (CLS and padded tiles included)
    tokens: int                   # image embeddings placed into the text sequence
    cross_positions: int          # image vectors the decoder reads through cross-attention


def image_plan(model: str | VisionLanguageModel, height: int, width: int) -> ImagePlan:
    m = VLMS[model] if isinstance(model, str) else model
    p = m.encoder.patch
    if m.key == 'llava':          # resize the short side to 336, then crop the centre square
        side = m.encoder.image_size
        g = side // p
        return ImagePlan((side, side), (side, side), (g, g), 1, g * g + 1, LLAVA_IMAGE_TOKENS, 0)
    if m.key == 'qwen':
        h, w = smart_resize(height, width, factor=p * QWEN_MERGE)
        gh, gw = h // p, w // p
        return ImagePlan((h, w), (h, w), (gh, gw), 1, gh * gw, gh * gw // QWEN_MERGE ** 2, 0)
    if m.key == 'llama':
        tile = m.encoder.image_size
        ch, cw = tiled_canvas(height, width, LLAMA_MAX_TILES, tile)
        rh, rw = fit_to_canvas(height, width, ch, cw, tile)
        g = tile // p
        per_tile = g * g + 1                                  # 1,600 patches + a class token
        # the encoder always runs on max_num_tiles tiles; empty ones are masked, but still cached
        return ImagePlan((rh, rw), (ch, cw), (g, g), (ch // tile) * (cw // tile),
                         LLAMA_MAX_TILES * per_tile, 1, LLAMA_MAX_TILES * per_tile)
    if m.key == 'gemma':          # squash to 896 × 896 (pan & scan, off by default, would add crops)
        side = m.encoder.image_size
        g = side // p
        return ImagePlan((side, side), (side, side), (g, g), 1, g * g, GEMMA_TOKENS, 0)
    raise KeyError(m.key)


def image_cache_values(model: str | VisionLanguageModel, height: int, width: int) -> int:
    """Keys and values one image adds to the decoder's cache, in numbers (× 2 bytes at 16 bits)."""
    m = VLMS[model] if isinstance(model, str) else model
    plan = image_plan(m, height, width)
    return plan.tokens * m.decoder.kv_values_per_token() + plan.cross_positions * m.decoder.cross_kv_values_per_position()


# ------------------------------------------------------------------------------- 3D positions (M-RoPE)


def mrope_positions(segments: list[tuple[str, int | tuple[int, int, int]]], merge: int = QWEN_MERGE) -> list[list[int]]:
    """Qwen3-VL position ids, one (temporal, height, width) triple per token.

    `segments` is the sequence as runs: ('text', n) or ('image', (t, h, w)) with the image's patch grid.
    Text advances all three by one per token. An image's tokens get (start, start + row, start + col),
    and the text after it continues at start + max(rows, cols): the image costs its side length in
    positions, not its token count."""
    out: list[list[int]] = [[], [], []]
    pos = 0
    for kind, size in segments:
        if kind == 'text':
            for axis in out:
                axis.extend(range(pos, pos + size))
            pos += size
        else:
            t, h, w = size
            rows, cols = h // merge, w // merge
            for f in range(t):
                for r in range(rows):
                    for c in range(cols):
                        out[0].append(pos + f)
                        out[1].append(pos + r)
                        out[2].append(pos + c)
            pos += max(rows, cols)
    return out


def mrope_pair_axes(head_dim: int = 128, section: tuple[int, int, int] = (24, 20, 20)) -> str:
    """Which position each rotary frequency pair rotates by, as a string of 'T', 'H', 'W'.

    Qwen3-VL interleaves them (T H W T H W …) so every axis gets fast and slow frequencies;
    the pairs after 3 × 20 stay temporal."""
    pairs = ['T'] * (head_dim // 2)
    for axis, offset in (('H', 1), ('W', 2)):
        for i in range(offset, 3 * section[1 if axis == 'H' else 2], 3):
            pairs[i] = axis
    return ''.join(pairs)


# ------------------------------------------------------------------------- Gemma 3: images see each other


def gemma_attention_mask(is_image: list[bool]) -> list[list[bool]]:
    """Causal, except that the tokens of one image see each other in both directions."""
    n = len(is_image)
    block = []
    current = -1
    for i, img in enumerate(is_image):
        if img and (i == 0 or not is_image[i - 1]):
            current += 1
        block.append(current if img else -1)
    return [[j <= i or (block[i] >= 0 and block[i] == block[j]) for j in range(n)] for i in range(n)]


# --------------------------------------------------------------------------------------- the connectors
# Weights are named as in transformers, so a module loads with load_state_dict from the HF submodule.

def _torch():
    import torch
    from torch import nn
    return torch, nn


def llava_projector(vision_hidden: int = 1024, text_hidden: int = 4096):
    """LLaVA: two linear layers with GELU, applied to each of the 576 patch vectors."""
    torch, nn = _torch()

    class LlavaProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(vision_hidden, text_hidden)
            self.act = nn.GELU()
            self.linear_2 = nn.Linear(text_hidden, text_hidden)

        def forward(self, x):                      # [N, 576, vision_hidden] → [N, 576, text_hidden]
            return self.linear_2(self.act(self.linear_1(x)))
    return LlavaProjector()


def qwen_merger(vision_hidden: int = 1152, text_hidden: int = 4096, merge: int = QWEN_MERGE, postshuffle_norm: bool = False):
    """Qwen3-VL: each 2 × 2 block of neighbouring patches becomes one token (4 × 1152 → 4608 → 4096).

    The processor already orders patches so that each block's four are consecutive; DeepStack's extra
    mergers normalize after concatenating (postshuffle_norm) instead of before."""
    torch, nn = _torch()
    width = vision_hidden * merge ** 2

    class QwenMerger(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width if postshuffle_norm else vision_hidden, eps=1e-6)
            self.linear_fc1 = nn.Linear(width, width)
            self.act_fn = nn.GELU()
            self.linear_fc2 = nn.Linear(width, text_hidden)

        def forward(self, x):                      # [patches, vision_hidden] → [patches / 4, text_hidden]
            x = self.norm(x.reshape(-1, width)) if postshuffle_norm else self.norm(x).reshape(-1, width)
            return self.linear_fc2(self.act_fn(self.linear_fc1(x)))
    return QwenMerger()


def gemma_projector(vision_hidden: int = 1152, text_hidden: int = 2560, patches_per_side: int = 64,
                    tokens: int = GEMMA_TOKENS, eps: float = 1e-6):
    """Gemma 3: average 4 × 4 patches (64 × 64 → 16 × 16 = 256), RMS-normalize, one linear map."""
    torch, nn = _torch()
    side = int(tokens ** 0.5)
    k = patches_per_side // side

    class GemmaProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_input_projection_weight = nn.Parameter(torch.zeros(vision_hidden, text_hidden))
            self.mm_soft_emb_norm = nn.Module()
            self.mm_soft_emb_norm.weight = nn.Parameter(torch.zeros(vision_hidden))   # Gemma's norm: x · (1 + w)

        def forward(self, x):                      # [N, 4096, vision_hidden] → [N, 256, text_hidden]
            n, _, d = x.shape
            grid = x.transpose(1, 2).reshape(n, d, patches_per_side, patches_per_side)
            pooled = nn.functional.avg_pool2d(grid, k).flatten(2).transpose(1, 2)
            xf = pooled.float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + self.mm_soft_emb_norm.weight.float())
            return (normed.type_as(x) @ self.mm_input_projection_weight).type_as(x)
    return GemmaProjector()


def gated_cross_attention_block(hidden: int = 4096, heads: int = 32, kv_heads: int = 8, head_dim: int = 128,
                                intermediate: int = 14336, eps: float = 1e-5):
    """Llama 3.2 Vision's cross-attention layer: text queries read image keys and values.

    h ← h + tanh(g_attn) · CrossAttn(norm(h), image);  h ← h + tanh(g_mlp) · MLP(norm(h)).
    Both gates start at 0, so the added layers begin as the identity and the text model is untouched.
    `mask` says which image positions each text token may read (the images before it, their real tiles).
    As in transformers, a row that may read nothing is left unmasked, and its MLP output is zeroed."""
    torch, nn = _torch()

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * self.weight

    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        def forward(self, h, image, mask):         # h [B, T, d], image [B, P, d], mask [B, T, P] bool
            B, T, _ = h.shape
            q = self.q_norm(self.q_proj(h).view(B, T, heads, head_dim)).transpose(1, 2)
            k = self.k_norm(self.k_proj(image).view(B, -1, kv_heads, head_dim)).transpose(1, 2)
            v = self.v_proj(image).view(B, -1, kv_heads, head_dim).transpose(1, 2)
            k, v = (t.repeat_interleave(heads // kv_heads, dim=1) for t in (k, v))
            s = (q @ k.transpose(-1, -2)) * head_dim ** -0.5
            s = s.masked_fill(~mask[:, None], torch.finfo(s.dtype).min)
            out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, heads * head_dim)
            return self.o_proj(out)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x):
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class GatedCrossAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(hidden)
            self.cross_attn = CrossAttention()
            self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))
            self.post_attention_layernorm = RMSNorm(hidden)
            self.mlp = MLP()
            self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

        def forward(self, h, image, mask):
            sees = mask.any(-1, keepdim=True)                            # [B, T, 1]: False for rows with no image
            a = self.cross_attn(self.input_layernorm(h), image, mask | ~sees)
            sees_image = sees.to(h.dtype)
            h = h + self.cross_attn_attn_gate.tanh() * a
            f = self.mlp(self.post_attention_layernorm(h))
            return h + self.cross_attn_mlp_gate.tanh() * (sees_image * f)
    return GatedCrossAttentionBlock()


# ------------------------------------------------------------------------------- transformers configs


def hf_config(model: str | VisionLanguageModel):
    """The transformers config of the full model, for parameter counts on the meta device."""
    import transformers as tf
    m = VLMS[model] if isinstance(model, str) else model
    e, d = m.encoder, m.decoder
    if m.key == 'llava':
        return tf.LlavaConfig(
            vision_config=tf.CLIPVisionConfig(hidden_size=e.hidden, intermediate_size=e.intermediate, num_hidden_layers=e.layers,
                                              num_attention_heads=e.heads, image_size=e.image_size, patch_size=e.patch, projection_dim=768),
            text_config=tf.LlamaConfig(hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                                       num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, vocab_size=d.vocab,
                                       max_position_embeddings=d.max_positions, rms_norm_eps=1e-5, pad_token_id=0),
            image_token_index=m.image_token_id, image_seq_length=LLAVA_IMAGE_TOKENS, vision_feature_layer=-2)
    if m.key == 'qwen':
        return tf.Qwen3VLConfig(
            vision_config=dict(depth=e.layers, hidden_size=e.hidden, num_heads=e.heads, intermediate_size=e.intermediate,
                               patch_size=e.patch, spatial_merge_size=QWEN_MERGE, temporal_patch_size=2, out_hidden_size=d.hidden,
                               num_position_embeddings=2304, deepstack_visual_indexes=[8, 16, 24], hidden_act='gelu_pytorch_tanh'),
            text_config=dict(hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                             num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, head_dim=d.head_dim, vocab_size=d.vocab,
                             max_position_embeddings=d.max_positions,
                             rope_parameters={'rope_type': 'default', 'rope_theta': 5_000_000.0, 'mrope_section': [24, 20, 20],
                                              'mrope_interleaved': True}),
            image_token_id=m.image_token_id, tie_word_embeddings=False)
    if m.key == 'llama':
        from transformers.models.mllama.configuration_mllama import (
            MllamaTextConfig,
            MllamaVisionConfig,
        )
        return tf.MllamaConfig(
            vision_config=MllamaVisionConfig(hidden_size=e.hidden, num_hidden_layers=e.layers, num_global_layers=e.global_layers,
                                             attention_heads=e.heads, intermediate_size=e.intermediate, image_size=e.image_size,
                                             patch_size=e.patch, max_num_tiles=LLAMA_MAX_TILES, vision_output_dim=7680,
                                             intermediate_layers_indices=[3, 7, 15, 23, 30],
                                             supported_aspect_ratios=[list(a) for a in tile_arrangements()]),
            text_config=MllamaTextConfig(hidden_size=d.hidden, num_hidden_layers=d.layers, num_attention_heads=d.heads,
                                         num_key_value_heads=d.kv_heads, intermediate_size=d.intermediate, vocab_size=d.vocab,
                                         cross_attention_layers=list(d.cross_layers), max_position_embeddings=d.max_positions),
            image_token_index=m.image_token_id)
    if m.key == 'gemma':
        return tf.Gemma3Config(
            vision_config=dict(hidden_size=e.hidden, intermediate_size=e.intermediate, num_hidden_layers=e.layers,
                               num_attention_heads=e.heads, image_size=e.image_size, patch_size=e.patch, vision_use_head=False),
            text_config=dict(hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                             num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, head_dim=d.head_dim, vocab_size=d.vocab,
                             sliding_window=d.sliding_window, max_position_embeddings=d.max_positions),
            mm_tokens_per_image=GEMMA_TOKENS, image_token_index=m.image_token_id)
    raise KeyError(m.key)
