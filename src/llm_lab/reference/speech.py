"""How two models speak (the Speech out tab), and how Qwen3-Omni lays out sound and video in one
sequence (the All together tab).

Both turn speech into a neural codec's tokens: 12.5 frames per second, several codebooks per frame,
each frame a stack of codes from coarse to fine (residual vector quantization).

    qwen3omni  Qwen3-Omni-30B-A3B   a Thinker (the 30B-A3B language model) writes text; a Talker (3B MoE)
                                    reads its hidden states and writes codebook 1 of each frame; a 5-layer
                                    code predictor adds codebooks 2–16; Code2Wav turns them into 24 kHz audio
    moshi      Moshi (moshiko)      one 7B model that listens and speaks at once: every 80 ms step it reads
                                    17 streams (its text, its 8 codebooks, the user's 8) and a 6-layer
                                    depth transformer writes its 8 codes for the step; Mimi decodes them

Numbers from Qwen/Qwen3-Omni-30B-A3B-Instruct's and kmhf/hf-moshiko's config.json;
tests/test_speech_models.py checks them against transformers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

OUT_SAMPLE_RATE = 24_000
FRAME_RATE = 12.5                     # codec frames per second, both models
SAMPLES_PER_FRAME = 1920              # 24,000 / 12.5


@dataclass(frozen=True)
class SpeechModel:
    key: str
    name: str
    source: str
    codebooks: int                    # codes per frame the model writes
    codebook_size: int
    main: str                         # the network that runs once per frame
    main_layers: int
    main_hidden: int
    main_kv_heads: int
    main_head_dim: int
    inner: str                        # the network that fills in the rest of the frame
    inner_layers: int
    inner_hidden: int
    decoder: str                      # codes → waveform


QWEN3_OMNI = SpeechModel(
    'qwen3omni', 'Qwen3-Omni-30B-A3B', 'Qwen/Qwen3-Omni-30B-A3B-Instruct', codebooks=16, codebook_size=2048,
    main='Talker (MoE, 128 experts, 6 active)', main_layers=20, main_hidden=1024, main_kv_heads=2, main_head_dim=128,
    inner='code predictor', inner_layers=5, inner_hidden=1024, decoder='Code2Wav')
MOSHI = SpeechModel(
    'moshi', 'Moshi', 'kmhf/hf-moshiko', codebooks=8, codebook_size=2048,
    main='Helium temporal transformer', main_layers=32, main_hidden=4096, main_kv_heads=32, main_head_dim=128,
    inner='depth transformer', inner_layers=6, inner_hidden=1024, decoder='Mimi')

SPEECH_MODELS = {m.key: m for m in (QWEN3_OMNI, MOSHI)}

# Qwen3-Omni: (Thinker, Talker incl. code predictor, Code2Wav); Moshi: (Mimi, Helium incl. embeddings, depth).
PARAMETERS = {
    'qwen3omni': (31_719_205_488, 3_324_596_480, 216_016_577),
    'moshi': (79_308_609, 6_973_366_272, 714_362_880),
}
QWEN3_OMNI_THINKER = {'audio': 647_927_168, 'vision': 538_631_408, 'text': 30_532_646_912}   # text incl. LM head
CODE2WAV_UPSAMPLE = (8, 5, 4, 3, 2, 2)    # upsample_rates [8, 5, 4, 3] after upsampling_ratios [2, 2]: 1,920
MIMI_CODEBOOKS = 32                       # Mimi has 32; Moshi uses the first 8
MOSHI_STREAMS = 1 + 2 * MOSHI.codebooks   # its text, its 8 codebooks, the user's 8


@dataclass(frozen=True)
class SpeechPlan:
    frames: int              # codec frames
    codes: int               # codes written
    main_steps: int          # passes of the per-frame network
    inner_steps: int         # passes of the per-code network
    samples: int             # 24-kHz samples out
    kv_values: int           # keys and values the per-frame network caches for this speech


def speech_plan(model: str, seconds: float) -> SpeechPlan:
    m = SPEECH_MODELS[model]
    frames = math.ceil(seconds * FRAME_RATE - 1e-9)
    kv = frames * 2 * m.main_kv_heads * m.main_head_dim * m.main_layers
    if model == 'qwen3omni':
        # Talker: codebook 1 of each frame; the code predictor: codebooks 2–16, one pass each.
        return SpeechPlan(frames, frames * m.codebooks, frames, frames * (m.codebooks - 1), frames * SAMPLES_PER_FRAME, kv)
    if model == 'moshi':
        # Helium: one pass per 80-ms step; the depth transformer: one pass per code (its text token comes from Helium).
        return SpeechPlan(frames, frames * m.codebooks, frames, frames * m.codebooks, frames * SAMPLES_PER_FRAME, kv)
    raise KeyError(model)


def moshi_delays(codebooks: int = MOSHI.codebooks) -> list[int]:
    """Steps each codebook lags behind: the semantic codebook 1 is on time, the acoustic 2–8 one step late."""
    return [0] + [1] * (codebooks - 1)


def delayed_grid(codes, delays):
    """Shift each codebook row right by its delay (padding with None), the layout Moshi is trained on."""
    steps = len(codes[0]) + max(delays)
    return [[None] * d + list(row) + [None] * (steps - d - len(row)) for row, d in zip(codes, delays)]


# --------------------------------------------------------------------------------- all together: Qwen3-Omni

POSITIONS_PER_SECOND = 13             # Qwen3-Omni: one position id per audio token, 13 per second
OMNI_VIDEO_FPS = 2
OMNI_SECONDS_PER_GRID = 1.0           # 2 frames per temporal grid at 2 fps


def omni_audio_tokens(mel_frames: int, n_window: int = 50) -> int:
    """Qwen3-Omni's audio encoder: 100 mel frames per second, 13 tokens per 100-frame chunk (8× downsampling)."""
    chunk = 2 * n_window
    rest = mel_frames % chunk
    feat = (rest - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // chunk) * 13


def omni_interleave(grids: int, tokens_per_grid: int, audio_tokens: int, seconds_per_grid: float = OMNI_SECONDS_PER_GRID,
                    positions_per_second: float = POSITIONS_PER_SECOND) -> str:
    """The order of a video's tokens ('v') and its sound's ('a'), as Qwen3-Omni's processor writes them:
    each video grid at its start time, then the audio tokens up to the next grid. Times are counted in
    position ids, one per audio token."""
    video_t = [g * seconds_per_grid * positions_per_second for g in range(grids) for _ in range(tokens_per_grid)]
    out, i, j = [], 0, 0
    while i < len(video_t) and j < audio_tokens:
        if video_t[i] <= j:
            out.append('v')
            i += 1
        else:
            out.append('a')
            j += 1
    out += ['v'] * (len(video_t) - i) + ['a'] * (audio_tokens - j)
    return ''.join(out)


def hf_config(model: str):
    import transformers as tf
    if model == 'moshi':
        return tf.MoshiConfig(audio_vocab_size=MOSHI.codebook_size)
    if model == 'qwen3omni':
        return tf.Qwen3OmniMoeConfig(
            thinker_config=dict(
                audio_config=dict(num_mel_bins=128, encoder_layers=32, encoder_attention_heads=20, encoder_ffn_dim=5120, d_model=1280,
                                  output_dim=2048, n_window=50, n_window_infer=800, conv_chunksize=500, downsample_hidden_size=480,
                                  max_source_positions=1500),
                vision_config=dict(depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304, patch_size=16, spatial_merge_size=2,
                                   temporal_patch_size=2, out_hidden_size=2048, deepstack_visual_indexes=[8, 16, 24]),
                text_config=dict(hidden_size=2048, num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
                                 intermediate_size=768, moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8,
                                 vocab_size=152064, max_position_embeddings=65536, decoder_sparse_step=1),
                position_id_per_seconds=POSITIONS_PER_SECOND),
            talker_config=dict(
                accept_hidden_layer=24, num_code_groups=QWEN3_OMNI.codebooks, thinker_hidden_size=2048, spatial_merge_size=2,
                position_id_per_seconds=POSITIONS_PER_SECOND,
                text_config=dict(hidden_size=1024, num_hidden_layers=20, num_attention_heads=16, num_key_value_heads=2, head_dim=128,
                                 intermediate_size=2048, moe_intermediate_size=384, num_experts=128, num_experts_per_tok=6,
                                 shared_expert_intermediate_size=768, vocab_size=3072),
                code_predictor_config=dict(hidden_size=1024, num_hidden_layers=5, num_attention_heads=16, num_key_value_heads=8,
                                           head_dim=128, intermediate_size=3072, vocab_size=2048, num_code_groups=QWEN3_OMNI.codebooks)),
            code2wav_config=dict(codebook_size=2048, codebook_dim=512, semantic_codebook_size=4096, hidden_size=1024, num_hidden_layers=8,
                                 num_attention_heads=16, num_key_value_heads=16, intermediate_size=3072, decoder_dim=1536,
                                 upsample_rates=[8, 5, 4, 3], upsampling_ratios=[2, 2], num_quantizers=16, num_semantic_quantizers=1,
                                 sliding_window=72),
            enable_audio_output=True)
    raise KeyError(model)


@dataclass(frozen=True)
class OmniPlan:
    grids: int               # video grids: 2 frames, 1 s each (2 frames per second, as Qwen's own utilities sample)
    tokens_per_grid: int
    audio_tokens: int
    order: str               # 'v' and 'a' in sequence order
    positions: int           # time positions the clip spans (13 per second)


def omni_plan(seconds: int, height: int = 360, width: int = 640) -> OmniPlan:
    """A clip with sound in Qwen3-Omni: one video grid per second, its sound at 13 tokens per second, interleaved."""
    from .vision import smart_resize
    rh, rw = smart_resize(height, width)
    per_grid = (rh // 32) * (rw // 32)
    audio = omni_audio_tokens(seconds * 100)
    return OmniPlan(seconds, per_grid, audio, omni_interleave(seconds, per_grid, audio), seconds * POSITIONS_PER_SECOND)
