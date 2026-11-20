"""How two audio-language models turn sound into tokens (the Audio tab of the Multimodal page).

Both use the encoder of Whisper large-v3 and differ in what they do with its output:

    qwen2audio  Qwen2-Audio-7B   average-pool 2 frames → 25 tokens per second, one clip of at most 30 s
    voxtral     Voxtral Mini 3B  each 30-s chunk encoded alone, 4 frames stacked → 12.5 tokens per second

Sound becomes a log-mel spectrogram first: 16,000 samples per second, a 400-sample window every
160 samples (100 frames per second), 128 mel bands. Whisper's encoder always reads 30 s (3,000 frames),
padded with silence, and halves the frame rate with a strided convolution: 1,500 positions.

The numbers come from the models' config.json and preprocessor_config.json; tests/test_audio_models.py
checks the token counts and the adapters against transformers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .vision import Decoder

SAMPLE_RATE = 16_000
HOP = 160                    # samples between spectrogram frames: 100 frames per second
N_FFT = 400                  # 25 ms window
MELS = 128
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE
CHUNK_FRAMES = CHUNK_SAMPLES // HOP          # 3,000 spectrogram frames
ENCODER_POSITIONS = CHUNK_FRAMES // 2        # 1,500 after the stride-2 convolution


@dataclass(frozen=True)
class AudioEncoder:
    name: str
    layers: int
    hidden: int
    heads: int
    intermediate: int
    mels: int


@dataclass(frozen=True)
class AudioLanguageModel:
    key: str
    name: str
    source: str
    encoder: AudioEncoder
    decoder: Decoder
    audio_token_id: int


WHISPER_LARGE_V3 = AudioEncoder('Whisper large-v3 encoder', layers=32, hidden=1280, heads=20, intermediate=5120, mels=MELS)

QWEN2_AUDIO = AudioLanguageModel(
    'qwen2audio', 'Qwen2-Audio-7B', 'Qwen/Qwen2-Audio-7B-Instruct', WHISPER_LARGE_V3,
    Decoder('Qwen-7B', layers=32, hidden=4096, heads=32, kv_heads=32, head_dim=128, intermediate=11008,
            vocab=156032, max_positions=8192),
    audio_token_id=151646)
VOXTRAL = AudioLanguageModel(
    'voxtral', 'Voxtral Mini 3B', 'mistralai/Voxtral-Mini-3B-2507', WHISPER_LARGE_V3,
    Decoder('Ministral 3B', layers=30, hidden=3072, heads=32, kv_heads=8, head_dim=128, intermediate=8192,
            vocab=131072, max_positions=131072),
    audio_token_id=24)

ALMS = {m.key: m for m in (QWEN2_AUDIO, VOXTRAL)}

VOXTRAL_STACK = 4                            # encoder frames per Voxtral token
VOXTRAL_TOKENS_PER_CHUNK = ENCODER_POSITIONS // VOXTRAL_STACK   # 375

# Parameters (audio encoder, adapter, language model incl. LM head), counted from transformers.
PARAMETERS = {
    'qwen2audio': (636_968_960, 5_246_976, 7_754_878_976),
    'voxtral': (636_968_960, 25_165_824, 4_014_136_320),
}


# -------------------------------------------------------------------------------------- what audio becomes


@dataclass(frozen=True)
class AudioPlan:
    samples: int                 # 16-kHz samples the model reads (after truncation)
    chunks: int                  # 30-second encoder windows
    mel_frames: int              # spectrogram frames that hold sound (not padding)
    encoder_positions: int       # encoder outputs computed, padding included
    tokens: int                  # audio tokens in the text sequence
    dropped_seconds: float       # audio beyond what the model reads


def audio_plan(model: str | AudioLanguageModel, seconds: float) -> AudioPlan:
    """What `seconds` of audio becomes in each model, as their processors compute it."""
    key = model if isinstance(model, str) else model.key
    samples = max(1, round(seconds * SAMPLE_RATE))
    if key == 'qwen2audio':
        # The feature extractor pads or truncates to 30 s; its mask marks the frames that hold sound.
        kept = min(samples, CHUNK_SAMPLES)
        frames = math.ceil(kept / HOP)
        after_conv = (frames - 1) // 2 + 1
        tokens = (after_conv - 2) // 2 + 1             # Qwen2AudioProcessor: average pooling by 2
        return AudioPlan(kept, 1, frames, ENCODER_POSITIONS, tokens, (samples - kept) / SAMPLE_RATE)
    if key == 'voxtral':
        # The processor pads to a whole number of 30-s chunks; every chunk gives 375 tokens.
        chunks = math.ceil(samples / CHUNK_SAMPLES)
        return AudioPlan(samples, chunks, math.ceil(samples / HOP), chunks * ENCODER_POSITIONS,
                         chunks * VOXTRAL_TOKENS_PER_CHUNK, 0.0)
    raise KeyError(key)


def audio_cache_values(model: str, seconds: float) -> int:
    """Keys and values the audio tokens store in the decoder's cache."""
    m = ALMS[model]
    return audio_plan(model, seconds).tokens * m.decoder.kv_values_per_token()


# ------------------------------------------------------------------------------------------------ adapters


def _torch():
    import torch
    from torch import nn
    return torch, nn


def qwen2_audio_adapter(audio_hidden: int = 1280, text_hidden: int = 4096):
    """Qwen2-Audio after the 32 encoder layers: average every 2 positions, the encoder's final LayerNorm,
    and one linear map. Weights: `layer_norm` is `audio_tower.layer_norm`, `linear` is
    `multi_modal_projector.linear`."""
    torch, nn = _torch()

    class Qwen2AudioAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = nn.LayerNorm(audio_hidden)
            self.linear = nn.Linear(audio_hidden, text_hidden)

        def forward(self, h):                          # [B, 1500, 1280]
            pooled = nn.functional.avg_pool1d(h.transpose(1, 2), 2, 2).transpose(1, 2)   # [B, 750, 1280]
            return self.linear(self.layer_norm(pooled))

    return Qwen2AudioAdapter()


def voxtral_adapter(audio_hidden: int = 1280, text_hidden: int = 3072, stack: int = VOXTRAL_STACK):
    """Voxtral after the encoder: 4 consecutive positions side by side (4 × 1,280 = 5,120), then
    Linear → GELU → Linear, as `multi_modal_projector.linear_1 / linear_2` (no biases)."""
    torch, nn = _torch()

    class VoxtralAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(stack * audio_hidden, text_hidden, bias=False)
            self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=False)

        def forward(self, h):                          # [chunks, 1500, 1280]
            stacked = h.reshape(-1, stack * audio_hidden)              # [chunks · 375, 5120]
            return self.linear_2(nn.functional.gelu(self.linear_1(stacked)))

    return VoxtralAdapter()


def log_mel_frames(samples: int) -> int:
    """Spectrogram frames Whisper's feature extractor computes for `samples` (centred windows)."""
    return samples // HOP + 1


def hf_config(model: str | AudioLanguageModel):
    """The transformers config of the full model, for parameter counts on the meta device."""
    import transformers as tf
    m = ALMS[model] if isinstance(model, str) else model
    e, d = m.encoder, m.decoder
    if m.key == 'qwen2audio':
        return tf.Qwen2AudioConfig(
            audio_config=dict(model_type='qwen2_audio_encoder', num_mel_bins=e.mels, encoder_layers=e.layers, encoder_attention_heads=e.heads,
                              encoder_ffn_dim=e.intermediate, d_model=e.hidden, max_source_positions=ENCODER_POSITIONS),
            text_config=dict(model_type='qwen2', hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                             num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, vocab_size=d.vocab,
                             max_position_embeddings=d.max_positions, tie_word_embeddings=False),
            audio_token_index=m.audio_token_id)
    if m.key == 'voxtral':
        return tf.VoxtralConfig(
            audio_config=dict(hidden_size=e.hidden, intermediate_size=e.intermediate, num_hidden_layers=e.layers, num_attention_heads=e.heads,
                              num_mel_bins=e.mels, max_source_positions=ENCODER_POSITIONS),
            text_config=dict(hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers, num_attention_heads=d.heads,
                             num_key_value_heads=d.kv_heads, head_dim=d.head_dim, vocab_size=d.vocab, max_position_embeddings=d.max_positions),
            audio_token_id=m.audio_token_id)
    raise KeyError(m.key)
