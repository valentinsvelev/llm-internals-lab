"""Softmax attention with H query heads and G key/value heads.

One module covers the first three tabs of the explorer:
  MHA  G = H   (GPT-2: 12 / 12)
  MQA  G = 1   (Gemma 2B: 8 / 1)
  GQA  1 < G < H, query head h reads KV head ⌈h·G/H⌉  (Qwen2.5-0.5B: 14 / 2)
and, with `qk_norm`, `output_gate` and a partial `rope_dim`, the gated attention layers of Qwen3-Next.

The KV cache stores K and V with G heads, never H: that is the memory MQA and GQA save.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .layers import RMSNorm, Rotary, causal_mask, masked_softmax_attention


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, G, d = cfg.heads, cfg.kv_heads, cfg.head_dim
        assert H % G == 0, 'query heads must split evenly into KV groups'
        self.H, self.G, self.d = H, G, d
        self.q_proj = nn.Linear(cfg.hidden, H * d, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden, G * d, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden, G * d, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(H * d, cfg.hidden, bias=cfg.out_bias)
        # Qwen3-Next: a per-channel sigmoid gate on the output, and RMSNorm on every q and k head
        self.gate_proj = nn.Linear(cfg.hidden, H * d, bias=False) if cfg.output_gate else None
        self.q_norm = RMSNorm(d, cfg.norm_eps, cfg.norm_offset) if cfg.qk_norm else None
        self.k_norm = RMSNorm(d, cfg.norm_eps, cfg.norm_offset) if cfg.qk_norm else None
        # GPT-2 has learned absolute positions (added to the embeddings) and no rotation here
        self.rotary = None if cfg.learned_positions else Rotary(cfg.rotary_dim, cfg.rope_theta)
        self.scale = d ** -0.5

    def forward(self, x: Tensor, positions: Tensor, cache: dict | None = None) -> Tensor:
        B, T, _ = x.shape
        H, G, d = self.H, self.G, self.d
        # every token gets its own query (H heads), key and value (G heads each)
        q = self.q_proj(x).view(B, T, H, d)
        k = self.k_proj(x).view(B, T, G, d)
        v = self.v_proj(x).view(B, T, G, d)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # [B, heads, T, d]
        if self.rotary is not None:                                 # rotate q and k by position, never v
            cos, sin = self.rotary.angles(positions)
            q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        if cache is not None:                                       # append to the KV cache, read it all back
            if 'k' in cache:
                k, v = torch.cat([cache['k'], k], dim=2), torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        # each KV head serves H/G consecutive query heads
        k, v = k.repeat_interleave(H // G, dim=1), v.repeat_interleave(H // G, dim=1)
        o = masked_softmax_attention(q, k, v, causal_mask(T, k.shape[2], x.device), self.scale)
        o = o.transpose(1, 2).reshape(B, T, H * d)
        if self.gate_proj is not None:
            o = o * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(o)
