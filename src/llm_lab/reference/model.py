"""A decoder-only language model assembled from a ModelConfig: the "whole model" of the explorer.

    h₀ = E[x] (+ P[positions] for GPT-2, · √d for Gemma)
    for every layer:  h ← h + Mixer(Norm(h));  h ← h + FFN(Norm(h))
    logits = Norm_f(h) W_U

Mixer is softmax attention (MHA/MQA/GQA), MLA (with the DSA indexer for DeepSeek-V3.2) or Gated
DeltaNet, chosen per layer by the config. `Cache` holds what each layer keeps between decoding steps.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .gated_deltanet import GatedDeltaNet
from .layers import make_ffn, make_norm
from .mla import MultiHeadLatentAttention


class Cache:
    """Per-layer state for generation. Attention layers grow with every token (K and V, or MLA's
    latent and rotary key, plus DSA's index keys); Gated DeltaNet layers keep a fixed state."""

    def __init__(self, layers: int):
        self.layers: list[dict] = [{} for _ in range(layers)]
        self.length = 0                         # tokens processed so far

    def numbers_per_layer(self) -> list[int]:
        """How many numbers each layer currently stores (per sequence in the batch)."""
        return [sum(t[0].numel() for t in layer.values()) for layer in self.layers]


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int):
        super().__init__()
        self.kind = cfg.layer_type(i)
        self.norm1, self.norm2 = make_norm(cfg), make_norm(cfg)
        if self.kind == 'gdn':
            self.mixer = GatedDeltaNet(cfg)
        elif cfg.attention == 'mla':
            self.mixer = MultiHeadLatentAttention(cfg)
        else:
            self.mixer = GroupedQueryAttention(cfg)
        self.ffn = make_ffn(cfg, i)

    def forward(self, h: Tensor, positions: Tensor, cache: dict | None) -> Tensor:
        h = h + self.mixer(self.norm1(h), positions, cache)
        return h + self.ffn(self.norm2(h))


class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos_embed = nn.Embedding(cfg.max_positions, cfg.hidden) if cfg.learned_positions else None
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.norm = make_norm(cfg)
        self.lm_head = None if cfg.tied else nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(self, ids: Tensor, cache: Cache | None = None) -> Tensor:
        """ids [B, T] → logits [B, T, V]. With a cache, ids are the new tokens only."""
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        h = self.embed(ids)
        if self.cfg.embed_scale:
            h = h * torch.tensor(math.sqrt(self.cfg.hidden), dtype=h.dtype)
        if self.pos_embed is not None:
            h = h + self.pos_embed(positions)
        for i, block in enumerate(self.blocks):
            h = block(h, positions, cache.layers[i] if cache is not None else None)
        if cache is not None:
            cache.length += ids.shape[1]
        h = self.norm(h)
        return h @ (self.embed.weight if self.lm_head is None else self.lm_head.weight).T

    def new_cache(self) -> Cache:
        return Cache(self.cfg.layers)

    @torch.no_grad()
    def generate(self, ids: Tensor, steps: int, pick=None) -> Tensor:
        """Append `steps` tokens. `pick(logits [B, V]) → [B]` chooses them (default: greedy).
        The prompt runs once; after that only the new token passes through the model."""
        from .decoding import greedy
        pick = pick or greedy
        cache = self.new_cache()
        logits = self(ids, cache)[:, -1]
        for _ in range(steps):
            nxt = pick(logits)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            logits = self(nxt[:, None], cache)[:, -1]
        return ids
