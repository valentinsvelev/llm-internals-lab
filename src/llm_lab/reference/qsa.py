"""Qwen Sparse Attention (QSA): one attention layer of Qwen3.8-Flash-Next.

A lightweight indexer picks whole blocks of `block` tokens:
  * every token stores one raw index key (one shared key head of `index_dim`);
  * for a query, the visible tokens are cut into complete blocks; each block's raw keys are averaged,
    normalized and rotated at the block's first position, giving one block key;
  * block score = Σ_j ReLU(q^I_j · k̄_b) / √index_dim over the `index_heads` query heads;
  * the best `budget / block` blocks are kept, plus the tokens of the current, incomplete block.
Then gated grouped-query attention (as in Qwen3-Next) runs over the kept positions only, with the
original, uncompressed keys and values.

The block pooling is written per query, as a loop, to keep it readable: this is a reference, not a kernel.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import QSAConfig
from .layers import RMSNorm, Rotary, causal_mask, masked_softmax_attention


class QSAIndexer(nn.Module):
    def __init__(self, cfg: QSAConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden, cfg.index_heads * cfg.index_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.index_dim, bias=False)
        self.q_norm = RMSNorm(cfg.index_dim, cfg.norm_eps, offset=True)
        self.k_norm = RMSNorm(cfg.index_dim, cfg.norm_eps, offset=True)

    def forward(self, x: Tensor, positions: Tensor, rotary: Rotary, allowed: Tensor, cache: dict | None) -> Tensor:
        c = self.cfg
        B, T, _ = x.shape
        cos, sin = rotary.angles(positions)
        q = rotary(self.q_norm(self.q_proj(x).view(B, T, c.index_heads, c.index_dim)).transpose(1, 2), cos, sin)
        raw = self.k_proj(x)                                    # raw keys are cached; pooling happens per query
        if cache is not None:
            if 'k_index' in cache:
                raw = torch.cat([cache['k_index'], raw], dim=1)
            cache['k_index'] = raw
        Tk = raw.shape[1]
        keep = torch.zeros(B, T, Tk, dtype=torch.bool, device=x.device)
        for b in range(B):
            for t in range(T):
                visible = allowed[0, 0, t].nonzero().flatten()
                n_blocks = visible.numel() // c.block
                tail = visible[n_blocks * c.block:]             # the current, incomplete block: always kept
                keep[b, t, tail] = True
                if n_blocks == 0:
                    continue
                blocks = visible[:n_blocks * c.block].view(n_blocks, c.block)
                pooled = self.k_norm(raw[b, blocks].float().mean(1).to(raw.dtype))     # one key per block
                bc, bs = rotary.angles(blocks[:, 0])                                    # rotated at the block start
                block_keys = rotary(pooled, bc, bs)
                scores = torch.relu(q[b, :, t].float() @ block_keys.float().T).sum(0) / math.sqrt(c.index_dim)
                best = scores.topk(min(c.top_blocks, n_blocks)).indices
                keep[b, t, blocks[best].flatten()] = True
        return keep.unsqueeze(1)


class QwenSparseAttention(nn.Module):
    def __init__(self, cfg: QSAConfig):
        super().__init__()
        H, G, d = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.H, self.G, self.d = H, G, d
        self.q_proj = nn.Linear(cfg.hidden, H * d, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden, H * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, G * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, G * d, bias=False)
        self.o_proj = nn.Linear(H * d, cfg.hidden, bias=False)
        self.q_norm = RMSNorm(d, cfg.norm_eps, offset=True)
        self.k_norm = RMSNorm(d, cfg.norm_eps, offset=True)
        self.rotary = Rotary(cfg.rope_dim, cfg.rope_theta)
        self.indexer = QSAIndexer(cfg)

    def forward(self, x: Tensor, positions: Tensor, cache: dict | None = None) -> Tensor:
        B, T, _ = x.shape
        H, G, d = self.H, self.G, self.d
        Tk = T + (cache['k'].shape[2] if cache and 'k' in cache else 0)
        allowed = causal_mask(T, Tk, x.device)[None, None]
        allowed = allowed & self.indexer(x, positions, self.rotary, allowed, cache)
        q = self.q_norm(self.q_proj(x).view(B, T, H, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, G, d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, G, d).transpose(1, 2)
        cos, sin = self.rotary.angles(positions)
        q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        if cache is not None:
            if 'k' in cache:
                k, v = torch.cat([cache['k'], k], dim=2), torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        k, v = k.repeat_interleave(H // G, dim=1), v.repeat_interleave(H // G, dim=1)
        o = masked_softmax_attention(q, k, v, allowed, d ** -0.5).transpose(1, 2).reshape(B, T, H * d)
        return self.o_proj(o * torch.sigmoid(self.gate_proj(x)))
