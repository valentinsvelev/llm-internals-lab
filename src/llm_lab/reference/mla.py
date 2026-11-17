"""Multi-head Latent Attention (DeepSeek-V2/V3) and DeepSeek Sparse Attention (DeepSeek-V3.2).

MLA caches one normalized latent c (d_c numbers) and one rotary key k^R (d_R numbers) per token.
Per-head keys and values are rebuilt from c:

    k_h = [ c W^UK_h ,  RoPE(k^R) ]        v_h = c W^UV_h

Two equivalent ways to attend are implemented:
  * `expanded`: rebuild K and V for every head, then ordinary attention (how training is usually written);
  * `absorbed`: fold W^UK into the query and W^UV into the output, attend over the cached latents
    directly (how inference is done: K and V are never materialized). The test suite checks they agree.

With `index_heads > 0` a lightning indexer (DSA) restricts every query to its top-k earlier tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig
from .layers import RMSNorm, Rotary, causal_mask, masked_softmax_attention, yarn_softmax_scale


class LightningIndexer(nn.Module):
    """DeepSeek Sparse Attention's indexer:  I[t,s] = Σ_j w[t,j] · ReLU(q^I[t,j] · k^I[s]),
    then keep the top-k tokens s ≤ t for query t.

    The reference implementation runs this in FP8 after a Hadamard rotation; both preserve the dot
    products up to rounding, so this float version selects the same tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads, self.dim, self.topk = cfg.index_heads, cfg.index_dim, cfg.index_topk
        self.q_up = nn.Linear(cfg.q_rank, self.heads * self.dim, bias=False)   # from MLA's query latent c^Q
        self.k_proj = nn.Linear(cfg.hidden, self.dim, bias=False)              # one key per token, all heads share it
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.head_weights = nn.Linear(cfg.hidden, self.heads, bias=False)      # w[t, j]
        # the indexer rotates its first d_R dims with the half-split layout (MLA itself uses interleaved pairs)
        self.rotary = Rotary(cfg.rope_head_dim, cfg.rope_theta, cfg.yarn, interleaved=False)

    def forward(self, x: Tensor, q_latent: Tensor, positions: Tensor, allowed: Tensor, cache: dict | None) -> Tensor:
        """Boolean mask [B, 1, T, T_keys]: True where query t keeps key s."""
        B, T, _ = x.shape
        cos, sin = self.rotary.angles(positions)
        q = self.rotary(self.q_up(q_latent).view(B, T, self.heads, self.dim).transpose(1, 2), cos, sin)
        k = self.rotary(self.k_norm(self.k_proj(x)), cos, sin)                  # [B, T, dim]
        if cache is not None:
            if 'k_index' in cache:
                k = torch.cat([cache['k_index'], k], dim=1)
            cache['k_index'] = k
        scores = F.relu((q.float() @ k.float().transpose(-1, -2).unsqueeze(1)) * self.dim ** -0.5)   # [B, heads, T, Tk]
        w = self.head_weights(x).float() * self.heads ** -0.5                                         # [B, T, heads]
        index = torch.einsum('bth,bhts->bts', w, scores).masked_fill(~allowed[:, 0], float('-inf'))
        top = index.topk(min(self.topk, index.shape[-1]), dim=-1).indices
        keep = torch.zeros_like(index, dtype=torch.bool).scatter_(-1, top, True)
        return keep.unsqueeze(1)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, n, r, dv = cfg.heads, cfg.nope_dim, cfg.rope_head_dim, cfg.v_dim
        self.H, self.n, self.r, self.dv, self.dc = H, n, r, dv, cfg.kv_rank
        if cfg.q_rank:     # DeepSeek-V3/V3.2 compress queries too:  q = RMSNorm(x W^DQ) W^UQ
            self.q_down = nn.Linear(cfg.hidden, cfg.q_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_rank, 1e-6)
            self.q_up = nn.Linear(cfg.q_rank, H * (n + r), bias=False)
        else:              # DeepSeek-V2-Lite projects queries directly
            self.q_proj = nn.Linear(cfg.hidden, H * (n + r), bias=False)
        self.kv_down = nn.Linear(cfg.hidden, cfg.kv_rank + r, bias=False)   # [W^DKV  W^KR]: latent and rotary key
        self.kv_norm = RMSNorm(cfg.kv_rank, 1e-6)
        self.kv_up = nn.Linear(cfg.kv_rank, H * (n + dv), bias=False)       # [W^UK  W^UV] for every head
        self.o_proj = nn.Linear(H * dv, cfg.hidden, bias=False)
        self.rotary = Rotary(r, cfg.rope_theta, cfg.yarn, interleaved=cfg.rope_interleaved)
        self.scale = (n + r) ** -0.5 * yarn_softmax_scale(cfg.yarn)
        self.indexer = LightningIndexer(cfg) if cfg.index_heads else None

    def forward(self, x: Tensor, positions: Tensor, cache: dict | None = None, absorbed: bool | None = None) -> Tensor:
        B, T, _ = x.shape
        H, n, r = self.H, self.n, self.r
        q_latent = self.q_norm(self.q_down(x)) if hasattr(self, 'q_down') else None
        q = (self.q_up(q_latent) if q_latent is not None else self.q_proj(x)).view(B, T, H, n + r).transpose(1, 2)
        q_nope, q_rope = q.split([n, r], dim=-1)
        c, k_rope = self.kv_down(x).split([self.dc, r], dim=-1)
        c = self.kv_norm(c)                                         # the latent: this is what is cached
        cos, sin = self.rotary.angles(positions)
        q_rope = self.rotary(q_rope, cos, sin)
        k_rope = self.rotary(k_rope, cos, sin)                      # one rotary key, shared by all heads
        if cache is not None:
            if 'c' in cache:
                c, k_rope = torch.cat([cache['c'], c], dim=1), torch.cat([cache['k_rope'], k_rope], dim=1)
            cache['c'], cache['k_rope'] = c, k_rope
        allowed = causal_mask(T, c.shape[1], x.device)[None, None]
        if self.indexer is not None:                                # DSA: keep only the indexer's top-k tokens
            allowed = allowed & self.indexer(x, q_latent, positions, allowed, cache)
        if absorbed is None:
            absorbed = cache is not None and T == 1                 # decode steps attend over latents directly
        o = (self._absorbed if absorbed else self._expanded)(q_nope, q_rope, c, k_rope, allowed)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * self.dv))

    def _expanded(self, q_nope, q_rope, c, k_rope, allowed):
        B, Tk = c.shape[:2]
        kv = self.kv_up(c).view(B, Tk, self.H, self.n + self.dv).transpose(1, 2)
        k_nope, v = kv.split([self.n, self.dv], dim=-1)
        k = torch.cat([k_nope, k_rope[:, None].expand(-1, self.H, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        return masked_softmax_attention(q, k, v, allowed, self.scale)

    def _absorbed(self, q_nope, q_rope, c, k_rope, allowed):
        W = self.kv_up.weight.view(self.H, self.n + self.dv, self.dc)
        W_uk, W_uv = W[:, :self.n], W[:, self.n:]                  # [H, n, d_c], [H, d_v, d_c]
        # q_nope · (c W^UK)ᵀ = (q_nope W^UKᵀ) · cᵀ: fold W^UK into the query, score against latents
        q_lat = torch.einsum('bhtn,hnc->bhtc', q_nope.float(), W_uk.float())
        scores = q_lat @ c.float().transpose(-1, -2)[:, None] + q_rope.float() @ k_rope.float().transpose(-1, -2)[:, None]
        scores = (scores * self.scale).masked_fill(~allowed, float('-inf'))
        o_lat = scores.softmax(-1) @ c.float()[:, None]             # mix latents, then expand once with W^UV
        return torch.einsum('bhtc,hvc->bhtv', o_lat, W_uv.float()).to(q_nope.dtype)
