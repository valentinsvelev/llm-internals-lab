"""Gated DeltaNet (Yang, Kautz & Hatamizadeh, 2024), as used in 36 of Qwen3-Next's 48 layers.

Instead of keeping every key and value, each value head keeps one matrix S that maps keys to values
and updates it once per token:

    α_t = exp(−e^A · softplus(a_t + δ))      decay: how much of the memory survives
    β_t = σ(b_t)                             write strength
    S_t = α_t S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ      the gated delta rule
    o_t = S_t q_t

Here S is stored transposed, [d_k, d_v], as in the reference code; the math is the same.

The rule is computed two ways: `delta_rule_recurrent` (one token at a time, what generation does) and
`delta_rule_chunked` (whole chunks with matrix products, what training and prefill do). They agree.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig
from .layers import GatedRMSNorm


def _prepare(q, k, v, g, beta):
    """Shared preprocessing: grouped value heads, float32, unit-length q and k, q scaled by 1/√d_k.
    Shapes in: q, k [B, T, Hk, dk], v [B, T, Hv, dv], g and beta [B, T, Hv]. Out: heads first."""
    r = v.shape[2] // k.shape[2]                 # each query/key head serves r value heads
    q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
    q, k, v, g, beta = (t.transpose(1, 2).float() for t in (q, k, v, g, beta))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    return q * q.shape[-1] ** -0.5, k, v, g, beta


def delta_rule_recurrent(q, k, v, g, beta, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """One step per token. Returns outputs [B, T, Hv, dv] and the final state [B, Hv, dk, dv]."""
    dtype = v.dtype
    q, k, v, g, beta = _prepare(q, k, v, g, beta)
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], device=v.device) if state is None else state.float()
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]                       # decay:  S ← α S
        recalled = torch.einsum('bhk,bhkv->bhv', k[:, :, t], S)          # what S stores under k_t
        delta = beta[:, :, t, None] * (v[:, :, t] - recalled)           # erase it, write v_t, strength β
        S = S + k[:, :, t, :, None] * delta[:, :, None, :]
        out[:, :, t] = torch.einsum('bhk,bhkv->bhv', q[:, :, t], S)      # read:   o_t = S_t q_t
    return out.transpose(1, 2).to(dtype), S


def delta_rule_chunked(q, k, v, g, beta, state: Tensor | None = None, chunk: int = 64) -> tuple[Tensor, Tensor]:
    """The same outputs, chunk by chunk. Inside a chunk the C sequential rank-1 updates are combined
    into matrix products: with D[i,j] = exp(γ_i − γ_j) the decay between positions j ≤ i (γ = cumulative
    log decay), solving the unit lower-triangular system (I + tril(βk kᵀ ⊙ D, −1)) once gives every
    token's corrected value u (the "WY representation"). Only the chunk-to-chunk scan is sequential."""
    dtype = v.dtype
    q, k, v, g, beta = _prepare(q, k, v, g, beta)
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:   # padded steps have β = 0 (no write) and g = 0 (no decay), so they change nothing
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (t.reshape(B, H, n, chunk, -1) for t in (q, k, v))
    g, beta = g.reshape(B, H, n, chunk), beta.reshape(B, H, n, chunk)
    gamma = g.cumsum(-1)                                                    # log decay since the chunk start
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = (gamma[..., :, None] - gamma[..., None, :]).masked_fill(~lower, float('-inf')).exp()
    kb, vb = k * beta[..., None], v * beta[..., None]
    system = (kb @ k.transpose(-1, -2)) * decay                             # the diagonal is ignored (unit triangular)
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, kb * gamma.exp()[..., None], upper=False, unitriangular=True)
    attn = (q @ k.transpose(-1, -2)) * decay                                # causal, decayed q·k inside the chunk
    q_dec = q * gamma.exp()[..., None]
    k_dec = k * (gamma[..., -1:] - gamma).exp()[..., None]
    S = torch.zeros(B, H, dk, dv, device=v.device) if state is None else state.float()
    out = torch.empty(B, H, n, chunk, dv, device=v.device)
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ S                                 # values corrected for what S already stores
        out[:, :, i] = q_dec[:, :, i] @ S + attn[:, :, i] @ v_new
        S = S * gamma[:, :, i, -1].exp()[..., None, None] + k_dec[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, n * chunk, dv)[:, :, :T]
    return out.transpose(1, 2).to(dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        nk, nv, d = cfg.lin_qk_heads, cfg.lin_v_heads, cfg.lin_head_dim
        self.nk, self.nv, self.d, self.kernel = nk, nv, d, cfg.conv_kernel
        self.q_proj = nn.Linear(cfg.hidden, nk * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, nk * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, nv * d, bias=False)
        self.z_proj = nn.Linear(cfg.hidden, nv * d, bias=False)        # output gate
        self.a_proj = nn.Linear(cfg.hidden, nv, bias=False)            # → decay α
        self.b_proj = nn.Linear(cfg.hidden, nv, bias=False)            # → write strength β
        self.channels = 2 * nk * d + nv * d
        self.conv = nn.Parameter(torch.randn(self.channels, cfg.conv_kernel) * 0.1)   # causal, one filter per channel
        self.A_log = nn.Parameter(torch.log(torch.empty(nv).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.ones(nv))
        self.out_norm = GatedRMSNorm(d, cfg.norm_eps)
        self.out_proj = nn.Linear(nv * d, cfg.hidden, bias=False)

    def forward(self, x: Tensor, positions: Tensor | None = None, cache: dict | None = None,
                mode: str = 'auto') -> Tensor:
        """`positions` is unused: these layers have no RoPE. `cache` holds the last k−1 convolution
        inputs and the state S. mode: 'chunked', 'recurrent', or 'auto' (recurrent for one cached token)."""
        B, T, _ = x.shape
        nk, nv, d = self.nk, self.nv, self.d
        mixed = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1).transpose(1, 2)   # [B, C, T]
        history = cache.get('conv') if cache else None
        if history is None:
            history = mixed.new_zeros(B, self.channels, self.kernel - 1)
        full = torch.cat([history, mixed], dim=-1)
        if cache is not None:
            cache['conv'] = full[..., -(self.kernel - 1):]
        mixed = F.silu(F.conv1d(full, self.conv[:, None, :], groups=self.channels)).transpose(1, 2)
        q, k, v = mixed.split([nk * d, nk * d, nv * d], dim=-1)
        q, k, v = q.view(B, T, nk, d), k.view(B, T, nk, d), v.view(B, T, nv, d)
        beta = torch.sigmoid(self.b_proj(x))
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)   # log α_t ≤ 0
        state = cache.get('state') if cache else None
        if mode == 'auto':
            mode = 'recurrent' if (state is not None and T == 1) else 'chunked'
        rule = delta_rule_recurrent if mode == 'recurrent' else delta_rule_chunked
        o, state = rule(q, k, v, g, beta, state)
        if cache is not None:
            cache['state'] = state
        z = self.z_proj(x).view(B, T, nv, d)
        o = self.out_norm(o, z)
        return self.out_proj(o.reshape(B, T, nv * d))
