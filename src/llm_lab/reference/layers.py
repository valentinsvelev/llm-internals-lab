"""Building blocks shared by the reference models: norms, activations, RoPE (with YaRN),
feed-forward layers and mixtures of experts.

Everything is written for clarity first: no fused kernels, float32 math where the original
implementations upcast, and comments that map each line to the equations on the explorer page.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig, MoE, YaRN

# ---------------------------------------------------------------- normalization

class RMSNorm(nn.Module):
    """x / rms(x) · w.  With `offset`, the weight is stored as w and applied as (1 + w), as in Gemma
    and Qwen3-Next (so a zero-initialized weight is the identity)."""

    def __init__(self, dim: int, eps: float = 1e-6, offset: bool = False):
        super().__init__()
        self.eps, self.offset = eps, offset
        self.weight = nn.Parameter(torch.zeros(dim) if offset else torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.offset:
            return (x * (1.0 + self.weight.float())).to(dtype)
        return self.weight * x.to(dtype)


def make_norm(cfg: ModelConfig, dim: int | None = None) -> nn.Module:
    dim = dim or cfg.hidden
    if cfg.norm == 'layer':
        return nn.LayerNorm(dim, eps=cfg.norm_eps)
    return RMSNorm(dim, cfg.norm_eps, cfg.norm_offset)


class GatedRMSNorm(nn.Module):
    """Gated DeltaNet's output norm: RMSNorm(o) · w ⊙ SiLU(z)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(gate.float())).to(dtype)


# ---------------------------------------------------------------- activations

def gelu_tanh(x: Tensor) -> Tensor:
    """GELU(x) ≈ x/2 · (1 + tanh(√(2/π) · (x + 0.044715 x³))), the approximation GPT-2 and Gemma use."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


ACTIVATIONS = {'silu': F.silu, 'gelu_tanh': gelu_tanh}


# ---------------------------------------------------------------- rotary position embedding

def yarn_inv_freq(dim: int, base: float, yarn: YaRN) -> tuple[Tensor, float]:
    """YaRN: keep the fast-rotating pairs (they encode local order), slow the slow ones down by
    `factor` (so long contexts stay in the trained range), and blend linearly in between.
    Returns the per-pair frequencies and the factor applied to cos and sin."""
    def correction_dim(rotations: float) -> float:
        # the pair index whose wavelength fits `rotations` times into the original context
        return dim * math.log(yarn.original_max_positions / (rotations * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(correction_dim(yarn.beta_fast)), 0)
    high = min(math.ceil(correction_dim(yarn.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    interpolation = extrapolation / yarn.factor
    inv_freq = interpolation * ramp + extrapolation * (1 - ramp)

    def mscale(m: float) -> float:
        return 0.1 * m * math.log(yarn.factor) + 1.0 if yarn.factor > 1 else 1.0
    return inv_freq, mscale(yarn.mscale) / mscale(yarn.mscale_all_dim)


def yarn_softmax_scale(yarn: YaRN | None) -> float:
    """DeepSeek multiplies the attention scale by mscale² (1.59 for V2-Lite, 1.87 for V3.2)."""
    if yarn is None or yarn.factor <= 1 or not yarn.mscale_all_dim:
        return 1.0
    m = 0.1 * yarn.mscale_all_dim * math.log(yarn.factor) + 1.0
    return m * m


class Rotary(nn.Module):
    """RoPE over the first `dim` dimensions of a head. Pair i turns by position · θ_i, θ_i = base^(−2i/dim).

    `interleaved=False` pairs dimension i with i + dim/2 (GPT-NeoX / Llama layout);
    `interleaved=True` pairs 2i with 2i + 1 (the layout DeepSeek uses for MLA)."""

    def __init__(self, dim: int, base: float, yarn: YaRN | None = None, interleaved: bool = False):
        super().__init__()
        self.dim, self.interleaved = dim, interleaved
        if yarn is None:
            inv_freq, self.scale = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim), 1.0
        else:
            inv_freq, self.scale = yarn_inv_freq(dim, base, yarn)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def angles(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """cos and sin of every pair's angle, [T, dim/2]."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        return freqs.cos() * self.scale, freqs.sin() * self.scale

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate x [..., T, d] (d ≥ dim); dimensions beyond `dim` pass through unchanged."""
        rot, rest = x[..., :self.dim], x[..., self.dim:]
        if self.interleaved:
            x1, x2 = rot[..., 0::2], rot[..., 1::2]
            # output layout follows the reference code: rotated evens first, then rotated odds
            rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        else:
            x1, x2 = rot[..., :self.dim // 2], rot[..., self.dim // 2:]
            rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([rot.to(x.dtype), rest], dim=-1)


# ---------------------------------------------------------------- feed-forward

class MLP(nn.Module):
    """GPT-2: FFN(u) = GELU(u W₁ + b₁) W₂ + b₂."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.hidden, cfg.intermediate, bias=cfg.mlp_bias)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden, bias=cfg.mlp_bias)
        self.act = ACTIVATIONS[cfg.act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.up(x)))


class GLU(nn.Module):
    """Gated MLP: FFN(u) = (act(u W_gate) ⊙ u W_up) W_down  (SwiGLU with SiLU, GeGLU with GELU)."""

    def __init__(self, hidden: int, inner: int, act: str = 'silu', bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(hidden, inner, bias=bias)
        self.up = nn.Linear(hidden, inner, bias=bias)
        self.down = nn.Linear(inner, hidden, bias=bias)
        self.act = ACTIVATIONS[act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class MixtureOfExperts(nn.Module):
    """Routed experts plus shared experts. Three routers cover the three models:

    - softmax, plain top-k weights (DeepSeek-V2-Lite);
    - softmax, top-k weights renormalized to 1, shared expert behind a sigmoid gate (Qwen3-Next);
    - sigmoid scores, choice biased for load balance and limited to the best expert groups,
      weights renormalized and scaled (DeepSeek-V3/V3.2).
    Experts are stored stacked, [experts, out, in], and only the chosen ones run."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m: MoE = cfg.moe
        self.m, self.act = m, ACTIVATIONS[cfg.act]
        self.router = nn.Linear(cfg.hidden, m.experts, bias=False)
        if m.router == 'sigmoid':
            self.register_buffer('choice_bias', torch.zeros(m.experts))   # e_score_correction_bias
        self.w_gate = nn.Parameter(torch.empty(m.experts, m.expert_dim, cfg.hidden))
        self.w_up = nn.Parameter(torch.empty(m.experts, m.expert_dim, cfg.hidden))
        self.w_down = nn.Parameter(torch.empty(m.experts, cfg.hidden, m.expert_dim))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)
        self.shared = GLU(cfg.hidden, m.shared_dim, cfg.act) if m.shared_dim else None
        self.shared_gate = nn.Linear(cfg.hidden, 1, bias=False) if m.shared_gate else None

    def route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """For each token, the chosen experts [N, k] and their weights [N, k]."""
        m = self.m
        logits = F.linear(x.float(), self.router.weight.float())
        if m.router == 'softmax':
            scores = logits.softmax(-1)
            weights, experts = scores.topk(m.top_k, dim=-1)
        else:
            scores = logits.sigmoid()
            choice = scores + self.choice_bias                       # the bias only affects which experts run
            per_group = m.experts // m.groups
            group_score = choice.view(-1, m.groups, per_group).topk(2, dim=-1).values.sum(-1)
            best_groups = group_score.topk(m.top_groups, dim=-1).indices
            allowed = torch.zeros_like(group_score).scatter_(1, best_groups, 1.0)
            allowed = allowed.repeat_interleave(per_group, dim=1).bool()
            experts = choice.masked_fill(~allowed, float('-inf')).topk(m.top_k, dim=-1).indices
            weights = scores.gather(1, experts)
        if m.normalize:
            weights = weights / (weights.sum(-1, keepdim=True) + (1e-20 if m.router == 'sigmoid' else 0.0))
        return experts, (weights * m.scaling).to(x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        experts, weights = self.route(x)
        out = torch.zeros_like(x)
        for e in experts.unique().tolist():
            tok, slot = torch.where(experts == e)
            h = self.act(x[tok] @ self.w_gate[e].T) * (x[tok] @ self.w_up[e].T)
            out.index_add_(0, tok, (h @ self.w_down[e].T) * weights[tok, slot, None])
        if self.shared is not None:
            shared = self.shared(x)
            if self.shared_gate is not None:
                shared = torch.sigmoid(self.shared_gate(x)) * shared
            out = out + shared
        return out.reshape(shape)


def make_ffn(cfg: ModelConfig, layer: int) -> nn.Module:
    if cfg.uses_moe(layer):
        return MixtureOfExperts(cfg)
    if cfg.ffn == 'mlp':
        return MLP(cfg)
    return GLU(cfg.hidden, cfg.intermediate, cfg.act, cfg.mlp_bias)


# ---------------------------------------------------------------- masks

def causal_mask(t_query: int, t_key: int, device=None) -> Tensor:
    """True where query i (the last t_query of t_key positions) may see key j."""
    q = torch.arange(t_key - t_query, t_key, device=device)[:, None]
    k = torch.arange(t_key, device=device)[None, :]
    return k <= q


def masked_softmax_attention(q: Tensor, k: Tensor, v: Tensor, allowed: Tensor, scale: float) -> Tensor:
    """softmax(q kᵀ · scale, over allowed keys) v, in float32. q [B,H,Tq,d], k [B,H,Tk,d], v [B,H,Tk,dv]."""
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allowed, float('-inf'))
    return (scores.softmax(-1) @ v.float()).to(q.dtype)
