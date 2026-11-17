"""The eight decoding methods of the explorer, on PyTorch tensors.

Sampling filters work on the probabilities p(τ) = softmax(z/τ), applied after temperature as in
Hugging Face transformers, and return the renormalized distribution p′ that is sampled from.
"""
from __future__ import annotations

import torch
from torch import Tensor


def probabilities(logits: Tensor, tau: float = 1.0) -> Tensor:
    return torch.softmax(logits.float() / tau, dim=-1)


def greedy(logits: Tensor) -> Tensor:
    return logits.argmax(-1)


def _renormalize(p: Tensor, keep: Tensor) -> Tensor:
    p = p * keep
    return p / p.sum(-1, keepdim=True)


def top_k(p: Tensor, k: int) -> Tensor:
    """Keep the k most likely tokens."""
    kth = p.topk(k, dim=-1).values[..., -1:]
    return _renormalize(p, p >= kth)


def top_p(p: Tensor, mass: float) -> Tensor:
    """Keep the smallest set of most likely tokens whose probabilities reach `mass` (nucleus sampling)."""
    sorted_p, order = p.sort(-1, descending=True)
    before = sorted_p.cumsum(-1) - sorted_p                  # mass ranked above each token
    keep_sorted = before < mass
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    return _renormalize(p, keep)


def min_p(p: Tensor, base: float) -> Tensor:
    """Keep every token at least `base` times as likely as the most likely one."""
    return _renormalize(p, p >= base * p.max(-1, keepdim=True).values)


def typical(p: Tensor, mass: float) -> Tensor:
    """Locally typical sampling: keep the tokens whose surprisal −log p is closest to the entropy H,
    until their probabilities reach `mass`."""
    logp = p.clamp_min(1e-30).log()
    entropy = -(p * logp).sum(-1, keepdim=True)
    distance = (-logp - entropy).abs()
    order = distance.argsort(-1)
    sorted_p = p.gather(-1, order)
    before = sorted_p.cumsum(-1) - sorted_p
    keep = torch.zeros_like(p, dtype=torch.bool).scatter(-1, order, before < mass)
    return _renormalize(p, keep)


def sample(logits: Tensor, method: str = 'random', tau: float = 1.0, *, k: int = 50, p: float = 0.9,
           base: float = 0.1, mass: float = 0.9, generator: torch.Generator | None = None) -> Tensor:
    """One token per row of `logits` [B, V]. method: random, topk, topp, minp or typical."""
    probs = probabilities(logits, tau)
    probs = {'random': lambda x: x, 'topk': lambda x: top_k(x, k), 'topp': lambda x: top_p(x, p),
             'minp': lambda x: min_p(x, base), 'typical': lambda x: typical(x, mass)}[method](probs)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def beam_search(model, ids: Tensor, width: int, steps: int) -> tuple[Tensor, float]:
    """Keep the `width` most likely continuations (by total log-probability) after every step;
    return the best sequence and its score. One sequence, no length penalty, recomputed each step."""
    beams = [(ids[0], 0.0)]
    for _ in range(steps):
        candidates = []
        for seq, score in beams:
            logp = torch.log_softmax(model(seq[None])[0, -1].float(), -1)
            top = logp.topk(width)
            candidates += [(torch.cat([seq, i.view(1)]), score + float(v)) for v, i in zip(top.values, top.indices)]
        beams = sorted(candidates, key=lambda c: c[1], reverse=True)[:width]
    return beams[0][0][None], beams[0][1]


def speculative_accept(p: Tensor, q: Tensor, drafted: int, generator: torch.Generator | None = None) -> tuple[int, bool]:
    """One verification step: the draft model proposed `drafted` from q. Accept it with probability
    min(1, p/q); otherwise draw from the residual r ∝ max(0, p − q). The returned token is distributed
    exactly as p. Returns (token, accepted)."""
    u = torch.rand((), generator=generator)
    if u < torch.clamp(p[drafted] / q[drafted], max=1.0):
        return drafted, True
    residual = (p - q).clamp_min(0)
    return int(torch.multinomial(residual / residual.sum(), 1, generator=generator)), False


@torch.no_grad()
def speculative_generate(target, draft, ids: Tensor, steps: int, gamma: int = 4, tau: float = 1.0,
                         generator: torch.Generator | None = None) -> tuple[Tensor, float]:
    """Speculative sampling (Leviathan et al., 2023; Chen et al., 2023) for one sequence.
    The draft proposes `gamma` tokens; the target scores all of them in one forward pass and accepts a
    prefix; a rejected position is resampled from the residual, and if all are accepted the target adds
    one more token. Returns the sequence and the fraction of drafted tokens accepted."""
    seq, accepted, proposed = ids[0], 0, 0
    while seq.numel() < ids.shape[1] + steps:
        ctx, qs = seq, []
        for _ in range(gamma):                                   # cheap: the draft runs gamma times
            q = probabilities(draft(ctx[None])[0, -1], tau)
            t = torch.multinomial(q, 1, generator=generator)
            qs.append(q)
            ctx = torch.cat([ctx, t])
        ps = probabilities(target(ctx[None])[0, -gamma - 1:], tau)    # expensive: one target pass for all
        out = seq
        for i in range(gamma):
            token, ok = speculative_accept(ps[i], qs[i], int(ctx[seq.numel() + i]), generator)
            out = torch.cat([out, torch.tensor([token])])
            proposed += 1
            accepted += ok
            if not ok:
                break
        else:                                                    # every draft accepted: one bonus token
            out = torch.cat([out, torch.multinomial(ps[gamma], 1, generator=generator)])
        seq = out
    return seq[None, :ids.shape[1] + steps], accepted / max(proposed, 1)
