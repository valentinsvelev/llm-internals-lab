"""The Reasoning page's base model in numbers: DeepSeek-R1 (the DeepSeek-V3 architecture).

`parameters` counts every weight by part, as transformers' DeepseekV3ForCausalLM holds them (the
multi-token-prediction module aside); `active_parameters` counts those one token uses (8 of the 256
routed experts); `cache_bytes_per_token` is what MLA stores per token for generation; `route` is
DeepSeek-V3's expert choice for one token. The page (web/src/reasoning-content.js) computes the same,
and tests/test_reasoning.py checks both against transformers and against the reference MoE layer.
"""
from __future__ import annotations

from .config import DEEPSEEK_R1, ModelConfig


def parameters(cfg: ModelConfig = DEEPSEEK_R1) -> dict[str, int]:
    m, H = cfg.moe, cfg.hidden
    moe_layers = cfg.layers - m.first_dense
    glu = 3 * H * m.expert_dim
    attn = (H * cfg.q_rank + cfg.q_rank + cfg.q_rank * cfg.heads * cfg.qk_head_dim          # queries via a latent
            + H * (cfg.kv_rank + cfg.rope_head_dim) + cfg.kv_rank                            # the cached latent and rotary key
            + cfg.kv_rank * cfg.heads * (cfg.nope_dim + cfg.v_dim)                          # keys and values from the latent
            + cfg.heads * cfg.v_dim * H)                                                     # output
    return {'embedding': cfg.vocab_size * H, 'attention': cfg.layers * attn,
            'dense_ffn': m.first_dense * 3 * H * cfg.intermediate, 'router': moe_layers * m.experts * H,
            'shared_experts': moe_layers * m.shared * glu, 'routed_experts': moe_layers * m.experts * glu,
            'norms': cfg.layers * 2 * H + H, 'head': cfg.vocab_size * H}


def active_parameters(cfg: ModelConfig = DEEPSEEK_R1) -> int:
    p = parameters(cfg)
    return sum(p.values()) - p['routed_experts'] + p['routed_experts'] * cfg.moe.top_k // cfg.moe.experts


def cache_bytes_per_token(cfg: ModelConfig = DEEPSEEK_R1, bytes_per_value: int = 2) -> int:
    return sum(cfg.cache_values_per_token(i) for i in range(cfg.layers)) * bytes_per_value


def route(scores: list[float], cfg: ModelConfig = DEEPSEEK_R1) -> tuple[list[int], list[float]]:
    """Sigmoid scores of all routed experts → the chosen experts (best first) and their weights."""
    m = cfg.moe
    per = m.experts // m.groups
    group_score = [sum(sorted(scores[g * per:(g + 1) * per], reverse=True)[:2]) for g in range(m.groups)]
    best = sorted(range(m.groups), key=lambda g: (-group_score[g], g))[:m.top_groups]
    allowed = [e for e in range(m.experts) if e // per in best]
    experts = sorted(allowed, key=lambda e: (-scores[e], e))[:m.top_k]
    total = sum(scores[e] for e in experts)
    return experts, [m.scaling * scores[e] / total for e in experts]
