"""Bridges to Hugging Face transformers: configs in both directions, and loading HF weights.

`load_hf_weights` maps every tensor of a transformers checkpoint onto the reference modules (fused
projections are split, GPT-2's Conv1D weights transposed, stacked experts unpacked) and loads them with
strict=True, so nothing is silently left out. The tests use it with random weights to check that the
reference models compute exactly what transformers computes; with a real checkpoint it runs real models:

    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
    model = load_hf_weights(DecoderLM(from_hf_config(hf.config)), hf.state_dict())
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig, MoE, QSAConfig, YaRN


def family(cfg: ModelConfig) -> str:
    if cfg.learned_positions:
        return 'gpt2'
    if cfg.full_attention_interval:
        return 'qwen3_next'
    if cfg.attention == 'mla':
        return 'deepseek_v32' if cfg.index_heads else 'deepseek_v2'
    return 'gemma' if cfg.embed_scale else 'qwen2'


def _yarn_params(cfg: ModelConfig) -> dict:
    p = {'rope_theta': cfg.rope_theta, 'rope_type': 'default'}
    if cfg.yarn:
        y = cfg.yarn
        p.update(rope_type='yarn', factor=y.factor, original_max_position_embeddings=y.original_max_positions,
                 beta_fast=y.beta_fast, beta_slow=y.beta_slow, mscale=y.mscale, mscale_all_dim=y.mscale_all_dim)
    return p


def to_hf_config(cfg: ModelConfig):
    """The transformers config of the same architecture (used by the tests)."""
    import transformers as tf
    f = family(cfg)
    common = dict(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden, num_hidden_layers=cfg.layers,
                  num_attention_heads=cfg.heads, num_key_value_heads=cfg.kv_heads, max_position_embeddings=cfg.max_positions,
                  rms_norm_eps=cfg.norm_eps, tie_word_embeddings=cfg.tied, intermediate_size=cfg.intermediate)
    if f == 'gpt2':
        return tf.GPT2Config(vocab_size=cfg.vocab_size, n_positions=cfg.max_positions, n_embd=cfg.hidden, n_layer=cfg.layers,
                             n_head=cfg.heads, n_inner=cfg.intermediate, activation_function='gelu_new',
                             layer_norm_epsilon=cfg.norm_eps, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    if f == 'gemma':
        return tf.GemmaConfig(**common, head_dim=cfg.head_dim, hidden_act='gelu_pytorch_tanh', rope_parameters=_yarn_params(cfg))
    if f == 'qwen2':
        return tf.Qwen2Config(**common, hidden_act=cfg.act, rope_parameters=_yarn_params(cfg))
    m = cfg.moe
    if f == 'qwen3_next':
        rope = _yarn_params(cfg) | {'partial_rotary_factor': cfg.rotary_dim / cfg.head_dim}
        types = ['linear_attention' if cfg.layer_type(i) == 'gdn' else 'full_attention' for i in range(cfg.layers)]
        return tf.Qwen3NextConfig(**common, head_dim=cfg.head_dim, rope_parameters=rope, layer_types=types,
                                  linear_conv_kernel_dim=cfg.conv_kernel, linear_key_head_dim=cfg.lin_head_dim,
                                  linear_value_head_dim=cfg.lin_head_dim, linear_num_key_heads=cfg.lin_qk_heads,
                                  linear_num_value_heads=cfg.lin_v_heads, decoder_sparse_step=1, mlp_only_layers=[],
                                  moe_intermediate_size=m.expert_dim, shared_expert_intermediate_size=m.shared_dim,
                                  num_experts_per_tok=m.top_k, num_experts=m.experts, norm_topk_prob=m.normalize)
    mla = dict(kv_lora_rank=cfg.kv_rank, q_lora_rank=cfg.q_rank, qk_nope_head_dim=cfg.nope_dim, qk_rope_head_dim=cfg.rope_head_dim,
               v_head_dim=cfg.v_dim, moe_intermediate_size=m.expert_dim, n_routed_experts=m.experts, n_shared_experts=m.shared,
               num_experts_per_tok=m.top_k, first_k_dense_replace=m.first_dense, routed_scaling_factor=m.scaling,
               n_group=m.groups, topk_group=m.top_groups, norm_topk_prob=m.normalize, rope_parameters=_yarn_params(cfg))
    if f == 'deepseek_v2':
        return tf.DeepseekV2Config(**common, **mla, topk_method='greedy')
    return tf.DeepseekV32Config(**common, **mla, index_n_heads=cfg.index_heads, index_head_dim=cfg.index_dim,
                                index_topk=cfg.index_topk,
                                mlp_layer_types=['dense' if i < m.first_dense else 'sparse' for i in range(cfg.layers)])


def from_hf_config(c) -> ModelConfig:
    """A ModelConfig from a transformers config of one of the six supported families."""
    t = c.model_type
    rp = getattr(c, 'rope_parameters', None) or {}
    yarn = None
    if rp.get('rope_type') == 'yarn':
        yarn = YaRN(rp['factor'], rp['original_max_position_embeddings'], rp.get('beta_fast') or 32, rp.get('beta_slow') or 1,
                    rp.get('mscale') or 1.0, rp.get('mscale_all_dim') or 1.0)
    if t == 'gpt2':
        return ModelConfig(name=t, vocab_size=c.vocab_size, hidden=c.n_embd, layers=c.n_layer, attention='gqa', heads=c.n_head,
                           kv_heads=c.n_head, head_dim=c.n_embd // c.n_head, qkv_bias=True, out_bias=True, learned_positions=True,
                           max_positions=c.n_positions, norm='layer', norm_eps=c.layer_norm_epsilon, tied=True, ffn='mlp',
                           act='gelu_tanh', intermediate=c.n_inner or 4 * c.n_embd, mlp_bias=True)
    common = dict(name=t, vocab_size=c.vocab_size, hidden=c.hidden_size, layers=c.num_hidden_layers, heads=c.num_attention_heads,
                  kv_heads=c.num_key_value_heads, max_positions=c.max_position_embeddings, rope_theta=rp.get('rope_theta', 10000.0),
                  norm_eps=c.rms_norm_eps, tied=c.tie_word_embeddings, intermediate=c.intermediate_size, yarn=yarn)
    if t == 'gemma':
        return ModelConfig(**common, attention='gqa', head_dim=c.head_dim, norm_offset=True, embed_scale=True, act='gelu_tanh')
    if t == 'qwen2':
        return ModelConfig(**common, attention='gqa', head_dim=c.hidden_size // c.num_attention_heads, qkv_bias=True, act='silu')
    if t == 'qwen3_next':
        return ModelConfig(**common, attention='gqa', head_dim=c.head_dim, qk_norm=True, output_gate=True,
                           rope_dim=int(c.head_dim * rp.get('partial_rotary_factor', 1.0)), norm_offset=True,
                           full_attention_interval=c.layer_types.index('full_attention') + 1, lin_qk_heads=c.linear_num_key_heads,
                           lin_v_heads=c.linear_num_value_heads, lin_head_dim=c.linear_key_head_dim, conv_kernel=c.linear_conv_kernel_dim,
                           moe=MoE(c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, shared=1,
                                   shared_dim=c.shared_expert_intermediate_size, normalize=c.norm_topk_prob, shared_gate=True))
    if t in ('deepseek_v2', 'deepseek_v32'):
        sparse = t == 'deepseek_v32'
        moe = MoE(c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size, shared=c.n_shared_experts,
                  shared_dim=c.moe_intermediate_size * c.n_shared_experts, router='sigmoid' if sparse else 'softmax',
                  normalize=bool(c.norm_topk_prob) if sparse else False, scaling=c.routed_scaling_factor,
                  groups=c.n_group if sparse else 1, top_groups=c.topk_group if sparse else 1, first_dense=c.first_k_dense_replace)
        return ModelConfig(**common, attention='mla', head_dim=c.qk_nope_head_dim + c.qk_rope_head_dim, q_rank=c.q_lora_rank,
                           kv_rank=c.kv_lora_rank, nope_dim=c.qk_nope_head_dim, rope_head_dim=c.qk_rope_head_dim, v_dim=c.v_head_dim,
                           rope_interleaved=True, act='silu', moe=moe, index_heads=getattr(c, 'index_n_heads', 0) if sparse else 0,
                           index_dim=getattr(c, 'index_head_dim', 0) if sparse else 0, index_topk=getattr(c, 'index_topk', 0) if sparse else 0)
    raise ValueError(f'unsupported model type {t!r}')


# ---------------------------------------------------------------- weights

def _ffn(out: dict, sd: dict, src: str, dst: str, moe: MoE | None, family_: str):
    if moe is None or f'{src}.gate.weight' not in sd:
        for name in ('gate', 'up', 'down'):
            out[f'{dst}.{name}.weight'] = sd[f'{src}.{name}_proj.weight']
        return
    out[f'{dst}.router.weight'] = sd[f'{src}.gate.weight']
    if f'{src}.gate.e_score_correction_bias' in sd:
        out[f'{dst}.choice_bias'] = sd[f'{src}.gate.e_score_correction_bias']
    gate_up = sd[f'{src}.experts.gate_up_proj']                    # [E, 2I, H]: gate rows, then up rows
    out[f'{dst}.w_gate'], out[f'{dst}.w_up'] = gate_up.chunk(2, dim=1)
    out[f'{dst}.w_down'] = sd[f'{src}.experts.down_proj']
    shared = 'shared_expert' if family_ == 'qwen3_next' else 'shared_experts'
    for name in ('gate', 'up', 'down'):
        out[f'{dst}.shared.{name}.weight'] = sd[f'{src}.{shared}.{name}_proj.weight']
    if family_ == 'qwen3_next':
        out[f'{dst}.shared_gate.weight'] = sd[f'{src}.shared_expert_gate.weight']


def _split_q_gate(w: Tensor, heads: int, d: int) -> tuple[Tensor, Tensor]:
    """Qwen's q_proj interleaves [query | gate] per head."""
    w = w.view(heads, 2, d, w.shape[-1])
    return w[:, 0].reshape(heads * d, -1), w[:, 1].reshape(heads * d, -1)


def load_hf_weights(model: nn.Module, sd: dict) -> nn.Module:
    from .model import DecoderLM
    assert isinstance(model, DecoderLM)
    cfg, f, out = model.cfg, family(model.cfg), {}
    sd = {k: v for k, v in sd.items() if not k.endswith(('.attn.bias', '.attn.masked_bias'))}
    if f == 'gpt2':
        def T(w):                                                  # Conv1D stores [in, out]
            return w.T.contiguous()
        out['embed.weight'], out['pos_embed.weight'] = sd['transformer.wte.weight'], sd['transformer.wpe.weight']
        out['norm.weight'], out['norm.bias'] = sd['transformer.ln_f.weight'], sd['transformer.ln_f.bias']
        for i in range(cfg.layers):
            p, q = f'transformer.h.{i}', f'blocks.{i}'
            for a, b in (('ln_1', 'norm1'), ('ln_2', 'norm2')):
                out[f'{q}.{b}.weight'], out[f'{q}.{b}.bias'] = sd[f'{p}.{a}.weight'], sd[f'{p}.{a}.bias']
            w, bias = T(sd[f'{p}.attn.c_attn.weight']).chunk(3, 0), sd[f'{p}.attn.c_attn.bias'].chunk(3, 0)
            for j, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
                out[f'{q}.mixer.{name}.weight'], out[f'{q}.mixer.{name}.bias'] = w[j], bias[j]
            out[f'{q}.mixer.o_proj.weight'], out[f'{q}.mixer.o_proj.bias'] = T(sd[f'{p}.attn.c_proj.weight']), sd[f'{p}.attn.c_proj.bias']
            out[f'{q}.ffn.up.weight'], out[f'{q}.ffn.up.bias'] = T(sd[f'{p}.mlp.c_fc.weight']), sd[f'{p}.mlp.c_fc.bias']
            out[f'{q}.ffn.down.weight'], out[f'{q}.ffn.down.bias'] = T(sd[f'{p}.mlp.c_proj.weight']), sd[f'{p}.mlp.c_proj.bias']
    else:
        out['embed.weight'], out['norm.weight'] = sd['model.embed_tokens.weight'], sd['model.norm.weight']
        if not cfg.tied:
            out['lm_head.weight'] = sd['lm_head.weight']
        for i in range(cfg.layers):
            p, q = f'model.layers.{i}', f'blocks.{i}'
            out[f'{q}.norm1.weight'] = sd[f'{p}.input_layernorm.weight']
            out[f'{q}.norm2.weight'] = sd[f'{p}.post_attention_layernorm.weight']
            _ffn(out, sd, f'{p}.mlp', f'{q}.ffn', cfg.moe if cfg.uses_moe(i) else None, f)
            a, m = f'{p}.self_attn', f'{q}.mixer'
            if cfg.layer_type(i) == 'gdn':
                _load_gdn(out, sd, f'{p}.linear_attn', m, cfg)
            elif cfg.attention == 'mla':
                names = {'kv_a_proj_with_mqa': 'kv_down', 'kv_a_layernorm': 'kv_norm', 'kv_b_proj': 'kv_up', 'o_proj': 'o_proj',
                         'q_proj': 'q_proj', 'q_a_proj': 'q_down', 'q_a_layernorm': 'q_norm', 'q_b_proj': 'q_up',
                         'indexer.wq_b': 'indexer.q_up', 'indexer.wk': 'indexer.k_proj', 'indexer.k_norm': 'indexer.k_norm',
                         'indexer.weights_proj': 'indexer.head_weights'}
                for k, v in sd.items():
                    if k.startswith(a + '.'):
                        rest = k[len(a) + 1:]
                        mod, _, param = rest.rpartition('.')
                        out[f'{m}.{names[mod]}.{param}'] = v
            else:
                for name in ('k_proj', 'v_proj', 'o_proj'):
                    for kind in ('weight', 'bias'):
                        if f'{a}.{name}.{kind}' in sd:
                            out[f'{m}.{name}.{kind}'] = sd[f'{a}.{name}.{kind}']
                if cfg.output_gate:
                    out[f'{m}.q_proj.weight'], out[f'{m}.gate_proj.weight'] = _split_q_gate(sd[f'{a}.q_proj.weight'], cfg.heads, cfg.head_dim)
                    out[f'{m}.q_norm.weight'], out[f'{m}.k_norm.weight'] = sd[f'{a}.q_norm.weight'], sd[f'{a}.k_norm.weight']
                else:
                    out[f'{m}.q_proj.weight'] = sd[f'{a}.q_proj.weight']
                    if f'{a}.q_proj.bias' in sd:
                        out[f'{m}.q_proj.bias'] = sd[f'{a}.q_proj.bias']
    missing, unexpected = model.load_state_dict(out, strict=False)
    missing = [k for k in missing if not k.endswith('inv_freq')]
    if missing or unexpected:
        raise KeyError(f'weights do not line up: missing {missing[:5]}, unexpected {unexpected[:5]}')
    return model


def _load_gdn(out: dict, sd: dict, src: str, dst: str, cfg: ModelConfig):
    """Qwen3-Next fuses q, k, v, z (and b, a) per key head; split them into separate projections."""
    nk, nv, d = cfg.lin_qk_heads, cfg.lin_v_heads, cfg.lin_head_dim
    r = nv // nk
    w = sd[f'{src}.in_proj_qkvz.weight'].view(nk, 2 * d + 2 * r * d, -1)
    q, k, v, z = w.split([d, d, r * d, r * d], dim=1)
    for name, t in (('q_proj', q), ('k_proj', k), ('v_proj', v), ('z_proj', z)):
        out[f'{dst}.{name}.weight'] = t.reshape(-1, t.shape[-1])
    b, a = sd[f'{src}.in_proj_ba.weight'].view(nk, 2 * r, -1).split([r, r], dim=1)
    out[f'{dst}.b_proj.weight'], out[f'{dst}.a_proj.weight'] = b.reshape(nv, -1), a.reshape(nv, -1)
    out[f'{dst}.conv'] = sd[f'{src}.conv1d.weight'][:, 0]
    out[f'{dst}.A_log'], out[f'{dst}.dt_bias'] = sd[f'{src}.A_log'], sd[f'{src}.dt_bias']
    out[f'{dst}.out_norm.weight'] = sd[f'{src}.norm.weight']
    out[f'{dst}.out_proj.weight'] = sd[f'{src}.out_proj.weight']


def load_hf_qsa_weights(module: nn.Module, sd: dict, cfg: QSAConfig) -> nn.Module:
    """Weights of a transformers Qwen4Exp (Qwen3.8-Flash-Next) attention layer into QwenSparseAttention."""
    out = {}
    out['q_proj.weight'], out['gate_proj.weight'] = _split_q_gate(sd['q_proj.weight'], cfg.heads, cfg.head_dim)
    for name in ('k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'):
        out[f'{name}.weight'] = sd[f'{name}.weight']
    qk = sd['indexer.index_qk_proj.weight']
    out['indexer.q_proj.weight'], out['indexer.k_proj.weight'] = qk.split([cfg.index_heads * cfg.index_dim, cfg.index_dim], 0)
    out['indexer.q_norm.weight'], out['indexer.k_norm.weight'] = sd['indexer.q_layernorm.weight'], sd['indexer.k_layernorm.weight']
    module.load_state_dict(out, strict=True)
    return module


def from_pretrained(hf_id: str, **kwargs):
    """Download a checkpoint with transformers and return (our model, transformers model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model import DecoderLM
    hf = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32, trust_remote_code=False, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False)
    return load_hf_weights(DecoderLM(from_hf_config(hf.config)), hf.state_dict()).eval(), hf, tok
