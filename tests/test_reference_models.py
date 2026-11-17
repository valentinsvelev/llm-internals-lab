"""The PyTorch reference implementations compute what the transformers implementations compute.

Strategy: build the same architecture twice (ours from `ModelConfig.tiny()`, transformers' from the
equivalent config), give transformers random weights, copy them into ours with `load_hf_weights`, and
compare logits. Tiny configs keep every mechanism (GQA sharing, MLA latents and YaRN, the DSA indexer
with a budget smaller than the sequence, Gated DeltaNet with a 3:1 layer pattern, every MoE router).
Full-size configs are compared by parameter count on the meta device, without allocating memory.
"""
from __future__ import annotations

import math
import warnings

import pytest

torch = pytest.importorskip('torch')

from llm_lab.reference import CONFIGS, DecoderLM
from llm_lab.reference.config import QSAConfig
from llm_lab.reference.decoding import beam_search, min_p, speculative_accept, top_k, top_p, typical
from llm_lab.reference.gated_deltanet import delta_rule_chunked, delta_rule_recurrent
from llm_lab.reference.layers import Rotary, yarn_softmax_scale

KEYS = list(CONFIGS)


def _hf():
    tf = pytest.importorskip('transformers')
    warnings.filterwarnings('ignore')
    tf.logging.set_verbosity_error()
    return tf


def _random_hf_model(cfg, seed=0):
    from llm_lab.reference.hf import to_hf_config
    tf = _hf()
    hcfg = to_hf_config(cfg)
    hcfg._attn_implementation = 'eager'
    torch.manual_seed(seed)
    model = tf.AutoModelForCausalLM.from_config(hcfg).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():   # random everything, norms and gates included, so no parameter goes untested
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.5 if p.ndim == 1 else 0.08))
        for name, b in model.named_buffers():
            if name.endswith('e_score_correction_bias'):
                b.copy_(torch.randn(b.shape, generator=g) * 0.1)
    return model


def _ours_from(hf_model, cfg):
    from llm_lab.reference.hf import load_hf_weights
    return load_hf_weights(DecoderLM(cfg), hf_model.state_dict()).eval()


@pytest.mark.parametrize('key', KEYS)
def test_logits_match_transformers(key):
    cfg = CONFIGS[key].tiny()
    hf = _random_hf_model(cfg)
    ours = _ours_from(hf, cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 17), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, got = hf(ids).logits, ours(ids)
    assert expected.abs().max() > 0.5                      # a non-trivial model
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize('key', ['gqa', 'gdn', 'dsa'])
def test_greedy_generation_matches_transformers(key):
    cfg = CONFIGS[key].tiny()
    hf = _random_hf_model(cfg, seed=3)
    ours = _ours_from(hf, cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = hf.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
    assert torch.equal(ours.generate(ids, 8), expected)


@pytest.mark.parametrize('key', KEYS)
def test_full_size_parameter_counts_match_transformers(key):
    from llm_lab.reference.hf import to_hf_config
    tf = _hf()
    with torch.device('meta'):
        ours = sum(p.numel() for p in DecoderLM(CONFIGS[key]).parameters())
        theirs = sum(p.numel() for p in tf.AutoModelForCausalLM.from_config(to_hf_config(CONFIGS[key])).parameters())
    assert ours == theirs
    published = {'mha': 124_439_808, 'mqa': 2_506_172_416, 'gqa': 494_032_768,      # exact, from the model cards / hub
                 'mla': 15.7e9, 'gdn': 80e9, 'dsa': 671e9}[key]                      # as stated: 15.7B, 80B, 671B
    assert abs(ours - published) / published < 0.005


def test_config_round_trip():
    from llm_lab.reference.hf import from_hf_config, to_hf_config
    _hf()
    for key, cfg in CONFIGS.items():
        back = from_hf_config(to_hf_config(cfg))
        for field in ('hidden', 'layers', 'heads', 'kv_rank', 'rope_head_dim', 'index_topk', 'lin_v_heads', 'rotary_dim', 'moe'):
            assert getattr(back, field) == getattr(cfg, field), (key, field)


@pytest.mark.parametrize('key', KEYS)
def test_cache_gives_the_same_logits_and_stores_what_the_page_says(key):
    cfg = CONFIGS[key].tiny()
    torch.manual_seed(0)
    model = DecoderLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ids = torch.randint(0, cfg.vocab_size, (1, 11))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        steps = [model(ids[:, :6], cache)] + [model(ids[:, t:t + 1], cache) for t in range(6, 11)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=5e-5, rtol=1e-4)
    for i, stored in enumerate(cache.numbers_per_layer()):
        if cfg.layer_type(i) == 'gdn':   # fixed size: one d_k × d_v state per value head + 3 convolution inputs
            channels = 2 * cfg.lin_qk_heads * cfg.lin_head_dim + cfg.lin_v_heads * cfg.lin_head_dim
            assert stored == cfg.lin_v_heads * cfg.lin_head_dim ** 2 + (cfg.conv_kernel - 1) * channels
        else:                            # grows: the same number of values for every token
            assert stored == 11 * cfg.cache_values_per_token(i)


def test_mla_absorbed_attention_equals_expanded():
    """Folding W^UK into the query and W^UV into the output attends over cached latents directly."""
    for key in ('mla', 'dsa'):
        cfg = CONFIGS[key].tiny()
        torch.manual_seed(0)
        layer = DecoderLM(cfg).blocks[1].mixer.eval()
        x = torch.randn(2, 9, cfg.hidden)
        with torch.no_grad():
            a = layer(x, torch.arange(9), absorbed=False)
            b = layer(x, torch.arange(9), absorbed=True)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_dsa_without_sparsity_is_plain_mla():
    """With a budget of at least the sequence length, the indexer keeps every token."""
    from dataclasses import replace
    cfg = CONFIGS['dsa'].tiny(index_topk=64)
    torch.manual_seed(0)
    sparse = DecoderLM(cfg).eval()
    dense = DecoderLM(replace(cfg, index_heads=0, index_dim=0, index_topk=0)).eval()
    dense.load_state_dict({k: v for k, v in sparse.state_dict().items() if '.indexer.' not in k})
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        torch.testing.assert_close(sparse(ids), dense(ids))
        tight = DecoderLM(replace(cfg, index_topk=3)).eval()
        tight.load_state_dict(sparse.state_dict())
        assert (tight(ids) - dense(ids)).abs().max() > 1e-3          # a small budget does change the result


def test_gated_delta_rule_chunked_equals_recurrent():
    g = torch.Generator().manual_seed(0)
    B, T, nk, nv, d = 2, 37, 2, 4, 8
    q, k = torch.randn(B, T, nk, d, generator=g), torch.randn(B, T, nk, d, generator=g)
    v = torch.randn(B, T, nv, d, generator=g)
    decay = -torch.rand(B, T, nv, generator=g)            # log α ≤ 0
    beta = torch.rand(B, T, nv, generator=g)
    state = torch.randn(B, nv, d, d, generator=g) * 0.1
    o1, s1 = delta_rule_recurrent(q, k, v, decay, beta, state)
    o2, s2 = delta_rule_chunked(q, k, v, decay, beta, state, chunk=8)   # 5 chunks, the last one padded
    torch.testing.assert_close(o1, o2, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(s1, s2, atol=1e-5, rtol=1e-4)
    # without decay and with β = 1, a key written once is read back exactly: S k = v
    kk = torch.nn.functional.normalize(torch.randn(1, 1, 1, d, generator=g), dim=-1)
    vv = torch.randn(1, 1, 1, d, generator=g)
    _, s = delta_rule_recurrent(kk, kk, vv, torch.zeros(1, 1, 1), torch.ones(1, 1, 1))
    torch.testing.assert_close(torch.einsum('k,kv->v', kk[0, 0, 0], s[0, 0]), vv[0, 0, 0], atol=1e-5, rtol=1e-4)


def test_yarn_matches_transformers_and_the_page():
    tf = _hf()
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from llm_lab.reference.hf import to_hf_config
    for key, page_scale in (('mla', 1.59), ('dsa', 1.87)):
        cfg = CONFIGS[key]
        inv_freq, factor = ROPE_INIT_FUNCTIONS['yarn'](to_hf_config(cfg))
        ours = Rotary(cfg.rope_head_dim, cfg.rope_theta, cfg.yarn)
        torch.testing.assert_close(ours.inv_freq, inv_freq.float())
        assert ours.scale == pytest.approx(factor)
        assert yarn_softmax_scale(cfg.yarn) == pytest.approx(page_scale, abs=0.005)
        # the page: pairs 0–10 keep their speed, pairs 23–31 are slowed 40×
        plain = 1.0 / cfg.rope_theta ** (torch.arange(0, 64, 2) / 64)
        assert torch.allclose(ours.inv_freq[:11], plain[:11]) and torch.allclose(ours.inv_freq[23:], plain[23:] / 40)
    assert tf is not None


def test_qsa_matches_the_qwen_reference_layer():
    tf = _hf()
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as Q
    except ImportError:
        pytest.skip('this transformers version has no Qwen3.8-Flash-Next (qwen4_exp)')
    from llm_lab.reference.hf import load_hf_qsa_weights
    from llm_lab.reference.qsa import QwenSparseAttention
    cfg = QSAConfig(hidden=64, heads=4, kv_heads=2, head_dim=32, rope_dim=8, index_heads=2, index_dim=16, block=4, budget=8)
    hc = tf.Qwen4ExpTextConfig(
        vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        rope_parameters={'rope_type': 'default', 'rope_theta': cfg.rope_theta, 'partial_rotary_factor': 0.25, 'mrope_section': [2, 1, 1]},
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2, hc_count=2, hc_lowrank=8)
    hc._attn_implementation = 'eager'
    torch.manual_seed(0)
    reference = Q.Qwen4ExpTextAttention(hc, layer_idx=3).eval()
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    T = 23                                  # 5 complete blocks, budget 2 blocks: the selection matters
    x = torch.randn(1, T, 64)
    cos, sin = Q.Qwen4ExpTextRotaryEmbedding(hc)(x, torch.arange(T)[None])
    with torch.no_grad():
        expected, _ = reference(x, (cos, sin), torch.full((1, 1, T, T), float('-inf')).triu(1))
        ours = load_hf_qsa_weights(QwenSparseAttention(cfg), reference.state_dict(), cfg).eval()
        got = ours(x, torch.arange(T))
        cache = {}
        steps = [ours(x[:, :9], torch.arange(9), cache)] + [ours(x[:, t:t + 1], torch.arange(t, t + 1), cache) for t in range(9, T)]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(torch.cat(steps, 1), got, atol=1e-5, rtol=1e-4)
    assert cache['k_index'].shape[1] == T     # one raw index key per token; pooling happens when a query reads them


# ---------------------------------------------------------------- decoding

PAGE_LOGITS = torch.tensor([9.4, 9.1, 8.3, 7.9, 7.4, 7.0, 6.8, 6.5, 6.1, 5.6, 5.2, 4.8, 4.5, 4.1])   # the page's demo
WORDS = ['smaller', 'pieces', 'multiple', 'sub', 'several', 'tokens', 'parts', 'two', 'chunks', 'fragments', 'short', 'units', 'many', 'shorter']


def _kept(p):
    return [WORDS[i] for i in (p > 0).nonzero().flatten().tolist()]


def test_filters_reproduce_the_page():
    p1, p08 = torch.softmax(PAGE_LOGITS, 0), torch.softmax(PAGE_LOGITS / 0.8, 0)
    assert _kept(top_k(p1, 3)) == ['smaller', 'pieces', 'multiple']
    assert _kept(top_p(p1, 0.6)) == ['smaller', 'pieces']
    assert _kept(min_p(p1, 0.2)) == ['smaller', 'pieces', 'multiple', 'sub']
    assert _kept(typical(p1, 0.5)) == ['smaller', 'pieces', 'multiple', 'sub']
    assert len(_kept(min_p(p08, 0.1))) == 4 and len(_kept(typical(p08, 0.9))) == 5     # the page's defaults at τ = 0.8
    for q in (top_k(p08, 5), top_p(p08, 0.9), min_p(p08, 0.1), typical(p08, 0.9)):
        assert q.sum().item() == pytest.approx(1.0)
    assert top_k(p08, 5)[0].item() == pytest.approx(0.460, abs=5e-4)                   # “smaller”: 46% after top-k on the page


def test_speculative_sampling_reproduces_the_target_distribution():
    """Draft x ~ q, accept w.p. min(1, p/q), else resample from r: the output is distributed as p."""
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(PAGE_LOGITS[:6], 0)
    q = torch.softmax(PAGE_LOGITS[:6] + torch.tensor([-0.5, 0.45, -0.3, 0.4, 0.1, -0.35]), 0)   # the page's draft
    counts, accepted, n = torch.zeros(6), 0, 40_000
    for _ in range(n):
        drafted = int(torch.multinomial(q, 1, generator=g))
        token, ok = speculative_accept(p, q, drafted, g)
        counts[token] += 1
        accepted += ok
    assert torch.allclose(counts / n, p, atol=0.01)
    assert accepted / n == pytest.approx(float(torch.minimum(p, q).sum()), abs=0.01)    # acceptance rate Σ min(p, q)


def test_beam_search_can_beat_greedy():
    """A toy two-step model where the most likely first token does not start the most likely sequence."""
    probs = {0: [0, .40, .35, .25, 0],        # first step: token 1 is the most likely
             1: [0, 0, 0, .46, .54],         # after 1, the best continuation has .54 → .40 × .54 = .216
             2: [0, 0, 0, .70, .30]}         # after 2, .70 → .35 × .70 = .245
    class Toy(torch.nn.Module):
        def forward(self, ids):
            z = torch.full((ids.shape[0], ids.shape[1], 5), -30.0)
            row = torch.tensor(probs.get(ids[0, -1].item(), [.2] * 5))
            z[0, -1] = torch.where(row > 0, row.clamp_min(1e-12).log(), torch.tensor(-30.0))
            return z
    greedy_seq = [0]
    for _ in range(2):
        greedy_seq.append(int(Toy()(torch.tensor([greedy_seq]))[0, -1].argmax()))
    seq, score = beam_search(Toy(), torch.tensor([[0]]), width=2, steps=2)
    assert greedy_seq == [0, 1, 4]
    assert seq.tolist() == [[0, 2, 3]]
    assert score == pytest.approx(math.log(0.35 * 0.70), abs=1e-4) and score > math.log(0.40 * 0.54)
