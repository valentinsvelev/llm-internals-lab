"""The numbers in the Reasoning page's diagram: DeepSeek-R1's parameters, KV cache and expert routing, in the
page (web/src/reasoning-content.js, run in node), in llm_lab.reference.reasoning, in transformers and in
the reference MoE layer."""
import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from llm_lab.reference import reasoning
from llm_lab.reference.config import DEEPSEEK_R1, DEEPSEEK_V3_2

CONTENT = Path(__file__).resolve().parents[1] / 'web/src/reasoning-content.js'
SEEDS = list(range(1, 41))


@pytest.fixture(scope='module')
def page():
    node = shutil.which('node')
    if node is None:
        pytest.skip('node not installed')
    script = (f'const R = require({json.dumps(str(CONTENT))});'
              f'console.log(JSON.stringify({{arch: R.ARCH, params: R.parameters(), cache: R.cacheBytesPerToken(),'
              f' routes: {json.dumps(SEEDS)}.map(s => {{ const sc = R.demoScores(s); return {{scores: sc, ...R.route(sc)}}; }})}}))')
    return json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)


def test_page_config_is_r1(page):
    a, c, m = page['arch'], DEEPSEEK_R1, DEEPSEEK_R1.moe
    assert (a['vocab'], a['hidden'], a['layers'], a['heads']) == (c.vocab_size, c.hidden, c.layers, c.heads)
    assert (a['qRank'], a['kvRank'], a['nopeDim'], a['ropeDim'], a['vDim']) == (c.q_rank, c.kv_rank, c.nope_dim, c.rope_head_dim, c.v_dim)
    assert (a['intermediate'], a['experts'], a['topK'], a['expertDim'], a['shared']) == (c.intermediate, m.experts, m.top_k, m.expert_dim, m.shared)
    assert (a['denseLayers'], a['groups'], a['topGroups'], a['scaling'], a['maxPositions']) == (m.first_dense, m.groups, m.top_groups, m.scaling, c.max_positions)
    assert DEEPSEEK_R1 == replace(DEEPSEEK_V3_2, name='DeepSeek-R1', index_heads=0, index_dim=0, index_topk=0, source=DEEPSEEK_R1.source)


def test_parameters_and_cache(page):
    ref = reasoning.parameters()
    keys = {'embedding': 'embedding', 'attention': 'attention', 'denseFfn': 'dense_ffn', 'router': 'router',
            'sharedExperts': 'shared_experts', 'routedExperts': 'routed_experts', 'norms': 'norms', 'head': 'head'}
    assert {keys[k]: v for k, v in page['params']['parts'].items()} == ref
    assert page['params']['total'] == sum(ref.values())
    assert page['params']['active'] == reasoning.active_parameters()
    assert page['cache'] == reasoning.cache_bytes_per_token() == 61 * (512 + 64) * 2


def test_parameters_match_transformers():
    """Build DeepseekV3ForCausalLM with R1's config on the meta device and count its weights by part."""
    torch = pytest.importorskip('torch')
    tf = pytest.importorskip('transformers')
    if not hasattr(tf, 'DeepseekV3Config'):
        pytest.skip('transformers has no DeepSeek-V3')
    c, m = DEEPSEEK_R1, DEEPSEEK_R1.moe
    cfg = tf.DeepseekV3Config(vocab_size=c.vocab_size, hidden_size=c.hidden, intermediate_size=c.intermediate,
                              moe_intermediate_size=m.expert_dim, num_hidden_layers=c.layers, num_attention_heads=c.heads,
                              num_key_value_heads=c.kv_heads, n_shared_experts=m.shared, n_routed_experts=m.experts,
                              routed_scaling_factor=m.scaling, kv_lora_rank=c.kv_rank, q_lora_rank=c.q_rank,
                              qk_rope_head_dim=c.rope_head_dim, v_head_dim=c.v_dim, qk_nope_head_dim=c.nope_dim,
                              n_group=m.groups, topk_group=m.top_groups, num_experts_per_tok=m.top_k,
                              first_k_dense_replace=m.first_dense, norm_topk_prob=m.normalize,
                              max_position_embeddings=c.max_positions, tie_word_embeddings=False)
    with torch.device('meta'):
        model = tf.DeepseekV3ForCausalLM(cfg)
    got = {k: 0 for k in reasoning.parameters()}
    for name, p in model.named_parameters():
        part = ('embedding' if 'embed_tokens' in name else 'head' if name.startswith('lm_head') else 'routed_experts' if '.experts.' in name
                else 'shared_experts' if 'shared_experts' in name else 'router' if name.endswith('mlp.gate.weight')
                else 'dense_ffn' if '.mlp.' in name else 'attention' if 'self_attn' in name else 'norms')
        got[part] += p.numel()
    assert got == reasoning.parameters()
    assert sum(got.values()) == 671_026_404_352


def test_routing_matches_the_reference_layer(page):
    """The page's router, the plain Python one and the reference MoE layer choose the same experts."""
    torch = pytest.importorskip('torch')
    from llm_lab.reference.layers import MixtureOfExperts
    cfg = replace(DEEPSEEK_R1, hidden=DEEPSEEK_R1.moe.experts, moe=replace(DEEPSEEK_R1.moe, expert_dim=1, shared_dim=1))
    layer = MixtureOfExperts(cfg)
    with torch.no_grad():
        layer.router.weight.copy_(torch.eye(cfg.hidden))           # the router's logits are the input itself
    for r in page['routes']:
        experts, weights = reasoning.route(r['scores'])
        assert r['experts'] == experts and r['weights'] == pytest.approx(weights, abs=1e-12)
        assert len(set(e // 32 for e in experts)) <= 4 and math.fsum(weights) == pytest.approx(2.5)
        logits = torch.tensor([[math.log(s / (1 - s)) for s in r['scores']]], dtype=torch.float64)
        chosen, w = layer.route(logits.float())
        assert chosen[0].tolist() == experts
        assert w[0].tolist() == pytest.approx(weights, abs=1e-5)
