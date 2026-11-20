"""The explorer page: complete, offline, consistent with the published configs and the PyTorch reference."""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import embedded_specs, embedded_tokens

from llm_lab.explorer import page_tokens, render_html
from llm_lab.reference.config import CONFIGS

ROOT = Path(__file__).resolve().parents[1]


def test_page_is_complete_and_offline():
    html = render_html()
    assert html.startswith('<!DOCTYPE html>')
    for marker in ('__MATH_FONT__', '__CONTENT_JS__', '__MM_CONTENT_JS__', '__MM_MEDIA_JS__', '__MM_APP_JS__', '__MATHML__', '__APP_JS__', '__SPECS_JSON__'):
        assert marker not in html
    assert 'href="https://github.com/valentinsvelev/llm-internals-lab"' in html     # source link in the header
    # Nothing is fetched at runtime: no external scripts, stylesheets or fonts.
    assert not re.search(r'<script[^>]+src=', html)
    assert '<link' not in html
    assert not re.search(r'url\((?!data:|#)', html)
    for mech in ('mha', 'mqa', 'gqa', 'mla', 'gdn', 'dsa'):
        assert f'data-mech="{mech}"' in html
    assert 'var temml=' in html            # TeX is typeset in the page by the inlined Temml (MIT)
    assert 'font-family: "LabMath"' in html  # with an inlined subset of Latin Modern Math
    assert len(html) < 1_100_000          # about 50 KB: the Retrieval page's vectors; about 45 KB: the Reasoning page
    assert 'id="arch"' in html and 'id="diagram"' in html
    for page in ('architectures', 'multimodal', 'retrieval', 'reasoning', 'agents'):
        assert f'data-page="{page}"' in html
    assert 'id="mm"' in html and 'id="mmc"' in html
    for mode in ('images', 'audio', 'video', 'speech', 'imagegen', 'omni'):
        assert f'data-mode="{mode}"' in html


def test_dimensions_match_the_reference_configs():
    """The page's numbers and llm_lab.reference.config, both transcribed from the published config.json files."""
    from llm_lab.next_token import HF_IDS
    specs = embedded_specs(render_html())
    assert list(specs) == list(CONFIGS) == ['mha', 'mqa', 'gqa', 'mla', 'gdn', 'dsa']
    for key, cfg in CONFIGS.items():
        s = specs[key]
        assert s['hf'] == HF_IDS[key] and s['hf'] in cfg.source, key
        assert (s['d'], s['L'], s['V'], s['tied']) == (cfg.hidden, cfg.layers, cfg.vocab_size, cfg.tied), key
        # DeepSeek-V2-Lite: the config allows 163,840 positions; the page shows the model card's 32K context
        assert s['ctx'] == (32768 if key == 'mla' else cfg.max_positions), key
        if s['dff'] is not None:
            assert s['dff'] == cfg.intermediate, key
        if cfg.attention == 'gqa':
            assert (s['H'], s['G'], s['dh']) == (cfg.heads, cfg.kv_heads, cfg.head_dim), key
        else:
            assert (s['H'], s['dh'], s['dr'], s['dv'], s['dc']) == (cfg.heads, cfg.nope_dim, cfg.rope_head_dim, cfg.v_dim, cfg.kv_rank), key
        if cfg.moe:
            m = cfg.moe
            assert (s['experts'], s['active'], s['shared'], s['dExpert'], s['firstDense']) == (
                m.experts, m.top_k, m.shared, m.expert_dim, m.first_dense), key
        attention_layer = cfg.full_attention_interval - 1 if cfg.full_attention_interval else 0
        assert s['cache_per_token_per_layer'] == cfg.cache_values_per_token(attention_layer), key
        assert s['cache_layers'] == sum(cfg.layer_type(i) == 'attention' for i in range(cfg.layers)), key
    # and the published numbers themselves, for the values the page is about
    assert {k: s['cache_per_token_per_layer'] for k, s in specs.items()} == {
        'mha': 2 * 12 * 64, 'mqa': 2 * 1 * 256, 'gqa': 2 * 2 * 64, 'mla': 512 + 64, 'gdn': 2 * 2 * 256, 'dsa': 512 + 64 + 128}
    gdn, dsa = CONFIGS['gdn'], CONFIGS['dsa']
    assert specs['gdn']['lin'] == {'qk': gdn.lin_qk_heads, 'v': gdn.lin_v_heads, 'dk': gdn.lin_head_dim, 'dv': gdn.lin_head_dim, 'conv': gdn.conv_kernel}
    assert (specs['gdn']['interval'], specs['gdn']['ropeDim'], specs['gdn']['thetaN']) == (gdn.full_attention_interval, gdn.rotary_dim, gdn.rope_theta)
    assert specs['dsa']['index'] == {'heads': dsa.index_heads, 'dim': dsa.index_dim, 'topk': dsa.index_topk}
    assert specs['dsa']['dq'] == dsa.q_rank
    # Gated DeltaNet keeps 32 states of 128 × 128 plus the convolution's last 3 inputs, whatever the length
    channels = 2 * gdn.lin_qk_heads * gdn.lin_head_dim + gdn.lin_v_heads * gdn.lin_head_dim
    assert gdn.lin_v_heads * gdn.lin_head_dim ** 2 + 3 * channels == 548_864


def test_decoding_math_in_node():
    """The page's decoding functions, run directly: filters, beam search and speculative sampling."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('node not installed')
    content = ROOT / 'web/src/content.js'
    script = f"""
const L = require({json.dumps(str(content))});
const d = L.nextTokenData('mha', null);
const kept = (m, o, t = 1) => L.decodeDist(d, m, t, o).filter(c => c.kept).map(c => c.text.trim());
const out = {{
  topk: kept('topk', {{k: 3}}), topp: kept('topp', {{p: 0.6}}), minp: kept('minp', {{minp: 0.2}}),
  typical: kept('typical', {{typ: 0.5}}), greedy: kept('greedy', {{}}),
  sums: ['topk', 'topp', 'minp', 'typical', 'temperature'].map(m => L.decodeDist(d, m, 0.8, {{k: 4, p: 0.9, minp: 0.1, typ: 0.9}}).reduce((a, c) => a + c.p, 0)),
}};
const b = L.beamSearch(d, 2);
out.beam = [b.best.text, d.cands[b.best.from].text.trim(), b.greedy.text.trim(), b.best.score > b.greedy.score];
const sp = L.specDist(d, 0.8);
out.spec = sp.map(c => Math.min(c.p, c.q) + (1 - c.acc) * c.r - c.p);
out.specR = sp.reduce((a, c) => a + c.r, 0);
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    assert r['topk'] == ['smaller', 'pieces', 'multiple'] and r['greedy'] == ['smaller']
    assert r['topp'] == ['smaller', 'pieces']                       # 36% + 27% ≥ 0.6
    assert r['minp'] == ['smaller', 'pieces', 'multiple', 'sub']   # p ≥ 0.2 × 36%
    assert r['typical'] == ['smaller', 'pieces', 'multiple', 'sub']  # closest to H first, until 50%
    assert all(abs(x - 1) < 1e-9 for x in r['sums'])
    assert r['beam'] == ['.', 'pieces', 'pieces', True]             # beam prefers " pieces." over greedy " smaller pieces"
    assert max(abs(x) for x in r['spec']) < 1e-12 and abs(r['specR'] - 1) < 1e-12   # speculative sampling reproduces p


def test_multimodal_page_matches_the_vision_reference():
    """The multimodal page's numbers and arithmetic (web/src/mm-content.js), run in node, against
    llm_lab.reference.vision, which tests/test_vision_models.py checks against transformers."""
    from llm_lab.reference import vision as V
    node = shutil.which('node')
    if node is None:
        pytest.skip('node not installed')
    rng = random.Random(0)
    sizes = [(336, 336), (480, 640), (768, 1024), (1080, 1920), (1200, 800), (900, 300), (100, 100), (3024, 4032), (64, 2000)]
    sizes += [(rng.randint(28, 5000), rng.randint(28, 5000)) for _ in range(400)]
    segments = [['text', 3], ['image', [1, 6, 10]], ['text', 4], ['image', [1, 4, 2]], ['text', 2]]
    is_image = [False, False, True, True, True, False, False, True, True, False]
    script = f"""
const M = require({json.dumps(str(ROOT / 'web/src/mm-content.js'))});
const sizes = {json.dumps(sizes)};
const out = {{ models: {{}}, plans: {{}}, cache: {{}} }};
for (const k of M.ORDER) {{
  const m = M.VLMS[k];
  out.models[k] = {{ name: m.name, hf: m.hf, enc: m.enc, dec: m.dec, params: [m.enc.params, m.conn.params, m.dec.params], total: m.total }};
  out.plans[k] = sizes.map(([h, w]) => {{ const p = M.imagePlan(k, h, w); return [p.resized, p.canvas, p.patches, p.tiles, p.encoderPositions, p.tokens, p.cross]; }});
  out.cache[k] = sizes.map(([h, w]) => M.cacheValues(k, h, w));
}}
out.mrope = M.mropePositions({json.dumps(segments)});
out.axes = M.mropePairAxes();
out.mask = M.gemmaMask({json.dumps(is_image)});
out.tex = M.allTex().length;
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    assert list(r['models']) == list(V.VLMS)
    for key, m in V.VLMS.items():
        js, e, d = r['models'][key], m.encoder, m.decoder
        assert (js['name'], js['hf'], js['enc']['name'], js['dec']['name']) == (m.name, m.source, e.name, d.name), key
        assert (js['enc']['layers'], js['enc'].get('global', 0), js['enc']['d'], js['enc']['heads'], js['enc']['ff'], js['enc']['patch'], js['enc']['size']) == (
            e.layers, e.global_layers, e.hidden, e.heads, e.intermediate, e.patch, e.image_size), key
        assert (js['dec']['layers'], js['dec']['d'], js['dec']['heads'], js['dec']['kv'], js['dec']['dh'], js['dec']['ff'], js['dec']['V'], js['dec']['ctx'], tuple(js['dec']['cross'])) == (
            d.layers, d.hidden, d.heads, d.kv_heads, d.head_dim, d.intermediate, d.vocab, d.max_positions, d.cross_layers), key
        assert (js['dec'].get('window'), js['dec'].get('globalEvery', 0)) == (d.sliding_window, d.sliding_pattern), key
        assert tuple(js['params']) == V.PARAMETERS[key] and js['total'] == sum(V.PARAMETERS[key]), key
        for (h, w), plan, cost in zip(sizes, r['plans'][key], r['cache'][key]):
            p = V.image_plan(key, h, w)
            assert plan == [list(p.resized), list(p.canvas), list(p.patches), p.tiles, p.encoder_positions, p.tokens, p.cross_positions], (key, h, w)
            assert cost == V.image_cache_values(key, h, w), (key, h, w)
    assert r['mrope'] == V.mrope_positions([(k, tuple(v) if isinstance(v, list) else v) for k, v in segments])
    assert r['axes'] == V.mrope_pair_axes()
    assert r['mask'] == V.gemma_attention_mask(is_image)
    assert r['tex'] > 50          # the equations the page can show (build.mjs renders each one)


def test_media_views_match_the_reference():
    """Audio, video, speech out, image out and all together (web/src/mm-media.js), run in node, against
    llm_lab.reference.{audio, video, speech, imagegen}, which tests/test_media_models.py checks against transformers."""
    from llm_lab.reference import audio as A
    from llm_lab.reference import imagegen as G
    from llm_lab.reference import speech as SP
    from llm_lab.reference import video as V
    node = shutil.which('node')
    if node is None:
        pytest.skip('node not installed')
    rng = random.Random(1)
    seconds = [0.3, 1, 2.5, 5, 7.31, 29.99, 30, 30.01, 45, 90, 600, 1800] + [round(rng.uniform(0.1, 2400), 2) for _ in range(150)]
    videos = [(s, h, w, fps) for s in (1, 3, 10, 60, 300, 1200, 4.7) for h, w in ((720, 1280), (1080, 1920), (480, 640), (1280, 720), (240, 320))
              for fps in (30, 25)] + [(round(rng.uniform(1, 3000), 1), rng.randint(100, 2200), rng.randint(100, 2200), 30) for _ in range(80)]
    interleave = [[3, 4, 30], [5, 1, 65], [10, 16, 130], [2, 220, 3]]
    script = f"""
const MM = require({json.dumps(str(ROOT / 'web/src/mm-content.js'))});
const X = require({json.dumps(str(ROOT / 'web/src/mm-media.js'))});
const seconds = {json.dumps(seconds)}, videos = {json.dumps(videos)};
const out = {{ audio: {{}}, video: {{}}, speech: {{}}, gen: {{}}, models: X.MODELS }};
for (const k of ['qwen2audio', 'voxtral']) out.audio[k] = seconds.map(s => X.audioPlan(k, s));
for (const k of ['qwen3vl', 'llava_ov']) out.video[k] = videos.map(([s, h, w, f]) => X.videoPlan(k, s, h, w, f));
out.frames = videos.map(([s, h, w, f]) => X.sampleFrames(s, f));
for (const k of ['qwen3omni', 'moshi']) out.speech[k] = seconds.map(s => X.speechPlan(k, s));
out.omniAudio = Array.from({{ length: 3200 }}, (_, i) => X.omniAudioTokens(i + 1));
out.interleave = {json.dumps(interleave)}.map(a => X.omniInterleave(...a));
out.omni = [3, 10, 30].map(s => X.omniPlan(s));
out.delays = X.moshiDelays();
out.gen = {{ janus: X.genPlan('janus'), emu3: X.GEN.ratios.map(r => [X.emu3Grid(r), X.genPlan('emu3', r)]) }};
out.tex = X.allTex().length;
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    for key in ('qwen2audio', 'voxtral'):
        m, js = A.ALMS[key], r['models'][key]
        assert (js['name'], js['hf'], tuple(js['params'])) == (m.name, m.source, A.PARAMETERS[key])
        assert (js['enc']['layers'], js['enc']['d'], js['enc']['heads'], js['enc']['ff'], js['enc']['mels']) == (
            m.encoder.layers, m.encoder.hidden, m.encoder.heads, m.encoder.intermediate, m.encoder.mels)
        d = js['dec']
        assert (d['layers'], d['d'], d['heads'], d['kv'], d['dh'], d['ff'], d['V']) == (
            m.decoder.layers, m.decoder.hidden, m.decoder.heads, m.decoder.kv_heads, m.decoder.head_dim, m.decoder.intermediate, m.decoder.vocab)
        for s, plan in zip(seconds, r['audio'][key]):
            p = A.audio_plan(key, s)
            assert [plan['samples'], plan['chunks'], plan['melFrames'], plan['encoderPositions'], plan['tokens']] == [
                p.samples, p.chunks, p.mel_frames, p.encoder_positions, p.tokens], (key, s)
            assert plan['dropped'] == pytest.approx(p.dropped_seconds)
    for key in ('qwen3vl', 'llava_ov'):
        m, js = V.VIDEO_MODELS[key], r['models'][key]
        assert (js['hf'], tuple(js['params'])) == (m.source, V.PARAMETERS[key])
        assert (js['enc']['layers'], js['enc']['d'], js['dec']['layers'], js['dec']['d'], js['dec']['kv']) == (
            m.encoder.layers, m.encoder.hidden, m.decoder.layers, m.decoder.hidden, m.decoder.kv_heads)
        for (s, h, w, f), plan in zip(videos, r['video'][key]):
            p = V.video_plan(key, s, h, w, f)
            assert [plan['frames'], plan['frameSize'], plan['groups'], plan['perGroup'], plan['tokens']] == [
                p.frames, list(p.frame_size), p.groups, p.tokens_per_group, p.tokens], (key, s, h, w, f)
            assert plan['stamps'] == pytest.approx(list(p.timestamps))
    assert r['frames'] == [V.sample_frames(s, f) for s, h, w, f in videos]
    for key in ('qwen3omni', 'moshi'):
        m, js = SP.SPEECH_MODELS[key], r['models'][key]
        assert (js['hf'], tuple(js['params']), js['codebooks'], js['codebook']) == (m.source, SP.PARAMETERS[key], m.codebooks, m.codebook_size)
        assert (js['main']['layers'], js['main']['d'], js['main']['kv'], js['inner']['layers']) == (m.main_layers, m.main_hidden, m.main_kv_heads, m.inner_layers)
        for s, plan in zip(seconds, r['speech'][key]):
            p = SP.speech_plan(key, s)
            assert [plan['frames'], plan['codes'], plan['mainSteps'], plan['innerSteps'], plan['samples'], plan['kv']] == [
                p.frames, p.codes, p.main_steps, p.inner_steps, p.samples, p.kv_values], (key, s)
    assert r['omniAudio'] == [SP.omni_audio_tokens(i + 1) for i in range(3200)]
    assert r['interleave'] == [SP.omni_interleave(*a) for a in interleave]
    for s, plan in zip((3, 10, 30), r['omni']):
        p = SP.omni_plan(s)
        assert [plan['grids'], plan['perGrid'], plan['audioTokens'], plan['order'], plan['positions']] == [p.grids, p.tokens_per_grid, p.audio_tokens, p.order, p.positions]
    assert r['delays'] == SP.moshi_delays()
    omni = r['models']['omni']
    assert omni['params'] == list(SP.QWEN3_OMNI_THINKER.values()) + [SP.PARAMETERS['qwen3omni'][1] + SP.PARAMETERS['qwen3omni'][2]]
    for key in ('janus', 'emu3'):
        m, js = G.IMAGEGEN_MODELS[key], r['models'][key]
        assert (js['hf'], tuple(js['params']), js['codebook'], js['down'], js['codeDim']) == (m.source, G.PARAMETERS[key], m.codebook, m.downsample, m.code_dim)
        assert (js['dec']['layers'], js['dec']['d'], js['dec']['kv'], js['dec']['V'], js['dec']['ctx']) == (
            m.decoder.layers, m.decoder.hidden, m.decoder.kv_heads, m.decoder.vocab, m.decoder.max_positions)
    as_list = lambda p: [list(p.grid), list(p.pixels), p.codes, p.steps, p.sequences, p.kv_values]  # noqa: E731
    js = r['gen']['janus']
    assert [js['grid'], js['pixels'], js['codes'], js['steps'], js['sequences'], js['kv']] == as_list(G.gen_plan('janus'))
    for ratio, (grid, js) in zip(G.EMU3_RATIOS, r['gen']['emu3']):
        assert grid == list(G.emu3_grid(ratio))
        assert [js['grid'], js['pixels'], js['codes'], js['steps'], js['sequences'], js['kv']] == as_list(G.gen_plan('emu3', ratio))
    assert r['tex'] > 50


def test_example_tokenization_is_consistent():
    examples = page_tokens(render_html())['examples']
    assert [e['key'] for e in examples] == ['subwords', 'fact', 'story', 'attention']
    for ex in examples:
        assert len(ex['candidates']) == len(ex['logits']) == 14 and ex['logits'] == sorted(ex['logits'], reverse=True)
        assert set(ex['beam']) == set(ex['candidates'][:3])          # second steps for the three most likely tokens
        for slug, t in ex['models'].items():
            text = ''.join('' if p.startswith('<') else p.replace('Ġ', ' ').replace('▁', ' ') for p in t['pieces'])
            assert text == ex['sentence'], (ex['key'], slug)
            assert len(t['ids']) == len(t['pieces']) == len(t['words']) <= 10      # the diagram has 10 rows
            assert len(t['candidate_ids']) == len(ex['candidates'])
            assert len(t['continuation_ids']) == len(ex['continuations'])
            assert t['words'][-1] == len(ex['sentence'].split(' ')) - 1
        assert ex['models']['gdn']['ids'] == ex['models']['gqa']['ids']     # Qwen3-Next shares Qwen's tokenizer
        assert ex['models']['mqa']['ids'][0] == 2 and ex['models']['mla']['ids'][0] == 100000   # BOS
        assert ex['models']['dsa']['ids'][0] == 0                            # DeepSeek-V3 tokenizer: BOS is ID 0
    toks = embedded_tokens(render_html())
    assert toks['sentence'] == 'Tokenizers split unpronounceable words into'
    for t in toks['models'].values():
        # 'unpronounceable' is split into four subword tokens by every tokenizer
        assert t['words'].count(2) == 4 and t['words'].count(0) == 2
    # GPT-2 IDs, checked against two independent tokenizer implementations when generated
    assert toks['models']['mha']['ids'] == [30642, 11341, 6626, 555, 31186, 8652, 540, 2456, 656]
    att = embedded_tokens(render_html(), 'attention')['models']       # only GPT-2 has no token for “Attention”
    assert att['mha']['pieces'][:2] == ['Att', 'ention'] and all(t['words'].count(0) == 1 for k, t in att.items() if k != 'mha')
    assert embedded_tokens(render_html(), 'fact')['candidates'][0] == ' France'


def test_cli_writes_standalone_file(tmp_path):
    out = tmp_path / 'explorer.html'
    subprocess.run([sys.executable, '-m', 'llm_lab.explorer', '--out', str(out)], check=True)
    assert out.read_text(encoding='utf-8') == render_html()
