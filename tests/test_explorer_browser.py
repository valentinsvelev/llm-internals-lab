"""The explorer in a real browser: every tab, step, pop-up, decoding method, example input and page.

Needs Playwright and a Chromium: pip install -e ".[dev]" && python -m playwright install chromium
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from helpers import embedded_specs, embedded_tokens, fake_bundle

from llm_lab.explorer import render_html


def flat(text: str) -> str:
    """Pop-up text without the line breaks MathML puts between tokens, math italic folded to plain letters."""
    import unicodedata
    return unicodedata.normalize('NFKC', ''.join(text.replace('\u2061', '').split()))


def test_explorer_in_a_real_browser():
    playwright = pytest.importorskip('playwright.sync_api', reason='browser test: pip install -e ".[dev]"')
    from conftest import launch_chromium
    specs = embedded_specs(render_html())
    tk = embedded_tokens(render_html())['models']
    with playwright.sync_playwright() as p:
        browser = launch_chromium(p)
        if browser is None:
            pytest.skip('no Chromium for Playwright: run `python -m playwright install chromium`')
        try:
            for width in (1280, 420):
                page = browser.new_page(viewport={'width': width, 'height': 900})
                errors = []
                page.on('pageerror', lambda err, errors=errors: errors.append(str(err)))
                page.set_content(render_html(), wait_until='load')
                for slug in ('mha', 'mqa', 'gqa', 'mla', 'gdn', 'dsa'):
                    page.locator(f'.tab[data-mech="{slug}"]').click()
                    heads = {'gdn': 32, 'dsa': 0}.get(slug, specs[slug]['H'])
                    assert page.locator('#diagram .head').count() == heads
                    assert page.locator('#steps .step').count() == 9  # overview + 8 steps
                    assert page.locator('#steps math').count() >= 1
                    if heads:  # following a head updates the per-head walkthrough
                        page.locator('#diagram .head').nth(heads - 1).click()
                        assert f'Inside {"value" if slug == "gdn" else "query"} head {heads}' in page.locator('#diagram .cap').text_content()
                    for i in range(1, 9):  # each step highlights part of the diagram and shows its equation
                        page.locator(f'#step-{i}').click()
                        assert page.locator('#diagram.focusing').count() == 1
                        assert page.locator('#diagram .on').count() > 0, (slug, i)
                        assert page.locator('.step.active math').count() >= 1
                    page.keyboard.press('Escape')
                    assert page.locator('#diagram.focusing').count() == 0
                # Year labels and permanently visible definitions work at desktop and mobile widths.
                assert page.locator('#symbol-key [data-symbol]').count() >= 10
                for slug, year in (('mha', '2017'), ('mqa', '2019'), ('gqa', '2023'),
                                   ('mla', '2024'), ('gdn', '2024'), ('dsa', '2025–26')):
                    assert page.locator(f'#tabs .tab[data-mech="{slug}"] .tab-year').inner_text() == year
                # The legend belongs inside the SVG (and not above it as separate chips).
                assert page.locator('#arch foreignObject#symbol-key').count() == 1
                assert page.locator('#arch .diagram-legend').count() == 1
                assert page.locator('section#symbol-key').count() == 0
                assert page.locator('#arch text').filter(has_text='Legend').count() == 1
                assert page.locator('#arch .diagram-legend').evaluate(
                    '(el) => el.scrollHeight <= el.clientHeight')
                # The legend must clear the transformer block, including at mobile widths.
                assert page.locator('#arch foreignObject#symbol-key').evaluate(
                    '''el => {
                        const block = document.querySelector('#arch .blockbox');
                        if (!block) return false;
                        const gap = block.getBoundingClientRect().top - el.getBoundingClientRect().bottom;
                        return gap >= 4;
                    }''')
                # Legend math should be real MathML, including x-hat; P belongs to GPT-2,
                # RoPE's dial belongs to rotary models, and theta in the stats is typeset.
                page.locator('.tab[data-mech="mha"]').click()
                assert page.locator('#symbol-key [data-symbol="xhat"] math').count() == 1
                assert page.locator('#symbol-key [data-symbol="pos"]').is_visible()
                assert page.locator('#symbol-key [data-symbol="rope"]').is_hidden()
                assert page.locator('#symbol-key [data-symbol="att"] math').count() == 1
                assert page.locator('#symbol-key [data-symbol="y"] math').count() == 1
                page.locator('.tab[data-mech="mqa"]').click()
                assert page.locator('#symbol-key [data-symbol="pos"]').is_hidden()
                assert page.locator('#symbol-key [data-symbol="rope"]').is_visible()
                assert page.locator('#symbol-key .symbol-dial').count() == 1
                assert page.locator('#stats math').count() >= 1    # theta in RoPE position stats
                page.locator('.tab[data-mech="gdn"]').click()
                assert page.locator('#symbol-key [data-symbol="rope"]').is_hidden()
                assert page.locator('#symbol-key [data-symbol="att"]').is_hidden()
                page.locator('#block-toggle button[data-v="gattn"]').click()
                assert page.locator('#symbol-key [data-symbol="rope"]').is_visible()
                assert page.locator('#symbol-key [data-symbol="att"]').is_visible()
                # v5: output arrows, sorted references and model-specific symbols.
                names = [li.inner_text().split(' (')[0] for li in page.locator('#sources li').all()]
                assert names == sorted(names, key=str.casefold)
                page.locator('.tab[data-mech="mha"]').click()
                norm = page.locator('#arch g[data-h="fnorm"] .pill-l')
                assert norm.text_content() == 'LayerNorm'
                assert float(page.locator('#arch g[data-h="fnorm"] .pillbox').get_attribute('width')) >= 90
                lengths = [page.locator(f'#arch line[data-h="{key}"]').evaluate(
                    'el => +el.getAttribute("y2") - +el.getAttribute("y1")')
                    for key in ('logit', 'dist', 'decbox')]
                assert lengths == [22, 22, 22], lengths
                for slug, keys in (
                    ('mla', ('ckv', 'kr', 'moe')),
                    ('gdn', ('conv', 'gate-a', 'gate-b', 'state', 'read', 'zg', 'moe')),
                    ('dsa', ('ckv', 'kr', 'ki', 'ibar', 'moe')),
                ):
                    page.locator(f'.tab[data-mech="{slug}"]').click()
                    if slug == 'gdn':
                        page.locator('#block-toggle button[data-v="gdn"]').click()
                    for key in keys:
                        assert page.locator(f'#symbol-key [data-symbol="{key}"]').is_visible(), (slug, key)
                    assert page.locator('#arch .diagram-legend').evaluate(
                        'el => el.scrollHeight <= el.clientHeight + 1'), slug
                page.locator('.tab[data-mech="gdn"]').click()
                page.locator('#block-toggle button[data-v="gattn"]').click()
                assert page.locator('#symbol-key [data-symbol="gt"]').is_visible()
                assert page.locator('#symbol-key [data-symbol="gate-a"]').count() == 0
                page.locator('.tab[data-mech="gdn"]').click()
                if width >= 1000:
                    lead = page.locator('#lead')
                    assert lead.evaluate('el => el.clientHeight <= 1.5 * parseFloat(getComputedStyle(el).lineHeight)')
                # v6: detailed legend pop-ups for base and model-specific symbols.
                page.locator('.tab[data-mech="mha"]').click()
                for sym, phrase in (('emb', 'Embedding table'), ('xhat', 'Normalized token vector'),
                                    ('q', 'Query'), ('k', 'Key'), ('v', 'Value'),
                                    ('att', 'Attention weights'), ('y', 'Mixer output'),
                                    ('logit', 'Logits'), ('pos', 'Position table')):
                    item = page.locator(f'#symbol-key [data-symbol="{sym}"]')
                    assert item.get_attribute('tabindex') == '0'
                    assert item.get_attribute('data-pop') == f'legend:{sym}'
                    item.hover(force=True)
                    assert page.locator('#pop').is_visible(), sym
                    assert phrase in page.locator('#pop').inner_text(), sym
                page.locator('#symbol-key [data-symbol="q"]').focus()
                assert 'What am I looking for?' in page.locator('#pop').inner_text()
                page.locator('.tab[data-mech="mla"]').click()
                for sym in ('rope', 'ckv', 'kr', 'moe'):
                    page.locator(f'#symbol-key [data-symbol="{sym}"]').focus()
                    assert page.locator('#pop').is_visible(), sym
                    assert len(page.locator('#pop .pop-t').inner_text()) > 65, sym
                page.locator('.tab[data-mech="dsa"]').click()
                for sym in ('ki', 'ibar'):
                    page.locator(f'#symbol-key [data-symbol="{sym}"]').focus()
                    assert page.locator('#pop').is_visible(), sym
                page.locator('.tab[data-mech="gdn"]').click()
                page.locator('#block-toggle button[data-v="gdn"]').click()
                for sym in ('conv', 'gate-a', 'gate-b', 'state', 'read', 'zg'):
                    page.locator(f'#symbol-key [data-symbol="{sym}"]').focus()
                    assert page.locator('#pop').is_visible(), sym
                page.locator('#block-toggle button[data-v="gattn"]').click()
                page.locator('#symbol-key [data-symbol="gt"]').focus()
                assert page.locator('#pop').is_visible()
                # Leave the tour controls with a clean, un-focused legend.
                page.locator('#tour-start').focus()
                # The guided tour: one idea per step on the GPT-2 diagram, ending in the six mechanisms.
                page.locator('.tab[data-mech="gqa"]').click()
                page.locator('#tour-start').click()
                assert page.locator('.tab.active').get_attribute('data-mech') == 'mha'
                assert page.locator('#arch-card').is_hidden()
                for i in range(8):
                    assert f'step {i + 1} of 8' in page.locator('#tour').inner_text()
                    assert page.locator('#arch.touring').count() == 1 and page.locator('#arch .on').count() > 0
                    assert page.locator('#tour .tour-term').count() == 1
                    if i == 1:
                        assert page.locator('#symbol-key [data-symbol="emb"].active').count() == 1
                    if i == 2:
                        assert page.locator('#symbol-key [data-symbol].active').count() == 4
                    if i == 5:
                        assert page.locator('#symbol-key [data-symbol="logit"].active').count() == 1
                    if i == 6:
                        assert page.locator('#symbol-key [data-symbol].active').count() == 2
                    if i < 7:
                        page.locator('#tour .next').click()
                assert page.locator('#tour .tour-ans button').count() == 6
                assert '2025–26' in page.locator('#tour .tour-ans').inner_text()
                page.keyboard.press('ArrowLeft')
                assert 'step 7 of 8' in page.locator('#tour').inner_text()
                page.keyboard.press('ArrowRight')
                page.locator('#tour .tour-ans button[data-mech="gdn"]').click()
                assert page.locator('.tab.active').get_attribute('data-mech') == 'gdn'
                assert page.locator('#arch.touring').count() == 0 and page.locator('#tour-start').count() == 1
                # Qwen's block-sparse variant replaces the token-level indexer in "Attention up close".
                page.locator('.tab[data-mech="dsa"]').click()
                assert page.locator('#sparse-toggle').is_visible()
                assert page.locator('#diagram .ibar.on-s').count() == 8           # toy budget: 8 tokens
                page.locator('#sparse-toggle button[data-v="qsa"]').click()
                assert page.locator('#diagram .ibar.on-s').count() == 2           # 2 blocks of 4 …
                assert page.locator('#diagram .kvk.on-s').count() == 10           # … plus the 2 tokens of the current block
                assert 'Qwen3.8-Flash-Next' in page.locator('#steps').inner_text()
                page.locator('#sparse-toggle button[data-v="dsa"]').click()
                # The whole-model diagram: overview + one chip per step, each highlighting a part.
                for slug, views in (('mha', [None]), ('mla', ['moe', 'dense']), ('gdn', ['gdn', 'gattn']), ('dsa', ['moe', 'dense'])):
                    page.locator(f'.tab[data-mech="{slug}"]').click()
                    for v in views:
                        if v:
                            page.locator(f'#block-toggle button[data-v="{v}"]').click()
                        n = 17 if v == 'gdn' or slug == 'dsa' else 16
                        assert page.locator('#arch-chips .chipbtn').count() == n + 1, (slug, v)
                        for i in range(n):
                            page.locator(f'#arch-chips .chipbtn[data-i="{i}"]').click()
                            assert page.locator('#arch.focusing').count() == 1
                            assert page.locator('#arch .on').count() > 0, (slug, v, i)
                            assert page.locator('#arch-card math').count() >= 1
                    page.locator('#arch .hit[aria-label$="sublayer"]').first.click(position={'x': 12, 'y': 12})
                    assert page.locator('#arch-card h3').inner_text()[1:].strip()
                page.locator('.tab[data-mech="mla"]').click()
                page.locator('#block-toggle button[data-v="moe"]').click()
                page.locator('#arch .hit[aria-label="Mixture-of-experts layer"]').click(position={'x': 12, 'y': 12})
                assert 'top 6 run' in page.locator('#arch-card').inner_text()
                assert page.locator('#arch .tick.pick').count() == 6 * len(tk['mla']['ids'])
                # Gated DeltaNet: a chain of states, one per token, instead of an attention matrix.
                page.locator('.tab[data-mech="gdn"]').click()
                page.locator('#block-toggle button[data-v="gdn"]').click()
                T = len(tk['gdn']['ids'])
                assert page.locator('#arch .scell').count() == T and page.locator('#arch .chainl').count() == T - 1
                assert page.locator('#arch .acell').count() == 0 and page.locator('#arch .dial').count() == 0
                page.locator('#arch [data-pop="S:3"]').hover()
                assert 'α3S2' in flat(page.locator('#pop').inner_text()) and page.locator('#pop .pop-c math').count() >= 2
                page.locator('#block-toggle button[data-v="gattn"]').click()
                assert page.locator('#arch .rope-l').text_content() == 'partial RoPE'
                assert page.locator('#arch .gtcell').count() == 4 * T
                # DeepSeek Sparse Attention: index scores, and attention only over the selected tokens.
                page.locator('.tab[data-mech="dsa"]').click()
                T = len(tk['dsa']['ids'])
                assert page.locator('#arch .kicell').count() == T
                assert page.locator('#arch #ibars .ibar.on-s').count() == 4      # toy budget for the last row
                assert page.locator('#arch .acell.drop').count() == sum(max(0, t + 1 - 4) for t in range(T))
                # Every token row carries its own q, k and v: hovering a token lights its row everywhere.
                page.locator('.tab[data-mech="gqa"]').click()
                shown = "[...document.querySelectorAll('#arch #arcs .qkvarc')].filter(e => getComputedStyle(e).display !== 'none').length"
                page.locator('#arch-chips .chipbtn[data-i="0"]').click()
                page.keyboard.press('Escape')
                assert page.evaluate(shown) == 0                                  # arrows only while hovering a row
                page.locator('#arch [data-pop="tok:4"]').hover()
                assert page.evaluate(shown) == 3
                assert page.locator('#arch .zrow.rowhl').count() == 1             # … and its row of logits
                lit = page.locator('#arch [data-row="4"].rowhl')
                assert lit.count() > 14 + 2 + 2          # q cells of 14 heads, k and v of 2 KV heads, ...
                assert 'pron' in page.locator('#pop').inner_text()
                assert page.locator('#arch .lookup').count() == 1           # its row of E, linked while hovering
                page.locator('#arch [data-pop="word:2"]').hover()
                assert '4 tokens' in page.locator('#pop').inner_text()
                # Decoding: logits → probabilities → the chosen method.
                page.locator('#arch-chips .chipbtn[data-i="15"]').click()
                rows = '#arch .prow:not(.cut):not(.tail)'
                page.select_option('#dec-method', 'greedy')
                assert page.locator(rows).count() == 1
                page.select_option('#dec-method', 'topk')
                page.locator('#dec-k').fill('3')
                page.locator('#dec-k').dispatch_event('input')
                assert page.locator(rows).count() == 3
                assert page.locator('#arch .lbar').count() == 8                 # a logit bar per listed token
                page.locator('#arch [data-pop="cand:0"]').hover()
                pop = page.locator('#pop').inner_text()
                assert 'illustrative' in pop and "p′=" in flat(pop) and 'log∑' in flat(pop)
                assert page.locator('#pop .pop-c math').count() >= 5           # every calculation line is typeset
                for meth, n in (('minp', 4), ('typical', 5), ('temperature', 8)):   # at τ = 0.8
                    page.locator(f'#arch [data-pop="meth:{meth}"]').click()
                    assert page.locator(rows).count() == n, meth
                page.locator('#arch [data-pop="meth:beam"]').click()
                assert 'best: “␣pieces.”' in page.locator('#arch').text_content()
                assert page.locator('#ghost1 text').text_content() == '.'      # beam search appends two tokens
                page.locator('#arch [data-pop="meth:spec"]').click()
                assert page.locator('#arch .prow .qbar').count() == 8 and 'acceptance rate' in page.locator('#arch').text_content()
                page.locator('#arch [data-pop="cand:1"]').hover()
                assert '=p' in flat(page.locator('#pop').inner_text())         # min(p,q) + (1 − Σ min) r = p
                page.locator('#arch [data-pop="meth:topk"]').click()
                # RoPE: dials on q and k for every RoPE model, none for GPT-2; MLA's is decoupled.
                for slug, label in (('mha', None), ('mqa', 'RoPE'), ('gqa', 'RoPE'), ('mla', 'decoupled RoPE'), ('dsa', 'decoupled RoPE')):
                    page.locator(f'.tab[data-mech="{slug}"]').click()
                    dials = page.locator('#arch .dial').count()
                    assert dials == (0 if label is None else 2 * len(tk[slug]['ids'])), slug
                    if label:
                        assert page.locator('#arch .rope-l').first.text_content() == label
                page.locator('.tab[data-mech="mla"]').click()
                page.locator('#arch [data-pop="rope:6:k"]').first.hover()
                assert 'pair31:2.0×10−5rad' in flat(page.locator('#pop').inner_text())   # YaRN slows pair 31 by 40×
                page.locator('#arch [data-pop="emb:4"]').first.hover()
                assert 'onehot(405)' in flat(page.locator('#pop').inner_text())
                # The loop label only appears on hover.
                assert page.evaluate("getComputedStyle(document.querySelector('#arch .loop-t')).opacity") == '0'
                gqa = page.locator('.tab[data-mech="gqa"]')
                if width == 1280:
                    import pathlib
                    import tempfile
                    tmp = pathlib.Path(tempfile.mkdtemp()) / 'n.json'
                    tmp.write_text(json.dumps(fake_bundle()), encoding='utf-8')
                    p2 = browser.new_page(viewport={'width': 1280, 'height': 900})
                    p2.set_content(render_html(tmp), wait_until='load')
                    p2.locator('#arch [data-pop="cand:0"]').hover()
                    assert 'computed with openai-community/gpt2' in p2.locator('#pop').inner_text()
                    assert p2.locator('#arch .prow:not(.tail)').count() == 3
                    p2.close()
                gqa.click()
                assert '14 query heads into 2 groups of 7' in page.locator('#steps').inner_text()
                assert 'Repeat 24 blocks' in page.locator('#arch-chips').inner_text()
                # Logits: one row per token; the last one, z_T, feeds the distribution.
                T = len(tk['gqa']['ids'])
                assert page.locator('#arch .zrow').count() == T and page.locator('#arch .zrow.last .zlog').count() > 0
                # Example inputs: each is tokenized by every model's own tokenizer.
                page.locator('#arch .expill', has_text='A fact').click()
                fact = embedded_tokens(render_html(), 'fact')['models']
                assert page.locator('#arch .chip:not(.ghostchip)').count() == len(fact['gqa']['ids'])
                assert page.locator('#arch .zrow').count() == len(fact['gqa']['ids'])
                assert 'Parisisthecapital' in page.locator('#arch').text_content()
                page.locator('#arch [data-pop="meth:greedy"]').click()
                assert page.locator('#ghost0 text').text_content() == '␣France'
                page.locator('.tab[data-mech="mha"]').click()                    # the example stays when switching models
                assert page.locator('#arch .chip:not(.ghostchip)').count() == len(fact['mha']['ids'])
                page.locator('#arch .expill', has_text='Attention').click()
                assert [page.locator('#arch .chip-t').nth(i).text_content() for i in range(2)] == ['Att', 'ention']
                page.locator('#arch .expill', has_text='Subwords').click()
                assert page.locator('#arch .chip:not(.ghostchip)').count() == len(tk['mha']['ids'])
                gqa.click()
                # Pages: placeholders for what comes next; the mechanism keys only work on this page.
                page.locator('.pages a[data-page="agents"]').click()
                assert page.locator('.page[data-page="agents"]').is_visible() and page.locator('#arch').is_hidden()
                assert page.locator('.page[data-page="agents"] .plan-list li').count() == 5
                page.keyboard.press('3')
                page.locator('.pages a[data-page="architectures"]').click()
                assert page.locator('#arch').is_visible() and page.locator('.tab.active').get_attribute('data-mech') == 'gqa'
                assert page.locator('.pages a[aria-current="page"]').get_attribute('data-page') == 'architectures'
                assert errors == []
                page.close()
        finally:
            browser.close()


def _png(width: int, height: int) -> bytes:
    """A plain red PNG, for the "Your image…" button."""
    import struct
    import zlib
    raw = b''.join(b'\x00' + b'\xc8\x3c\x3c' * width for _ in range(height))
    chunk = lambda kind, data: struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))  # noqa: E731
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def test_multimodal_page_in_a_real_browser():
    playwright = pytest.importorskip('playwright.sync_api', reason='browser test: pip install -e ".[dev]"')
    from conftest import launch_chromium

    from llm_lab.reference import vision as V
    mib = lambda k, h, w: V.image_cache_values(k, h, w) * 2 / 2 ** 20  # noqa: E731
    with playwright.sync_playwright() as p:
        browser = launch_chromium(p)
        if browser is None:
            pytest.skip('no Chromium for Playwright: run `python -m playwright install chromium`')
        try:
            for width in (1280, 420):
                page = browser.new_page(viewport={'width': width, 'height': 900})
                errors = []
                page.on('pageerror', lambda err, errors=errors: errors.append(str(err)))
                page.set_content(render_html(), wait_until='load')
                page.locator('.pages a[data-page="multimodal"]').click()
                assert page.locator('#mm').is_visible() and page.locator('#arch').is_hidden()
                stat = lambda name, page=page: page.locator('#mm-stats div', has_text=name).locator('dd').text_content()  # noqa: E731
                for key in V.VLMS:
                    page.locator(f'#mm-tabs .tab[data-k="{key}"]').click()
                    assert page.locator('#mm-tabs .tab.active').get_attribute('data-k') == key
                    total = sum(V.PARAMETERS[key])
                    assert stat('Parameters') == f'{total / 1e9:.2f}B'
                    for j, (h, w) in enumerate([(336, 336), (480, 640), (768, 1024), (1080, 1920), (1200, 800)]):
                        page.locator(f'#mm-size button[data-i="{j}"]').click()
                        plan = V.image_plan(key, h, w)
                        assert stat('Image tokens here') == ('1 (+ 6,404 beside)' if key == 'llama' else f'{plan.tokens:,}'), (key, h, w)
                        assert page.locator('#mm .st-s').first.text_content() == f'{w} × {h} pixels'
                        rows = page.locator('#mm-cost-rows .crow .c-v').all_text_contents()
                        assert [float(r.split()[0].replace(',', '')) for r in rows] == pytest.approx([mib(k, h, w) for k in V.VLMS], rel=0.01)
                    # every step has its equation and highlights part of the diagram
                    chips = page.locator('#mm-chips .chipbtn')
                    assert chips.count() == 9
                    for s in range(1, 9):
                        chips.nth(s).click()
                        assert page.locator('#mm-card math').count() >= 1
                        assert page.locator('#mm.focusing [data-h].on').count() >= 1
                    page.locator('#mm-chips .chipbtn').first.click()
                    assert page.locator('#mm.focusing').count() == 0
                    # pop-ups on the encoder and on one decoder layer
                    page.locator('#mm [data-pop="enc"]').hover(position={'x': 12, 'y': 12})      # the box, not one of its layers
                    assert page.locator('#mm-pop').is_visible() and V.VLMS[key].encoder.name in page.locator('#mm-pop').text_content()
                    page.locator('#mm [data-pop="dlayer:3"]').hover()
                    assert 'Layer 4' in page.locator('#mm-pop b').text_content()
                    assert page.locator('#mmc rect').count() > 20                      # the up-close figure
                    page.mouse.move(0, 0)
                # model-specific details
                page.locator('#mm-tabs .tab[data-k="llama"]').click()
                page.locator('#mm-size button[data-i="2"]').click()
                assert page.locator('#mm .dlayer.cross').count() == 8
                assert page.locator('#mm .tile.on').count() == V.image_plan('llama', 768, 1024).tiles
                page.locator('#mm-tabs .tab[data-k="qwen"]').click()
                assert page.locator('#mm path.ds').count() == 6                           # three DeepStack mergers, in and out
                assert page.locator('#mmc .mnum').all_text_contents()[:15] == list('000111222333334')   # (t, row, col) per token
                page.locator('#mm-tabs .tab[data-k="gemma"]').click()
                assert page.locator('#mmc .mcell.bi:not(.key)').count() == 6                     # 4 image tokens: 3 + 2 + 1 later ones
                # keys: ← → steps, 1–4 models
                page.locator('#mm-title').click()
                page.keyboard.press('ArrowRight')
                assert page.locator('#mm-chips .chipbtn.active').get_attribute('data-i') == '0'
                page.keyboard.press('1')
                assert page.locator('#mm-tabs .tab.active').get_attribute('data-k') == 'llava'
                # your own image: only its size is used (and its colours, for the mosaics)
                page.locator('#mm-tabs .tab[data-k="qwen"]').click()
                page.set_input_files('#mm-file', files=[{'name': 'tall.png', 'mimeType': 'image/png', 'buffer': _png(300, 900)}])
                page.wait_for_function("document.querySelector('#mm-mine').textContent.startsWith('Yours')")
                assert page.locator('#mm-mine').text_content() == 'Yours · 300 × 900'
                assert stat('Image tokens here') == f'{V.image_plan("qwen", 900, 300).tokens:,}'
                assert page.locator('#mm image').count() == 2
                # the retrieval placeholder
                page.locator('.pages a[data-page="retrieval"]').click()
                assert page.locator('.page[data-page="retrieval"]').is_visible() and page.locator('#mm').is_hidden()
                assert page.locator('#retrieval-lab').is_visible()
                assert page.locator('#retr-flow .retr-stage').count() == 4
                page.locator('.pages a[data-page="multimodal"]').click()
                assert page.locator('#mm-tabs .tab.active').get_attribute('data-k') == 'qwen'
                assert errors == []
                page.close()
        finally:
            browser.close()


def _wav(seconds: float, rate: int = 16000) -> bytes:
    """A quiet 220 Hz tone as a 16-bit WAV, for the "Your audio…" button."""
    import io
    import math
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b''.join(struct.pack('<h', int(3000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(int(seconds * rate))))
    return buf.getvalue()


def test_multimodal_modalities_in_a_real_browser():
    playwright = pytest.importorskip('playwright.sync_api', reason='browser test: pip install -e ".[dev]"')
    from conftest import launch_chromium

    from llm_lab.reference import audio as A
    from llm_lab.reference import imagegen as G
    from llm_lab.reference import speech as SP
    from llm_lab.reference import video as V
    expected = {
        'audio': ('Audio tokens here', [5, 30, 120, 600, 1800], lambda k, v: A.audio_plan(k, v).tokens),
        'video': ('Video tokens here', [10, 60, 300, 1200], lambda k, v: V.video_plan(k, v).tokens),
        'speech': ('Codes written', [2, 10, 30, 120], lambda k, v: SP.speech_plan(k, v).codes),
        'imagegen': ('Tokens written', list(G.EMU3_RATIOS), lambda k, v: G.gen_plan(k, v).steps),
        'omni': ('Clip tokens', [3, 10, 30], lambda k, v: len(SP.omni_plan(v).order)),
    }
    models = {'audio': ['qwen2audio', 'voxtral'], 'video': ['qwen3vl', 'llava_ov'], 'speech': ['qwen3omni', 'moshi'],
              'imagegen': ['janus', 'emu3'], 'omni': ['omni']}
    with playwright.sync_playwright() as p:
        browser = launch_chromium(p)
        if browser is None:
            pytest.skip('no Chromium for Playwright: run `python -m playwright install chromium`')
        try:
            for width in (1280, 420):
                page = browser.new_page(viewport={'width': width, 'height': 900})
                errors = []
                page.on('pageerror', lambda err, errors=errors: errors.append(str(err)))
                page.set_content(render_html(), wait_until='load')
                page.locator('.pages a[data-page="multimodal"]').click()
                stat = lambda name, page=page: page.locator('#mm-stats div', has_text=name).locator('dd').text_content()  # noqa: E731
                for mode, (label, values, count) in expected.items():
                    page.locator(f'#mm-modes button[data-mode="{mode}"]').click()
                    assert page.locator('#mm-modes button.active').get_attribute('data-mode') == mode
                    assert page.locator('#mm-tabs .tab').count() == len(models[mode])
                    for key in models[mode]:
                        page.locator(f'#mm-tabs .tab[data-k="{key}"]').click()
                        assert page.locator('#mm-tabs .tab.active').get_attribute('data-k') == key
                        for i, v in enumerate(values):
                            page.locator(f'#mm-size button[data-i="{i}"]').click()
                            assert stat(label) == f'{count(key, v):,}', (mode, key, v)
                        chips = page.locator('#mm-chips .chipbtn')
                        assert chips.count() >= 6
                        for s in range(1, chips.count()):
                            chips.nth(s).click()
                            assert page.locator('#mm-card math').count() >= 1, (mode, key, s)
                            assert page.locator('#mm.focusing [data-h].on').count() >= 1, (mode, key, s)
                        chips.first.click()
                        page.locator('#mm [data-pop="step:cost"] .cost-v').first.hover()
                        assert page.locator('#mm-pop').is_visible()
                        page.mouse.move(0, 0)
                        assert page.locator('#mmc rect, #mmc text').count() > 10            # the up-close figure
                        assert page.locator('#mm-sources li').count() >= 2
                # your own recording: only its length and waveform are used
                page.locator('#mm-modes button[data-mode="audio"]').click()
                page.locator('#mm-tabs .tab[data-k="voxtral"]').click()
                page.set_input_files('#mm-file', files=[{'name': 'tone.wav', 'mimeType': 'audio/wav', 'buffer': _wav(42.5)}])
                page.wait_for_function("document.querySelector('#mm-mine').textContent.startsWith('Yours')")
                assert page.locator('#mm-mine').text_content() == 'Yours · 43 s'
                assert stat('Audio tokens here') == f'{A.audio_plan("voxtral", 42.5).tokens:,}'
                # your own video, if ffmpeg can make one: its length and size are used
                if shutil.which('ffmpeg'):
                    with tempfile.TemporaryDirectory() as tmp:
                        clip = Path(tmp) / 'clip.webm'
                        subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=4:size=320x240:rate=25',
                                        '-c:v', 'libvpx', '-b:v', '200k', str(clip)], check=True)
                        page.locator('#mm-modes button[data-mode="video"]').click()
                        page.locator('#mm-tabs .tab[data-k="qwen3vl"]').click()
                        page.set_input_files('#mm-file', str(clip))
                        page.wait_for_function("document.querySelector('#mm-mine').textContent.startsWith('Yours')", timeout=20000)
                        assert page.locator('#mm-mine').text_content() == 'Yours · 4 s, 320 × 240'
                        assert stat('Video tokens here') == f'{V.video_plan("qwen3vl", 4, 240, 320).tokens:,}'
                        assert page.locator('#mm image').count() >= 4                       # its frames, not the drawn example
                # links into a modality and model
                page.evaluate("location.hash = '#multimodal/speech/moshi'")
                page.wait_for_function("document.querySelector('#mm-tabs .tab.active')?.dataset.k === 'moshi'")
                assert page.locator('#mm-modes button.active').get_attribute('data-mode') == 'speech'
                page.evaluate("location.hash = '#multimodal/gemma'")              # older links still open the image view
                page.wait_for_function("document.querySelector('#mm-tabs .tab.active')?.dataset.k === 'gemma'")
                assert errors == []
                page.close()
        finally:
            browser.close()
