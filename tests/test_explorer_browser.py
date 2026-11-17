"""The explorer in a real browser: every tab, step, pop-up, decoding method, example input and page.

Needs Playwright and a Chromium: pip install -e ".[dev]" && python -m playwright install chromium
"""
from __future__ import annotations

import json

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
                # The guided tour: one idea per step on the GPT-2 diagram, ending in the six mechanisms.
                page.locator('.tab[data-mech="gqa"]').click()
                page.locator('#tour-start').click()
                assert page.locator('.tab.active').get_attribute('data-mech') == 'mha'
                assert page.locator('#arch-card').is_hidden()
                for i in range(8):
                    assert f'step {i + 1} of 8' in page.locator('#tour').inner_text()
                    assert page.locator('#arch.touring').count() == 1 and page.locator('#arch .on').count() > 0
                    if i < 7:
                        page.locator('#tour .next').click()
                assert page.locator('#tour .tour-ans button').count() == 6
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
