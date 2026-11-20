"""The Reasoning page in a real browser: one diagram of thinking. Each pass writes one token into the
context; the newest token is the next pass's input; attention reads one more position each pass and the
KV cache grows by one token; the check after every pass stops the thinking at </think> and the reply at <eos>;
every part opens its details."""
from __future__ import annotations

import shutil

import pytest

from llm_lab.explorer import render_html
from llm_lab.reference import reasoning


def _num(page, sel):
    return int(page.locator(sel).first.text_content().replace(',', ''))


def _go(page, k):
    page.locator('#think-slider').fill(str(k))
    page.locator('#think-slider').dispatch_event('input')


def test_reasoning_page():
    playwright = pytest.importorskip('playwright.sync_api')
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    from conftest import launch_chromium
    total_b = f'{sum(reasoning.parameters().values()) / 1e9:.1f}B'
    with playwright.sync_playwright() as p:
        browser = launch_chromium(p)
        if browser is None:
            pytest.skip('Install Chromium for the Playwright tests')
        try:
            for width in (1280, 420):
                page = browser.new_page(viewport={'width': width, 'height': 900})
                errors = []
                page.on('pageerror', lambda err, errors=errors: errors.append(str(err)))
                page.set_content(render_html(), wait_until='load')
                page.locator('.pages a[data-page="reasoning"]').click()
                lab = page.locator('.page[data-page="reasoning"]')
                assert lab.is_visible() and page.locator('.pages a[data-page="reasoning"] .soon').count() == 0
                # one diagram, nothing else: no tabs, no training, no results
                assert lab.locator('.tabs, .tab').count() == 0 and lab.locator('figure').count() == 1
                text = lab.text_content()
                for word in ('GRPO', 'reward', 'AIME', 'MATH-500', 'pass@1'):
                    assert word not in text, word
                assert page.locator('#reas-stats dt').all_text_contents() == ['Model', 'Layers', 'Active parameters', 'KV cache']
                assert total_b in page.locator('#reas-stats').text_content() and '68.6 KiB' in page.locator('#reas-stats').text_content()
                info = page.evaluate('({n: REASONINGAPP.N, end: REASONINGAPP.END, wait: REASONINGAPP.WAIT, prompt: REASONINGAPP.PROMPT_N, k: REASONINGAPP.state.k})')
                assert 1 < info['wait'] < info['end'] < info['n'] and info['k'] == info['wait']
                assert page.locator('#think-pass .think-out .think-tok').text_content() == 'Wait'
                stop = page.locator('#think-stop .think-stopbox')
                assert 'yes' not in stop.get_attribute('class') and '“Wait” instead' in stop.text_content()
                assert f'{info["wait"]} of 32,768 tokens' in stop.text_content()
                # the pass: input → embedding → 61 layers (attention, 8 of 256 experts) → LM head → sample → output
                assert page.locator('#think-pass [data-part]').count() == 7
                assert page.locator('#think-pass .think-group').count() == 8 and page.locator('#think-pass .think-group i').count() == 256
                assert page.locator('#think-pass .think-group i.on').count() == 8
                assert page.locator('#think-pass .think-group:not(.off)').count() == 4
                assert page.locator('#think-pass .think-group.off i.on').count() == 0          # experts only from the allowed groups
                assert page.locator('#think-pass .think-bars i.pick').count() == 1
                # one step: the output becomes the next input; attention reads one more; the cache grows by one
                k, reads = info['k'], _num(page, '#think-pass .think-attn > b')
                assert reads == info['prompt'] + k - 1
                out = page.locator('#think-pass .think-out .think-tok').text_content()
                experts = page.locator('#think-pass .think-group i').evaluate_all('els => els.map(e => e.className)')
                page.locator('.think-controls [data-step="1"]').click()
                assert page.locator('#think-pass .think-in .think-tok').text_content() == out
                assert _num(page, '#think-pass .think-attn > b') == reads + 1
                assert f'{reads + 1}' in page.locator('#think-cache .think-cachefoot').text_content()
                assert page.locator('#think-ctx .think-tok.out').text_content() == page.locator('#think-pass .think-out .think-tok').text_content()
                assert page.locator('#think-pass .think-group i').evaluate_all('els => els.map(e => e.className)') != experts
                # the first pass is prefill: the whole prompt goes in at once
                page.locator('[data-jump="start"]').click()
                assert 'prefill' in page.locator('#think-status').text_content()
                assert page.locator('#think-pass .think-in .think-all').count() == 1 and page.locator('#think-ctx .think-prompt.in').count() == 1
                assert _num(page, '#think-pass .think-attn > b') == info['prompt']
                # </think> closes the thought; the answer follows and reads it
                page.locator('[data-jump="end"]').click()
                assert page.locator('#think-pass .think-out .think-tok').text_content() == '</think>'
                assert page.locator('#think-ctx .think-tok.ans').count() == 0
                assert 'yes' in stop.get_attribute('class') and '</think> drawn' in stop.text_content()
                _go(page, info['end'] + 3)
                assert 'yes' not in stop.get_attribute('class') and 'answering' in page.locator('#think-status').text_content()
                page.locator('[data-jump="done"]').click()
                assert 'ending the reply' in page.locator('#think-status').text_content()
                assert page.locator('#think-pass .think-out .think-tok').text_content() == '<eos>'
                assert 'yes' in stop.get_attribute('class') and '<eos> drawn' in stop.text_content()
                assert '(√(4a − 3) − 1)/2' in page.locator('#think-ctx').text_content().replace('( ', '(')
                _go(page, info['wait'])
                assert page.locator('#think-ctx .think-tok.wait').count() == 1
                # play advances the passes
                page.locator('#think-play').click()
                page.wait_for_timeout(700)
                page.locator('#think-play').click()
                assert page.evaluate('REASONINGAPP.state.k') > info['wait']
                # every part opens its details; the cache's cost table is computed from the architecture
                for part in ('context', 'input', 'embed', 'attention', 'experts', 'head', 'sample', 'output', 'stop', 'cache', 'context'):
                    page.locator(f'[data-part="{part}"]').first.click()
                    assert page.locator(f'[data-part="{part}"].open').count() == 1
                    assert page.locator('#think-detail .retr-dhead b').text_content()
                page.locator('[data-part="cache"]').click()
                rows = page.locator('#think-detail .think-prow:not(.head)').all_text_contents()
                assert len(rows) == 4 and '4.29 GiB' in rows[3]                       # 65,536 tokens × 70,272 B
                page.locator('[data-part="stop"]').click()
                assert page.locator('#think-detail .think-srow:not(.head)').count() == 4
                assert '32,768' in page.locator('#think-detail').text_content() and '65,536' in page.locator('#think-detail').text_content()
                page.locator('[data-part="attention"]').press('Enter')
                assert page.locator('[data-part="attention"].open').count() == 1
                assert page.locator('#think-detail math').count() >= 1
                # no sideways scrolling
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                assert errors == [], errors
                page.close()
        finally:
            browser.close()
