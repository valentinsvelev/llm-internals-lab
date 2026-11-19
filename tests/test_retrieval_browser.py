"""Retrieval v4: numeric sparse matrices, a computed trace and in-pipeline PCA."""
from __future__ import annotations

import pytest
from llm_lab.explorer import render_html


def test_retrieval_visuals():
    playwright = pytest.importorskip('playwright.sync_api')
    from conftest import launch_chromium
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
                page.locator('.pages a[data-page="retrieval"]').click()
                assert page.locator('#retr-flow .retr-stage').count() == 5
                assert page.locator('#retr-flow .retr-stage-projection #retr-map').count() == 1
                assert page.locator('.retr-vector-entry.zero').count() > 0
                assert page.locator('.retr-vector-entry:not(.zero)').count() > 0
                assert page.locator('#retr-eqn-idf math').count() == 1
                assert 'nonzero' in page.locator('#retr-weight-trace').text_content()
                zero = page.locator('.retr-vector-entry.zero').first
                zero.click()
                assert 'normalized = 0.000' in page.locator('#retr-weight-trace').text_content()
                nonzero = page.locator('.retr-vector-entry:not(.zero)').first
                nonzero.click()
                assert page.locator('#retr-weight-trace .retr-trace-result').text_content() != '0.000'
                assert page.locator('.retr-vector-entry.active').count() == 1
                assert page.locator('#retr-map .retr-map-dot').count() == 10
                assert page.locator('#retr-flow-top .retr-top-row').count() == 2
                assert page.locator('#retr-map').evaluate('el => !!el.closest("#retr-flow")')
                assert 'General admission is free' in page.locator('#retr-answer').text_content()
                assert 'visits:1' in page.locator('#retr-context').text_content()
                first = page.locator('#retr-map .retr-map-dot').first
                first.focus()
                first.press('Enter')
                assert 'Selected passage' in page.locator('#retr-inspect-title').text_content()
                page.locator('#retr-preset').select_option('tram')
                page.locator('#retr-k').fill('1')
                assert page.locator('#retr-flow-top .retr-top-row').count() == 1
                page.locator('#retr-k').fill('2')
                assert 'Take tram line 4' in page.locator('#retr-answer').text_content()
                page.locator('#retr-method').select_option('cosine')
                assert 'TF-IDF cosine' in page.locator('#retr-equation-note').text_content()
                assert page.locator('.retr-score-equation').text_content().strip()
                page.locator('#retr-chunk-switch').click()
                assert page.locator('#retr-map .retr-map-dot').count() == 5
                page.locator('#retr-question').fill('purple submarines')
                assert page.locator('#retr-flow-top .retr-top-row').count() == 0
                assert page.locator('#retr-map .retr-map-query').count() == 1
                assert not errors, errors
                page.close()
        finally:
            browser.close()


def test_distinct_rankers_and_reproducible_pca():
    playwright = pytest.importorskip('playwright.sync_api')
    from conftest import launch_chromium
    with playwright.sync_playwright() as p:
        browser = launch_chromium(p)
        if browser is None:
            pytest.skip('Install Chromium for the Playwright tests')
        try:
            page = browser.new_page()
            page.set_content(render_html(), wait_until='load')
            result = page.evaluate('''() => {
              const R=window.RLAB, q='free admission Thursday',
                a=R.retrieve(q,'parts',2,'bm25','free'),
                b=R.retrieve(q,'parts',2,'cosine','free'),
                c=R.projection('parts',q), d=R.projection('parts',q);
              return {bm25:a.ranked.ranks[0].id, cosine:b.ranked.ranks[0].id,
                bm25Check:Math.abs(a.ranked.ranks[0].score-a.ranked.ranks[0].breakdown.reduce((s,x)=>s+x.score,0))<1e-9,
                cosineCheck:Math.abs(b.ranked.ranks[0].score-b.ranked.ranks[0].breakdown.reduce((s,x)=>s+x.score,0))<1e-9,
                cosineBound:b.ranked.ranks.every(x=>x.score>=-1e-9&&x.score<=1+1e-9),
                points:c.points.length,finite:c.points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))&&Number.isFinite(c.query.x),
                stable:JSON.stringify(c)===JSON.stringify(d), different:Math.abs(a.ranked.ranks[0].score-b.ranked.ranks[0].score)>.01,
                trace:R.tfidfTrace(q,'parts','admission','visits:1')};
            }''')
            trace = result.pop('trace')
            assert trace['known'] and trace['tf'] == 1 and trace['df'] > 0
            assert trace['zero'] > 0 and trace['nnz'] + trace['zero'] == trace['V']
            assert abs(trace['raw'] - trace['tf'] * trace['idf']) < 1e-12
            assert abs(trace['weight'] - trace['raw'] / trace['norm']) < 1e-12
            assert result == {'bm25': 'visits:1', 'cosine': 'visits:1', 'bm25Check': True,
                              'cosineCheck': True, 'cosineBound': True, 'points': 10,
                              'finite': True, 'stable': True, 'different': True}
            page.close()
        finally:
            browser.close()
