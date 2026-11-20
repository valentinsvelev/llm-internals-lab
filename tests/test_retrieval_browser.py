"""The Retrieval page in a real browser: input → unique words → vocabulary → weights → sparse vectors →
q · d → output, with the three boxes on the arrows opening their details."""
from __future__ import annotations

import pytest

from llm_lab.explorer import render_html
from llm_lab.reference import retrieval as RR


def test_retrieval_page():
    playwright = pytest.importorskip('playwright.sync_api')
    from conftest import launch_chromium
    docs, questions = RR.page_corpus()
    index = RR.build_index(docs)
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
                assert page.locator('#retr-flow .retr-stage').count() == 4
                # the header, as on the Architectures page: tabs, title, lead and key facts
                assert page.locator('#retr-tabs .tab').all_inner_texts()[0].startswith('Sparse')
                assert page.locator('#retr-tabs .tab.active').get_attribute('data-mode') == 'sparse'
                assert page.locator('#retr-title').text_content() == 'Sparse retrieval'
                assert page.locator('#retr-lead').text_content()
                assert page.locator('#retr-stats dt').all_text_contents() == ['Model', 'Vocabulary', 'Dimensions', 'Index']
                # a clean figure: no stage numbers, notes or "show all"; both dropdowns use the same caret
                assert page.locator('.retr-stage-index, #retr-show-all, #retr-top-note, #retr-matrix-note, #retr-vocab-n').count() == 0
                assert page.locator('.retr-qpick .retr-caret').count() == page.locator('.retr-select .retr-caret').count() == 1
                assert page.locator('#retr-link-weights-info').is_hidden()
                assert page.locator('#retr-flow .retr-link').count() == 3
                # vocabulary and vectors show the same rows: 01–16, the query's terms, the last term, with ⋮ between
                q = RR.query_vector(index, questions[0])
                rows = sorted(set(range(16)) | {i for i, x in enumerate(q) if x} | {len(index.vocab) - 1})
                assert page.locator('#retr-vocab .retr-vrow .retr-vid').all_text_contents() == [f'{i + 1:02d}' for i in rows]
                assert page.locator('#retr-matrix .retr-mid').all_text_contents() == [f'{i + 1:02d}' for i in rows]
                assert page.locator('#retr-vocab .retr-vgap').count() >= 1
                boxes = [page.locator(sel).last.bounding_box() for sel in ('#retr-vocab .retr-vrow', '#retr-matrix .retr-mid')]
                mid = [b['y'] + b['height'] / 2 for b in boxes]
                if width >= 1100:
                    assert abs(mid[0] - mid[1]) < 1.5, mid        # the rows line up across the Weighting box
                assert page.locator('.retr-link-more, .retr-rank-no').count() == 0
                # the query box is a single dropdown of the example questions, showing the whole question
                assert page.locator('.retr-qbox select, .retr-qbox input, .retr-qbox textarea').count() == 1
                assert page.locator('#retr-preset option').all_text_contents() == questions
                assert page.locator('#retr-qtext').text_content() == questions[0]
                # the query vector: a 1 exactly in the query's rows; q and d_i in math italic
                assert page.locator('.retr-mid.q').count() == int(sum(q)) == page.locator('.retr-cell.q.one').count()
                assert page.locator('.retr-cell.q.zero').count() > 0
                assert page.locator('#retr-matrix .retr-mhead var').all_text_contents() == ['𝑞', '𝑑1', '𝑑2', '𝑑3', '𝑑4', '𝑑5']
                assert page.locator('#retr-docs .retr-doc-no var').first.text_content() == '𝑑1'
                # the output: scores sorted, the same numbers as the reference; nothing picked until k is chosen
                ref = RR.scores(index, questions[0], 'bm25')
                shown = [float(x) for x in page.locator('#retr-ranks .retr-rank-score').all_text_contents()]
                assert shown == sorted([round(x, 2) for x in ref], reverse=True)
                assert page.locator('#retr-k option:checked').text_content() == 'select k'
                assert page.locator('#retr-ranks .retr-rank-row.hit').count() == 0
                assert '(select k in the Output)' in page.locator('#retr-context').text_content()
                page.locator('#retr-k').select_option('4')
                assert page.locator('#retr-ranks .retr-rank-row.hit').count() == 4
                page.locator('#retr-k').select_option('2')
                assert page.locator('#retr-ranks .retr-rank-row.hit').count() == 2
                # the three boxes on the arrows
                assert page.locator('.retr-link[data-link="score"].open').count() == 1
                assert page.locator('#retr-detail .retr-sum').count() == 5
                assert page.locator('#retr-detail math').count() >= 1
                page.locator('.retr-link[data-link="words"] .retr-link-main').click()
                assert page.locator('.retr-link[data-link="words"] .retr-link-main').get_attribute('aria-expanded') == 'true'
                assert 'Lowercase' in page.locator('#retr-detail').text_content()
                assert page.locator('#retr-detail .retr-w.stop').count() > 0
                page.locator('#retr-docs [data-doc="2"]').click()
                assert page.locator('#retr-detail .retr-dhead b').text_content().startswith('𝑑3 · ')
                page.locator('.retr-link[data-link="weights"] .retr-link-main').click()
                assert page.locator('#retr-detail math').count() == 2
                cell = page.locator('.retr-cell:not(.q):not(.zero)').first
                term = cell.get_attribute('data-term')
                cell.click()
                assert page.locator('.retr-cell.active').count() == 1
                assert term in page.locator('#retr-detail .retr-trace').text_content()
                # TF-IDF: the box keeps its name, and the scores follow the reference
                assert page.locator('#retr-link-weights-title').text_content() == 'Weighting'
                page.locator('[data-method="tfidf"]').click()
                assert page.locator('#retr-link-weights-title').text_content() == 'Weighting'
                ref = RR.scores(index, questions[0], 'tfidf')
                shown = [float(x) for x in page.locator('#retr-ranks .retr-rank-score').all_text_contents()]
                assert shown == sorted([round(x, 2) for x in ref], reverse=True)
                assert 'General admission is free' in page.locator('#retr-answer').text_content()
                # another example: the box, the vectors and the ranking follow
                page.locator('#retr-preset').select_option('wheelchair')
                assert page.locator('#retr-qtext').text_content() == questions[2]
                assert page.locator('.retr-cell.q.one').count() == int(sum(RR.query_vector(index, questions[2])))
                # a query whose terms lie past 16 still gets their rows after the ⋮
                page.locator('#retr-preset').select_option('vegan')
                ids = [i for i, x in enumerate(RR.query_vector(index, questions[4])) if x]
                assert ids and min(ids) >= 16
                assert {f'{i + 1:02d}' for i in ids} <= set(page.locator('#retr-vocab .retr-vrow.q .retr-vid').all_text_contents())
                assert page.locator('.retr-cell.q.one').count() == len(ids)
                page.locator('#retr-preset').select_option('wheelchair')
                assert page.locator('#retr-ranks .retr-rank-row.hit .retr-rank-id').first.text_content() == '𝑑3'
                page.locator('.retr-link[data-link="words"] .retr-link-main').click()
                assert 'borrow' in page.locator('#retr-detail code.out').first.text_content()
                assert questions[2] in page.locator('#retr-context').text_content()
                # dense mode: same input and output, tokens and 384-number vectors in between
                emb = RR.page_embeddings()
                page.locator('#retr-preset').select_option('eat')
                page.locator('#retr-tabs [data-mode="dense"]').click()
                assert page.locator('#retr-title').text_content() == 'Dense retrieval'
                assert page.locator('#retr-tabs .tab.active').get_attribute('data-mode') == 'dense'
                assert 'all-MiniLM-L6-v2' in page.locator('#retr-stats').text_content()
                assert [page.locator(f'#retr-h-{x}').text_content() for x in ('vocab', 'vectors')] == ['Tokens', 'Dense vectors']
                assert page.locator('#retr-link-weights-title').text_content() == 'Encoder'
                assert page.locator('.retr-seg2.retr-method').is_hidden()
                assert page.locator('#retr-matrix .retr-mid').all_text_contents() == [str(i) for i in range(1, 17)] + ['384']
                assert page.locator('#retr-matrix .retr-cell.dense').count() == 17 * 6
                q = next(x for x in emb['queries'] if x['id'] == 'eat')
                assert page.locator('#retr-vocab .retr-vterm').all_text_contents() == q['tokens']
                assert page.locator('#retr-vocab .retr-vid').all_text_contents() == [str(i) for i in q['ids']]
                # every vector column, q included, has cells of one size
                sizes = {(round(b['width']), round(b['height'])) for b in page.locator('#retr-matrix .retr-cell.dense').evaluate_all(
                    'els => els.map(e => { const r = e.getBoundingClientRect(); return {width: r.width, height: r.height}; })')}
                assert len(sizes) == 1, sizes
                ref = RR.dense_scores(emb, 'eat')
                shown = [float(x) for x in page.locator('#retr-ranks .retr-rank-score').all_text_contents()]
                assert shown == sorted([round(x, 2) for x in ref], reverse=True)
                assert page.locator('#retr-ranks .retr-rank-row.hit').count() == 2
                page.locator('.retr-link[data-link="words"] .retr-link-main').click()
                assert 'WordPiece' in page.locator('#retr-detail').text_content()
                page.locator('.retr-link[data-link="weights"] .retr-link-main').click()
                assert page.locator('#retr-detail math').count() == 2 and 'mean pooling' in page.locator('#retr-detail').text_content().lower()
                page.locator('.retr-link[data-link="score"] .retr-link-main').click()
                assert page.locator('#retr-detail .retr-cmp .retr-cmp-col').count() == 3     # sparse, learned sparse, dense
                page.locator('#retr-matrix .retr-cell.dense[data-dim="3"][data-doc="4"]').click()
                assert 'Dimension 4 of' in page.locator('#retr-detail .retr-trace').first.text_content()
                # learned sparse mode: a weight per vocabulary token, expansions marked
                ls = RR.page_learned()
                page.locator('#retr-tabs [data-mode="learned"]').click()
                assert page.locator('#retr-title').text_content() == 'Learned sparse retrieval'
                assert page.locator('#retr-stats dt').all_text_contents() == ['Model', 'Vocabulary', 'Dimensions', 'Index']
                cut = page.locator('#retr-matrix .retr-mid.lterm').evaluate_all('els => els.filter(e => e.scrollWidth > e.clientWidth + 0.5).length')
                assert cut == 0          # every token name fits its column
                assert page.locator('#retr-h-vectors').text_content() == 'Learned sparse vectors'
                assert page.locator('#retr-link-weights-title').text_content() == 'Encoder + MLM head'
                lq = next(x for x in ls['queries'] if x['id'] == 'eat')
                assert page.locator('#retr-matrix .retr-mid.lterm').count() == min(16, len(lq['terms']))
                exp = {t for _, t, _ in lq['terms'] if t not in lq['tokens']}
                assert {e.split('\u00a0')[0] for e in page.locator('#retr-matrix .retr-mid.lterm.exp').evaluate_all('els => els.map(e => e.firstChild.textContent)')} == exp
                ref = RR.learned_scores(ls, 'eat')
                shown = [float(x) for x in page.locator('#retr-ranks .retr-rank-score').all_text_contents()]
                assert shown == sorted([round(x, 2) for x in ref], reverse=True)
                page.locator('.retr-link[data-link="weights"] .retr-link-main').click()
                assert page.locator('#retr-detail math').count() == 2 and 'expansion' in page.locator('#retr-detail').text_content()
                page.locator('#retr-matrix .retr-cell[data-lterm]:not(.zero)').first.click()
                assert page.locator('#retr-detail .retr-trace').count() == 1
                page.locator('.retr-link[data-link="score"] .retr-link-main').click()
                assert page.locator('#retr-detail .retr-sum').count() == 5 and page.locator('#retr-detail .retr-cmp-col').count() == 3
                # hybrid: the BM25 and the dense ranking, fused
                page.locator('#retr-tabs [data-mode="hybrid"]').click()
                assert page.locator('#retr-title').text_content() == 'Hybrid retrieval'
                assert [page.locator(f'#retr-h-{x}').text_content() for x in ('vocab', 'vectors')] == ['Two rankings', 'Fused scores']
                assert page.locator('.retr-seg2.retr-method').is_hidden() and page.locator('.retr-seg2.retr-fusion').is_visible()
                data = RR.page_data()
                keys = [d['key'] for d in data['docs']]
                sparse = RR.scores(index, 'Where can I get something to eat?', 'bm25')
                for fusion in ('rrf', 'convex'):
                    page.locator(f'[data-fusion="{fusion}"]').click()
                    ref = RR.hybrid_scores(sparse, RR.dense_scores(emb, 'eat'), keys, fusion)
                    digits = 4 if fusion == 'rrf' else 2
                    shown = [float(x) for x in page.locator('#retr-matrix .retr-cell.fused').all_text_contents()]
                    assert shown == [round(x, digits) for x in ref], fusion
                    assert [float(x) for x in page.locator('#retr-ranks .retr-rank-score').all_text_contents()] == sorted([round(x, digits) for x in ref], reverse=True)
                    assert page.locator('#retr-detail math').count() == 1          # the fusion formula is open
                page.locator('[data-fusion="rrf"]').click()
                assert page.locator('#retr-vocab .retr-rkrow').count() == 5
                assert page.locator('#retr-vocab .retr-rk.none').count() == 5      # no shared term: the BM25 list is empty
                assert page.locator('#retr-matrix .retr-cell:not(.fused)').all_text_contents()[::2] == ['—'] * 5
                page.locator('.retr-link[data-link="words"] .retr-link-main').click()
                assert page.locator('#retr-detail .retr-cmp-col').count() == 2
                page.locator('.retr-link[data-link="score"] .retr-link-main').click()
                assert page.locator('#retr-detail .retr-cmp-col').count() == 3 and page.locator('#retr-detail .retr-sum').count() == 5
                # back to sparse: the vocabulary, the weighting switch and the term rows return
                page.locator('#retr-tabs [data-mode="sparse"]').click()
                assert page.locator('#retr-title').text_content() == 'Sparse retrieval'
                assert page.locator('.retr-seg2.retr-method').is_visible() and page.locator('#retr-tokpick').is_hidden()
                assert page.locator('#retr-h-vocab').text_content() == 'Vocabulary'
                assert page.locator('#retr-ranks .retr-rank-row.hit').count() == 0      # no term shared: nothing retrieved
                assert not errors, errors
                page.close()
        finally:
            browser.close()
