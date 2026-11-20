"""The Retrieval page's numbers: llm_lab.reference.retrieval against scikit-learn and an independent BM25,
and the page's own arithmetic (web/src/retrieval-content.js, run in node) against llm_lab.reference.retrieval."""
import json
import math
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from llm_lab.reference import retrieval as R

ROOT = Path(__file__).resolve().parents[1]
node = shutil.which('node')
pytestmark = pytest.mark.skipif(node is None, reason='node not installed')


@pytest.fixture(scope='module')
def corpus():
    return R.page_corpus(ROOT)


def _queries(questions, vocab):
    rng = random.Random(0)
    extra = ['purple submarines', 'Café vegan dishes', 'tickets for the night-ocean exhibit', '10:00 Monday', 'shuttle shuttle shuttle', '']
    extra += [' '.join(rng.sample(vocab, rng.randint(1, 5))) for _ in range(40)]
    return questions + extra


def test_tfidf_weights_match_scikit_learn(corpus):
    sk = pytest.importorskip('sklearn.feature_extraction.text')
    docs, _ = corpus
    index = R.build_index(docs)
    vec = sk.TfidfVectorizer(tokenizer=R.tokenize, lowercase=False, token_pattern=None, smooth_idf=True, norm='l2')
    X = vec.fit_transform(docs).toarray()
    assert sorted(vec.vocabulary_) == sorted(index.vocab)
    for d in range(index.n_docs):
        ours = R.tfidf_vector(index, d)
        for j, t in enumerate(index.vocab):
            assert ours[j] == pytest.approx(X[d, vec.vocabulary_[t]], abs=1e-12), (d, t)


def test_bm25_dot_product_is_the_bm25_score(corpus):
    """q · d with binary q equals BM25 summed over the query's distinct terms, written out independently."""
    docs, questions = corpus
    index = R.build_index(docs)
    lengths = [len(R.tokenize(d)) for d in docs]
    avg, N = sum(lengths) / len(docs), len(docs)
    df = Counter(t for d in docs for t in set(R.tokenize(d)))
    for q in _queries(questions, index.vocab):
        for d, text in enumerate(docs):
            tf = Counter(R.tokenize(text))
            expected = sum(math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * lengths[d] / avg))
                           for t in set(R.tokenize(q)) if tf[t])
            assert R.scores(index, q, 'bm25')[d] == pytest.approx(expected, abs=1e-12), (q, d)


def test_page_matches_the_reference(corpus):
    docs, questions = corpus
    index = R.build_index(docs)
    queries = _queries(questions, index.vocab)
    script = f"""
const R = require({json.dumps(str(ROOT / 'web/src/retrieval-content.js'))});
const qs = {json.dumps(queries)};
const out = {{ vocab: R.INDEX.vocab, tokens: R.INDEX.docs.map(d => d.terms), qtokens: qs.map(q => R.tokenize(q)),
  vectors: {{}}, scores: {{}}, qvec: qs.map(q => Array.from(R.queryVector(q).vector)) }};
for (const m of ['bm25', 'tfidf']) {{
  out.vectors[m] = R.INDEX.docs.map(d => Array.from(R.docVector(m, d)));
  out.scores[m] = qs.map(q => R.retrieve(q, m, 5).rows.map(r => r.score));
}}
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    assert r['vocab'] == index.vocab
    assert r['tokens'] == index.docs
    assert r['qtokens'] == [R.tokenize(q) for q in queries]
    assert r['qvec'] == [R.query_vector(index, q) for q in queries]
    for method, vec in (('bm25', R.bm25_vector), ('tfidf', R.tfidf_vector)):
        for d in range(index.n_docs):
            assert r['vectors'][method][d] == pytest.approx(vec(index, d), abs=1e-12)
        for q, s in zip(queries, r['scores'][method]):
            assert s == pytest.approx(R.scores(index, q, method), abs=1e-12), (method, q)


def test_examples_and_where_sparse_fails():
    """The first six examples rank their document first with both weightings. The added ones are worded
    unlike their document: two share no term with any document, one only a term every document has."""
    data = R.page_data(ROOT)
    index = R.build_index([d['text'] for d in data['docs']])
    keys = [d['key'] for d in data['docs']]
    by_id = {p['id']: p for p in data['presets']}
    for p in data['presets'][:6]:
        for method in ('bm25', 'tfidf'):
            s = R.scores(index, p['question'], method)
            assert keys[max(range(len(s)), key=s.__getitem__)] == p['source'], (p['id'], method, s)
    for pid in ('eat', 'parking'):
        assert not any(t in index.df for t in R.tokenize(by_id[pid]['question'])), pid
        assert max(R.scores(index, by_id[pid]['question'], 'bm25')) == 0, pid
    assert [t for t in R.tokenize(by_id['train']['question']) if t in index.df] == ['aquarium']
    assert index.df['aquarium'] == index.n_docs
    for method in ('bm25', 'tfidf'):
        s = R.scores(index, by_id['train']['question'], method)
        assert keys[max(range(len(s)), key=s.__getitem__)] != by_id['train']['source'], method
    assert by_id['parking']['source'] is None


def test_tokenizer():
    assert R.tokenize('The Tidal Café serves vegan dishes; visitors borrowed wheelchairs at 10:00.') == [
        'tidal', 'café', 'serve', 'vegan', 'dish', 'visitor', 'borrowed', 'wheelchair', '10', '00']
    assert [R.singular(w) for w in ('galleries', 'passes', 'boxes', 'bus', 'axis', 'has', 'runs')] == [
        'gallery', 'pass', 'box', 'bus', 'axis', 'has', 'run']


# ---------- dense: web/src/embeddings.json, written by web/scripts/make_embeddings.py ----------

def test_embeddings_file():
    """The dense vectors are of exactly the page's texts, 384 numbers of length 1 each."""
    emb, data = R.page_embeddings(ROOT), R.page_data(ROOT)
    assert emb['model'] == 'sentence-transformers/all-MiniLM-L6-v2'
    assert (emb['dim'], emb['layers'], emb['heads'], emb['vocab_size']) == (384, 6, 12, 30522)
    assert (emb['max_length'], emb['pooling'], emb['normalized']) == (256, 'mean', True)
    assert [(d['key'], d['text']) for d in emb['docs']] == [(d['key'], d['text']) for d in data['docs']], \
        'the documents changed: run web/scripts/make_embeddings.py'
    assert [(q['id'], q['text']) for q in emb['queries']] == [(p['id'], p['question']) for p in data['presets']], \
        'the examples changed: run web/scripts/make_embeddings.py'
    for x in emb['docs'] + emb['queries']:
        assert len(x['vector']) == 384 and abs(math.fsum(v * v for v in x['vector']) - 1) < 1e-4, x['text']
        assert len(x['ids']) == len(x['tokens']) and x['tokens'][0] == '[CLS]' and x['tokens'][-1] == '[SEP]'
        assert x['ids'][0] == 101 and x['ids'][-1] == 102 and all(0 <= i < emb['vocab_size'] for i in x['ids'])


def test_embeddings_are_minilm():
    """Real model output, not the placeholder (make_embeddings.py --placeholder): token ids and lengths."""
    emb = R.page_embeddings(ROOT)
    if emb.get('placeholder'):
        pytest.skip('placeholder embeddings: run web/scripts/make_embeddings.py for all-MiniLM-L6-v2')
    assert 22_000_000 < emb['params'] < 23_000_000
    for x in emb['docs'] + emb['queries']:
        assert len(x['ids']) == len(x['tokens']) and x['ids'][0] == 101 and x['ids'][-1] == 102 and x['norm'] > 0
    if emb.get('check_sentence_transformers') is not None:
        assert emb['check_sentence_transformers'] < 1e-4


def test_page_dense_matches_the_file():
    emb = R.page_embeddings(ROOT)
    ids = [q['id'] for q in emb['queries']]
    script = f"""
const R = require({json.dumps(str(ROOT / 'web/src/retrieval-content.js'))});
console.log(JSON.stringify({json.dumps(ids)}.map(id => {{ const r = R.denseRetrieve(id, 3);
  return {{scores: r.rows.map(x => x.score), ranks: r.ranks.map(x => x.key), picks: r.picks.map(x => x.key)}}; }})));
"""
    out = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    keys = [d['key'] for d in emb['docs']]
    for pid, r in zip(ids, out):
        ref = R.dense_scores(emb, pid)
        assert r['scores'] == pytest.approx(ref, abs=1e-12), pid
        assert r['ranks'] == [k for _, k in sorted(zip(ref, keys), key=lambda x: (-x[0], x[1]))], pid
        assert r['picks'] == r['ranks'][:3]


def test_learned_sparse_file():
    """The learned sparse weights are of exactly the page's texts: positive, over the 30,522 WordPiece tokens."""
    ls, data = R.page_learned(ROOT), R.page_data(ROOT)
    assert ls['model'] == 'opensearch-project/opensearch-neural-sparse-encoding-v2-distill'
    assert (ls['vocab_size'], ls['pooling'], ls['activation']) == (30522, 'max', 'log(1 + relu)')
    assert [(d['key'], d['text']) for d in ls['docs']] == [(d['key'], d['text']) for d in data['docs']], \
        'the documents changed: run web/scripts/make_embeddings.py'
    assert [(q['id'], q['text']) for q in ls['queries']] == [(p['id'], p['question']) for p in data['presets']], \
        'the examples changed: run web/scripts/make_embeddings.py'
    for x in ls['docs'] + ls['queries']:
        ids = [j for j, _, _ in x['terms']]
        assert ids == sorted(set(ids)) and all(0 <= j < 30522 for j in ids), x['text']
        assert x['terms'] and all(w > 0 for _, _, w in x['terms']), x['text']
        assert len(x['ids']) == len(x['tokens']) and x['tokens'][0] == '[CLS]' and x['tokens'][-1] == '[SEP]'
    if not ls.get('placeholder'):
        assert 60_000_000 < ls['params'] < 75_000_000
        if ls.get('check_sparse_encoder') is not None:
            assert ls['check_sparse_encoder'] < 1e-3


def test_page_learned_matches_the_file():
    ls = R.page_learned(ROOT)
    ids = [q['id'] for q in ls['queries']]
    script = f"""
const R = require({json.dumps(str(ROOT / 'web/src/retrieval-content.js'))});
console.log(JSON.stringify({json.dumps(ids)}.map(id => {{ const r = R.learnedRetrieve(id, 2);
  return {{scores: r.rows.map(x => x.score), picks: r.picks.map(x => x.key), shared: r.rows.map(x => x.products.length)}}; }})));
"""
    out = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    keys = [d['key'] for d in ls['docs']]
    for pid, r in zip(ids, out):
        ref = R.learned_scores(ls, pid)
        assert r['scores'] == pytest.approx(ref, abs=1e-9), pid
        ranked = [k for s, k in sorted(zip(ref, keys), key=lambda x: (-x[0], x[1])) if s > 0]
        assert r['picks'] == ranked[:2], pid
        q = {j for j, _, _ in next(x['terms'] for x in ls['queries'] if x['id'] == pid)}
        assert r['shared'] == [len(q & {j for j, _, _ in d['terms']}) for d in ls['docs']], pid


def test_learned_sparse_match_the_model():
    """Recompute every weight with transformers (needs the model: skipped offline or for placeholders)."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_embeddings', ROOT / 'web/scripts/make_embeddings.py')
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    ls = R.page_learned(ROOT)
    if ls.get('placeholder'):
        pytest.skip('placeholder weights: run web/scripts/make_embeddings.py for the learned sparse model')
    try:
        tokenizer, model = mk.load_learned(ls['model'], ls['revision'])
    except Exception as e:  # no network or no cached copy
        pytest.skip(f'cannot load {ls["model"]}: {type(e).__name__}')
    for x in ls['docs'] + ls['queries']:
        got = mk.learned(x['text'], tokenizer, model)
        assert got['ids'] == x['ids'], x['text']
        mine, ref = {j: w for j, _, w in got['terms']}, {j: w for j, _, w in x['terms']}
        assert set(mine) == set(ref), x['text']
        assert all(abs(mine[j] - ref[j]) < 2e-3 for j in ref), x['text']


def test_page_hybrid_matches_the_reference():
    """Reciprocal rank fusion and the weighted sum, in the page and in Python, for every example."""
    data, emb = R.page_data(ROOT), R.page_embeddings(ROOT)
    index = R.build_index([d['text'] for d in data['docs']])
    keys = [d['key'] for d in data['docs']]
    script = f"""
const R = require({json.dumps(str(ROOT / 'web/src/retrieval-content.js'))});
const out = [];
for (const p of R.PRESETS) for (const f of ['rrf', 'convex']) for (const m of ['bm25', 'tfidf']) {{
  const r = R.hybridRetrieve(p.question, m, p.id, 2, f);
  out.push({{id: p.id, f, m, scores: r.rows.map(x => x.score), ranks: r.ranks.map(x => x.key), sparse: r.lists.sparse}});
}}
console.log(JSON.stringify(out));
"""
    out = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    by_id = {p['id']: p for p in data['presets']}
    assert len(out) == len(data['presets']) * 4
    for r in out:
        sparse, dense = R.scores(index, by_id[r['id']]['question'], r['m']), R.dense_scores(emb, r['id'])
        ref = R.hybrid_scores(sparse, dense, keys, r['f'])
        assert r['scores'] == pytest.approx(ref, abs=1e-12), (r['id'], r['f'], r['m'])
        assert r['ranks'] == [k for _, k in sorted(zip(ref, keys), key=lambda x: (-x[0], x[1]))]
        assert r['sparse'] == [k for s, k in sorted(zip(sparse, keys), key=lambda x: (-x[0], x[1])) if s > 1e-12]
    # no shared term: the sparse list is empty, so RRF is the dense list's 1/(60 + rank) alone
    eat = next(r for r in out if r['id'] == 'eat' and r['f'] == 'rrf' and r['m'] == 'bm25')
    assert eat['sparse'] == [] and sorted(eat['scores'], reverse=True) == pytest.approx([1 / (60 + i) for i in range(1, 6)])


def test_embeddings_match_the_model():
    """Recompute every vector with transformers (needs the model: skipped offline)."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_embeddings', ROOT / 'web/scripts/make_embeddings.py')
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    emb = R.page_embeddings(ROOT)
    if emb.get('placeholder'):
        pytest.skip('placeholder embeddings: run web/scripts/make_embeddings.py for all-MiniLM-L6-v2')
    try:
        tokenizer, model = mk.load(emb['model'], emb['revision'])
    except Exception as e:  # no network or no cached copy
        pytest.skip(f'cannot load {emb["model"]}: {type(e).__name__}')
    for x in emb['docs'] + emb['queries']:
        got = mk.embed(x['text'], tokenizer, model)
        assert got['ids'] == x['ids'] and got['tokens'] == x['tokens'], x['text']
        assert got['vector'] == pytest.approx(x['vector'], abs=1e-4), x['text']
