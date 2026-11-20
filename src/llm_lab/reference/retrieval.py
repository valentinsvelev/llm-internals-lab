"""Retrieval as a dot product (the Retrieval page), in plain Python.

Sparse:

    query     q ∈ {0,1}^n   a 1 for every vocabulary term in the query
    document  d ∈ R^n       BM25 or TF-IDF weights of its terms, 0 elsewhere
    score     s(q, d) = q · d = Σ_{t ∈ q} d_t

BM25 uses k1 = 1.2, b = 0.75 and Lucene's idf, ln(1 + (N − df + 0.5) / (df + 0.5)); with these
weights q · d is the usual BM25 score. TF-IDF uses scikit-learn's smoothed idf, 1 + ln((N + 1) /
(df + 1)), and scales each document vector to length 1, as TfidfVectorizer does; tests/test_retrieval.py
checks that, and that the page (web/src/retrieval-content.js) computes the same numbers.

Dense: the page's vectors come from all-MiniLM-L6-v2 and are computed in advance by
web/scripts/make_embeddings.py (web/src/embeddings.json); the score is again q · d, the cosine
similarity of two vectors of length 1. `page_embeddings` and `dense_scores` read and use that file.

Learned sparse: SPLADE-style weights over the 30,522 WordPiece tokens from
opensearch-neural-sparse-encoding-v2-distill (web/src/learned-sparse.json, same script); the score is
q · d over the tokens where both are nonzero. `page_learned` and `learned_scores` read and use it.

Hybrid: the sparse and the dense ranking fused, by reciprocal rank (Cormack, Clarke & Büttcher 2009:
Σ 1 / (k + rank), k = 60) or by a weighted sum of min-max normalized scores. The sparse list holds only
the documents with a score above 0. `hybrid_scores` does both.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

K1, B = 1.2, 0.75
STOP = frozenset('a an and are at be but by can do does every for from how i in into is it me my of on or the their there '
                 'this to what when where with you'.split())
_WORD = re.compile(r'[^\W_]+')          # letters and digits, as /[\p{L}\p{N}]+/u


def singular(w: str) -> str:
    """Simplified plurals: -ies → -y, -sses/-shes/-ches/-xes/-zes → drop -es, other final -s → drop it."""
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 4 and re.search(r'(ss|sh|ch|x|z)es$', w):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not re.search(r'(ss|us|is)$', w):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase, split at everything that is not a letter or digit, drop stop words, singular."""
    return [singular(w) for w in _WORD.findall(text.lower()) if w not in STOP]


@dataclass
class Index:
    docs: list[list[str]]                       # each document's terms, in order
    vocab: list[str] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for terms in self.docs:                 # ids in order of first appearance, d1 to dN
            for t in terms:
                if t not in self.df:
                    self.vocab.append(t)
                    self.df[t] = 0
        for terms in self.docs:
            for t in set(terms):
                self.df[t] += 1

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_length(self) -> float:
        return sum(map(len, self.docs)) / self.n_docs


def build_index(texts: list[str]) -> Index:
    return Index([tokenize(t) for t in texts])


def bm25_vector(index: Index, d: int) -> list[float]:
    terms, N = index.docs[d], index.n_docs
    norm = 1 - B + B * len(terms) / index.avg_length
    out = []
    for t in index.vocab:
        tf = terms.count(t)
        idf = math.log(1 + (N - index.df[t] + 0.5) / (index.df[t] + 0.5))
        out.append(idf * tf * (K1 + 1) / (tf + K1 * norm) if tf else 0.0)
    return out


def tfidf_vector(index: Index, d: int) -> list[float]:
    terms, N = index.docs[d], index.n_docs
    raw = [terms.count(t) * (1 + math.log((N + 1) / (index.df[t] + 1))) for t in index.vocab]
    length = math.sqrt(sum(x * x for x in raw))
    return [x / length for x in raw]


def query_vector(index: Index, question: str) -> list[float]:
    terms = set(tokenize(question))
    return [1.0 if t in terms else 0.0 for t in index.vocab]


def scores(index: Index, question: str, method: str = 'bm25') -> list[float]:
    """q · d for every document."""
    q = query_vector(index, question)
    vec = bm25_vector if method == 'bm25' else tfidf_vector
    return [sum(a * b for a, b in zip(q, vec(index, d))) for d in range(index.n_docs)]


def _root(root: Path | None) -> Path:
    return root or Path(__file__).resolve().parents[3]


def page_data(root: Path | None = None) -> dict:
    """The page's documents (key, id, text) and examples (id, question, key of the answering document or
    None), read from web/src/retrieval-content.js with node."""
    script = (f"const R=require({json.dumps(str(_root(root) / 'web/src/retrieval-content.js'))});"
              "const key=id=>{const d=R.DOCS.find(d=>d.id===id);return d?d.key:null};"
              "console.log(JSON.stringify({docs:R.DOCS.map(d=>({key:d.key,id:d.id,text:d.text})),"
              "presets:R.PRESETS.map(p=>({id:p.id,question:p.question,source:key(p.source)}))}))")
    return json.loads(subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True).stdout)


def page_corpus(root: Path | None = None) -> tuple[list[str], list[str]]:
    """The page's five document texts and example questions."""
    data = page_data(root)
    return [d['text'] for d in data['docs']], [p['question'] for p in data['presets']]


def page_embeddings(root: Path | None = None) -> dict:
    """web/src/embeddings.json: the dense vectors of the documents and examples (make_embeddings.py)."""
    return json.loads((_root(root) / 'web/src/embeddings.json').read_text())


def page_learned(root: Path | None = None) -> dict:
    """web/src/learned-sparse.json: [vocabulary id, token, weight] of every nonzero entry (make_embeddings.py)."""
    return json.loads((_root(root) / 'web/src/learned-sparse.json').read_text())


def learned_scores(data: dict, preset_id: str) -> list[float]:
    """q · d for every document, over the vocabulary tokens where both are nonzero."""
    q = {j: w for j, _, w in next(x['terms'] for x in data['queries'] if x['id'] == preset_id)}
    return [math.fsum(w * q[j] for j, _, w in d['terms'] if j in q) for d in data['docs']]


RRF_K, ALPHA = 60, 0.5


def _ranking(scores: list[float], keys: list[str], only_positive: bool) -> list[str]:
    order = sorted(range(len(keys)), key=lambda i: (-scores[i], keys[i]))
    return [keys[i] for i in order if scores[i] > 1e-12 or not only_positive]


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def hybrid_scores(sparse: list[float], dense: list[float], keys: list[str], fusion: str = 'rrf',
                  k: int = RRF_K, alpha: float = ALPHA) -> list[float]:
    """The fused score of every document."""
    if fusion == 'rrf':
        lists = [_ranking(sparse, keys, True), _ranking(dense, keys, False)]
        return [math.fsum(1 / (k + lst.index(key) + 1) for lst in lists if key in lst) for key in keys]
    return [alpha * s + (1 - alpha) * d for s, d in zip(_minmax(sparse), _minmax(dense))]


def dense_scores(emb: dict, preset_id: str) -> list[float]:
    """q · d for every document, q being the example's embedding."""
    q = next(x['vector'] for x in emb['queries'] if x['id'] == preset_id)
    return [sum(a * b for a, b in zip(q, d['vector'])) for d in emb['docs']]
