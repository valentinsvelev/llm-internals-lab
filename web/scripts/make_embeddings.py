"""Run the Retrieval page's two learned retrievers on its documents and example questions, and write
their outputs: src/embeddings.json (dense mode) and src/learned-sparse.json (learned sparse mode).

    pip install torch transformers
    python3 web/scripts/make_embeddings.py                       # from the repository root (reads the texts with node)
    python3 web/scripts/make_embeddings.py --texts texts.json    # or from a file: {"docs": [{key, text}], "queries": [{id, text}]}
    python3 web/scripts/make_embeddings.py --only dense          # or: --only learned
    python3 web/scripts/make_embeddings.py --placeholder         # no model: illustrative random numbers, marked as placeholders

Dense: all-MiniLM-L6-v2 as sentence-transformers runs it: WordPiece tokens (at most 256), its 6 transformer
layers, the mean of the last layer's token vectors, scaled to length 1.

Learned sparse: opensearch-neural-sparse-encoding-v2-distill, SPLADE-style: the masked-language-model head
gives every token position a logit for each of the 30,522 vocabulary tokens; the weight of vocabulary token j
is max over positions of log(1 + ReLU(logit)), special tokens set to 0. Query and documents use the same model.

Each text is run on its own, so there is no padding. If sentence-transformers is installed, both outputs are
checked against it (SentenceTransformer.encode, SparseEncoder.encode). Needs huggingface.co the first time.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import unicodedata
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LEARNED_MODEL = 'opensearch-project/opensearch-neural-sparse-encoding-v2-distill'
MAX_LENGTH = 256          # sentence-transformers' max_seq_length for all-MiniLM-L6-v2
LEARNED_MAX_LENGTH = 512
DIGITS = 6
STOP = set('a an and are at be but by can do does every for from how i in into is it me my of on or the their there '
           'this to what when where with you'.split())   # only for the placeholder


def page_texts() -> dict:
    """The documents and example questions, read from src/retrieval-content.js."""
    script = (f"const R=require({json.dumps(str(HERE.parent / 'src/retrieval-content.js'))});"
              "console.log(JSON.stringify({docs:R.DOCS.map(d=>({key:d.key,text:d.text})),"
              "queries:R.PRESETS.map(p=>({id:p.id,text:p.question}))}))")
    return json.loads(subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True).stdout)


# ---------- dense ----------

def load(model_id: str = MODEL, revision: str | None = None):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).eval()
    return tokenizer, model


def embed(text: str, tokenizer, model) -> dict:
    """WordPiece tokens → transformer → mean of the token vectors → divided by its length."""
    import torch
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0].double()          # (tokens, 384)
    mean = h.mean(0)
    norm = mean.norm().item()
    ids = enc['input_ids'][0].tolist()
    return {'ids': ids, 'tokens': tokenizer.convert_ids_to_tokens(ids), 'norm': round(norm, DIGITS),
            'vector': [round(x, DIGITS) for x in (mean / norm).tolist()]}


def check_sentence_transformers(texts: list[str], vectors: list[list[float]], model_id: str, revision: str | None):
    """Largest difference to SentenceTransformer.encode, or None if the library is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    st = SentenceTransformer(model_id, revision=revision, device='cpu')
    ref = st.encode(texts, normalize_embeddings=True, batch_size=1)
    return max(abs(float(a) - float(b)) for v, r in zip(vectors, ref) for a, b in zip(v, r))


# ---------- learned sparse ----------

def load_learned(model_id: str = LEARNED_MODEL, revision: str | None = None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForMaskedLM.from_pretrained(model_id, revision=revision).eval()
    return tokenizer, model


def learned(text: str, tokenizer, model) -> dict:
    """MLM logits (positions × vocabulary) → max over positions of log(1 + ReLU) → the nonzero entries."""
    import torch
    enc = tokenizer(text, truncation=True, max_length=LEARNED_MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        logits = model(**enc).logits[0].double()                  # (tokens, 30522)
    w = torch.log1p(torch.relu(logits)).max(0).values
    w[tokenizer.all_special_ids] = 0
    ids = enc['input_ids'][0].tolist()
    nz = torch.nonzero(w).flatten().tolist()
    return {'ids': ids, 'tokens': tokenizer.convert_ids_to_tokens(ids),
            'terms': [[j, tokenizer.convert_ids_to_tokens(j), round(w[j].item(), 4)] for j in nz if round(w[j].item(), 4) > 0]}


def check_sparse_encoder(texts: list[str], terms: list[list], model_id: str, revision: str | None):
    """Largest difference to sentence-transformers' SparseEncoder.encode, or None if not available."""
    try:
        from sentence_transformers import SparseEncoder
    except ImportError:
        return None
    enc = SparseEncoder(model_id, revision=revision, device='cpu')
    ref = enc.encode(texts, batch_size=1, convert_to_tensor=True).to_dense()
    worst = 0.0
    for row, ts in zip(ref, terms):
        mine = {j: w for j, _, w in ts}
        for j in set(mine) | set(row.nonzero().flatten().tolist()):
            worst = max(worst, abs(mine.get(j, 0.0) - float(row[j])))
    return worst


# ---------- placeholders: illustrative only, not model output ----------

def plain_tokens(text: str) -> tuple[list[str], list[int]]:
    """A plain word split (lowercase, accents stripped, punctuation apart, as BERT does before WordPiece) and
    provisional ids: BERT's 101 and 102 for [CLS] and [SEP], the others from a hash of the token."""
    plain = ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')
    words = re.findall(r'\w+|[^\w\s]', plain)
    return ['[CLS]', *words, '[SEP]'], [101, *(1000 + zlib.crc32(w.encode()) % 29522 for w in words), 102]


def placeholder(text: str, rng: random.Random, dim: int = 384) -> dict:
    """Plain tokens and a random vector of length 1."""
    tokens, ids = plain_tokens(text)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return {'ids': ids, 'tokens': tokens, 'norm': None, 'vector': [round(x / n, DIGITS) for x in v]}


def placeholder_learned(text: str, rng: random.Random, pool: list[str], extra: int) -> dict:
    """Plain tokens; random weights for the text's own words (stop words and punctuation left out) and for a
    few words of the other texts, standing in for expansion terms."""
    tokens, ids = plain_tokens(text)
    own = {t: i for t, i in zip(tokens[1:-1], ids[1:-1]) if t.isalnum() and t not in STOP}
    terms = {i: [i, t, round(rng.uniform(0.4, 2.4), 4)] for t, i in own.items()}
    for t in rng.sample([t for t in pool if t not in own], extra):
        i = 1000 + zlib.crc32(t.encode()) % 29522
        terms.setdefault(i, [i, t, round(rng.uniform(0.1, 0.9), 4)])
    return {'ids': ids, 'tokens': tokens, 'terms': sorted(terms.values())}


def write(path: Path, out: dict, what: str):
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'wrote {what} to {path} ({path.stat().st_size // 1024} KiB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--texts', type=Path, help='JSON with docs [{key, text}] and queries [{id, text}]; default: read the page source')
    ap.add_argument('--out', type=Path, default=HERE.parent / 'src/embeddings.json', help='dense output')
    ap.add_argument('--learned-out', type=Path, default=HERE.parent / 'src/learned-sparse.json', help='learned sparse output')
    ap.add_argument('--only', choices=('dense', 'learned'), help='run only one of the two')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--learned-model', default=LEARNED_MODEL)
    ap.add_argument('--revision', default=None, help='a commit of the dense model repository (default: the latest)')
    ap.add_argument('--placeholder', action='store_true', help='write illustrative random numbers instead (no model, no download)')
    args = ap.parse_args()
    texts = json.loads(args.texts.read_text()) if args.texts else page_texts()
    items = [('docs', 'key', d) for d in texts['docs']] + [('queries', 'id', q) for q in texts['queries']]

    if args.placeholder:
        note = 'Placeholder: illustrative random numbers, a plain word split and provisional ids, not model output. ' \
               'Run web/scripts/make_embeddings.py for the models\' outputs.'
        if args.only != 'learned':
            rng = random.Random(0)
            out = {'placeholder': True, 'note': note,
                   # the model the real file comes from, as its model card and configuration describe it
                   'model': args.model, 'revision': None, 'dim': 384, 'layers': 6, 'heads': 12, 'vocab_size': 30522,
                   'params': None, 'max_length': MAX_LENGTH, 'pooling': 'mean', 'normalized': True,
                   'docs': [], 'queries': [], 'check_sentence_transformers': None}
            for part, key, x in items:
                out[part].append({key: x[key], 'text': x['text'], **placeholder(x['text'], rng)})
            write(args.out, out, 'placeholder dense vectors')
        if args.only != 'dense':
            rng = random.Random(1)
            pool = sorted({t for _, _, x in items for t in plain_tokens(x['text'])[0][1:-1] if t.isalpha() and t not in STOP})
            out = {'placeholder': True, 'note': note, 'model': args.learned_model, 'revision': None, 'vocab_size': 30522,
                   'params': None, 'max_length': LEARNED_MAX_LENGTH, 'pooling': 'max', 'activation': 'log(1 + relu)',
                   'docs': [], 'queries': [], 'check_sparse_encoder': None}
            for part, key, x in items:
                out[part].append({key: x[key], 'text': x['text'], **placeholder_learned(x['text'], rng, pool, 6 if part == 'docs' else 3)})
            write(args.learned_out, out, 'placeholder learned sparse vectors')
        return

    if args.only != 'learned':
        tokenizer, model = load(args.model, args.revision)
        cfg = model.config
        out = {'model': args.model, 'revision': getattr(cfg, '_commit_hash', None) or args.revision,
               'transformers': __import__('transformers').__version__,
               'dim': cfg.hidden_size, 'layers': cfg.num_hidden_layers, 'heads': cfg.num_attention_heads,
               'vocab_size': cfg.vocab_size, 'params': sum(p.numel() for p in model.parameters()),
               'max_length': MAX_LENGTH, 'pooling': 'mean', 'normalized': True, 'docs': [], 'queries': []}
        for part, key, x in items:
            out[part].append({key: x[key], 'text': x['text'], **embed(x['text'], tokenizer, model)})
        everything = out['docs'] + out['queries']
        out['check_sentence_transformers'] = check_sentence_transformers(
            [x['text'] for x in everything], [x['vector'] for x in everything], args.model, args.revision)
        write(args.out, out, f"{args.model} vectors ({out['dim']} dimensions, {out['params']:,} parameters)")
        if out['check_sentence_transformers'] is not None:
            print(f"  largest difference to sentence-transformers: {out['check_sentence_transformers']:.2e}")
        for q in out['queries']:
            scores = sorted(((sum(a * b for a, b in zip(q['vector'], d['vector'])), d['key']) for d in out['docs']), reverse=True)
            print(f"  {q['text']:<52}" + '  '.join(f'{k} {s:.3f}' for s, k in scores))

    if args.only != 'dense':
        tokenizer, model = load_learned(args.learned_model)
        cfg = model.config
        out = {'model': args.learned_model, 'revision': getattr(cfg, '_commit_hash', None),
               'transformers': __import__('transformers').__version__, 'vocab_size': cfg.vocab_size,
               'params': sum(p.numel() for p in model.parameters()), 'max_length': LEARNED_MAX_LENGTH,
               'pooling': 'max', 'activation': 'log(1 + relu)', 'docs': [], 'queries': []}
        for part, key, x in items:
            out[part].append({key: x[key], 'text': x['text'], **learned(x['text'], tokenizer, model)})
        everything = out['docs'] + out['queries']
        out['check_sparse_encoder'] = check_sparse_encoder([x['text'] for x in everything], [x['terms'] for x in everything],
                                                           args.learned_model, None)
        write(args.learned_out, out, f"{args.learned_model} weights ({out['params']:,} parameters)")
        if out['check_sparse_encoder'] is not None:
            print(f"  largest difference to SparseEncoder: {out['check_sparse_encoder']:.2e}")
        for q in out['queries']:
            qw = {j: w for j, _, w in q['terms']}
            scores = sorted(((sum(w * qw.get(j, 0) for j, _, w in d['terms']), d['key']) for d in out['docs']), reverse=True)
            print(f"  {q['text']:<52}" + '  '.join(f'{k} {s:.2f}' for s, k in scores))


if __name__ == '__main__':
    main()
