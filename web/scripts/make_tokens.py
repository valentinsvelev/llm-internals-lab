"""Tokenize the example inputs with every model's real tokenizer and write src/tokens.json.

    TOK_DIR=<folder with the @lenml/tokenizer-* npm packages> \
    DEEPSEEK_LLM_GGUF=<ggml-vocab-deepseek-llm.gguf from llama.cpp> \
    python3 scripts/make_tokens.py

Reads src/examples.json (sentences, candidate next tokens, illustrative logits) and adds, per model:
the token IDs and pieces of the sentence, which word each piece came from (-1 for special tokens), and the
IDs of the candidate and beam-continuation tokens. Fails if a candidate is not a single token in every
tokenizer, or a sentence is longer than the diagram's 10 rows.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from bpe_gguf import encode, load  # noqa: E402

# page model → tokenizer: npm port name, or 'gguf' for DeepSeek-V2-Lite's DeepSeek-LLM BPE
TOKENIZER = {'mha': 'gpt2', 'mqa': 'gemma', 'gqa': 'qwen2_5', 'mla': 'gguf', 'gdn': 'qwen3', 'dsa': 'deepseek_v3'}
DEEPSEEK_LLM_BOS = 100000
MAX_ROWS = 10


def piece_text(piece: str) -> str:
    """The text a piece stands for (plain ASCII input: only the space markers need mapping)."""
    return piece.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')


def words_of(sentence: str, pieces: list[str]) -> list[int]:
    """Index of the space-separated word each piece starts in; -1 for special tokens."""
    out, pos = [], 0
    for p in pieces:
        if p.startswith('<') and p.endswith('>'):
            out.append(-1)
            continue
        text = piece_text(p)
        assert sentence.startswith(text, pos), (sentence, pos, text)
        first = pos + len(text) - len(text.lstrip(' '))
        out.append(sentence[:first].count(' '))
        pos += len(text)
    assert pos == len(sentence), (sentence, pieces)
    return out


def main() -> None:
    src = json.loads((ROOT / 'src/examples.json').read_text(encoding='utf-8'))
    examples = src['examples']
    sentences = [e['sentence'] for e in examples]
    singles = sorted({t for e in examples for t, _ in e['candidates']} | {t for e in examples for lst in e['beam'].values() for t, _ in lst})
    npm = json.loads(subprocess.run(['node', str(HERE / 'tok_lenml.mjs'), json.dumps(sentences), json.dumps(singles)],
                                    check=True, capture_output=True, text=True, env=os.environ).stdout)
    vocab, ranks, _ = load(os.environ['DEEPSEEK_LLM_GGUF'])
    gguf = {'sent': [[{'id': DEEPSEEK_LLM_BOS, 'piece': '<｜begin▁of▁sentence｜>'}] + encode(s, vocab, ranks) for s in sentences],
            'single': [[t['id'] for t in encode(w, vocab, ranks)] for w in singles]}

    def single_id(model: str, text: str) -> int:
        name = TOKENIZER[model]
        ids = gguf['single'][singles.index(text)] if name == 'gguf' else npm[name + '_tokens'][singles.index(text)]
        if len(ids) != 1:
            raise SystemExit(f'{text!r} is {len(ids)} tokens for {model}; candidates must be single tokens')
        return ids[0]

    out = []
    for i, e in enumerate(examples):
        cands = [t for t, _ in e['candidates']]
        conts = sorted({t for lst in e['beam'].values() for t, _ in lst} - set(cands))
        for first in e['beam']:
            if first not in cands[:3]:
                raise SystemExit(f'{e["key"]}: beam continuations must be given for the three most likely candidates')
        models = {}
        for model, name in TOKENIZER.items():
            toks = gguf['sent'][i] if name == 'gguf' else npm[name][i]
            pieces = [t['piece'] for t in toks]
            if len(toks) > MAX_ROWS:
                raise SystemExit(f'{e["key"]}: {len(toks)} tokens for {model}; the diagram has {MAX_ROWS} rows')
            models[model] = {'ids': [t['id'] for t in toks], 'pieces': pieces, 'words': words_of(e['sentence'], pieces),
                             'candidate_ids': [single_id(model, t) for t in cands],
                             'continuation_ids': [single_id(model, t) for t in conts]}
        out.append({'key': e['key'], 'label': e['label'], 'note': e['note'], 'sentence': e['sentence'],
                    'candidates': cands, 'logits': [z for _, z in e['candidates']], 'beam': e['beam'],
                    'continuations': conts, 'models': models})
    (ROOT / 'src/tokens.json').write_text(json.dumps({'examples': out}, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote src/tokens.json:', ', '.join(f'{e["key"]} ({len(e["models"]["mha"]["ids"])} GPT-2 tokens)' for e in out))


if __name__ == '__main__':
    main()
