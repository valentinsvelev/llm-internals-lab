"""Real next-token logits for the explorer's example inputs, computed on CPU.

The explorer runs no model, so its decoding demo starts with clearly labeled
illustrative logits. This module replaces them with the real thing:

    pip install -e ".[pretrained]"
    python -m llm_lab.next_token --models mha gqa --out next_token.json
    python -m llm_lab.explorer --serve --next-token next_token.json

For each model and each example input of the page (or the ones chosen with --examples) it
tokenizes the same sentence the page shows (with the tokenizer's
default special tokens), runs one forward pass, and stores the top logits of the
last position plus the log-sum-exp and the entropy of the full vocabulary at every
temperature the page offers, so the page's probabilities (and locally typical
sampling) are exact over the whole vocabulary. For beam search it also runs the
three most likely continuations once more and stores their next-token
log-probabilities. Only pretrained weights from Hugging Face are used; no remote
code is executed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HF_IDS = {
    'mha': 'openai-community/gpt2',
    'mqa': 'google/gemma-2b',            # gated: accept the licence on Hugging Face first
    'gqa': 'Qwen/Qwen2.5-0.5B',
    'mla': 'deepseek-ai/DeepSeek-V2-Lite',  # 15.7B parameters: needs ~32 GB of RAM in bf16
    'gdn': 'Qwen/Qwen3-Next-80B-A3B-Instruct',  # 80B parameters: ~160 GB in bf16
    'dsa': 'deepseek-ai/DeepSeek-V3.2',        # 685B parameters: a multi-GPU server
}
BEAM_PREFIXES = 3     # the page's beam width goes up to 3
BEAM_TOP = 5          # continuations stored per prefix (the page needs at least the beam width)
TEMPERATURES = [round(0.1 * i, 1) for i in range(1, 21)]   # 0.1 … 2.0, the page's slider


def page_examples() -> dict[str, str]:
    """The page's example inputs, key → sentence."""
    from .explorer import page_tokens
    return {e['key']: e['sentence'] for e in page_tokens()['examples']}


def export(models: list[str], *, examples: list[str] | None = None, top: int = 50,
           local_files_only: bool = False, dtype: str = 'float32') -> dict:
    """Run each model on the page's example inputs and return the bundle the page embeds:
    ``{'kind': 'next-token-logits', 'examples': {key: {'prompt': sentence, 'models': {…}}}}``."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError('Install the optional dependencies: pip install -e ".[pretrained]"') from exc
    sentences = page_examples()
    for ex in examples or []:
        if ex not in sentences:
            raise ValueError(f'Unknown example {ex!r}; choose from {", ".join(sentences)}')
    chosen = {k: v for k, v in sentences.items() if not examples or k in examples}
    bundle = {'kind': 'next-token-logits', 'examples': {k: {'prompt': v, 'models': {}} for k, v in chosen.items()}}
    for key in models:
        if key not in HF_IDS:
            raise ValueError(f'Unknown model {key!r}; choose from {", ".join(HF_IDS)}')
        hf_id = HF_IDS[key]
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), trust_remote_code=False,
            local_files_only=local_files_only, low_cpu_mem_usage=True).eval()
        for ex, sentence in chosen.items():
            bundle['examples'][ex]['models'][key] = _run(torch, tok, model, hf_id, sentence, top)
        del model
    return bundle


def _run(torch, tok, model, hf_id: str, sentence: str, top: int) -> dict:
    """One forward pass on the sentence; top logits, log-sum-exp and entropy per temperature, beam step."""
    ids = tok(sentence, return_tensors='pt')['input_ids']
    with torch.no_grad():
        z = model(input_ids=ids).logits[0, -1].float()
    values, indices = torch.topk(z, min(top, z.numel()))
    entry = _describe(tok, values, indices)
    entropy = {}
    for t in TEMPERATURES:
        logp = torch.log_softmax(z / t, dim=0)
        entropy[f'{t:.1f}'] = round(float(-(logp.exp() * logp).sum()), 4)
    beam = []
    for first in indices[:BEAM_PREFIXES].tolist():
        with torch.no_grad():
            z2 = model(input_ids=torch.cat([ids, torch.tensor([[first]])], dim=1)).logits[0, -1].float()
        lp = torch.log_softmax(z2, dim=0)
        v2, i2 = torch.topk(lp, min(BEAM_TOP, lp.numel()))
        beam.append({'id': int(first), 'top': [{'id': c['id'], 'piece': c['piece'], 'text': c['text'], 'logprob': c['logit']}
                                             for c in _describe(tok, v2, i2)]})
    return {
        'hf_id': hf_id,
        'token_ids': [int(i) for i in ids[0].tolist()],
        'top': entry,
        'lse': {f'{t:.1f}': round(float(torch.logsumexp(z / t, dim=0)), 4) for t in TEMPERATURES},
        'entropy': entropy,
        'beam': beam,
    }


def _describe(tok, values, indices) -> list[dict]:
    return [{'id': int(i), 'piece': tok.convert_ids_to_tokens(int(i)), 'text': tok.decode([int(i)]),
             'logit': round(float(v), 4)} for v, i in zip(values.tolist(), indices.tolist())]


def validate(bundle: dict, page_tokens: dict) -> None:
    """Refuse a bundle that does not belong to the page: other prompt, model or tokenization.

    Bundles hold several examples (``{'examples': {key: {'prompt', 'models'}}}``); older ones hold one
    prompt (``{'prompt', 'models'}``), which must be one of the page's example sentences."""
    if bundle.get('kind') != 'next-token-logits':
        raise ValueError('Not a next-token-logits bundle')
    page = {e['key']: e for e in page_tokens['examples']}
    if 'examples' in bundle:
        examples = bundle['examples']
        if not isinstance(examples, dict) or not examples:
            raise ValueError('Bundle has no examples')
    else:
        key = next((k for k, e in page.items() if e['sentence'] == bundle.get('prompt')), None)
        if key is None:
            raise ValueError('Bundle was computed for a sentence that is not one of the page\'s examples')
        examples = {key: bundle}
    for ex, entry in examples.items():
        if ex not in page:
            raise ValueError(f'Unknown example {ex!r}')
        if entry.get('prompt') != page[ex]['sentence']:
            raise ValueError(f'{ex}: bundle was computed for a different sentence than the page shows')
        _validate_models(entry.get('models'), page[ex])


def _validate_models(models, example: dict) -> None:
    if not isinstance(models, dict) or not models:
        raise ValueError('Bundle has no models')
    for key, entry in models.items():
        if HF_IDS.get(key) != entry.get('hf_id'):
            raise ValueError(f'{key}: expected {HF_IDS.get(key)}, got {entry.get("hf_id")}')
        if entry.get('token_ids') != example['models'][key]['ids']:
            raise ValueError(f'{key}: token IDs differ from the tokenization shown on the page')
        top = entry.get('top')
        if not top or not all(isinstance(c.get('id'), int) and math.isfinite(c.get('logit', math.nan)) for c in top):
            raise ValueError(f'{key}: missing or invalid top logits')
        lse = entry.get('lse', {})
        if sorted(lse) != sorted(f'{t:.1f}' for t in TEMPERATURES):
            raise ValueError(f'{key}: log-sum-exp must cover temperatures 0.1 to 2.0')
        if any(c['logit'] > lse['1.0'] + 1e-3 for c in top):
            raise ValueError(f'{key}: a logit exceeds the log-sum-exp; the bundle is inconsistent')
        entropy = entry.get('entropy')       # optional: older bundles have none
        if entropy is not None and (sorted(entropy) != sorted(lse) or not all(0 <= h < 20 for h in entropy.values())):
            raise ValueError(f'{key}: entropy must be given for every temperature, in nats')
        ids = {c['id'] for c in top}
        for b in entry.get('beam') or []:   # optional: second-step log-probabilities for beam search
            if b.get('id') not in ids or not b.get('top') or not all(c.get('logprob', 1) <= 1e-6 for c in b['top']):
                raise ValueError(f'{key}: invalid beam-search continuations')


def main() -> None:
    ap = argparse.ArgumentParser(description='Export real next-token logits for the architecture explorer')
    ap.add_argument('--models', nargs='+', default=['mha', 'gqa'], choices=sorted(HF_IDS))
    ap.add_argument('--examples', nargs='+', help='example inputs of the page (default: all)')
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--local-files-only', action='store_true')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', type=Path, default=Path('next_token.json'))
    a = ap.parse_args()
    bundle = export(a.models, examples=a.examples, top=a.top, local_files_only=a.local_files_only, dtype=a.dtype)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(bundle, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {a.out}. Use it with: python -m llm_lab.explorer --serve --next-token {a.out}')


if __name__ == '__main__':
    main()
