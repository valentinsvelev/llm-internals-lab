"""Reading the data embedded in the explorer page, and fake next-token bundles."""
from __future__ import annotations

import json
import re

from llm_lab.explorer import page_tokens, render_html


def embedded_specs(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="model-specs">(.*?)</script>', html, re.S)
    assert match, 'model-specs block missing'
    return json.loads(match.group(1))


def embedded_tokens(html: str, example: str = 'subwords') -> dict:
    """One example input of the page (the first one, shown on load, by default)."""
    return next(e for e in page_tokens(html)['examples'] if e['key'] == example)


def fake_entry(example: str = 'subwords', models=('mha', 'gqa')) -> dict:
    from llm_lab.next_token import HF_IDS, TEMPERATURES
    toks = embedded_tokens(render_html(), example)
    logits = [8.0, 7.5, 6.0]
    entry = {'prompt': toks['sentence'], 'models': {}}
    for key in models:
        entry['models'][key] = {
            'hf_id': HF_IDS[key], 'token_ids': toks['models'][key]['ids'],
            'top': [{'id': 100 + i, 'piece': f'Ġw{i}', 'text': f' w{i}', 'logit': z} for i, z in enumerate(logits)],
            'lse': {f'{t:.1f}': max(logits) / t + 1.0 for t in TEMPERATURES}}
    return entry


def fake_bundle(models=('mha', 'gqa'), examples=('subwords', 'fact')) -> dict:
    return {'kind': 'next-token-logits', 'examples': {ex: fake_entry(ex, models) for ex in examples}}
