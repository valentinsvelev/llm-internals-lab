"""Real next-token logits: the export (without network), validation, and embedding into the page."""
from __future__ import annotations

import json
import sys

import pytest
from helpers import fake_bundle, fake_entry

from llm_lab.explorer import page_tokens, render_html


def test_next_token_bundle_is_validated_and_embedded(tmp_path):
    path = tmp_path / 'next.json'
    path.write_text(json.dumps(fake_bundle()), encoding='utf-8')
    html = render_html(path)
    assert '"hf_id":"openai-community/gpt2"' in html
    bad = fake_bundle()
    bad['examples']['fact']['models']['mha']['token_ids'] = bad['examples']['fact']['models']['mha']['token_ids'][1:]
    path.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(ValueError, match='token IDs'):
        render_html(path)
    bad = fake_bundle()
    bad['examples']['fact']['prompt'] = 'Something else'
    path.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(ValueError, match='sentence'):
        render_html(path)
    # older bundles: one prompt, which must be one of the page's examples
    old = {'kind': 'next-token-logits', **fake_entry('story')}
    path.write_text(json.dumps(old), encoding='utf-8')
    assert '"prompt":"Once upon a time, there was a"' in render_html(path)
    old['prompt'] = 'Something else'
    path.write_text(json.dumps(old), encoding='utf-8')
    with pytest.raises(ValueError, match='sentence'):
        render_html(path)


def test_next_token_export_without_network(monkeypatch):
    import types

    torch = pytest.importorskip('torch')
    from llm_lab import next_token
    examples = {e['sentence']: e for e in page_tokens(render_html())['examples']}
    z = torch.tensor([0.5, 3.0, -1.0, 2.0, 1.0])

    class FakeTok:
        def __call__(self, text, return_tensors=None):
            return {'input_ids': torch.tensor([examples[text]['models']['mha']['ids']])}
        def convert_ids_to_tokens(self, i):
            return f'Ġt{i}'
        def decode(self, ids):
            return f' t{ids[0]}'

    class FakeModel:
        def eval(self):
            return self
        def __call__(self, input_ids):
            logits = torch.zeros(1, input_ids.shape[1], z.numel())
            logits[0, -1] = z
            return types.SimpleNamespace(logits=logits)

    class Auto:
        @staticmethod
        def from_pretrained(hf_id, **kw):
            assert kw['trust_remote_code'] is False and hf_id == 'openai-community/gpt2'
            return FakeTok() if 'torch_dtype' not in kw else FakeModel()

    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=Auto, AutoModelForCausalLM=Auto))
    bundle = next_token.export(['mha'], top=3)
    assert set(bundle['examples']) == {'subwords', 'fact', 'story', 'attention'}
    assert next_token.export(['mha'], examples=['story'], top=3)['examples'].keys() == {'story'}
    entry = bundle['examples']['story']['models']['mha']
    assert [c['id'] for c in entry['top']] == [1, 3, 4]
    assert entry['top'][0]['piece'] == 'Ġt1'
    assert entry['lse']['0.5'] == pytest.approx(float(torch.logsumexp(z / 0.5, 0)), abs=1e-3)
    p = torch.softmax(z, 0)
    assert entry['entropy']['1.0'] == pytest.approx(float(-(p * p.log()).sum()), abs=1e-3)
    assert [b['id'] for b in entry['beam']] == [1, 3, 4]               # the three most likely first tokens
    assert entry['beam'][0]['top'][0]['logprob'] == pytest.approx(float(torch.log_softmax(z, 0)[1]), abs=1e-3)
    toks = page_tokens(render_html())
    next_token.validate(bundle, toks)
    entry['beam'][0]['top'][0]['logprob'] = 0.5
    with pytest.raises(ValueError, match='beam'):
        next_token.validate(bundle, toks)
