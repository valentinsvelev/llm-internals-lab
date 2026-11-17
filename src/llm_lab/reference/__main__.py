"""python -m llm_lab.reference                 summary of the six reference models (no download)
   python -m llm_lab.reference --run HF_ID      run a real checkpoint through the reference code
                                               and compare it with transformers, e.g. openai-community/gpt2"""
from __future__ import annotations

import argparse

import torch

from .config import CONFIGS
from .model import DecoderLM


def summary() -> None:
    print(f'{"tab":4} {"model":20} {"params":>9} {"active":>8} {"cache/token":>12}  mixer per layer')
    for key, cfg in CONFIGS.items():
        with torch.device('meta'):
            total = sum(p.numel() for p in DecoderLM(cfg).parameters())
        idle = 0
        if cfg.moe:   # experts a token does not use
            idle = (cfg.moe.experts - cfg.moe.top_k) * 3 * cfg.moe.expert_dim * cfg.hidden * sum(map(cfg.uses_moe, range(cfg.layers)))
        # bytes per token over all layers: 16-bit values, DSA's index keys in FP8
        per_token = sum(2 * cfg.cache_values_per_token(i) - (cfg.index_dim if cfg.layer_type(i) == 'attention' else 0)
                        for i in range(cfg.layers))
        kinds = {cfg.layer_type(i) for i in range(cfg.layers)}
        mixer = ('Gated DeltaNet ×3 + gated attention' if 'gdn' in kinds else
                 'MLA + lightning indexer' if cfg.index_heads else 'MLA' if cfg.attention == 'mla' else
                 f'attention {cfg.heads}q/{cfg.kv_heads}kv')
        print(f'{key.upper():4} {cfg.name:20} {total / 1e9:8.2f}B {(total - idle) / 1e9:7.2f}B {per_token / 1024:9.1f} KiB  {mixer}')


@torch.no_grad()
def run(hf_id: str, prompt: str, steps: int) -> None:
    from .hf import from_pretrained
    ours, hf, tok = from_pretrained(hf_id)
    ids = tok(prompt, return_tensors='pt')['input_ids']
    diff = (ours(ids) - hf(ids).logits).abs().max().item()
    out = ours.generate(ids, steps)
    print(f'max |logit difference| vs transformers: {diff:.2e}')
    print(tok.decode(out[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', metavar='HF_ID')
    ap.add_argument('--prompt', default='Tokenizers split unpronounceable words into')
    ap.add_argument('--steps', type=int, default=12)
    a = ap.parse_args()
    run(a.run, a.prompt, a.steps) if a.run else summary()


if __name__ == '__main__':
    main()
