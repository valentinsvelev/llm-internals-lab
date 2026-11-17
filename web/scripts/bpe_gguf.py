"""usage: pip install gguf; python bpe_gguf.py ggml-vocab-deepseek-llm.gguf '["text"]'

Byte-level BPE from a llama.cpp vocab GGUF (tokens + merges). For plain English text only:
pre-tokenization splits into ' ?letters', ' ?digits' and ' ?punctuation' chunks, which is what the
GPT-2, Qwen2 and DeepSeek-LLM pre-tokenizer regexes produce for such text."""
import json, re, sys
from gguf import GGUFReader

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))
B2U = bytes_to_unicode()

def load(path):
    r = GGUFReader(path)
    def arr(key):
        f = r.fields[key]
        return [bytes(f.parts[i]).decode('utf-8') for i in f.data]
    def val(key):
        f = r.fields.get(key)
        return None if f is None else f.parts[f.data[0]].tolist()[0]
    tokens = arr('tokenizer.ggml.tokens'); merges = arr('tokenizer.ggml.merges')
    meta = {k: val(k) for k in ['tokenizer.ggml.bos_token_id', 'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.eos_token_id']}
    pre = r.fields.get('tokenizer.ggml.pre'); meta['pre'] = bytes(pre.parts[pre.data[0]]).decode() if pre else None
    return tokens, {tuple(m.split(' ', 1)): i for i, m in enumerate(merges)}, meta

def bpe(word, ranks):
    parts = list(word)
    while len(parts) > 1:
        best = min(((ranks.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))))
        if best[0] == 1e18: break
        i = best[1]; parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts

def encode(text, tokens, ranks):
    index = {t: i for i, t in enumerate(tokens)}
    out = []
    for chunk in re.findall(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+", text):
        u = ''.join(B2U[b] for b in chunk.encode('utf-8'))
        for p in bpe(u, ranks): out.append({'id': index[p], 'piece': p})
    return out

if __name__ == '__main__':
    sentences = json.loads(sys.argv[2])
    tokens, ranks, meta = load(sys.argv[1])
    print(json.dumps({'meta': meta, 'n_vocab': len(tokens), 'out': [encode(s, tokens, ranks) for s in sentences]}))
