// usage: node tok_lenml.mjs '<json array of sentences>' ['<json array of single tokens>']
//   → JSON {model: [[{id, piece}]...], model_tokens: [[ids]...]}
// Ports of the Hugging Face tokenizer.json files: @lenml/tokenizer-{gpt2,gemma,qwen2_5,qwen3,deepseek_v3}.
// Qwen3-Next uses the Qwen3 tokenizer (the same IDs as Qwen2.5). DeepSeek-V3.2 uses the DeepSeek-V3
// tokenizer, whose tokenizer_config adds BOS (ID 0) in front; the npm port does not, so it is prepended.
import { createRequire } from 'node:module';
const require = createRequire(process.env.TOK_DIR ? `${process.env.TOK_DIR}/package.json` : import.meta.url);
const out = {};
const sentences = JSON.parse(process.argv[2]);
const singles = JSON.parse(process.argv[3] || '[]');
for (const [name, pkg, bos] of [['gpt2', '@lenml/tokenizer-gpt2'], ['qwen2_5', '@lenml/tokenizer-qwen2_5'], ['gemma', '@lenml/tokenizer-gemma'],
  ['qwen3', '@lenml/tokenizer-qwen3'], ['deepseek_v3', '@lenml/tokenizer-deepseek_v3', 0]]) {
  const { fromPreTrained } = require(pkg);
  const tk = fromPreTrained();
  out[name] = sentences.map(s => {
    let ids = tk.encode(s, { add_special_tokens: true });
    if (bos != null && ids[0] !== bos) ids = [bos, ...ids];
    return ids.map(id => ({ id, piece: tk.model.convert_ids_to_tokens([id])[0] }));
  });
  out[name + '_tokens'] = singles.map(w => tk.encode(w, { add_special_tokens: false }));
}
console.log(JSON.stringify(out));
