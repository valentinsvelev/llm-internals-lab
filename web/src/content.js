/* Attention Explorer — content and arithmetic.
 *
 * Pure data + pure functions, no DOM. The same file runs in the browser and in
 * the Node build script (which pre-renders every TeX string to MathML).
 *
 * Every dimension below is copied from the model's published config.json.
 * Nothing here is a model output: there are no weights and no activations.
 */
(function (root) {
'use strict';

const MODELS = {
  mha: {
    key: 'mha', abbr: 'MHA', name: 'Multi-Head Attention',
    model: 'GPT-2 small', hf: 'openai-community/gpt2',
    config: 'https://huggingface.co/openai-community/gpt2/blob/main/config.json',
    d: 768, H: 12, G: 12, dh: 64, L: 12, ctx: 1024,
    norm: 'LayerNorm', ffn: 'MLP', pos: 'learned',
    posLabel: 'Learned, added to the input',
    V: 50257, ffnKind: 'mlp', act: 'GELU', dff: 3072, eps: '10^{-5}', epsText: '10⁻⁵', tied: true,
    tokenizer: 'byte-level BPE', ffnLabel: 'GELU MLP',
  },
  mqa: {
    key: 'mqa', abbr: 'MQA', name: 'Multi-Query Attention',
    model: 'Gemma 2B', hf: 'google/gemma-2b',
    config: 'https://huggingface.co/google/gemma-2b/blob/main/config.json',
    d: 2048, H: 8, G: 1, dh: 256, L: 18, ctx: 8192,
    norm: 'RMSNorm', ffn: 'MLP', pos: 'rope', theta: '10,000',
    posLabel: 'RoPE, θ = 10,000',
    V: 256000, ffnKind: 'glu', act: 'GELU', dff: 16384, eps: '10^{-6}', epsText: '10⁻⁶', tied: true,
    tokenizer: 'SentencePiece', ffnLabel: 'GeGLU', embScale: true, normPlusOne: true, thetaN: 10000, ropeDim: 256,
  },
  gqa: {
    key: 'gqa', abbr: 'GQA', name: 'Grouped-Query Attention',
    model: 'Qwen2.5-0.5B', hf: 'Qwen/Qwen2.5-0.5B',
    config: 'https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json',
    d: 896, H: 14, G: 2, dh: 64, L: 24, ctx: 32768,
    norm: 'RMSNorm', ffn: 'MLP', pos: 'rope', theta: '1,000,000',
    posLabel: 'RoPE, θ = 1,000,000',
    V: 151936, ffnKind: 'glu', act: 'SiLU', dff: 4864, eps: '10^{-6}', epsText: '10⁻⁶', tied: true,
    tokenizer: 'byte-level BPE', ffnLabel: 'SwiGLU', thetaN: 1000000, ropeDim: 64,
  },
  mla: {
    key: 'mla', abbr: 'MLA', name: 'Multi-head Latent Attention',
    model: 'DeepSeek-V2-Lite', hf: 'deepseek-ai/DeepSeek-V2-Lite',
    config: 'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json',
    d: 2048, H: 16, G: null, dh: 128, dr: 64, dv: 128, dc: 512, L: 27, ctx: 32768,
    norm: 'RMSNorm', ffn: 'MoE', pos: 'decoupled',
    posLabel: 'Decoupled RoPE + YaRN',
    V: 102400, ffnKind: 'moe', act: 'SiLU', dff: 10944, eps: '10^{-6}', epsText: '10⁻⁶', tied: false,
    tokenizer: 'byte-level BPE', ffnLabel: 'SwiGLU MoE', thetaN: 10000, ropeDim: 64, yarn: { factor: 40, low: 10, high: 23 },  // YaRN correction range for rope_dim 64, base 10,000, 4K → ×40
    experts: 64, active: 6, shared: 2, dExpert: 1408, firstDense: 1, ms2: 1.59,  // YaRN softmax factor (0.1·0.707·ln 40 + 1)²
  },
  gdn: {
    key: 'gdn', abbr: 'GDN', name: 'Gated DeltaNet', tab: 'linear · hybrid',
    model: 'Qwen3-Next-80B-A3B', hf: 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    config: 'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json',
    d: 2048, H: 16, G: 2, dh: 256, L: 48, ctx: 262144,
    norm: 'RMSNorm', ffn: 'MoE', pos: 'rope', theta: '10,000,000', thetaN: 10000000, ropeDim: 64,
    posLabel: 'RoPE on 64 of 256 dims, attention layers only',
    V: 151936, ffnKind: 'moe', act: 'SiLU', dff: null, eps: '10^{-6}', epsText: '10⁻⁶', tied: false,
    tokenizer: 'byte-level BPE', ffnLabel: 'SwiGLU MoE', normPlusOne: true,
    lin: { qk: 16, v: 32, dk: 128, dv: 128, conv: 4 }, interval: 4,   // full_attention_interval: every 4th layer is attention
    experts: 512, active: 10, shared: 1, dExpert: 512, firstDense: 0, params: '80B total · 3B active',
  },
  dsa: {
    key: 'dsa', abbr: 'DSA', name: 'Sparse Attention', tab: 'DSA · QSA',
    model: 'DeepSeek-V3.2', hf: 'deepseek-ai/DeepSeek-V3.2',
    config: 'https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json',
    d: 7168, H: 128, G: null, dh: 128, dr: 64, dv: 128, dc: 512, dq: 1536, L: 61, ctx: 163840,
    norm: 'RMSNorm', ffn: 'MoE', pos: 'decoupled',
    posLabel: 'Decoupled RoPE + YaRN',
    V: 129280, ffnKind: 'moe', act: 'SiLU', dff: 18432, eps: '10^{-6}', epsText: '10⁻⁶', tied: false,
    tokenizer: 'byte-level BPE', ffnLabel: 'SwiGLU MoE', thetaN: 10000, ropeDim: 64, yarn: { factor: 40, low: 10, high: 23 },
    experts: 256, active: 8, shared: 1, dExpert: 2048, firstDense: 3, ms2: 1.87,  // (0.1·1.0·ln 40 + 1)²
    index: { heads: 64, dim: 128, topk: 2048 }, drawH: 16, params: '671B main model',
  },
};
const ORDER = ['mha', 'mqa', 'gqa', 'mla', 'gdn', 'dsa'];
/* First publications of the named designs, not the release years of the model examples.
 * "Sparse" groups the DSA (2025) and QSA (2026) implementations, not all sparse attention. */
const ARCH_YEARS = {
  mha: { year: '2017', note: 'Multi-head attention · Attention Is All You Need (2017)' },
  mqa: { year: '2019', note: 'Multi-query attention · Shazeer (2019)' },
  gqa: { year: '2023', note: 'Grouped-query attention · Ainslie et al. (2023)' },
  mla: { year: '2024', note: 'Multi-head latent attention · DeepSeek-V2 (2024)' },
  gdn: { year: '2024', note: 'Gated DeltaNet · preprint 2024, ICLR 2025' },
  dsa: { year: '2025–26', note: 'DeepSeek Sparse Attention (2025) and Qwen Sparse Attention (2026); sparse attention itself is older' },
};
/* Qwen's block-sparse variant, shown in "Attention up close" of the sparse tab.
 * Qwen3.8-Flash-Next text config (via Ollama / LMSYS): 48 layers, 3 GDN : 1 QSA. */
const QSA = { model: 'Qwen3.8-Flash-Next', H: 24, G: 2, dh: 256, L: 48, attnLayers: 12, block: 4, blocks: 512, heads: 4, dim: 128, budget: 2048 };

/* The example inputs, each tokenized by every model's own tokenizer (scripts/make_tokens.py, from
 * src/examples.json): sentence, candidate next tokens with ILLUSTRATIVE logits, illustrative beam-search
 * probabilities, and per model the pieces, IDs, the word each piece came from (-1 = special token) and the
 * IDs of the candidates. The logits are not computed from any model: this page runs no model. Real ones
 * can be embedded with `python -m llm_lab.next_token` (see the README); then `source` becomes "computed".
 * TOKENS is the example on show; setExample() switches it in place, so every reader sees the change. */
const TOKEN_DATA = (typeof module !== 'undefined' && module.exports) ? require('./tokens.json') : root.LAB_TOKENS;
const EXAMPLES = TOKEN_DATA.examples;
const TOKENS = {};
function setExample(key) {
  const ex = EXAMPLES.find(e => e.key === key) || EXAMPLES[0];
  Object.keys(TOKENS).forEach(k => delete TOKENS[k]);
  return Object.assign(TOKENS, ex);
}
setExample(EXAMPLES[0].key);
/* Speculative decoding: the draft model's logits are the target's plus these offsets (illustrative). */
const DRAFT_SHIFT = [-0.5, 0.45, -0.3, 0.4, 0.1, -0.35, 0.25, 0.05, -0.15, 0.2, 0.0, -0.2, 0.1, 0.0];

function pieceOf(key, text) {
  if (!/^ /.test(text)) return text;
  return (key === 'mqa' ? '▁' : 'Ġ') + text.slice(1);
}
function idOf(key, text) {
  const tk = TOKENS.models[key];
  let i = TOKENS.candidates.indexOf(text);
  if (i >= 0) return tk.candidate_ids[i];
  i = TOKENS.continuations.indexOf(text);
  return i >= 0 ? tk.continuation_ids[i] : null;
}
/* Real logits for the example on show: a bundle holds several examples, or (older bundles) one prompt. */
function realEntry(real, key) {
  if (!real) return null;
  const ex = real.examples ? real.examples[TOKENS.key] : real.prompt === TOKENS.sentence ? real : null;
  return ex && ex.models && ex.models[key] ? ex.models[key] : null;
}
function nextTokenData(key, real) {
  const r = realEntry(real, key);
  if (r) {
    const cands = r.top.map(c => ({ text: c.text, piece: c.piece, id: c.id, z: c.logit }));
    const beam = {};
    (r.beam || []).forEach(b => {
      const i = cands.findIndex(c => c.id === b.id);
      if (i >= 0) beam[i] = b.top.map(c => ({ text: c.text, piece: c.piece, id: c.id, lp: c.logprob }));
    });
    return { source: 'computed', model: r.hf_id, cands, lse: r.lse, entropy: r.entropy || null, beam };
  }
  const cands = TOKENS.candidates.map((text, i) => ({ text, piece: pieceOf(key, text), id: idOf(key, text), z: TOKENS.logits[i] }));
  const beam = {};
  Object.entries(TOKENS.beam).forEach(([first, list]) => {
    beam[TOKENS.candidates.indexOf(first)] = list.map(([text, p]) => ({ text, piece: pieceOf(key, text), id: idOf(key, text), lp: Math.log(p) }));
  });
  return { source: 'illustrative', cands, lse: null, entropy: null, beam };
}

/* The decoding methods, in the order of the buttons. */
const METHODS = [
  { key: 'greedy', label: 'Greedy', short: 'Greedy', kind: 'search' },
  { key: 'beam', label: 'Beam search', short: 'Beam', kind: 'search' },
  { key: 'temperature', label: 'Random sampling', short: 'Random', kind: 'sample' },
  { key: 'spec', label: 'Speculative decoding', short: 'Speculative', kind: 'fast' },
  { key: 'topk', label: 'Top-k sampling', short: 'Top-k', kind: 'filter' },
  { key: 'topp', label: 'Top-p (nucleus) sampling', short: 'Top-p', kind: 'filter' },
  { key: 'minp', label: 'Min-p sampling', short: 'Min-p', kind: 'filter' },
  { key: 'typical', label: 'Locally typical sampling', short: 'Typical', kind: 'filter' },
];
const usesTau = method => !['greedy', 'beam'].includes(method);

/* Decoding, as implemented in Hugging Face transformers (temperature first, then the filter).
 * Returns the sampling distribution over the listed candidates: p(τ) = softmax(z/τ), normalized over
 * the full vocabulary when its log-sum-exp is known, then filtered and renormalized.
 * o: { k, p, minp, typ } */
function decodeDist(data, method, tau, o = {}) {
  const t = usesTau(method) ? tau : 1;
  const key = t.toFixed(1);
  const zs = data.cands.map(c => c.z / t);
  const full = !!(data.lse && data.lse[key] != null);
  let lse;
  if (full) lse = data.lse[key];
  else { const mx = Math.max(...zs); lse = mx + Math.log(zs.reduce((a, z) => a + Math.exp(z - mx), 0)); }
  const base = data.cands.map((c, i) => ({ ...c, p0: Math.exp(zs[i] - lse) }));
  const order = base.map((c, i) => i).sort((a, b) => base[b].p0 - base[a].p0);
  // entropy of p(τ): exported for the full vocabulary, otherwise over the listed candidates
  const H = full && data.entropy && data.entropy[key] != null ? data.entropy[key]
    : -base.reduce((a, c) => a + (c.p0 > 0 ? c.p0 * Math.log(c.p0) : 0), 0);
  const pmax = base[order[0]].p0;
  const keep = new Set();
  let typOrder = null;
  if (method === 'greedy' || method === 'beam') keep.add(order[0]);
  else if (method === 'topk') order.slice(0, o.k).forEach(i => keep.add(i));
  else if (method === 'topp') { let cum = 0; for (const i of order) { keep.add(i); cum += base[i].p0; if (cum >= o.p) break; } }
  else if (method === 'minp') order.forEach(i => { if (base[i].p0 >= o.minp * pmax) keep.add(i); });
  else if (method === 'typical') {
    typOrder = base.map((c, i) => i).sort((a, b) => Math.abs(-Math.log(base[a].p0) - H) - Math.abs(-Math.log(base[b].p0) - H));
    let cum = 0; for (const i of typOrder) { keep.add(i); cum += base[i].p0; if (cum >= o.typ) break; }
  } else order.forEach(i => keep.add(i));
  const z = [...keep].reduce((a, i) => a + base[i].p0, 0);
  const cum = {}; let run = 0;
  order.forEach(i => { cum[i] = run; run += base[i].p0; });
  const tcum = {}; run = 0;
  if (typOrder) typOrder.forEach(i => { tcum[i] = run; run += base[i].p0; });
  return base.map((c, i) => ({ ...c, kept: keep.has(i), p: keep.has(i) ? c.p0 / z : 0, rank: order.indexOf(i),
    zt: zs[i], lse, mass: z, cum: cum[i], tau: t, full, H, pmax,
    dev: Math.abs(-Math.log(c.p0) - H), trank: typOrder ? typOrder.indexOf(i) : null, tcum: typOrder ? tcum[i] : null }));
}

/* Beam search, width B, two steps. Exact as long as every beam lists at least B continuations:
 * the B best two-token sequences are then among the listed ones. Scores are log-probabilities at τ = 1. */
function beamSearch(data, B) {
  const p1 = decodeDist(data, 'temperature', 1);
  const order = p1.map((c, i) => i).sort((a, b) => p1[b].p0 - p1[a].p0);
  const beams = order.slice(0, B).map(i => ({ i, lp: Math.log(p1[i].p0) }));
  const exps = [];
  beams.forEach(b => (data.beam[b.i] || []).forEach(c => exps.push({ from: b.i, ...c, score: b.lp + c.lp })));
  const ranked = exps.slice().sort((a, b) => b.score - a.score);
  ranked.forEach((e, r) => { e.rank = r; e.kept = r < B; });
  const greedy = (data.beam[order[0]] || []).slice().sort((a, b) => b.lp - a.lp)[0];
  return { beams, exps, best: ranked[0], p1, greedy: greedy && { from: order[0], ...greedy, score: Math.log(p1[order[0]].p0) + greedy.lp } };
}

/* Speculative sampling (Leviathan et al. 2023; Chen et al. 2023) for one position: draft x̃ ~ q,
 * accept with probability min(1, p/q), otherwise resample from r ∝ max(0, p − q). The result is
 * distributed exactly as p. Over the listed candidates: p is renormalized over them. */
function specDist(data, tau) {
  const base = decodeDist(data, 'temperature', tau);
  const mass = base.reduce((a, c) => a + c.p0, 0);
  const zq = data.cands.map((c, i) => (c.z + DRAFT_SHIFT[i % DRAFT_SHIFT.length]) / tau);
  const mx = Math.max(...zq), sq = zq.reduce((a, z) => a + Math.exp(z - mx), 0);
  const rows = base.map((c, i) => {
    const p = c.p0 / mass, q = Math.exp(zq[i] - mx) / sq;
    return { ...c, p, q, a: Math.min(1, p / q), over: Math.max(0, p - q), both: Math.min(p, q) };
  });
  const acc = rows.reduce((a, c) => a + c.both, 0);
  rows.forEach(c => { c.r = c.over / (1 - acc); c.kept = true; c.mass = mass; c.acc = acc; });
  return rows;
}
/* Same-width comparison used in the DeepSeek-V2 paper (Table 1):
 * 16 heads of dimension 128; MLA with d_c = 4·d_h and d_R = d_h/2; DSA adds a 128-dim index key;
 * Gated DeltaNet stores no per-token cache, one d_k × d_v state per head instead. */
const COMPARE = { H: 16, dh: 128, G: 4, dc: 512, dr: 64, di: 128 };

const fmt = n => n.toLocaleString('en-US');
const showPiece = p => p.startsWith('<') ? 'BOS' : p.replace(/^[Ġ▁]/, '·');

/* Values cached per token in one layer that keeps a per-token cache. */
function cacheValues(m) {
  if (m.key === 'mla') return m.dc + m.dr;
  if (m.key === 'dsa') return m.dc + m.dr + m.index.dim;
  return 2 * m.G * m.dh;
}
/* Layers with a per-token cache: all of them, except Gated DeltaNet layers. */
function cacheLayers(m) { return m.key === 'gdn' ? m.L / m.interval : m.L; }
/* Bytes per token: 16-bit values; DSA's index keys are stored in FP8 (1 byte, per-block scales ignored). */
function cacheBytesPerToken(m) {
  if (m.key === 'dsa') return ((m.dc + m.dr) * 2 + m.index.dim) * m.L;
  return cacheValues(m) * cacheLayers(m) * 2;
}
/* Gated DeltaNet: fixed state per layer (all value heads) plus the convolution's last k−1 inputs. */
function gdnState(m) {
  const { qk, v, dk, dv, conv } = m.lin;
  const heads = v * dk * dv, channels = 2 * qk * dk + v * dv;
  return { heads, convState: (conv - 1) * channels, channels, total: heads + (conv - 1) * channels };
}
function bytes(n) {
  const units = [['GiB', 2 ** 30], ['MiB', 2 ** 20], ['KiB', 2 ** 10]];
  for (const [u, s] of units) {
    if (n >= s) {
      const v = n / s;
      const txt = Number.isInteger(v) ? String(v) : (v >= 100 ? v.toFixed(0) : v.toFixed(1));
      return `${txt} ${u}`;
    }
  }
  return `${n} B`;
}
function compareRows() {
  const { H, dh, G, dc, dr, di } = COMPARE;
  return [
    { key: 'mha', label: 'MHA', formula: '2 H d_h', value: 2 * H * dh },
    { key: 'gqa', label: 'GQA', formula: '2 G d_h', value: 2 * G * dh },
    { key: 'dsa', label: 'DSA', formula: 'd_c + d_R + d_I', value: dc + dr + di },
    { key: 'mla', label: 'MLA', formula: 'd_c + d_R', value: dc + dr },
    { key: 'mqa', label: 'MQA', formula: '2 d_h', value: 2 * dh },
    { key: 'gdn', label: 'GDN', formula: '0', value: 0, star: true },   // a fixed state instead: see the note under the chart
  ];
}

/* Which KV head (1-based) does query head h (1-based) read? */
function kvHeadOf(m, h) { return m.key === 'mla' || m.key === 'dsa' ? h : Math.ceil((h * m.G) / m.H); }

/* ------------------------------------------------------------------------ */
/* Text. TeX strings are rendered to MathML at build time.                   */

const MASK_TEX = String.raw`A_h = \operatorname{softmax}\!\big(S_h + M\big), \qquad M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s > t \end{cases}`;

function overview(m, variant) { const o = overviewRaw(m, variant); return { ...o, tex: splitTex(o.tex) }; }
function overviewRaw(m, variant) {
  const block = String.raw`x \leftarrow x + \operatorname{Attn}\!\big(\operatorname{${m.norm === 'LayerNorm' ? 'LN' : 'RMSNorm'}}(x)\big)`;
  if (m.key === 'mha') return {
    lead: 'Every query head has its own key head and value head.',
    tex: [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_h^{\top}}{\sqrt{d_h}} + M\Big)\, V_h`],
    text: `GPT-2 projects each token into 12 queries, 12 keys and 12 values of 64 dimensions. Head h compares its queries only with its own keys, so the 12 heads can learn different patterns. The price is memory: every key and value head has to be cached while the model generates text.`,
    block, blockNote: `Attention is the first sublayer of each of the ${m.L} blocks; an MLP follows.`,
  };
  if (m.key === 'mqa') return {
    lead: 'All query heads share a single key head and value head.',
    tex: [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K^{\top}}{\sqrt{d_h}} + M\Big)\, V`],
    text: `Gemma 2B keeps 8 query heads of 256 dimensions but computes one key head and one value head that all of them read. The cache is 8× smaller than multi-head attention with the same heads. The cost is less capacity in the keys and values, which can lower quality.`,
    block, blockNote: `Attention is the first sublayer of each of the ${m.L} blocks; an MLP follows.`,
  };
  if (m.key === 'gqa') return {
    lead: 'Query heads form groups, and each group shares one key head and value head.',
    tex: [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_{g(h)}^{\top}}{\sqrt{d_h}} + M\Big)\, V_{g(h)}`],
    text: `Qwen2.5-0.5B splits its 14 query heads into 2 groups of 7. GQA covers the range between the other two: with G = H groups it is MHA, with G = 1 it is MQA. Here the cache is 7× smaller than MHA with the same heads.`,
    block, blockNote: `Attention is the first sublayer of each of the ${m.L} blocks; an MLP follows.`,
  };
  if (m.key === 'gdn') return {
    lead: 'Most layers replace the growing KV cache with fixed-size memory.',
    tex: [String.raw`S_t = \alpha_t\, S_{t-1}\big(I - \beta_t\, k_t k_t^{\top}\big) + \beta_t\, v_t k_t^{\top}`, String.raw`o_t = S_t\, q_t`],
    text: `Each Gated DeltaNet head keeps one 128 × 128 matrix S that maps keys to values. Every new token decays it by α (forget), overwrites what it stores under the token's key k with the token's value v (strength β), and then reads it with the token's query q. Qwen3-Next-80B-A3B has 36 such layers and 12 layers of gated softmax attention, so only a quarter of its layers keep a per-token cache.`,
    block: String.raw`x \leftarrow x + \operatorname{Mixer}_{\ell}\!\big(\operatorname{RMSNorm}(x)\big)`,
    blockNote: 'In every group of four layers, layers 1–3 use Gated DeltaNet and layer 4 gated attention; a mixture-of-experts layer follows each of them.',
  };
  if (m.key === 'dsa' && variant === 'qsa') return {
    lead: 'Qwen selects whole blocks of four tokens instead of single tokens.',
    tex: [String.raw`I_{t,b} = \frac{1}{\sqrt{128}} \sum_{j=1}^{4} \operatorname{ReLU}\big(q^{I}_{t,j} \cdot \bar{k}^{I}_{b}\big)`, String.raw`\bar{k}^{I}_{b} = \operatorname{Norm}\Big(\tfrac{1}{4} \textstyle\sum_{s=4b}^{4b+3} k^{I}_{s}\Big)`],
    text: `Qwen Sparse Attention (QSA, Qwen3.8-Flash-Next) averages the index keys of every four tokens into one block key and keeps whole blocks: up to 512 blocks, so 2,048 tokens, plus the 0–3 tokens of the current, unfinished block. The main attention is grouped-query attention with 24 query and 2 KV heads of 256. Only 12 of the model's 48 layers use QSA; the other 36 are Gated DeltaNet.`,
    block: String.raw`x \leftarrow x + \operatorname{QSA}\!\big(\operatorname{RMSNorm}(x)\big)`,
    blockNote: 'Every fourth layer of Qwen3.8-Flash-Next; the other three are Gated DeltaNet layers.',
  };
  if (m.key === 'dsa') return {
    lead: 'A cheap indexer scores every earlier token, and attention reads only the 2,048 best.',
    tex: [String.raw`I_{t,s} = \sum_{j=1}^{64} w_{t,j}\, \operatorname{ReLU}\big(q^{I}_{t,j} \cdot k^{I}_{s}\big)`, String.raw`u_t = \operatorname{Attn}\big(q_t,\ \{ c_s : I_{t,s} \in \operatorname{Top}_{2048}(I_{t,:}) \}\big)`],
    text: `DeepSeek-V3.2 keeps MLA (see that tab) and adds a lightning indexer to every layer. For each new token the indexer scores every earlier token with 64 small heads, ReLU instead of softmax, in FP8; the main attention then reads only the 2,048 highest-scoring tokens. Qwen's QSA, switchable above the diagram, applies the same idea to blocks of four tokens.`,
    block: String.raw`x \leftarrow x + \operatorname{SparseMLA}\!\big(\operatorname{RMSNorm}(x)\big)`,
    blockNote: 'Every one of the 61 blocks has its own indexer; blocks 1–3 use a dense MLP, blocks 4–61 a mixture of experts.',
  };
  return {
    lead: 'Keys and values are rebuilt from one small latent vector per token.',
    tex: [String.raw`K_h = \big[\, C^{KV} W^{UK}_h \;\; K^{R} \,\big], \qquad V_h = C^{KV} W^{UV}_h`],
    text: `DeepSeek-V2-Lite compresses each token's keys and values into a 512-dimensional latent, plus a 64-dimensional rotary key that all 16 heads share. Only these 576 numbers are cached. The per-head keys and values are expanded from the latent, and at inference that expansion can be folded into the query and output weights.`,
    block, blockNote: `Attention is the first sublayer of each of the ${m.L} blocks; a mixture-of-experts layer follows (a dense MLP in block 1).`,
  };
}

function stats(m) {
  if (m.key === 'gdn') return [
    ['Example model', m.model],
    ['Layers', `${m.L}: ${m.L - m.L / m.interval} GDN + ${m.L / m.interval} attention`],
    ['Width d', fmt(m.d)],
    ['Vocabulary', fmt(m.V)],
    ['Gated DeltaNet', `${m.lin.qk} qk · ${m.lin.v} v heads × ${m.lin.dk}`],
    ['Gated attention', `${m.H} query · ${m.G} KV heads × ${m.dh}`],
    ['Feed-forward', `MoE: ${m.shared} shared + ${m.active} of ${m.experts}`],
  ];
  const attn = m.key === 'dsa' ? `MLA, ${m.H} heads · top ${fmt(m.index.topk)} tokens`
    : m.key === 'mla' ? `${m.H} heads · latent ${m.dc} + ${m.dr}`
    : `${m.H} query · ${m.G} KV head${m.G > 1 ? 's' : ''} × ${m.dh}`;
  const ffn = m.ffnKind === 'moe'
    ? `MoE: ${m.shared} shared + ${m.active} of ${m.experts}`
    : `${m.ffnLabel} · ${fmt(m.dff)}`;
  return [
    ['Example model', m.model],
    ['Blocks', String(m.L)],
    ['Width d', fmt(m.d)],
    ['Vocabulary', fmt(m.V)],
    ['Attention', attn],
    ['Feed-forward', ffn],
    ['Positions', m.posLabel],
  ];
}

function cacheSentence(m) {
  return `At the full ${fmt(m.ctx)}-token context, one sequence needs ${bytes(cacheBytesPerToken(m) * m.ctx)} of cache in 16-bit precision${m.key === 'dsa' ? ' (index keys in FP8)' : ''}.`;
}

function stepsStandard(m) {
  const { d, H, G, dh, L } = m;
  const per = `${H} × ${dh}`;
  const kvName = m.key === 'mha' ? 'K_h' : m.key === 'mqa' ? 'K' : 'K_{g(h)}';
  const vName = m.key === 'mha' ? 'V_h' : m.key === 'mqa' ? 'V' : 'V_{g(h)}';
  const projNote = {
    mha: 'GPT-2 computes Q, K and V together with one fused 768 × 2304 matrix and a bias.',
    mqa: 'Gemma uses no bias in these projections.',
    gqa: 'Qwen2 adds a learned bias to the Q, K and V projections.',
  }[m.key];
  const kvText = {
    mha: `With G = H = ${H}, every query head gets a dedicated key head and value head.`,
    mqa: `With G = 1, one key head and one value head are computed. Each of W^K and W^V is ${fmt(d)} × ${dh}, an eighth the size of W^Q.`,
    gqa: `With G = ${G}, only ${G} key heads and ${G} value heads are computed. Query heads 1–7 read KV head 1; heads 8–14 read KV head 2.`,
  }[m.key];
  const scoreTex = m.key === 'gqa'
    ? String.raw`S_h = \frac{Q_h K_{g(h)}^{\top}}{\sqrt{d_h}}, \qquad g(h) = \left\lceil \frac{h\,G}{H} \right\rceil`
    : String.raw`S_h = \frac{Q_h ${kvName}^{\top}}{\sqrt{d_h}}`;
  const shareNote = {
    mha: '',
    mqa: ' All 8 heads score against the same keys, each with its own queries.',
    gqa: ' The 7 heads of a group score against the same keys, each with its own queries.',
  }[m.key];
  const pos = m.pos === 'learned' ? {
    tex: [String.raw`X_0 = E_{\text{token}} + E_{\text{position}}`],
    shape: `[T, ${d}] + [T, ${d}] → [T, ${d}], before block 1`,
    text: `GPT-2 has no rotary step inside attention. A learned position vector (one of ${fmt(m.ctx)}) is added to each token embedding before the first block, so queries and keys already carry position.`,
  } : {
    tex: [String.raw`\tilde{q}_t = R_t\, q_t, \quad \tilde{k}_s = R_s\, k_s \;\;\Rightarrow\;\; \tilde{q}_t^{\top} \tilde{k}_s = q_t^{\top} R_{s-t}\, k_s`],
    shape: `Q [T, ${H}, ${dh}], K [T, ${G}, ${dh}]: shapes unchanged`,
    text: `RoPE rotates each pair of dimensions in a query or key by an angle proportional to the token's position (base θ = ${m.theta}). The score then depends on position only through the offset s − t between the two tokens. Values are not rotated.`,
  };
  const cache = 2 * G * dh;
  return [
    { id: 'q', title: 'Project queries',
      tex: [String.raw`Q = X W^{Q}, \qquad W^{Q} \in \mathbb{R}^{d \times H d_h}`],
      shape: `X [T, ${d}] → Q [T, ${per}]`,
      text: `Each token's normalized hidden state is mapped to ${H} query vectors of ${dh} dimensions, one per head. ${projNote}` },
    { id: 'kv', title: 'Project keys and values',
      tex: [String.raw`K = X W^{K}, \quad V = X W^{V}, \qquad W^{K}\!, W^{V} \in \mathbb{R}^{d \times G d_h}`],
      shape: `X [T, ${d}] → K, V [T, ${G} × ${dh}]`,
      text: kvText },
    { id: 'pos', title: m.pos === 'learned' ? 'Position (added earlier)' : 'Rotate by position',
      ...pos },
    { id: 'score', title: 'Score every pair',
      tex: [scoreTex],
      shape: `[T, ${dh}] × [${dh}, T] → S_h [T, T], for each of ${H} heads`,
      text: `The dot product of query t with key s says how relevant token s is to token t. Dividing by √${dh} = ${Math.sqrt(dh)} keeps the scores in a range where softmax is not saturated.${shareNote}` },
    { id: 'mask', title: 'Mask and normalize',
      tex: [MASK_TEX],
      shape: 'S_h [T, T] → A_h [T, T], each row sums to 1',
      text: 'The causal mask removes future tokens, so each token can attend only to itself and earlier tokens. Softmax turns every row into non-negative weights that sum to 1.' },
    { id: 'mix', title: 'Mix the values',
      tex: [String.raw`O_h = A_h\, ${vName}`],
      shape: `[T, T] × [T, ${dh}] → O_h [T, ${dh}]`,
      text: 'Each output row is a weighted average of the value vectors that the token is allowed to see.' },
    { id: 'merge', title: 'Merge the heads',
      tex: [String.raw`Y = \big[\, O_1 \;\; O_2 \;\; \cdots \;\; O_H \,\big]\, W^{O}, \qquad W^{O} \in \mathbb{R}^{H d_h \times d}`],
      shape: `[T, ${H} · ${dh} = ${fmt(H * dh)}] → Y [T, ${fmt(d)}]`,
      text: 'The head outputs are concatenated and mixed by one linear projection. Y is then added to the residual stream.' },
    { id: 'cache', title: 'Cache for generation',
      tex: [String.raw`2\, G\, d_h = 2 \cdot ${G} \cdot ${dh} = ${cache.toLocaleString('en-US').replace(',', '{,}')}`],
      shape: `${fmt(cache)} values per layer × ${L} layers × 2 bytes = ${bytes(cacheBytesPerToken(m))} per token`,
      text: `While generating, each new token computes only its own query, key and value; the keys and values of earlier tokens are read back from the cache. ${cacheSentence(m)}` },
  ];
}

function stepsMLA(m) {
  const { d, H, dh, dr, dv, dc, L } = m;
  const cache = dc + dr;
  const mhaEq = 2 * H * dh;
  return [
    { id: 'q', title: 'Project queries',
      tex: [String.raw`Q_h = \big[\, Q^{C}_h \;\; Q^{R}_h \,\big] = X W^{Q}_h`],
      shape: `X [T, ${fmt(d)}] → Q [T, ${H} × (${dh} + ${dr})]`,
      text: `Each head's query has a ${dh}-dimensional content part and a ${dr}-dimensional rotary part. V2-Lite projects queries straight from X; the full DeepSeek-V2 also compresses queries, to a 1,536-dimensional latent.` },
    { id: 'kv', title: 'Compress keys and values',
      tex: [String.raw`C^{KV} = X W^{DKV}, \qquad W^{DKV} \in \mathbb{R}^{d \times d_c}`],
      shape: `X [T, ${fmt(d)}] → C^KV [T, ${dc}]`,
      text: `One ${dc}-dimensional latent per token stands in for the keys and values of all ${H} heads. This latent is what goes into the cache. It is normalized with RMSNorm before it is expanded.` },
    { id: 'pos', title: 'Add a shared rotary key',
      tex: [String.raw`K^{R} = \operatorname{RoPE}\!\big(X W^{KR}\big), \qquad Q^{R}_h \leftarrow \operatorname{RoPE}\!\big(Q^{R}_h\big)`],
      shape: `X [T, ${fmt(d)}] → K^R [T, ${dr}], one key shared by all ${H} heads`,
      text: `Position travels in a separate ${dr}-dimensional key that every head shares, the way MQA shares its key, and it is cached next to the latent. RoPE stays out of the latent path on purpose: a position-dependent rotation there would block the weight folding described in the last step.` },
    { id: 'expand', title: 'Expand per head',
      tex: [String.raw`K_h = \big[\, C^{KV} W^{UK}_h \;\; K^{R} \,\big], \qquad V_h = C^{KV} W^{UV}_h`],
      shape: `C^KV [T, ${dc}] → K_h [T, ${dh} + ${dr}], V_h [T, ${dv}], for each of ${H} heads`,
      text: `Up-projections rebuild a ${dh}-dimensional content key and a ${dv}-dimensional value for every head. Every head's key ends with the same shared rotary part.` },
    { id: 'score', title: 'Score every pair',
      tex: [String.raw`S_h = \frac{Q^{C}_h \big(K^{C}_h\big)^{\top} + Q^{R}_h \big(K^{R}\big)^{\top}}{\sqrt{d_h + d_R}}`],
      shape: `[T, ${dh + dr}] × [${dh + dr}, T] → S_h [T, T], for each of ${H} heads`,
      text: `The content and position terms are the two halves of one ${dh + dr}-dimensional dot product, scaled by 1/√${dh + dr}. DeepSeek's YaRN context extension multiplies this scale by a further constant.` },
    { id: 'mask', title: 'Mask and normalize',
      tex: [MASK_TEX],
      shape: 'S_h [T, T] → A_h [T, T], each row sums to 1',
      text: 'The causal mask removes future tokens, so each token can attend only to itself and earlier tokens. Softmax turns every row into non-negative weights that sum to 1.' },
    { id: 'merge', title: 'Mix values, merge heads',
      tex: [String.raw`O_h = A_h V_h, \qquad Y = \big[\, O_1 \;\; \cdots \;\; O_{${H}} \,\big]\, W^{O}`],
      shape: `[T, ${H} · ${dv} = ${fmt(H * dv)}] → Y [T, ${fmt(d)}]`,
      text: 'Values are mixed per head exactly as in multi-head attention, then concatenated and projected back to the model width.' },
    { id: 'cache', title: 'Cache the latent, fold the weights',
      tex: [
        String.raw`d_c + d_R = ${dc} + ${dr} = ${cache}`,
        String.raw`Q^{C}_h \big(C^{KV} W^{UK}_h\big)^{\top} = \big(Q^{C}_h \, {W^{UK}_h}^{\top}\big)\big(C^{KV}\big)^{\top}`,
      ],
      shape: `${cache} values per layer × ${L} layers × 2 bytes ≈ ${bytes(cacheBytesPerToken(m))} per token`,
      text: `Only C^KV and K^R are stored, ${(mhaEq / cache).toFixed(1)}× less than MHA with ${H} heads of ${dh} dimensions (${fmt(mhaEq)} values). Because matrix products can be regrouped, W^UK can be folded into the query side and W^UV into W^O, so inference can attend directly over cached latents without rebuilding K and V. ${cacheSentence(m)}` },
  ];
}

function stepsGDN(m) {
  const { qk, v, dk, dv } = m.lin, st = gdnState(m), nG = m.L - m.L / m.interval, nA = m.L / m.interval;
  return [
    { id: 'proj', title: 'Project q, k, v and the gates',
      tex: [String.raw`q_t = \hat{x}_t W^{Q}, \quad k_t = \hat{x}_t W^{K}, \quad v_t = \hat{x}_t W^{V}`, String.raw`a_t = \hat{x}_t W^{a}, \quad b_t = \hat{x}_t W^{b}, \quad z_t = \hat{x}_t W^{Z}`],
      shape: `x̂ [${fmt(m.d)}] → q, k [${qk} × ${dk}] · v, z [${v} × ${dv}] · a, b [${v}]`,
      text: `Every token gets ${qk} query and ${qk} key heads and ${v} value heads of ${dk} dimensions; each query/key head serves two value heads, the way GQA shares keys. a and b hold one number per value head and become the two gates; z gates the output.` },
    { id: 'conv', title: 'Mix in the previous three tokens',
      tex: [String.raw`\tilde{q}_t = \operatorname{SiLU}\Big(\sum_{i=0}^{3} w_i \odot q_{t-i}\Big) \quad \text{(same for } k, v\text{)}`, String.raw`\hat{q}_t = \frac{\tilde{q}_t}{\lVert \tilde{q}_t \rVert \sqrt{d_k}}, \qquad \hat{k}_t = \frac{\tilde{k}_t}{\lVert \tilde{k}_t \rVert}`],
      shape: `causal convolution, kernel ${m.lin.conv}, one filter per channel (${fmt(st.channels)} channels)`,
      text: 'A short causal convolution blends each channel with the same channel of the three previous tokens, followed by SiLU. Then q and k are scaled to unit length, and q by a further 1/√128. These layers use no RoPE: order enters through the convolution and the recurrence itself.' },
    { id: 'gates', title: 'Decay α and write strength β',
      tex: [String.raw`\alpha_t = \exp\!\big(-e^{A} \operatorname{softplus}(a_t + \delta)\big) \in (0, 1)`, String.raw`\beta_t = \sigma(b_t) \in (0, 1)`],
      shape: `one α and one β per value head and token: ${v} + ${v} numbers`,
      text: 'α decides how much of the memory survives this token: near 1 it remembers, near 0 it resets. β decides how strongly the token writes. A and δ are learned per head. Both gates depend on the token itself, so the model chooses when to forget; that is the "gated" in the name.' },
    { id: 'delta', title: 'Erase and write: the delta rule',
      tex: [String.raw`S_t = \alpha_t\, S_{t-1} \big(I - \beta_t\, \hat{k}_t \hat{k}_t^{\top}\big) + \beta_t\, v_t \hat{k}_t^{\top}`, String.raw`S_t \in \mathbb{R}^{d_v \times d_k}`],
      shape: `S: ${dv} × ${dk} per value head · ${v} × ${fmt(dk * dv)} = ${fmt(st.heads)} numbers per layer`,
      text: `S maps keys to values: S k is what the memory returns for key k. The update decays S by α, then replaces what S returns for k_t: the old value is erased in proportion β and v_t is written in its place. Without α this is one step of gradient descent on the error ‖S k_t − v_t‖², so the memory corrects itself instead of piling up every value.` },
    { id: 'read', title: 'Read the memory with q',
      tex: [String.raw`o_t = S_t\, \hat{q}_t`],
      shape: `S_t [${dv} × ${dk}] · q̂_t [${dk}] → o_t [${dv}], for each of ${v} value heads`,
      text: 'The query asks the memory which value is stored in its direction. No score is computed against earlier tokens one by one: everything this token can see has already been folded into S_t.' },
    { id: 'out', title: 'Normalize, gate, merge heads',
      tex: [String.raw`y_t = \big[\operatorname{RMSNorm}(o_{t,h}) \odot \operatorname{SiLU}(z_{t,h})\big]_{h=1}^{${v}}\, W^{O}`],
      shape: `[${v} × ${dv} = ${fmt(v * dv)}] → y_t [${fmt(m.d)}]`,
      text: 'Each head output is normalized and multiplied elementwise by SiLU of the output gate z; W^O then merges the 32 heads back to the model width.' },
    { id: 'cost', title: 'Constant cost per token',
      tex: [String.raw`\text{GDN: } O(d_k\, d_v) \text{ per head and token}`, String.raw`\text{softmax attention: } O(t\, d_h) \text{ at position } t`],
      shape: 'training: the sequence is cut into chunks that are processed in parallel',
      text: 'Generating a token touches only S, so its cost does not grow with the context. For training, the same outputs are computed for whole chunks of tokens with matrix multiplies (the chunkwise form), so the layer still trains in parallel like attention.' },
    { id: 'cache', title: 'Fixed-size cache',
      tex: [String.raw`${v} \cdot ${dk} \cdot ${dv} + 3 \cdot ${fmt(st.channels).replace(',', '{,}')} = ${fmt(st.total).replace(',', '{,}')}`],
      shape: `per GDN layer, whatever the length · × ${nG} layers ≈ ${bytes(st.total * nG * 2)} in 16-bit`,
      text: `A GDN layer keeps only its ${v} states and the last three inputs of the convolution. At the full ${fmt(m.ctx)}-token context the ${nA} gated-attention layers need ${bytes(cacheBytesPerToken(m) * m.ctx)} of KV cache, the ${nG} GDN layers still ${bytes(st.total * nG * 2)}.` },
  ];
}

function stepsSparse(m, variant) {
  if (variant === 'qsa') {
    const Q = QSA;
    return [
      { id: 'iq', title: 'Index queries: 4 small heads',
        tex: [String.raw`q^{I}_{t,j} \in \mathbb{R}^{${Q.dim}}, \qquad j = 1, \dots, ${Q.heads}`],
        shape: `${Q.heads} index query heads × ${Q.dim}, one shared index key head`,
        text: `QSA's indexer (${Q.model}) has ${Q.heads} query heads of ${Q.dim} dimensions and one key head that all of them share.` },
      { id: 'ik', title: 'One key per block of four tokens',
        tex: [String.raw`\bar{k}^{I}_{b} = \operatorname{Norm}\Big(\tfrac{1}{4} \textstyle\sum_{s=4b}^{4b+3} k^{I}_{s}\Big), \ \text{RoPE at position } 4b`],
        shape: 'four raw index keys → one cached block key',
        text: 'Every four index keys are averaged (in FP32), normalized and rotated with RoPE at the first position of the block. Only this compressed key is cached; the raw keys of the unfinished block wait in a four-slot buffer.' },
      { id: 'iscore', title: 'Score every complete block',
        tex: [String.raw`I_{t,b} = \frac{1}{\sqrt{${Q.dim}}} \sum_{j=1}^{${Q.heads}} \operatorname{ReLU}\big(q^{I}_{t,j} \cdot \bar{k}^{I}_{b}\big)`],
        shape: 'one score per complete block before the query',
        text: 'One score per block of four instead of one per token: a quarter of the comparisons, and no per-head weights.' },
      { id: 'select', title: 'Keep 512 blocks and the current block',
        tex: [String.raw`\mathcal{S}_t = \bigcup_{b\, \in\, \operatorname{Top}_{${Q.blocks}}(I_{t,:})} \{4b, \dots, 4b + 3\} \;\cup\; \text{current block}`],
        shape: `${Q.blocks} × ${Q.block} = ${fmt(Q.budget)} positions + 0–3 → at most ${fmt(Q.budget + 3)}`,
        text: 'The best blocks are expanded back to their four positions. The tokens of the current block, which is not complete yet, are always included. The diagram uses a toy budget of 2 blocks.' },
      { id: 'attend', title: 'Grouped-query attention over them',
        tex: [String.raw`\operatorname{head}_h = \sum_{s \in \mathcal{S}_t} \operatorname{softmax}_{s}\!\Big(\frac{q_{t,h} \cdot k_{s,g(h)}}{\sqrt{${Q.dh}}}\Big)\, v_{s,g(h)}`],
        shape: `${Q.H} query heads, ${Q.G} KV heads × ${Q.dh}, original keys and values`,
        text: 'The pooled keys only serve as the index. Attention itself uses the full, uncompressed keys and values of the selected positions.' },
      { id: 'merge', title: 'Gate and merge the heads',
        tex: [String.raw`Y = \big([\, O_1 \; \cdots \; O_{${Q.H}} \,] \odot \sigma(G)\big)\, W^{O}`],
        shape: `[T, ${Q.H} · ${Q.dh} = ${fmt(Q.H * Q.dh)}] → Y`,
        text: 'A sigmoid gate computed from the token scales the head outputs before W^O, as in the gated attention of Qwen3-Next.' },
      { id: 'cost', title: 'Hybrid: 12 of 48 layers',
        tex: [String.raw`48 = 12 \times \big(3\ \text{GDN} + 1\ \text{QSA}\big)`],
        shape: '36 Gated DeltaNet layers (fixed state) + 12 QSA layers (sparse, per-token cache)',
        text: `${Q.model} combines both newer ideas on this page: Gated DeltaNet in three of every four layers, and in the fourth a sparse attention layer that can still reach any earlier token.` },
      { id: 'cache', title: 'Cache',
        tex: [String.raw`2 \cdot ${Q.G} \cdot ${Q.dh} = 1{,}024 \ \text{per token} \; + \; \text{one index key per 4 tokens}`],
        shape: `per QSA layer; only the ${Q.attnLayers} QSA layers grow with the length`,
        text: 'Every token keeps its keys and values, because a later token may select any block. The index cache holds one pooled key per four tokens instead of one per token.' },
    ];
  }
  const { H, dh, dr, dv, dc, dq, index: I } = m;
  return [
    { id: 'iq', title: 'Index queries and head weights',
      tex: [String.raw`q^{I}_{t,j} = c^{Q}_t W^{IQ}_j, \qquad w_{t,j} = \hat{x}_t W^{w}_j, \qquad j = 1, \dots, ${I.heads}`],
      shape: `c^Q [${fmt(dq)}] → ${I.heads} × ${I.dim} · x̂ [${fmt(m.d)}] → ${I.heads} weights`,
      text: `The lightning indexer has its own small heads: ${I.heads} index queries of ${I.dim} dimensions, computed from the query latent c^Q that MLA already has, and one weight per index head. It runs in FP8.` },
    { id: 'ik', title: 'One index key per token',
      tex: [String.raw`k^{I}_s = \operatorname{LN}\big(\hat{x}_s W^{IK}\big) \in \mathbb{R}^{${I.dim}}`],
      shape: `x̂ [${fmt(m.d)}] → k^I [${I.dim}], shared by all ${I.heads} index heads, cached`,
      text: 'Each token stores one index key next to its MLA latent. All index heads compare against this same key, the multi-query layout, which keeps the extra cache small.' },
    { id: 'iscore', title: 'Score every earlier token',
      tex: [String.raw`I_{t,s} = \sum_{j=1}^{${I.heads}} w_{t,j} \cdot \operatorname{ReLU}\big(q^{I}_{t,j} \cdot k^{I}_{s}\big), \qquad s \le t`],
      shape: 'one score per earlier token',
      text: 'ReLU instead of softmax and FP8 arithmetic make every score cheap. The indexer still looks at all earlier tokens, so its cost grows with the context, but far more slowly than full attention would.' },
    { id: 'select', title: `Keep the top ${fmt(I.topk)}`,
      tex: [String.raw`\mathcal{S}_t = \operatorname{Top}_{k}\big(\{ I_{t,s} : s \le t \}\big), \qquad k = ${fmt(I.topk).replace(',', '{,}')}`],
      shape: `at most ${fmt(I.topk)} tokens per query; with fewer earlier tokens, all are kept`,
      text: `Each query keeps its own set of highest-scoring tokens. The diagram uses a toy budget of 8 of 30 tokens; with the real budget of ${fmt(I.topk)}, sparsity begins only after ${fmt(I.topk)} tokens of context.` },
    { id: 'attend', title: 'Attend over the selected tokens',
      tex: [String.raw`O_{t,h} = \sum_{s \in \mathcal{S}_t} \operatorname{softmax}_{s}\!\Big(\frac{q_{t,h} \cdot k_{s,h}}{\sqrt{d_h + d_R}}\Big)\, v_{s,h}`],
      shape: `${H} heads, all reading the same ≤ ${fmt(I.topk)} selected latents`,
      text: `The main attention is MLA, unchanged, except that unselected tokens are masked out like future tokens. Because MLA's keys and values all come from one latent per token, the ${H} heads share one selection and each selected latent is loaded once.` },
    { id: 'merge', title: 'Merge the heads',
      tex: [String.raw`Y = \big[\, O_1 \;\; \cdots \;\; O_{${H}} \,\big]\, W^{O}`],
      shape: `[T, ${H} · ${dv} = ${fmt(H * dv)}] → Y [T, ${fmt(m.d)}]`,
      text: 'Exactly as in multi-head attention: concatenate the head outputs, project back to the model width.' },
    { id: 'cost', title: 'What gets cheaper',
      tex: [String.raw`\text{main attention per query: } O(t) \;\to\; O(k)`, String.raw`\text{indexer: still } O(t), \text{ but light}`],
      shape: `at a 131,072-token context, ${fmt(I.topk)} of 131,072 latents are read (1.6%)`,
      text: 'Long prompts and long generations both get cheaper. DeepSeek reports that V3.2-Exp matches V3.1-Terminus on benchmarks while being substantially more efficient at long context.' },
    { id: 'cache', title: 'The cache keeps everything',
      tex: [String.raw`d_c + d_R + d_I = ${dc} + ${dr} + ${I.dim} = ${dc + dr + I.dim}`],
      shape: `values per token per layer (index key in FP8) × ${m.L} layers ≈ ${bytes(cacheBytesPerToken(m))} per token`,
      text: `Sparse attention does not shrink the cache: a later token may select any earlier one, so every latent and index key stays. What shrinks is how much of the cache each new token reads. ${cacheSentence(m)}` },
  ];
}

/* One equation per line: split at \qquad so nothing has to wrap mid-formula. */
const splitTex = list => list.flatMap(t => t.split(/\s*,?\s*\\qquad\s*/));
function steps(m, variant) {
  const list = m.key === 'mla' ? stepsMLA(m) : m.key === 'gdn' ? stepsGDN(m) : m.key === 'dsa' ? stepsSparse(m, variant) : stepsStandard(m);
  return list.map(s => ({ ...s, tex: splitTex(s.tex) }));
}

/* ------------------------------------------------------------------------ */
/* The whole model: one forward pass, tokens in, next token out.             */


function archOverview(m, view = {}) {
  const N = m.norm === 'LayerNorm' ? 'LN' : 'RMSNorm';
  const emb = m.pos === 'learned' ? String.raw`h_0 = E[x] + P[1{:}T]` : m.embScale ? String.raw`h_0 = \sqrt{d}\; E[x]` : String.raw`h_0 = E[x]`;
  const ffnName = view.moe ? 'MoE' : 'FFN';
  const last = String.raw`p = \operatorname{softmax}\!\big(\operatorname{${N}}_f(h_T)\, W_U\big)`;
  if (m.key === 'gdn') return {
    tex: [emb, String.raw`h \leftarrow h + \operatorname{Mixer}_{\ell}\!\big(\operatorname{${N}}(h)\big)`, String.raw`h \leftarrow h + \operatorname{MoE}\!\big(\operatorname{${N}}(h)\big)`, last],
    text: `${m.model} turns T tokens into a T × ${fmt(m.d)} residual stream h and refines it with ${m.L} layers. In every group of four, three mix tokens with Gated DeltaNet, a linear-attention layer with a fixed-size memory, and the fourth with gated softmax attention; every layer then applies a mixture of ${m.experts} experts. Switch between the two layer types above the diagram.`,
  };
  const what = m.key === 'mla' ? `${m.L} blocks whose feed-forward layers are mixtures of experts (except block 1)`
    : m.key === 'dsa' ? `${m.L} blocks with sparse attention and mixture-of-experts layers (dense in blocks 1–3)`
    : `${m.L} blocks of identical shape`;
  return {
    tex: [emb,
      String.raw`h \leftarrow h + \operatorname{${m.key === 'dsa' ? 'SparseAttn' : 'Attn'}}\!\big(\operatorname{${N}}(h)\big)`,
      String.raw`h \leftarrow h + \operatorname{${ffnName}}\!\big(\operatorname{${N}}(h)\big)`, last],
    text: `${m.model} turns T tokens into a T × ${fmt(m.d)} residual stream h, refines it with ${what}, and reads the next-token distribution off the last position. Attention is the only step where tokens exchange information; every other step acts on each token on its own.`,
  };
}

/* Decoding equations, one set per method (the card shows the selected one). */
const DECODE_TEX = {
  greedy: [String.raw`x_{T+1} = \arg\max_v \, p_v`, String.raw`p = \operatorname{softmax}(z_T)`],
  beam: [String.raw`\operatorname{score}(y_{1:n}) = \sum_{i=1}^{n} \log p\big(y_i \mid x,\, y_{<i}\big)`, String.raw`\text{keep the } B \text{ best prefixes after every step}`],
  temperature: [String.raw`p(\tau) = \operatorname{softmax}(z_T / \tau)`, String.raw`x_{T+1} \sim p(\tau)`],
  spec: [String.raw`\tilde{x} \sim q \ \text{(draft)}, \qquad \text{accept with prob. } \min\!\Big(1, \frac{p(\tilde{x})}{q(\tilde{x})}\Big)`, String.raw`\text{else } x \sim r, \quad r(v) = \frac{\max(0,\, p_v - q_v)}{\sum_u \max(0,\, p_u - q_u)}`, String.raw`\Rightarrow\ x_{T+1} \sim p \ \text{exactly}`],
  topk: [String.raw`p(\tau) = \operatorname{softmax}(z_T / \tau)`, String.raw`S = \text{the } k \text{ most likely tokens}`, String.raw`x_{T+1} \sim p(\tau) \text{ renormalized over } S`],
  topp: [String.raw`p(\tau) = \operatorname{softmax}(z_T / \tau)`, String.raw`S = \text{smallest top set with } \textstyle\sum_{v \in S} p_v \ge p`, String.raw`x_{T+1} \sim p(\tau) \text{ renormalized over } S`],
  minp: [String.raw`p(\tau) = \operatorname{softmax}(z_T / \tau)`, String.raw`S = \{ v : p_v \ge p_{\text{base}} \cdot \max_u p_u \}`, String.raw`x_{T+1} \sim p(\tau) \text{ renormalized over } S`],
  typical: [String.raw`H = -\textstyle\sum_v p_v \log p_v`, String.raw`S = \text{smallest set, in order of } \big|{-\log p_v} - H\big|, \text{ with } \textstyle\sum_{v \in S} p_v \ge m`, String.raw`x_{T+1} \sim p(\tau) \text{ renormalized over } S`],
};

/* view: { moe: bool, layer: 'gdn' | 'gattn' } */
function archSteps(m, view = {}) {
  const moe = !!view.moe, layer = m.key === 'gdn' ? (view.layer || 'gdn') : null;
  const { d, L, V } = m;
  const N = m.norm === 'LayerNorm' ? 'LN' : 'RMSNorm';
  const D = fmt(d), Vt = fmt(V);
  const tokText = {
    mha: `A byte-level BPE tokenizer splits the text into pieces from a vocabulary of ${Vt} and replaces each piece with its ID. Common words are one token; rare words split into several. A leading space belongs to the token that follows it.`,
    mqa: `A SentencePiece tokenizer with ${Vt} entries splits the text and maps each piece to an ID. Gemma's tokenizer also puts a beginning-of-sequence token in front.`,
    gqa: `A byte-level BPE tokenizer maps the text to IDs. The embedding matrix has ${Vt} rows, slightly more than the tokenizer uses, padded for efficient matrix shapes.`,
    mla: `A byte-level BPE tokenizer with about 100K entries maps the text to IDs; the embedding matrix is padded to ${Vt} rows. DeepSeek's tokenizer also puts a beginning-of-sequence token in front.`,
    gdn: `Qwen3-Next uses the same byte-level BPE tokenizer as Qwen2.5 and Qwen3, so the IDs match the GQA tab. The embedding matrix has ${Vt} rows, padded beyond the tokenizer's entries.`,
    dsa: `A byte-level BPE tokenizer with about 128K entries maps the text to IDs; the embedding matrix has ${Vt} rows. DeepSeek's tokenizer puts a beginning-of-sequence token in front.`,
  }[m.key];
  const embTex = m.pos === 'learned'
    ? [String.raw`h_0 = E[x] + P[1{:}T]`, String.raw`E \in \mathbb{R}^{V \times d}, \;\; P \in \mathbb{R}^{${m.ctx} \times d}`]
    : m.embScale ? [String.raw`h_0 = \sqrt{d}\; E[x]`, String.raw`E \in \mathbb{R}^{V \times d}`]
    : [String.raw`h_0 = E[x]`, String.raw`E \in \mathbb{R}^{V \times d}`];
  const embText = m.pos === 'learned'
    ? `Each ID selects one row of the learned embedding matrix E (${Vt} × ${D}). GPT-2 then adds a learned position vector for each position, the only place it encodes order explicitly.`
    : m.embScale
    ? `Each ID selects one row of E (${Vt} × ${D}). Gemma multiplies the result by √${d} ≈ ${Math.sqrt(d).toFixed(1)}. No position is added here: RoPE handles order inside attention.`
    : `Each ID selects one row of E (${Vt} × ${D}). No position is added here: ${m.key === 'mla' || m.key === 'dsa' ? 'the decoupled rotary key handles order inside attention' : m.key === 'gdn' ? 'Gated DeltaNet layers see order through their recurrence, attention layers through RoPE' : 'RoPE handles order inside attention'}.`;
  const normTexS = m.norm === 'LayerNorm'
    ? String.raw`\operatorname{LN}(h_t) = \gamma \odot \frac{h_t - \mu_t}{\sqrt{\sigma_t^2 + \varepsilon}} + \beta`
    : String.raw`\operatorname{RMSNorm}(h_t) = ${m.normPlusOne ? '(1 + \\gamma)' : '\\gamma'} \odot \frac{h_t}{\sqrt{\frac{1}{d}\sum_i h_{t,i}^2 + \varepsilon}}`;
  const normText = m.norm === 'LayerNorm'
    ? `Pre-norm: each token's vector is rescaled before it enters the sublayer, while the stream itself stays unnormalized. μ and σ² are the mean and variance over the ${D} features of one token; γ and β are learned.`
    : `Pre-norm: each token's vector is rescaled before it enters the sublayer, while the stream itself stays unnormalized. RMSNorm divides by the root mean square, with no mean subtraction and no bias.${m.normPlusOne ? ` ${m.key === 'gdn' ? 'Qwen3-Next' : 'Gemma'} stores the scale as 1 + γ.` : ''}`;
  const mixName = layer === 'gdn' ? 'GDN' : m.key === 'dsa' ? 'SparseAttn' : 'Attn';

  /* ---- mixer steps: what replaces "q · k · v → RoPE → Attend" ---- */
  let mixer;
  if (layer === 'gdn') {
    const { qk, v, dk, dv } = m.lin, st = gdnState(m);
    mixer = [
      { id: 'qkv', chip: 'q · k · v', title: 'Give every token its own q, k, v and gates',
        tex: [String.raw`q_t = \hat{x}_t W^{Q}, \quad k_t = \hat{x}_t W^{K}, \quad v_t = \hat{x}_t W^{V}`, String.raw`\alpha_t, \beta_t \text{ from } \hat{x}_t W^{a},\ \hat{x}_t W^{b}, \qquad z_t = \hat{x}_t W^{Z}`],
        shape: `x̂ [${D}] per token → q, k [${qk} × ${dk}], v [${v} × ${dv}], α, β [${v}]`,
        text: `As in attention, every token gets its own query, key and value: ${qk} query and key heads and ${v} value heads of ${dk}, each query/key head serving two value heads. It also gets two gates per value head, a decay α and a write strength β, and an output gate z.` },
      { id: 'conv', chip: 'Conv', title: 'Short convolution, unit-length q and k',
        tex: [String.raw`\tilde{q}_t = \operatorname{SiLU}\Big(\sum_{i=0}^{3} w_i \odot q_{t-i}\Big), \ \text{same for } k, v`, String.raw`\hat{q}_t = \tilde{q}_t / \lVert \tilde{q}_t \rVert, \qquad \hat{k}_t = \tilde{k}_t / \lVert \tilde{k}_t \rVert`],
        shape: `kernel ${m.lin.conv}: token t sees its own and the 3 previous rows, per channel`,
        text: 'A short causal convolution mixes each channel of q, k and v with the same channel of the three previous tokens. q and k are then scaled to unit length. This layer has no RoPE: order comes from the convolution and from the recurrence in the next step.' },
      { id: 'state', chip: 'Memory S', title: 'Update the memory, token by token',
        tex: [String.raw`S_t = \alpha_t\, S_{t-1} \big(I - \beta_t\, \hat{k}_t \hat{k}_t^{\top}\big) + \beta_t\, v_t \hat{k}_t^{\top}`, String.raw`S_t \in \mathbb{R}^{${dv} \times ${dk}} \text{ per value head}`],
        shape: `S_0 = 0 → S_1 → … → S_T, one ${dv} × ${dk} matrix per head, the same size at every step`,
        text: `Where attention keeps all keys and values and builds a T × T matrix, a GDN head keeps one matrix S that maps keys to values and updates it once per token: decay by α, erase what S stored under k_t, write v_t there with strength β (the delta rule). Row t of the column is the state after token t; arrows show that each state is built from the previous one.` },
      { id: 'attn', chip: 'Read out', title: 'Read the memory, gate, project',
        tex: [String.raw`o_t = S_t\, \hat{q}_t`, String.raw`y_t = \big[\operatorname{RMSNorm}(o_{t,h}) \odot \operatorname{SiLU}(z_{t,h})\big]_{h}\, W^{O}`],
        shape: `o_t [${v} × ${dv}] → y_t [${D}]`,
        text: `Each token's query reads the state that includes it: o_t = S_t q_t. The ${v} head outputs are normalized, multiplied by SiLU of the output gate and merged by W^O. Only S_T and the last three convolution inputs have to be kept for generation: ${fmt(st.total)} numbers per layer, however long the text.`, more: true },
    ];
  } else if (m.key === 'dsa') {
    const I = m.index;
    mixer = [
      { id: 'qkv', chip: 'q · c · k^I', title: 'Queries, latents and index keys',
        tex: [String.raw`q_t = \operatorname{RMSNorm}\big(\hat{x}_t W^{DQ}\big) W^{UQ}, \quad c^{KV}_t = \hat{x}_t W^{DKV}, \quad k^{R}_t = \operatorname{RoPE}\big(\hat{x}_t W^{KR}\big)`, String.raw`q^{I}_{t,j} = c^{Q}_t W^{IQ}_j, \quad w_{t,j} = \hat{x}_t W^{w}_j, \quad k^{I}_t = \operatorname{LN}\big(\hat{x}_t W^{IK}\big)`],
        shape: `x̂ [${D}] → q [${m.H} × (${m.dh}+${m.dr})] via c^Q [${fmt(m.dq)}]; c [${m.dc}], k^R [${m.dr}]; k^I [${I.dim}]`,
        text: `The attention is MLA, at a larger size than on the MLA tab: ${m.H} heads, and queries compressed through a ${fmt(m.dq)}-dim latent c^Q. The lightning indexer adds its own small projections: ${I.heads} index queries of ${I.dim} from c^Q, ${I.heads} head weights, and one ${I.dim}-dim index key k^I per token, cached next to the latent.` },
      { id: 'rope', chip: 'Decoupled RoPE', title: 'Decoupled RoPE',
        tex: [String.raw`\tilde{q}^{R}_{t,h} = R_t\, q^{R}_{t,h}, \qquad \tilde{k}^{R}_{t} = R_t\, k^{R}_{t}`, String.raw`R_t: \text{ pair } i \text{ turns by } t\,\theta_i, \quad \theta_i = 10000^{-2i/${m.dr}}\ \text{(YaRN-scaled)}`],
        shape: `q^R: ${m.H} heads × ${m.dr} and the shared k^R [${m.dr}] rotate; c [${m.dc}] does not`,
        text: `As in MLA, only the ${m.dr}-dim rotary parts carry position, so the latent stays position-free. YaRN slows the lowest frequencies by ${m.yarn.factor}× for the ${fmt(m.ctx)}-token context. The indexer's queries and keys are also rotated, on ${m.dr} of their ${I.dim} dimensions.` },
      { id: 'index', chip: 'Indexer', title: `Lightning indexer: keep the top ${fmt(I.topk)}`,
        tex: [String.raw`I_{t,s} = \sum_{j=1}^{${I.heads}} w_{t,j}\, \operatorname{ReLU}\big(q^{I}_{t,j} \cdot k^{I}_{s}\big), \qquad s \le t`, String.raw`\mathcal{S}_t = \operatorname{Top}_{${fmt(I.topk).replace(',', '{,}')}}\big(I_{t,:}\big)`],
        shape: `one score per earlier token → at most ${fmt(I.topk)} selected tokens per query`,
        text: `For each query the indexer scores every earlier token with ${I.heads} cheap FP8 heads and keeps the ${fmt(I.topk)} best. This sentence has only ${TOKENS.models[m.key].ids.length} tokens, so nothing would be dropped; the diagram uses a toy budget of 4 to show the selection. The bars next to A score the earlier tokens for the highlighted row.` },
      { id: 'attn', chip: 'Attend', title: 'Attend over the selected tokens',
        tex: [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_h^{\top}}{\sqrt{d_h + d_R}} + M + M^{\text{sel}}\Big)\, V_h`],
        shape: `[T, ${D}] → ${m.H} heads over ≤ ${fmt(I.topk)} selected latents each → [T, ${D}]`,
        text: `MLA as usual, except that the selection mask M^sel hides every token the indexer did not keep, exactly like the causal mask hides future tokens. All ${m.H} heads share one selection. The latents, rotary keys and index keys of every token stay in the cache, because a later query may pick any of them.`, more: true },
    ];
  } else {
    const gated = layer === 'gattn';
    const ov = overview(m);
    const attnTex = m.key === 'mla'
      ? [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_h^{\top}}{\sqrt{d_h + d_R}} + M\Big)\, V_h`]
      : gated ? [String.raw`\operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_{g(h)}^{\top}}{\sqrt{d_h}} + M\Big)\, V_{g(h)}`, String.raw`Y = \big([\operatorname{head}_1 \cdots \operatorname{head}_{${m.H}}] \odot \sigma(G)\big)\, W^{O}`]
      : ov.tex;
    const attnShape = m.key === 'mla'
      ? `[T, ${D}] → ${m.H} heads, keys and values rebuilt from a ${m.dc}-dim latent → [T, ${D}]`
      : `[T, ${D}] → ${m.H} query heads, ${m.G} key/value head${m.G > 1 ? 's' : ''} of ${m.dh} → [T, ${D}]`;
    const posNote = m.pos === 'learned' ? ' Position is already inside h from the embedding step.' : m.key === 'mla' ? '' : ' Then q and k (not v) are rotated by the token\'s position (RoPE).';
    const qkvTex = m.key === 'mla'
      ? [String.raw`q_t = \hat{x}_t W^{Q}, \quad c^{KV}_t = \hat{x}_t W^{DKV}, \quad k^{R}_t = \operatorname{RoPE}\big(\hat{x}_t W^{KR}\big)`]
      : gated ? [String.raw`q_t = \operatorname{RMSNorm}\big(\hat{x}_t W^{Q}\big), \quad k_t = \operatorname{RMSNorm}\big(\hat{x}_t W^{K}\big), \quad v_t = \hat{x}_t W^{V}`, String.raw`g_t = \hat{x}_t W^{G} \quad \text{(output gate)}`]
      : [String.raw`q_t = \hat{x}_t W^{Q}, \quad k_t = \hat{x}_t W^{K}, \quad v_t = \hat{x}_t W^{V}`,
         String.raw`Q = \hat{X} W^{Q}, \quad K = \hat{X} W^{K}, \quad V = \hat{X} W^{V}`];
    const qkvShape = m.key === 'mla'
      ? `x̂ [${D}] per token → q_t [${m.H} × (${m.dh}+${m.dr})], c_t [${m.dc}], k_t^R [${m.dr}]`
      : `x̂ [${D}] per token → q_t [${m.H} × ${m.dh}], k_t and v_t [${m.G} × ${m.dh}]${gated ? `, g_t [${m.H} × ${m.dh}]` : ''}`;
    const qkvText = {
      mha: `Every token's vector x̂ is multiplied by the same three matrices, so each token gets its own query (what it looks for), key (what it can be found by) and value (what it passes on). Doing this for all rows at once is one matrix multiply that yields Q, K and V. In MHA each token has 12 query, 12 key and 12 value heads.`,
      mqa: `Every token's vector x̂ is multiplied by the same three matrices, so each token gets its own query, key and value. Here a token has 8 query heads but only one key head and one value head. Doing all rows at once gives Q, K and V.`,
      gqa: `Every token's vector x̂ is multiplied by the same three matrices, so each token gets its own query, key and value. Here a token has 14 query heads but only 2 key and 2 value heads, each shared by a group of 7 query heads. Doing all rows at once gives Q, K and V.`,
      mla: `Every token's vector x̂ gives that token its own 16 query heads. Instead of per-head keys and values, it gets one ${m.dc}-dim latent c_t and one ${m.dr}-dim rotary key k_t^R; the per-head keys and values are expanded from c_t when attention needs them.`,
      gdn: `Every fourth layer of Qwen3-Next is softmax attention: each token gets 16 query heads and 2 key and value heads of 256 (GQA), with an RMSNorm on every query and key head. The same projection that makes q also makes a gate g for every output channel.`,
    }[m.key] + posNote;
    const ropeStep = m.pos === 'learned' ? {
      chip: 'Position', title: 'Position: already in h',
      tex: [String.raw`h_0 = E[x] + P[1{:}T]`, String.raw`q_t = \hat{x}_t W^{Q}, \quad k_s = \hat{x}_s W^{K}`],
      shape: 'no rotation inside attention',
      text: 'GPT-2 has no RoPE. Each token carries its learned position vector from the embedding step, so its query and key already depend on where it stands, and attention sees absolute positions mixed into the content.' }
    : m.key === 'mla' ? {
      chip: 'Decoupled RoPE', title: 'Decoupled RoPE: rotate only the rotary parts',
      tex: [String.raw`\tilde{q}^{R}_{t,h} = R_t\, q^{R}_{t,h}, \qquad \tilde{k}^{R}_{t} = R_t\, k^{R}_{t}`,
        String.raw`R_t: \text{ pair } i \text{ turns by } t\,\theta_i, \quad \theta_i = 10000^{-2i/${m.dr}}\ \text{(YaRN-scaled)}`,
        String.raw`K_h = \big[\, C^{KV} W^{UK}_h \;\; \tilde{K}^{R} \,\big]`],
      shape: `q^R: ${m.H} heads × ${m.dr} and the shared k^R [${m.dr}] rotate; c [${m.dc}] does not`,
      text: `Only the ${m.dr}-dim rotary parts carry position: each head's q^R and the one shared k^R are rotated by the token's position. The latent c is never rotated. That separation is the point of "decoupled": a position-dependent rotation between c and W^UK would make it impossible to fold W^UK into the queries at inference. YaRN slows the lowest frequencies by 40× to extend the context. The dials show the fastest pair, which turns 1 radian per position.` }
    : gated ? {
      chip: 'Partial RoPE', title: 'Rotate a quarter of every q and k head',
      tex: [String.raw`\tilde{q}_t = \big[\, R_t\, q_t^{[1:64]} \;\; q_t^{[65:256]} \,\big], \quad \text{same for } k`, String.raw`\theta_i = 10^{7 \cdot (-2i/64)}, \quad i = 0, \dots, 31`],
      shape: `rotates ${m.ropeDim} of ${m.dh} dims per head (partial_rotary_factor 0.25); v unchanged`,
      text: `Qwen3-Next rotates only the first ${m.ropeDim} of each head's ${m.dh} dimensions, with base θ = 10,000,000; the other ${m.dh - m.ropeDim} carry no position. Only these 12 attention layers use RoPE at all. The dials show the fastest pair, 1 radian per position.` }
    : {
      chip: 'RoPE', title: 'Rotate q and k by position (RoPE)',
      tex: [String.raw`\tilde{q}_t = R_t\, q_t, \qquad \tilde{k}_s = R_s\, k_s`,
        String.raw`R_t: \text{ pair } i \text{ turns by } t\,\theta_i, \quad \theta_i = ${m.thetaN === 10000 ? '10000' : '1{,}000{,}000'}^{-2i/${m.dh}}`,
        String.raw`\tilde{q}_t^{\top} \tilde{k}_s = q_t^{\top} R_{s-t}\, k_s`],
      shape: `every q and k head [${m.dh}] = ${m.dh / 2} rotated pairs; v unchanged`,
      text: `Every query and key head of a token is rotated pairwise by angles that grow with its position t; pair i turns by t·θ_i, from 1 radian per position for the first pair to almost nothing for the last. Because rotations compose, the score between positions t and s only depends on their offset s − t. Values are not rotated. The dials show the fastest pair. (${m.model} uses plain RoPE; the decoupled variant is MLA's.)` };
    mixer = [
      { id: 'qkv', chip: 'q · k · v', title: 'Give every token its own q, k and v', tex: qkvTex, shape: qkvShape, text: qkvText },
      { id: 'rope', ...ropeStep },
      { id: 'attn', chip: 'Attend', title: `Attend (${gated ? 'gated GQA' : m.abbr})`, tex: attnTex, shape: attnShape,
        text: `Row t of A holds how much token t draws from each token s ≤ t: query t is compared with every allowed key, the scores are scaled and softmaxed, and the weights mix the value vectors. The ${m.H} heads do this in parallel and W^O merges them back to width ${D}.${gated ? ' Before W^O, a sigmoid gate computed from the token itself scales every output channel; Qwen adds it for training stability.' : ''} ${m.key === 'mla' ? 'The latents and rotary keys' : 'The keys and values'} of every token are what the KV cache stores.`, more: true },
    ];
  }

  /* ---- feed-forward ---- */
  let ffn;
  const silu = String.raw`\operatorname{SiLU}(x) \;=\; x\,\sigma(x) \;=\; \frac{x}{1 + e^{-x}}`;
  const gelu = String.raw`\operatorname{GELU}(x) \;=\; x\,\Phi(x) \;\approx\; \tfrac{x}{2}\Big(1 + \tanh\!\Big(\sqrt{2/\pi}\,\big(x + 0.044715\,x^{3}\big)\Big)\Big)`;
  const expertTex = String.raw`F_i(u) = \big(\operatorname{SiLU}(u W^{i}_{\text{gate}}) \odot u W^{i}_{\text{up}}\big) W^{i}_{\text{down}}`;
  if (m.ffnKind === 'mlp') ffn = {
    tex: [String.raw`\operatorname{FFN}(u) = \operatorname{GELU}(u W_1 + b_1)\, W_2 + b_2`, gelu],
    shape: `[T, ${D}] → [T, ${fmt(m.dff)}] → [T, ${D}]`,
    text: `Two linear layers with a GELU between them, applied to every token separately; the hidden layer is ${m.dff / d}× wider than the stream. GELU weights x by Φ(x), the probability that a standard normal value is below x; GPT-2 uses the tanh approximation shown.` };
  else if (m.ffnKind === 'glu' || !moe) ffn = {
    tex: [String.raw`\operatorname{FFN}(u) = \big(\operatorname{${m.act}}(u W_{\text{gate}}) \odot u W_{\text{up}}\big)\, W_{\text{down}}`, m.act === 'GELU' ? gelu : silu],
    shape: `[T, ${D}] → 2 × [T, ${fmt(m.dff)}] → [T, ${D}]`,
    text: `A gated MLP (${m.act === 'GELU' ? 'GeGLU' : 'SwiGLU'}): one projection passes through ${m.act} and multiplies the other elementwise before the down-projection (σ is the logistic sigmoid${m.act === 'GELU' ? ', Φ the standard normal CDF' : ''}). Applied to every token separately; the hidden layer is ${(m.dff / d).toFixed(1).replace('.0', '')}× wider than the stream.${m.key === 'mla' ? ` In DeepSeek-V2-Lite only block 1 is dense like this; blocks 2–${L} use experts.` : m.key === 'dsa' ? ` In DeepSeek-V3.2 only blocks 1–3 are dense like this; blocks 4–${L} use experts.` : ''}` };
  else if (m.key === 'gdn') ffn = {
    tex: [String.raw`\operatorname{MoE}(u) = \sigma(u\, w_{\text{sh}})\, F^{\text{shared}}(u) + \sum_{i \in \operatorname{Top}_{${m.active}}(s)} \frac{s_i}{\sum_{j \in \operatorname{Top}_{${m.active}}} s_j}\, F_i(u)`,
      String.raw`s = \operatorname{softmax}\big(u\, W_{\text{router}}\big) \in \mathbb{R}^{${m.experts}}`, expertTex, silu],
    shape: `[T, ${D}] → each expert ${D} → ${m.dExpert} → ${D}; 1 shared + ${m.active} of ${m.experts} routed per token`,
    text: `A router scores all ${m.experts} experts for each token; the ${m.active} best run, with their scores renormalized to sum to 1, plus one shared expert whose output is scaled by a learned sigmoid gate. Each expert is a SwiGLU MLP with a hidden size of only ${m.dExpert}. Every one of the ${L} layers uses this block, and it is the main reason only about 3B of the 80B parameters are used per token.` };
  else if (m.key === 'dsa') ffn = {
    tex: [String.raw`\operatorname{MoE}(u) = F^{\text{shared}}(u) + \sum_{i \in \operatorname{Top}_{${m.active}}(s + b)} 2.5\, \frac{s_i}{\sum_{j} s_j}\, F_i(u)`,
      String.raw`s = \sigma\big(u\, W_{\text{router}}\big) \in \mathbb{R}^{${m.experts}}`, expertTex, silu],
    shape: `[T, ${D}] → each expert ${D} → ${fmt(m.dExpert)} → ${D}; 1 shared + ${m.active} of ${m.experts} routed per token`,
    text: `A router scores ${m.experts} experts with a sigmoid; a per-expert bias b, adjusted during training to balance the load, decides which ${m.active} run, but only the scores weight their outputs (normalized, times 2.5). Routing is limited to 4 of 8 expert groups. One shared expert always runs. Blocks 1–3 use a dense MLP instead.` };
  else ffn = {
    tex: [String.raw`\operatorname{MoE}(u) = \sum_{i=1}^{${m.shared}} F^{\text{shared}}_i(u) + \sum_{i=1}^{${m.experts}} g_i\, F^{\text{routed}}_i(u)`,
      String.raw`g_i = \begin{cases} s_i & s_i \in \operatorname{Top}_{${m.active}}(s) \\ 0 & \text{otherwise} \end{cases} \qquad s = \operatorname{softmax}\big(u\, W_{\text{router}}\big)`,
      expertTex, silu],
    shape: `[T, ${D}] → each expert ${D} → ${fmt(m.dExpert)} → ${D}; ${m.shared} shared + ${m.active} of ${m.experts} routed per token`,
    text: `A router scores all ${m.experts} routed experts for each token; only the top ${m.active} run, weighted by their scores, and the ${m.shared} shared experts always run. Each expert is a small SwiGLU MLP. This is the main reason only 2.4B of the 15.7B parameters are used per token.` };
  const moeName = moe ? 'MoE' : 'FFN';
  const repeatText = m.key === 'gdn'
    ? `Layers 1–3 of every group of four are Gated DeltaNet layers, layer 4 is gated attention: 36 + 12 = ${L}. Each has its own weights; the stream never changes shape.`
    : `Every block has the same structure and its own weights; the stream never changes shape.${m.key === 'mla' ? ' In DeepSeek-V2-Lite block 1 uses a dense feed-forward layer and blocks 2–27 use experts.' : m.key === 'dsa' ? ' Blocks 1–3 use a dense feed-forward layer, blocks 4–61 experts; every block has its own indexer.' : ''}`;
  return [
    { id: 'tok', chip: 'Tokenize', title: 'Tokenize',
      tex: [String.raw`x = (x_1, \dots, x_T), \qquad x_t \in \{0, \dots, V - 1\}`],
      shape: `text → [T] token IDs, V = ${Vt}`, text: tokText },
    { id: 'emb', chip: 'Embed', title: 'Embed: look up one row per token', tex: embTex,
      shape: `[T] → h_0 [T, ${D}]`, text: embText },
    { id: 'stream', chip: 'Stream h', title: 'The residual stream h',
      tex: [String.raw`h \in \mathbb{R}^{T \times d}, \qquad h \leftarrow h_0`,
        String.raw`h \leftarrow h + \operatorname{${mixName}}\!\big(\operatorname{${N}}(h)\big)`,
        String.raw`h \leftarrow h + \operatorname{${moe ? 'MoE' : 'FFN'}}\!\big(\operatorname{${N}}(h)\big)`],
      shape: `h: [T, ${D}], one row per token, the same shape in every block`,
      text: `h holds one ${D}-number vector per token, one row each. It starts as the embeddings h_0, runs along the top of every block, and ends at the LM head. Each sublayer reads a normalized copy of h and adds its output back; nothing ever overwrites it. That is why it is called a stream (information flows along it from input to output) and residual (sublayers only add corrections). It also gives gradients a direct path through all ${L} blocks.` },
    { id: 'norm1', chip: 'Norm', title: `Normalize (${m.norm})`, tex: [normTexS],
      shape: `[T, ${D}] → [T, ${D}], ε = ${m.epsText}`, text: normText },
    ...mixer,
    { id: 'res1', chip: 'Add', title: 'Add to the stream',
      tex: [String.raw`h \leftarrow h + \operatorname{${mixName}}\!\big(\operatorname{${N}}(h)\big)`],
      shape: `[T, ${D}] + [T, ${D}] → [T, ${D}]`,
      text: `The sublayer's output is added to the stream instead of replacing it. Each block makes an incremental edit, and gradients get a direct path through all ${L} blocks.` },
    { id: 'norm2', chip: 'Norm', title: `Normalize again`, tex: [normTexS],
      shape: `[T, ${D}] → [T, ${D}]`, text: `A second ${m.norm} with its own learned scale${m.norm === 'LayerNorm' ? ' and bias' : ''} prepares the stream for the feed-forward sublayer.` },
    { id: 'ffn', chip: moe ? 'Experts' : 'Feed-forward', title: moe ? 'Mixture of experts' : 'Feed-forward', ...ffn },
    { id: 'res2', chip: 'Add', title: 'Add to the stream',
      tex: [String.raw`h \leftarrow h + \operatorname{${moeName}}\!\big(\operatorname{${N}}(h)\big)`],
      shape: `[T, ${D}] + [T, ${D}] → [T, ${D}]`,
      text: 'The second residual addition completes the block.' },
    { id: 'repeat', chip: `Repeat ${L} ${m.key === 'gdn' ? 'layers' : 'blocks'}`, title: `Repeat for ${L} ${m.key === 'gdn' ? 'layers' : 'blocks'}`,
      tex: [String.raw`h_{\ell} = \operatorname{Block}_{\ell}\big(h_{\ell-1}\big), \qquad \ell = 1, \dots, ${L}`],
      shape: `[T, ${D}] in and out of every block`, text: repeatText },
    { id: 'fnorm', chip: 'Final norm', title: 'Final normalization',
      tex: [String.raw`\bar{h} = \operatorname{${N}}_f\big(h_{${L}}\big)`],
      shape: `[T, ${D}] → [T, ${D}]`,
      text: 'One more normalization after the last block, with its own weights.' },
    { id: 'head', chip: 'LM head', title: 'Score the vocabulary',
      tex: m.tied ? [String.raw`z = \bar{h}\, W_U`, String.raw`W_U = E^{\top} \in \mathbb{R}^{d \times V}`]
                  : [String.raw`z = \bar{h}\, W_U`, String.raw`W_U \in \mathbb{R}^{d \times V}`],
      shape: `[T, ${D}] → logits [T, ${Vt}]`,
      text: `A linear map gives every vocabulary entry a score at every position. ${m.tied ? `${m.model} reuses the embedding matrix here (tied weights).` : `${m.model} uses a separate output matrix (untied).`} Training uses all T rows; generation only needs the last one.` },
    { id: 'dist', chip: 'Logits → p', title: 'From logits to probabilities',
      tex: [String.raw`p_v(\tau) = \frac{e^{z_{T,v} / \tau}}{\sum_{u=1}^{V} e^{z_{T,u} / \tau}}`, String.raw`\textstyle\sum_v p_v = 1, \qquad p_v > 0`],
      shape: `logits z_T [${Vt}] → probabilities p [${Vt}]`,
      text: `The last row of logits holds one unbounded score per vocabulary entry. Softmax exponentiates and normalizes them, so every token gets a probability and they sum to 1; only differences between logits matter. Dividing by a temperature τ first makes p sharper (τ < 1) or flatter (τ > 1). The panel lists the most likely candidates; the logits are illustrative values for real ${m.model} tokens, since this page runs no model.`, dist: true },
    { id: 'decode', chip: 'Decode', title: 'Decode: choose the next token',
      tex: DECODE_TEX.temperature,
      shape: 'choose → append → run again (earlier tokens come from the cache)',
      text: `Search methods are deterministic: greedy takes the most likely token, beam search keeps the B most likely continuations and picks the best sequence. Sampling methods draw at random: random sampling from all of p; top-k, top-p, min-p and typical sampling from a filtered, renormalized subset. Speculative decoding does not change the distribution; it makes sampling faster by letting a small draft model propose tokens that the large model only checks.`, decode: true },
  ];
}

/* Which diagram parts belong to each step (ids are drawn in app.js). */
const HIGHLIGHT = {
  standard: {
    q: ['x', 'wq', 'qrow'], kv: ['x', 'wk', 'wv', 'krow', 'vrow'],
    pos: ['pos', 'qrow', 'krow'], score: ['qsel', 'ksel', 'link', 'pq', 'pk', 'ps'],
    mask: ['ps', 'pa'], mix: ['vsel', 'pa', 'pv', 'po'],
    merge: ['po', 'pcat', 'pwo', 'py'], cache: ['krow', 'vrow', 'cache'],
  },
  mla: {
    q: ['x', 'wq', 'qrow'], kv: ['x', 'wdkv', 'latent'],
    pos: ['x', 'wkr', 'kr', 'rope', 'qrope'], expand: ['latent', 'wuk', 'wuv', 'krow', 'vrow', 'kr', 'rope'],
    score: ['qsel', 'ksel', 'link', 'pq', 'pk', 'ps'], mask: ['ps', 'pa'],
    merge: ['vsel', 'pa', 'pv', 'po', 'pcat', 'pwo', 'py'], cache: ['latent', 'kr', 'cache'],
  },
  gdn: {
    proj: ['qrow', 'krow', 'vrow', 'gate', 'links'], conv: ['qrow', 'krow', 'vrow', 'conv'], gates: ['gate', 'pg'],
    delta: ['ksel', 'vsel', 'gate', 'state', 'pk', 'pv', 'pg', 'ps', 'chain'], read: ['qsel', 'state', 'pq', 'ps', 'po'],
    out: ['po', 'pz', 'py'], cost: ['chain', 'ps', 'state'], cache: ['state', 'cache'],
  },
  sparse: {
    iq: ['qi'], ik: ['ki'], iscore: ['qi', 'ki', 'bars'], select: ['bars', 'sel', 'budget'],
    attend: ['sel', 'lat', 'pq', 'pk', 'ps', 'pa', 'pv', 'po'], merge: ['po', 'pcat', 'pwo', 'py'],
    cost: ['bars', 'lat', 'sel', 'budget'], cache: ['ki', 'lat', 'cache'],
  },
};

const ARCH_HIGHLIGHT = {
  tok: ['text', 'tok'], emb: ['tok', 'emb'], stream: ['stream', 'emb'], norm1: ['norm1', 'xhat'], qkv: ['xhat', 'q', 'k', 'v', 'arc'],
  rope: ['rope', 'q', 'k', 'kr', 'pos'],
  attn: ['q', 'k', 'v', 'att', 'y'], res1: ['y', 'res1', 'attnout', 'stream'], norm2: ['norm2', 'fin'],
  ffn: ['fin', 'ffn'], res2: ['ffn', 'ffnout', 'res2', 'stream'], repeat: ['block', 'stream'],
  fnorm: ['fnorm'], head: ['head', 'logit'], dist: ['logit', 'dist'], decode: ['dist', 'decbox', 'chosen', 'loop'],
  conv: ['conv', 'q', 'k', 'v'], state: ['state', 'k', 'v', 'gate', 'chain', 'arc2'], index: ['ki', 'iq', 'ibar', 'att', 'sel'],
};
/* Per-view differences from ARCH_HIGHLIGHT. */
function archHighlight(m, view = {}) {
  if (m.key === 'gdn' && view.layer !== 'gattn') return { ...ARCH_HIGHLIGHT, qkv: ['xhat', 'q', 'k', 'v', 'gate', 'zg', 'arc'], attn: ['q', 'state', 'o', 'y', 'zg', 'arc2'] };
  if (m.key === 'gdn') return { ...ARCH_HIGHLIGHT, qkv: ['xhat', 'q', 'k', 'v', 'gt', 'arc'], attn: ['q', 'k', 'v', 'att', 'gt', 'y'] };
  if (m.key === 'dsa') return { ...ARCH_HIGHLIGHT, qkv: ['xhat', 'q', 'k', 'v', 'kr', 'ki', 'iq', 'arc'], attn: ['q', 'k', 'v', 'att', 'y', 'sel'], rope: ['rope', 'q', 'kr', 'pos'] };
  return ARCH_HIGHLIGHT;
}
/* The layer views a model offers in the whole-model diagram. */
function views(m) {
  if (m.key === 'gdn') return [{ moe: true, layer: 'gdn' }, { moe: true, layer: 'gattn' }];
  if (m.ffnKind === 'moe') return [{ moe: true }, { moe: false }];
  return [{ moe: false }];
}


/* A guided tour for newcomers: one idea per step, told on the GPT-2 diagram, revealing only the
 * parts the step talks about (ids as in ARCH_HIGHLIGHT). It ends with the six mechanisms as six
 * answers to the same problem, the growing KV cache. */
const TOUR = [
  { id: 'tokens', title: 'Text becomes numbers', on: ['text', 'tok'],
    term: ['Token', 'A piece of text represented by one number.'],
    get text() {     // names the most-split word of the example on show, as GPT-2 cuts it
      const tk = TOKENS.models.mha, words = TOKENS.sentence.split(' ');
      const n = words.map((w, i) => tk.pieces.filter((p, t) => tk.words[t] === i && /[A-Za-z]/.test(p)).length);
      const most = n.indexOf(Math.max(...n)), num = ['', 'one', 'two', 'three', 'four', 'five', 'six'][n[most]] || String(n[most]);
      const split = n[most] > 1 ? `a rarer word like “${words[most].replace(/[^A-Za-z]+$/, '')}” is built from ${num}` : 'a rare word is built from several (try the Subwords example)';
      return `A language model never sees letters. A tokenizer cuts the text into pieces from a fixed vocabulary (GPT-2’s has 50,257 entries) and replaces each piece with its number. Frequent words are a single piece; ${split}. Hover a word to see how it was cut.`;
    } },
  { id: 'vectors', title: 'Numbers become vectors', on: ['tok', 'emb', 'stream'],
    term: ['E · embedding table', 'Turns each token ID into a vector; the residual stream carries these vectors through the model.'],
    text: 'Each number selects one row of a learned table, E: a list of 768 numbers that stands for that piece. From here on every token is one row of numbers, and all the rows travel through the model side by side. This bundle, the residual stream, is what every later step reads and adds to.' },
  { id: 'qkv', title: 'Every token asks, offers and passes on', on: ['norm1', 'attnbox', 'xhat', 'q', 'k', 'v', 'arc'],
    term: ['$\\hat{x}$, $Q$, $K$, $V$', '$\\hat{x}$ is the normalized token vector; $Q$ asks, $K$ is matched, and $V$ contributes information. See the highlighted diagram key.'],
    text: 'To share information, each token computes three new vectors from its row: a query (what it is looking for), a key (what it can be found by) and a value (what it hands on when it is found). Every token gets its own three, made with the same weights for all tokens; GPT-2 splits each into 12 heads, the small cells in every row. The arrows show it for the last token.' },
  { id: 'attention', title: 'Attention: looking back at earlier tokens', on: ['attnbox', 'q', 'k', 'v', 'att', 'y'],
    term: ['Attention weights', 'Numbers describing how much of each earlier token’s value to use.'],
    text: 'Each token compares its query with the key of every earlier token. The better a key matches, the more weight that token gets; the weights in each row of the grid add up to 1. The token then takes the weighted mix of those tokens’ values. No token can look ahead, so the upper triangle stays blank. GPT-2 does this 12 times in parallel (12 heads), each free to look for something different.' },
  { id: 'block', title: 'A block, repeated', on: ['attnbox', 'y', 'res1', 'attnout', 'stream', 'norm2', 'ffnbox', 'fin', 'ffn', 'ffnout', 'res2', 'block'],
    term: ['Transformer block', 'Attention plus a feed-forward layer, adding updates back to each token’s representation.'],
    text: 'The mixed result is added onto each token’s row. A second part, the feed-forward layer, then works on every token on its own. Attention plus feed-forward make one block; GPT-2 stacks 12 of them, each with its own weights, and each refines the rows a little more.' },
  { id: 'next', title: 'Predicting the next piece', on: ['stream', 'fnorm', 'head', 'logit', 'dist', 'decbox', 'chosen', 'loop'],
    term: ['Logits', 'The scores for possible next tokens, before converting them into probabilities.'],
    text: 'After the last block, the last token’s row is turned into one score for every entry of the vocabulary, and the scores into probabilities. A decoding rule picks one; here it samples among the five most likely. The chosen piece is appended to the text and the model runs again: text is generated one piece at a time.' },
  { id: 'cache', title: 'The memory problem', on: ['tok', 'k', 'v', 'loop'], extra: 'cache',
    term: ['KV cache', 'Saved keys and values from earlier tokens so they do not need to be recomputed.'],
    text: 'Running again does not mean starting over. The keys and values of earlier tokens never change, so they are kept in memory: the KV cache, shaded here. But it grows with every token, in every block. GPT-2 stores 36 KiB per token, 36 MiB for its full context of 1,024 tokens. Today’s models read contexts of 100,000 tokens and more with much larger layers, and this cache becomes the main limit on memory and speed.' },
  { id: 'answers', title: 'Six answers', on: ['attnbox', 'q', 'k', 'v', 'att', 'y'], extra: 'answers',
    term: ['The years', 'Dates on the tabs describe these named designs, not the ages of the example models; Sparse shows DSA (2025) and QSA (2026).'],
    text: 'Each tab of this page is one answer to that problem, shown in a real model. They differ only inside the attention box. Pick one to explore it in full; the bars show what each stores per token and layer at the same width.' },
];
const TOUR_ANSWERS = {
  mha: 'Every head keeps its own keys and values: the original design.',
  mqa: 'All heads share one key and one value: 8× less to store.',
  gqa: 'Groups of heads share keys and values: the common middle ground.',
  mla: 'Store one small compressed vector per token and rebuild keys and values from it.',
  gdn: 'In most layers, replace the growing cache with a fixed-size memory.',
  dsa: 'Keep everything, but let each token read only the 2,048 most relevant.',
};

/* References: authors (year). Title. Where published. */
const SOURCES = [
  { by: 'Vaswani et al.', year: 2017, title: 'Attention Is All You Need', in: 'NeurIPS 2017', url: 'https://arxiv.org/abs/1706.03762' },
  { by: 'Shazeer', year: 2019, title: 'Fast Transformer Decoding: One Write-Head is All You Need', in: 'arXiv:1911.02150', url: 'https://arxiv.org/abs/1911.02150' },
  { by: 'Ainslie et al.', year: 2023, title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints', in: 'EMNLP 2023', url: 'https://arxiv.org/abs/2305.13245' },
  { by: 'DeepSeek-AI', year: 2024, title: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model', in: 'arXiv:2405.04434', url: 'https://arxiv.org/abs/2405.04434' },
  { by: 'Su et al.', year: 2024, title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', in: 'Neurocomputing 568', url: 'https://arxiv.org/abs/2104.09864' },
  { by: 'Yang, Kautz & Hatamizadeh', year: 2025, title: 'Gated Delta Networks: Improving Mamba2 with Delta Rule', in: 'ICLR 2025', url: 'https://arxiv.org/abs/2412.06464' },
  { by: 'Qwen Team', year: 2025, title: 'Qwen3-Next: Towards Ultimate Training & Inference Efficiency', in: 'Qwen blog', url: 'https://qwenlm.github.io/blog/qwen3_next/' },
  { by: 'DeepSeek-AI', year: 2025, title: 'DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models', in: 'arXiv:2512.02556', url: 'https://arxiv.org/abs/2512.02556' },
  { by: 'SGLang Team', year: 2026, title: 'Qwen3.8-Flash-Next: Day-0 Support in SGLang', in: 'LMSYS blog', url: 'https://www.lmsys.org/blog/2026-08-26-qwen-flash-next' },
  { by: 'Holtzman et al.', year: 2020, title: 'The Curious Case of Neural Text Degeneration', in: 'ICLR 2020', url: 'https://arxiv.org/abs/1904.09751' },
  { by: 'Meister et al.', year: 2023, title: 'Locally Typical Sampling', in: 'Transactions of the ACL 11', url: 'https://arxiv.org/abs/2202.00666' },
  { by: 'Nguyen et al.', year: 2025, title: 'Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs', in: 'ICLR 2025', url: 'https://arxiv.org/abs/2407.01082' },
  { by: 'Leviathan, Kalman & Matias', year: 2023, title: 'Fast Inference from Transformers via Speculative Decoding', in: 'ICML 2023', url: 'https://arxiv.org/abs/2211.17192' },
];

function allTex() {
  const out = new Set();
  for (const k of ORDER) {
    const m = MODELS[k];
    for (const variant of k === 'dsa' ? ['dsa', 'qsa'] : [undefined]) {
      overview(m, variant).tex.forEach(t => out.add(t));
      out.add(overview(m, variant).block);
      steps(m, variant).forEach(s => s.tex.forEach(t => out.add(t)));
    }
    for (const view of views(m)) {
      splitTex(archOverview(m, view).tex).forEach(t => out.add(t));
      archSteps(m, view).forEach(s => splitTex(s.tex).forEach(t => out.add(t)));
    }
  }
  for (const list of Object.values(DECODE_TEX)) splitTex(list).forEach(t => out.add(t));
  return [...out];
}

const API = { TOUR, TOUR_ANSWERS, MODELS, ORDER, ARCH_YEARS, QSA, COMPARE, HIGHLIGHT, ARCH_HIGHLIGHT, SOURCES, TOKENS, METHODS, fmt, bytes, cacheValues, cacheLayers, gdnState, showPiece,
  nextTokenData, decodeDist, beamSearch, specDist, usesTau, DRAFT_SHIFT, DECODE_TEX, EXAMPLES, setExample, realEntry,
  cacheBytesPerToken, compareRows, kvHeadOf, overview, stats, steps, allTex, views, archHighlight,
  archOverview: (m, view) => { const o = archOverview(m, view); return { ...o, tex: splitTex(o.tex) }; },
  archSteps: (m, view) => archSteps(m, view).map(s => ({ ...s, tex: splitTex(s.tex) })) };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
else root.LAB = API;
})(typeof window !== 'undefined' ? window : globalThis);
