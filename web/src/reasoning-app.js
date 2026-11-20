/* Reasoning page: one diagram of thinking.
 *   Context window (prompt · <think> · thinking · </think> · answer)
 *   One forward pass: input token → embedding → 61 layers (attention reads the KV cache; 8 of 256
 *   experts) → LM head → sample → output token, appended to the context
 *   KV cache: one entry per token so far
 * Step or play through a trace written for the page; click a part for its details below. */
(function () {
'use strict';
const R = window.REASON, $ = s => document.querySelector(s), root = $('#reasoning-lab');
if (!R || !root) return;
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const tex = s => { const out = window.LABUI?.tex?.(s); return out && out !== s ? out : `<code>${esc(s)}</code>`; };
const N_ = n => n.toLocaleString('en-US');
const B = n => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : N_(n));
const bytes = n => (n >= 2 ** 40 ? `${(n / 2 ** 40).toFixed(1)} TiB` : n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`);
const A = R.ARCH, P = R.parameters(), CACHE = R.cacheBytesPerToken(), PER_LAYER = A.kvRank + A.ropeDim;

/* ---------- the trace: prompt, then generated pieces (about one token each) ---------- */
const spaced = text => [...text.matchAll(/(\s*)([\p{L}\p{N}²]+|[^\s\p{L}\p{N}])/gu)].map(m => [m[1] ? ' ' : '', m[2]]);
const PROMPT_TEXT = R.TEMPLATE.replace('{question}', R.TRACE.question);
const PROMPT_N = R.pieces(PROMPT_TEXT).length + 1;                    // + <think>
const GEN = [
  ...spaced(R.TRACE.thinking).map(([sp, p]) => ({sp, p, phase: 'think'})),
  {sp: ' ', p: '</think>', phase: 'end'},
  {sp: ' ', p: '<answer>', phase: 'answer'},
  ...spaced(R.TRACE.answer).map(([sp, p], i) => ({sp: i ? sp : ' ', p, phase: 'answer'})),
  {sp: ' ', p: '</answer>', phase: 'answer'},
  {sp: ' ', p: '<eos>', phase: 'eos'},
];
const N = GEN.length, END = GEN.findIndex(g => g.phase === 'end') + 1, WAIT = GEN.findIndex(g => /^wait$/i.test(g.p)) + 1;
const st = {k: WAIT, part: 'loop', ready: false};
let timer = null;

/* pass k (1 … N) writes GEN[k − 1]; its input is the token before (pass 1: the whole prompt, at once) */
function pass(k) {
  const out = GEN[k - 1], input = k === 1 ? null : GEN[k - 2];
  const phase = k === 1 ? 'prefill' : out.phase === 'think' ? 'thinking' : out.phase === 'end' ? 'closing the thought' : out.phase === 'eos' ? 'ending the reply' : 'answering';
  return {k, out, input, phase, reads: PROMPT_N + k - 1, cached: PROMPT_N + k - 1};
}

/* ---------- the diagram ---------- */
const part = (id, cls, inner, label) => `<div role="button" tabindex="0" class="think-part ${cls}${st.part === id ? ' open' : ''}" data-part="${id}" aria-label="${label}: details" aria-pressed="${st.part === id}">${inner}</div>`;
const arrow = '<span class="think-arrow" aria-hidden="true"></span>';
// which candidate the sampler draws: mostly the most likely one, sometimes another (the bars are schematic)
const drawn = k => (GEN[k - 1] && /^wait$/i.test(GEN[k - 1].p) ? 2 : [0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0][(k * 7) % 13]);
const kind = p => [/^<\/?\w+>$/.test(p) ? 'tag' : '', /^wait$/i.test(p) ? 'wait' : ''].filter(Boolean).join(' ');
const tok = (p, cls = '') => `<span class="think-tok ${cls} ${kind(p)}">${esc(p)}</span>`;
function draw() {
  const s = pass(st.k), r = R.route(R.demoScores(st.k * 7919 + 13));
  const on = new Set(r.experts), allowed = new Set(r.groups), per = A.experts / A.groups;

  // status and controls
  $('#think-status').innerHTML = `<b>Pass ${N_(s.k)}</b> of ${N_(N)} <span class="think-phase ${s.phase.split(' ')[0]}">${s.phase}</span>`;
  $('#think-slider').max = String(N);
  $('#think-slider').value = String(st.k);
  $('#think-play').textContent = timer ? 'pause' : 'play';

  // context window
  const written = GEN.slice(0, s.k).map((g, i) => {
    const cls = [i === s.k - 1 ? 'out' : '', i === s.k - 2 ? 'in' : '', g.phase === 'answer' ? 'ans' : ''].filter(Boolean).join(' ');
    return `${g.sp}${tok(g.p, cls)}`;
  }).join('');
  $('#think-ctx').innerHTML = part('context', 'think-ctxbox', `<span class="think-cap">Context window <small>${N_(PROMPT_N + s.k)} tokens: what the model reads</small></span><span class="think-scroll"><span class="think-prompt${s.k === 1 ? ' in' : ''}">${esc(PROMPT_TEXT)} ${tok('<think>', 'tag')}</span>${written}<span class="think-caret" aria-hidden="true"></span></span>`, 'The context window');

  // one forward pass
  const grid = Array.from({length: A.groups}, (_, g) => `<span class="think-group${allowed.has(g) ? '' : ' off'}">${Array.from({length: per}, (_, j) => `<i class="${on.has(g * per + j) ? 'on' : ''}"></i>`).join('')}</span>`).join('');
  const pick = drawn(s.k);
  const bars = [0.46, 0.21, 0.12, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015].map((h, i) => `<i style="height:${(h * 100 / 0.46).toFixed(0)}%"${i === pick ? ' class="pick"' : ''}></i>`).join('');
  $('#think-pass').innerHTML = [
    part('input', 'think-io think-in', `<span class="think-cap">input</span>${s.input ? tok(s.input.p, `in${s.input.phase === 'answer' ? ' ans' : ''}`) : `<span class="think-all">the whole prompt, ${PROMPT_N} tokens at once</span>`}`, 'The input of this pass'),
    arrow,
    part('embed', 'think-box', `<span class="think-cap">Embedding</span><b>${N_(A.hidden)}</b><small>numbers for the token</small>`, 'Embedding'),
    arrow,
    `<div class="think-layers"><span class="think-lcap">× ${A.layers} layers <small>the same for every token</small></span>
      ${part('attention', 'think-box think-attn', `<span class="think-cap">Attention</span><b>${N_(s.reads)}</b><small>${s.k === 1 ? 'positions, each reading the ones before' : 'positions read, itself included'}</small>`, 'Attention')}
      ${arrow}
      ${part('experts', 'think-box think-moe', `<span class="think-cap">Experts <small>${A.topK} of ${A.experts} + ${A.shared}</small></span><span class="think-grid" aria-hidden="true">${grid}</span>`, 'Experts')}
    </div>`,
    arrow,
    part('head', 'think-box', `<span class="think-cap">LM head</span><b>${N_(A.vocab)}</b><small>scores, one per vocabulary token</small>`, 'LM head'),
    arrow,
    part('sample', 'think-box think-sample', `<span class="think-cap">Sample</span><span class="think-bars" aria-hidden="true">${bars}</span><small>one token is drawn</small>`, 'Sampling'),
    arrow,
    part('output', 'think-io think-out', `<span class="think-cap">output</span>${tok(s.out.p, `out${s.out.phase === 'answer' ? ' ans' : ''}`)}<small class="think-back">appended to the context</small>`, 'The output of this pass'),
  ].join('');
  const [stop, verdict, tone] = STOP(s.out);
  $('#think-stop').innerHTML = part('stop', `think-stopbox ${stop ? `yes ${tone}` : ''}`, `<span class="think-cap">Stop? <small>checked after every token</small></span>
    <span class="think-verdict ${tone}">${verdict}</span><small class="think-limit">${N_(s.k)} of ${N_(R.MAX_GENERATION)} tokens: length limit</small>`, 'Stopping');
  $('#think-rw').innerHTML = `<b>reads ${N_(s.reads)}</b><span>${s.k === 1 ? `writes ${PROMPT_N}` : 'writes 1'}</span>`;

  // KV cache
  const gen = GEN.slice(0, Math.max(0, s.k - 1)), think = gen.filter(g => g.phase !== 'answer').length, ans = gen.length - think;
  const seg = (n, cls) => (n > 0 ? `<i class="${cls}" style="flex:${n}"></i>` : '');
  $('#think-cache').innerHTML = part('cache', 'think-cachebox', `<span class="think-cap">KV cache <small>one entry per token so far, in every layer</small></span>
    <span class="think-cachebar">${seg(PROMPT_N, 'p')}${seg(think, 't')}${seg(ans, 'a')}${seg(PROMPT_N + N - 1 - s.cached, 'rest')}</span>
    <span class="think-cachefoot"><span><b>${N_(s.cached)}</b> tokens × ${A.layers} layers × ${PER_LAYER} numbers × 2 bytes = <b>${bytes(s.cached * CACHE)}</b></span><span class="think-key"><b class="p">prompt</b><b class="t">thinking</b><b class="a">answer</b></span></span>`, 'The KV cache');
  $('#think-detail').innerHTML = DETAIL[st.part](s, r);
  const ctx = $('#think-ctx .think-scroll');
  if (ctx) ctx.scrollTop = ctx.scrollHeight;
  bind();
}

/* the check after every pass: has the model closed its thought, or ended the reply? */
const STOP = out => (out.phase === 'end' ? [true, '<b>&lt;/think&gt;</b> drawn: the thinking stops', 'k']
  : out.phase === 'eos' ? [true, '<b>&lt;eos&gt;</b> drawn: the reply is done', 'v']
  : out.phase === 'answer' ? [false, 'no <b>&lt;eos&gt;</b> yet: keep answering', 'v']
  : /^wait$/i.test(out.p) ? [false, 'no <b>&lt;/think&gt;</b>: “Wait” instead', 'q']
  : [false, 'no <b>&lt;/think&gt;</b>: keep thinking', 'q']);

/* ---------- details ---------- */
const box = (left, right) => `<div class="retr-dgrid"><div>${left}</div><div>${right}</div></div>`;
const rules = items => `<ul class="retr-rules">${items.map(x => `<li>${x}</li>`).join('')}</ul>`;
const eq = (...fs) => `<div class="retr-eqs">${fs.map(f => `<div class="retr-eq">${tex(f)}</div>`).join('')}</div>`;
const dhead = (title, meta = '') => `<div class="retr-dhead"><b>${title}</b>${meta ? `<span class="retr-meta">${meta}</span>` : ''}</div>`;
const costRows = () => [1000, 8000, 32768, 65536].map(L => `<div class="think-prow"><span>${N_(L)} tokens</span><em>${bytes(L * CACHE)}</em><em>${(2 * (P.active - P.parts.embedding) * L / 1e15).toFixed(2)}</em><em>${bytes(L * (L - 1) / 2 * CACHE)}</em></div>`).join('');
const DETAIL = {
  loop: s => box(`${dhead('Thinking is a loop of forward passes', `pass ${s.k}: ${s.phase}`)}
      ${eq(String.raw`x_{t+1}\sim p_\theta\bigl(\,\cdot \mid \text{prompt},\,x_1,\dots,x_t\bigr)`)}
      <p class="retr-dnote">Each pass reads the whole context so far and writes one token; the token joins the context, and the next pass begins, until the model writes an end token or the length limit is reached. Step through the trace above, or play it.</p>`,
    rules(['<b>Reasoning is writing.</b> There is no separate reasoning module and no hidden scratchpad: the thinking is the tokens between <code>&lt;think&gt;</code> and <code>&lt;/think&gt;</code>.',
      `<b>The same network every time</b>: ${A.layers} layers and ${B(P.active)} of its ${B(P.total)} parameters per token, whether the token is part of the question, the thinking or the answer.`,
      '<b>A fixed amount of work per token.</b> A harder problem gets more computation only by getting more tokens: thinking longer means more passes.',
      '<b>The context is the working memory.</b> Nothing else carries over from one pass to the next.'])),
  context: () => box(`${dhead('The context window', `up to ${N_(A.maxPositions)} tokens`)}
      <p class="think-text">${esc(PROMPT_TEXT)} <span class="think-tag">&lt;think&gt;</span></p>
      <p class="retr-dnote">The prompt (the paper’s template, paraphrased) ends right after <code>&lt;think&gt;</code>, so the first token the model writes is already reasoning.</p>`,
    rules(['<b>Thinking tokens are ordinary tokens.</b> They come from the same vocabulary and enter the network like the words of the question.',
      '<b>Everything stays.</b> Earlier steps, including wrong ones, remain in the context; a correction is new text after them, not an erasure.',
      '<b>Only the past.</b> A token sees the tokens before it, never after: the thinking is built left to right.'])),
  input: s => box(`${dhead(s.k === 1 ? 'Prefill: the prompt in one pass' : 'Decode: one new token in', s.k === 1 ? `${PROMPT_N} positions in parallel` : 'the token written by the pass before')}
      <p class="retr-dnote">${s.k === 1 ? `The first pass runs over all ${PROMPT_N} prompt tokens at once: each position attends to the ones before it, every position’s keys and values go into the cache, and the last position predicts the first thinking token.` : `From now on, each pass takes only the newest token, <code>${esc(s.input.p)}</code>. Everything earlier is already in the cache, so it is read, not recomputed.`}</p>`,
    rules(['<b>Prefill</b> is parallel and happens once; <b>decode</b> is one token at a time, for every token of the thinking and the answer.',
      `<b>The cache saves work</b>: without it, every pass would recompute all ${N_(s.reads)} earlier positions through all ${A.layers} layers.`])),
  embed: () => box(`${dhead('From token to vector', `${B(P.parts.embedding)} parameters`)}
      ${eq(String.raw`h^{(0)}=E[\text{token}],\qquad E\in\mathbb{R}^{${A.vocab}\times ${A.hidden}}`)}
      <p class="retr-dnote">The token’s id picks one row of a learned table: ${N_(A.hidden)} numbers, the same for this token wherever it appears. Its position is added later, inside attention.</p>`,
    rules(['<b>“Wait” is a row too</b>: a self-correction starts as an ordinary vector, like any other word.',
      `<b>Positions</b>: rotary embeddings turn 64 numbers of each query and key, stretched (YaRN) to ${N_(A.maxPositions)} positions.`])),
  attention: s => box(`${dhead('Attention: reading the thinking so far', `${N_(s.reads)} positions in every layer`)}
      ${eq(String.raw`\operatorname{softmax}\!\Bigl(\frac{q_t\,K_{\le t}^{\top}}{\sqrt{d}}\Bigr)\,V_{\le t}`)}
      <p class="retr-dnote">In each of ${A.layers} layers, ${A.heads} heads compare the new token’s query with the keys of every earlier token and take a weighted mix of their values. This is how a later step can use an earlier one, or return to it after a “Wait”.</p>`,
    rules([`<b>Latent attention (MLA)</b>: keys and values are rebuilt from a ${A.kvRank}-number latent per token, so the cache keeps only that latent and a ${A.ropeDim}-number rotary key: ${PER_LAYER} numbers per layer and token, instead of ${N_(A.heads * (A.nopeDim + A.ropeDim + A.vDim))} for the keys and values of all ${A.heads} heads.`,
      `<b>Causal</b>: position t reads positions 1 … t, never later ones.`,
      '<b>Cost grows with the context</b>: every new token reads the whole cache, so each pass of a long thought reads more than the one before.',
      `${B(P.parts.attention)} parameters in all layers; the Architectures page shows MLA step by step.`])),
  experts: (s, r) => box(`${dhead('Experts: a few specialists per token', `this pass, one layer: experts ${r.experts.slice().sort((x, y) => x - y).join(', ')}`)}
      ${eq(String.raw`y=\sum_{i\in\text{top-}8} g_i\,\operatorname{Expert}_i(x)+\operatorname{Shared}(x)`)}
      <p class="retr-dnote">After attention, a router scores all ${A.experts} experts; only ${A.topK} of them run for this token, plus ${A.shared} shared expert. The figure draws demo scores with DeepSeek-V3’s rule: the experts come in ${A.groups} groups of ${A.experts / A.groups}, only the best ${A.topGroups} groups may be used (faded: ruled out), and the best ${A.topK} experts among them run.</p>`,
    rules([`<b>Different tokens, different experts</b>: every token, and every layer, gets its own choice; step through the trace to see it change.`,
      `<b>Big, but sparse</b>: ${B(P.parts.routedExperts)} parameters sit in the routed experts, of which ${B(P.parts.routedExperts * A.topK / A.experts)} run per token. The first ${A.denseLayers} layers use one dense feed-forward network instead.`])),
  head: () => box(`${dhead('From vector to scores', `${B(P.parts.head)} parameters`)}
      ${eq(String.raw`z=W_{\text{head}}\,\operatorname{RMSNorm}\bigl(h^{(${A.layers})}\bigr)\in\mathbb{R}^{${A.vocab}}`)}
      <p class="retr-dnote">The last layer’s vector becomes one score for each of the ${N_(A.vocab)} tokens of the vocabulary: how well each would continue the text.</p>`,
    rules(['<b>The only output is these scores.</b> Whatever the model “thinks”, it has to come out through them, one token at a time.',
      `<b>One more predictor</b>: the config has ${A.mtp} multi-token-prediction module, trained to guess a further token; it can speed up decoding.`])),
  sample: () => box(`${dhead('Choosing the token', 'the bars are schematic')}
      ${eq(String.raw`p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}`)}
      <p class="retr-dnote">The scores become probabilities; a temperature T flattens (T > 1) or sharpens (T < 1) them, and one token is drawn.</p>`,
    rules(['<b>Runs differ</b>: drawn at random, the same question can lead to different chains of thought, and so to different answers.',
      '<b>Small choices steer the thought</b>: whether the next token is “So” or “Wait” decides whether the text moves on or checks the last step.'])),
  output: s => box(`${dhead('The new token joins the context', `“${esc(s.out.p)}”`)}
      <p class="retr-dnote">The written token is appended to the context; in the next pass it is the input, and its keys and values join the cache.</p>`,
    rules(['<b>Thinking ends with a token</b>: when <code>&lt;/think&gt;</code> is drawn, the thought is closed and the answer follows (see “Stop?”).',
      '<b>The answer reads the thinking</b>: its tokens attend to the question and to every thinking token before them.',
      '<b>Shown or hidden</b>: an application can hide the thinking from the user; the model had to write it either way.'])),
  stop: s => box(`${dhead('How and when thinking stops', `this pass: “${esc(s.out.p)}”`)}
      <div class="think-stops">
        <div class="think-srow head"><span>what happens</span><span>who decides</span><span>the result</span></div>
        <div class="think-srow"><span>the model draws <code>&lt;/think&gt;</code></span><span>the model</span><span>the thinking ends and the answer follows, reading the whole thought</span></div>
        <div class="think-srow"><span>the model draws <code>&lt;eos&gt;</code></span><span>the model</span><span>the reply is done (an end-of-sequence token, shown here as &lt;eos&gt;)</span></div>
        <div class="think-srow"><span>the length limit is reached</span><span>the software</span><span>generation is cut off, mid-thought if need be: no answer</span></div>
        <div class="think-srow"><span>a thinking budget runs out</span><span>the software</span><span>it appends <code>&lt;/think&gt;</code> itself, and the model answers from the thought so far</span></div>
      </div>
      <p class="retr-dnote">Nothing in the network counts tokens or checks whether the problem is solved. <code>&lt;/think&gt;</code> is one of ${N_(A.vocab)} tokens, and the thinking stops when the sampler draws it, which becomes likely once the model gives it a high probability.</p>`,
    rules(['<b>When is learned, not programmed.</b> In RL, outputs were rewarded for a correct answer in the right format; the length of the thinking was never set, and R1-Zero’s thinking grew steadily longer during training.',
      `<b>A hard limit</b>: R1-Zero’s RL allowed ${N_(R.RL_MAX_LENGTH[0])} tokens per output, later ${N_(R.RL_MAX_LENGTH[1])}; DeepSeek evaluates R1 with at most ${N_(R.MAX_GENERATION)}. A thought that reaches the limit is cut off before it has an answer.`,
      '<b>It adapts, imperfectly</b>: the paper reports fewer tokens for simple problems and more for hard ones, but also excessive reasoning on some simple questions.',
      '<b>Longer or shorter from outside</b> (budget forcing, s1): to cap the thinking, the software appends <code>&lt;/think&gt;</code> once the budget is used up; to extend it, it suppresses <code>&lt;/think&gt;</code> and appends “Wait”, as the model itself does in this trace.',
      '<b>Or not at all</b>: R1 sometimes closes the thought at once (<code>&lt;think&gt;</code> then <code>&lt;/think&gt;</code>); DeepSeek recommends starting every reply with <code>&lt;think&gt;</code> and a newline to make it think.'])),
  cache: s => box(`${dhead('The KV cache: memory that grows with the thought', `${bytes(CACHE)} per token`)}
      <div class="think-parts"><div class="think-prow head"><span>context length</span><em>cache</em><em>PFLOP</em><em>cache reads, all passes</em></div>${costRows()}</div>
      <p class="retr-dnote">Computed from the architecture: ${A.layers} layers × ${PER_LAYER} numbers × 2 bytes per token; about 2 × ${B(P.active - P.parts.embedding)} operations per token for the weight matrices; and since every pass reads the whole cache, the reads grow with the square of the length. In the trace above: ${N_(s.cached)} tokens, ${bytes(s.cached * CACHE)}.</p>`,
    rules(['<b>Grows by one token per pass</b>, prompt, thinking and answer alike, and is kept until the reply is done.',
      '<b>Long thinking is memory-bound</b>: generating each token means reading the whole cache, so MLA’s small cache is what makes thousands of thinking tokens affordable.'])),
};

/* ---------- interaction ---------- */
function stop() { if (timer) { clearInterval(timer); timer = null; } }
function go(k) { st.k = Math.max(1, Math.min(N, k)); draw(); }
function bind() {
  root.querySelectorAll('[data-part]').forEach(b => {
    const open = () => { st.part = b.dataset.part; draw(); root.querySelector(`[data-part="${st.part}"]`)?.focus({preventScroll: true}); };
    b.addEventListener('click', open);
    b.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });
}
function init() {
  if (st.ready) return;
  st.ready = true;
  const stat = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
  $('#reas-stats').innerHTML = [stat('Model', `<a href="${A.source}" target="_blank" rel="noopener noreferrer">${A.name}</a>`), stat('Layers', A.layers),
    stat('Active parameters', `${B(P.active)} of ${B(P.total)}`), stat('KV cache', `${bytes(CACHE)} per token`)].join('');
  $('#think-slider').addEventListener('input', e => { stop(); go(Number(e.target.value)); });
  root.querySelectorAll('[data-step]').forEach(b => b.addEventListener('click', () => { stop(); go(st.k + Number(b.dataset.step)); }));
  root.querySelectorAll('[data-jump]').forEach(b => b.addEventListener('click', () => { stop(); go({start: 1, wait: WAIT, end: END, done: N}[b.dataset.jump]); }));
  $('#think-play').addEventListener('click', () => {
    if (timer) { stop(); draw(); return; }
    if (st.k >= N) st.k = 1;
    timer = setInterval(() => { if (st.k >= N || root.closest('.page').hidden) { stop(); draw(); } else go(st.k + 1); }, 130);
    draw();
  });
  draw();
}
window.REASONINGAPP = {show: init, state: st, pass, N, END, WAIT, PROMPT_N};
if (!root.closest('.page').hidden) init();
})();
