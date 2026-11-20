/* Retrieval page, three modes on one figure.
 *   sparse:  Input → (unique words) → Vocabulary → (Weighting: BM25 / TF-IDF) → Sparse vectors → (q · d) → Output
 *   learned: Input → (Tokenize) → Tokens → (Encoder + MLM head, SPLADE pooling) → Learned sparse vectors → (q · d) → Output
 *   dense:   Input → (Tokenize) → Tokens → (Encoder: MiniLM, mean pooling) → Dense vectors → (q · d) → Output
 *   hybrid:  Input → (Two retrievers) → Two rankings → (Fusion: RRF / weighted) → Fused scores → (sort) → Output
 * The three labelled boxes on the arrows open the details of their step. */
(function () {
'use strict';
const R = window.RLAB, $ = s => document.querySelector(s), root = $('#retrieval-lab');
if (!R || !root) return;
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const st = {preset: R.PRESETS[0].id, question: R.PRESETS[0].question, mode: 'sparse', method: 'bm25', k: 0, doc: 0, term: null, dim: null, lterm: null, tokOf: 'q', fusion: 'rrf', open: 'score', ready: false};
let res = null, sres = null, dres = null, lres = null, hres = null;     // the shown result; the sparse, dense, learned sparse and hybrid one
const DENSE = R.DENSE, HAS_DENSE = !!DENSE, PLACEHOLDER = !!(DENSE && DENSE.placeholder);   // placeholder: random vectors, not model output
const LEARNED = R.LEARNED, HAS_LEARNED = !!LEARNED, L_PLACEHOLDER = !!(LEARNED && LEARNED.placeholder);
const MODEL_DATA = () => (st.mode === 'learned' ? LEARNED : DENSE);   // the tokenizer's data in the two model modes
const V = () => MODEL_DATA().vocab_size.toLocaleString('en-US');
const n = R.INDEX.vocab.length;
const FIRST = 16;                       // vocabulary rows shown before the ⋮ (plus the query's terms and the last term)
const id2 = i => String(i + 1).padStart(2, '0');
const f2 = v => (v === 0 ? '0' : v.toFixed(2));
const f3 = v => v.toFixed(3);
const fscore = v => (st.mode === 'hybrid' && st.fusion === 'rrf' ? v.toFixed(4) : v.toFixed(2));   // RRF scores are small
const FUSION = {rrf: 'RRF', convex: 'weighted'};
const tex = s => {
  const out = window.LABUI?.tex?.(s);
  return out && out !== s ? out : `<span class="retr-math-fallback">${esc(s)}</span>`;
};
// q and d_i in math italic, as the rest of the lab writes symbols (<var> with Unicode math letters)
const MI = window.LABUI?.mi || (x => x);
const qv = () => `<var class="retr-var q">${MI('q')}</var>`;
const dv = i => `<var class="retr-var d">${MI('d')}<sub>${i + 1}</sub></var>`;
const dvKey = key => dv(R.DOCS.findIndex(d => d.key === key));
const fd = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3).replace(/^0/, '');   // .052, −.034
const tokOf = () => (st.tokOf === 'q' ? res.query : res.rows[st.tokOf]);
const METHOD = {bm25: {name: 'BM25', info: 'idf × saturated tf, length-normalized'}, tfidf: {name: 'TF-IDF', info: 'tf × idf, each vector scaled to length 1'}};

/* ---------- 01 Input ---------- */
function drawInput() {
  $('#retr-docs').innerHTML = R.DOCS.map((d, i) => `<button type="button" class="retr-doc${st.doc === i ? ' selected' : ''}" data-doc="${i}" aria-pressed="${st.doc === i}">
    <span class="retr-doc-no">${dv(i)}</span><span class="retr-doc-name">${esc(d.title)}</span><span class="retr-doc-text">${esc(d.text)}</span></button>`).join('');
  root.querySelectorAll('#retr-docs [data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- rows shared by the vocabulary and the vectors: 01–16, the query's terms, the last term ---------- */
function shownRows() {
  const ids = [...new Set([...Array(Math.min(FIRST, n)).keys(), ...res.query.ids, n - 1])].sort((a, b) => a - b);
  const out = [];
  let prev = -1;
  ids.forEach(i => { if (i - prev > 1) out.push(null); out.push(i); prev = i; });   // null = ⋮
  return out;
}

/* ---------- 02 Vocabulary ---------- */
function drawVocab() {
  const rows = shownRows(), qset = new Set(res.query.ids), doc = R.INDEX.docs[st.doc];
  $('#retr-vocab').innerHTML = '<div class="retr-vhead"><span>id</span><span>term</span></div>' + rows.map(i => (i === null
    ? '<div class="retr-vgap" aria-hidden="true"><span>⋮</span><span>⋮</span></div>'
    : `<button type="button" class="retr-vrow${qset.has(i) ? ' q' : ''}${doc.tf.has(R.INDEX.vocab[i]) ? ' indoc' : ''}${st.term === R.INDEX.vocab[i] ? ' active' : ''}" data-term="${esc(R.INDEX.vocab[i])}" title="${esc(R.INDEX.vocab[i])}: in ${R.INDEX.df.get(R.INDEX.vocab[i])} of ${R.INDEX.N} documents">
    <span class="retr-vid">${id2(i)}</span><span class="retr-vterm">${esc(R.INDEX.vocab[i])}</span></button>`)).join('');
  root.querySelectorAll('#retr-vocab [data-term]').forEach(b => b.addEventListener('click', () => selectTerm(b.dataset.term, st.doc, true)));
}

/* ---------- 03 Sparse vectors: one row per term, one column per vector ---------- */
function drawVectors() {
  const rows = shownRows(), qset = new Set(res.query.ids), cells = [], ncol = res.rows.length;
  cells.push('<span class="retr-mhead retr-mcorner">id</span>', `<span class="retr-mhead q" title="the query">${qv()}</span>`);
  res.rows.forEach((r, d) => cells.push(`<button type="button" class="retr-mhead${st.doc === d ? ' selected' : ''}" data-doc="${d}" title="${esc(r.title)}" aria-label="${r.key}, ${esc(r.title)}">${dv(d)}</button>`));
  rows.forEach(i => {
    if (i === null) { for (let c = 0; c < ncol + 2; c++) cells.push('<span class="retr-mgap" aria-hidden="true">⋮</span>'); return; }
    const t = R.INDEX.vocab[i], inq = qset.has(i);
    cells.push(`<button type="button" class="retr-mid${inq ? ' q' : ''}${st.term === t ? ' active' : ''}" data-term="${esc(t)}" title="term ${id2(i)}: ${esc(t)}">${id2(i)}</button>`);
    cells.push(`<span class="retr-cell q ${inq ? 'one' : 'zero'}" title="q, ${esc(t)}: ${inq ? 1 : 0}">${inq ? 1 : 0}</span>`);
    res.rows.forEach((r, d) => {
      const v = r.vector[i], active = st.term === t && st.doc === d;
      const shade = v > 0 ? Math.min(1, v / (st.method === 'bm25' ? 3 : 0.6)) : 0;
      cells.push(`<button type="button" class="retr-cell${v === 0 ? ' zero' : ''}${inq ? ' qrow' : ''}${active ? ' active' : ''}${st.doc === d ? ' colsel' : ''}" style="--w:${shade.toFixed(2)}" data-term="${esc(t)}" data-doc="${d}" aria-pressed="${active}" aria-label="${METHOD[st.method].name} weight of ${esc(t)} in ${r.key}: ${v.toFixed(3)}">${f2(v)}</button>`);
    });
  });
  const grid = $('#retr-matrix');
  grid.style.gridTemplateColumns = `24px 40px repeat(${ncol}, minmax(38px,1fr))`;
  grid.classList.remove('dense', 'learned');
  grid.innerHTML = cells.join('');
  grid.querySelectorAll('.retr-cell[data-term]').forEach(b => b.addEventListener('click', () => selectTerm(b.dataset.term, Number(b.dataset.doc), true)));
  grid.querySelectorAll('.retr-mid').forEach(b => b.addEventListener('click', () => selectTerm(b.dataset.term, st.doc, true)));
  grid.querySelectorAll('.retr-mhead[data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- 02 Tokens (dense): the WordPiece tokens of the query or of a document ---------- */
function tokenRows(tokens) {
  if (tokens.length <= FIRST + 1) return tokens.map((_, i) => i);
  return [...Array(FIRST).keys(), null, tokens.length - 1];
}
const tokClass = t => (/^\[(CLS|SEP|UNK|PAD)\]$/.test(t) ? ' special' : t.startsWith('##') ? ' piece' : '');
function drawTokens() {
  const x = tokOf(), rows = tokenRows(x.tokens);
  $('#retr-tokpick').innerHTML = `<span class="retr-docpick" role="group" aria-label="Text">${[['q', qv()], ...R.DOCS.map((d, i) => [i, dv(i)])].map(([k, lab]) =>
    `<button type="button" data-tok="${k}" class="${String(st.tokOf) === String(k) ? 'active' : ''}" aria-pressed="${String(st.tokOf) === String(k)}" aria-label="${k === 'q' ? 'query' : R.DOCS[k].key}">${lab}</button>`).join('')}</span>`;
  $('#retr-vocab').innerHTML = '<div class="retr-vhead"><span>id</span><span>token</span></div>' + rows.map(i => (i === null
    ? '<div class="retr-vgap" aria-hidden="true"><span>⋮</span><span>⋮</span></div>'
    : `<div class="retr-vrow tok${tokClass(x.tokens[i])}" title="token ${i + 1} of ${x.tokens.length}"><span class="retr-vid">${x.ids ? x.ids[i] : '–'}</span><span class="retr-vterm">${esc(x.tokens[i])}</span></div>`)).join('');
  root.querySelectorAll('#retr-tokpick [data-tok]').forEach(b => b.addEventListener('click', () => {
    st.tokOf = b.dataset.tok === 'q' ? 'q' : Number(b.dataset.tok);
    if (st.tokOf !== 'q') st.doc = st.tokOf;
    draw(false);
  }));
}

/* ---------- 03 Dense vectors: one row per dimension, one column per vector ---------- */
function drawDense() {
  const dim = DENSE.dim, rows = [...Array(FIRST).keys(), null, dim - 1], ncol = res.rows.length, cells = [];
  const shade = v => Math.min(1, Math.abs(v) / 0.12).toFixed(2);
  const gap = '<span class="retr-mgap" aria-hidden="true">⋮</span>', spacer = '<span aria-hidden="true"></span>';
  cells.push('<span class="retr-mhead retr-mcorner">dim</span>', `<span class="retr-mhead q" title="the query">${qv()}</span>`, spacer);
  res.rows.forEach((r, d) => cells.push(`<button type="button" class="retr-mhead${st.doc === d ? ' selected' : ''}" data-doc="${d}" title="${esc(r.title)}" aria-label="${r.key}, ${esc(r.title)}">${dv(d)}</button>`));
  rows.forEach(j => {
    if (j === null) { cells.push(gap, gap, spacer); for (let c = 0; c < ncol; c++) cells.push(gap); return; }
    cells.push(`<span class="retr-mid${st.dim === j ? ' active' : ''}">${j + 1}</span>`);
    const qj = res.query.vector[j];
    cells.push(`<span class="retr-cell dense qd ${qj < 0 ? 'neg' : 'pos'}" style="--w:${shade(qj)}" title="q, dimension ${j + 1}: ${qj}">${fd(qj)}</span>`, spacer);
    res.rows.forEach((r, d) => {
      const v = r.vector[j], active = st.dim === j && st.doc === d;
      cells.push(`<button type="button" class="retr-cell dense ${v < 0 ? 'neg' : 'pos'}${active ? ' active' : ''}${st.doc === d ? ' colsel' : ''}" style="--w:${shade(v)}" data-dim="${j}" data-doc="${d}" aria-pressed="${active}" aria-label="dimension ${j + 1} of ${r.key}: ${v}">${fd(v)}</button>`);
    });
  });
  const grid = $('#retr-matrix');
  grid.style.gridTemplateColumns = `28px minmax(40px,1fr) 2px repeat(${ncol}, minmax(40px,1fr))`;   // every vector column the same width
  grid.classList.add('dense');
  grid.classList.remove('learned');
  grid.innerHTML = cells.join('');
  grid.querySelectorAll('.retr-cell[data-dim]').forEach(b => b.addEventListener('click', () => { st.dim = Number(b.dataset.dim); st.doc = Number(b.dataset.doc); st.open = 'score'; draw(false); }));
  grid.querySelectorAll('.retr-mhead[data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- 03 Learned sparse vectors: one row per vocabulary token the query uses ---------- */
const own = x => new Set(x.tokens);                         // a weighted token the text does not contain is an expansion
function learnedRows() {
  const q = res.query.terms, top = [...q].sort((a, b) => b[2] - a[2]).slice(0, FIRST);
  const ids = top.map(t => t[0]).sort((a, b) => a - b), out = [];
  let prev = -1;
  ids.forEach(j => { if (j - prev > 1) out.push(null); out.push(j); prev = j; });
  if (prev < LEARNED.vocab_size - 1) out.push(null);
  return out;
}
function drawLearned() {
  const rows = learnedRows(), qtok = own(res.query), ncol = res.rows.length, cells = [], name = new Map(res.query.terms.map(([j, t]) => [j, t]));
  const gap = '<span class="retr-mgap" aria-hidden="true">⋮</span>', spacer = '<span aria-hidden="true"></span>';
  cells.push('<span class="retr-mhead retr-mcorner">token</span>', `<span class="retr-mhead q" title="the query">${qv()}</span>`, spacer);
  res.rows.forEach((r, d) => cells.push(`<button type="button" class="retr-mhead${st.doc === d ? ' selected' : ''}" data-doc="${d}" title="${esc(r.title)}" aria-label="${r.key}, ${esc(r.title)}">${dv(d)}</button>`));
  rows.forEach(j => {
    if (j === null) { cells.push(gap, gap, spacer); for (let c = 0; c < ncol; c++) cells.push(gap); return; }
    const t = name.get(j), qw = res.query.map.get(j), exp = !qtok.has(t);
    cells.push(`<button type="button" class="retr-mid lterm${exp ? ' exp' : ''}${st.lterm === j ? ' active' : ''}" data-lterm="${j}" data-doc="${st.doc}" title="${esc(t)}: vocabulary id ${j}${exp ? ', an expansion, not in the query' : ''}">${esc(t)}</button>`);
    cells.push(`<span class="retr-cell q one lq${exp ? ' exp' : ''}" title="q, ${esc(t)}: ${qw}">${qw.toFixed(2)}</span>`, spacer);
    res.rows.forEach((r, d) => {
      const v = r.map.get(j) || 0, active = st.lterm === j && st.doc === d, dexp = v > 0 && !own(r).has(t);
      cells.push(`<button type="button" class="retr-cell${v === 0 ? ' zero' : ''}${dexp ? ' exp' : ''}${active ? ' active' : ''}${st.doc === d ? ' colsel' : ''}" style="--w:${Math.min(1, v / 2.5).toFixed(2)}" data-lterm="${j}" data-doc="${d}" aria-pressed="${active}" aria-label="weight of ${esc(t)} in ${r.key}: ${v}${dexp ? ', an expansion' : ''}">${f2(v)}</button>`);
    });
  });
  const grid = $('#retr-matrix');
  grid.style.gridTemplateColumns = `minmax(0,max-content) minmax(31px,1fr) 1px repeat(${ncol}, minmax(31px,1fr))`;   // the token column fits its longest token
  grid.style.gridTemplateRows = ['var(--retr-head)', ...rows.map(j => (j === null ? '14px' : 'var(--retr-row)'))].join(' ');
  grid.classList.add('learned');
  grid.classList.remove('dense');
  grid.innerHTML = cells.join('');
  grid.querySelectorAll('[data-lterm]').forEach(b => b.addEventListener('click', () => { st.lterm = Number(b.dataset.lterm); st.doc = Number(b.dataset.doc); st.open = 'weights'; draw(false); }));
  grid.querySelectorAll('.retr-mhead[data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- 02 Two rankings (hybrid): what each retriever returns ---------- */
function drawRankings() {
  const entry = (list, r) => (key => (key ? `<button type="button" class="retr-rk" data-doc="${R.DOCS.findIndex(d => d.key === key)}">${dvKey(key)}<small>${(r.find(x => x.key === key).score).toFixed(2)}</small></button>` : '<span class="retr-rk none">—</span>'));
  const sp = entry(res.lists.sparse, res.sparse.rows), dn = entry(res.lists.dense, res.dense.rows);
  $('#retr-vocab').innerHTML = `<div class="retr-rkhead"><span>#</span><span>${METHOD[st.method].name}</span><span>dense</span></div>`
    + R.DOCS.map((_, i) => `<div class="retr-rkrow"><span class="retr-vid">${i + 1}</span>${sp(res.lists.sparse[i])}${dn(res.lists.dense[i])}</div>`).join('');
  root.querySelectorAll('#retr-vocab .retr-rk[data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- 03 Fused scores (hybrid): each list's share, and their sum ---------- */
function drawFused() {
  const rrf = st.fusion === 'rrf', cells = [], fused = res.rows.map(r => r.score), lo = Math.min(...fused), hi = Math.max(...fused);
  const maxPart = Math.max(1e-9, ...res.rows.flatMap(r => [r.part.sparse, r.part.dense]));
  const shade = v => (hi > lo ? (v - lo) / (hi - lo) : 1);        // the fused column: from the lowest to the highest
  const part = (r, side) => (rrf ? (r.rank[side] ? `1/${R.RRF_K + r.rank[side]}` : '—') : r.part[side].toFixed(2));
  cells.push('<span class="retr-mhead retr-mcorner"></span>', `<span class="retr-mhead txt">${METHOD[st.method].name}</span>`, '<span class="retr-mhead txt">dense</span>', '<span class="retr-mhead txt">fused</span>');
  res.rows.forEach((r, d) => {
    const sel = st.doc === d ? ' hsel' : '';
    cells.push(`<button type="button" class="retr-mid hdoc${sel}" data-doc="${d}" aria-label="${r.key}, ${esc(r.title)}">${dv(d)}</button>`);
    for (const side of ['sparse', 'dense']) {
      cells.push(`<button type="button" class="retr-cell${r.part[side] === 0 ? ' zero' : ''}${sel}" style="--w:${(0.5 * r.part[side] / maxPart).toFixed(2)}" data-doc="${d}" title="${r.rank[side] ? `rank ${r.rank[side]} in the ${side === 'sparse' ? METHOD[st.method].name : 'dense'} list` : 'not in the list'}">${part(r, side)}</button>`);
    }
    cells.push(`<button type="button" class="retr-cell fused${sel}" style="--w:${shade(r.score).toFixed(2)}" data-doc="${d}">${fscore(r.score)}</button>`);
  });
  const grid = $('#retr-matrix');
  grid.style.gridTemplateColumns = 'minmax(26px,auto) repeat(3, minmax(52px,1fr))';
  grid.style.gridTemplateRows = '';
  grid.classList.add('hybrid');
  grid.classList.remove('dense', 'learned');
  grid.innerHTML = cells.join('');
  grid.querySelectorAll('[data-doc]').forEach(b => b.addEventListener('click', () => { st.doc = Number(b.dataset.doc); st.open = 'weights'; draw(false); }));
}

/* ---------- 04 Output ---------- */
function drawOutput() {
  const max = Math.max(1e-9, ...res.ranks.map(r => r.score)), picked = new Set(res.picks.map(p => p.key));
  $('#retr-ranks').innerHTML = res.ranks.map(r => `<button type="button" class="retr-rank-row${picked.has(r.key) ? ' hit' : ''}${R.INDEX.docs[st.doc].key === r.key ? ' selected' : ''}" data-doc="${R.DOCS.findIndex(d => d.key === r.key)}" aria-label="${r.key}, ${esc(r.title)}: score ${fscore(r.score)}${picked.has(r.key) ? ', in the top-k' : ''}">
    <span class="retr-rank-id">${dvKey(r.key)}</span><span class="retr-rank-bar"><i style="width:${(100 * Math.max(0, r.score) / max).toFixed(1)}%"></i></span><span class="retr-rank-score">${fscore(r.score)}</span></button>`).join('');
  root.querySelectorAll('#retr-ranks [data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- the header: tabs, title, lead and key facts, as on the Architectures page ---------- */
const TABS = () => document.querySelectorAll('#retr-tabs [data-mode]');
const span = xs => { const a = Math.min(...xs), b = Math.max(...xs); return a === b ? String(a) : `${a}–${b}`; };
const card = (id, label = id.split('/').pop()) => `<a href="https://huggingface.co/${esc(id)}" target="_blank" rel="noopener noreferrer" title="${esc(id)}">${esc(label)}</a>`;
function drawHeader() {   // stat values with markup are this file's own strings
  const H = {
    sparse: ['Sparse retrieval', 'Counted words: a 1 per query term, a BM25 or TF-IDF weight per document term.',
      () => [['Model', 'BM25 · TF-IDF'], ['Vocabulary', `${n} terms, from the documents`], ['Dimensions', `${n}, one per term`], ['Index', 'inverted']]],
    learned: ['Learned sparse retrieval', 'A trained model weights the vocabulary and adds related terms the text does not contain.',
      () => [['Model', card(LEARNED.model, LEARNED.model.includes('opensearch-neural-sparse-encoding-') ? `OpenSearch · ${LEARNED.model.split('encoding-').pop()}` : undefined)],
        ['Vocabulary', `${V()} WordPiece tokens`], ['Dimensions', `${V()}, one per token`], ['Index', 'inverted']]],
    dense: ['Dense retrieval', `A trained encoder maps every text to ${DENSE ? DENSE.dim : 384} numbers, so related meanings point the same way.`,
      () => [['Model', card(DENSE.model)], ['Vocabulary', `${DENSE.vocab_size.toLocaleString('en-US')} WordPiece tokens`], ['Dimensions', String(DENSE.dim)], ['Index', 'nearest-neighbour search']]],
    hybrid: ['Hybrid retrieval', 'Two retrievers, one list: the BM25 and the dense ranking, fused into one.',
      () => [['Model', `BM25 + ${card(DENSE.model)}`], ['Vocabulary', `${n} terms + ${DENSE.vocab_size.toLocaleString('en-US')} tokens`],
        ['Dimensions', `${n} + ${DENSE.dim}`], ['Index', 'inverted + nearest-neighbour']]],
  }[st.mode];
  $('#retr-title').textContent = H[0];
  $('#retr-lead').textContent = H[1];
  $('#retr-stats').innerHTML = H[2]().map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v.includes('<') ? v : esc(v)}</dd></div>`).join('');
  TABS().forEach(b => { const on = b.dataset.mode === st.mode; b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on)); });
}

/* ---------- the boxes on the arrows ---------- */
function drawLinks() {
  const mode = st.mode, hybrid = mode === 'hybrid', model = mode === 'learned' || mode === 'dense', words = R.INDEX.docs.reduce((s, d) => s + R.analyze(d.text).length, 0);
  $('#retr-h-vocab').textContent = hybrid ? 'Two rankings' : model ? 'Tokens' : 'Vocabulary';
  $('#retr-h-vectors').textContent = {sparse: 'Sparse vectors', learned: 'Learned sparse vectors', dense: 'Dense vectors', hybrid: 'Fused scores'}[mode];
  $('#retr-link-words-title').textContent = hybrid ? 'Two retrievers' : model ? 'Tokenize' : 'unique words';
  $('#retr-link-words-info').textContent = hybrid ? `${METHOD[st.method].name} and dense, same query` : model ? `WordPiece, ${V()} tokens` : `${words} words → ${n} terms`;
  $('#retr-link-weights-title').textContent = {sparse: 'Weighting', learned: 'Encoder + MLM head', dense: 'Encoder', hybrid: 'Fusion'}[mode];
  $('#retr-link-weights-info').textContent = {sparse: '', learned: 'a weight for every vocabulary token', dense: `MiniLM, ${DENSE && DENSE.layers} layers, mean pooling`, hybrid: ''}[mode];
  $('#retr-link-score-title').textContent = hybrid ? 'Sort by fused score' : 'Score query against each doc.';
  $('#retr-link-score-info').innerHTML = hybrid ? (st.fusion === 'rrf' ? `Σ 1 / (${R.RRF_K} + rank)` : `½ · ${METHOD[st.method].name} + ½ · dense, normalized`)
    : `<var>${MI('s')}</var> = <var class="q">${MI('q')}</var> · <var>${MI('d')}</var>`;
  root.querySelector('.retr-seg2.retr-method').hidden = mode !== 'sparse';
  root.querySelector('.retr-seg2.retr-fusion').hidden = !hybrid;
  $('#retr-tokpick').hidden = !model;
  $('#retr-vocab').classList.toggle('tokens', model);
  $('#retr-vocab').classList.toggle('rankings', hybrid);
  root.querySelectorAll('[data-method]').forEach(b => { const on = b.dataset.method === st.method; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
  root.querySelectorAll('[data-fusion]').forEach(b => { const on = b.dataset.fusion === st.fusion; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
  root.querySelectorAll('.retr-link').forEach(l => {
    const on = l.dataset.link === st.open;
    l.classList.toggle('open', on);
    l.querySelector('.retr-link-main').setAttribute('aria-expanded', String(on));
  });
  $('#retr-flow').dataset.open = st.open;
}

/* ---------- details ---------- */
function detailWords() {
  const d = R.INDEX.docs[st.doc], words = R.analyze(d.text), seenBefore = new Set();
  R.INDEX.docs.slice(0, st.doc).forEach(x => x.terms.forEach(t => seenBefore.add(t)));
  const kept = words.filter(w => w.term), distinct = new Set(kept.map(w => w.term)), fresh = [...distinct].filter(t => !seenBefore.has(t));
  const word = w => (w.term ? `<span class="retr-w kept${w.why === 'plural' ? ' plural' : ''}${res.query.known.includes(w.term) ? ' q' : ''}" title="term ${id2(R.INDEX.pos.get(w.term))}">${esc(w.why === 'plural' ? `${w.word}→${w.term}` : w.term)}<sub>${id2(R.INDEX.pos.get(w.term))}</sub></span>`
    : `<span class="retr-w stop" title="stop word: dropped">${esc(w.word)}</span>`);
  const q = res.query;
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>${dv(st.doc)} · ${esc(d.title)}</b>${docPicker()}</div>
      <p class="retr-words">${words.map(word).join(' ')}</p>
      <p class="retr-dnote">${words.length} words → ${kept.length} terms (${distinct.size} different). ${fresh.length ? `${fresh.length} of them are new to the vocabulary, and get the next free ids.` : 'All of them were already in the vocabulary.'}</p></div>
    <div><ol class="retr-rules">
      <li><b>Lowercase</b>, split at everything that is not a letter or digit: “10:00” → “10”, “00”.</li>
      <li><b>Drop stop words</b> (${R.STOP.size} common words like ${['the', 'is', 'at', 'from'].map(w => `“${w}”`).join(', ')}): they are in nearly every text.</li>
      <li><b>Singular</b>, roughly: “visitors” → “visitor”, “dishes” → “dish”. No other stemming: “borrowed” stays “borrowed”.</li>
      <li><b>Unique words</b>, reading ${R.DOCS[0].key} to ${R.DOCS[R.DOCS.length - 1].key}: each new term gets the next id, 01 … ${n}.</li></ol>
      <p class="retr-dnote"><b>The query</b> goes through the same steps: ${q.words.map(w => (w.term ? `<code class="${R.INDEX.pos.has(w.term) ? 'in' : 'out'}">${esc(w.term)}</code>` : `<s>${esc(w.word)}</s>`)).join(' ')}.
      ${q.unknown.length ? `${q.unknown.map(t => `<code class="out">${esc(t)}</code>`).join(', ')} ${q.unknown.length > 1 ? 'are' : 'is'} in no document, so ${q.unknown.length > 1 ? 'they have' : 'it has'} no row in the vectors and can never match.` : 'Every query term is in the vocabulary.'}</p></div></div>`;
}
function detailWeights() {
  const d = R.INDEX.docs[st.doc], t = st.term && R.INDEX.pos.has(st.term) ? st.term : (res.query.known[0] || R.INDEX.vocab[0]), w = R.weight(st.method, t, d);
  const row = res.rows[st.doc], nz = R.nonzero(row.vector);
  const formula = st.method === 'bm25'
    ? [String.raw`\displaystyle d_t=\operatorname{idf}(t)\cdot\frac{\operatorname{tf}\,(k_1+1)}{\operatorname{tf}+k_1\big(1-b+b\,\frac{|d|}{\overline{|d|}}\big)}`, String.raw`\displaystyle \operatorname{idf}(t)=\ln\!\Big(1+\frac{N-\operatorname{df}+0.5}{\operatorname{df}+0.5}\Big)`]
    : [String.raw`\displaystyle d_t=\frac{\operatorname{tf}\cdot\operatorname{idf}(t)}{\lVert \operatorname{tf}\cdot\operatorname{idf}\rVert_2}`, String.raw`\displaystyle \operatorname{idf}(t)=1+\ln\frac{N+1}{\operatorname{df}+1}`];
  const nums = st.method === 'bm25'
    ? `tf = <b>${w.tf}</b> · df = <b>${w.df}</b> of N = ${w.N} · idf = <b>${f3(w.idf)}</b> · |d| = <b>${w.length}</b> terms, average ${w.avgLength.toFixed(1)} · tf part = <b>${f3(w.tfPart)}</b> → d<sub>t</sub> = <b class="retr-result">${f3(w.value)}</b>`
    : `tf = <b>${w.tf}</b> · df = <b>${w.df}</b> of N = ${w.N} · idf = <b>${f3(w.idf)}</b> · tf × idf = <b>${f3(w.raw)}</b> · row length = <b>${f3(w.norm)}</b> → d<sub>t</sub> = <b class="retr-result">${f3(w.value)}</b>`;
  const why = st.method === 'bm25'
    ? ['<b>Rare terms count more</b>: idf is high for a term in few documents, near 0 for one in all of them.', '<b>Repeats help less and less</b>: the tf part grows toward k₁ + 1 = 2.2, never past it (k₁ = 1.2).', '<b>Long documents are discounted</b>: b = 0.75 of a document’s length relative to the average enters the denominator.']
    : ['<b>Rare terms count more</b>: idf is high for a term in few documents.', '<b>More occurrences, more weight</b>: tf counts them, without a ceiling.', '<b>Every row has length 1</b>: each weight is divided by the length of its document’s vector, so long documents don’t win by being long.'];
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>${METHOD[st.method].name}: one weight per term and document</b>${docPicker()}</div>
      <div class="retr-eqs">${formula.map(f => `<div class="retr-eq">${tex(f)}</div>`).join('')}</div>
      <p class="retr-trace"><span>Entry <code>${esc(t)}</code> (term ${id2(R.INDEX.pos.get(t))}) in ${dv(st.doc)}:</span> <span>${nums}</span></p>
      <p class="retr-dnote">${dv(st.doc)} has ${nz} nonzero weights out of ${n}: ${(100 * (n - nz) / n).toFixed(0)}% of its vector is zero. Click any entry of the matrix to see its numbers.</p></div>
    <div><ul class="retr-rules">${why.map(x => `<li>${x}</li>`).join('')}</ul>
      <p class="retr-dnote">${st.method === 'bm25' ? 'With these weights, q · d is exactly the BM25 score: BM25 adds up the weights of the query’s terms.' : 'Cosine similarity would also scale q to length 1, dividing every score by the same √m (m = query terms): the order stays the same.'}</p></div></div>`;
}
function detailScore() {
  const q = res.query, rows = res.ranks;
  const line = r => {
    const parts = r.products.map(p => `<span class="retr-prod${p.w > 0 ? ' hit' : ''}" title="${esc(p.term)}">1 × ${f2(p.w)}</span>`);
    return `<button type="button" class="retr-sum${R.INDEX.docs[st.doc].key === r.key ? ' selected' : ''}" data-doc="${R.DOCS.findIndex(d => d.key === r.key)}"><span class="retr-sum-id">${qv()} · ${dvKey(r.key)}</span><span class="retr-sum-terms">${parts.length ? parts.join(' + ') : '0'}</span><span class="retr-sum-eq">= <b>${r.score.toFixed(2)}</b></span></button>`;
  };
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>One dot product per document</b><span class="retr-meta">terms where q = 1: ${q.ids.map(i => `<code>${esc(R.INDEX.vocab[i])}</code>`).join(' ') || 'none'}</span></div>
      <div class="retr-eqs"><div class="retr-eq">${tex(String.raw`\displaystyle s(q,d)=q\cdot d=\sum_{t=1}^{${n}} q_t\,d_t=\sum_{t\,\in\,q} d_t`)}</div></div>
      <div class="retr-sums">${rows.map(line).join('')}</div></div>
    <div><ul class="retr-rules">
      <li><b>Only the query’s terms count.</b> In ${q.ids.length ? `the other ${n - q.ids.length}` : `all ${n}`} rows q<sub>t</sub> = 0, so whatever a document holds there adds nothing.</li>
      <li><b>So the sum skips them.</b> A search engine stores, for each term, the documents that contain it (an inverted index), and reads only the lists of the query’s terms${q.ids.length ? `: ${q.ids.length} here` : ': none here'}.</li>
      <li><b>No shared term, no score.</b> A document that contains none of the query’s terms scores 0, however related its meaning: “purchase” never matches “buy”.</li></ul></div></div>`;
}
function firstSplit(lists) {
  for (const ts of lists) for (let i = 0; i < ts.length - 1; i++) {
    if (ts[i].startsWith('##') || !ts[i + 1].startsWith('##')) continue;
    let j = i + 1;
    while (j < ts.length && ts[j].startsWith('##')) j++;
    return ts.slice(i, j);
  }
  return null;
}
function detailTokens() {
  const d = res.rows[st.doc], q = res.query;
  const chip = (t, id) => (id == null ? `<span class="retr-w tok${tokClass(t)}">${esc(t)}</span>` : `<span class="retr-w tok${tokClass(t)}" title="id ${id}">${esc(t)}<sub>${id}</sub></span>`);
  const idOf = (x, i) => (x.ids ? x.ids[i] : null);
  const splits = d.tokens.filter((t, i) => !t.startsWith('##') && (d.tokens[i + 1] || '').startsWith('##')).length;
  const ex = firstSplit([...res.rows, q].map(x => x.tokens));   // a word the tokenizer split, for the rule below
  const exText = ex ? `“${ex.map(t => t.replace(/^##/, '')).join('')}” → ${ex.map(t => `<code>${esc(t)}</code>`).join(' ')}` : '';
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>${dv(st.doc)} · ${esc(d.title)}</b>${docPicker()}</div>
      <p class="retr-words">${d.tokens.map((t, i) => chip(t, idOf(d, i))).join(' ')}</p>
      <p class="retr-dnote">${d.tokens.length} tokens, counting [CLS] and [SEP]${splits ? `; ${splits} word${splits > 1 ? 's are' : ' is'} split into pieces (##)` : ''}. ${PLACEHOLDER ? 'Each id stands for one entry of the vocabulary: the same token always gets the same id.' : 'The ids are positions in the model’s vocabulary, the same for every text.'}</p></div>
    <div><ol class="retr-rules">
      <li><b>Lowercase</b> and strip accents, split into words and punctuation: “10:00” → <code>10</code> <code>:</code> <code>00</code>.</li>
      <li><b>WordPiece</b>: each word becomes the longest pieces found in a fixed vocabulary of ${V()}${exText ? `; a rarer word is split: ${exText}` : ''}.</li>
      <li><b>Nothing is dropped</b>: stop words and punctuation stay, because the model reads every word in context.</li>
      <li><b>[CLS]</b> first, <b>[SEP]</b> last, at most ${MODEL_DATA().max_length} tokens.</li></ol>
      <p class="retr-dnote"><b>The query</b> ${qv()}: ${q.tokens.map((t, i) => chip(t, idOf(q, i))).join(' ')}. Unlike the sparse vocabulary, this one does not come from the documents: a word that no document contains still gets tokens and a vector.</p></div></div>`;
}
function detailEncoder() {
  const d = res.rows[st.doc], q = res.query, M = DENSE.params ? `, ${(DENSE.params / 1e6).toFixed(1)} M parameters` : '';
  const trace = (lab, x, sym) => `<p class="retr-trace"><span>${lab}:</span> <span>T = <b>${x.tokens.length}</b> tokens · ${x.norm == null ? `${sym} = v / ‖v‖, length <b class="retr-result">1</b>`
    : `‖v‖ = <b>${x.norm.toFixed(3)}</b> → ${sym} = v / ${x.norm.toFixed(3)}, length <b class="retr-result">1</b>`}</span></p>`;
  const eqs = [String.raw`h_1,\dots,h_T=\operatorname{Encoder}(\text{tokens}),\quad h_i\in\mathbb{R}^{${DENSE.dim}}`, String.raw`\displaystyle v=\frac{1}{T}\sum_{i=1}^{T}h_i,\qquad d=\frac{v}{\lVert v\rVert}`];
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>MiniLM: tokens → one vector of length 1</b>${docPicker()}</div>
      <div class="retr-eqs">${eqs.map(f => `<div class="retr-eq">${tex(f)}</div>`).join('')}</div>
      ${trace(dv(st.doc), d, `${MI('d')}<sub>${st.doc + 1}</sub>`)}${trace(qv(), q, MI('q'))}
      <p class="retr-dnote">Model: <code>${esc(DENSE.model)}</code>, ${DENSE.layers} transformer layers, ${DENSE.heads} attention heads${M}. Documents are embedded once, in advance; at search time only the query goes through the model.</p></div>
    <div><ul class="retr-rules">
      <li><b>A vector per token</b>: each of the ${DENSE.layers} layers lets every token look at all the others, so a token’s ${DENSE.dim} numbers depend on the whole text.</li>
      <li><b>Mean pooling</b>: the average of the token vectors stands for the text; dividing by its length puts every text on the unit sphere.</li>
      <li><b>Trained for this</b>: fine-tuned with a contrastive objective on over a billion pairs of related texts, such as a question and its answer, so that related texts get similar vectors.</li>
      <li><b>The numbers have no names</b>: unlike a row of the vocabulary, dimension 17 does not stand for a word.</li></ul></div></div>`;
}
function detailDenseScore() {
  const q = res.query, dim = DENSE.dim, preset = R.PRESETS.find(p => p.id === st.preset);
  const line = r => {
    const first = [0, 1, 2].map(j => `<span class="retr-prod hit">${fd(q.vector[j])}·${fd(r.vector[j])}</span>`).join(' + ');
    return `<button type="button" class="retr-sum${R.INDEX.docs[st.doc].key === r.key ? ' selected' : ''}" data-doc="${R.DOCS.findIndex(d => d.key === r.key)}"><span class="retr-sum-id">${qv()} · ${dvKey(r.key)}</span><span class="retr-sum-terms">${first} + … <span class="retr-prod">(${dim} products)</span></span><span class="retr-sum-eq">= <b>${r.score.toFixed(2)}</b></span></button>`;
  };
  const picked = st.dim !== null ? (() => {
    const j = st.dim, r = res.rows[st.doc], p = q.vector[j] * r.vector[j];
    return `<p class="retr-trace"><span>Dimension ${j + 1} of ${dv(st.doc)}:</span> <span>${MI('q')}<sub>${j + 1}</sub>${MI('d')}<sub>${j + 1}</sub> = ${fd(q.vector[j])} × ${fd(r.vector[j])} = <b class="retr-result">${p < 0 ? '−' : ''}${Math.abs(p).toFixed(4)}</b>, one of ${dim} products that add up to ${r.score.toFixed(2)}.</span></p>`;
  })() : '';
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>One dot product per document</b><span class="retr-meta">${dim} products each</span></div>
      <div class="retr-eqs"><div class="retr-eq">${tex(String.raw`\displaystyle s(q,d)=q\cdot d=\sum_{j=1}^{${dim}} q_j\,d_j=\cos(q,d)`)}</div></div>
      ${picked}
      <div class="retr-sums">${res.ranks.map(line).join('')}</div>
      ${compareBlock()}</div>
    <div><ul class="retr-rules">
      <li><b>Every dimension counts.</b> No entry is 0, so there is nothing to skip and no inverted index: large collections use approximate nearest-neighbour search instead (such as HNSW).</li>
      <li><b>Meaning over words.</b> Related texts score high without sharing a word: try “Where can I get something to eat?”.</li>
      <li><b>Something is always nearest.</b> Scores are never all 0. Even when no document answers (try the parking question), the top k go to the model, so the prompt must let it say it does not know.</li>
      <li><b>Exact words can slip.</b> Names, numbers and rare codes are where sparse retrieval is strong; many systems run both and merge the rankings (hybrid search).</li></ul></div></div>`;
}
function compareBlock(cols = [[`Sparse (${METHOD[st.method].name})`, sres], ['Learned sparse', lres], [PLACEHOLDER ? 'Dense' : 'Dense (MiniLM)', dres]].filter(c => c[1])) {
  const preset = R.PRESETS.find(p => p.id === st.preset);
  const answers = key => preset && preset.source && R.DOCS.find(d => d.key === key).id === preset.source;
  const col = ([title, r, fmt = v => v.toFixed(2)]) => `<div class="retr-cmp-col"><b>${title}</b>${r.ranks.map((x, i) => `<span class="retr-cmp-row${answers(x.key) ? ' ans' : ''}"><i>${i + 1}</i>${dvKey(x.key)}<em>${fmt(x.score)}</em></span>`).join('')}</div>`;
  return `<div class="retr-cmp" style="--cols:${cols.length}" aria-label="The rankings for this question">${cols.map(col).join('')}
    <p class="retr-dnote">${preset && preset.source ? `Marked: ${dvKey(R.DOCS.find(d => d.id === preset.source).key)}, the document that answers this question.` : 'No document answers this question: whatever is retrieved, the model should say it does not know.'}</p></div>`;
}
function detailLearnedEncoder() {
  const d = res.rows[st.doc], q = res.query, nz = d.terms.length;
  const chips = (x, n) => [...x.terms].sort((a, b) => b[2] - a[2]).slice(0, n).map(([j, t, w]) =>
    `<span class="retr-w lw${own(x).has(t) ? '' : ' exp'}${st.lterm === j ? ' active' : ''}" title="vocabulary id ${j}">${esc(t)}<sub>${w.toFixed(2)}</sub></span>`).join(' ');
  const eqs = [String.raw`w_i=\operatorname{MLM}(h_i)\in\mathbb{R}^{${LEARNED.vocab_size}}`, String.raw`\displaystyle d_j=\max_{1\le i\le T}\,\log\bigl(1+\operatorname{ReLU}(w_{ij})\bigr)`];
  const sel = st.lterm !== null ? (() => {
    const t = (q.terms.find(x => x[0] === st.lterm) || d.terms.find(x => x[0] === st.lterm) || [, '?'])[1], w = d.map.get(st.lterm) || 0;
    return `<p class="retr-trace"><span>Entry <code>${esc(t)}</code> (id ${st.lterm}) in ${dv(st.doc)}:</span> <span>${MI('d')}<sub>j</sub> = <b class="retr-result">${w.toFixed(3)}</b>${w === 0 ? ': no position gave it a positive logit, so it stays 0.' : own(d).has(t) ? ', a word of the text.' : `, an <b>expansion</b>: ${dv(st.doc)} never says “${esc(t)}”.`}</span></p>`;
  })() : '';
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>Encoder + MLM head: a weight for every vocabulary token</b>${docPicker()}</div>
      <div class="retr-eqs">${eqs.map(f => `<div class="retr-eq">${tex(f)}</div>`).join('')}</div>
      ${sel}
      <p class="retr-dnote"><b>${dv(st.doc)}</b>, largest weights: ${chips(d, 18)}</p>
      <p class="retr-dnote"><b>The query</b> ${qv()}: ${chips(q, 18)}</p>
      <p class="retr-dnote">${dv(st.doc)} has ${nz} nonzero weights out of ${V()} (${(100 * nz / LEARNED.vocab_size).toFixed(2)}%), the query ${q.terms.length}. Purple: expansions. Model: <code>${esc(LEARNED.model)}</code>${LEARNED.params ? `, ${(LEARNED.params / 1e6).toFixed(0)} M parameters` : ''}; it encodes the query with the same network.</p></div>
    <div><ul class="retr-rules">
      <li><b>Which words fit here?</b> The masked-language-model head scores every vocabulary token at every position, so words the text never says can get weight too: expansion.</li>
      <li><b>max and log(1 + ReLU)</b>: negative scores become 0, a token counts with its strongest position, and the log damps large values, much as tf saturates in BM25.</li>
      <li><b>Learned, and kept sparse</b>: SPLADE-style models are trained on query–passage pairs with a penalty on nonzero weights (FLOPS regularization), so most of the ${V()} stay exactly 0.</li>
      <li><b>Readable</b>: every dimension is a vocabulary token, unlike the numbered dimensions of a dense vector.</li></ul></div></div>`;
}
function detailLearnedScore() {
  const q = res.query;
  const line = r => {
    const parts = r.products.slice(0, 3).map(p => `<span class="retr-prod hit" title="${esc(p.term)}">${esc(p.term)} ${p.q.toFixed(2)}×${p.w.toFixed(2)}</span>`);
    const more = r.products.length > 3 ? ` + … <span class="retr-prod">(${r.products.length} shared terms)</span>` : '';
    return `<button type="button" class="retr-sum${R.INDEX.docs[st.doc].key === r.key ? ' selected' : ''}" data-doc="${R.DOCS.findIndex(d => d.key === r.key)}"><span class="retr-sum-id">${qv()} · ${dvKey(r.key)}</span><span class="retr-sum-terms">${parts.length ? parts.join(' + ') + more : '0 <span class="retr-prod">(no shared term)</span>'}</span><span class="retr-sum-eq">= <b>${r.score.toFixed(2)}</b></span></button>`;
  };
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>One dot product per document</b><span class="retr-meta">over the terms where both are nonzero</span></div>
      <div class="retr-eqs"><div class="retr-eq">${tex(String.raw`\displaystyle s(q,d)=q\cdot d=\sum_{j=1}^{${LEARNED.vocab_size}} q_j\,d_j=\sum_{j:\;q_j>0,\;d_j>0} q_j\,d_j`)}</div></div>
      <div class="retr-sums">${res.ranks.map(line).join('')}</div>
      ${compareBlock()}</div>
    <div><ul class="retr-rules">
      <li><b>Still sparse.</b> Only the ${q.terms.length} nonzero terms of the query can add anything, so a search engine keeps an inverted index, one list per vocabulary token, as for BM25.</li>
      <li><b>Expansion bridges wording.</b> A term the model adds to the query or to a document can match without either text sharing the word: try “Where can I get something to eat?”.</li>
      <li><b>Weights, not counts.</b> The model decides how much each term matters for this text, instead of tf and idf.</li>
      <li><b>Nothing shared, no score.</b> If no weighted term overlaps, the score is 0, as with BM25.</li></ul></div></div>`;
}
function detailRetrievers() {
  const s = res.sparse, d = res.dense, top = Math.max(...s.rows.map(r => r.score));
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>Two retrievers, the same question</b><span class="retr-meta">each returns its own list</span></div>
      ${compareBlock([[`${METHOD[st.method].name} (sparse)`, s], [PLACEHOLDER ? 'Dense' : 'Dense (MiniLM)', d]])}
      <p class="retr-dnote">The ${METHOD[st.method].name} list holds ${res.lists.sparse.length ? `the ${res.lists.sparse.length} document${res.lists.sparse.length > 1 ? 's' : ''} that share a term with the question` : 'no document: none shares a term with the question'}; the dense list holds all ${R.DOCS.length}.</p></div>
    <div><ul class="retr-rules">
      <li><b>Two indexes.</b> The sparse vectors sit in an inverted index, the dense ones in a nearest-neighbour index; both answer the same question independently.</li>
      <li><b>Different scales.</b> ${METHOD[st.method].name} scores have no upper bound (here up to ${top.toFixed(2)}), cosine lies between −1 and 1: adding them as they are would let ${METHOD[st.method].name} decide alone.</li>
      <li><b>Different misses.</b> ${METHOD[st.method].name} finds exact words, names and numbers; dense retrieval finds paraphrases. A document one of them misses can still be found by the other.</li></ul></div></div>`;
}
function detailFusion() {
  const r = res.rows[st.doc], rrf = st.fusion === 'rrf', M = METHOD[st.method].name;
  const f = rrf ? String.raw`\displaystyle s(d)=\sum_{\text{lists } L}\frac{1}{k+\operatorname{rank}_L(d)},\qquad k=${R.RRF_K}`
    : String.raw`\displaystyle \hat s=\frac{s-\min s}{\max s-\min s},\qquad s(d)=\alpha\,\hat s_{\text{sparse}}(d)+(1-\alpha)\,\hat s_{\text{dense}}(d),\quad \alpha=${R.ALPHA}`;
  const term = side => (rrf ? (r.rank[side] ? `1/(${R.RRF_K} + ${r.rank[side]})` : `0 (not in the ${side === 'sparse' ? M : 'dense'} list)`)
    : `${R.ALPHA} × ${r.norm[side].toFixed(2)}`);
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>${rrf ? 'Reciprocal rank fusion' : 'Weighted sum of normalized scores'}</b>${docPicker()}</div>
      <div class="retr-eqs"><div class="retr-eq">${tex(f)}</div></div>
      <p class="retr-trace"><span>${dv(st.doc)}:</span> <span>${rrf ? `rank ${r.rank.sparse || '–'} in ${M}, rank ${r.rank.dense || '–'} in dense` : `${M} ${r.raw.sparse.toFixed(2)} → ${r.norm.sparse.toFixed(2)}, dense ${r.raw.dense.toFixed(3)} → ${r.norm.dense.toFixed(2)}`} · s = ${term('sparse')} + ${term('dense')} = <b class="retr-result">${fscore(r.score)}</b></span></p></div>
    <div><ul class="retr-rules">${rrf ? `
      <li><b>Only ranks count.</b> RRF ignores the scores, so their different scales do not matter and nothing needs tuning.</li>
      <li><b>k = ${R.RRF_K}</b> keeps the top ranks from deciding alone: 1/61 and 1/65 differ little, so a document both lists rank fairly high beats one that only a single list ranks first. The value comes from the original paper, which found it near-optimal but not critical.</li>
      <li><b>Missing means nothing.</b> A document that a list does not return adds 0 from that list.</li>` : `
      <li><b>Normalize first.</b> Min-max scaling puts both lists on 0 to 1, so neither scale wins by default.</li>
      <li><b>Scores, not only ranks.</b> A clear winner stays clear, unlike in RRF; but the result depends on α and on the other documents, through the min and max.</li>
      <li><b>α</b> is tuned on example questions in practice; here it is ${R.ALPHA}, both lists weighing the same.</li>`}</ul></div></div>`;
}
function detailHybridScore() {
  const rrf = st.fusion === 'rrf', M = METHOD[st.method].name;
  const line = r => `<button type="button" class="retr-sum${R.INDEX.docs[st.doc].key === r.key ? ' selected' : ''}" data-doc="${R.DOCS.findIndex(d => d.key === r.key)}"><span class="retr-sum-id">${dvKey(r.key)}</span><span class="retr-sum-terms">${rrf
    ? ['sparse', 'dense'].map(side => (r.rank[side] ? `<span class="retr-prod hit">1/${R.RRF_K + r.rank[side]}</span>` : '<span class="retr-prod">0</span>')).join(' + ')
    : ['sparse', 'dense'].map(side => `<span class="retr-prod${r.part[side] ? ' hit' : ''}">${r.part[side].toFixed(2)}</span>`).join(' + ')}</span><span class="retr-sum-eq">= <b>${fscore(r.score)}</b></span></button>`;
  return `<div class="retr-dgrid">
    <div><div class="retr-dhead"><b>Sort by the fused score</b><span class="retr-meta">${FUSION[st.fusion]}, then the top k</span></div>
      <div class="retr-sums">${res.ranks.map(line).join('')}</div>
      ${compareBlock([[M, res.sparse], [PLACEHOLDER ? 'Dense' : 'Dense (MiniLM)', res.dense], [`Hybrid (${FUSION[st.fusion]})`, res, fscore]])}</div>
    <div><ul class="retr-rules">
      <li><b>Agreement wins.</b> A document that both lists rank high rises to the top, even if neither ranks it first.</li>
      <li><b>Both kinds of question.</b> The ${M} list covers exact words and names, the dense list paraphrases such as “something to eat”.</li>
      <li><b>No new model.</b> Fusion works on the two result lists only; search engines offer it as a built-in step after running both retrievers.</li></ul></div></div>`;
}
function docPicker() {
  return `<span class="retr-docpick" role="group" aria-label="Document">${R.DOCS.map((d, i) => `<button type="button" data-doc="${i}" class="${st.doc === i ? 'active' : ''}" aria-pressed="${st.doc === i}" aria-label="${d.key}">${dv(i)}</button>`).join('')}</span>`;
}
function drawDetail() {
  const box = $('#retr-detail');
  box.innerHTML = {hybrid: {words: detailRetrievers, weights: detailFusion, score: detailHybridScore},
    dense: {words: detailTokens, weights: detailEncoder, score: detailDenseScore},
    learned: {words: detailTokens, weights: detailLearnedEncoder, score: detailLearnedScore},
    sparse: {words: detailWords, weights: detailWeights, score: detailScore}}[st.mode][st.open]();
  box.dataset.open = st.open;
  box.querySelectorAll('[data-doc]').forEach(b => b.addEventListener('click', () => selectDoc(Number(b.dataset.doc))));
}

/* ---------- what the model receives ---------- */
function drawContext() {
  $('#retr-prompt-stat').textContent = !st.k ? '· select k first' : `· ${res.picks.length} document${res.picks.length === 1 ? '' : 's'}`;
  $('#retr-context').textContent = `QUESTION\n${st.question}\n\nRETRIEVED DOCUMENTS\n${!st.k ? '(select k in the Output)' : res.picks.length ? res.picks.map((p, i) => `[${i + 1}] ${p.title} (${p.key}, score ${fscore(p.score)})\n${p.text}`).join('\n\n') : '(nothing retrieved)'}\n\nINSTRUCTION\nAnswer only from these documents and cite them. If they do not answer the question, say so.`;
  $('#retr-answer').textContent = !st.k ? 'Select k in the Output to choose how many documents the model gets.'
    : res.answer ? (res.answer.citation ? `Example of a supported answer: ${res.answer.text} [${res.answer.citation}] (written in advance, not generated).`
      : `Example of a careful answer: ${res.answer.text} (written in advance, not generated).`)
      : 'No answer is shown: the document that answers this question is not among the retrieved ones. This page runs no language model.';
}

/* ---------- state ---------- */
function selectDoc(d) { st.doc = d; if (st.mode !== 'sparse' && st.tokOf !== 'q') st.tokOf = d; draw(false); }
function selectTerm(t, d, open) { st.term = t; st.doc = d; if (open) st.open = 'weights'; draw(false); }
function draw(recompute = true) {
  if (recompute) {
    sres = R.retrieve(st.question, st.method, st.k, st.preset);
    dres = HAS_DENSE ? R.denseRetrieve(st.preset, st.k) : null;
    lres = HAS_LEARNED ? R.learnedRetrieve(st.preset, st.k) : null;
    hres = HAS_DENSE ? R.hybridRetrieve(st.question, st.method, st.preset, st.k, st.fusion) : null;
  }
  if ((st.mode === 'dense' && !dres) || (st.mode === 'learned' && !lres) || (st.mode === 'hybrid' && !hres)) st.mode = 'sparse';
  res = {sparse: sres, learned: lres, dense: dres, hybrid: hres}[st.mode];
  if (st.mode === 'learned' && st.lterm !== null && !res.query.map.has(st.lterm) && !res.rows[st.doc].map.has(st.lterm)) st.lterm = null;
  if (st.term && !R.INDEX.pos.has(st.term)) st.term = null;
  if (st.mode !== 'learned') $('#retr-matrix').style.gridTemplateRows = '';
  if (st.mode !== 'hybrid') $('#retr-matrix').classList.remove('hybrid');
  drawInput();
  if (st.mode === 'dense') { drawTokens(); drawDense(); } else if (st.mode === 'learned') { drawTokens(); drawLearned(); }
  else if (st.mode === 'hybrid') { drawRankings(); drawFused(); } else { drawVocab(); drawVectors(); }
  drawOutput(); drawLinks(); drawHeader(); drawDetail(); drawContext();
}
function init() {
  if (st.ready) return;
  st.ready = true;
  const sel = $('#retr-preset');
  // The query box is one dropdown: the visible box shows the whole question (it wraps), and a
  // transparent native <select> on top of it opens the list, so mouse, keyboard and phones work as usual.
  sel.innerHTML = R.PRESETS.map(p => `<option value="${esc(p.id)}">${esc(p.question)}</option>`).join('');
  sel.value = st.preset;
  $('#retr-qtext').textContent = st.question;
  sel.addEventListener('change', () => {
    st.preset = sel.value;
    st.question = R.PRESETS.find(p => p.id === st.preset).question;
    $('#retr-qtext').textContent = st.question;
    st.term = null; draw();
  });
  $('#retr-dense-source').textContent = [
    PLACEHOLDER || L_PLACEHOLDER ? `The ${[PLACEHOLDER && 'dense', L_PLACEHOLDER && 'learned sparse'].filter(Boolean).join(' and ')} numbers in this build are illustrative; the script writes the models’.` : '',
    HAS_DENSE && !PLACEHOLDER && DENSE.revision ? `MiniLM revision ${DENSE.revision.slice(0, 10)}.` : '',
    HAS_LEARNED && !L_PLACEHOLDER && LEARNED.revision ? `Learned sparse model revision ${LEARNED.revision.slice(0, 10)}.` : ''].filter(Boolean).join(' ');
  $('#retr-k').value = st.k ? String(st.k) : '';
  $('#retr-k').addEventListener('change', e => { st.k = Number(e.target.value) || 0; draw(); });
  root.querySelectorAll('[data-method]').forEach(b => b.addEventListener('click', () => { st.method = b.dataset.method; st.open = 'weights'; draw(); }));
  root.querySelectorAll('[data-fusion]').forEach(b => b.addEventListener('click', () => { st.fusion = b.dataset.fusion; st.open = 'weights'; draw(); }));
  root.querySelectorAll('.retr-link-main').forEach(b => b.addEventListener('click', () => { st.open = b.closest('.retr-link').dataset.link; drawLinks(); drawDetail(); }));
  TABS().forEach(b => {
    if (((b.dataset.mode === 'dense' || b.dataset.mode === 'hybrid') && !HAS_DENSE) || (b.dataset.mode === 'learned' && !HAS_LEARNED)) { b.disabled = true; b.title = 'Needs the output of web/scripts/make_embeddings.py'; }
    b.addEventListener('click', () => { st.mode = b.dataset.mode; st.term = null; st.dim = null; st.lterm = null; st.tokOf = 'q'; draw(false); });
  });
  draw();
}
window.RETRIEVALAPP = {show: init, state: st};
if (!root.closest('.page').hidden) init();
})();
