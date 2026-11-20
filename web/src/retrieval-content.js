/* Retrieval page: sparse retrieval as a dot product of sparse vectors.
 *   query     q ∈ {0,1}^n   a 1 for every vocabulary term in the query (the sum of their one-hot vectors)
 *   document  d ∈ R^n       BM25 or TF-IDF weights of its terms, 0 for every other term
 *   score     s(q, d) = q · d = Σ_t q_t d_t = the sum of d's weights over the query's terms
 * Five fictional documents (a made-up aquarium). Everything is computed here, in the page; no model,
 * embeddings or network. llm_lab/reference/retrieval.py has the same rules in Python, tested against
 * scikit-learn's TfidfVectorizer; tests/test_retrieval.py checks this file against it. */
(function (root) {
'use strict';
const DOCS = [
  {id: 'visits', title: 'Visitor guide', parts: [
    'The Moonlight Aquarium opens Tuesday through Sunday from 10:00 to 18:00. The aquarium is closed on Monday.',
    'On Thursday, general admission is free after 16:00. Visitors still need a timed-entry ticket.']},
  {id: 'travel', title: 'Getting here', parts: [
    'Tram line 4 runs from Central Station to Harbor Square every 12 minutes. The aquarium is a five-minute walk from Harbor Square.',
    'A free weekend shuttle runs every 20 minutes from Central Station to the aquarium. The last shuttle departs at 17:30.']},
  {id: 'access', title: 'Accessibility', parts: [
    'The aquarium has step-free entry at the east gate. Elevators serve every exhibit floor.',
    'Wheelchairs can be borrowed without charge at the information desk. Advance reservation is recommended.']},
  {id: 'events', title: 'Exhibits and events', parts: [
    'The jellyfish feeding talk begins at 15:00 every day the aquarium is open. Seats are first come, first served.',
    'The night-ocean exhibit stays open until 21:00 on the first Friday of each month. Special-entry tickets are required.']},
  {id: 'food', title: 'Food and drinks', parts: [
    'The Tidal Café serves lunch from 11:00 to 15:00. Vegan dishes and nut-free snacks are available.',
    'Outside food is allowed at the harbor terrace picnic tables, but not inside the aquarium exhibits.']},
].map((d, i) => ({...d, key: `d${i + 1}`, text: d.parts.join(' ')}));
const PRESETS = [
  {id: 'free', label: 'When is admission free?', question: 'When is admission free at the aquarium?', source: 'visits', answer: 'General admission is free on Thursday after 16:00, but a timed-entry ticket is still required.'},
  {id: 'tram', label: 'How do I get there?', question: 'How do I get to the aquarium from Central Station?', source: 'travel', answer: 'Take tram line 4 from Central Station to Harbor Square; the aquarium is a five-minute walk away.'},
  {id: 'wheelchair', label: 'Can I borrow a wheelchair?', question: 'Can I borrow a wheelchair at the aquarium?', source: 'access', answer: 'Yes. Wheelchairs can be borrowed without charge from the information desk; advance reservation is recommended.'},
  {id: 'jellyfish', label: 'When is the jellyfish talk?', question: 'What time is the jellyfish feeding talk?', source: 'events', answer: 'The jellyfish feeding talk starts at 15:00 on days when the aquarium is open.'},
  {id: 'vegan', label: 'Vegan food at the café?', question: 'Does the café serve vegan dishes?', source: 'food', answer: 'Yes. The Tidal Café has vegan dishes and nut-free snacks.'},
  {id: 'outside', label: 'Can I bring my own food?', question: 'Can I bring outside food into the aquarium?', source: 'food', answer: 'Outside food is allowed at the harbor terrace picnic tables, but not inside the exhibits.'},
  // Worded unlike the document that answers them: no shared term (eat), or only one that every document has (train).
  {id: 'eat', label: 'Something to eat?', question: 'Where can I get something to eat?', source: 'food', answer: 'The Tidal Café serves lunch from 11:00 to 15:00, and you can eat your own food at the harbor terrace picnic tables.'},
  {id: 'train', label: 'A train to the aquarium?', question: 'Is there a train to the aquarium?', source: 'travel', answer: 'No train is mentioned, but tram line 4 runs from Central Station to Harbor Square, a five-minute walk from the aquarium.'},
  // No document answers this one.
  {id: 'parking', label: 'Parking for cars?', question: 'Is there parking for cars?', source: null, answer: 'None of the documents mentions parking, so the model should say it does not know.'},
];
const STOP = new Set('a an and are at be but by can do does every for from how i in into is it me my of on or the their there this to what when where with you'.split(' '));
const K1 = 1.2, B = 0.75;

/* ---------- text → terms ---------- */
// Simplified plurals: -ies → -y, -sses/-shes/-ches/-xes/-zes → drop -es, other final -s → drop it.
function singular(w) {
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 4 && /(ss|sh|ch|x|z)es$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !/(ss|us|is)$/.test(w)) return w.slice(0, -1);
  return w;
}
// Every word of a text, with what becomes of it: kept as a term, or dropped as a stop word.
function analyze(text) {
  const words = String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  return words.map(w => (STOP.has(w) ? {word: w, term: null, why: 'stop'} : {word: w, term: singular(w), why: singular(w) === w ? 'kept' : 'plural'}));
}
const tokenize = text => analyze(text).filter(x => x.term).map(x => x.term);

/* ---------- the vocabulary and the document vectors ---------- */
// Ids in order of first appearance, reading d1, d2, … d5: the way the "unique words" step builds them.
function buildIndex() {
  const docs = DOCS.map(d => {
    const terms = tokenize(d.text), tf = new Map();
    terms.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
    return {...d, terms, tf, length: terms.length};
  });
  const vocab = [], pos = new Map(), df = new Map();
  docs.forEach(d => d.terms.forEach(t => { if (!pos.has(t)) { pos.set(t, vocab.length); vocab.push(t); } }));
  docs.forEach(d => d.tf.forEach((_, t) => df.set(t, (df.get(t) || 0) + 1)));
  const N = docs.length, avgLength = docs.reduce((s, d) => s + d.length, 0) / N;
  return {docs, vocab, pos, df, N, avgLength};
}
const INDEX = buildIndex();

// One document weight, with every number that goes into it.
function weight(method, term, doc, index = INDEX) {
  const tf = doc.tf.get(term) || 0, df = index.df.get(term) || 0, N = index.N;
  if (method === 'bm25') {
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
    const lengthNorm = 1 - B + B * doc.length / index.avgLength;
    const tfPart = tf ? tf * (K1 + 1) / (tf + K1 * lengthNorm) : 0;
    return {method, term, tf, df, N, idf, length: doc.length, avgLength: index.avgLength, lengthNorm, tfPart, value: idf * tfPart};
  }
  const idfOf = t => 1 + Math.log((N + 1) / ((index.df.get(t) || 0) + 1));
  const norm = Math.sqrt([...doc.tf].reduce((s, [t, n]) => s + (n * idfOf(t)) ** 2, 0));
  const idf = idfOf(term), raw = tf * idf;
  return {method, term, tf, df, N, idf, raw, norm, value: raw / norm};
}
function docVector(method, doc, index = INDEX) {
  const v = new Float64Array(index.vocab.length);
  doc.tf.forEach((_, t) => { v[index.pos.get(t)] = weight(method, t, doc, index).value; });
  return v;
}
function queryVector(question, index = INDEX) {
  const words = analyze(question), terms = [...new Set(words.filter(w => w.term).map(w => w.term))];
  const known = terms.filter(t => index.pos.has(t)), unknown = terms.filter(t => !index.pos.has(t));
  const vector = new Float64Array(index.vocab.length);
  known.forEach(t => { vector[index.pos.get(t)] = 1; });
  return {words, terms, known, unknown, vector, ids: known.map(t => index.pos.get(t)).sort((a, b) => a - b)};
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

/* ---------- scores, ranking, top-k ---------- */
function retrieve(question, method = 'bm25', k = 2, presetId = null) {
  const index = INDEX, q = queryVector(question, index);
  const rows = index.docs.map(doc => {
    const vector = docVector(method, doc, index);
    const products = q.ids.map(i => ({id: i, term: index.vocab[i], q: 1, w: vector[i], product: vector[i]}));
    return {...doc, vector, products, score: dot(q.vector, vector)};
  });
  const ranks = [...rows].sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
  const picks = ranks.filter(r => r.score > 1e-12).slice(0, k);
  const preset = PRESETS.find(p => p.id === presetId && p.question === question);
  return {index, method, k, query: q, rows, ranks, picks, answer: answerFor(preset, picks)};
}
// The answer written in advance for an example, if the document it comes from was retrieved
// (or, when no document answers the question, as soon as anything was retrieved).
function answerFor(preset, picks) {
  if (!preset || !picks.length) return null;
  if (!preset.source) return {text: preset.answer, citation: null};
  const hit = picks.find(r => r.id === preset.source);
  return hit ? {text: preset.answer, citation: picks.indexOf(hit) + 1} : null;
}

/* ---------- dense: sentence embeddings computed in advance ----------
 * web/scripts/make_embeddings.py runs all-MiniLM-L6-v2 on every document and example question and
 * writes src/embeddings.json: WordPiece tokens, and the mean of the token vectors scaled to length 1.
 * The build inlines that file here; in node it is read from disk. The score is again a dot product,
 * q · d, which for vectors of length 1 is their cosine similarity. */
const DENSE = (() => {
  const inlined = /*__EMBEDDINGS__*/null;
  if (inlined) return inlined;
  if (typeof module !== 'undefined' && typeof require === 'function') { try { return require('./embeddings.json'); } catch (e) { return null; } }
  return null;
})();
function denseRetrieve(presetId, k = 2) {
  const preset = PRESETS.find(p => p.id === presetId), q = DENSE && DENSE.queries.find(x => x.id === presetId);
  if (!preset || !q) return null;
  const rows = DOCS.map(doc => {
    const e = DENSE.docs.find(x => x.key === doc.key);
    return {...doc, vector: e.vector, tokens: e.tokens, ids: e.ids, norm: e.norm, score: dot(q.vector, e.vector)};
  });
  const ranks = [...rows].sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
  const picks = ranks.slice(0, k);              // some document is always nearest: dense scores are never all 0
  return {method: 'dense', k, query: q, rows, ranks, picks, answer: answerFor(preset, picks)};
}
const nonzero = v => v.reduce((n, x) => n + (x !== 0 ? 1 : 0), 0);

/* ---------- learned sparse: SPLADE-style term weights computed in advance ----------
 * The same script runs opensearch-neural-sparse-encoding-v2-distill and writes src/learned-sparse.json:
 * for each text, the nonzero entries [vocabulary id, token, weight] of a vector over the 30,522 WordPiece
 * tokens, weight = max over positions of log(1 + ReLU(MLM logit)). The query is encoded the same way.
 * The score is q · d over the vocabulary, as in sparse retrieval: only shared nonzero terms add up. */
const LEARNED = (() => {
  const inlined = /*__LEARNED_SPARSE__*/null;
  if (inlined) return inlined;
  if (typeof module !== 'undefined' && typeof require === 'function') { try { return require('./learned-sparse.json'); } catch (e) { return null; } }
  return null;
})();
function learnedRetrieve(presetId, k = 2) {
  const preset = PRESETS.find(p => p.id === presetId), q = LEARNED && LEARNED.queries.find(x => x.id === presetId);
  if (!preset || !q) return null;
  const qmap = new Map(q.terms.map(([j, , w]) => [j, w]));
  const rows = DOCS.map(doc => {
    const e = LEARNED.docs.find(x => x.key === doc.key), map = new Map(e.terms.map(([j, , w]) => [j, w]));
    const products = q.terms.filter(([j]) => map.has(j)).map(([j, t, w]) => ({id: j, term: t, q: w, w: map.get(j), product: w * map.get(j)}))
      .sort((a, b) => b.product - a.product);
    return {...doc, terms: e.terms, map, tokens: e.tokens, ids: e.ids, products, score: products.reduce((s, p) => s + p.product, 0)};
  });
  const ranks = [...rows].sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
  const picks = ranks.filter(r => r.score > 1e-12).slice(0, k);
  return {method: 'learned', k, query: {...q, map: qmap}, rows, ranks, picks, answer: answerFor(preset, picks)};
}
/* ---------- hybrid: fuse the sparse (BM25 or TF-IDF) and the dense ranking ----------
 * Each retriever returns its own list: the sparse one only the documents that share a term with the
 * query (score > 0), the dense one all of them. Two ways to fuse:
 *   rrf:    s(d) = Σ_lists 1 / (RRF_K + rank(d)), RRF_K = 60 (Cormack, Clarke & Büttcher 2009);
 *           a document missing from a list gets nothing from it.
 *   convex: s(d) = α ŝ_sparse(d) + (1 − α) ŝ_dense(d), each score min-max normalized over the
 *           documents (0 for a document the sparse list misses), α = ALPHA. */
const RRF_K = 60, ALPHA = 0.5;
function minmax(xs) {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  return xs.map(x => (hi > lo ? (x - lo) / (hi - lo) : 0));
}
function hybridRetrieve(question, method = 'bm25', presetId = null, k = 2, fusion = 'rrf') {
  const sparse = retrieve(question, method, DOCS.length, presetId), dense = denseRetrieve(presetId, DOCS.length);
  if (!dense) return null;
  const lists = {sparse: sparse.ranks.filter(r => r.score > 1e-12).map(r => r.key), dense: dense.ranks.map(r => r.key)};
  const raw = {sparse: DOCS.map(d => sparse.rows.find(r => r.key === d.key).score), dense: DOCS.map(d => dense.rows.find(r => r.key === d.key).score)};
  const norm = {sparse: minmax(raw.sparse), dense: minmax(raw.dense)};
  const rows = DOCS.map((doc, i) => {
    const rank = {sparse: lists.sparse.indexOf(doc.key) + 1 || null, dense: lists.dense.indexOf(doc.key) + 1 || null};
    const part = fusion === 'rrf'
      ? {sparse: rank.sparse ? 1 / (RRF_K + rank.sparse) : 0, dense: rank.dense ? 1 / (RRF_K + rank.dense) : 0}
      : {sparse: ALPHA * norm.sparse[i], dense: (1 - ALPHA) * norm.dense[i]};
    return {...doc, rank, raw: {sparse: raw.sparse[i], dense: raw.dense[i]}, norm: {sparse: norm.sparse[i], dense: norm.dense[i]}, part, score: part.sparse + part.dense};
  });
  const ranks = [...rows].sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
  const picks = ranks.filter(r => r.score > 1e-12).slice(0, k);
  const preset = PRESETS.find(p => p.id === presetId && p.question === question);
  return {method: 'hybrid', fusion, k, sparse, dense, lists, rows, ranks, picks, answer: answerFor(preset, picks)};
}
const API = {DOCS, PRESETS, STOP, K1, B, singular, analyze, tokenize, buildIndex, INDEX, weight, docVector, queryVector, dot, retrieve, nonzero, answerFor, DENSE, denseRetrieve, LEARNED, learnedRetrieve, RRF_K, ALPHA, minmax, hybridRetrieve};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
root.RLAB = API;
})(typeof window === 'undefined' ? globalThis : window);
