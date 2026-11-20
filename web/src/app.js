/* Attention Explorer — rendering and interaction. Depends on content.js (window.LAB), the
 * pre-rendered MathML table (window.MATHML) and Temml (window.temml) for the math in pop-ups. */
(function () {
'use strict';
const LAB = window.LAB;
const MATH = window.MATHML || {};
const NS = 'http://www.w3.org/2000/svg';
const VW = 720, VH = 612, HX0 = 150, HX1 = 708, PIPE_Y = 508;
const state = { mech: 'mha', step: -1, head: 1, arch: -1, block: 'moe', layer: 'gdn', sparse: 'dsa', keys: 'arch', row: null, tour: -1 };
const $ = s => document.querySelector(s);

/* ---------- small helpers ---------- */
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) {
    if (attrs[k] == null) continue;
    if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]);
  }
  if (parent) parent.appendChild(e);
  return e;
}
function mi(s) { // ASCII letters → Unicode mathematical italic (h is U+210E)
  return [...String(s)].map(c => {
    if (c === 'h') return 'ℎ';
    const u = c.charCodeAt(0);
    if (u >= 65 && u <= 90) return String.fromCodePoint(0x1D434 + u - 65);
    if (u >= 97 && u <= 122) return String.fromCodePoint(0x1D44E + u - 97);
    return c;
  }).join('');
}
/* Math label: italic base with optional subscript and superscript stacked. */
function mlab(p, x, y, base, sub, sup, o = {}) {
  const t = S('text', { x, y, class: 'ml ' + (o.cls || ''), 'text-anchor': o.anchor || 'middle', 'data-h': o.h }, p);
  S('tspan', { text: o.upright ? base : mi(base) }, t);
  if (sub) S('tspan', { text: /^[a-z]+$/.test(sub) ? mi(sub) : sub, class: 'ss', dy: 4 }, t);
  if (sup) S('tspan', { text: sup, class: 'ss', dy: sub ? -11 : -7, dx: sub ? -(String(sub).length * 6) : 0 }, t);
  return t;
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
/* TeX → MathML: pre-rendered at build time when possible, otherwise rendered here by Temml. */
const texCache = new Map();
function tex(t) {
  if (MATH[t]) return MATH[t];
  if (texCache.has(t)) return texCache.get(t);
  let html;
  try { html = window.temml.renderToString(t, { displayMode: false, throwOnError: false }); }
  catch (e) { html = `<code>${esc(t)}</code>`; }
  texCache.set(t, html);
  return html;
}
/* Inline notation in prose: W^UK → W with superscript, S_h → S with subscript, $…$ → TeX. */
function fmtPlain(s) {
  return esc(s)
    .replace(/\b([A-Za-z])\^([A-Z]{1,3})\b/g, (_, b, p) => `<var>${mi(b)}</var><sup>${p}</sup>`)
    .replace(/\b([A-Za-z])_([a-zA-Z0-9]{1,2})\b/g, (_, b, p) => `<var>${mi(b)}</var><sub>${/^[a-z]+$/.test(p) ? mi(p) : p}</sub>`)
    .replace(/x̂/g, () => tex(String.raw`\hat{x}`))
    .replace(/θ/g, () => tex(String.raw`\theta`));
}
function fmtText(s) { return String(s).split(/\$([^$]+)\$/).map((part, i) => i % 2 ? tex(part) : fmtPlain(part)).join(''); }
function math(t) { return `<div class="eq">${tex(t)}</div>`; }
/* TeX number helpers */
const tn = (x, d = 2) => (x < 0 ? '-' : '') + Math.abs(x).toFixed(d);
const tf = n => LAB.fmt(n).replace(/,/g, '{,}');
const tp = x => `${(100 * x).toFixed(1)}\\%`;
const tt = s => `\\text{${String(s).replace(/([{}$&#%_])/g, '\\$1')}}`;
/* Deterministic pseudo-random number in [0, 1): illustrative scores only. */
const hash01 = (a, b) => { const x = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453; return x - Math.floor(x); };

/* ---------- diagram: shared pieces ---------- */
function defs(svg) {
  const d = S('defs', {}, svg);
  for (const c of ['q', 'k']) {
    const p = S('pattern', { id: 'hatch-' + c, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, d);
    S('rect', { width: 5, height: 5, class: 'hatch-bg ' + c }, p);
    S('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'hatch-line ' + c }, p);
  }
  const pp = S('pattern', { id: 'hatch-p', width: 4, height: 4, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, d);
  S('rect', { width: 4, height: 4, class: 'hatch-bg p' }, pp);
  S('line', { x1: 0, y1: 0, x2: 0, y2: 4, class: 'hatch-line p' }, pp);
  const mi2 = S('marker', { id: 'arrow-ink', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, d);
  S('path', { d: 'M0,0.8 L7,4 L0,7.2', class: 'arrowhead ink' }, mi2);
  const mk = S('marker', { id: 'arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, d);
  S('path', { d: 'M0,0.8 L7,4 L0,7.2', class: 'arrowhead' }, mk);
  const ml = S('marker', { id: 'arrow-lat', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, d);
  S('path', { d: 'M0,0.8 L7,4 L0,7.2', class: 'arrowhead lat' }, ml);
  const mq = S('marker', { id: 'arrow-q', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, d);
  S('path', { d: 'M0,0.8 L7,4 L0,7.2', class: 'arrowhead q' }, mq);
}
function rowLabel(svg, y, name, dims, h, extra) {
  S('text', { x: HX0 - 22, y: y - 3, class: 'rl', 'text-anchor': 'end', 'data-h': h, text: name }, svg);
  S('text', { x: HX0 - 22, y: y + 13, class: 'rd', 'text-anchor': 'end', 'data-h': h, text: dims }, svg);
  if (extra) {
    const g = S('g', { 'data-h': extra.h }, svg);
    const w = extra.text.length * 6.4 + 12;
    S('rect', { x: HX0 - 22 - w, y: y + 21, width: w, height: 16, rx: 8, class: 'pill' }, g);
    S('text', { x: HX0 - 22 - w / 2, y: y + 32.5, 'text-anchor': 'middle', class: 'pill-t', text: extra.text }, g);
  }
}
function legend(svg, m) {
  const items = [['cache', m.key === 'gdn' ? 'kept for generation' : 'kept in the KV cache']];
  if (m.key === 'mla') items.unshift(['rope', 'rotary part']);
  let x = HX1;
  for (const [kind, label] of items.slice().reverse()) {
    const t = S('text', { x, y: 19, 'text-anchor': 'end', class: 'lg', text: label }, svg);
    x -= t.getComputedTextLength() + 19;
    if (kind === 'cache') S('rect', { x, y: 8, width: 13, height: 13, rx: 3, class: 'cache' }, svg);
    else S('rect', { x, y: 8, width: 13, height: 13, rx: 2, class: 'bar k rope' }, svg);
    x -= 22;
  }
}
function headButton(svg, m, i, label) {
  const g = S('g', { class: 'head' + (i === state.head ? ' sel' : ''), tabindex: 0, role: 'button',
    'aria-pressed': String(i === state.head), 'aria-label': label }, svg);
  g.addEventListener('click', () => selectHead(i));
  g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectHead(i); } });
  S('title', { text: label }, g);
  return g;
}

/* ---------- diagram: MHA / MQA / GQA ---------- */
function drawStandard(svg, m) {
  const { H, G } = m, unit = 60;
  const qTop = 68, qBot = qTop + unit, kTop = 220, kBot = kTop + unit, vTop = kBot + 12, vBot = vTop + unit;
  const slot = (HX1 - HX0) / H, bw = Math.min(22, slot * 0.55);
  const qx = i => HX0 + slot * (i - 0.5);
  const per = H / G;
  const kvx = g => (qx((g - 1) * per + 1) + qx(g * per)) / 2;
  const selG = LAB.kvHeadOf(m, state.head);

  if (G > 1 && G < H) {
    const pad = Math.min(12, (slot - bw) / 2 - 3);
    for (let g = 1; g <= G; g++) {
      const x0 = qx((g - 1) * per + 1) - bw / 2 - pad, x1 = qx(g * per) + bw / 2 + pad;
      S('rect', { x: x0, y: qTop - 40, width: x1 - x0, height: vBot - qTop + 52, rx: 12, class: 'grp', 'data-h': 'grp' }, svg);
      S('text', { x: x0 + 10, y: qTop - 26, class: 'grp-t', 'data-h': 'grp', text: `group ${g} · ${per} query heads` }, svg);
    }
  }
  const cpad = Math.min(9, (slot - bw) / 2 - 2);
  for (let g = 1; g <= G; g++)
    S('rect', { x: kvx(g) - bw / 2 - cpad, y: kTop - cpad, width: bw + 2 * cpad, height: vBot - kTop + 2 * cpad, rx: 6, class: 'cache', 'data-h': 'cache' }, svg);

  rowLabel(svg, (qTop + qBot) / 2, 'Queries', `${H} × ${m.dh}`, 'qrow', m.pos === 'rope' ? { text: 'RoPE', h: 'pos' } : null);
  rowLabel(svg, (kTop + kBot) / 2, 'Keys', `${G} × ${m.dh}`, 'krow', m.pos === 'rope' ? { text: 'RoPE', h: 'pos' } : null);
  rowLabel(svg, (vTop + vBot) / 2, 'Values', `${G} × ${m.dh}`, 'vrow');

  for (let i = 1; i <= H; i++) {
    const g = LAB.kvHeadOf(m, i), x1 = qx(i), x2 = kvx(g), dy = (kTop - qBot) * 0.5;
    S('path', { d: `M${x1},${qBot} C${x1},${qBot + dy} ${x2},${kTop - dy} ${x2},${kTop}`,
      class: 'link' + (i === state.head ? ' sel' : ''), 'data-h': 'links' + (i === state.head ? ' link' : '') }, svg);
  }
  for (let i = 1; i <= H; i++) {
    S('text', { x: qx(i), y: qTop - 8, 'text-anchor': 'middle', class: 'hn' + (i === state.head ? ' sel' : ''), 'data-h': 'qrow', text: i }, svg);
    const g = headButton(svg, m, i, `Query head ${i}: reads key/value head ${LAB.kvHeadOf(m, i)}`);
    S('rect', { x: qx(i) - bw / 2, y: qTop, width: bw, height: unit, rx: 3, class: 'bar q' + (i === state.head ? ' sel' : ''), 'data-h': 'qrow' + (i === state.head ? ' qsel' : '') }, g);
  }
  for (let g = 1; g <= G; g++) {
    const on = g === selG;
    S('rect', { x: kvx(g) - bw / 2, y: kTop, width: bw, height: unit, rx: 3, class: 'bar k' + (on ? ' sel' : ''), 'data-h': 'krow' + (on ? ' ksel' : '') }, svg);
    S('rect', { x: kvx(g) - bw / 2, y: vTop, width: bw, height: unit, rx: 3, class: 'bar v' + (on ? ' sel' : ''), 'data-h': 'vrow' + (on ? ' vsel' : '') }, svg);
  }
  S('text', { x: HX0 - 22, y: vBot + 40, class: 'note', text: `one bar = one head of ${m.dh} values` }, svg);
}

/* ---------- diagram: MLA ---------- */
function drawMLA(svg, m) {
  const { H } = m, u = 48, ur = 24;               // 128 dims → 48 px, 64 dims → 24 px
  const qTop = 60, qBot = qTop + ur + u, kTop = 186, kBot = kTop + ur + u;
  const latY = 300, latH = 12, vTop = 346, vBot = vTop + u;
  const X0 = HX0 + 44, slot = (HX1 - X0) / H, bw = Math.min(22, slot * 0.55);
  const hx = i => X0 + slot * (i - 0.5);
  const krX = HX0 + 4, latX0 = X0 - 6, latX1 = latX0 + 4 * u;

  S('rect', { x: latX0 - 9, y: latY - 9, width: latX1 - latX0 + 18, height: latH + 18, rx: 8, class: 'cache', 'data-h': 'cache' }, svg);
  S('rect', { x: krX - 8, y: kTop - 8, width: bw + 16, height: ur + 16, rx: 7, class: 'cache', 'data-h': 'cache' }, svg);

  rowLabel(svg, (qTop + qBot) / 2, 'Queries', `${H} × (${m.dh}+${m.dr})`, 'qrow');
  rowLabel(svg, (kTop + kBot) / 2, 'Keys', `${H} × (${m.dh}+${m.dr})`, 'krow');
  rowLabel(svg, latY + latH / 2, 'Latent', `${m.dc}`, 'latent');
  rowLabel(svg, (vTop + vBot) / 2, 'Values', `${H} × ${m.dv}`, 'vrow');

  const kL = hx(1) - bw / 2, kR = hx(H) + bw / 2;
  S('path', { d: `M${latX0},${latY} L${latX1},${latY} L${kR},${kBot + 4} L${kL},${kBot + 4} Z`, class: 'fan', 'data-h': 'wuk' }, svg);
  S('path', { d: `M${latX0},${latY + latH} L${latX1},${latY + latH} L${kR},${vTop - 4} L${kL},${vTop - 4} Z`, class: 'fan', 'data-h': 'wuv' }, svg);
  mlab(svg, latX1 + 26, latY - 12, 'W', null, 'UK', { cls: 'fan-t', anchor: 'start', h: 'wuk' });
  mlab(svg, latX1 + 26, latY + latH + 24, 'W', null, 'UV', { cls: 'fan-t', anchor: 'start', h: 'wuv' });

  S('line', { x1: krX + bw, y1: kTop + ur / 2, x2: kR, y2: kTop + ur / 2, class: 'rail', 'data-h': 'rope' }, svg);
  S('rect', { x: krX, y: kTop, width: bw, height: ur, rx: 2, class: 'bar k rope', 'data-h': 'kr' }, svg);
  mlab(svg, krX + bw / 2, kTop - 14, 'K', null, 'R', { h: 'kr' });

  for (let i = 1; i <= H; i++) {
    S('line', { x1: hx(i), y1: qBot, x2: hx(i), y2: kTop, class: 'link' + (i === state.head ? ' sel' : ''), 'data-h': 'links' + (i === state.head ? ' link' : '') }, svg);
  }
  for (let i = 1; i <= H; i++) {
    const sel = i === state.head, x = hx(i) - bw / 2;
    S('text', { x: hx(i), y: qTop - 8, 'text-anchor': 'middle', class: 'hn' + (sel ? ' sel' : ''), 'data-h': 'qrow', text: i }, svg);
    const g = headButton(svg, m, i, `Query head ${i}: uses its own rebuilt key and value, plus the shared rotary key`);
    S('rect', { x, y: qTop, width: bw, height: ur, rx: 2, class: 'bar q rope' + (sel ? ' sel' : ''), 'data-h': 'qrow qrope' + (sel ? ' qsel' : '') }, g);
    S('rect', { x, y: qTop + ur, width: bw, height: u, rx: 2, class: 'bar q' + (sel ? ' sel' : ''), 'data-h': 'qrow' + (sel ? ' qsel' : '') }, g);
    S('rect', { x, y: kTop, width: bw, height: ur, rx: 2, class: 'bar k rope' + (sel ? ' sel' : ''), 'data-h': 'krow rope' + (sel ? ' ksel' : '') }, svg);
    S('rect', { x, y: kTop + ur, width: bw, height: u, rx: 2, class: 'bar k' + (sel ? ' sel' : ''), 'data-h': 'krow' + (sel ? ' ksel' : '') }, svg);
    S('rect', { x, y: vTop, width: bw, height: u, rx: 2, class: 'bar v' + (sel ? ' sel' : ''), 'data-h': 'vrow' + (sel ? ' vsel' : '') }, svg);
  }
  S('rect', { x: latX0, y: latY, width: latX1 - latX0, height: latH, rx: 3, class: 'bar lat', 'data-h': 'latent' }, svg);
  S('text', { x: HX0 - 22, y: vBot + 32, class: 'note', text: 'bar height ∝ dimensions · per-head keys and values are rebuilt from the latent, not stored' }, svg);
}

/* ---------- diagram: Gated DeltaNet (one GDN layer of Qwen3-Next) ---------- */
function drawGDN(svg, m) {
  const { qk, v, dk } = m.lin, unit = 40;
  const qTop = 58, qBot = qTop + unit, kTop = 124, kBot = kTop + unit, vTop = 190, vBot = vTop + unit;
  const gTop = 270, sTop = 306, sz = 13;
  const slot = (HX1 - HX0) / v, bw = Math.min(10, slot * 0.6);
  const vx = j => HX0 + slot * (j - 0.5);
  const qkx = i => (vx(2 * i - 1) + vx(2 * i)) / 2;
  const j0 = Math.min(state.head, v), i0 = Math.ceil(j0 / 2);
  rowLabel(svg, (qTop + qBot) / 2, 'Queries', `${qk} × ${dk}`, 'qrow', { text: 'conv 4', h: 'conv' });
  rowLabel(svg, (kTop + kBot) / 2, 'Keys', `${qk} × ${dk}`, 'krow', { text: 'conv 4', h: 'conv' });
  rowLabel(svg, (vTop + vBot) / 2, 'Values', `${v} × ${m.lin.dv}`, 'vrow', { text: 'conv 4', h: 'conv' });
  rowLabel(svg, gTop + 6, 'Gates α, β', `${v} + ${v}`, 'gate');
  rowLabel(svg, sTop + sz / 2, 'State S', `${v} × ${dk}×${m.lin.dv}`, 'state');
  // one cache tile behind all states
  S('rect', { x: HX0 - 4, y: sTop - 6, width: HX1 - HX0 + 8, height: sz + 12, rx: 6, class: 'cache', 'data-h': 'cache' }, svg);
  for (let i = 1; i <= qk; i++) {
    const on = i === i0, x = qkx(i);
    for (const [y0, y1] of [[qBot, vTop], [kBot, vTop]]) {
      for (const j of [2 * i - 1, 2 * i]) {
        if (y0 === qBot) continue;
        const dy = (y1 - y0) * 0.5;
        S('path', { d: `M${x},${y0} C${x},${y0 + dy} ${vx(j)},${y1 - dy} ${vx(j)},${y1}`, class: 'link' + (on && j === j0 ? ' sel' : ''), 'data-h': 'links' }, svg);
      }
    }
    S('text', { x, y: qTop - 8, 'text-anchor': 'middle', class: 'hn' + (on ? ' sel' : ''), 'data-h': 'qrow', text: i }, svg);
    S('rect', { x: x - bw, y: qTop, width: 2 * bw, height: unit, rx: 3, class: 'bar q' + (on ? ' sel' : ''), 'data-h': 'qrow' + (on ? ' qsel' : '') }, svg);
    S('rect', { x: x - bw, y: kTop, width: 2 * bw, height: unit, rx: 3, class: 'bar k' + (on ? ' sel' : ''), 'data-h': 'krow' + (on ? ' ksel' : '') }, svg);
  }
  for (let j = 1; j <= v; j++) {
    const on = j === j0, x = vx(j);
    const g = headButton(svg, m, j, `Value head ${j}: own state and gates, uses query/key head ${Math.ceil(j / 2)}`);
    S('rect', { x: x - slot / 2, y: vTop, width: slot, height: sTop + sz - vTop, class: 'hitpad' }, g);   // one target for the whole column
    S('rect', { x: x - bw / 2, y: vTop, width: bw, height: unit, rx: 2, class: 'bar v' + (on ? ' sel' : ''), 'data-h': 'vrow' + (on ? ' vsel' : '') }, g);
    S('rect', { x: x - bw / 2, y: gTop, width: bw / 2 - 0.5, height: 12, rx: 1, class: 'gcell a' + (on ? ' sel' : ''), 'data-h': 'gate' }, g);
    S('rect', { x: x + 0.5, y: gTop, width: bw / 2 - 0.5, height: 12, rx: 1, class: 'gcell b' + (on ? ' sel' : ''), 'data-h': 'gate' }, g);
    S('rect', { x: x - sz / 2, y: sTop, width: sz, height: sz, rx: 2, class: 'scell' + (on ? ' sel' : ''), 'data-h': 'state' }, g);
    if (j % 4 === 1 || on) S('text', { x, y: vBot + 12, 'text-anchor': 'middle', class: 'hn' + (on ? ' sel' : ''), 'data-h': 'vrow', text: j }, svg);
  }
  S('text', { x: HX0 - 22, y: sTop + sz + 34, class: 'note', text: `each query/key head serves two value heads · each value head keeps one ${dk} × ${m.lin.dv} state` }, svg);
}

/* ---------- diagram: sparse attention (DSA or QSA) ---------- */
const SPN = 30;                       // positions drawn; the query is the last one
function sparseScores() {             // illustrative index scores I_{t,s} for the query t = SPN − 1
  return Array.from({ length: SPN }, (_, s) => 0.18 + 0.55 * hash01(29, s) ** 2 + (s === 0 ? 0.5 : 0) + (s >= SPN - 3 ? 0.35 : 0) + (s === SPN - 1 ? 0.3 : 0) + (s === 11 || s === 12 ? 0.45 : 0));
}
function sparseSelection(variant) {
  const sc = sparseScores();
  if (variant === 'qsa') {
    const tail = SPN % 4, nb = (SPN - tail) / 4;
    const bs = Array.from({ length: nb }, (_, b) => (sc[4 * b] + sc[4 * b + 1] + sc[4 * b + 2] + sc[4 * b + 3]) / 4);
    const top = bs.map((x, b) => b).sort((a, b) => bs[b] - bs[a]).slice(0, 2);
    const sel = new Set();
    top.forEach(b => { for (let i = 0; i < 4; i++) sel.add(4 * b + i); });
    for (let s = SPN - tail; s < SPN; s++) sel.add(s);
    return { sc, bs, top: new Set(top), sel, tail, nb };
  }
  const order = sc.map((x, s) => s).sort((a, b) => sc[b] - sc[a]);
  return { sc, sel: new Set(order.slice(0, 8)) };
}
function drawSparse(svg, m) {
  const qsa = state.sparse === 'qsa', Q = LAB.QSA;
  const pitch = (HX1 - HX0) / SPN, w = pitch * 0.66;
  const px = s => HX0 + pitch * (s + 0.5);
  const base = 150, hmax = 84, kiY = 168, latY = 206, latH = 26;
  const sel = sparseSelection(state.sparse);
  const smax = Math.max(...sel.sc);
  rowLabel(svg, base - 34, 'Index scores', qsa ? 'one per block' : 'one per token', 'bars');
  rowLabel(svg, kiY + 6, 'Index keys', qsa ? `1 per block × ${Q.dim}` : `1 × ${m.index.dim}, FP8`, 'ki');
  rowLabel(svg, latY + latH / 2, qsa ? 'Keys, values' : 'Latents', qsa ? `${Q.G} × ${Q.dh} each` : `${m.dc} + ${m.dr}`, 'lat');
  S('rect', { x: HX0 - 4, y: kiY - 5, width: HX1 - HX0 + 8, height: latY + latH - kiY + 10, rx: 6, class: 'cache', 'data-h': 'cache' }, svg);
  if (qsa) {
    for (let b = 0; b < sel.nb; b++) {
      const x0 = px(4 * b) - w / 2, x1 = px(4 * b + 3) + w / 2, on = sel.top.has(b);
      S('rect', { x: x0, y: base - hmax * sel.bs[b] / smax, width: x1 - x0, height: hmax * sel.bs[b] / smax, rx: 2, class: 'ibar' + (on ? ' on-s' : ''), 'data-h': 'bars' }, svg);
      S('rect', { x: x0, y: kiY, width: x1 - x0, height: 12, rx: 2, class: 'kicell', 'data-h': 'ki' }, svg);
      S('path', { d: `M${x0},${latY + latH + 8} V${latY + latH + 12} H${x1} V${latY + latH + 8}`, class: 'blockbr', 'data-h': 'sel' }, svg);
    }
    for (let s = SPN - sel.tail; s < SPN; s++) S('rect', { x: px(s) - w / 2, y: kiY, width: w, height: 12, rx: 1.5, class: 'kicell raw', 'data-h': 'ki' }, svg);
    S('text', { x: (px(SPN - sel.tail) + px(SPN - 1)) / 2, y: latY + latH + 42, 'text-anchor': 'middle', class: 'hn', 'data-h': 'sel', text: 'current' }, svg);
  } else {
    for (let s = 0; s < SPN; s++) {
      const h = hmax * sel.sc[s] / smax, on = sel.sel.has(s);
      S('rect', { x: px(s) - w / 2, y: base - h, width: w, height: h, rx: 1.5, class: 'ibar' + (on ? ' on-s' : ''), 'data-h': 'bars' }, svg);
      S('rect', { x: px(s) - w / 2, y: kiY, width: w, height: 12, rx: 1.5, class: 'kicell', 'data-h': 'ki' }, svg);
    }
  }
  S('line', { x1: HX0, y1: base + 0.5, x2: HX1, y2: base + 0.5, class: 'axis', 'data-h': 'bars' }, svg);
  for (let s = 0; s < SPN; s++) {
    const on = sel.sel.has(s);
    if (qsa) {
      S('rect', { x: px(s) - w / 2, y: latY, width: w, height: latH / 2 - 1, rx: 1.5, class: 'kvk' + (on ? ' on-s' : ''), 'data-h': 'lat' + (on ? ' sel' : '') }, svg);
      S('rect', { x: px(s) - w / 2, y: latY + latH / 2 + 1, width: w, height: latH / 2 - 1, rx: 1.5, class: 'kvv' + (on ? ' on-s' : ''), 'data-h': 'lat' + (on ? ' sel' : '') }, svg);
    } else S('rect', { x: px(s) - w / 2, y: latY, width: w, height: latH, rx: 1.5, class: 'latc' + (on ? ' on-s' : ''), 'data-h': 'lat' + (on ? ' sel' : '') }, svg);
    if (s % 5 === 0 || s === SPN - 1) S('text', { x: px(s), y: latY + latH + 26, 'text-anchor': 'middle', class: 'hn', text: s }, svg);
  }
  // the query
  const qx = px(SPN - 1);
  S('path', { d: `M${qx},${base - hmax - 22} V${base - hmax - 8}`, class: 'branch', 'marker-end': 'url(#arrow-ink)', 'data-h': 'qi' }, svg);
  const qt = S('text', { x: qx - 8, y: base - hmax - 26, 'text-anchor': 'end', class: 'cap', 'data-h': 'qi' }, svg);
  S('tspan', { text: `query t = ${SPN - 1}` }, qt);
  S('tspan', { class: 'cap-m', text: qsa ? `  · ${Q.heads} index heads × ${Q.dim}` : `  · ${m.index.heads} index heads × ${m.index.dim} + ${m.index.heads} weights` }, qt);
  const kept = sel.sel.size;
  const note = qsa
    ? [`toy budget: 2 of ${sel.nb} complete blocks + the current block = ${kept} of ${SPN} tokens`, `(real: ${Q.blocks} blocks = ${LAB.fmt(Q.budget)} tokens). Attention reads only the dark positions.`]
    : [`toy budget: ${kept} of ${SPN} tokens (real: ${LAB.fmt(m.index.topk)}). The indexer scores every earlier token;`, 'attention reads only the dark ones.'];
  note.forEach((line, n) => S('text', { x: HX0 - 22, y: latY + latH + 62 + n * 16, class: 'note', 'data-h': 'budget', text: line }, svg));
}

/* ---------- diagram: inside one head ---------- */
function mat(p, x, y, rows, cols, kind, h, drop) {
  const c = 8, gap = 2, w = cols * c + (cols - 1) * gap, hh = rows * c + (rows - 1) * gap;
  const g = S('g', { class: 'mat ' + kind, 'data-h': h }, p);
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
    const masked = kind === 'att' && k > r, dropped = !masked && drop && drop(r, k);
    S('rect', { x: x + k * (c + gap), y: y + r * (c + gap), width: c, height: c, rx: 1.6, class: masked ? 'cell off' : dropped ? 'cell drop' : 'cell' }, g);
  }
  return { w, h: hh };
}
/* Toy selection for the pipeline: each row keeps its 3 best allowed keys. */
function toyDrop(r, k) {
  if (r < 3) return false;
  const sc = Array.from({ length: r + 1 }, (_, s) => hash01(r + 40, s) + (s === 0 ? 0.6 : 0) + (s === r ? 0.5 : 0));
  const keep = sc.map((x, s) => s).sort((a, b) => sc[b] - sc[a]).slice(0, 3);
  return !keep.includes(k);
}
function drawPipeline(svg, m) {
  const y0 = PIPE_Y, T = 5, mla = m.key === 'mla' || (m.key === 'dsa' && state.sparse === 'dsa');
  const qsa = m.key === 'dsa' && state.sparse === 'qsa', sparse = m.key === 'dsa';
  const H = qsa ? LAB.QSA.H : m.H, G = qsa ? LAB.QSA.G : m.G, dh = qsa ? LAB.QSA.dh : m.dh;
  const h = Math.min(state.head, H), g = qsa ? Math.ceil(h * G / H) : LAB.kvHeadOf(m, h);
  const qd = mla ? m.dh + m.dr : dh, vd = mla ? m.dv : dh, cat = H * vd;
  const reads = sparse ? (qsa ? `reads KV head ${g} at the selected positions only` : 'reads only the selected latents, like every head')
    : mla ? 'with its own rebuilt key and value'
    : m.key === 'mha' ? `reads key/value head ${g}, its own`
    : m.key === 'mqa' ? 'reads the single shared key/value head'
    : `reads key/value head ${g}, shared by heads ${(g - 1) * (m.H / m.G) + 1}–${g * (m.H / m.G)}`;
  S('line', { x1: 16, y1: y0 - 66, x2: HX1, y2: y0 - 66, class: 'sep' }, svg);
  const cap = S('text', { x: 16, y: y0 - 38, class: 'cap' }, svg);
  S('tspan', { text: `Inside query head ${h}` }, cap);
  S('tspan', { text: `  ·  ${reads}`, class: 'cap-m' }, cap);
  S('text', { x: HX1, y: y0 - 38, 'text-anchor': 'end', class: 'hint', text: sparse ? 'pick a head in the whole model above' : 'click a query head to follow it' }, svg);

  const kSub = m.key === 'mqa' ? null : String(mla ? h : g);
  const vSub = kSub;
  let x = 22;
  const lab = (cx, base, sub, sup, hk) => mlab(svg, cx, y0 - 10, base, sub, sup, { h: hk });
  const dim = (cx, txt, hk, yy) => S('text', { x: cx, y: yy, 'text-anchor': 'middle', class: 'dim', 'data-h': hk, text: txt }, svg);
  const op = (txt, hk) => { S('text', { x: x + 9, y: y0 + 29, 'text-anchor': 'middle', class: 'op', 'data-h': hk, text: txt }, svg); x += 18; };
  const arrow = (len, label, hk) => {
    S('line', { x1: x + 4, y1: y0 + 24, x2: x + len - 4, y2: y0 + 24, class: 'arr', 'marker-end': 'url(#arrow)', 'data-h': hk }, svg);
    const [above, below] = String(label || '').split('|');
    if (above) S('text', { x: x + len / 2, y: y0 + 15, 'text-anchor': 'middle', class: 'arr-t', 'data-h': hk, text: above }, svg);
    if (below) S('text', { x: x + len / 2, y: y0 + 39, 'text-anchor': 'middle', class: 'arr-t', 'data-h': hk, text: below }, svg);
    x += len;
  };
  const block = (rows, cols, kind, hk, base, sub, sup, dimTxt, drop) => {
    const r = mat(svg, x, y0, rows, cols, kind, hk, drop);
    lab(x + r.w / 2, base, sub, sup, hk);
    dim(x + r.w / 2, dimTxt, hk, y0 + 66);
    x += r.w;
    return r;
  };
  block(T, 3, 'q', 'pq', 'Q', String(h), null, `T×${qd}`);
  op('·', 'pk');
  block(3, T, 'k', 'pk', 'K', kSub, '⊤', `${qd}×T`);
  arrow(46, `÷√${qd}`, 'ps');
  block(T, T, 'score', 'ps', 'S', String(h), null, 'T×T');
  arrow(70, 'mask|softmax', 'pa');
  block(T, T, 'att', 'pa', 'A', String(h), null, 'T×T', sparse ? toyDrop : null);
  op('·', 'pv');
  block(T, 3, 'v', 'pv', 'V', vSub, null, `T×${vd}`);
  op('=', 'po');
  block(T, 3, 'out', 'po', 'O', String(h), null, `T×${vd}`);
  arrow(40, `all ${H}`, 'pcat');
  const cw = 50, ch = 48, slices = Math.min(H, 32);
  const cg = S('g', { 'data-h': 'pcat', class: 'mat out' }, svg);
  S('rect', { x, y: y0, width: cw, height: ch, rx: 2, class: 'cell' }, cg);
  for (let i = 1; i < slices; i++) S('line', { x1: x + (cw * i) / slices, y1: y0, x2: x + (cw * i) / slices, y2: y0 + ch, class: 'slice' }, cg);
  S('rect', { x: x + (cw * Math.floor((h - 1) * slices / H)) / slices, y: y0, width: cw / slices, height: ch, class: 'slice-sel' }, cg);
  S('text', { x: x + cw / 2, y: y0 - 8, 'text-anchor': 'middle', class: 'ml-up', 'data-h': 'pcat', text: 'concat' }, svg);
  dim(x + cw / 2, `T×${LAB.fmt(cat)}`, 'pcat', y0 + 66);
  x += cw;
  op('·', 'pwo');
  const wo = mat(svg, x, y0 + 5, 4, 4, 'w', 'pwo');
  lab(x + wo.w / 2, 'W', null, 'O', 'pwo');
  dim(x + wo.w / 2, `${LAB.fmt(cat)}×${LAB.fmt(m.d)}`, 'pwo', y0 + 82);
  x += wo.w;
  op('=', 'py');
  block(T, 4, 'y', 'py', 'Y', null, null, `T×${LAB.fmt(m.d)}`);
  S('text', { x: 16, y: y0 + 102, class: 'note', text: sparse
    ? 'T = 5 drawn, toy budget of 3 keys per query. Light cells: future tokens; outlined cells: earlier tokens the indexer did not select.'
    : 'T = number of tokens (5 drawn). Light cells in A are future tokens hidden by the causal mask. Shading is schematic.' }, svg);
}

/* Gated DeltaNet: the recurrence of one value head, unrolled over three tokens. */
function drawGDNPipeline(svg, m) {
  const y0 = 462, j = Math.min(state.head, m.lin.v), i = Math.ceil(j / 2);
  S('line', { x1: 16, y1: y0 - 66, x2: HX1, y2: y0 - 66, class: 'sep' }, svg);
  const cap = S('text', { x: 16, y: y0 - 38, class: 'cap' }, svg);
  S('tspan', { text: `Inside value head ${j}` }, cap);
  S('tspan', { text: `  ·  uses query/key head ${i} · one state, updated once per token`, class: 'cap-m' }, cap);
  S('text', { x: HX1, y: y0 - 38, 'text-anchor': 'end', class: 'hint', text: 'click a value head to follow it' }, svg);
  let x = 22, lastO = null;
  const lab = (cx, base, sub, sup, hk, y = y0 - 10) => mlab(svg, cx, y, base, sub, sup, { h: hk });
  const state0 = mat(svg, x, y0, 5, 5, 'st0', 'ps');
  lab(x + state0.w / 2, 'S', '0', null, 'ps');
  S('text', { x: x + state0.w / 2, y: y0 + 66, 'text-anchor': 'middle', class: 'dim', 'data-h': 'ps', text: '= 0' }, svg);
  x += state0.w;
  for (let t = 1; t <= 3; t++) {
    const len = 62;
    S('line', { x1: x + 4, y1: y0 + 24, x2: x + len - 4, y2: y0 + 24, class: 'arr', 'marker-end': 'url(#arrow)', 'data-h': 'chain' }, svg);
    const ta = S('text', { x: x + len / 2, y: y0 + 14, 'text-anchor': 'middle', class: 'arr-t ml-s', 'data-h': 'pk pv' }, svg);
    S('tspan', { text: `${mi('k')}`, class: 'kc' }, ta); S('tspan', { text: String(t), class: 'ss', dy: 3 }, ta);
    S('tspan', { text: ', ', dy: -3 }, ta);
    S('tspan', { text: `${mi('v')}`, class: 'vc' }, ta); S('tspan', { text: String(t), class: 'ss', dy: 3 }, ta);
    const tb = S('text', { x: x + len / 2, y: y0 + 41, 'text-anchor': 'middle', class: 'arr-t ml-s', 'data-h': 'pg' }, svg);
    S('tspan', { text: 'α' }, tb); S('tspan', { text: String(t), class: 'ss', dy: 3 }, tb);
    S('tspan', { text: ', β', dy: -3 }, tb); S('tspan', { text: String(t), class: 'ss', dy: 3 }, tb);
    x += len;
    const r = mat(svg, x, y0, 5, 5, 'st', 'ps');
    lab(x + r.w / 2, 'S', String(t), null, 'ps');
    // read-out below the state
    S('line', { x1: x + r.w / 2, y1: y0 + r.h + 4, x2: x + r.w / 2, y2: y0 + r.h + 24, class: 'arr', 'marker-end': 'url(#arrow)', 'data-h': 'pq po' }, svg);
    const tq = S('text', { x: x + r.w / 2 + 5, y: y0 + r.h + 19, class: 'arr-t ml-s', 'data-h': 'pq' }, svg);
    S('tspan', { text: '· ' }, tq); S('tspan', { text: mi('q'), class: 'qc' }, tq); S('tspan', { text: String(t), class: 'ss', dy: 3 }, tq);
    mat(svg, x + r.w / 2 - 24, y0 + r.h + 28, 1, 5, 'out', 'po');
    const to = S('text', { x: x + r.w / 2 + 34, y: y0 + r.h + 37, class: 'arr-t ml-s', 'data-h': 'po' }, svg);
    S('tspan', { text: mi('o') }, to); S('tspan', { text: String(t), class: 'ss', dy: 3 }, to);
    lastO = to;
    x += r.w;
  }
  // the update rule, written out once, to the right of the last read-out and its label
  const oBox = lastO.getBBox();
  const rx = Math.max(x + 22, Math.ceil(oBox.x + oBox.width) + 20);
  S('rect', { x: rx - 8, y: y0 - 6, width: HX1 - rx + 8, height: 104, rx: 8, class: 'rulebox', 'data-h': 'ps pk pv pg' }, svg);
  const I = mi;
  const lines = [
    ['decay', `${I('S')} ← α ${I('S')}`, null, 'pg'],
    ['erase', `${I('S')} ← ${I('S')} − β (${I('S')}${I('k')}) ${I('k')}`, '⊤', 'pk'],
    ['write', `${I('S')} ← ${I('S')} + β ${I('v')}${I('k')}`, '⊤', 'pv'],
    ['read', `${I('o')} = ${I('S')}${I('q')}`, null, 'pq po'],
    ['output', `${I('y')} = (norm(${I('o')}) ⊙ SiLU(${I('z')})) ${I('W')}`, 'O', 'pz py'],
  ];
  lines.forEach(([k, txt, sup, hk], n) => {
    S('text', { x: rx, y: y0 + 12 + n * 19, class: 'rule-k', 'data-h': hk, text: k }, svg);
    const t = S('text', { x: rx + 50, y: y0 + 12 + n * 19, class: 'rule-v', 'data-h': hk }, svg);
    S('tspan', { text: txt }, t);
    if (sup) S('tspan', { text: sup, class: 'ss', dy: -6 }, t);
  });
  S('text', { x: 16, y: y0 + 128, class: 'note', text: `S is ${m.lin.dv} × ${m.lin.dk} (5 × 5 drawn). Every token rewrites the same matrix: nothing is appended.` }, svg);
}

function drawDiagram(m) {
  const svg = $('#diagram');
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${VW} ${VH}`);
  defs(svg);
  legend(svg, m);
  if (m.key === 'mla') { drawMLA(svg, m); drawPipeline(svg, m); }
  else if (m.key === 'gdn') { drawGDN(svg, m); drawGDNPipeline(svg, m); }
  else if (m.key === 'dsa') { drawSparse(svg, m); drawPipeline(svg, m); }
  else { drawStandard(svg, m); drawPipeline(svg, m); }
  applyFocus();
}
function hlKey(m) { return m.key === 'mla' ? 'mla' : m.key === 'gdn' ? 'gdn' : m.key === 'dsa' ? 'sparse' : 'standard'; }
function applyFocus() {
  const svg = $('#diagram'), m = LAB.MODELS[state.mech];
  const step = LAB.steps(m, state.sparse)[state.step];
  const map = LAB.HIGHLIGHT[hlKey(m)];
  const on = new Set(step ? map[step.id] || [] : []);
  svg.classList.toggle('focusing', !!step);
  svg.querySelectorAll('[data-h]').forEach(el => {
    const ids = el.getAttribute('data-h').split(' ');
    el.classList.toggle('on', ids.some(i => on.has(i)));
  });
}

/* ---------- the whole model ---------- */
/* Layout: every token occupies one row (y = rowY(t)) across tokens, embeddings, x̂, Q, K, V,
   the mixer and the feed-forward layer, so a token's own q, k and v sit on its row. */
const AW = 1440, AH = 744, SY = 132, Y0 = 288, RH = 24;
const BX0 = 316, BX1 = 1136;                 // decoder block
const AX0 = 332, AX1 = 832;                  // attention / mixer sublayer
const FX0 = 864, FX1 = 1112;                 // feed-forward sublayer
const SB0 = 190;                             // top of both sublayer boxes
let SB1 = 542;                               // their bottom: set per example, from its longest tokenization
const FN = 1190;                             // final norm
const RX0 = 1162, RX1 = 1432, RC = (RX0 + RX1) / 2;
const LY0 = 185, LY1 = 253, DY0 = 277, DY1 = 477, CY0 = 501, CY1 = 705;   // logits, distribution, decoding
const rowY = t => Y0 + t * RH;
const NEXT_REAL = (() => { try { return JSON.parse(document.getElementById('next-token-data').textContent || '{}'); } catch (e) { return {}; } })();

function moeShown(m) { return m.ffnKind === 'moe' && (m.key === 'gdn' || state.block !== 'dense'); }
function view(m) { return { moe: moeShown(m), layer: m.key === 'gdn' ? state.layer : undefined }; }
function archStepsNow() { const m = LAB.MODELS[state.mech]; return LAB.archSteps(m, view(m)); }
function archIndex(id) { return archStepsNow().findIndex(s => s.id === id); }
function toks(m) { return LAB.TOKENS.models[m.key]; }
function pieceParts(p) {           // [marker, text] so the leading-space marker can be drawn muted
  if (p.startsWith('<')) return ['', 'BOS'];
  return /^[Ġ▁]/.test(p) ? ['␣', p.slice(1)] : ['', p];
}
function pieceText(p) { const [mk, rest] = pieceParts(p); return mk + rest; }
function pieceInto(textEl, p) {
  const [mk, rest] = pieceParts(p);
  if (mk) S('tspan', { class: 'mk', text: mk }, textEl);
  S('tspan', { text: rest }, textEl);
  return textEl;
}

/* An element that selects a step on click and shows a pop-up on hover/focus. */
function act(el, stepId, pop, label) {
  el.setAttribute('tabindex', '0');
  el.setAttribute('role', 'button');
  if (label) el.setAttribute('aria-label', label);
  el.classList.add('hit');
  if (pop) el.dataset.pop = pop;
  const go = () => { if (stepId && state.tour < 0 && archIndex(stepId) >= 0) setArch(archIndex(stepId), false); };
  el.addEventListener('click', go);
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  return el;
}
function rowAttr(el, t) { el.dataset.row = String(t); return el; }
function header(svg, x, y, base, sup, h, anchor = 'middle', sub) {
  return mlab(svg, x, y, base, sub || null, sup || null, { anchor, h, cls: 'hd' });
}
function dimText(svg, x, y, text, h) { return S('text', { x, y, 'text-anchor': 'middle', class: 'dim', 'data-h': h, text }, svg); }
function pillA(svg, cx, cy, w, text, h, stepId, pop) {
  const g = act(S('g', { 'data-h': h }, svg), stepId, pop, text);
  S('rect', { x: cx - w / 2, y: cy - 13, width: w, height: 26, rx: 13, class: 'pillbox' }, g);
  S('text', { x: cx, y: cy + 4.5, 'text-anchor': 'middle', class: 'pill-l', text }, g);
}
function plusA(svg, cx, cy, h, stepId, pop) {
  const g = act(S('g', { 'data-h': h }, svg), stepId, pop, 'Residual addition');
  S('circle', { cx, cy, r: 10, class: 'plus' }, g);
  S('path', { d: `M${cx - 5},${cy} H${cx + 5} M${cx},${cy - 5} V${cy + 5}`, class: 'plus-x' }, g);
}
function cells(svg, x, t, n, pitch, w, cls, h, step, popFn, extra = {}) {
  for (let i = 0; i < n; i++) {
    const r = S('rect', { x: x + i * pitch, y: rowY(t) - 6, width: w, height: 12, rx: Math.min(1.5, w / 3), class: cls + (extra.sel === i ? ' sel' : ''), 'data-h': h }, svg);
    rowAttr(r, t);
    if (popFn) { r.dataset.pop = popFn(i); r.classList.add('cellhit'); }
    if (extra.onClick) r.addEventListener('click', () => extra.onClick(i));
    else if (step) r.addEventListener('click', () => setArch(archIndex(step), false));
  }
  return n * pitch - (pitch - w);
}
function bracketUp(svg, x, t0, t1, toX, h) {
  S('path', { d: `M${x - 6},${rowY(t0)} H${x} V${rowY(t1)} H${x - 6} M${x},${(rowY(t0) + rowY(t1)) / 2} H${toX} V${SY + 10}`, class: 'branch', 'marker-end': 'url(#arrow-ink)', 'data-h': h }, svg);
}
function dials(svg, dx, T, which) {   // RoPE: needle angle = t radians, the fastest rotating pair at position t
  for (let t = 0; t < T; t++) {
    const y = rowY(t);
    const g = rowAttr(S('g', { 'data-h': 'rope', 'data-pop': `rope:${t}:${which}`, class: 'cellhit' }, svg), t);
    g.addEventListener('click', () => setArch(archIndex('rope'), false));
    S('circle', { cx: dx, cy: y, r: 5, class: 'dial' }, g);
    S('line', { x1: dx, y1: y, x2: dx + 4.2 * Math.sin(t), y2: y - 4.2 * Math.cos(t), class: 'needle' }, g);
  }
}
function ropeLabel(svg, dQ, dK, label) {
  act(S('text', { x: (dQ + dK) / 2, y: Y0 - 58, 'text-anchor': 'middle', class: 'rope-l', 'data-h': 'rope', text: label }, svg), 'rope', 'ropelabel', 'Rotary position embedding');
  S('path', { d: `M${dQ},${Y0 - 53} V${Y0 - 46} M${dK},${Y0 - 53} V${Y0 - 46} M${dQ},${Y0 - 53} H${dK}`, class: 'rope-br', 'data-h': 'rope' }, svg);
}
/* Illustrative choices (the real ones depend on the weights). */
function expertsOf(m, t) { return Array.from({ length: m.active }, (_, j) => (t * 37 + j * 53 + 3) % m.experts).sort((a, b) => a - b); }
function idxScore(t, s) { return 0.2 + 0.6 * hash01(t + 3, s + 1) + (s === 0 ? 0.5 : 0) + (s === t ? 0.4 : 0); }
const TOY_K = 4;
function idxSel(t) {
  const s = Array.from({ length: t + 1 }, (_, i) => i).sort((a, b) => idxScore(t, b) - idxScore(t, a));
  return new Set(s.slice(0, TOY_K));
}
function attnShade(t, s) { return 0.25 + 0.75 * hash01(t + 11, s + 5); }

/* v6: every legend symbol shares the main diagram tooltip behavior. */
function drawSymbolKey(svg) {
  const m = LAB.MODELS[state.mech], v = view(m);
  const recurrent = m.key === 'gdn' && v.layer !== 'gattn';
  const latent = m.key === 'mla' || m.key === 'dsa';
  const isSparse = m.key === 'dsa';
  const gatedAttention = m.key === 'gdn' && !recurrent;
  const item = (name, formula, meaning, title) => `<div data-symbol="${name}" data-pop="legend:${name}" tabindex="0" aria-describedby="pop"><dt data-tex="${esc(formula)}">${esc(name)}</dt><dd>${esc(meaning)}</dd></div>`;
  const extra = [
    ...(latent ? [
      item('ckv', String.raw`C^{KV}`, 'compressed KV', 'Compact latent used to reconstruct keys and values'),
      item('kr', String.raw`k^{R}`, 'rotary key', 'Position-dependent key kept separately from the latent'),
    ] : []),
    ...(isSparse ? [
      item('ki', String.raw`k^{I}`, 'index key', 'Key used to choose which past tokens to read'),
      item('ibar', String.raw`I_t`, 'selection scores', 'Scores used by the sparse indexer to select past tokens'),
    ] : []),
    ...(recurrent ? [
      item('conv', String.raw`\operatorname{conv}_4`, 'short conv', 'Causal convolution that mixes the latest four positions'),
      item('gate-a', String.raw`\alpha`, 'forget gate', 'Controls how much of the previous state is retained'),
      item('gate-b', String.raw`\beta`, 'write gate', 'Controls the strength of the update to the state'),
      item('state', String.raw`S_t`, 'fixed memory', 'Fixed-size memory after token t'),
      item('read', String.raw`o`, 'state readout', 'Result of reading the state with the query'),
      item('zg', String.raw`z`, 'output gate', 'Controls how much of the readout to pass through'),
    ] : []),
    ...(gatedAttention ? [item('gt', String.raw`\sigma(G)`, 'output gate', 'Gate on gated-attention head outputs')] : []),
    ...(moeShown(m) ? [item('moe', String.raw`\operatorname{MoE}`, 'routed experts', 'Only a selected subset of expert networks is used per token')] : []),
  ];
  S('text', { x: 472, y: 16, class: 'st', text: 'Legend' }, svg);
  const host = S('foreignObject', { id: 'symbol-key', x: 472, y: 24, width: 662, height: 60,
    role: 'group', 'aria-label': 'Symbols in the model diagram' }, svg);
  const frame = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
  frame.className = 'diagram-legend';
  frame.innerHTML = `<dl class="symbol-key-list">
    <div data-symbol="emb" data-pop="legend:emb" tabindex="0" aria-describedby="pop"><dt data-tex="E">E</dt><dd>embedding table</dd></div>
    <div data-symbol="pos" data-pop="legend:pos" tabindex="0" aria-describedby="pop"><dt data-tex="P">P</dt><dd>position table</dd></div>
    <div data-symbol="xhat" data-pop="legend:xhat" tabindex="0" aria-describedby="pop"><dt data-tex="\\hat{x}">x</dt><dd>normalized token</dd></div>
    <div data-symbol="q" data-pop="legend:q" tabindex="0" aria-describedby="pop"><dt class="q" data-tex="Q">Q</dt><dd>query</dd></div>
    <div data-symbol="k" data-pop="legend:k" tabindex="0" aria-describedby="pop"><dt class="k" data-tex="K">K</dt><dd>key</dd></div>
    <div data-symbol="v" data-pop="legend:v" tabindex="0" aria-describedby="pop"><dt class="v" data-tex="V">V</dt><dd>value</dd></div>
    <div data-symbol="rope" data-pop="legend:rope" tabindex="0" aria-describedby="pop"><dt><svg class="symbol-dial" width="15" height="15" viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.2"/><line x1="8" y1="8" x2="11.5" y2="5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>RoPE</dt><dd>rotations</dd></div>
    ${extra.join('')}
    <div data-symbol="att" data-pop="legend:att" tabindex="0" aria-describedby="pop"><dt data-tex="A_{1}">A₁</dt><dd>head weights</dd></div>
    <div data-symbol="y" data-pop="legend:y" tabindex="0" aria-describedby="pop"><dt data-tex="Y">Y</dt><dd>attention output</dd></div>
    <div data-symbol="logit" data-pop="legend:logit" tabindex="0" aria-describedby="pop"><dt>logits</dt><dd>next-token scores</dd></div>
  </dl>`;
  host.appendChild(frame);
}

function drawArch(m) {
  const svg = $('#arch');
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${AW} ${AH}`);
  defs(svg);
  const v = view(m), tk = toks(m), T = tk.ids.length, last = T - 1;
  if (state.row == null || state.row > last) state.row = last;
  SB1 = rowY(Math.max(...LAB.ORDER.map(k => LAB.TOKENS.models[k].ids.length)) - 1) + 44;

  // block boundary and row bands (behind everything)
  const blk = S('g', { 'data-h': 'block' }, svg);
  S('rect', { x: BX0 + 7, y: 92 + 7, width: BX1 - BX0, height: SB1 + 40 - 92, rx: 14, class: 'ghost' }, blk);
  S('rect', { x: BX0, y: 92, width: BX1 - BX0, height: SB1 + 40 - 92, rx: 14, class: 'blockbox' }, blk);
  for (let t = 0; t < T; t++) rowAttr(S('rect', { x: 20, y: rowY(t) - 11, width: BX1 - 30, height: 22, rx: 5, class: 'band' }, svg), t);
  const bl = act(S('g', { 'data-h': 'block' }, svg), 'repeat', 'block', `Decoder block × ${m.L}`);
  const blockName = m.key === 'gdn' ? (v.layer === 'gattn' ? 'Decoder layer · gated attention, every 4th layer' : 'Decoder layer · Gated DeltaNet, layers 1–3 of every 4')
    : m.key === 'mla' ? (v.moe ? `Decoder block · blocks 2–${m.L}` : 'Decoder block · block 1')
    : m.key === 'dsa' ? (v.moe ? `Decoder block · blocks 4–${m.L}` : 'Decoder block · blocks 1–3')
    : 'Decoder block';
  S('text', { x: BX0 + 16, y: SB1 + 26, class: 'st', text: blockName }, bl);
  const times = m.key === 'gdn' ? (v.layer === 'gattn' ? m.L / m.interval : m.L - m.L / m.interval) : m.L;
  S('text', { x: BX1 - 16, y: SB1 + 28, 'text-anchor': 'end', class: 'times', text: `× ${times}` }, bl);

  drawInput(svg, m, tk);

  // ---- residual stream
  act(S('line', { x1: BX0 - 12, y1: SY, x2: FN + 32, y2: SY, class: 'stream', 'data-h': 'stream' }, svg), 'stream', 'stream', 'Residual stream');
  const sl = act(S('g', { 'data-h': 'stream' }, svg), 'stream', 'stream', 'Residual stream');
  const st = S('text', { x: BX0 + 16, y: 112, class: 'st' }, sl);
  S('tspan', { text: 'Residual stream h' }, st);
  S('tspan', { text: `   T × ${LAB.fmt(m.d)}: one vector per token, carried through every block; each sublayer adds to it`, class: 'st-m small' }, st);

  // ---- mixer sublayer (attention, Gated DeltaNet or sparse attention); its content is centred in the box
  const gdnLayer = m.key === 'gdn' && v.layer !== 'gattn';
  const ab = act(S('g', { 'data-h': 'attnbox' }, svg), 'attn', null, gdnLayer ? 'Gated DeltaNet sublayer' : 'Attention sublayer');
  S('rect', { x: AX0, y: SB0, width: AX1 - AX0, height: SB1 - SB0, rx: 10, class: 'subbox' }, ab);
  const at = S('text', { x: (AX0 + AX1) / 2, y: 209, 'text-anchor': 'middle', class: 'st', 'data-h': 'attnbox' }, svg);
  S('tspan', { text: gdnLayer ? 'Gated DeltaNet' : m.key === 'dsa' ? 'Sparse attention' : m.key === 'gdn' ? 'Gated attention' : 'Attention' }, at);
  S('tspan', { text: gdnLayer ? `  ${m.lin.qk} qk · ${m.lin.v} v heads · fixed-size memory` : m.key === 'dsa' ? `  MLA · ${m.H} heads · indexer keeps top ${LAB.fmt(m.index.topk)}` : `  ${m.key === 'gdn' ? 'GQA' : m.abbr} · ${m.H} heads${m.key === 'gdn' ? ' · output gate' : ''}`, class: 'st-m' }, at);
  const mg = S('g', {}, svg);
  S('text', { x: AX0 + 14 + 19, y: Y0 - 26, 'text-anchor': 'middle', class: 'ml hd', 'data-h': 'xhat', text: mi('x') + '̂' }, mg);
  for (let t = 0; t < T; t++) cells(mg, AX0 + 14, t, 5, 8, 6, 'xcell', 'xhat', 'norm1', () => `xhat:${t}`);
  const right = gdnLayer ? mixGDN(mg, m, T) : (m.key === 'mla' || m.key === 'dsa') ? mixMLA(mg, m, T) : mixStd(mg, m, T, m.key === 'gdn');
  const dx = Math.max(0, Math.round((AX1 - right - 14) / 2));
  mg.setAttribute('transform', `translate(${dx},0)`);
  S('g', { id: 'arcs' }, mg);
  bracketUp(svg, right + dx, 0, last, AX1 + 13, 'attnout');
  const nx = AX0 + 36 + dx;
  S('circle', { cx: nx, cy: SY, r: 3.5, class: 'fork', 'data-h': 'norm1' }, svg);
  S('line', { x1: nx, y1: SY, x2: nx, y2: 153, class: 'branch', 'data-h': 'norm1' }, svg);
  pillA(svg, nx, 166, 84, m.norm, 'norm1', 'norm1', 'norm1');
  S('line', { x1: nx, y1: 179, x2: nx, y2: SB0, class: 'branch', 'data-h': 'norm1' }, svg);
  plusA(svg, AX1 + 13, SY, 'res1', 'res1', 'res1');

  drawFFN(svg, m, v.moe, T);
  drawOutput(svg, m, T);
  drawDist(m);
  hoverRow = null;                 // a fresh diagram: the next hover draws its arrows again
  drawArcs(m, state.row);
  applyArchFocus();
  drawSymbolKey(svg);
  renderSymbolKey();
  wireRows(svg);
}

/* ---- left: text → tokens → IDs → rows of the embedding matrix */
function drawInput(svg, m, tk) {
  const L = LAB.TOKENS, T = tk.ids.length, last = T - 1;
  S('text', { x: 16, y: 16, class: 'st', 'data-h': 'text', text: 'Input text' }, svg);
  // wide enough for the longest example on one line (words are set 5 apart)
  const wordW = w => { const el = S('text', { class: 'word', text: w }, svg); const n = el.getComputedTextLength(); el.remove(); return n; };
  const lineW = sent => sent.split(' ').reduce((a, w) => a + wordW(w) + 5, -5);
  const boxR = Math.min(BX0 + 140, Math.max(300, 16 + 10 + Math.max(...LAB.EXAMPLES.map(e => lineW(e.sentence))) + 12));
  const sbox = S('rect', { x: 16, y: 24, width: boxR - 16, height: 60, rx: 8, class: 'sentbox', 'data-h': 'text' }, svg);
  // example inputs: one pill each, the active one filled
  let px = 24;
  LAB.EXAMPLES.forEach(ex => {
    const on = ex.key === LAB.TOKENS.key;
    const b = S('g', { class: 'expill' + (on ? ' on-x' : ''), 'data-h': 'text', tabindex: 0, role: 'button', 'aria-pressed': String(on), 'aria-label': `Example input: ${ex.label}`, 'data-pop': `ex:${ex.key}` }, svg);
    const t = S('text', { x: px + 9, y: 44, class: 'expill-t', text: ex.label }, b);
    const w = t.getComputedTextLength() + 18;
    b.insertBefore(S('rect', { x: px, y: 31, width: w, height: 18, rx: 9, class: 'expill-r' }), t);
    px += w + 5;
    const go = () => setExample(ex.key);
    b.addEventListener('click', go);
    b.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
  const sg = act(S('g', { 'data-h': 'text' }, svg), 'tok', 'text', 'Input sentence');
  let wx = 26, wy = 70;
  L.sentence.split(' ').forEach((w, i) => {
    const t = S('text', { x: wx, y: wy, class: 'word', 'data-word': i, 'data-pop': `word:${i}` }, sg);
    t.textContent = w;
    const wlen = t.getComputedTextLength();
    if (wx + wlen > boxR - 10 && wx > 26) { wx = 26; wy += 20; t.setAttribute('x', wx); t.setAttribute('y', wy); }
    wx += wlen + 5;
  });
  sbox.setAttribute('height', wy + 14 - 24);
  const tz = act(S('g', { 'data-h': 'tok' }, svg), 'tok', 'tokenizer', 'Tokenizer');
  S('line', { x1: 40, y1: wy + 16, x2: 40, y2: rowY(0) - 22, class: 'branch', 'marker-end': 'url(#arrow-ink)' }, tz);
  S('text', { x: 52, y: 132, class: 'st', text: 'Tokenizer' }, tz);
  S('text', { x: 52, y: 148, class: 'st-s', text: m.tokenizer }, tz);
  S('text', { x: 52, y: 163, class: 'st-s', text: `vocab ${LAB.fmt(m.V)}` }, tz);
  S('text', { x: 52, y: 178, class: 'st-s', text: `${T} tokens` }, tz);

  const words = tk.words;
  for (let w = 0; w <= Math.max(...words); w++) {
    const rows = words.map((x, i) => x === w ? i : -1).filter(i => i >= 0);
    if (rows.length > 1) S('path', { d: `M30,${rowY(rows[0]) - 8} H25 V${rowY(rows[rows.length - 1]) + 8} H30`, class: 'wordbr', 'data-h': 'tok', 'data-word': w }, svg);
  }
  S('text', { x: 152, y: Y0 - 26, 'text-anchor': 'end', class: 'st-s', 'data-h': 'tok', text: 'ID' }, svg);
  tk.pieces.forEach((p, t) => {
    const g = rowAttr(act(S('g', { 'data-h': 'tok' }, svg), 'tok', `tok:${t}`, `Token ${t + 1}: ${pieceText(p)}`), t);
    if (words[t] >= 0) g.dataset.word = words[t];
    S('rect', { x: 34, y: rowY(t) - 9, width: 72, height: 18, rx: 5, class: 'chip' + (p.startsWith('<') ? ' special' : '') }, g);
    pieceInto(S('text', { x: 70, y: rowY(t) + 4.5, 'text-anchor': 'middle', class: 'chip-t' }, g), p);
    S('text', { x: 152, y: rowY(t) + 4, 'text-anchor': 'end', class: 'tid', text: tk.ids[t] }, g);
  });
  // appended tokens (filled by the decoding demo; beam search appends two)
  for (let n = 0; n < 2; n++) {
    const gh = S('g', { 'data-h': 'loop', class: 'ghostg', id: `ghost${n}` }, svg);
    S('rect', { x: 34, y: rowY(T + n) - 9, width: 72, height: 18, rx: 5, class: 'chip ghostchip' }, gh);
    S('text', { x: 70, y: rowY(T + n) + 4.5, 'text-anchor': 'middle', class: 'chip-t ghost-t', text: '' }, gh);
  }

  // token ID → dense vector: each ID picks one row of the matrix E drawn above its column
  const EX = 162, EW = 46, ET = 128, EB = 234, PX = 228;
  const tile = (x, name, label, h, pop, stripeOf) => {
    const g = act(S('g', { 'data-h': h }, svg), 'emb', pop, label);
    mlab(g, x + EW / 2, ET - 9, name, null, null, { cls: 'hd' });
    S('rect', { x, y: ET, width: EW, height: EB - ET, rx: 3, class: 'etable' }, g);
    for (let i = 1; i < 6; i++) S('line', { x1: x + i * 8 - 1, y1: ET + 2, x2: x + i * 8 - 1, y2: EB - 2, class: 'ecol' }, g);
    for (let t = 0; t < T; t++) rowAttr(S('rect', { x: x + 1, y: stripeOf(t) - 1, width: EW - 2, height: 2, rx: 0.5, class: 'estripe', 'data-h': h }, svg), t);
  };
  const stripeE = t => ET + 3 + (EB - ET - 6) * tk.ids[t] / m.V;
  tile(EX, 'E', `Embedding matrix E, ${LAB.fmt(m.V)} rows`, 'emb', 'etable', stripeE);
  for (let t = 0; t < T; t++) cells(svg, EX, t, 6, 8, 6, 'ecell', 'emb', 'emb', () => `emb:${t}`);
  let xr = EX + EW, stripeP = null;
  if (m.pos === 'learned') {
    S('text', { x: 218, y: rowY(Math.floor(last / 2)) + 5, 'text-anchor': 'middle', class: 'op', 'data-h': 'emb pos', text: '+' }, svg);
    stripeP = t => ET + 3 + (EB - ET - 6) * t / m.ctx;
    tile(PX, 'P', `Position matrix P, ${LAB.fmt(m.ctx)} rows`, 'emb pos', 'ptable', stripeP);
    for (let t = 0; t < T; t++) cells(svg, PX, t, 6, 8, 6, 'pcell', 'emb pos', 'emb', () => `pos:${t}`);
    xr = PX + EW;
  } else if (m.embScale) {
    act(S('text', { x: 214, y: rowY(Math.floor(last / 2)) + 4, class: 'scale', 'data-h': 'emb', text: '× √d' }, svg), 'emb', 'scale');
    xr = 246;
  }
  state._lookup = { EX, PX, stripeE, stripeP };
  S('g', { id: 'lookup' }, svg);
  for (let t = 0; t < T; t++) S('line', { x1: xr + 3, y1: rowY(t), x2: 290, y2: rowY(t), class: 'gather', 'data-h': 'emb' }, svg);
  const mid = (rowY(0) + rowY(last)) / 2;
  S('path', { d: `M290,${rowY(0)} V${rowY(last)} M290,${mid} C310,${mid} 288,${SY} ${BX0 - 12},${SY}`, class: 'gather', 'data-h': 'emb stream' }, svg);
}

/* ---- standard attention (MHA / MQA / GQA, and Qwen3-Next's gated attention) */
function mixStd(svg, m, T, gated) {
  const last = T - 1, H = m.H, G = m.G, rope = m.pos === 'rope';
  const sel = (state.head - 1) % H, selG = LAB.kvHeadOf(m, sel + 1);
  const qx = AX0 + 80, qw = 5 * H - 1, kw = 5 * G - 1;
  for (let t = 0; t < T; t++) cells(svg, qx, t, H, 5, 4, 'qcell', 'q', 'qkv', i => `q:${t}:${i + 1}`, { sel, onClick: i => selectHead(i + 1) });
  header(svg, qx + qw / 2, Y0 - 26, 'Q', null, 'q');
  // K and V move right until the label under them ("1×256 each") clears the one under Q
  const textW = str => { const el = S('text', { class: 'dim', text: str }, svg); const n = el.getComputedTextLength(); el.remove(); return n; };
  const qLab = `${H}×${m.dh}`, kvLab = `${G}×${m.dh} each`, kv0 = qx + qw + (rope ? 22 : 16), kvMid = rope ? kw + 11 : kw + 6;
  const gap = (kv0 + kvMid - textW(kvLab) / 2) - (qx + qw / 2 + textW(qLab) / 2);
  const kx = kv0 + Math.max(0, Math.ceil(16 - gap));
  const dQ = rope ? qx + qw + 10 : null, dK = rope ? kx + kw + 10 : null;
  const vx = kx + kw + (rope ? 22 : 12);
  act(S('rect', { x: kx - 5, y: rowY(0) - 11, width: vx + kw + 5 - (kx - 5), height: rowY(last) - rowY(0) + 22, rx: 5, class: 'cache', 'data-h': 'k v' }, svg), 'attn', 'cache', 'KV cache');
  for (let t = 0; t < T; t++) {
    cells(svg, kx, t, G, 5, 4, 'kcell', 'k', 'qkv', i => `k:${t}:${i + 1}`, { sel: selG - 1 });
    cells(svg, vx, t, G, 5, 4, 'vcell', 'v', 'qkv', i => `v:${t}:${i + 1}`, { sel: selG - 1 });
  }
  header(svg, kx + kw / 2, Y0 - 26, 'K', null, 'k');
  header(svg, vx + kw / 2, Y0 - 26, 'V', null, 'v');
  dimText(svg, qx + qw / 2, rowY(last) + 28, qLab, 'q');
  dimText(svg, (kx + vx + kw) / 2, rowY(last) + 28, kvLab, 'k v');
  state._qkv = { qx, kx, vx };
  if (rope) { dials(svg, dQ, T, 'q'); dials(svg, dK, T, 'k'); ropeLabel(svg, dQ, dK, gated ? 'partial RoPE' : 'RoPE'); }
  const ax = vx + kw + 26;
  drawA(svg, m, T, ax, null);
  let yx = ax + T * 8 + 22;
  if (gated) {
    const gx = ax + T * 8 + 14;
    for (let t = 0; t < T; t++) cells(svg, gx, t, 4, 5, 4, 'gtcell', 'gt', 'attn', () => `gt:${t}`);
    header(svg, gx + 9, Y0 - 26, 'σ(G)', null, 'gt');
    S('text', { x: gx + 25, y: rowY(Math.floor(last / 2)) + 4, 'text-anchor': 'middle', class: 'op small', 'data-h': 'gt', text: '⊙' }, svg);
    yx = gx + 34;
  }
  return yCells(svg, m, T, yx);
}
function drawA(svg, m, T, ax, selOf) {
  const last = T - 1;
  for (let t = 0; t < T; t++) {
    const keep = selOf ? selOf(t) : null;
    for (let c = 0; c < T; c++) {
      const cls = c > t ? 'acell off' : keep && !keep.has(c) ? 'acell drop' : 'acell';
      const r = S('rect', { x: ax + c * 8, y: rowY(t) - 6, width: 6.5, height: 12, rx: 1.4, class: cls, 'data-h': keep && keep.has(c) ? 'att sel' : 'att', 'data-pop': `att:${t}:${c}` }, svg);
      if (cls === 'acell') r.style.fillOpacity = String(attnShade(t, c).toFixed(2));
      rowAttr(r, t); r.classList.add('cellhit');
      r.addEventListener('click', () => setArch(archIndex('attn'), false));
    }
  }
  header(svg, ax + T * 4, Y0 - 26, 'A', null, 'att', 'middle', String(state.head));
  dimText(svg, ax + T * 4, rowY(last) + 28, 'T×T', 'att');
}
function yCells(svg, m, T, yx) {
  const last = T - 1;
  for (let t = 0; t < T; t++) cells(svg, yx, t, 5, 8, 6, 'ycell', 'y', 'attn', () => `y:${t}`);
  header(svg, yx + 19, Y0 - 26, 'Y', null, 'y');
  dimText(svg, yx + 19, rowY(last) + 28, `T×${LAB.fmt(m.d)}`, 'y');
  return yx + 46;                  // where the output bracket goes
}

/* ---- MLA, with DeepSeek-V3.2's lightning indexer when m.key === 'dsa' */
function mixMLA(svg, m, T) {
  const last = T - 1, dsa = m.key === 'dsa';
  const Hd = m.drawH || m.H, per = m.H / Hd, sel = Math.floor((state.head - 1) / per);
  const qx = AX0 + 80, qw = 5 * Hd - 1;
  for (let t = 0; t < T; t++) cells(svg, qx, t, Hd, 5, 4, 'qcell', 'q', 'qkv', i => `q:${t}:${i * per + 1}`, { sel, onClick: i => selectHead(i * per + 1) });
  header(svg, qx + qw / 2, Y0 - 26, 'Q', null, 'q');
  const dQ = qx + qw + 10, kx = qx + qw + 24, cw = 30, rx = kx + cw + 6, dK = rx + 16;
  const kix = dK + 12, kEnd = dsa ? kix + 6 : dK + 6;
  act(S('rect', { x: kx - 5, y: rowY(0) - 11, width: kEnd + 5 - (kx - 5), height: rowY(last) - rowY(0) + 22, rx: 5, class: 'cache', 'data-h': 'k v' }, svg), 'attn', 'cache', 'KV cache');
  for (let t = 0; t < T; t++) {
    cells(svg, kx, t, 4, 8, 6, 'lcell', 'k', 'qkv', () => `c:${t}`);
    cells(svg, rx, t, 1, 8, 6, 'krcell', 'v kr', 'qkv', () => `kr:${t}`);
    if (dsa) cells(svg, kix, t, 1, 8, 6, 'kicell', 'ki', 'index', () => `ki:${t}`);
  }
  header(svg, kx + cw / 2, Y0 - 26, 'C', 'KV', 'k');
  header(svg, rx + 3, Y0 - 26, 'k', 'R', 'v kr');
  if (dsa) header(svg, kix + 3, Y0 - 26, 'k', 'I', 'ki');
  dimText(svg, qx + qw / 2, rowY(last) + 28, `${m.H}×(${m.dh}+${m.dr})`, 'q');
  dimText(svg, (kx + kEnd) / 2, rowY(last) + 28, dsa ? `${m.dc}+${m.dr}+${m.index.dim}` : `${m.dc}+${m.dr} cached`, 'k v');
  state._qkv = { qx, kx, vx: rx };
  dials(svg, dQ, T, 'q'); dials(svg, dK, T, 'k');
  ropeLabel(svg, dQ, dK, 'decoupled RoPE');
  let ax = kEnd + 26;
  if (dsa) {
    const ibx = kEnd + 14;
    state._ibars = { x: ibx, w: 30 };
    S('g', { id: 'ibars', 'data-h': 'ibar' }, svg);
    header(svg, ibx + 15, Y0 - 26, 'I', null, 'ibar', 'middle', 't');
    dimText(svg, ibx + 15, rowY(last) + 28, `top ${TOY_K}`, 'ibar');
    ax = ibx + 30 + 14;
  }
  drawA(svg, m, T, ax, dsa ? idxSel : null);
  return yCells(svg, m, T, ax + T * 8 + 22);
}

/* ---- Gated DeltaNet: per-token q, k, v and gates, then a chain of states S_1 … S_T */
function mixGDN(svg, m, T) {
  const last = T - 1, { qk, v } = m.lin;
  const qx = AX0 + 68, kx = qx + 71, vx = kx + 71, gx = vx + 105, sx = gx + 29, ox = sx + 28, zx = ox + 37, yx = zx + 33;
  const selV = Math.min(state.head, v) - 1, selQ = Math.floor(selV / 2);
  // what generation keeps: the last state and the last three inputs of the convolution
  act(S('rect', { x: qx - 4, y: rowY(Math.max(0, last - 2)) - 11, width: vx + 3 * v - qx + 6, height: rowY(last) - rowY(Math.max(0, last - 2)) + 22, rx: 5, class: 'cache', 'data-h': 'state' }, svg), 'state', 'gcache', 'Convolution cache');
  act(S('rect', { x: sx - 5, y: rowY(last) - 12, width: 24, height: 24, rx: 5, class: 'cache', 'data-h': 'state' }, svg), 'state', 'gcache', 'State cache');
  for (let t = 0; t < T; t++) {
    cells(svg, qx, t, qk, 4, 3, 'qcell', 'q', 'qkv', i => `gq:${t}:${i + 1}`, { sel: selQ, onClick: i => selectHead(2 * i + 1) });
    cells(svg, kx, t, qk, 4, 3, 'kcell', 'k', 'qkv', i => `gk:${t}:${i + 1}`, { sel: selQ });
    cells(svg, vx, t, v, 3, 2.2, 'vcell', 'v', 'qkv', i => `gv:${t}:${i + 1}`, { sel: selV, onClick: i => selectHead(i + 1) });
    rowAttr(S('rect', { x: gx, y: rowY(t) - 6, width: 6, height: 12, rx: 1.5, class: 'gcell a', 'data-h': 'gate', 'data-pop': `ga:${t}` }, svg), t).classList.add('cellhit');
    rowAttr(S('rect', { x: gx + 8, y: rowY(t) - 6, width: 6, height: 12, rx: 1.5, class: 'gcell b', 'data-h': 'gate', 'data-pop': `gb:${t}` }, svg), t).classList.add('cellhit');
    const sq = rowAttr(S('rect', { x: sx, y: rowY(t) - 7, width: 14, height: 14, rx: 2.5, class: 'scell', 'data-h': 'state', 'data-pop': `S:${t}` }, svg), t);
    sq.classList.add('cellhit'); sq.addEventListener('click', () => setArch(archIndex('state'), false));
    if (t > 0) S('line', { x1: sx + 7, y1: rowY(t - 1) + 8, x2: sx + 7, y2: rowY(t) - 9, class: 'chainl', 'marker-end': 'url(#arrow-lat)', 'data-h': 'chain' }, svg);
    cells(svg, ox, t, 4, 7, 5, 'ocell', 'o', 'attn', () => `o:${t}`);
    cells(svg, zx, t, 3, 7, 5, 'zcell', 'zg', 'attn', () => `zg:${t}`);
  }
  header(svg, qx + 31, Y0 - 26, 'Q', null, 'q');
  header(svg, kx + 31, Y0 - 26, 'K', null, 'k');
  header(svg, vx + 47, Y0 - 26, 'V', null, 'v');
  S('text', { x: gx + 7, y: Y0 - 26, 'text-anchor': 'middle', class: 'ml hd', 'data-h': 'gate', text: 'α β' }, svg);
  header(svg, sx + 7, Y0 - 26, 'S', null, 'state', 'middle', 't');
  header(svg, ox + 12, Y0 - 26, 'o', null, 'o');
  header(svg, zx + 9, Y0 - 26, 'z', null, 'zg');
  act(S('text', { x: (qx + vx + 3 * v) / 2, y: Y0 - 58, 'text-anchor': 'middle', class: 'rope-l', 'data-h': 'conv', text: 'causal conv, kernel 4' }, svg), 'conv', 'convlabel', 'Short convolution');
  S('path', { d: `M${qx},${Y0 - 53} V${Y0 - 46} M${vx + 3 * v},${Y0 - 53} V${Y0 - 46} M${qx},${Y0 - 53} H${vx + 3 * v}`, class: 'rope-br', 'data-h': 'conv' }, svg);
  dimText(svg, qx + 31, rowY(last) + 28, `${qk}×${m.lin.dk}`, 'q');
  dimText(svg, kx + 31, rowY(last) + 28, `${qk}×${m.lin.dk}`, 'k');
  dimText(svg, vx + 47, rowY(last) + 28, `${v}×${m.lin.dv}`, 'v');
  dimText(svg, sx + 7, rowY(last) + 28, `${v}×${m.lin.dk}²`, 'state');
  state._qkv = { qx, kx, vx, gx, sx, ox };
  return yCells(svg, m, T, yx);
}

/* ---- feed-forward sublayer */
function drawFFN(svg, m, moe, T) {
  const last = T - 1, nx = FX0 + 46;
  S('circle', { cx: nx, cy: SY, r: 3.5, class: 'fork', 'data-h': 'norm2' }, svg);
  S('line', { x1: nx, y1: SY, x2: nx, y2: 153, class: 'branch', 'data-h': 'norm2' }, svg);
  pillA(svg, nx, 166, 84, m.norm, 'norm2', 'norm2', 'norm2');
  S('line', { x1: nx, y1: 179, x2: nx, y2: 190, class: 'branch', 'data-h': 'norm2' }, svg);
  const fb = act(S('g', { 'data-h': 'ffnbox' }, svg), 'ffn', null, moe ? 'Mixture-of-experts layer' : 'Feed-forward layer');
  S('rect', { x: FX0, y: SB0, width: FX1 - FX0, height: SB1 - SB0, rx: 10, class: 'subbox' }, fb);
  const ft = S('text', { x: (FX0 + FX1) / 2, y: 209, 'text-anchor': 'middle', class: 'st', 'data-h': 'ffnbox' }, svg);
  S('tspan', { text: moe ? 'Experts' : 'Feed-forward' }, ft);
  S('tspan', { text: `  ${moe ? 'MoE' : m.ffnKind === 'mlp' ? 'MLP' : m.act === 'GELU' ? 'GeGLU' : 'SwiGLU'}`, class: 'st-m' }, ft);
  // content: input cells, hidden layer (or expert ticks), output cells, bracket; centred in the box
  const bins = moe ? Math.min(64, m.experts) : 0, per = moe ? m.experts / bins : 1;
  const hw = moe ? 0 : Math.min(140, 16 * m.dff / m.d);
  const span = moe ? 24 + 2 * bins + 8 + 14 + 24 : 24 + hw + 10 + 24;   // fx … bracket
  const fx = Math.round((FX0 + FX1) / 2 - span / 2);
  let outX;
  for (let t = 0; t < T; t++) rowAttr(S('rect', { x: fx, y: rowY(t) - 5, width: 16, height: 10, rx: 2, class: 'fin', 'data-h': 'fin', 'data-pop': `fin:${t}` }, svg), t).classList.add('cellhit');
  if (moe) {
    const ex = fx + 24, sx = ex + 2 * bins + 8;
    S('text', { x: ex, y: Y0 - 26, class: 'st-s', 'data-h': 'ffn', text: `router: ${m.active} of ${m.experts}` }, svg);
    S('text', { x: sx + 6, y: Y0 - 26, 'text-anchor': 'middle', class: 'st-s', 'data-h': 'ffn', text: 'shared' }, svg);
    for (let t = 0; t < T; t++) {
      const chosen = new Set(expertsOf(m, t).map(e => Math.floor(e / per)));
      for (let e = 0; e < bins; e++) rowAttr(S('rect', { x: ex + e * 2, y: rowY(t) - 5, width: 1.4, height: 10, class: chosen.has(e) ? 'tick pick' : 'tick', 'data-h': 'ffn', 'data-pop': `moe:${t}` }, svg), t).classList.add('cellhit');
      for (let e = 0; e < m.shared; e++) rowAttr(S('rect', { x: sx + e * 5, y: rowY(t) - 5, width: 3, height: 10, class: 'tick shared', 'data-h': 'ffn', 'data-pop': `moe:${t}` }, svg), t);
    }
    outX = sx + 14;
    dimText(svg, (fx + outX + 16) / 2, rowY(last) + 28, `each ${LAB.fmt(m.d)}→${LAB.fmt(m.dExpert)}→${LAB.fmt(m.d)}${per > 1 ? ` · tick = ${per}` : ''}`, 'ffn');
  } else {
    const hx = fx + 24;
    S('text', { x: hx + hw / 2, y: Y0 - 26, 'text-anchor': 'middle', class: 'st-s', 'data-h': 'ffn', text: m.ffnKind === 'mlp' ? 'GELU' : `${m.act}(gate) ⊙ up` }, svg);
    for (let t = 0; t < T; t++) {
      if (m.ffnKind === 'mlp') rowAttr(S('rect', { x: hx, y: rowY(t) - 5, width: hw, height: 10, rx: 2, class: 'fhid', 'data-h': 'ffn', 'data-pop': `ffn:${t}` }, svg), t).classList.add('cellhit');
      else {
        rowAttr(S('rect', { x: hx, y: rowY(t) - 6, width: hw, height: 5, rx: 1.5, class: 'fhid', 'data-h': 'ffn', 'data-pop': `ffn:${t}` }, svg), t).classList.add('cellhit');
        rowAttr(S('rect', { x: hx, y: rowY(t) + 1, width: hw, height: 5, rx: 1.5, class: 'fhid alt', 'data-h': 'ffn', 'data-pop': `ffn:${t}` }, svg), t).classList.add('cellhit');
      }
    }
    outX = hx + hw + 10;
    dimText(svg, hx + hw / 2, rowY(last) + 28, `${LAB.fmt(m.d)}→${m.ffnKind === 'mlp' ? '' : '2×'}${LAB.fmt(m.dff)}→${LAB.fmt(m.d)}`, 'ffn');
  }
  for (let t = 0; t < T; t++) rowAttr(S('rect', { x: outX, y: rowY(t) - 5, width: 16, height: 10, rx: 2, class: 'fout', 'data-h': 'ffn', 'data-pop': `fout:${t}` }, svg), t).classList.add('cellhit');
  bracketUp(svg, outX + 24, 0, last, FX1 + 10, 'ffnout');
  plusA(svg, FX1 + 10, SY, 'res2', 'res2', 'res2');
}

/* ---- output: final norm, LM head, logits, distribution box, decoding box, loop back */
function drawOutput(svg, m, T) {
  S('text', { x: FN, y: SY - 26, 'text-anchor': 'middle', class: 'st', 'data-h': 'fnorm', text: 'Final norm' }, svg);
  const finalNormWidth = m.norm === 'LayerNorm' ? 96 : 86;
  pillA(svg, FN, SY, finalNormWidth, m.norm, 'fnorm', 'fnorm', 'fnorm');
  S('line', { x1: FN + finalNormWidth / 2, y1: SY, x2: RC - 24, y2: SY, class: 'thin' }, svg);
  const hd = act(S('g', { 'data-h': 'head' }, svg), 'head', 'head', 'Language-model head');
  S('rect', { x: RC - 22, y: SY - 28, width: 44, height: 56, rx: 7, class: 'wbox' }, hd);
  mlab(hd, RC, SY + 6, 'W', 'U', null);
  S('text', { x: RC, y: SY - 36, 'text-anchor': 'middle', class: 'st', text: 'LM head' }, hd);
  S('text', { x: RC + 30, y: SY + 5, class: 'st-s', text: m.tied ? 'tied to E' : 'separate' }, hd);
  S('line', { x1: RC, y1: SY + 29, x2: RC, y2: LY0 - 2, class: 'branch', 'marker-end': 'url(#arrow-ink)', 'data-h': 'logit' }, svg);

  // logits: one row of V scores per position; only the last row, z_T, goes on
  const lg = act(S('g', { 'data-h': 'logit' }, svg), 'head', 'logits', 'Logits');
  S('rect', { x: RX0, y: LY0, width: RX1 - RX0, height: LY1 - LY0, rx: 10, class: 'subbox' }, lg);
  const lt = S('text', { x: RX0 + 12, y: LY0 + 17, class: 'st' }, lg);
  S('tspan', { text: 'Logits ' }, lt);
  S('tspan', { text: mi('Z'), class: 'mlt' }, lt);
  S('text', { x: RX1 - 12, y: LY0 + 17, 'text-anchor': 'end', class: 'st-s', text: `T × ${LAB.fmt(m.V)}` }, lg);
  // earlier rows as thin lines, the last row as a vector of cells
  const MX = RX0 + 12, N = 54, pitch = 4, MW = N * pitch - 1, rp = Math.min(3, 26 / Math.max(1, T - 1));
  for (let t = 0; t < T; t++) {
    const lastRow = t === T - 1, y = lastRow ? LY1 - 17 : LY0 + 25 + t * rp;
    const row = rowAttr(S('g', { class: 'zrow' + (lastRow ? ' last' : ''), 'data-pop': `zrow:${t}` }, svg), t);
    row.classList.add('cellhit');
    row.addEventListener('click', () => setArch(archIndex('head'), false));
    S('rect', { x: MX - 2, y: y - (lastRow ? 2 : 0.7), width: MW + 4, height: lastRow ? 12 : rp, class: 'hitpad' }, row);
    if (!lastRow) S('rect', { x: MX, y, width: MW, height: Math.max(1.2, rp - 1.2), rx: 0.6, class: 'zline' }, row);
    else for (let j = 0; j < N; j++) {
      const r = S('rect', { x: MX + j * pitch, y, width: pitch - 1, height: 8, rx: 0.8, class: 'zlog' }, row);
      r.style.opacity = (0.2 + 0.8 * hash01(t + 7, j + 3) ** 2).toFixed(2);
    }
  }
  const zl = act(S('g', { 'data-h': 'logit' }, svg), 'head', 'zT', 'Logits of the last position');
  S('path', { d: `M${MX + MW + 4},${LY1 - 13} h7`, class: 'thin' }, zl);
  mlab(zl, MX + MW + 14, LY1 - 8.5, 'z', 'T', null, { anchor: 'start' });
  S('line', { x1: RC, y1: LY1, x2: RC, y2: DY0 - 2, class: 'branch', 'marker-end': 'url(#arrow-ink)', 'data-h': 'dist' }, svg);

  const db = act(S('g', { 'data-h': 'dist' }, svg), 'dist', null, 'Next-token distribution');
  S('rect', { x: RX0, y: DY0, width: RX1 - RX0, height: DY1 - DY0, rx: 10, class: 'subbox' }, db);
  S('line', { x1: RC, y1: DY1, x2: RC, y2: CY0 - 2, class: 'branch', 'marker-end': 'url(#arrow-ink)', 'data-h': 'decbox' }, svg);
  const dbx = act(S('g', { 'data-h': 'decbox' }, svg), 'decode', null, 'Decoding');
  S('rect', { x: RX0, y: CY0, width: RX1 - RX0, height: CY1 - CY0, rx: 10, class: 'subbox' }, dbx);
  S('g', { id: 'distg' }, svg);
  const lp = S('g', { class: 'loopg', 'data-h': 'loop', tabindex: 0, 'aria-label': 'Append the chosen token and run again' }, svg);
  S('path', { id: 'loophit', d: '', class: 'loophit' }, lp);
  S('path', { id: 'loop', d: '', class: 'loop', 'marker-end': 'url(#arrow)' }, lp);
  S('text', { x: (BX0 + BX1) / 2, y: AH - 20, 'text-anchor': 'middle', class: 'loop-t', text: 'append the chosen token and run again · earlier tokens are not recomputed: their keys and values (or states) come from the cache' }, lp);
}
function setLoop(n) {          // n appended tokens
  const T = toks(LAB.MODELS[state.mech]).ids.length;
  const d = `M${RC},${CY1} V${AH - 14} H72 V${rowY(T + n - 1) + 12}`;
  $('#loop')?.setAttribute('d', d);
  $('#loophit')?.setAttribute('d', d);
}

/* For the focused row: arrows from x̂_t to its own q_t, k_t, v_t, the lookup of its embedding row,
   and per-mechanism extras (convolution window and memory arrows, index scores). */
function drawArcs(m, t, hovering) {
  const g = $('#arcs'); if (!g) return;
  g.replaceChildren();
  g.classList.toggle('idle', !hovering);   // arrows only for the hovered row (or when a step highlights them)
  const y = rowY(t), Q = state._qkv, x0 = AX0 + 14 + 38 + 3;
  const arc = (x2, lift) => S('path', { d: lift ? `M${x0},${y} C${x0 + 11},${y - lift} ${x2 - 14},${y - lift} ${x2 - 2},${y}` : `M${x0},${y} H${x2 - 2}`, class: 'qkvarc', 'marker-end': 'url(#arrow-ink)', 'data-h': 'arc' }, g);
  arc(Q.qx, 0); arc(Q.kx, 15); arc(Q.vx, 22);
  if (Q.sx != null) {             // Gated DeltaNet
    const t0 = Math.max(0, t - 3);
    S('rect', { x: Q.qx - 3, y: rowY(t0) - 10, width: Q.vx + 96 - Q.qx + 3, height: rowY(t) - rowY(t0) + 20, rx: 4, class: 'convwin', 'data-h': 'conv' }, g);
    S('path', { d: `M${Q.vx + 98},${y - 2} C${Q.vx + 112},${y - 18} ${Q.sx - 12},${y - 18} ${Q.sx - 2},${y - 3}`, class: 'qkvarc', 'marker-end': 'url(#arrow-ink)', 'data-h': 'arc2' }, g);
    S('path', { d: `M${Q.sx + 16},${y} H${Q.ox - 2}`, class: 'qkvarc', 'marker-end': 'url(#arrow-ink)', 'data-h': 'arc2' }, g);
  }
  const ib = $('#ibars');
  if (ib && state._ibars) {       // DSA: index scores of row t against every earlier token
    ib.replaceChildren();
    const { x, w } = state._ibars, keep = idxSel(t);
    const mx = Math.max(...Array.from({ length: t + 1 }, (_, s) => idxScore(t, s)));
    for (let s = 0; s <= t; s++) {
      const r = rowAttr(S('rect', { x, y: rowY(s) - 4, width: Math.max(1.5, w * idxScore(t, s) / mx), height: 8, rx: 1.5, class: 'ibar' + (keep.has(s) ? ' on-s' : ''), 'data-pop': `ib:${t}:${s}` }, ib), s);
      r.classList.add('cellhit');
    }
    S('path', { d: `M${x - 4},${y} h-4`, class: 'qkvarc', 'data-h': 'ibar' }, ib);
  }
  const lk = $('#lookup');
  if (lk) {
    lk.replaceChildren();
    $('#arch').querySelectorAll('.estripe').forEach(el => el.classList.toggle('focus', !!hovering && Number(el.dataset.row) === t));
    if (!hovering) { applyArchFocus(); return; }
    const { EX, PX, stripeE, stripeP } = state._lookup;
    const link = (x, ys) => S('path', { d: `M${x},${ys} H${x - 4} V${y} H${x - 1}`, class: 'lookup', 'marker-end': 'url(#arrow-q)', 'data-h': 'emb' }, lk);
    link(EX, stripeE(t));
    if (stripeP) link(PX, stripeP(t));
  }
  applyArchFocus();
}

/* ---------- decoding: logits → probabilities → one token ---------- */
const dec = { method: 'topk', tau: 0.8, k: 5, p: 0.9, minp: 0.1, typ: 0.9, B: 2, pick: null, u: null, su: [0.30, 0.42, 0.55] };
const METHOD = k => LAB.METHODS.find(x => x.key === k);
function decodeNow(m) {
  const data = LAB.nextTokenData(m.key, NEXT_REAL), n = data.cands.length;
  if (dec.method === 'beam') {
    const beam = LAB.beamSearch(data, dec.B), dist = LAB.decodeDist(data, 'temperature', 1);
    dec.pick = beam.best ? beam.best.from : 0;
    return { data, dist, beam };
  }
  if (dec.method === 'spec') {
    const dist = LAB.specDist(data, dec.tau), run = specRun(dist, dec.su);
    dec.pick = run.x;
    return { data, dist, run };
  }
  const dist = LAB.decodeDist(data, dec.method, dec.tau, { k: Math.min(dec.k, n), p: dec.p, minp: dec.minp, typ: dec.typ });
  if (dec.pick == null || !dist[dec.pick] || !dist[dec.pick].kept || dec.method === 'greedy') { dec.pick = argmax(dist); dec.u = null; }
  return { data, dist };
}
const argmax = dist => dist.reduce((b, c, i) => (c.p > dist[b].p ? i : b), 0);
function intervals(dist, key = 'p') {   // cumulative intervals of the kept tokens, in rank order
  let a = 0; const out = {};
  dist.map((c, i) => i).filter(i => dist[i].kept && dist[i][key] > 0).sort((i, j) => (dist[j][key] - dist[i][key]) || (i - j))
    .forEach(i => { out[i] = [a, a + dist[i][key]]; a += dist[i][key]; });
  return out;
}
function draw(iv, u) {
  const keys = Object.keys(iv).map(Number);
  return keys.find(i => u >= iv[i][0] && u < iv[i][1]) ?? keys[keys.length - 1];
}
/* One run of speculative sampling with fixed random numbers u = [draft draw, accept test, residual draw]. */
function specRun(dist, [u1, u2, u3]) {
  const d = draw(intervals(dist, 'q'), u1), a = dist[d].a, acc = u2 < a;
  return { u1, u2, u3, d, a, acc, x: acc ? d : draw(intervals(dist, 'r'), u3) };
}
function resample(m) {
  if (dec.method === 'greedy' || dec.method === 'beam') return;
  if (dec.method === 'spec') { dec.su = [Math.random(), Math.random(), Math.random()]; return; }
  const { dist } = decodeNow(m);
  dec.u = Math.random();
  dec.pick = draw(intervals(dist), dec.u);
}
function setMethod(m, method) {
  dec.method = method; dec.pick = null; dec.u = null;
  drawDist(m); applyArchFocus();
  if (archStepsNow()[state.arch]?.decode) renderArchSteps(m);
}
const pct = (x, d) => `${(100 * x).toFixed(d != null ? d : x < 0.1 ? 1 : 0)}%`;

function drawDist(m) {
  const g = $('#distg'); if (!g) return;
  g.replaceChildren();
  const now = decodeNow(m), { data, dist } = now, spec = dec.method === 'spec', beam = dec.method === 'beam';
  // ---- distribution box: logit bars → softmax → probability bars
  S('text', { x: RC, y: DY0 + 18, 'text-anchor': 'middle', class: 'st', 'data-h': 'dist', text: 'Next-token distribution' }, g);
  const TX = RX0 + 66, LB0 = RX0 + 72, LB1 = RX0 + 116, LV = RX0 + 150, AR = RX0 + 160, PB0 = RX0 + 170, PB1 = RX0 + 226, PV = RX1 - 6;
  const hz = S('text', { x: (LB0 + LV) / 2, y: DY0 + 35, 'text-anchor': 'middle', class: 'st-s', 'data-h': 'dist' }, g);
  S('tspan', { text: 'logit ' }, hz); S('tspan', { text: mi('z'), class: 'mlt' }, hz);
  const hp = S('text', { x: PV, y: DY0 + 35, 'text-anchor': 'end', class: 'st-s', 'data-h': 'dist' }, g);
  S('tspan', { text: mi('p'), class: 'mlt' }, hp); S('tspan', { text: usesTauNow() ? ` = softmax(${mi('z')}/τ)` : ` = softmax(${mi('z')})` }, hp);
  const shown = dist.slice().sort((a, b) => a.rank - b.rank).slice(0, 8);
  const zs = shown.map(c => c.z), zhi = Math.max(...zs), zlo0 = Math.min(...zs), zlo = zlo0 - 0.35 * Math.max(0.5, zhi - zlo0);
  const P = c => spec ? c.p : c.p0, Pf = c => spec ? c.p : c.p;
  const maxp = Math.max(...shown.map(c => Math.max(P(c), Pf(c), spec ? c.q : 0)));
  const beams = beam ? new Set(now.beam.beams.map(b => b.i)) : null;
  shown.forEach((c, i) => {
    const y = DY0 + 52 + i * 17, idx = dist.indexOf(c), picked = idx === dec.pick;
    const cls = 'prow' + (c.kept || spec ? '' : ' cut') + (picked ? ' picked' : '') + (beams && beams.has(idx) ? ' beam' : '');
    const row = act(S('g', { 'data-h': 'dist', class: cls }, g), 'dist', `cand:${idx}`, `${pieceText(c.piece)}: logit ${c.z.toFixed(2)}, probability ${pct(c.p)}`);
    S('rect', { x: RX0 + 4, y: y - 8.5, width: RX1 - RX0 - 8, height: 17, class: 'hitpad' }, row);   // the whole row is one target
    pieceInto(S('text', { x: TX, y: y + 4, 'text-anchor': 'end', class: 'ptok' }, row), c.piece);
    S('rect', { x: LB0, y: y - 4.5, width: Math.max(1.5, (LB1 - LB0) * (c.z - zlo) / (zhi - zlo)), height: 9, rx: 2, class: 'lbar' }, row);
    S('text', { x: LV, y: y + 3.5, 'text-anchor': 'end', class: 'lval', text: c.z.toFixed(1).replace('-', '−') }, row);
    S('text', { x: AR, y: y + 4, 'text-anchor': 'middle', class: 'parr', text: '›' }, row);
    const W = PB1 - PB0;
    if (!spec) S('rect', { x: PB0, y: y - 4.5, width: Math.max(1, W * c.p0 / maxp), height: 9, rx: 2, class: 'pbar0' }, row);
    if (c.kept || spec) S('rect', { x: PB0, y: y - 4.5, width: Math.max(1.5, W * Pf(c) / maxp), height: 9, rx: 2, class: 'pbar' }, row);
    if (spec) S('rect', { x: PB0, y: y - 4.5, width: Math.max(1, W * c.q / maxp), height: 9, rx: 2, class: 'qbar' }, row);
    S('text', { x: PV, y: y + 4, 'text-anchor': 'end', class: 'ppct', text: c.kept || spec ? pct(Pf(c)) : '–' }, row);
  });
  // the rest of the vocabulary
  const restN = (data.source === 'computed' ? m.V : dist.length) - shown.length;
  const shownSet = new Set(shown.map(c => dist.indexOf(c)));
  const rest0 = data.source === 'computed' && dist[0].full ? 1 - shown.reduce((a, c) => a + c.p0, 0) : dist.filter((c, i) => !shownSet.has(i)).reduce((a, c) => a + (spec ? c.p : c.p0), 0);
  const restK = dist.filter((c, i) => !shownSet.has(i)).reduce((a, c) => a + (spec ? c.p : c.p), 0);
  const tr = act(S('g', { 'data-h': 'dist', class: 'prow tail' }, g), 'dist', 'tail', 'The rest of the vocabulary');
  const ty = DY0 + 52 + 8 * 17 + 2;
  S('text', { x: TX, y: ty + 4, 'text-anchor': 'end', class: 'ptok', text: `+ ${LAB.fmt(restN)} more` }, tr);
  S('text', { x: (LB0 + LV) / 2, y: ty + 4, 'text-anchor': 'middle', class: 'lval', text: '⋯' }, tr);
  if (!spec) S('text', { x: PV, y: ty + 4, 'text-anchor': 'end', class: 'ppct', text: `${pct(Math.max(0, rest0))} in total` }, tr);
  else {
    const lg = S('g', { 'data-h': 'dist' }, g);
    S('rect', { x: PB0 - 8, y: ty - 3, width: 10, height: 7, rx: 1.5, class: 'pbar' }, lg);
    S('text', { x: PB0 + 6, y: ty + 3.5, class: 'st-s tiny', text: 'target p' }, lg);
    S('rect', { x: PB0 + 54, y: ty - 3, width: 10, height: 7, rx: 1.5, class: 'qbar' }, lg);
    S('text', { x: PB0 + 68, y: ty + 3.5, class: 'st-s tiny', text: 'draft q' }, lg);
  }
  state._restK = restK;

  // ---- decoding box: method buttons, parameters, the method's own picture, chosen token
  S('text', { x: RC, y: CY0 + 18, 'text-anchor': 'middle', class: 'st', 'data-h': 'decbox', text: 'Decoding' }, g);
  const BW = [[54, 54, 62, 72], [62, 62, 62, 62]], bx0 = [];
  LAB.METHODS.forEach((M, i) => {
    const col = i % 4, row = Math.floor(i / 4), by = CY0 + 28 + row * 24;
    const bw = BW[row][col], bx = RX0 + 8 + BW[row].slice(0, col).reduce((a, w) => a + w + 4, 0);
    const b = S('g', { class: 'mbtn' + (dec.method === M.key ? ' on-m' : ''), 'data-h': 'decbox', tabindex: 0, role: 'button', 'aria-pressed': String(dec.method === M.key), 'aria-label': `Decoding method ${M.label}`, 'data-pop': `meth:${M.key}` }, g);
    S('rect', { x: bx, y: by, width: bw, height: 20, rx: 10, class: 'mbtn-r' }, b);
    S('text', { x: bx + bw / 2, y: by + 13.5, 'text-anchor': 'middle', class: 'mbtn-t', text: M.short }, b);
    const go = () => setMethod(m, M.key);
    b.addEventListener('click', go);
    b.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
  const kept = dist.filter(c => c.kept).length, n = dist.length, tau = dec.tau.toFixed(1);
  const par = {
    greedy: 'the most likely token · τ not used',
    beam: `width B = ${dec.B} · two steps · scores log p at τ = 1`,
    temperature: `τ = ${tau} · every token can be drawn`,
    spec: `τ = ${tau} · draft q is illustrative`,
    topk: `k = ${Math.min(dec.k, n)} · τ = ${tau}`,
    topp: `p = ${dec.p.toFixed(2)} · τ = ${tau} · keeps ${kept}`,
    minp: `p₍base₎ = ${dec.minp.toFixed(2)} · τ = ${tau} · keeps ${kept}`,
    typical: `m = ${dec.typ.toFixed(2)} · τ = ${tau} · keeps ${kept}`,
  }[dec.method];
  const pt = S('text', { x: RC, y: CY0 + 88, 'text-anchor': 'middle', class: 'st-s', 'data-h': 'decbox' }, g);
  par.split('₍base₎').forEach((part, n) => { if (n) S('tspan', { text: 'base', class: 'sub', dy: 3 }, pt); S('tspan', { text: part, dy: n ? -3 : null }, pt); });
  if (beam) drawBeam(g, m, now);
  else if (spec) drawSpec(g, m, now);
  else drawStrip(g, m, now);
  // appended tokens
  const ghosts = beam && now.beam.best ? [dist[now.beam.best.from].piece, now.beam.best.piece] : [dist[dec.pick].piece];
  [0, 1].forEach(i => {
    const gh = $(`#ghost${i}`); if (!gh) return;
    gh.style.display = ghosts[i] ? '' : 'none';
    gh.querySelector('text').textContent = ghosts[i] ? pieceText(ghosts[i]) : '';
  });
  setLoop(ghosts.length);
}
function usesTauNow() { return LAB.usesTau(dec.method); }
function chosenChip(g, piece, id, note, sampleable) {
  const ch = act(S('g', { 'data-h': 'chosen' }, g), 'decode', 'chosen', 'Chosen token');
  S('rect', { x: RX0 + 8, y: CY0 + 172, width: 180, height: 24, rx: 6, class: 'chosen' }, ch);
  const ct = S('text', { x: RX0 + 98, y: CY0 + 188, 'text-anchor': 'middle', class: 'chosen-t' }, ch);
  S('tspan', { text: 'chosen ', class: 'st-m' }, ct);
  S('tspan', { text: `“${pieceText(piece)}”` }, ct);
  S('tspan', { text: note ? `  ${note}` : `  ${id}`, class: 'st-m small' }, ct);
  const sb = S('g', { class: 'mbtn' + (sampleable ? '' : ' dis'), 'data-h': 'decbox', tabindex: 0, role: 'button', 'aria-label': 'Sample again', 'data-pop': 'sample' }, g);
  S('rect', { x: RX0 + 194, y: CY0 + 173, width: RX1 - RX0 - 202, height: 22, rx: 11, class: 'mbtn-r' }, sb);
  S('text', { x: (RX0 + 194 + RX1 - 8) / 2, y: CY0 + 188, 'text-anchor': 'middle', class: 'mbtn-t', text: '↻ Resample' }, sb);
  const m = LAB.MODELS[state.mech];
  const again = () => { if (!sampleable) return; resample(m); drawDist(m); applyArchFocus(); if (archStepsNow()[state.arch]?.decode) renderArchSteps(m); };
  sb.addEventListener('click', again);
  sb.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); again(); } });
}
/* Sampling methods: the kept tokens laid end to end on [0, 1); u picks one. */
function drawStrip(g, m, now) {
  const { dist } = now, iv = intervals(dist), X0 = RX0 + 12, X1 = RX1 - 12, W = X1 - X0, y = CY0 + 112;
  const sg = act(S('g', { 'data-h': 'decbox' }, g), 'decode', null, 'Sampling interval');
  Object.keys(iv).map(Number).forEach((i, n) => {
    const [a, b] = iv[i], c = dist[i], w = (b - a) * W;
    const r = S('rect', { x: X0 + a * W, y, width: Math.max(0.8, w - 0.8), height: 16, class: 'seg ' + (n % 2 ? 'b' : 'a') + (i === dec.pick ? ' picked' : ''), 'data-pop': `seg:${i}` }, sg);
    r.classList.add('cellhit');
    if (w > 38) S('text', { x: X0 + a * W + w / 2, y: y + 11.5, 'text-anchor': 'middle', class: 'seg-t' + (i === dec.pick ? ' on' : ''), text: pieceText(c.piece) }, sg);
  });
  S('text', { x: X0, y: y + 28, 'text-anchor': 'middle', class: 'tiny', text: '0' }, sg);
  S('text', { x: X1, y: y + 28, 'text-anchor': 'middle', class: 'tiny', text: '1' }, sg);
  if (dec.u != null && dec.method !== 'greedy') {
    const ux = X0 + dec.u * W;
    S('path', { d: `M${ux},${y - 2} V${y + 18} M${ux - 4},${y - 8} L${ux + 4},${y - 8} L${ux},${y - 2} Z`, class: 'umark' }, sg);
    S('text', { x: Math.min(X1 - 30, Math.max(X0 + 30, ux)), y: y - 11, 'text-anchor': 'middle', class: 'tiny ink', text: `u = ${dec.u.toFixed(3)}` }, sg);
  }
  const kept = dist.filter(c => c.kept), mass = kept.length ? kept[0].mass : 1;
  const note = {
    greedy: 'one token holds the whole interval',
    temperature: `all ${dist.length} listed tokens, widths = p(τ)`,
    topk: `the top ${kept.length}, which held ${pct(mass, 1)}, rescaled to 100%`,
    topp: `${kept.length} tokens reach ${pct(mass, 1)} ≥ p, rescaled to 100%`,
    minp: `${kept.length} tokens have p ≥ ${dec.minp.toFixed(2)} × ${pct(dist[0].pmax, 1)}`,
    typical: `${kept.length} tokens closest to H = ${dist[0].H.toFixed(2)} nats, rescaled`,
  }[dec.method];
  S('text', { x: RC, y: y + 44, 'text-anchor': 'middle', class: 'st-s', text: note }, sg);
  const c = dist[dec.pick];
  chosenChip(g, c.piece, c.id, null, dec.method !== 'greedy');
}
/* Speculative sampling: draft proposal, acceptance test, and the residual when it fails. */
function drawSpec(g, m, now) {
  const { dist, run } = now, d = dist[run.d], X0 = RX0 + 12, X1 = RX1 - 12, W = X1 - X0, y = CY0 + 118;
  const sg = act(S('g', { 'data-h': 'decbox' }, g), 'decode', 'spec', 'Speculative sampling');
  const t1 = S('text', { x: RC, y: CY0 + 103, 'text-anchor': 'middle', class: 'st-s' }, sg);
  S('tspan', { text: 'draft proposes ' }, t1);
  S('tspan', { text: `“${pieceText(d.piece)}”`, class: 'ink' }, t1);
  S('tspan', { text: ` · q ${pct(d.q)}, p ${pct(d.p)}` }, t1);
  S('rect', { x: X0, y, width: W * run.a, height: 14, class: 'seg acc' }, sg);
  if (run.a < 1) S('rect', { x: X0 + W * run.a, y, width: W * (1 - run.a), height: 14, class: 'seg rej' }, sg);
  if (run.a > 0.25) S('text', { x: X0 + W * run.a / 2, y: y + 10.5, 'text-anchor': 'middle', class: 'seg-t on', text: `accept ${run.a.toFixed(2)}` }, sg);
  if (run.a < 0.8) S('text', { x: X0 + W * (1 + run.a) / 2, y: y + 10.5, 'text-anchor': 'middle', class: 'seg-t', text: 'reject' }, sg);
  const ux = X0 + run.u2 * W;
  S('path', { d: `M${ux},${y - 2} V${y + 16} M${ux - 4},${y - 8} L${ux + 4},${y - 8} L${ux},${y - 2} Z`, class: 'umark' }, sg);
  S('text', { x: RC, y: y + 29, 'text-anchor': 'middle', class: 'st-s', text: run.acc
    ? `u = ${run.u2.toFixed(2)} < min(1, p/q) = ${run.a.toFixed(2)} → accepted`
    : `u = ${run.u2.toFixed(2)} ≥ ${run.a.toFixed(2)} → rejected, draw from r ∝ max(0, p − q)` }, sg);
  S('text', { x: RC, y: y + 44, 'text-anchor': 'middle', class: 'st-s', text: `acceptance rate Σ min(p, q) = ${pct(dist[0].acc, 1)}` }, sg);
  const c = dist[run.x];
  chosenChip(g, c.piece, c.id, run.acc ? 'draft' : 'from r', true);
}
/* Beam search: B beams after step 1, their best continuations, the B best pairs survive. */
function drawBeam(g, m, now) {
  const { dist, beam } = now, y0 = CY0 + 100;
  const bg = act(S('g', { 'data-h': 'decbox chosen' }, g), 'decode', 'beamtree', 'Beam search tree');
  S('text', { x: RX0 + 12, y: y0, class: 'tiny', text: 'step 1 · log p' }, bg);
  S('text', { x: RX0 + 120, y: y0, class: 'tiny', text: 'step 2' }, bg);
  S('text', { x: RX1 - 12, y: y0, 'text-anchor': 'end', class: 'tiny', text: 'Σ log p' }, bg);
  let row = 0;
  beam.beams.forEach((b, bi) => {
    const exps = beam.exps.filter(e => e.from === b.i).sort((a, c) => c.score - a.score).slice(0, 2);
    const yb = y0 + 14 + row * 14 + (exps.length - 1) * 7;
    const bt = S('text', { x: RX0 + 12, y: yb + 4, class: 'btok' + (beam.best && beam.best.from === b.i ? ' best' : ''), 'data-pop': `beam:${bi}` }, bg);
    S('tspan', { text: '“' }, bt); pieceInto(bt, dist[b.i].piece); S('tspan', { text: '”' }, bt);
    bt.classList.add('cellhit');
    S('text', { x: RX0 + 96, y: yb + 4, 'text-anchor': 'end', class: 'lval', text: b.lp.toFixed(2).replace('-', '−') }, bg);
    exps.forEach((e, k) => {
      const ye = y0 + 14 + row * 14, best = beam.best === e;
      S('path', { d: `M${RX0 + 100},${yb} C${RX0 + 108},${yb} ${RX0 + 106},${ye} ${RX0 + 116},${ye}`, class: 'blink' + (e.kept ? ' kept' : '') + (best ? ' best' : '') }, bg);
      const et = S('text', { x: RX0 + 120, y: ye + 4, class: 'btok' + (e.kept ? '' : ' pruned') + (best ? ' best' : ''), 'data-pop': `bexp:${beam.exps.indexOf(e)}` }, bg);
      S('tspan', { text: '“' }, et); pieceInto(et, e.piece); S('tspan', { text: '”' }, et);
      et.classList.add('cellhit');
      S('text', { x: RX1 - 12, y: ye + 4, 'text-anchor': 'end', class: 'lval' + (e.kept ? ' ink' : ''), text: `${e.score.toFixed(2).replace('-', '−')}${e.kept ? '  ✓' : ''}` }, bg);
      row++;
    });
  });
  if (beam.best) {
    const t = S('text', { x: RC, y: CY1 - 8, 'text-anchor': 'middle', class: 'st-s' }, bg);
    S('tspan', { text: 'best: ' }, t);
    S('tspan', { class: 'ink', text: `“${pieceText(dist[beam.best.from].piece)}${pieceText(beam.best.piece)}”` }, t);
    S('tspan', { text: beam.greedy && beam.greedy !== beam.best && beam.greedy.text !== beam.best.text ? ` · greedy: “${pieceText(dist[beam.greedy.from].piece)}${pieceText(beam.greedy.piece)}”` : '' }, t);
  }
}

/* Linked highlighting: hovering anything on a token's row lights up that row everywhere. */
function wireRows(svg) {
  if (svg.dataset.wired) return;
  svg.dataset.wired = '1';
  svg.addEventListener('pointerover', e => {
    const el = e.target.closest('[data-row]');
    const w = e.target.closest('[data-word]');
    if (el && el.closest('#ibars')) return;        // index bars describe the focused row, not their own
    setHover(el ? Number(el.dataset.row) : null, w && !el ? Number(w.dataset.word) : null);
  });
  svg.addEventListener('pointerleave', () => setHover(null, null));
}
let hoverRow = null;
function setHover(t, word) {
  const svg = $('#arch'), m = LAB.MODELS[state.mech], tk = toks(m);
  const rows = new Set();
  if (t != null) rows.add(t);
  if (word != null) tk.words.forEach((w, i) => { if (w === word) rows.add(i); });
  svg.querySelectorAll('[data-row]').forEach(el => el.classList.toggle('rowhl', rows.has(Number(el.dataset.row)) && !el.closest('#ibars')));
  svg.querySelectorAll('.word').forEach(el => el.classList.toggle('rowhl', word != null ? Number(el.dataset.word) === word : (t != null && tk.words[t] === Number(el.dataset.word))));
  svg.querySelectorAll('.wordbr').forEach(el => el.classList.toggle('rowhl', word != null ? Number(el.dataset.word) === word : (t != null && tk.words[t] === Number(el.dataset.word))));
  const target = t != null ? t : state.row, key = `${target}:${t != null}`;
  if (key !== hoverRow) { hoverRow = key; drawArcs(m, target, t != null); }
}

/* ---------- pop-ups: title, subtitle, text, calculation lines (TeX, or {t} for plain text) ---------- */
/* Expanded, keyboard-accessible explanations for the compact diagram legend. */
function legendPopupContent(symbol, m) {
  const recurrent = m.key === 'gdn' && view(m).layer !== 'gattn';
  const details = {
    emb: ['Embedding table $E$', 'Tokens become vectors',
      'Each token ID selects a learned row from this table. That vector is the starting numerical representation of the token; the model updates it as the text passes through its layers.',
      [String.raw`e_t = E[\operatorname{token\ ID}_t]`]],
    pos: ['Position table $P$', 'Learned positions · GPT-2',
      'GPT-2 learns a separate vector for each position in the input. The position vector is added to the token embedding so the model can distinguish the same token appearing in different places.',
      [String.raw`h_t = E[\operatorname{ID}_t] + P_t`]],
    xhat: ['Normalized token vector $\\hat{x}$', `Normalization · ${m.norm}`,
      'A normalized copy of a token’s current representation. It keeps numerical scales stable before the model computes queries, keys, values or other mixer inputs.',
      [String.raw`\hat{x}_t = \operatorname{${m.norm}}(h_t)`]],
    q: ['Query $Q$', 'What am I looking for?',
      'A query is a learned representation used to request information from earlier tokens. Attention compares it with keys to find useful context. Different query heads can look for different things.',
      [String.raw`\operatorname{score}(t,s) \propto q_t \cdot k_s`]],
    k: ['Key $K$', 'How can this token be found?',
      'A key represents what information a token makes available for matching. Queries are compared with keys; in ordinary attention, past keys can be cached so they do not need to be recomputed.',
      [String.raw`\operatorname{score}(t,s) \propto q_t \cdot k_s`]],
    v: ['Value $V$', 'What information does this token contribute?',
      'A value holds information that another token can retrieve. Attention mixes past value vectors in the proportions determined by query–key matching. Keys determine where to look; values supply the result.',
      [String.raw`o_t = \textstyle\sum_{s\le t} a_{t,s}v_s`]],
    rope: ['RoPE rotation dials', 'Position information in queries and keys',
      'Each circle illustrates one pair of coordinates being rotated according to token position. This lets attention account for relative positions without adding a separate learned position table. The dials are schematic.',
      [String.raw`\tilde q_t = R_t q_t,\qquad \tilde k_t = R_t k_t`]],
    ckv: ['Compressed KV vector $C^{KV}$', 'Multi-head latent attention',
      'Instead of caching full keys and values for every attention head, MLA stores a compact latent representation for each token. When needed, the model uses projections of this vector to construct keys and values.',
      [String.raw`c_t^{KV} = \hat{x}_t W^{DKV}`]],
    kr: ['Rotary key $k^{R}$', 'Position-dependent key · MLA',
      'MLA stores a small rotary key separately from the compressed content vector. It carries position information shared across attention heads, while the compressed latent carries much of the content.',
      [String.raw`k_t^{R} = \operatorname{RoPE}(\hat{x}_t W^{KR})`]],
    ki: ['Index key $k^{I}$', 'Sparse attention · finding candidates',
      'The sparse indexer uses a lightweight key for each past token to estimate which positions are worth reading with the full attention mechanism. It is separate from the main attention key.',
      [String.raw`k_s^{I} = \operatorname{LN}(\hat{x}_s W^{IK})`]],
    ibar: ['Selection scores $I_t$', 'Sparse attention · choosing what to read',
      'The indexer scores earlier positions and selects the highest-scoring ones. Full attention then operates on the selected tokens instead of reading the entire context. The diagram shows an illustrative selection.',
      [String.raw`\mathcal{S}_t = \operatorname{TopK}(I_{t,:})`]],
    conv: ['Short causal convolution', 'Gated DeltaNet · local order',
      'Before updating its recurrent memory, Gated DeltaNet combines each query, key and value with their recent history. The kernel covers the current token and up to three earlier tokens.',
      [String.raw`\tilde q_t = \operatorname{SiLU}(\operatorname{conv}_4(q)_t)`]],
    'gate-a': ['Forget gate $\\alpha$', 'Gated DeltaNet · retain old memory',
      'This gate controls how strongly the previous recurrent memory carries forward. A value near one keeps more of the old state; a smaller value forgets more.',
      [String.raw`0 < \alpha_t \le 1`]],
    'gate-b': ['Write gate $\\beta$', 'Gated DeltaNet · update memory',
      'This gate controls the strength of the change made by the current token’s key and value. It determines how strongly the new information corrects or overwrites the previous state.',
      [String.raw`0 < \beta_t < 1`]],
    state: ['Memory state $S_t$', 'Gated DeltaNet · fixed-size memory',
      'A matrix summarizes the information carried forward after processing token t. Unlike an attention KV cache, the size of this recurrent state does not grow with the number of tokens.',
      [String.raw`S_t \in \mathbb{R}^{d_v \times d_k}`]],
    read: ['State readout $o$', 'Gated DeltaNet · retrieving memory',
      'The current query reads from the fixed-size memory matrix to produce an output vector. This replaces the weighted sum over all earlier value vectors used by standard attention.',
      [String.raw`o_t = S_t\hat q_t`]],
    zg: ['Output gate $z$', 'Gated DeltaNet · shaping the readout',
      'The readout is normalized and multiplied by an activation of z before the head outputs are combined. The gate lets the layer modulate which components of the result are passed on.',
      [String.raw`y_t \propto \operatorname{SiLU}(z_t) \odot \operatorname{Norm}(o_t)`]],
    gt: ['Gated attention $\\sigma(G)$', 'An extra gate on attention output',
      'In the gated-attention layers, a learned gate scales each attention head’s output before the heads are projected back to the model width. This is distinct from Gated DeltaNet’s recurrent memory gates.',
      [String.raw`0 < \sigma(G) < 1`]],
    moe: ['Mixture of Experts (MoE)', 'Select a few feed-forward networks',
      'A router chooses a subset of specialized feed-forward networks for each token. The chosen experts process the token, and their contributions are combined. This lets the model have many parameters without using every expert for every token.',
      [String.raw`y = \textstyle\sum_{i\in\operatorname{selected}} w_i F_i(x)`]],
    att: [`Attention weights $A_{${state.head}}$`, `Head ${state.head} · who attends to whom`,
      'Each row describes how strongly one token reads earlier tokens in the selected head. Masked future positions receive no weight; the allowed weights sum to one. Colors in this diagram are illustrative, not model activations.',
      [String.raw`\textstyle\sum_{s\le t} a_{t,s}^{(h)} = 1`]],
    y: ['Mixer output $Y$', recurrent ? 'Gated DeltaNet output' : 'Attention output',
      recurrent ? 'The readout of the recurrent state is normalized, gated and projected back to the token-vector width. This result is added to the residual stream.' : 'Each attention head combines information from earlier token values; the head outputs are joined and projected back to the token-vector width. This result is added to the residual stream.',
      recurrent ? [String.raw`Y = \operatorname{Project}(\operatorname{gated\ readout})`] : [String.raw`Y = \operatorname{Concat}(o_1,\ldots,o_H)W^O`]],
    logit: ['Logits $Z$', 'Scores before probabilities',
      'The language-model head assigns a score to every vocabulary token. These raw scores are logits, not probabilities. A decoding rule uses their softmax probabilities to select the next token. The demo scores are illustrative unless stated otherwise.',
      [String.raw`p = \operatorname{softmax}(z)`]],
  };
  return details[symbol] || null;
}

function popContent(key) {
  const m = LAB.MODELS[state.mech], tk = toks(m), [kind, a, b] = key.split(':');
  if (kind === 'legend') return legendPopupContent(a, m);
  const t = Number(a), f = LAB.fmt, D = tf(m.d), H = m.H, G = m.G, T = tk.ids.length, v = view(m);
  const tokName = i => `“${pieceText(tk.pieces[i])}”`;
  const wordOf = i => LAB.TOKENS.sentence.split(' ')[tk.words[i]];
  const dims = (x, y) => `[${tf(x)}] \\cdot [${tf(x)} \\times ${tf(y)}] = [${tf(y)}]`;
  const N = m.norm, mla = m.key === 'mla' || m.key === 'dsa', gattn = m.key === 'gdn' && v.layer === 'gattn';
  const qdim = mla ? m.dh + m.dr : m.dh;
  const bias = m.key === 'mha' || m.key === 'gqa' ? ' + b' : '';
  const normCalc = N === 'LayerNorm'
    ? [String.raw`\mu = \tfrac{1}{${D}} \textstyle\sum_i h_i`, String.raw`\sigma^2 = \tfrac{1}{${D}} \textstyle\sum_i (h_i - \mu)^2`, String.raw`\hat{x} = \gamma \odot \frac{h - \mu}{\sqrt{\sigma^2 + ${m.eps}}} + \beta`]
    : [String.raw`\operatorname{rms} = \sqrt{\tfrac{1}{${D}} \textstyle\sum_i h_i^2 + ${m.eps}}`, String.raw`\hat{x} = ${m.normPlusOne ? '(1 + \\gamma)' : '\\gamma'} \odot h \,/\, \operatorname{rms}`];
  const geluDef = String.raw`\operatorname{GELU}(x) \approx \tfrac{x}{2}\big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^{3})\big)\big)`;
  const siluDef = String.raw`\operatorname{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}`;
  const mixName = m.key === 'gdn' && !gattn ? 'GDN' : 'Attn';
  const txt = s => ({ t: s });
  switch (kind) {
    case 'ex': { const ex = LAB.EXAMPLES.find(e => e.key === a); return [`Example: ${ex.label}`, `“${ex.sentence}”`, `${ex.note} Click to use this input.`, []]; }
    case 'text': return ['Input text', LAB.TOKENS.sentence, 'The model never sees letters: the tokenizer first turns this into token IDs.',
      [txt(`→ [${tk.ids.join(', ')}]`), String.raw`T = ${T} \text{ tokens}`]];
    case 'word': {
      const rows = tk.words.map((w, i) => w === t ? i : -1).filter(i => i >= 0);
      const letters = rows.filter(i => /[A-Za-z]/.test(tk.pieces[i])).length;
      return [`“${LAB.TOKENS.sentence.split(' ')[t]}”`, rows.length > 1 ? `${rows.length} tokens` : '1 token',
        letters > 1 ? 'Not in the vocabulary as a whole word, so it is built from more frequent pieces.'
          : rows.length > 1 ? 'The word is one vocabulary entry; the punctuation after it is a token of its own.' : 'Frequent enough to be a single vocabulary entry.',
        rows.map(i => txt(`${pieceText(tk.pieces[i])} → ID ${f(tk.ids[i])}`))];
    }
    case 'tokenizer': return ['Tokenizer', `${m.tokenizer} · vocabulary ${f(m.V)}`,
      `Splits the text into known pieces and maps each to an ID. ␣ marks a piece that starts with a space (stored as ${m.key === 'mqa' ? '▁' : 'Ġ'}).${tk.pieces[0].startsWith('<') ? ' This tokenizer also adds a beginning-of-sequence token.' : ''}`,
      [String.raw`x_t \in \{0, \dots, ${tf(m.V - 1)}\}, \quad t = 0, \dots, ${T - 1}`]];
    case 'tok': {
      const p = tk.pieces[t];
      if (p.startsWith('<')) return ['Beginning of sequence', `ID ${tk.ids[t]}`, 'Added by the tokenizer in front of the text. It gives every later token something to attend to.', [txt(`${p} → ${tk.ids[t]}`)]];
      return [`Position ${t}: ${tokName(t)}`, `ID ${f(tk.ids[t])}`, `From the word “${wordOf(t)}”. ${tk.words.filter(w => w === tk.words[t]).length > 1 ? 'The word was split into several tokens.' : 'A whole word.'}`,
        [String.raw`x_{${t}} = ${tk.ids[t]}`]];
    }
    case 'etable': return ['Embedding matrix E', `${f(m.V)} rows × ${f(m.d)} numbers`, 'One learned row per vocabulary entry; a token ID selects its row. The marks show where this sentence’s tokens sit, in proportion to their IDs; the blue one is the highlighted token.',
      [String.raw`E \in \mathbb{R}^{${tf(m.V)} \times ${D}}`, String.raw`${tf(m.V)} \cdot ${D} = ${tf(m.V * m.d)} \text{ numbers}`, m.tied ? String.raw`W_U = E^{\top} \text{ (tied)}` : String.raw`\text{not shared with } W_U`]];
    case 'ptable': return ['Position matrix P', `${f(m.ctx)} rows × ${f(m.d)}`, `One learned row per position; row t is added to the embedding of the token at position t. This sentence uses rows 0–${T - 1}, the first ${T} of ${f(m.ctx)}, so all marks sit at the top.`,
      [String.raw`P \in \mathbb{R}^{${tf(m.ctx)} \times ${D}}`, String.raw`${tf(m.ctx)} \cdot ${D} = ${tf(m.ctx * m.d)} \text{ numbers}`]];
    case 'emb': return [`Embedding of ${tokName(t)}`, `row ${f(tk.ids[t])} of E · ${f(m.d)} numbers`, 'A lookup, not a computation: multiplying a one-hot vector by E just selects one row.',
      [String.raw`e_{${t}} = \operatorname{onehot}(${tk.ids[t]})\, E = E[${tk.ids[t]}]`, String.raw`[1 \times ${tf(m.V)}] \cdot [${tf(m.V)} \times ${D}] = [1 \times ${D}]`,
        m.pos === 'learned' ? String.raw`h_{0,${t}} = E[${tk.ids[t]}] + P[${t}]` : m.embScale ? String.raw`h_{0,${t}} = \sqrt{${m.d}}\, E[${tk.ids[t]}] = ${Math.sqrt(m.d).toFixed(2)}\, E[${tk.ids[t]}]` : String.raw`h_{0,${t}} = E[${tk.ids[t]}]`]];
    case 'pos': return [`Position ${t} (counted from 0)`, `row ${t} of P · ${f(m.d)} numbers`, `GPT-2 adds a learned vector for each of its ${f(m.ctx)} positions, so the same token at another position gets a different input.`,
      [String.raw`h_{0,${t}} = E[${tk.ids[t]}] + P[${t}]`, String.raw`[${D}] + [${D}] = [${D}]`]];
    case 'scale': return ['Embedding scale', `× √${m.d}`, 'Gemma multiplies every embedding by √d before the first block.', [String.raw`\sqrt{${m.d}} = ${Math.sqrt(m.d).toFixed(2)}`, String.raw`h_0 = ${Math.sqrt(m.d).toFixed(2)} \cdot E[x]`]];
    case 'stream': return ['Residual stream h', `${T} × ${f(m.d)}: one row per token`, 'The running state of the model. It starts as the embeddings; every sublayer reads a normalized copy and adds its output back, so it collects the contribution of every block until the LM head reads it.',
      [String.raw`h_0: ${T} \times ${D} = ${tf(T * m.d)} \text{ numbers}`, String.raw`h \leftarrow h + \operatorname{${mixName}}\big(\operatorname{${N}}(h)\big)`, String.raw`h \leftarrow h + \operatorname{${moeShown(m) ? 'MoE' : 'FFN'}}\big(\operatorname{${N}}(h)\big)`, String.raw`\times ${m.L} \text{ blocks, shape unchanged}`]];
    case 'norm1': case 'norm2': return [m.norm, `per token · [${f(m.d)}] → [${f(m.d)}]`, m.norm === 'LayerNorm' ? 'Centers each token’s vector, scales it to unit variance, then applies a learned scale γ and bias β.' : 'Divides each token’s vector by its root mean square, then applies a learned scale. No mean subtraction.', normCalc];
    case 'xhat': return [`x̂ for ${tokName(t)}`, `${f(m.d)} numbers`, `The normalized copy of row ${t + 1} of the stream. Everything the ${m.key === 'gdn' && !gattn ? 'mixer' : 'attention'} sublayer computes for this token starts from it.`, [String.raw`\hat{x}_{${t}} = \operatorname{${N}}(h_{${t}})`, ...normCalc.slice(-1)]];
    case 'q': {
      const hd = Number(b), per = (m.drawH ? m.H / m.drawH : 1);
      const heads = per > 1 ? `heads ${hd}–${hd + per - 1}` : `head ${hd}`;
      if (m.key === 'dsa') return [`q for ${tokName(t)}, ${heads}`, `${per} × ${qdim} numbers`, `${tokName(t)} gets its own query in each of the ${H} heads, through the compressed query latent c^Q. Click to follow head ${hd}.`,
        [String.raw`c^{Q} = \operatorname{RMSNorm}(\hat{x}\, W^{DQ}) \in \mathbb{R}^{${tf(m.dq)}}`, String.raw`q = c^{Q} W^{UQ}: [${tf(m.dq)}] \to [${H} \times (${m.dh} + ${m.dr})]`]];
      if (gattn) return [`q for ${tokName(t)}, head ${hd}`, `${m.dh} numbers`, `Head ${hd} reads KV head ${LAB.kvHeadOf(m, hd)}. The same projection also yields the token’s output gate g.`,
        [String.raw`[\,q \;\; g\,] = \hat{x}\, W^{Q}: [${D}] \to [2 \times ${H} \times ${m.dh}]`, String.raw`q \leftarrow \operatorname{RMSNorm}(q) \text{ per head}`]];
      return [`q for ${tokName(t)}, head ${hd}`, `${qdim} numbers`, `${tokName(t)} gets its own query in each of the ${H} heads. ${mla ? '' : `Head ${hd} compares it with KV head ${LAB.kvHeadOf(m, hd)}.`} Click to follow head ${hd}.`,
        [String.raw`q = \hat{x}\, W^{Q}${bias}`, dims(m.d, H * qdim), String.raw`= ${H} \text{ heads} \times ${qdim}${mla ? String.raw` \;(${m.dh} + ${m.dr} \text{ rotary})` : ''}`]];
    }
    case 'k': return [`k for ${tokName(t)}, KV head ${b}`, `${m.dh} numbers · cached`, G === H ? `Only query head ${b} reads it.` : G === 1 ? `All ${H} query heads read this one key.` : `Read by query heads ${(Number(b) - 1) * H / G + 1}–${Number(b) * H / G}.`,
      [String.raw`k = ${gattn ? '\\operatorname{RMSNorm}(\\hat{x}\\, W^{K})' : `\\hat{x}\\, W^{K}${bias}`}`, dims(m.d, G * m.dh), String.raw`= ${G} \text{ head${G > 1 ? 's' : ''}} \times ${m.dh}`]];
    case 'v': return [`v for ${tokName(t)}, KV head ${b}`, `${m.dh} numbers · cached`, 'What this token passes on to every later token that attends to it.',
      [String.raw`v = \hat{x}\, W^{V}${bias}`, dims(m.d, G * m.dh), String.raw`= ${G} \text{ head${G > 1 ? 's' : ''}} \times ${m.dh}`]];
    case 'c': return [`Latent c for ${tokName(t)}`, `${m.dc} numbers · cached`, `Stands in for the keys and values of all ${H} heads; each head expands it with its own W^UK and W^UV.`,
      [String.raw`c = \hat{x}\, W^{DKV}`, dims(m.d, m.dc), String.raw`k^{C}_h = c\, W^{UK}_h: [${m.dc}] \to [${m.dh}]`, String.raw`v_h = c\, W^{UV}_h: [${m.dc}] \to [${m.dv}]`]];
    case 'kr': return [`Rotary key for ${tokName(t)}`, `${m.dr} numbers · cached`, 'Carries the position. One key shared by all heads.', [String.raw`k^{R} = \operatorname{RoPE}(\hat{x}\, W^{KR})`, dims(m.d, m.dr)]];
    case 'ki': return [`Index key for ${tokName(t)}`, `${m.index.dim} numbers · cached in FP8`, 'What the lightning indexer compares later queries against. One key per token, shared by all 64 index heads.',
      [String.raw`k^{I} = \operatorname{LN}(\hat{x}\, W^{IK})`, dims(m.d, m.index.dim)]];
    case 'ib': {
      const s2 = Number(b), keep = idxSel(t), order = Array.from({ length: t + 1 }, (_, i) => i).sort((x, y) => idxScore(t, y) - idxScore(t, x));
      return [`Index score: ${tokName(t)} → ${tokName(s2)}`, keep.has(s2) ? 'selected' : 'not selected', 'Illustrative score: the real one depends on the weights. Each query keeps its top scores; the toy budget here is 4 (DeepSeek-V3.2: 2,048).',
        [String.raw`I_{${t},${s2}} = \textstyle\sum_{j=1}^{64} w_{${t},j}\, \operatorname{ReLU}\big(q^{I}_{${t},j} \cdot k^{I}_{${s2}}\big)`, String.raw`\text{rank } ${order.indexOf(s2) + 1} \text{ of } ${t + 1} ${keep.has(s2) ? '\\le' : '>'} ${TOY_K} \;\Rightarrow\; \text{${keep.has(s2) ? 'attended' : 'masked'}}`]];
    }
    case 'rope': {
      const dd = m.key === 'gdn' ? m.ropeDim : mla ? m.dr : m.dh, pairs = dd / 2, th = m.thetaN;
      const ramp = i => mla ? Math.min(1, Math.max(0, (i - m.yarn.low) / (m.yarn.high - m.yarn.low))) : 0;
      const ang = i => t * Math.pow(th, -2 * i / dd) * ((1 - ramp(i)) + ramp(i) / (mla ? m.yarn.factor : 1));
      const num = x => { if (x >= 0.01 || x === 0) return x.toFixed(x >= 10 ? 1 : 3); const [mant, ex] = x.toExponential(1).split('e'); return String.raw`${mant} \times 10^{${Number(ex)}}`; };
      const what = b === 'q' ? (mla ? `each head’s rotary query q^R (${m.dr} of its ${qdim} numbers)` : gattn ? `the first ${m.ropeDim} of the ${m.dh} dims of every query head` : `all ${H} query heads`)
        : (mla ? 'the shared rotary key k^R' : gattn ? `the first ${m.ropeDim} dims of the ${G} key heads` : `the ${G} key head${G > 1 ? 's' : ''}`);
      return [`${mla ? 'Decoupled RoPE' : gattn ? 'Partial RoPE' : 'RoPE'} on ${b} for ${tokName(t)}`, `position ${t} · ${pairs} rotated pairs per head`, `Rotates ${what} pair by pair. The dial shows pair 0, which turns 1 radian per position.${mla ? ' The latent c is not rotated.' : ''}`,
        [String.raw`\theta_i = ${tf(th)}^{-2i/${dd}}, \quad \text{angle} = t\,\theta_i`, String.raw`\text{pair } 0: \; ${t} \cdot 1 = ${num(ang(0))} \text{ rad } (${(ang(0) * 180 / Math.PI).toFixed(0)}^{\circ})`,
          String.raw`\text{pair } ${pairs / 2}: \; ${num(ang(pairs / 2))} \text{ rad}`, String.raw`\text{pair } ${pairs - 1}: \; ${num(ang(pairs - 1))} \text{ rad}`,
          ...(mla ? [String.raw`\text{YaRN: pairs } 0\text{–}${m.yarn.low} \text{ unchanged, } ${m.yarn.high}\text{–}${pairs - 1} \text{ slowed } ${m.yarn.factor}\times`] : [])]];
    }
    case 'ropelabel': return mla ? ['Decoupled RoPE', `rotary part: ${m.dr} of ${qdim} per query head, k^R shared`, 'Position enters only through separate rotary parts, so the latent c stays position-free and W^UK can be folded into the queries at inference.', [String.raw`q \cdot k = q^{C} \cdot k^{C} + q^{R} \cdot k^{R}`, String.raw`\operatorname{RoPE} \text{ acts on } q^{R} \text{ and } k^{R} \text{ only}`]]
      : gattn ? ['Partial RoPE', `${m.ropeDim} of ${m.dh} dims per head · θ = 10,000,000`, 'Only a quarter of each query and key head is rotated; the rest is position-free. Only the gated-attention layers use RoPE; Gated DeltaNet layers get order from their recurrence.', [String.raw`\tilde{q}_t = \big[\,R_t\, q_t^{[1:64]} \;\; q_t^{[65:256]}\,\big]`]]
      : ['RoPE', `θ = ${f(m.thetaN)} · ${m.dh / 2} pairs per head`, 'Queries and keys are rotated by their position, so a query–key score depends only on the distance between the two tokens. Values are not rotated.', [String.raw`\tilde{q}_t^{\top} \tilde{k}_s = q_t^{\top} R_{s-t}\, k_s`]];
    case 'cache': {
      const per = LAB.cacheBytesPerToken(m), nL = LAB.cacheLayers(m);
      const first = m.key === 'mla' ? String.raw`${m.dc} + ${m.dr} = ${LAB.cacheValues(m)} \text{ values}`
        : m.key === 'dsa' ? String.raw`${m.dc} + ${m.dr} \;(\text{16-bit}) + ${m.index.dim} \;(\text{FP8})`
        : String.raw`2 \cdot ${G} \cdot ${m.dh} = ${tf(LAB.cacheValues(m))} \text{ values}`;
      return ['KV cache', `${f(LAB.cacheValues(m))} values per token per layer`, `Stored for every token so that later tokens can attend to it without recomputing it.${m.key === 'gdn' ? ' Only the 12 gated-attention layers keep one.' : ''}`,
        [first, String.raw`\times ${nL} \text{ layers} \Rightarrow \text{${LAB.bytes(per)} per token}`, String.raw`\times ${tf(m.ctx)} \text{ tokens} = \text{${LAB.bytes(per * m.ctx)}}`]];
    }
    case 'att': {
      const s2 = Number(b), h = state.head, scale = mla ? String.raw`\sqrt{${qdim}}` : String.raw`\sqrt{${m.dh}}`;
      if (s2 > t) return [`${tokName(t)} → ${tokName(s2)}`, 'masked', `${tokName(s2)} comes later in the text, so the causal mask hides it.`, [String.raw`s = -\infty \;\Rightarrow\; e^{-\infty} = 0 \;\Rightarrow\; a = 0`]];
      if (m.key === 'dsa' && !idxSel(t).has(s2)) return [`${tokName(t)} → ${tokName(s2)}`, 'not selected by the indexer', `Allowed by the causal mask, but its index score is not among the top ${TOY_K} for this row (toy budget), so it is masked like a future token.`, [String.raw`s \notin \mathcal{S}_{${t}} \;\Rightarrow\; a_{${t},${s2}} = 0`]];
      const over = m.key === 'dsa' ? String.raw`s' \in \mathcal{S}_{${t}}` : String.raw`s' \le ${t}`;
      return [`${tokName(t)} attends to ${tokName(s2)}`, `head ${h} · a(${t}, ${s2})`, 'Shading is schematic: this page runs no model, so the actual weights are not known.',
        [String.raw`s_{${t},${s2}} = q_{${t}} \cdot k_{${s2}} \,/\, ${scale}${mla ? String.raw` \cdot ${m.ms2}` : ''}`, String.raw`a_{${t},${s2}} = e^{s_{${t},${s2}}} \Big/ \textstyle\sum_{${over}} e^{s_{${t},s'}}`,
          String.raw`\text{row } ${t} \text{ sums to 1 over ${m.key === 'dsa' ? Math.min(t + 1, TOY_K) : t + 1} keys}`, ...(mla ? [String.raw`${m.ms2}: \text{ YaRN attention scale}`] : [])]];
    }
    case 'y': return [`Attention output for ${tokName(t)}`, `${f(m.d)} numbers`, `Each head mixes the value vectors with its row of A; the ${H} results are concatenated${gattn ? ', gated' : ''} and projected by W^O.`,
      [String.raw`o_h = \textstyle\sum_{s} a_{${t},s}\, v_s \quad (${mla ? m.dv : m.dh} \text{ each})`, gattn ? String.raw`Y_{${t}} = \big([\,o_1 \cdots o_{${H}}\,] \odot \sigma(g_{${t}})\big)\, W^{O}` : String.raw`Y_{${t}} = [\,o_1 \cdots o_{${H}}\,]\, W^{O}`, dims(H * (mla ? m.dv : m.dh), m.d)]];
    case 'gt': return [`Output gate for ${tokName(t)}`, `${H} × ${m.dh} numbers`, 'Computed from the token’s own x̂, it scales every channel of every head’s output before W^O, between 0 (closed) and 1 (open).',
      [String.raw`g_{${t}} = \hat{x}_{${t}} W^{G} \in \mathbb{R}^{${H} \times ${m.dh}}`, String.raw`\sigma(g) \in (0, 1)`]];
    case 'gq': case 'gk': case 'gv': {
      const L2 = m.lin, isV = kind === 'gv', name = kind[1];
      return [`${name} for ${tokName(t)}, ${isV ? 'value' : name === 'q' ? 'query' : 'key'} head ${b}`, `${L2.dk} numbers`,
        isV ? `Value head ${b} writes into its own state. It uses query/key head ${Math.ceil(Number(b) / 2)}.` : `Shared by value heads ${2 * Number(b) - 1} and ${2 * Number(b)}.${name === 'k' ? ' After the convolution it is scaled to unit length.' : ' After the convolution it is scaled to unit length and by 1/√128.'}`,
        [String.raw`${name} = \hat{x}\, W^{${name.toUpperCase()}}`, dims(m.d, (isV ? L2.v : L2.qk) * L2.dk), String.raw`\to \operatorname{SiLU}(\operatorname{conv}_4(\cdot))${isV ? '' : String.raw` \to \ell_2\text{-normalize}`}`]];
    }
    case 'ga': return [`Decay α for ${tokName(t)}`, `${m.lin.v} numbers, one per value head`, 'How much of each head’s memory survives this token. Near 1 the state is kept, near 0 it is wiped.',
      [String.raw`\alpha_{${t}} = \exp\big(-e^{A} \operatorname{softplus}(a_{${t}} + \delta)\big)`, String.raw`a_{${t}} = \hat{x}_{${t}} W^{a} \in \mathbb{R}^{${m.lin.v}}`]];
    case 'gb': return [`Write strength β for ${tokName(t)}`, `${m.lin.v} numbers, one per value head`, 'How strongly this token overwrites what the memory stores under its key.',
      [String.raw`\beta_{${t}} = \sigma\big(\hat{x}_{${t}} W^{b}\big) \in (0, 1)^{${m.lin.v}}`]];
    case 'S': return [`State after ${tokName(t)}`, `${m.lin.v} × ${m.lin.dv} × ${m.lin.dk} numbers`, t === 0 ? 'The memory starts empty; the first token writes into it.' : `Built from the state after ${tokName(t - 1)}: decayed, then corrected toward this token’s key → value. Same size for every token.`,
      [t === 0 ? String.raw`S_{0} = \beta_{0}\, v_{0} k_{0}^{\top} \quad (\text{from } 0)` : String.raw`S_{${t}} = \alpha_{${t}} S_{${t - 1}} \big(I - \beta_{${t}} k_{${t}} k_{${t}}^{\top}\big) + \beta_{${t}} v_{${t}} k_{${t}}^{\top}`,
        String.raw`${m.lin.v} \cdot ${m.lin.dv} \cdot ${m.lin.dk} = ${tf(LAB.gdnState(m).heads)} \text{ numbers}`]];
    case 'o': return [`Memory read-out for ${tokName(t)}`, `${m.lin.v} × ${m.lin.dv} numbers`, 'What the memory returns for the token’s own query.',
      [String.raw`o_{${t}} = S_{${t}}\, \hat{q}_{${t}}`, String.raw`[${m.lin.dv} \times ${m.lin.dk}] \cdot [${m.lin.dk}] = [${m.lin.dv}] \text{ per head}`]];
    case 'zg': return [`Output gate z for ${tokName(t)}`, `${m.lin.v} × ${m.lin.dv} numbers`, 'Each head’s read-out is normalized and multiplied by SiLU(z) before W^O merges the heads.',
      [String.raw`z_{${t}} = \hat{x}_{${t}} W^{Z}`, String.raw`y_{${t}} = \big[\operatorname{RMSNorm}(o_{${t},h}) \odot \operatorname{SiLU}(z_{${t},h})\big]_h W^{O}`, siluDef]];
    case 'gcache': {
      const s = LAB.gdnState(m);
      return ['What a GDN layer keeps', 'the last state and 3 convolution inputs', 'For the next token, the layer needs only the newest state S_T and the last three rows that its convolution reads. Nothing grows with the length of the text.',
        [String.raw`${m.lin.v} \cdot ${m.lin.dv} \cdot ${m.lin.dk} = ${tf(s.heads)}`, String.raw`3 \cdot ${tf(s.channels)} = ${tf(s.convState)} \;(\text{conv})`, String.raw`= ${tf(s.total)} \text{ values per layer}`]];
    }
    case 'convlabel': return ['Short causal convolution', `kernel ${m.lin.conv}, one filter per channel`, 'Each channel of q, k and v is mixed with the same channel of the previous three tokens, so local order is visible to the layer.',
      [String.raw`\tilde{q}_t = \operatorname{SiLU}\big(\textstyle\sum_{i=0}^{3} w_i \odot q_{t-i}\big)`, String.raw`${tf(LAB.gdnState(m).channels)} \text{ channels} \times 4 \text{ weights}`]];
    case 'fin': return [`FFN input for ${tokName(t)}`, `${f(m.d)} numbers`, 'The feed-forward layer sees one token at a time; no information moves between rows here.', [String.raw`u_{${t}} = \operatorname{${N}}(h_{${t}})`]];
    case 'ffn': return [`Hidden layer for ${tokName(t)}`, `${f(m.dff)} numbers`, m.ffnKind === 'mlp' ? `Expanded ${m.dff / m.d}× and passed through GELU.` : `Two projections; ${m.act} of the gate multiplies the up-projection elementwise.`,
      m.ffnKind === 'mlp' ? [String.raw`a = \operatorname{GELU}(u\, W_1 + b_1)`, dims(m.d, m.dff), geluDef]
        : [String.raw`g = \operatorname{${m.act}}(u\, W_{\text{gate}}), \quad r = u\, W_{\text{up}}`, dims(m.d, m.dff), String.raw`a = g \odot r`, m.act === 'GELU' ? geluDef : siluDef]];
    case 'moe': {
      const score = m.key === 'dsa' ? String.raw`s = \sigma(u\, W_{\text{router}}) \in \mathbb{R}^{${m.experts}}` : String.raw`s = \operatorname{softmax}(u\, W_{\text{router}}) \in \mathbb{R}^{${m.experts}}`;
      const out = m.key === 'gdn' ? String.raw`f = \sigma(u\, w_{\text{sh}})\, F^{\text{sh}}(u) + \textstyle\sum_{i \in \text{top}} \tilde{s}_i F_i(u)`
        : m.key === 'dsa' ? String.raw`f = F^{\text{sh}}(u) + 2.5 \textstyle\sum_{i \in \text{top}} \tilde{s}_i F_i(u)`
        : String.raw`f = \textstyle\sum F^{\text{sh}}(u) + \textstyle\sum_{i \in \text{top}} s_i F_i(u)`;
      return [`Experts for ${tokName(t)}`, `${m.active} of ${m.experts} routed + ${m.shared} shared`, 'The router picks a different set of experts for each token. The highlighted choice is illustrative.',
        [score, String.raw`\text{top } ${m.active}: \; ${expertsOf(m, t).join(', ')}`, out, String.raw`F_i: ${D} \to ${tf(m.dExpert)} \to ${D} \;(\text{SwiGLU})`]];
    }
    case 'fout': return [`FFN output for ${tokName(t)}`, `${f(m.d)} numbers`, 'Projected back down to the model width and added to the stream.',
      [m.ffnKind === 'mlp' ? String.raw`f = a\, W_2 + b_2` : moeShown(m) ? String.raw`f = \text{shared} + \textstyle\sum s_i F_i(u)` : String.raw`f = a\, W_{\text{down}}`, dims(moeShown(m) ? m.dExpert : m.dff, m.d)]];
    case 'res1': case 'res2': return ['Residual addition', `h ← h + ${kind === 'res1' ? 'Y' : 'f'}`, 'Every token’s row gets the sublayer’s result added to it, element by element.', [String.raw`h_t \leftarrow h_t + ${kind === 'res1' ? 'Y_t' : 'f_t'}`, String.raw`[${D}] + [${D}] = [${D}]`]];
    case 'block': return m.key === 'gdn'
      ? ['Repeated for 48 layers', '12 groups of 3 Gated DeltaNet + 1 gated attention', 'Use the switch above the diagram to see the other layer type. Every layer ends with the same mixture of experts.', [String.raw`48 = 12 \times (3 + 1)`, String.raw`h_{\ell} = \operatorname{Layer}_{\ell}(h_{\ell-1})`]]
      : [`Repeated for ${m.L} blocks`, `${m.L} blocks, own weights each`, `The output of block ℓ is the input of block ℓ+1.${m.key === 'mla' ? ' DeepSeek-V2-Lite: block 1 dense, blocks 2–27 MoE.' : m.key === 'dsa' ? ' DeepSeek-V3.2: blocks 1–3 dense, blocks 4–61 MoE.' : ''}`, [String.raw`h_{\ell} = \operatorname{Block}_{\ell}(h_{\ell-1}), \quad \ell = 1, \dots, ${m.L}`]];
    case 'fnorm': return ['Final norm', `[${f(m.d)}] → [${f(m.d)}]`, 'One last normalization after the final block, with its own weights.', normCalc];
    case 'head': return ['LM head', `[${f(m.d)}] → [${f(m.V)}]`, `${m.tied ? 'Reuses the embedding matrix (tied weights).' : 'A separate output matrix.'} Gives every vocabulary entry a score.`,
      [String.raw`z = \bar{h}\, W_U`, dims(m.d, m.V), m.tied ? String.raw`W_U = E^{\top}: \text{ the same } ${tf(m.V * m.d)} \text{ numbers}` : String.raw`${tf(m.V * m.d)} \text{ numbers of its own}`]];
    case 'logits': return ['Logits $Z$', `${T} rows × ${f(m.V)} scores`, 'The LM head scores every vocabulary entry at every position: row t predicts the token after position t. Training uses every row; generating the next token needs only the last one, z_T.',
      [String.raw`Z = \bar{H}\, W_U: [${T} \times ${D}] \cdot [${D} \times ${tf(m.V)}] = [${T} \times ${tf(m.V)}]`, String.raw`z_T = \text{row } ${T - 1} \text{ of } Z`]];
    case 'zT': return popContent(`zrow:${T - 1}`);
    case 'zrow': return t === T - 1
      ? ['Logits of the last token, $z_T$', `${f(m.V)} scores`, `The scores for the token after ${tokName(t)}, one per vocabulary entry. Softmax turns them into the next-token distribution below.`,
        [String.raw`z_T = \bar{h}_T\, W_U`, dims(m.d, m.V)]]
      : [`Logits at position ${t}`, `${f(m.V)} scores`, `Row ${t} predicts the token after ${tokName(t)}. That token is already known (${tokName(t + 1)}), so generation skips this row; in training it is compared with that token to compute the loss.`,
        [String.raw`z_{${t}} = \bar{h}_{${t}}\, W_U`, String.raw`\text{loss}_{${t}} = -\log \operatorname{softmax}(z_{${t}})_{x_{${t + 1}}}`]];
    case 'tail': {
      const { data, dist } = decodeNow(m);
      const n = (data.source === 'computed' ? m.V : dist.length) - 8;
      return ['The rest of the vocabulary', `${f(n)} more tokens`, `Every token gets a probability, however small; the panel lists the 8 most likely.${data.source === 'illustrative' ? ' The illustrative demo uses a toy vocabulary of 14 candidates.' : ''}`,
        [String.raw`p_v > 0 \text{ for every } v, \qquad \textstyle\sum_{v=1}^{V} p_v = 1`]];
    }
    case 'cand': return candPop(m, t);
    case 'seg': {
      const { dist } = decodeNow(m), c = dist[t], iv = intervals(dist)[t];
      return [`“${pieceText(c.piece)}”: its share of [0, 1)`, `p′ = ${pct(c.p, 1)}`, 'Sampling lays the kept tokens end to end and draws a uniform number u; the token whose interval contains u is chosen.',
        [String.raw`[${iv[0].toFixed(3)},\; ${iv[1].toFixed(3)})`, String.raw`\text{width} = p' = ${tp(c.p)}`]];
    }
    case 'chosen': {
      const { dist, beam, run } = decodeNow(m), c = dist[dec.pick];
      if (dec.method === 'beam') return ['Chosen continuation', `“${pieceText(c.piece)}${pieceText(beam.best.piece)}”`, 'Beam search returns a whole sequence; both tokens are appended.', [String.raw`\log p = ${tn(Math.log(dist[beam.best.from].p0))} + (${tn(beam.best.lp)}) = ${tn(beam.best.score)}`]];
      if (dec.method === 'spec') return ['Chosen token', `“${pieceText(c.piece)}” · ID ${f(c.id)}`, run.acc ? 'The target model accepted the draft token.' : 'The draft token was rejected; a replacement was drawn from the residual r.',
        run.acc ? [String.raw`u = ${run.u2.toFixed(3)} < ${run.a.toFixed(3)} = \min(1, p/q)`] : [String.raw`u = ${run.u2.toFixed(3)} \ge ${run.a.toFixed(3)}`, String.raw`x \sim r, \quad r(x) = ${tp(c.r)}`]];
      const iv = intervals(dist)[dec.pick];
      const calc = dec.method === 'greedy' ? [String.raw`\arg\max_v p_v: \; ${tp(c.p0)}`]
        : dec.u == null ? [txt('starts at the most likely token;'), String.raw`\text{Resample draws } u \sim U(0, 1)`]
        : [String.raw`u = ${dec.u.toFixed(3)} \sim U(0, 1)`, String.raw`u \in [${iv[0].toFixed(3)},\; ${iv[1].toFixed(3)}) \;\Rightarrow\; \text{chosen}`];
      return ['Chosen token', `“${pieceText(c.piece)}” · ID ${f(c.id)}`, 'Appended to the input; the model then predicts the token after it.', calc];
    }
    case 'spec': {
      const { dist, run } = decodeNow(m), d = dist[run.d];
      return ['One speculative step', `draft “${pieceText(d.piece)}” ${run.acc ? 'accepted' : 'rejected'}`, 'The draft model samples a token from q. The target model accepts it with probability min(1, p/q); if not, it samples from what p has in excess of q. The result is distributed exactly as p.',
        [String.raw`\tilde{x} \sim q: \; q(\tilde{x}) = ${tp(d.q)}, \;\; p(\tilde{x}) = ${tp(d.p)}`, String.raw`\min(1, p/q) = ${tn(run.a, 3)}`, String.raw`P(x) = \min(p_x, q_x) + \big(1 - \textstyle\sum_v \min(p_v, q_v)\big)\, r_x = p_x`]];
    }
    case 'beamtree': return ['Beam search, two steps', `width B = ${dec.B}`, 'Step 1 keeps the B most likely first tokens. Step 2 extends each with its most likely continuations, scores every pair by its total log-probability, and keeps the B best. Here only the two best continuations per beam are drawn.',
      [String.raw`\operatorname{score}(y_1, y_2) = \log p(y_1) + \log p(y_2 \mid y_1)`]];
    case 'beam': {
      const { dist, beam } = decodeNow(m), bm = beam.beams[t], c = dist[bm.i];
      return [`Beam ${t + 1}: “${pieceText(c.piece)}”`, `step 1, rank ${t + 1}`, 'One of the B most likely first tokens.', [String.raw`\log p = \log ${tp(c.p0)} = ${tn(bm.lp)}`]];
    }
    case 'bexp': {
      const { dist, beam } = decodeNow(m), e = beam.exps[t], c = dist[e.from];
      return [`“${pieceText(c.piece)}${pieceText(e.piece)}”`, e.kept ? (e === beam.best ? 'best sequence' : 'kept') : 'pruned', `Rank ${e.rank + 1} of all extended pairs; the B = ${dec.B} best survive. ${NEXT_REAL.models && NEXT_REAL.models[m.key] ? '' : 'Second-step probabilities are illustrative.'}`,
        [String.raw`${tn(Math.log(c.p0))} + (${tn(e.lp)}) = ${tn(e.score)}`, String.raw`p(y_2 \mid y_1) = ${tp(Math.exp(e.lp))}`]];
    }
    case 'meth': {
      const { dist } = decodeNow(m), kept = dist.filter(c => c.kept).length, tau = dec.tau.toFixed(1);
      const d = {
        greedy: ['Always the single most likely token. Deterministic, but tends to repeat itself.', [String.raw`x = \arg\max_v p_v`]],
        beam: ['Keeps the B most likely partial sequences, extends each, and returns the most likely whole sequence. Deterministic; common in translation and speech recognition, rare for chat.', [String.raw`\operatorname{score} = \textstyle\sum_i \log p(y_i \mid y_{<i}), \quad B = ${dec.B}`]],
        temperature: ['Draws from the whole distribution; τ below 1 sharpens it, above 1 flattens it. Every token can come out, even very unlikely ones.', [String.raw`x \sim \operatorname{softmax}(z/\tau), \quad \tau = ${tau}`]],
        spec: ['A small draft model proposes tokens and the large model checks them in one forward pass. Accepted drafts save time; the output still follows the large model’s distribution exactly.', [String.raw`\text{accept } \tilde{x} \sim q \text{ w.p. } \min(1, p/q)`, String.raw`\text{else } x \sim r \propto \max(0, p - q)`]],
        topk: ['Keeps the k most likely tokens, renormalizes, samples. k is fixed, so it keeps too many when the model is sure and too few when it is not.', [String.raw`k = ${dec.k}, \quad \tau = ${tau}`, String.raw`p' = p \,/\, \textstyle\sum_{\text{top } k} p`]],
        topp: ['Keeps the smallest set of most likely tokens whose probabilities reach p, renormalizes, samples. The set grows when p is spread out.', [String.raw`p = ${dec.p.toFixed(2)}, \quad \tau = ${tau}`, String.raw`\text{kept here: } ${kept}`]],
        minp: ['Keeps every token at least p_base times as likely as the most likely one. The cut-off follows the model’s confidence, which keeps text coherent at higher temperatures.', [String.raw`p_v \ge ${dec.minp.toFixed(2)} \cdot \max_u p_u`, String.raw`\text{kept here: } ${kept}`]],
        typical: ['Keeps tokens whose surprisal −log p is closest to the entropy H (the average surprisal) until their mass reaches m. Can drop the most likely token when it is unusually predictable.', [String.raw`\big|{-\log p_v} - H\big| \text{ smallest first}, \; \textstyle\sum p \ge ${dec.typ.toFixed(2)}`, String.raw`H = ${tn(dist[0].H)} \text{ nats}`]],
      }[a];
      return [METHOD(a).label, dec.method === a ? 'selected' : 'click to use', d[0], d[1]];
    }
    case 'sample': return dec.method === 'spec'
      ? ['Resample', 'three new random numbers', 'Draws a new draft token from q, a new acceptance test, and, if needed, a new draw from the residual.', [String.raw`u_1, u_2, u_3 \sim U(0, 1)`]]
      : ['Resample', 'u ~ U(0, 1)', 'Draws a new uniform number; the token whose interval contains it is chosen.', [String.raw`[\textstyle\sum_{j<i} p'_j,\; \sum_{j \le i} p'_j)`]];
  }
  return null;
}
/* The full chain for one candidate: logit → probability → the selected method's decision. */
function candPop(m, i) {
  const { data, dist, run } = decodeNow(m), c = dist[i], meth = dec.method;
  const over = c.full ? String.raw`\text{all } ${tf(m.V)} \text{ tokens}` : String.raw`\text{the } ${dist.length} \text{ candidates}`;
  const calc = [String.raw`z = ${tn(c.z)}`];
  if (meth === 'spec') {
    const sp = dist[i];
    calc.push(String.raw`p = ${tp(sp.p)}, \quad q = ${tp(sp.q)} \;(\text{draft})`, String.raw`\min(1, p/q) = ${tn(sp.a, 3)}`,
      String.raw`r = \max(0, p - q) \,/\, ${tn(1 - sp.acc, 3)} = ${tp(sp.r)}`, String.raw`\min(p, q) + ${tn(1 - sp.acc, 3)} \cdot r = ${tp(sp.p)} = p`);
    return [`“${pieceText(c.piece)}” · ID ${LAB.fmt(c.id)}`, `p ${pct(sp.p)} · q ${pct(sp.q)}`, `Speculative sampling outputs this token with probability exactly p. ${data.source === 'illustrative' ? 'Logits and the draft model are illustrative.' : `Target logits computed with ${data.model}; the draft q is illustrative, and p is renormalized over the listed candidates.`}`, calc];
  }
  if (c.tau !== 1) calc.push(String.raw`z / \tau = ${tn(c.z)} / ${c.tau.toFixed(1)} = ${tn(c.zt)}`);
  calc.push(String.raw`p = e^{${tn(c.zt)} - ${tn(c.lse)}} = ${tp(c.p0)}`);
  calc.push(String.raw`${tn(c.lse)} = \log \textstyle\sum_v e^{z_v${c.tau !== 1 ? '/\\tau' : ''}} \text{ over } ${over}`);
  if (meth === 'greedy') calc.push(c.rank === 0 ? String.raw`\text{largest } p \;\Rightarrow\; \text{chosen}` : String.raw`p < p_{\max} = ${tp(c.pmax)} \;\Rightarrow\; \text{never chosen}`);
  else if (meth === 'beam') calc.push(String.raw`\log p = ${tn(Math.log(c.p0))}`, c.rank < dec.B ? String.raw`\text{rank } ${c.rank + 1} \le B = ${dec.B} \;\Rightarrow\; \text{a beam}` : String.raw`\text{rank } ${c.rank + 1} > B = ${dec.B} \;\Rightarrow\; \text{dropped at step 1}`);
  else if (meth === 'temperature') calc.push(String.raw`\text{no filter: } p' = p = ${tp(c.p)}`);
  else if (meth === 'topk') calc.push(c.kept ? String.raw`\text{rank } ${c.rank + 1} \le k = ${Math.min(dec.k, dist.length)} \;\Rightarrow\; \text{kept}` : String.raw`\text{rank } ${c.rank + 1} > k = ${Math.min(dec.k, dist.length)} \;\Rightarrow\; \text{removed}`);
  else if (meth === 'topp') calc.push(String.raw`\text{mass ranked above: } ${tp(c.cum)} ${c.kept ? '<' : '\\ge'} ${dec.p.toFixed(2)} \;\Rightarrow\; \text{${c.kept ? 'kept' : 'removed'}}`);
  else if (meth === 'minp') calc.push(String.raw`p ${c.kept ? '\\ge' : '<'} ${dec.minp.toFixed(2)} \cdot ${tp(c.pmax)} = ${tp(dec.minp * c.pmax)} \;\Rightarrow\; \text{${c.kept ? 'kept' : 'removed'}}`);
  else if (meth === 'typical') calc.push(String.raw`-\log p = ${tn(-Math.log(c.p0))}, \quad H = ${tn(c.H)}`, String.raw`\big|{-\log p} - H\big| = ${tn(c.dev)} \;(\text{${ordinal(c.trank + 1)} closest})`,
    String.raw`\text{mass before it: } ${tp(c.tcum)} ${c.kept ? '<' : '\\ge'} ${dec.typ.toFixed(2)} \;\Rightarrow\; \text{${c.kept ? 'kept' : 'removed'}}`);
  if (c.kept && ['topk', 'topp', 'minp', 'typical'].includes(meth)) calc.push(String.raw`p' = ${tp(c.p0)} \,/\, ${tp(c.mass)} = ${tp(c.p)}`);
  return [`“${pieceText(c.piece)}” · ID ${LAB.fmt(c.id)}`, meth === 'beam' ? `log p = ${c.p0 > 0 ? Math.log(c.p0).toFixed(2) : '−∞'}` : c.kept ? `${pct(c.p, 1)} after decoding` : 'removed by the decoding filter',
    data.source === 'illustrative' ? 'Real token and ID; the logit is an illustrative value, since this page runs no model. Softmax depends only on differences between logits.' : `Logit computed with ${data.model}.`, calc];
}
const ordinal = n => n + (n % 10 === 1 && n % 100 !== 11 ? 'st' : n % 10 === 2 && n % 100 !== 12 ? 'nd' : n % 10 === 3 && n % 100 !== 13 ? 'rd' : 'th');
/* Hover and focus pop-ups for any figure: contentFn(key) → [title, subtitle, text, calculation lines]. */
function attachPopups(fig, box, contentFn) {
  let timer = null;
  const show = el => {
    const c = contentFn(el.dataset.pop); if (!c) return;
    clearTimeout(timer);
    box.innerHTML = `<b>${fmtText(c[0])}</b><span class="pop-s">${fmtText(c[1])}</span><span class="pop-t">${fmtText(c[2])}</span>` +
      (c[3] && c[3].length ? `<span class="pop-c">${c[3].map(l => typeof l === 'string' ? `<span class="pm">${tex(l)}</span>` : `<span class="pt">${esc(l.t)}</span>`).join('')}</span>` : '');
    box.hidden = false;
    const fr = fig.getBoundingClientRect(), r = el.getBoundingClientRect();
    let x = r.left - fr.left + fig.scrollLeft + r.width / 2 - box.offsetWidth / 2;
    let y = r.bottom - fr.top + 10;
    x = Math.max(4, Math.min(x, fig.scrollWidth - box.offsetWidth - 4));
    if (el.closest('#symbol-key')) {
      // The SVG scrolls sideways on mobile; keep legend explanations on screen.
      const left = fig.scrollLeft + 4;
      const right = fig.scrollLeft + fig.clientWidth - box.offsetWidth - 4;
      x = Math.max(left, Math.min(x, Math.max(left, right)));
    }
    if (y + box.offsetHeight > fig.scrollHeight - 4) y = r.top - fr.top - box.offsetHeight - 10;
    box.style.left = `${x}px`; box.style.top = `${Math.max(0, y)}px`;
  };
  const hide = () => { clearTimeout(timer); timer = setTimeout(() => { box.hidden = true; }, 60); };   // one pending hide at a time
  fig.addEventListener('pointerover', e => { const el = e.target.closest('[data-pop]'); if (el) show(el); else hide(); });
  fig.addEventListener('pointerleave', hide);
  fig.addEventListener('focusin', e => { const el = e.target.closest('[data-pop]'); if (el) show(el); });
  fig.addEventListener('focusout', hide);
  fig.addEventListener('click', e => { const el = e.target.closest('[data-pop]'); if (el) requestAnimationFrame(() => { const again = fig.querySelector(`[data-pop="${el.dataset.pop}"]`); if (again) show(again); }); });
}
function initPopups() { attachPopups($('#arch-fig'), $('#pop'), popContent); }
/* Shared with the other pages (mm-app.js). */
window.LABUI = { S, mi, mlab, esc, tex, fmtText, fmtPlain, math, attachPopups };

function applyArchFocus() {
  const svg = $('#arch'), m = LAB.MODELS[state.mech];
  const touring = state.tour >= 0, step = touring ? null : archStepsNow()[state.arch];
  const map = LAB.archHighlight(m, view(m));
  const on = new Set(touring ? LAB.TOUR[state.tour].on : step ? map[step.id] || [] : []);
  svg.classList.toggle('focusing', !!step);
  svg.classList.toggle('touring', touring);
  svg.querySelectorAll('[data-h]').forEach(el => {
    el.classList.toggle('on', el.getAttribute('data-h').split(' ').some(i => on.has(i)));
  });
}

/* ---------- step card ---------- */
function decodeControls(m) {
  const data = LAB.nextTokenData(m.key, NEXT_REAL), n = data.cands.length, M = dec.method;
  const slider = (id, label, min, max, step, val, show, digits) => show
    ? `<label>${label} <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${val}"><output>${Number(val).toFixed(digits)}</output></label>` : '';
  return `<div class="dec">
    <label>Method <select id="dec-method">${LAB.METHODS.map(x => `<option value="${x.key}"${x.key === M ? ' selected' : ''}>${esc(x.label)}</option>`).join('')}</select></label>
    ${slider('dec-tau', 'τ', 0.1, 2, 0.1, dec.tau, LAB.usesTau(M), 1)}
    ${slider('dec-k', 'k', 1, n, 1, Math.min(dec.k, n), M === 'topk', 0)}
    ${slider('dec-p', 'p', 0.05, 1, 0.05, dec.p, M === 'topp', 2)}
    ${slider('dec-minp', 'p<sub>base</sub>', 0.01, 0.5, 0.01, dec.minp, M === 'minp', 2)}
    ${slider('dec-typ', 'm', 0.05, 1, 0.05, dec.typ, M === 'typical', 2)}
    ${slider('dec-B', 'B', 1, 3, 1, dec.B, M === 'beam', 0)}
    ${M === 'greedy' || M === 'beam' ? '' : '<button type="button" id="dec-sample">Resample</button>'}
  </div>`;
}
function wireDecode(m) {
  const upd = () => { drawDist(m); applyArchFocus(); };
  $('#dec-method')?.addEventListener('change', e => setMethod(m, e.target.value));
  const bind = (id, key, digits) => $(id)?.addEventListener('input', e => { dec[key] = Number(e.target.value); e.target.nextElementSibling.textContent = dec[key].toFixed(digits); upd(); });
  bind('#dec-tau', 'tau', 1); bind('#dec-k', 'k', 0); bind('#dec-p', 'p', 2); bind('#dec-minp', 'minp', 2); bind('#dec-typ', 'typ', 2); bind('#dec-B', 'B', 0);
  $('#dec-sample')?.addEventListener('click', () => { resample(m); upd(); });
}
const DECODE_NOTES = {
  greedy: 'Greedy decoding always takes the most likely token: deterministic, and prone to loops in long text.',
  beam: 'Beam search keeps the B most likely partial sequences and extends each one; here it runs two steps. Because it ranks whole sequences, it can prefer a continuation whose first token is not the most likely one.',
  temperature: 'Random sampling draws from all of p. Temperature τ below 1 sharpens the distribution, above 1 flattens it.',
  spec: 'Speculative decoding: a small draft model proposes tokens, the large model scores all of them in one pass and accepts each with probability min(1, p/q). A rejected token is replaced by a draw from the residual, so the output follows p exactly; the speed-up comes from accepting several drafted tokens per large-model pass. Qwen3-Next and Qwen3.8-Flash-Next ship a built-in draft head (multi-token prediction) for this.',
  topk: 'Top-k keeps the k most likely tokens and renormalizes before drawing.',
  topp: 'Top-p (nucleus) keeps the smallest set of most likely tokens whose probabilities reach p, then renormalizes.',
  minp: 'Min-p keeps every token whose probability is at least $p_{\\text{base}}$ times that of the most likely token, then renormalizes.',
  typical: 'Locally typical sampling keeps the tokens whose surprisal −log p is closest to the entropy H of the distribution, until their mass reaches m, then renormalizes.',
};
function viewToggle(m) {
  const tog = $('#block-toggle');
  const opts = m.key === 'gdn' ? [['layer', 'gdn', 'Gated DeltaNet · 36 layers'], ['layer', 'gattn', 'Gated attention · 12 layers']]
    : m.key === 'mla' ? [['block', 'dense', 'Block 1 · dense'], ['block', 'moe', `Blocks 2–${m.L} · experts`]]
    : m.key === 'dsa' ? [['block', 'dense', 'Blocks 1–3 · dense'], ['block', 'moe', `Blocks 4–${m.L} · experts`]] : [];
  tog.hidden = !opts.length;
  tog.setAttribute('aria-label', m.key === 'gdn' ? 'Which Qwen3-Next layer to show' : 'Which block to show');
  tog.innerHTML = opts.map(([k, val, label]) => {
    const on = state[k] === val || (k === 'block' && val === 'moe' && state.block !== 'dense');
    return `<button type="button" data-k="${k}" data-v="${val}" class="${on ? 'active' : ''}" aria-pressed="${on}">${esc(label)}</button>`;
  }).join('');
  tog.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    state[b.dataset.k] = b.dataset.v;
    const n = archStepsNow().length; if (state.arch >= n) state.arch = n - 1;
    drawArch(m); renderArchSteps(m);
  }));
}
function renderArchSteps(m) {
  const v = view(m), steps = LAB.archSteps(m, v), ov = LAB.archOverview(m, v);
  const chips = $('#arch-chips');
  chips.innerHTML = [{ chip: 'Overview' }, ...steps].map((s, idx) => {
    const i = idx - 1, on = i === state.arch;
    return `<li><button type="button" class="chipbtn${on ? ' active' : ''}" data-i="${i}" aria-pressed="${on}">${i < 0 ? '' : `<span class="n">${idx}</span>`}${fmtText(s.chip)}</button></li>`;
  }).join('');
  chips.querySelectorAll('.chipbtn').forEach(b => b.addEventListener('click', () => setArch(Number(b.dataset.i), true)));
  const s = steps[state.arch];
  const card = $('#arch-card');
  if (!s) {
    card.innerHTML = `<div class="card-math">${ov.tex.map(math).join('')}</div>
      <div class="card-text"><h3>The whole model in four lines</h3><p>${fmtText(ov.text)}</p>
      <p class="hint-l">Hover any part of the diagram for details and the calculation behind it; hover a token to follow its row.</p>
      <div class="nav"><span></span><button type="button" class="next">Start: ${esc(steps[0].title)} →</button></div></div>`;
  } else {
    const texs = s.decode ? LAB.DECODE_TEX[dec.method] : s.tex;
    const text = s.decode ? `${DECODE_NOTES[dec.method]} The chosen token is appended and the model runs again; thanks to the cache, only the new token passes through the blocks.` : s.text;
    card.innerHTML = `<div class="card-math">${texs.map(math).join('')}<p class="shape">${fmtText(s.shape)}</p></div>
      <div class="card-text"><h3><span class="n">${state.arch + 1}</span>${esc(s.title)}</h3><p>${fmtText(text)}</p>
      ${s.decode ? decodeControls(m) : ''}
      ${s.more ? '<p><a href="#attention" class="more">Attention up close ↓</a></p>' : ''}
      <div class="nav"><button type="button" class="prev">← ${state.arch === 0 ? 'Overview' : 'Back'}</button>
      ${state.arch < steps.length - 1 ? `<button type="button" class="next">Next: ${esc(steps[state.arch + 1].title)} →</button>` : '<button type="button" class="next">Back to overview</button>'}</div></div>`;
    if (s.decode) wireDecode(m);
  }
  card.querySelector('.prev')?.addEventListener('click', () => setArch(state.arch - 1, true));
  card.querySelector('.next')?.addEventListener('click', () => setArch(state.arch < steps.length - 1 ? state.arch + 1 : -1, true));
  card.querySelector('.more')?.addEventListener('click', e => { e.preventDefault(); document.getElementById('attention').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  viewToggle(m);
}
function setArch(i, focus) {
  if (state.tour >= 0) return;          // the tour drives the diagram until it ends
  const n = archStepsNow().length;
  state.arch = Math.max(-1, Math.min(n - 1, i));
  state.keys = 'arch';
  renderArchSteps(LAB.MODELS[state.mech]);
  applyArchFocus();
  if (focus) document.querySelector(`#arch-chips .chipbtn[data-i="${state.arch}"]`)?.focus({ preventScroll: true });
}

/* ---------- side panel ---------- */
function renderPanel(m) {
  const ov = LAB.overview(m, state.sparse), steps = LAB.steps(m, state.sparse);
  const list = $('#steps');
  list.replaceChildren();
  const items = [{ title: 'Overview' }, ...steps];
  items.forEach((s, idx) => {
    const i = idx - 1, active = i === state.step;
    const li = document.createElement('li');
    li.className = 'step' + (active ? ' active' : '') + (i < 0 ? ' ov' : '');
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'step-h';
    b.id = `step-${idx}`;
    b.setAttribute('aria-expanded', String(active));
    b.setAttribute('aria-controls', `step-body-${idx}`);
    b.innerHTML = `<span class="n">${i < 0 ? '' : idx}</span><span class="t">${esc(s.title)}</span>`;
    b.addEventListener('click', () => setStep(i, true));
    li.appendChild(b);
    if (active) {
      const body = document.createElement('div');
      body.className = 'step-b';
      body.id = `step-body-${idx}`;
      if (i < 0) {
        body.innerHTML = ov.tex.map(math).join('') +
          `<p>${fmtText(ov.text)}</p>` +
          `<div class="ctx"><span class="ctx-l">Where it sits</span>${math(ov.block)}<p>${esc(ov.blockNote)}</p></div>` +
          `<div class="nav"><span></span><button type="button" class="next">Walk through the ${steps.length} steps →</button></div>`;
      } else {
        body.innerHTML = s.tex.map(math).join('') +
          `<p class="shape">${fmtText(s.shape)}</p>` +
          `<p>${fmtText(s.text)}</p>` +
          `<div class="nav"><button type="button" class="prev">← ${i === 0 ? 'Overview' : 'Back'}</button>` +
          (i < steps.length - 1 ? `<button type="button" class="next">Next: ${esc(steps[i + 1].title)} →</button>` : `<button type="button" class="next">Back to overview</button>`) + '</div>';
      }
      body.querySelector('.prev')?.addEventListener('click', () => setStep(i - 1, true));
      body.querySelector('.next')?.addEventListener('click', () => setStep(i < steps.length - 1 ? i + 1 : -1, true));
      li.appendChild(body);
    }
    list.appendChild(li);
  });
  const tog = $('#sparse-toggle');
  tog.hidden = m.key !== 'dsa';
  tog.querySelectorAll('button').forEach(b => {
    const on = b.dataset.v === state.sparse;
    b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on));
  });
  $('#attn-sub').textContent = m.key === 'gdn' ? 'Every head of one Gated DeltaNet layer, and the memory update inside one of them.'
    : m.key === 'dsa' ? (state.sparse === 'qsa' ? 'Qwen Sparse Attention (Qwen3.8-Flash-Next): the indexer picks blocks of four tokens, then attention reads only those.' : 'The lightning indexer of DeepSeek-V3.2 picks tokens, then attention reads only those.')
    : 'Every head of the attention sublayer, and the computation inside one of them.';
}

/* ---------- header, stats, compare ---------- */
function renderHeader(m) {
  document.querySelectorAll('#tabs .tab').forEach(t => {
    const stamp = LAB.ARCH_YEARS[t.dataset.mech];
    const year = t.querySelector('.tab-year');
    if (year && stamp) { year.textContent = stamp.year; year.title = stamp.note; }
    const on = t.dataset.mech === m.key;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
  });
  $('#title').textContent = m.name;
  $('#lead').textContent = LAB.overview(m, 'dsa').lead;
  $('#stats').innerHTML = LAB.stats(m).map(([k, v], i) =>
    `<div><dt>${esc(k)}</dt><dd>${i === 0 ? `<a href="${m.config}" target="_blank" rel="noopener noreferrer">${esc(v)}</a>` : esc(v).replace(/θ/g, () => tex(String.raw`\theta`))}</dd></div>`).join('');
  document.title = `${m.model} · ${m.abbr} · LLM Architecture Explorer`;
}
function renderCompare() {
  const rows = LAB.compareRows(), max = Math.max(...rows.map(r => r.value));
  $('#compare-rows').innerHTML = rows.map(r => `
    <button type="button" class="crow${r.key === state.mech ? ' active' : ''}" data-mech="${r.key}" aria-label="${r.label}: ${LAB.fmt(r.value)} values per token${r.star ? ', plus a fixed state (see note)' : ''}. Show ${r.label}.">
      <span class="c-l">${r.label}</span><span class="c-f">${r.formula.split(' ').map(t => /^[A-Za-z]$/.test(t) ? `<var>${mi(t)}</var>` : fmtPlain(t)).join(' ')}</span>
      <span class="c-bar">${r.value ? `<i style="width:${(100 * r.value / max).toFixed(2)}%"></i>` : ''}</span>
      <span class="c-v">${LAB.fmt(r.value)}${r.star ? '<span class="star" aria-hidden="true">*</span>' : ''}</span>
    </button>`).join('');
  document.querySelectorAll('.crow').forEach(b => b.addEventListener('click', () => setMech(b.dataset.mech)));
}


/* ---------- guided tour: one idea per step on the GPT-2 diagram ---------- */
function renderSymbolKey() {
  const byStep = { 1: ['emb', 'pos'], 2: ['xhat', 'q', 'k', 'v'], 3: ['q', 'k', 'v', 'rope', 'att'], 4: ['y'], 5: ['logit'], 6: ['k', 'v'] };
  const active = byStep[state.tour] || [];
  const model = LAB.MODELS[state.mech], layer = view(model);
  const showP = model.pos === 'learned';
  const showRoPE = model.pos === 'decoupled' || (model.pos === 'rope' && (model.key !== 'gdn' || layer.layer === 'gattn'));
  const showA = model.key !== 'gdn' || layer.layer === 'gattn';
  document.querySelectorAll('#symbol-key [data-symbol]').forEach(el => {
    const name = el.dataset.symbol;
    el.hidden = (name === 'pos' && !showP) || (name === 'rope' && !showRoPE) || (name === 'att' && !showA);
    const target = el.querySelector('[data-tex]');
    if (target) {
      const formula = name === 'att' ? `A_{${state.head}}` : target.dataset.tex;
      target.innerHTML = tex(formula);
    }
    el.classList.toggle('active', !el.hidden && active.includes(name));
  });
}
function renderTour() {
  renderSymbolKey();
  const box = $('#tour'), sec = box.closest('.whole'), N = LAB.TOUR.length;
  sec.classList.toggle('touring', state.tour >= 0);
  if (state.tour < 0) {
    box.innerHTML = `<div class="tour-invite"><span>New to transformers?</span>
      <button type="button" id="tour-start">Take the ${N}-step tour →</button>
      <span class="tour-m">one idea per step, from text to the next token, then six ways to make attention cheaper</span></div>`;
    $('#tour-start').addEventListener('click', startTour);
    return;
  }
  const s = LAB.TOUR[state.tour], i = state.tour;
  let extra = '';
  if (s.extra === 'answers') {
    const rows = LAB.compareRows(), max = Math.max(...rows.map(r => r.value));
    extra = `<div class="tour-ans">${LAB.ORDER.map(k => {
      const m = LAB.MODELS[k], r = rows.find(x => x.key === k);
      return `<button type="button" data-mech="${k}"><b>${esc(m.name)}</b><span class="tm">${esc(LAB.ARCH_YEARS[k].year)} · ${esc(m.abbr === 'DSA' ? 'Sparse' : m.abbr)} · ${esc(m.model)}</span>
        <span class="td">${esc(LAB.TOUR_ANSWERS[k])}</span>
        <span class="bar">${r.value ? `<i style="width:${(100 * r.value / max).toFixed(1)}%"></i>` : ''}</span>
        <span class="tv">${r.value ? `${LAB.fmt(r.value)} values per token` : 'nothing per token · fixed state'}</span></button>`;
    }).join('')}</div>`;
  }
  box.innerHTML = `<div class="tour-card">
    <div class="tour-top"><span>Tour · step ${i + 1} of ${N}</span>
      <span class="tour-dots">${LAB.TOUR.map((t, j) => `<button type="button" class="${j === i ? 'on' : ''}" data-i="${j}" aria-label="Step ${j + 1}: ${esc(t.title)}"></button>`).join('')}</span>
      <button type="button" class="tour-x">Exit tour ✕</button></div>
    <h3>${esc(s.title)}</h3><p>${fmtText(s.text)}</p>
    ${s.term ? `<p class="tour-term"><b>${fmtText(s.term[0])}</b> · ${fmtText(s.term[1])}</p>` : ''}${extra}
    <div class="nav"><button type="button" class="prev"${i === 0 ? ' disabled' : ''}>← Back</button>
      ${i < N - 1 ? `<button type="button" class="next">Next: ${esc(LAB.TOUR[i + 1].title)} →</button>` : '<button type="button" class="next">Explore on your own</button>'}</div></div>`;
  box.querySelector('.prev').addEventListener('click', () => setTour(i - 1));
  box.querySelector('.next').addEventListener('click', () => i < N - 1 ? setTour(i + 1) : endTour());
  box.querySelector('.tour-x').addEventListener('click', endTour);
  box.querySelectorAll('.tour-dots button').forEach(b => b.addEventListener('click', () => setTour(Number(b.dataset.i))));
  box.querySelectorAll('.tour-ans button').forEach(b => b.addEventListener('click', () => {
    endTour(); setMech(b.dataset.mech);
    document.querySelector('.intro').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));
}
function startTour() {
  if (state.mech !== 'mha') setMech('mha');
  dec.method = 'topk'; dec.tau = 0.8; dec.k = 5; dec.pick = null; dec.u = null;
  state.arch = -1; state.row = null;
  drawArch(LAB.MODELS[state.mech]); renderArchSteps(LAB.MODELS[state.mech]);
  setTour(0);
  $('#tour').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function setTour(i) {
  state.tour = Math.max(0, Math.min(LAB.TOUR.length - 1, i));
  state.keys = 'tour';
  renderTour(); applyArchFocus();
  $('#tour .next')?.focus({ preventScroll: true });
}
function endTour() {
  state.tour = -1; state.keys = 'arch';
  renderTour(); applyArchFocus();
}

/* ---------- state changes ---------- */
function setMech(k, fromHash) {
  if (!LAB.MODELS[k]) return;
  const changed = k !== state.mech;
  state.mech = k;
  if (changed) {
    state.tour = -1;
    state.head = 1; state.row = null; dec.pick = null; dec.u = null;
    const n = LAB.steps(LAB.MODELS[k], state.sparse).length; state.step = Math.min(state.step, n - 1);
    const na = LAB.archSteps(LAB.MODELS[k], view(LAB.MODELS[k])).length; state.arch = Math.min(state.arch, na - 1);
  }
  if (!fromHash) { try { history.replaceState(null, '', '#' + k); } catch (e) { /* sandboxed */ } }
  render();
}
function setExample(key) {
  if (key === LAB.TOKENS.key) return;
  LAB.setExample(key);
  state.row = null; dec.pick = null; dec.u = null; dec.su = [0.30, 0.42, 0.55];
  hoverRow = null;
  render();
  $('#arch .expill.on-x')?.focus({ preventScroll: true });
}
function setStep(i, focus) {
  const n = LAB.steps(LAB.MODELS[state.mech], state.sparse).length;
  state.step = Math.max(-1, Math.min(n - 1, i));
  state.keys = 'attn';
  renderPanel(LAB.MODELS[state.mech]);
  applyFocus();
  if (focus) document.getElementById(`step-${state.step + 1}`)?.focus({ preventScroll: true });
}
function selectHead(i) {
  state.head = i;
  state.keys = 'attn';
  drawDiagram(LAB.MODELS[state.mech]);
  drawArch(LAB.MODELS[state.mech]);
  const el = document.querySelector('#diagram .head.sel');
  el?.focus({ preventScroll: true });
}
function render() {
  const m = LAB.MODELS[state.mech];
  renderTour();
  renderHeader(m);
  drawArch(m);
  renderArchSteps(m);
  drawDiagram(m);
  renderPanel(m);
  renderCompare();
}

/* ---------- wiring ---------- */
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setMech(t.dataset.mech)));
$('#tabs').addEventListener('keydown', e => {
  const i = LAB.ORDER.indexOf(state.mech);
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    const k = LAB.ORDER[(i + (e.key === 'ArrowRight' ? 1 : LAB.ORDER.length - 1)) % LAB.ORDER.length];
    setMech(k);
    document.querySelector(`.tab[data-mech="${k}"]`).focus();
  }
});
document.addEventListener('keydown', e => {
  if (state.page !== 'architectures') return;
  if (e.target.closest && e.target.closest('#tabs, input, select, textarea')) return;
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const inSteps = !!e.target.closest('#steps');
  if (state.tour >= 0 && !inSteps) {
    if (e.key === 'ArrowRight') { e.preventDefault(); setTour(state.tour + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setTour(state.tour - 1); }
    else if (e.key === 'Escape') endTour();
    return;
  }
  const arch = state.keys === 'arch' && !inSteps;
  const move = d => arch ? setArch(state.arch + d, true) : setStep(state.step + d, true);
  if (e.key === 'ArrowRight' || (inSteps && e.key === 'ArrowDown')) { e.preventDefault(); move(1); }
  else if (e.key === 'ArrowLeft' || (inSteps && e.key === 'ArrowUp')) { e.preventDefault(); move(-1); }
  else if (e.key === 'Escape') { if (arch) setArch(-1, false); else setStep(-1, false); }
  else if (new RegExp(`^[1-${LAB.ORDER.length}]$`).test(e.key)) setMech(LAB.ORDER[Number(e.key) - 1]);
});
document.querySelectorAll('#sparse-toggle button').forEach(b => b.addEventListener('click', () => {
  state.sparse = b.dataset.v;
  const m = LAB.MODELS[state.mech];
  state.head = Math.min(state.head, state.sparse === 'qsa' ? LAB.QSA.H : m.H);
  drawDiagram(m); renderPanel(m);
}));
/* ---------- pages: #architectures (or a mechanism, #mla …), #multimodal (#multimodal/qwen …), #retrieval, #reasoning, #agents ---------- */
const PAGES = ['architectures', 'multimodal', 'retrieval', 'reasoning', 'agents'];
let archRendered = false;
function showPage(p, sub) {
  if (!PAGES.includes(p)) p = 'architectures';
  const changed = p !== state.page;
  state.page = p;
  document.querySelectorAll('.page').forEach(el => { el.hidden = el.dataset.page !== p; });
  document.querySelectorAll('.pages a').forEach(a => { if (a.dataset.page === p) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  const name = document.querySelector(`.pages a[data-page="${p}"]`)?.firstChild.textContent.trim();
  document.title = `${name} · LLM Internals Lab`;
  if (p === 'architectures' && !archRendered) { archRendered = true; render(); }   // text is measured, so draw while visible
  if (p === 'multimodal' && window.MMAPP) window.MMAPP.show(sub);
  if (p === 'retrieval' && window.RETRIEVALAPP) window.RETRIEVALAPP.show();
  if (p === 'reasoning' && window.REASONINGAPP) window.REASONINGAPP.show();
  if (changed) window.scrollTo(0, 0);
}
function route() {
  const [h, ...rest] = location.hash.slice(1).split('/');
  if (LAB.MODELS[h]) { showPage('architectures'); setMech(h, true); }
  else showPage(h, rest.join('/'));
}
document.querySelectorAll('a[data-page]').forEach(a => a.addEventListener('click', e => {
  e.preventDefault();
  const p = a.dataset.page;
  const hash = p === 'architectures' ? state.mech : p === 'multimodal' && window.MMAPP ? `multimodal/${window.MMAPP.key()}` : p;
  try { history.pushState(null, '', '#' + hash); } catch (err) { /* sandboxed */ }
  showPage(p);
}));
window.addEventListener('hashchange', route);
window.addEventListener('popstate', route);
$('#sources').innerHTML = LAB.SOURCES.slice().sort((a, b) => a.by.localeCompare(b.by, 'en') || a.year - b.year || a.title.localeCompare(b.title, 'en')).map(r => `<li>${esc(r.by)} (${r.year}). <a href="${r.url}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>. <i>${esc(r.in)}</i>.</li>`).join('');
$('#configs').innerHTML = LAB.ORDER.map(k => { const m = LAB.MODELS[k]; return `<a href="${m.config}" target="_blank" rel="noopener noreferrer">${esc(m.model)}</a>`; }).join(', ');

initPopups();
const initial = location.hash.slice(1);
if (LAB.MODELS[initial]) state.mech = initial;
route();
})();
