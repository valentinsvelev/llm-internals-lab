/* Multimodal page — rendering and interaction. Uses mm-content.js (window.MM) and the helpers
 * app.js shares (window.LABUI: S, esc, tex, fmtText, math, attachPopups). Draws on first show. */
(function () {
'use strict';
const MM = window.MM, fmt = MM.fmt;
const $ = s => document.querySelector(s);
let U, S, esc, fmtText, math;
const st = { key: 'llava', sel: 2, custom: null, step: -1, ready: false };

const hw = () => (st.sel === 'custom' ? [st.custom.h, st.custom.w] : MM.SIZES[st.sel]);
const model = () => MM.VLMS[st.key];
const plan = () => MM.imagePlan(st.key, ...hw());
const steps = () => MM.steps(st.key, ...hw());
const stepIndex = id => steps().findIndex(s => s.id === id);

/* ---------- the example image: a fixed little scene (an image keeps its colors in dark mode) ---------- */
const RGB = { skyTop: [134, 182, 228], skyLow: [204, 227, 245], sun: [247, 197, 72], far: [128, 172, 112], near: [76, 132, 72],
  wall: [238, 228, 208], roof: [186, 80, 58], door: [98, 72, 54] };
const rgb = c => `rgb(${c.map(v => Math.round(v)).join(',')})`;
function geom(W, H) {
  const s = Math.min(W, H);
  const far = u => H * (0.60 - 0.09 * Math.sin(Math.PI * (0.15 + 0.85 * u)));
  const near = u => H * (0.82 - 0.07 * Math.sin(Math.PI * (1.3 * u - 0.1)));
  const hx = 0.13 * W, hwid = 0.24 * s;
  return { s, far, near, sun: [0.76 * W, 0.27 * H, 0.11 * s],
    house: { x: hx, w: hwid, h: 0.17 * s, b: near((hx + hwid / 2) / W) + 0.05 * s, roof: 0.11 * s, eave: 0.025 * s, door: [0.055 * s, 0.10 * s] } };
}
function sceneRGB(x, y, W, H, g) {
  const hs = g.house, u = x / W;
  if (x >= hs.x && x <= hs.x + hs.w && y <= hs.b && y >= hs.b - hs.h) {
    const dx = hs.x + hs.w * 0.62 - hs.door[0] / 2;
    return x >= dx && x <= dx + hs.door[0] && y >= hs.b - hs.door[1] ? RGB.door : RGB.wall;
  }
  const ry = hs.b - hs.h, rx0 = hs.x - hs.eave, rx1 = hs.x + hs.w + hs.eave, apex = ry - hs.roof;
  if (y <= ry && y >= apex && Math.abs(x - (rx0 + rx1) / 2) <= (rx1 - rx0) / 2 * (y - apex) / (ry - apex)) return RGB.roof;
  if (y >= g.near(u)) return RGB.near;
  if (y >= g.far(u)) return RGB.far;
  const [sx, sy, sr] = g.sun;
  if ((x - sx) ** 2 + (y - sy) ** 2 <= sr * sr) return RGB.sun;
  const t = Math.min(1, y / (0.6 * H));
  return RGB.skyTop.map((c, i) => c + (RGB.skyLow[i] - c) * t);
}
/* Draw the image's region [vx, vy, vw, vh] (original pixels) into the box x, y, w, h, stretched to fit. */
function scene(p, x, y, w, h, [vx, vy, vw, vh]) {
  const v = S('svg', { x, y, width: w, height: h, viewBox: `${vx} ${vy} ${vw} ${vh}`, preserveAspectRatio: 'none', 'aria-hidden': 'true' }, p);
  const [H, W] = hw();
  if (st.sel === 'custom') { S('image', { href: st.custom.url, x: 0, y: 0, width: W, height: H, preserveAspectRatio: 'none' }, v); return v; }
  const g = geom(W, H), hill = f => {
    const pts = []; for (let i = 0; i <= 40; i++) pts.push(`${(W * i / 40).toFixed(1)},${f(i / 40).toFixed(1)}`);
    return `0,${H} ${pts.join(' ')} ${W},${H}`;
  };
  S('rect', { x: 0, y: 0, width: W, height: H, fill: 'url(#mm-sky)' }, v);
  S('circle', { cx: g.sun[0], cy: g.sun[1], r: g.sun[2], fill: rgb(RGB.sun) }, v);
  S('polygon', { points: hill(g.far), fill: rgb(RGB.far) }, v);
  S('polygon', { points: hill(g.near), fill: rgb(RGB.near) }, v);
  const hs = g.house, ry = hs.b - hs.h;
  S('polygon', { points: `${hs.x - hs.eave},${ry} ${hs.x + hs.w / 2},${ry - hs.roof} ${hs.x + hs.w + hs.eave},${ry}`, fill: rgb(RGB.roof) }, v);
  S('rect', { x: hs.x, y: ry, width: hs.w, height: hs.h, fill: rgb(RGB.wall) }, v);
  S('rect', { x: hs.x + hs.w * 0.62 - hs.door[0] / 2, y: hs.b - hs.door[1], width: hs.door[0], height: hs.door[1], fill: rgb(RGB.door) }, v);
  return v;
}
/* Average colour of each cell of an nx × ny grid over the region (original pixels), for the mosaics. */
function sample([x0, y0, x1, y1], nx, ny) {
  const [H, W] = hw(), out = [];
  if (st.sel === 'custom') {
    const c = document.createElement('canvas'); c.width = nx; c.height = ny;
    const ctx = c.getContext('2d'); ctx.drawImage(st.custom.img, x0, y0, x1 - x0, y1 - y0, 0, 0, nx, ny);
    const d = ctx.getImageData(0, 0, nx, ny).data;
    for (let r = 0; r < ny; r++) { const row = []; for (let q = 0; q < nx; q++) { const i = 4 * (r * nx + q); row.push([d[i], d[i + 1], d[i + 2]]); } out.push(row); }
    return out;
  }
  const g = geom(W, H), cw = (x1 - x0) / nx, ch = (y1 - y0) / ny;
  for (let r = 0; r < ny; r++) {
    const row = [];
    for (let q = 0; q < nx; q++) {
      const acc = [0, 0, 0];
      for (const [a, b] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) sceneRGB(x0 + (q + a) * cw, y0 + (r + b) * ch, W, H, g).forEach((v, i) => { acc[i] += v / 4; });
      row.push(acc);
    }
    out.push(row);
  }
  return out;
}

/* ---------- svg helpers ---------- */
function defs(svg) {
  const d = S('defs', {}, svg);
  const mk = (id, cls) => { const m = S('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, d); S('path', { d: 'M1 1 L9 5 L1 9', class: cls }, m); };
  mk('mm-arrow', 'arrowhead'); mk('mm-arrow-lat', 'arrowhead lat');
  const gr = S('linearGradient', { id: 'mm-sky', x1: 0, y1: 0, x2: 0, y2: 1 }, d);
  S('stop', { offset: 0, 'stop-color': rgb(RGB.skyTop) }, gr); S('stop', { offset: 0.6, 'stop-color': rgb(RGB.skyLow) }, gr);
  const pat = S('pattern', { id: 'mm-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, d);
  S('rect', { width: 6, height: 6, class: 'hatch-bg p' }, pat); S('line', { x1: 0, y1: 0, x2: 0, y2: 6, class: 'hatch-line p' }, pat);
}
const T = (p, x, y, text, cls, anchor = 'start', h) => S('text', { x, y, class: cls, 'text-anchor': anchor, 'data-h': h, text }, p);
const line = (p, pts, cls = 'arr', h, marker = 'mm-arrow') => S('path', { d: 'M' + pts.map(q => q.join(' ')).join(' L'), class: cls, fill: 'none', 'marker-end': marker ? `url(#${marker})` : null, 'data-h': h }, p);
function act(el, stepId, pop, label) {
  el.setAttribute('tabindex', '0'); el.setAttribute('role', 'button');
  if (label) el.setAttribute('aria-label', label);
  el.classList.add('hit');
  if (pop) el.dataset.pop = pop;
  const go = () => { const i = stepIndex(stepId); if (i >= 0) setStep(i, false); };
  el.addEventListener('click', go);
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  return el;
}
const short = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

/* ---------- the whole model ---------- */
const Y0 = 200;                     // the line every column is centred on
const COL = { a: 20, aw: 200, b: 262, bw: 270, c: 576, cw: 150, d: 770, dw: 116, e: 930, ew: 160, l: 1112, lw: 186, out: 1312 };
function rowsLayout(m, p) {
  let y = 64; const rows = [];
  for (const [label, kind] of m.template) {
    const hgt = kind === 'img' ? Math.max(44, Math.min(140, p.tokens / 6)) : 20;
    rows.push({ label, kind, y, h: hgt, capped: kind === 'img' && p.tokens / 6 > 140 });
    y += hgt + 6;
  }
  return { rows, top: 64, bottom: y - 6 };
}
function layerX(i, n) { const pitch = (COL.lw - 12) / n; return { x: COL.l + 6 + i * pitch, pitch }; }

function draw() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = model(), k = st.key, p = plan(), [h, w] = hw(), e = m.enc, dcd = m.dec;
  svg.setAttribute('aria-label', `${m.name}: a ${w} × ${h} image goes through ${e.name} and a ${m.conn.name} into ${dcd.name}.`);
  // column headers
  [[COL.a, 'Image', `${w} × ${h} pixels`], [COL.b, 'What the encoder sees', { llava: 'centre crop, resized', qwen: 'own size, rounded', llama: 'fitted into tiles', gemma: 'stretched to a square' }[k]],
    [COL.c, 'Vision encoder', e.name], [COL.d, 'Connector', m.conn.name], [COL.e, 'Language model', `${dcd.name.replace(' + 8 cross-attention layers', '')} · ${dcd.layers} layers`]]
    .forEach(([x, a, b]) => { T(svg, x, 30, a, 'st'); T(svg, x, 47, b, 'st-s'); });

  /* A: the image */
  const sa = Math.min(COL.aw / w, COL.aw / h), iw = w * sa, ih = h * sa, ix = COL.a + (COL.aw - iw) / 2, iy = Y0 - ih / 2;
  const gA = act(S('g', { 'data-h': 'img' }, svg), 'image', 'img', 'Input image');
  scene(gA, ix, iy, iw, ih, [0, 0, w, h]);
  S('rect', { x: ix, y: iy, width: iw, height: ih, class: 'frame' }, gA);
  if (k === 'llava' && w !== h) {
    const side = Math.min(w, h) * sa, cx = ix + (iw - side) / 2, cy = iy + (ih - side) / 2;
    const gc = act(S('g', { 'data-h': 'crop' }, svg), 'resize', 'crop', 'Centre crop');
    if (w > h) { S('rect', { x: ix, y: iy, width: cx - ix, height: ih, class: 'shade' }, gc); S('rect', { x: cx + side, y: iy, width: ix + iw - cx - side, height: ih, class: 'shade' }, gc); }
    else { S('rect', { x: ix, y: iy, width: iw, height: cy - iy, class: 'shade' }, gc); S('rect', { x: ix, y: cy + side, width: iw, height: iy + ih - cy - side, class: 'shade' }, gc); }
    S('rect', { x: cx, y: cy, width: side, height: side, class: 'crop' }, gc);
  }
  T(svg, COL.a + COL.aw / 2, Y0 + COL.aw / 2 + 22, `${fmt(h * w * 3)} numbers`, 'dim', 'middle', 'img');
  T(svg, COL.a + COL.aw / 2, Y0 + COL.aw / 2 + 38, `red, green, blue per pixel`, 'note', 'middle', 'img');
  line(svg, [[COL.a + COL.aw + 8, Y0], [COL.b - 8, Y0]], 'arr', 'resized');

  /* B: what the encoder sees */
  const [ch, cw] = p.canvas, sb = Math.min(COL.bw / cw, COL.bw / ch), bw = cw * sb, bh = ch * sb, bx = COL.b + (COL.bw - bw) / 2, by = Y0 - bh / 2;
  const gB = act(S('g', { 'data-h': 'resized' }, svg), 'resize', 'resized', 'Resized image');
  if (k === 'llava') { const side = Math.min(w, h); scene(gB, bx, by, bw, bh, [(w - side) / 2, (h - side) / 2, side, side]); }
  else if (k === 'llama') { S('rect', { x: bx, y: by, width: bw, height: bh, class: 'pad' }, gB); scene(gB, bx, by, p.resized[1] * sb, p.resized[0] * sb, [0, 0, w, h]); }
  else scene(gB, bx, by, bw, bh, [0, 0, w, h]);
  S('rect', { x: bx, y: by, width: bw, height: bh, class: 'frame' }, gB);
  const P = e.patch, pitch = P * sb, nx = cw / P, ny = ch / P;
  const gG = S('g', { 'data-h': 'grid', class: 'nohit' }, svg);
  const gridPath = (every, cls) => {
    let d = '';
    for (let i = every; i < nx; i += every) d += `M${(bx + i * pitch).toFixed(2)} ${by}V${(by + bh).toFixed(2)}`;
    for (let j = every; j < ny; j += every) d += `M${bx} ${(by + j * pitch).toFixed(2)}H${(bx + bw).toFixed(2)}`;
    if (d) S('path', { d, class: cls }, gG);
  };
  if (pitch >= 2.6) gridPath(1, 'pgrid');
  if (k === 'qwen') gridPath(2, 'pgrid2');
  if (k === 'gemma') gridPath(4, 'pgrid2');
  if (k === 'llama') gridPath(40, 'tileb');
  const capB = {
    llava: [`336 × 336 · 24 × 24 patches of 14`, w === h ? 'already square' : `the ${w > h ? 'sides are' : 'top and bottom are'} cut off`],
    qwen: [`${p.resized[1]} × ${p.resized[0]} · ${p.patches[1]} × ${p.patches[0]} patches of 16`, `thick lines: the 2 × 2 blocks that merge into one token`],
    llama: [`${cw} × ${ch} canvas · ${p.tiles} tile${p.tiles > 1 ? 's' : ''} of 560 · patches of 14`, `image ${p.resized[1]} × ${p.resized[0]}, the rest padding`],
    gemma: [`896 × 896 · 64 × 64 patches of 14`, w === h ? 'already square' : st.sel === 'custom' ? 'stretched to a square' : 'stretched: the round sun turns oval'],
  }[k];
  T(svg, COL.b + COL.bw / 2, Y0 + COL.bw / 2 + 22, capB[0], 'dim', 'middle', 'grid');
  T(svg, COL.b + COL.bw / 2, Y0 + COL.bw / 2 + 38, capB[1], 'note', 'middle', 'grid');
  if (k === 'llama') {           // the encoder always gets four tiles
    const gt = S('g', { 'data-h': 'grid enc-in' }, svg), tx0 = COL.b + COL.bw / 2 - 64;
    T(gt, tx0 - 8, Y0 + COL.bw / 2 + 64, 'into the encoder:', 'note', 'end');
    for (let i = 0; i < 4; i++) {
      const used = i < p.tiles, g1 = act(S('g', {}, gt), 'patches', `tile:${i}`, `Tile ${i + 1}${used ? '' : ', empty'}`);
      S('rect', { x: tx0 + i * 34, y: Y0 + COL.bw / 2 + 50, width: 26, height: 18, rx: 2, class: used ? 'tile on' : 'tile' }, g1);
      T(g1, tx0 + i * 34 + 13, Y0 + COL.bw / 2 + 63, used ? String(i + 1) : '—', 'tiny', 'middle');
    }
  }
  line(svg, [[COL.b + COL.bw + 8, Y0], [COL.c - 8, Y0]], 'arr', 'enc-in');

  /* C: the vision encoder, one column per layer */
  const top = Y0 - 135, bot = Y0 + 135;
  const gC = act(S('g', { 'data-h': 'enc' }, svg), 'encode', 'enc', `Vision encoder, ${e.layers} layers`);
  S('rect', { x: COL.c, y: top, width: COL.cw, height: 270, rx: 8, class: 'subbox' }, gC);
  T(gC, COL.c + COL.cw / 2, top + 18, e.global ? `${e.layers} + ${e.global} layers` : `${e.layers} layers`, 'st-s', 'middle');
  T(gC, COL.c + COL.cw / 2, top + 33, `d ${fmt(e.d)} · ${e.heads} heads`, 'dim', 'middle');
  const NE = e.layers + (e.global || 0), ex0 = COL.c + 16, ew = COL.cw - 32, epitch = ew / (NE + (e.global ? 1 : 0));
  const elX = i => ex0 + (i + (e.global && i >= e.layers ? 1 : 0) + 0.5) * epitch;
  for (let i = 0; i < NE; i++) {
    const off = k === 'llava' && i === NE - 1, gl = e.global && i >= e.layers;
    const g1 = S('g', { class: 'lay', 'data-pop': `elayer:${i}` }, gC);
    S('rect', { x: elX(i) - epitch / 2, y: top + 44, width: epitch, height: 186, class: 'lhit' }, g1);
    S('rect', { x: elX(i) - Math.max(1.6, epitch * 0.28), y: top + 50, width: Math.max(3.2, epitch * 0.56), height: 174, rx: 1, class: 'elayer' + (gl ? ' g' : '') + (off ? ' off' : '') }, g1);
  }
  e.taps.forEach((t, j) => {
    const ds = k === 'qwen' && j < 3;
    S('circle', { cx: elX(t), cy: ds ? top + 230 : top + 43, r: 3.2, class: 'tap' + (ds ? ' ds' : ''), 'data-h': ds ? 'enc dec' : 'enc' }, svg);
  });
  if (e.global) T(svg, elX(e.layers + 3.5), top + 246, 'global', 'tiny', 'middle', 'enc');
  if (k === 'llava') T(svg, elX(NE - 1) + 4, top + 246, 'unused', 'tiny', 'end', 'enc');
  T(svg, COL.c + COL.cw / 2, bot + 22, `in: ${k === 'llama' ? '4 × 1,601' : fmt(p.encoderPositions)} × ${fmt(e.d)}`, 'dim', 'middle', 'enc-in');
  T(svg, COL.c + COL.cw / 2, bot + 38, `no causal mask · ${(e.params / 1e6).toFixed(0)}M parameters`, 'note', 'middle', 'enc');
  line(svg, [[COL.c + COL.cw + 8, Y0], [COL.d - 8, Y0]], 'arr', 'conn');

  /* D: the connector */
  const gD = act(S('g', { 'data-h': 'conn' }, svg), 'connect', 'conn', `Connector: ${m.conn.name}`);
  const dTop = Y0 - 50, dx = COL.d, dw = COL.dw, cxm = dx + dw / 2;
  S('rect', { x: dx, y: dTop, width: dw, height: 100, rx: 8, class: 'subbox' }, gD);
  connectorIcon(gD, k, cxm, dTop);
  T(svg, cxm, dTop + 100 + 20, `out: ${{ llava: '576 × 4,096', qwen: `${fmt(p.tokens)} × 4,096`, llama: '6,404 × 4,096', gemma: '256 × 2,560' }[k]}`, 'dim', 'middle', 'conn');
  T(svg, cxm, dTop + 100 + 36, `${(m.conn.params / 1e6).toFixed(1)}M parameters`, 'note', 'middle', 'conn');

  /* E: the language model: one row per piece of the prompt, one column per layer */
  const L = rowsLayout(m, p), N = dcd.layers, boxTop = L.top - 8, boxBot = L.bottom + 8;
  const gE = S('g', { 'data-h': 'dec' }, svg);
  S('rect', { x: COL.l, y: boxTop, width: COL.lw, height: boxBot - boxTop, rx: 8, class: 'subbox' }, gE);
  L.rows.forEach((r, i) => S('rect', { x: COL.l + 2, y: r.y, width: COL.lw - 4, height: r.h, class: r.kind === 'img' ? 'imgband' : 'tband' }, gE));
  for (let i = 0; i < N; i++) {
    const { x, pitch: lp } = layerX(i, N);
    const cross = dcd.cross.includes(i), glob = dcd.globalEvery && (i + 1) % dcd.globalEvery === 0, dsl = k === 'qwen' && i < 3;
    const g1 = act(S('g', { class: 'lay', 'data-h': cross ? 'dec cross' : 'dec' }, gE), 'decode', `dlayer:${i}`, `Decoder layer ${i + 1}`);
    S('rect', { x, y: boxTop + 2, width: lp, height: boxBot - boxTop - 4, class: 'lhit' }, g1);
    S('rect', { x: x + lp * 0.22, y: boxTop + 4, width: Math.max(1.5, lp * 0.56), height: boxBot - boxTop - 8, rx: 1, class: 'dlayer' + (cross ? ' cross' : '') + (glob ? ' glob' : '') + (dsl ? ' dsl' : '') }, g1);
  }
  const gS = S('g', { 'data-h': 'seq' }, svg);
  L.rows.forEach((r, i) => {
    const g1 = act(S('g', {}, gS), 'sequence', r.kind === 'img' ? 'imgtok' : `tok:${i}`, r.label);
    if (r.kind === 'img') {
      S('rect', { x: COL.e, y: r.y, width: COL.ew, height: r.h, rx: 4, class: 'imgblock' }, g1);
      T(g1, COL.e + 8, r.y + 15, m.imageToken, 'tok', 'start');
      T(g1, COL.e + 8, r.y + 31, `× ${fmt(p.tokens)}`, 'tok', 'start');
      if (r.h >= 60) T(g1, COL.e + 8, r.y + 47, { llava: 'positions in a row', qwen: '3D positions: t, row, column', gemma: 'see each other both ways' }[k], 'tiny');
      if (r.capped) T(g1, COL.e + COL.ew - 8, r.y + r.h - 7, 'not to scale', 'tiny', 'end');
    } else {
      S('rect', { x: COL.e, y: r.y, width: COL.ew, height: r.h, rx: 4, class: 'chip' + (r.kind === 'mark' ? ' special' : '') }, g1);
      T(g1, COL.e + 8, r.y + 14, short(r.label, 25), 'tok');
    }
  });
  const imgRow = L.rows.find(r => r.kind === 'img') || L.rows.find(r => r.kind === 'mark');
  // connector → the image tokens, or (Llama) → the image memory beside the text
  if (k === 'llama') {
    const my = Math.max(Y0 - 24, L.bottom + 18), gM = act(S('g', { 'data-h': 'seq conn cost' }, svg), 'sequence', 'mem', 'Image memory');
    S('rect', { x: COL.e, y: my, width: COL.ew, height: 48, rx: 6, class: 'mem' }, gM);
    T(gM, COL.e + 8, my + 18, 'image memory', 'st-s');
    T(gM, COL.e + 8, my + 35, '6,404 × 4,096, beside the text', 'tiny');
    line(svg, [[COL.d + COL.dw + 8, Y0], [COL.e - 22, Y0], [COL.e - 22, my + 24], [COL.e - 6, my + 24]], 'arr', 'conn');
    const by2 = my + 24, gF = S('g', { 'data-h': 'dec cross' }, svg), xs = dcd.cross.map(i => layerX(i, N).x + layerX(i, N).pitch / 2);
    line(gF, [[COL.e + COL.ew + 2, by2], [xs[xs.length - 1], by2]], 'feed', null, null);
    xs.forEach(x => line(gF, [[x, by2], [x, boxBot + 3]], 'feed', null, 'mm-arrow-lat'));
    T(svg, COL.l + COL.lw, by2 + 18, '8 cross-attention layers read it', 'note', 'end', 'dec');
  } else {
    const yc = imgRow.y + Math.min(imgRow.h / 2, 26);
    line(svg, yc === Y0 ? [[COL.d + COL.dw + 8, Y0], [COL.e - 6, Y0]] : [[COL.d + COL.dw + 8, Y0], [COL.e - 22, Y0], [COL.e - 22, yc], [COL.e - 6, yc]], 'arr', 'conn');
  }
  let lowest = Math.max(L.bottom + 8, k === 'llama' ? Math.max(Y0 - 24, L.bottom + 18) + 66 : 0);
  if (k === 'qwen') {             // DeepStack: three more mergers, added into decoder layers 1–3
    const gDS = S('g', { 'data-h': 'dec' }, svg);
    [0, 1, 2].forEach(j => {
      const tx = elX(e.taps[j]), exitY = top + 259 - 12 * j, mY = Y0 + 144 - 22 * j, elbow = COL.c + COL.cw + 12 + 8 * j;
      const under = boxBot + 12 + 8 * j, turn = COL.e - 14 - 8 * j, lx = layerX(j, N), lxc = lx.x + lx.pitch / 2;
      const g1 = act(S('g', {}, gDS), 'decode', `merger:${j}`, `DeepStack merger for encoder layer ${e.taps[j] + 1}`);
      line(g1, [[tx, top + 233], [tx, exitY], [elbow, exitY], [elbow, mY], [dx - 2, mY]], 'ds', null, null);
      S('rect', { x: dx, y: mY - 9, width: dw, height: 18, rx: 4, class: 'chip special' }, g1);
      T(g1, cxm, mY + 4, `merger · layer ${e.taps[j] + 1}`, 'tiny', 'middle');
      line(g1, [[dx + dw, mY], [turn, mY], [turn, under], [lxc, under], [lxc, imgRow.y + imgRow.h + 2]], 'ds', null, 'mm-arrow-lat');
    });
    const ly2 = boxBot + 12 + 16 + 22;
    T(svg, COL.l, ly2, 'DeepStack: encoder layers 9, 17 and 25, merged,', 'note', 'start', 'dec');
    T(svg, COL.l, ly2 + 15, 'added to the image tokens after layers 1–3', 'note', 'start', 'dec');
    lowest = ly2 + 15;
  }
  if (k === 'gemma') {
    T(svg, COL.l, boxBot + 20, 'dark: global (every 6th); light: sliding window 1,024', 'note', 'start', 'dec');
    lowest = boxBot + 20;
  }
  // the next token
  const last = L.rows[L.rows.length - 1], ly = last.y + last.h / 2, gO = act(S('g', { 'data-h': 'dec' }, svg), 'decode', 'next', 'Next token');
  line(gO, [[COL.l + COL.lw + 4, ly], [COL.out + 4, ly]]);
  S('rect', { x: COL.out + 8, y: ly - 11, width: 72, height: 22, rx: 11, class: 'pillbox' }, gO);
  T(gO, COL.out + 44, ly + 4, 'next token', 'pill-l', 'middle');

  /* what the image costs */
  const gK = act(S('g', { 'data-h': 'cost' }, svg), 'cost', 'cost', 'Cost of the image');
  const cost = MM.cacheValues(k, h, w), cy0 = Math.max(Y0 + 157, lowest + 34);
  T(gK, COL.e, cy0, 'The image in the KV cache', 'st-s');
  T(gK, COL.e, cy0 + 26, MM.fmtMiB(cost), 'cost-v');
  T(gK, COL.e + 118, cy0 + 26 - 5, k === 'llama' ? `6,404 positions × 8 cross-attention layers, + 1 token` : `${fmt(p.tokens)} tokens × ${dcd.layers} layers · ${(100 * p.tokens / dcd.ctx).toFixed(p.tokens / dcd.ctx < 0.01 ? 2 : 0)}% of the ${fmt(dcd.ctx)}-token context`, 'note');
  const bottom = Math.max(cy0 + 22, k === 'llama' ? Y0 + 135 + 72 : Y0 + 135 + 44);
  svg.setAttribute('viewBox', `0 0 1400 ${Math.ceil(bottom + 14)}`);
  applyFocus();
}
function connectorIcon(g, k, cx, top) {
  if (k === 'llava') {
    [[-36, 18, 'W₁'], [0, 34, 'GELU'], [36, 34, 'W₂']].forEach(([dx, hgt, lab]) => {
      if (lab === 'GELU') { T(g, cx + dx, top + 46, lab, 'tiny', 'middle'); return; }
      S('rect', { x: cx + dx - 9, y: top + 42 - hgt / 2, width: 18, height: hgt, rx: 2, class: 'vec lat' }, g);
      T(g, cx + dx, top + 74, lab, 'tiny', 'middle');
    });
    T(g, cx, top + 90, '1,024 → 4,096 → 4,096', 'tiny', 'middle');
  } else if (k === 'qwen') {
    for (let i = 0; i < 4; i++) S('rect', { x: cx - 42 + (i % 2) * 15, y: top + 22 + Math.floor(i / 2) * 15, width: 13, height: 13, class: 'vec' }, g);
    line(g, [[cx - 8, top + 36], [cx + 14, top + 36]], 'arr');
    S('rect', { x: cx + 20, y: top + 29, width: 14, height: 14, class: 'vec lat' }, g);
    T(g, cx, top + 72, '2 × 2 patches → 1', 'tiny', 'middle');
    T(g, cx, top + 88, 'MLP 4,608 → 4,096', 'tiny', 'middle');
  } else if (k === 'llama') {
    for (let i = 0; i < 6; i++) S('rect', { x: cx - 46 + i * 7, y: top + 20, width: 5, height: 32, class: 'vec' }, g);
    line(g, [[cx - 2, top + 36], [cx + 18, top + 36]], 'arr');
    S('rect', { x: cx + 24, y: top + 20, width: 12, height: 32, class: 'vec lat' }, g);
    T(g, cx, top + 72, '6 layers side by side', 'tiny', 'middle');
    T(g, cx, top + 88, 'linear 7,680 → 4,096', 'tiny', 'middle');
  } else {
    for (let i = 0; i < 16; i++) S('rect', { x: cx - 46 + (i % 4) * 8, y: top + 18 + Math.floor(i / 4) * 8, width: 7, height: 7, class: 'vec' }, g);
    line(g, [[cx - 8, top + 34], [cx + 14, top + 34]], 'arr');
    S('rect', { x: cx + 20, y: top + 25, width: 16, height: 16, class: 'vec lat' }, g);
    T(g, cx, top + 72, 'average 4 × 4 → 1', 'tiny', 'middle');
    T(g, cx, top + 88, 'linear 1,152 → 2,560', 'tiny', 'middle');
  }
}

/* ---------- pop-ups ---------- */
function popContent(key) {
  const m = model(), k = st.key, p = plan(), [h, w] = hw(), e = m.enc, dcd = m.dec, [kind, a] = key.split(':'), i = Number(a);
  const tf = n => fmt(n).replace(/,/g, '{,}');
  switch (kind) {
    case 'img': return ['Input image', `${w} × ${h} pixels`, st.sel === 'custom' ? 'Your image. It stays in this browser tab; nothing is uploaded.' : 'An example image. Load your own with “Your image…”; it stays in this browser tab.',
      [String.raw`${tf(h)} \times ${tf(w)} \times 3 = ${tf(h * w * 3)}`]];
    case 'crop': { const side = Math.min(w, h); return ['Centre crop', `${side} × ${side} of ${w} × ${h}`,
      `CLIP's processor scales the short side to 336 and keeps the centre square: ${(100 * (1 - side / Math.max(w, h))).toFixed(0)}% of the long side is never seen. (LLaVA-1.5's training padded images to a square instead; the Hugging Face processor crops.)`, []]; }
    case 'resized': return ['What the encoder sees', k === 'llama' ? `${p.canvas[1]} × ${p.canvas[0]} canvas` : `${p.resized[1]} × ${p.resized[0]} pixels`, steps()[1].text, []];
    case 'tile': return [`Tile ${i + 1}`, i < p.tiles ? 'holds part of the image' : 'empty', i < p.tiles ? '560 × 560 pixels: 40 × 40 patches and a class token, 1,601 vectors.' : 'Padding only. The encoder still runs on it and the decoder caches it, but the cross-attention mask hides it.', []];
    case 'enc': return [`Vision encoder: ${e.name}`, `${e.layers}${e.global ? ` + ${e.global}` : ''} layers · d ${fmt(e.d)} · ${e.heads} heads · FFN ${fmt(e.ff)}`,
      `A transformer like the decoder, but every patch attends to every other (no causal mask). Positions: ${e.positions}. Output used: ${e.tapNote}.`,
      [String.raw`\text{${(e.params / 1e6).toFixed(1)}M parameters}`]];
    case 'elayer': {
      const gl = e.global && i >= e.layers, n = gl ? i - e.layers + 1 : i + 1, tap = e.taps.includes(i);
      const note = k === 'llava' && i === e.layers - 1 ? 'Computed but not used: LLaVA reads the layer before it, whose features are less specialised for CLIP’s caption matching.'
        : k === 'qwen' && tap && i !== e.layers - 1 ? `DeepStack reads this layer: its 2 × 2 blocks go through a merger of their own and are added to the image tokens after decoder layer ${e.taps.indexOf(i) + 1}.`
        : k === 'llama' && tap ? (gl ? 'The last global layer: its output, with five local layers, forms the 7,680-dim vector.' : `One of the five local layers whose output is kept and set beside the final one.`)
        : tap ? 'The connector reads this layer’s output.' : gl ? 'Global layer: sees all four tiles at once, with tanh gates on its outputs.' : 'Self-attention over all patches, then a feed-forward network.';
      return [gl ? `Global layer ${n}` : `Encoder layer ${n}`, `of ${e.layers}${e.global ? ` + ${e.global} global` : ''}`, note, []];
    }
    case 'conn': return [`Connector: ${m.conn.name}`, `${(m.conn.params / 1e6).toFixed(1)}M parameters`, steps()[4].text, steps()[4].tex];
    case 'merger': return [`DeepStack merger ${i + 1}`, `encoder layer ${e.taps[i] + 1} → decoder layer ${i + 1}`,
      'Same shape as the main merger (2 × 2 → 4,608 → 4,096). Its output is added to the hidden states of the image tokens only, right after decoder layer ' + (i + 1) + '.', []];
    case 'imgtok': return [k === 'llava' ? 'Image tokens' : 'Image tokens', `${fmt(p.tokens)} × ${m.imageToken}`, steps()[5].text, []];
    case 'tok': { const r = m.template[i]; return [r[0], r[1] === 'mark' ? 'special token' : 'text', r[1] === 'mark' ? (k === 'llama' ? 'Marks where the image is: text after it may read the image through cross-attention.' : 'Marks where an image starts or ends.') : 'Ordinary text tokens (shortened here). They go through the same layers as the image tokens.', []]; }
    case 'mem': return ['Image memory', '4 tiles × 1,601 × 4,096', 'Projected once per image, outside the text. Each cross-attention layer turns it into its own keys and values (8 heads of 128), so the cache holds 6,404 positions per cross-attention layer.', [String.raw`6404 \times 2 \cdot 8 \cdot 128 \times 8 = ${tf(6404 * 2 * 8 * 128 * 8)}`]];
    case 'dlayer': {
      const cross = dcd.cross.includes(i), glob = dcd.globalEvery && (i + 1) % dcd.globalEvery === 0;
      const t = cross ? 'Cross-attention: queries from the text, keys and values from the image memory; output scaled by tanh(g). No self-attention in this layer.'
        : k === 'gemma' ? (glob ? 'Global attention over the whole context.' : 'Sliding-window attention: each token sees the last 1,024 tokens (image tokens also see the rest of their image).')
        : k === 'qwen' && i < 3 ? `Self-attention, then DeepStack adds the layer-${e.taps[i] + 1} encoder features to the image tokens.`
        : k === 'llama' ? 'Self-attention over the text only, as in Llama 3.1 8B; frozen while the image layers were trained.' : 'Causal self-attention over text and image tokens, then a feed-forward network.';
      return [`Layer ${i + 1}${cross ? ' · cross-attention' : glob ? ' · global' : k === 'gemma' ? ' · sliding' : ''}`, `of ${dcd.layers}`, t, []];
    }
    case 'next': return ['Next token', 'text out', 'The model only ever outputs text. Like every token before it, the answer is decoded one token at a time; the image is read, never generated.', []];
    case 'cost': return ['The image in the KV cache', MM.fmtMiB(MM.cacheValues(k, h, w)), steps()[7].text, steps()[7].tex];
  }
  return null;
}

/* ---------- up close ---------- */
function drawClose() {
  const svg = $('#mmc'); svg.replaceChildren(); defs(svg);
  const k = st.key, c = MM.CLOSE[k];
  $('#mm-close-title').textContent = c.title;
  $('#mm-close-sub').textContent = c.sub;
  $('#mm-close-text').textContent = c.text;
  ({ llava: closeLlava, qwen: closeQwen, llama: closeLlama, gemma: closeGemma })[k](svg);
  svg.setAttribute('viewBox', `0 0 1400 ${{ llava: 262, qwen: 274, llama: 276, gemma: 292 }[k]}`);
}
function mosaic(p, x, y, cell, colors) {
  const g = S('g', { class: 'mosaic' }, p);
  colors.forEach((row, r) => row.forEach((c, q) => S('rect', { x: x + q * cell, y: y + r * cell, width: cell + 0.6, height: cell + 0.6, fill: rgb(c) }, g)));
  S('rect', { x, y, width: colors[0].length * cell, height: colors.length * cell, class: 'frame' }, p);
  return g;
}
function vec(p, x, y, wd, dims, lab, cls = 'vec') {
  S('rect', { x, y, width: wd, height: 16, rx: 2, class: cls }, p);
  T(p, x + wd / 2, y + 34, dims, 'dim', 'middle');
  if (lab) T(p, x + wd / 2, y - 8, lab, 'note', 'middle');
}
function closeLlava(svg) {
  const [h, w] = hw(), side = Math.min(w, h), x0 = (w - side) / 2, y0 = (h - side) / 2, cell = side / 24;
  let pr = 12, pc = 12;
  if (st.sel !== 'custom') {
    const g = geom(w, h), px = g.sun[0] - 0.7 * g.sun[2], py = g.sun[1] + 0.7 * g.sun[2];
    if (px > x0 && px < x0 + side && py > y0 && py < y0 + side) { pc = Math.floor((px - x0) / cell); pr = Math.floor((py - y0) / cell); }
  }
  const px0 = x0 + pc * cell, py0 = y0 + pr * cell, cols = sample([px0, py0, px0 + cell, py0 + cell], 14, 14);
  T(svg, 40, 40, `patch (${pr + 1}, ${pc + 1}) of 24 × 24`, 'st-s');
  mosaic(svg, 40, 56, 8, cols);
  T(svg, 96, 188, '14 × 14 pixels × 3', 'dim', 'middle');
  const y = 112;
  line(svg, [[166, y + 8], [208, y + 8]]); T(svg, 187, y - 4, 'flatten', 'tiny', 'middle');
  vec(svg, 214, y, 70, '588', 'patch');
  line(svg, [[290, y + 8], [340, y + 8]]); T(svg, 315, y - 4, '× W + pᵢ', 'tiny', 'middle');
  vec(svg, 346, y, 90, '1,024', 'embedded');
  line(svg, [[442, y + 8], [478, y + 8]]);
  S('rect', { x: 484, y: y - 22, width: 150, height: 60, rx: 8, class: 'subbox' }, svg);
  T(svg, 559, y + 2, 'CLIP layers 1–23', 'st-s', 'middle'); T(svg, 559, y + 20, 'every patch sees all 577', 'tiny', 'middle');
  line(svg, [[640, y + 8], [676, y + 8]]);
  vec(svg, 682, y, 90, '1,024', 'layer 23 output');
  line(svg, [[778, y + 8], [814, y + 8]]);
  S('rect', { x: 820, y: y - 22, width: 170, height: 60, rx: 8, class: 'subbox' }, svg);
  T(svg, 905, y + 2, 'MLP projector', 'st-s', 'middle'); T(svg, 905, y + 20, 'W₁ · GELU · W₂', 'tiny', 'middle');
  line(svg, [[996, y + 8], [1032, y + 8]]);
  vec(svg, 1038, y, 180, '4,096', 'token embedding', 'vec lat');
  T(svg, 1128, y + 60, `one of the 576 image tokens Vicuna reads`, 'note', 'middle');
  // the class token
  const cy = 236;
  S('rect', { x: 346, y: cy, width: 90, height: 16, rx: 2, class: 'vec off' }, svg);
  T(svg, 346 + 45, cy + 12, '[CLS]', 'tiny', 'middle');
  T(svg, 446, cy + 12, 'the class token, CLIP’s summary of the whole image, is dropped after layer 23', 'note');
}
function closeQwen(svg) {
  const seg = [['text', 3], ['image', [1, 6, 10]], ['text', 4]], pos = MM.mropePositions(seg);
  const labels = ['user', '\\n', '<vs>', ...Array.from({ length: 15 }, (_, j) => `${Math.floor(j / 5) + 1},${(j % 5) + 1}`), '<ve>', 'What', 'is', 'it?'];
  const x0 = 140, cw = 50, y0 = 56, rh = 26, n = labels.length;
  T(svg, x0 - 14, y0 + 14, 'token', 'st-s', 'end');
  [['time', 'ax-t'], ['row', 'ax-h'], ['column', 'ax-w']].forEach(([lab, cls], r) => {
    T(svg, x0 - 14, y0 + (r + 1) * rh + 18, lab, 'st-s', 'end');
    S('rect', { x: x0 - 8, y: y0 + (r + 1) * rh + 4, width: 4, height: rh - 8, class: cls }, svg);
  });
  for (let j = 0; j < n; j++) {
    const img = j >= 3 && j < 18, x = x0 + j * cw;
    S('rect', { x: x + 2, y: y0, width: cw - 4, height: rh - 4, rx: 4, class: img ? 'imgblock' : 'chip' }, svg);
    T(svg, x + cw / 2, y0 + 15, labels[j], img ? 'tiny' : 'tok', 'middle');
    for (let r = 0; r < 3; r++) {
      const v = pos[r][j], changed = r > 0 && v !== pos[0][j];
      T(svg, x + cw / 2, y0 + (r + 1) * rh + 18, String(v), 'mnum' + (changed ? ' ' + ['', 'ax-h-t', 'ax-w-t'][r] : ''), 'middle');
    }
  }
  const ix0 = x0 + 3 * cw, ix1 = x0 + 18 * cw;
  S('path', { d: `M${ix0 + 2} ${y0 - 8} V${y0 - 14} H${ix1 - 2} V${y0 - 8}`, class: 'rope-br' }, svg);
  T(svg, (ix0 + ix1) / 2, y0 - 20, 'image: 6 × 10 patches → 3 × 5 tokens, labelled row,column', 'note', 'middle');
  const jx = x0 + 18 * cw + cw / 2;
  T(svg, jx, y0 + 4 * rh + 20, 'continues at 3 + max(3, 5) = 8,', 'note', 'middle');
  T(svg, jx, y0 + 4 * rh + 35, 'not 3 + 15 = 18', 'note', 'middle');
  // the 64 frequency pairs of one 128-dim head
  const axes = MM.mropePairAxes(), fy = 230, fx = 140, fw = 16;
  T(svg, fx - 14, fy + 12, 'pairs', 'st-s', 'end');
  [...axes].forEach((a, j) => S('rect', { x: fx + j * fw, y: fy, width: fw - 2, height: 16, rx: 2, class: { T: 'ax-t', H: 'ax-h', W: 'ax-w' }[a] }, svg));
  T(svg, fx, fy + 34, 'pair 1: fastest rotation', 'tiny');
  T(svg, fx + 64 * fw - 2, fy + 34, 'pair 64: slowest', 'tiny', 'end');
  T(svg, fx + 32 * fw, fy + 34, `the 64 rotary frequency pairs of a 128-dim head: ${[...axes].filter(c => c === 'T').length} time, ${[...axes].filter(c => c === 'H').length} row, ${[...axes].filter(c => c === 'W').length} column, interleaved`, 'tiny', 'middle');
}
function closeLlama(svg) {
  const p = plan(), rows = ['<|begin_of_text|>', '…user<|end_header_id|>', '<|image|>', 'Describe', 'the', 'image', '.'];
  const img = 2, x0 = 230, y0 = 54, cw = 14, rh = 22, per = 10;
  rows.forEach((r, i) => T(svg, x0 - 12, y0 + i * rh + 15, r, 'tok', 'end'));
  for (let t = 0; t < 4; t++) {
    const used = t < p.tiles, tx = x0 + t * (per * cw + 10);
    T(svg, tx + per * cw / 2, y0 - 10, used ? `tile ${t + 1}` : `tile ${t + 1} · empty`, 'tiny', 'middle');
    for (let i = 0; i < rows.length; i++) for (let q = 0; q < per; q++) {
      const on = i >= img && used;
      S('rect', { x: tx + q * cw + 1, y: y0 + i * rh + 2, width: cw - 2, height: rh - 4, rx: 1.5, class: on ? 'mcell on' : 'mcell' }, svg);
    }
  }
  const right = x0 + 4 * (per * cw + 10) + 10;
  T(svg, right, y0 + 0.9 * rh + 6, '← no image yet: these rows read nothing,', 'note');
  T(svg, right, y0 + 0.9 * rh + 21, '   and the layer’s MLP output is set to 0 for them', 'note');
  T(svg, right, y0 + 4.4 * rh, `← from <|image|> on: every token reads the ${p.tiles} filled tile${p.tiles > 1 ? 's' : ''}`, 'note');
  T(svg, x0, y0 + rows.length * rh + 18, 'each block stands for one tile’s 1,601 positions (40 × 40 patches + class token)', 'tiny');
  // the gates
  const gy = 250;
  S('rect', { x: x0, y: gy - 22, width: 560, height: 36, rx: 8, class: 'subbox' }, svg);
  T(svg, x0 + 16, gy + 1, 'h ← h + tanh(g₁) · CrossAttn(h, image)', 'mlt');
  T(svg, x0 + 310, gy + 1, 'h ← h + tanh(g₂) · MLP(h)', 'mlt');
  T(svg, x0 + 580, gy + 1, 'g₁, g₂ start at 0, and tanh(0) = 0: before training, the layer passes text through unchanged', 'note');
}
function closeGemma(svg) {
  const [h, w] = hw(), patches = sample([0, 0, w, h], 64, 64);
  const pooled = Array.from({ length: 16 }, (_, r) => Array.from({ length: 16 }, (_, q) => {
    const acc = [0, 0, 0];
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) patches[4 * r + a][4 * q + b].forEach((v, i) => { acc[i] += v / 16; });
    return acc;
  }));
  T(svg, 40, 36, '64 × 64 patches', 'st-s');
  mosaic(svg, 40, 48, 3.25, patches);
  line(svg, [[262, 152], [318, 152]]); T(svg, 290, 140, 'average 4 × 4', 'tiny', 'middle');
  T(svg, 330, 36, '16 × 16 = 256 tokens', 'st-s');
  mosaic(svg, 330, 48, 13, pooled);
  T(svg, 330 + 104, 280, 'colours stand in for the vectors being averaged', 'tiny', 'middle');
  // attention mask for a short sequence, with 4 image tokens standing in for 256
  const toks = ['user', '\\n\\n', '<boi>', 'img', 'img', 'img', 'img', '<eoi>', '\\n\\n', 'What?'], isImg = toks.map(t => t === 'img');
  const mask = MM.gemmaMask(isImg), mx = 760, my = 56, c = 19;
  T(svg, mx, 36, 'attention mask (4 image tokens drawn for 256)', 'st-s');
  toks.forEach((t, i) => { T(svg, mx - 8, my + i * c + 13, t, 'tiny', 'end'); T(svg, mx + i * c + c / 2, my + toks.length * c + 14, String(i + 1), 'tiny', 'middle'); });
  mask.forEach((row, i) => row.forEach((on, j) => S('rect', { x: mx + j * c + 1, y: my + i * c + 1, width: c - 2, height: c - 2, rx: 2, class: on ? (j > i ? 'mcell bi' : 'mcell on') : 'mcell' }, svg)));
  const lx = mx + toks.length * c + 24;
  S('rect', { x: lx, y: my + 2 * c, width: 12, height: 12, rx: 2, class: 'mcell on key' }, svg); T(svg, lx + 20, my + 2 * c + 10, 'causal: earlier tokens', 'note');
  S('rect', { x: lx, y: my + 3.5 * c, width: 12, height: 12, rx: 2, class: 'mcell bi key' }, svg); T(svg, lx + 20, my + 3.5 * c + 10, 'later tokens of the same image', 'note');
  S('rect', { x: lx, y: my + 5 * c, width: 12, height: 12, rx: 2, class: 'mcell key' }, svg); T(svg, lx + 20, my + 5 * c + 10, 'not visible', 'note');
}

/* ---------- cost across the four models ---------- */
function renderCost() {
  const [h, w] = hw(), vals = MM.ORDER.map(k => MM.cacheValues(k, h, w)), max = Math.max(...vals);
  $('#mm-cost-sub').textContent = `For this ${w} × ${h} image, at 16 bits. Keys and values for every image token in every layer; for Llama 3.2, for every image position in its eight cross-attention layers.`;
  $('#mm-cost-rows').innerHTML = MM.ORDER.map((k, i) => {
    const m = MM.VLMS[k], p = MM.imagePlan(k, h, w);
    return `<button type="button" class="crow mm${k === st.key ? ' active' : ''}" data-k="${k}" aria-label="${esc(m.name)}: ${MM.fmtMiB(vals[i])}. Show ${esc(m.name)}.">
      <span class="c-l">${esc(m.name)}</span><span class="c-f">${k === 'llama' ? '1 token in the text · 6,404 beside it' : `${fmt(p.tokens)} tokens in the text`}</span>
      <span class="c-bar"><i style="width:${(100 * vals[i] / max).toFixed(2)}%"></i></span><span class="c-v">${MM.fmtMiB(vals[i])}</span></button>`;
  }).join('');
  document.querySelectorAll('#mm-cost-rows .crow').forEach(b => b.addEventListener('click', () => setModel(b.dataset.k)));
}

/* ---------- header, steps, controls ---------- */
function renderHeader() {
  const m = model(), p = plan(), [h, w] = hw();
  document.querySelectorAll('#mm-tabs .tab').forEach(t => { const on = t.dataset.k === st.key; t.classList.toggle('active', on); t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; });
  $('#mm-title').textContent = { llava: 'A projector into the token stream', qwen: 'Native resolution, 3D positions', llama: 'Gated cross-attention', gemma: 'A fixed budget of 256 tokens' }[st.key];
  $('#mm-lead').textContent = MM.LEAD[st.key];
  const stats = [['Model', `<a href="https://huggingface.co/${m.hf}" target="_blank" rel="noopener noreferrer">${esc(m.name)}</a>`],
    ['Parameters', `${(m.total / 1e9).toFixed(2)}B`], ['Vision encoder', `${(m.enc.params / 1e6).toFixed(0)}M`], ['Connector', `${(m.conn.params / 1e6).toFixed(1)}M`],
    ['Language model', `${(m.dec.params / 1e9).toFixed(2)}B`],
    ['Image tokens here', st.key === 'llama' ? '1 (+ 6,404 beside)' : fmt(p.tokens)], ['Context', `${fmt(m.dec.ctx)} tokens`]];
  $('#mm-stats').innerHTML = stats.map(([a, b]) => `<div><dt>${esc(a)}</dt><dd>${b}</dd></div>`).join('');
  document.querySelectorAll('#mm-size button[data-i]').forEach(b => { const on = String(st.sel) === b.dataset.i; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
  const mine = $('#mm-mine');
  mine.textContent = st.custom ? `Yours · ${st.custom.w} × ${st.custom.h}` : 'Your image…';
  mine.classList.toggle('active', st.sel === 'custom'); mine.setAttribute('aria-pressed', String(st.sel === 'custom'));
}
function renderSteps() {
  const S2 = steps(), [h, w] = hw(), ov = MM.overview(st.key, h, w);
  const chips = $('#mm-chips');
  chips.innerHTML = [{ chip: 'Overview' }, ...S2].map((s, idx) => {
    const i = idx - 1, on = i === st.step;
    return `<li><button type="button" class="chipbtn${on ? ' active' : ''}" data-i="${i}" aria-pressed="${on}">${i < 0 ? '' : `<span class="n">${idx}</span>`}${esc(s.chip)}</button></li>`;
  }).join('');
  chips.querySelectorAll('.chipbtn').forEach(b => b.addEventListener('click', () => setStep(Number(b.dataset.i), true)));
  const s = S2[st.step], card = $('#mm-card');
  if (!s) {
    card.innerHTML = `<div class="card-math">${ov.tex.map(math).join('')}</div>
      <div class="card-text"><h3>From pixels to tokens in four lines</h3><p>${fmtText(ov.text)}</p>
      <p class="hint-l">Hover any part of the diagram for details; try another image size, or your own image.</p>
      <div class="nav"><span></span><button type="button" class="next">Start: ${esc(S2[0].title)} →</button></div></div>`;
  } else {
    card.innerHTML = `<div class="card-math">${s.tex.map(math).join('')}<p class="shape">${esc(s.shape)}</p></div>
      <div class="card-text"><h3><span class="n">${st.step + 1}</span>${esc(s.title)}</h3><p>${fmtText(s.text)}</p>
      <div class="nav"><button type="button" class="prev">← ${st.step === 0 ? 'Overview' : 'Back'}</button>
      ${st.step < S2.length - 1 ? `<button type="button" class="next">Next: ${esc(S2[st.step + 1].title)} →</button>` : '<button type="button" class="next">Back to overview</button>'}</div></div>`;
  }
  card.querySelector('.prev')?.addEventListener('click', () => setStep(st.step - 1, true));
  card.querySelector('.next')?.addEventListener('click', () => setStep(st.step < S2.length - 1 ? st.step + 1 : -1, true));
}
function applyFocus() {
  const svg = $('#mm'), s = steps()[st.step], on = new Set(s ? s.on : []);
  svg.classList.toggle('focusing', !!s);
  svg.querySelectorAll('[data-h]').forEach(el => el.classList.toggle('on', el.getAttribute('data-h').split(' ').some(x => on.has(x))));
}
function setStep(i, focus) {
  st.step = Math.max(-1, Math.min(steps().length - 1, i));
  renderSteps(); applyFocus();
  if (focus) document.querySelector(`#mm-chips .chipbtn[data-i="${st.step}"]`)?.focus({ preventScroll: true });
}
function render() { renderHeader(); draw(); renderSteps(); drawClose(); renderCost(); }
function setModel(k, fromHash) {
  if (!MM.VLMS[k]) return;
  st.key = k;
  if (!fromHash) { try { history.replaceState(null, '', `#multimodal/${k}`); } catch (e) { /* sandboxed */ } }
  render();
}
function setSize(sel) { st.sel = sel; render(); }
function loadFile(file) {
  if (!file || !/^image\//.test(file.type)) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      st.custom = { url: reader.result, img, w: img.naturalWidth, h: img.naturalHeight, name: file.name };
      setSize('custom');
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function init() {
  U = window.LABUI; ({ S, esc, fmtText, math } = U);
  $('#mm-size').innerHTML = MM.SIZES.map(([h, w], i) => `<button type="button" data-i="${i}">${w} × ${h}</button>`).join('') +
    '<button type="button" id="mm-mine">Your image…</button><input type="file" id="mm-file" accept="image/*" hidden>';
  document.querySelectorAll('#mm-size button[data-i]').forEach(b => b.addEventListener('click', () => setSize(Number(b.dataset.i))));
  $('#mm-mine').addEventListener('click', () => (st.custom && st.sel !== 'custom' ? setSize('custom') : $('#mm-file').click()));
  $('#mm-file').addEventListener('change', e => loadFile(e.target.files[0]));
  const fig = $('#mm-fig');
  fig.addEventListener('dragover', e => { e.preventDefault(); });
  fig.addEventListener('drop', e => { e.preventDefault(); loadFile(e.dataTransfer.files[0]); });
  document.querySelectorAll('#mm-tabs .tab').forEach(t => t.addEventListener('click', () => setModel(t.dataset.k)));
  $('#mm-tabs').addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    const i = MM.ORDER.indexOf(st.key), k = MM.ORDER[(i + (e.key === 'ArrowRight' ? 1 : MM.ORDER.length - 1)) % MM.ORDER.length];
    setModel(k); document.querySelector(`#mm-tabs .tab[data-k="${k}"]`).focus();
  });
  document.addEventListener('keydown', e => {
    if ($('.page[data-page="multimodal"]').hidden) return;
    if (e.target.closest && e.target.closest('#mm-tabs, input, select, textarea')) return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); setStep(st.step + 1, true); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setStep(st.step - 1, true); }
    else if (e.key === 'Escape') setStep(-1, false);
    else if (/^[1-4]$/.test(e.key)) setModel(MM.ORDER[Number(e.key) - 1]);
  });
  U.attachPopups(fig, $('#mm-pop'), popContent);
  $('#mm-sources').innerHTML = MM.SOURCES.map(r => `<li>${esc(r.by)} (${r.year}). <a href="${r.url}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>. <i>${esc(r.in)}</i>.</li>`).join('');
  $('#mm-configs').innerHTML = MM.ORDER.map(k => `<a href="https://huggingface.co/${MM.VLMS[k].hf}" target="_blank" rel="noopener noreferrer">${esc(MM.VLMS[k].name)}</a>`).join(', ');
  st.ready = true;
}
window.MMAPP = {
  key: () => st.key,
  show(sub) {
    if (!st.ready) init();
    if (sub && MM.VLMS[sub]) st.key = sub;
    render();
  },
};
})();
