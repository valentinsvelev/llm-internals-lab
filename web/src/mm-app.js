/* Multimodal page — rendering and interaction. Uses mm-content.js (window.MM) and the helpers
 * app.js shares (window.LABUI: S, esc, tex, fmtText, math, attachPopups). Draws on first show. */
(function () {
'use strict';
const MM = window.MM, fmt = MM.fmt;
const $ = s => document.querySelector(s);
let U, S, esc, fmtText, math;
const st = { mode: 'images', key: 'llava', sel: 2, custom: null, step: -1, ready: false,
  keys: { images: 'llava', audio: 'qwen2audio', video: 'qwen3vl', speech: 'qwen3omni', imagegen: 'janus', omni: 'omni' },
  vals: { audio: 1, video: 1, speech: 1, imagegen: 0, omni: 0 },      // index into MEDIA.CONTROLS[mode].values, or 'custom'
  media: { audio: null, video: null } };                               // the visitor's own recording or video

const hw = () => (st.sel === 'custom' ? [st.custom.h, st.custom.w] : MM.SIZES[st.sel]);
const model = () => MM.VLMS[st.key];
const plan = () => MM.imagePlan(st.key, ...hw());
const val = () => (st.vals[st.mode] === 'custom' && st.media[st.mode] ? st.media[st.mode].seconds : MM.MEDIA.CONTROLS[st.mode].values[st.vals[st.mode]]);
const customVideo = () => (st.mode === 'video' && st.vals.video === 'custom' ? st.media.video : null);
const steps = () => (st.mode === 'images' ? MM.steps(st.key, ...hw()) : MM.MEDIA.steps(st.key, val(), customVideo()));
const stepIndex = id => steps().findIndex(s => s.id === id);

/* ---------- the example image: a fixed little scene (an image keeps its colors in dark mode) ---------- */
const RGB = { skyTop: [134, 182, 228], skyLow: [204, 227, 245], sun: [247, 197, 72], far: [128, 172, 112], near: [76, 132, 72],
  wall: [238, 228, 208], roof: [186, 80, 58], door: [98, 72, 54] };
const rgb = c => `rgb(${c.map(v => Math.round(v)).join(',')})`;
function geom(W, H, phase = null) {      // phase 0 … 1 moves the sun across the sky (video frames)
  const s = Math.min(W, H);
  const far = u => H * (0.60 - 0.09 * Math.sin(Math.PI * (0.15 + 0.85 * u)));
  const near = u => H * (0.82 - 0.07 * Math.sin(Math.PI * (1.3 * u - 0.1)));
  const hx = 0.13 * W, hwid = 0.24 * s;
  const sun = phase === null ? [0.76 * W, 0.27 * H, 0.11 * s] : [(0.18 + 0.64 * phase) * W, (0.42 - 0.24 * Math.sin(Math.PI * phase)) * H, 0.11 * s];
  return { s, far, near, sun,
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
function scene(p, x, y, w, h, [vx, vy, vw, vh], opts = {}) {
  const v = S('svg', { x, y, width: w, height: h, viewBox: `${vx} ${vy} ${vw} ${vh}`, preserveAspectRatio: 'none', 'aria-hidden': 'true' }, p);
  const [H, W] = opts.size || hw();
  const href = 'href' in opts ? opts.href : st.mode === 'images' && st.sel === 'custom' ? st.custom.url : null;
  if (href) { S('image', { href, x: 0, y: 0, width: W, height: H, preserveAspectRatio: 'none' }, v); return v; }
  const g = geom(W, H, opts.phase ?? null), hill = f => {
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

function drawImages() {
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
function popImages(key) {
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
  const k = st.key, c = st.mode === 'images' ? MM.CLOSE[k] : MM.MEDIA.CLOSE[k];
  $('#mm-close-title').textContent = c.title;
  $('#mm-close-sub').textContent = c.sub;
  $('#mm-close-text').textContent = c.text;
  if (st.mode === 'images') ({ llava: closeLlava, qwen: closeQwen, llama: closeLlama, gemma: closeGemma })[k](svg);
  else ({ audio: closeAudio, video: closeVideo, speech: closeSpeech, imagegen: closeGen, omni: closeOmni })[st.mode](svg);
  const heights = { llava: 262, qwen: 274, llama: 276, gemma: 292, qwen2audio: 214, voxtral: 268, qwen3vl: 200, llava_ov: 280, qwen3omni: 270, moshi: 270, janus: 260, emu3: 316, omni: 170 };
  svg.setAttribute('viewBox', `0 0 1400 ${heights[k]}`);
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
function renderCostImages() {
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

/* ========================================================================================================
 * Audio, video, speech out, image out, all together. Same layout as the image diagram: one column per
 * stage, centred on Y0; the language model as token rows × layer columns.
 * ======================================================================================================== */
const X = () => MM.MEDIA;
const mmodel = () => X().MODELS[st.key];

/* ---------- example signals (drawn, deterministic) ---------- */
function hash(i) { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }
function envelope(t) {            // syllables of 0.22 s with short gaps, phrases of 5–9 with pauses
  let start = 0, k = 0;
  while (true) {
    const n = 5 + Math.floor(hash(k) * 5), len = n * 0.28 + 0.45;
    if (t < start + len) {
      const u = t - start; if (u > n * 0.28) return 0.02;
      const s = Math.floor(u / 0.28), w = (u - s * 0.28) / 0.22;
      return w > 1 ? 0.03 : Math.pow(Math.sin(Math.PI * w), 0.6) * (0.55 + 0.45 * hash(k * 13 + s));
    }
    start += len; k++;
    if (k > 100000) return 0;
  }
}
const FORMANTS = [[700, 1200], [400, 2200], [300, 900], [600, 1800], [500, 1500]];
function synthSample(t) {          // a voiced sound at 140 Hz whose vowel changes every syllable
  const e = envelope(t); if (e < 0.05) return e * (hash(Math.floor(t * 16000)) - 0.5);
  const [f1, f2] = FORMANTS[Math.floor(t / 0.28) % FORMANTS.length];
  let v = 0;
  for (let h = 1; h <= 24; h++) { const f = 140 * h, g = Math.exp(-(((f - f1) / 180) ** 2)) + 0.6 * Math.exp(-(((f - f2) / 260) ** 2)) + 0.04; v += g * Math.sin(2 * Math.PI * f * t + h); }
  return e * v / 3;
}
function wavePeaks(seconds, n, offset = 0) {
  const a = st.media.audio && st.mode !== 'speech' ? st.media.audio : null, out = [];
  for (let i = 0; i < n; i++) {
    const t0 = offset + seconds * i / n, t1 = offset + seconds * (i + 1) / n;
    if (a) {
      const s0 = Math.floor(t0 * a.rate), s1 = Math.min(a.data.length, Math.max(s0 + 1, Math.floor(t1 * a.rate)));
      let lo = 0, hi = 0; for (let j = s0; j < s1; j += Math.max(1, Math.floor((s1 - s0) / 64))) { lo = Math.min(lo, a.data[j] || 0); hi = Math.max(hi, a.data[j] || 0); }
      out.push([lo, hi]);
    } else { const e = envelope((t0 + t1) / 2); out.push([-e * (0.7 + 0.3 * hash(i)), e * (0.7 + 0.3 * hash(i + 7))]); }
  }
  return out;
}
const BANDS = Array.from({ length: 16 }, (_, b) => 100 * Math.pow(75, b / 15));      // 100 Hz – 7.5 kHz, log-spaced
function bandEnergies(t) {          // Goertzel at 16 frequencies over a 25-ms window around t
  const a = st.media.audio, N = 400, out = [];
  const x = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const tt = t + (i - N / 2) / 16000;
    x[i] = a ? (a.data[Math.floor(tt * a.rate)] || 0) : synthSample(Math.max(0, tt));
    x[i] *= 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  }
  for (const f of BANDS) {
    const w = 2 * Math.PI * f / 16000, c = 2 * Math.cos(w); let s1 = 0, s2 = 0;
    for (let i = 0; i < N; i++) { const s0 = x[i] + c * s1 - s2; s2 = s1; s1 = s0; }
    out.push(Math.log10(1e-4 + s1 * s1 + s2 * s2 - c * s1 * s2));
  }
  return out;
}
function spectrogram(p, x, y, w, h, t0, t1, cols) {
  const g = S('g', { class: 'spec' }, p), cw = w / cols, rh = h / BANDS.length;
  const vals = []; for (let i = 0; i < cols; i++) vals.push(bandEnergies(t0 + (t1 - t0) * (i + 0.5) / cols));
  const all = vals.flat(), hi = Math.max(...all), lo = hi - 4;
  vals.forEach((col, i) => col.forEach((v, b) => {
    const o = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    if (o > 0.04) S('rect', { x: x + i * cw, y: y + h - (b + 1) * rh, width: cw + 0.4, height: rh + 0.4, 'fill-opacity': o.toFixed(2) }, g);
  }));
  return g;
}
function waveform(p, x, y, w, h, peaks, cls = 'wave') {
  const mid = y + h / 2, n = peaks.length, top = peaks.map(([, hi], i) => `${(x + (i + 0.5) * w / n).toFixed(1)},${(mid - hi * h / 2).toFixed(1)}`);
  const bot = peaks.map(([lo], i) => `${(x + (i + 0.5) * w / n).toFixed(1)},${(mid - lo * h / 2).toFixed(1)}`).reverse();
  S('polygon', { points: top.concat(bot).join(' '), class: cls }, p);
  S('line', { x1: x, y1: mid, x2: x + w, y2: mid, class: 'axis' }, p);
}

/* ---------- the language model column (token rows × layer columns), as in the image diagram ---------- */
function lmColumn(svg, { rows, tokens, label, note, dec, layerCls, stepSeq = 'sequence', stepDec = 'decode', next = true }) {
  let y = 64; const R = [];
  for (const [lab, kind] of rows) {
    const hgt = kind === 'media' ? Math.max(44, Math.min(140, tokens / 6)) : 20;
    R.push({ label: lab, kind, y, h: hgt, capped: kind === 'media' && tokens / 6 > 140 }); y += hgt + 6;
  }
  const bottom = y - 6, boxTop = 56, boxBot = bottom + 8, N = dec.layers;
  const gE = S('g', { 'data-h': 'dec' }, svg);
  S('rect', { x: COL.l, y: boxTop, width: COL.lw, height: boxBot - boxTop, rx: 8, class: 'subbox' }, gE);
  R.forEach(r => S('rect', { x: COL.l + 2, y: r.y, width: COL.lw - 4, height: r.h, class: r.kind === 'media' ? 'imgband' : 'tband' }, gE));
  for (let i = 0; i < N; i++) {
    const { x, pitch } = layerX(i, N);
    const g1 = act(S('g', { class: 'lay', 'data-h': 'dec' }, gE), stepDec, `dlayer:${i}`, `Decoder layer ${i + 1}`);
    S('rect', { x, y: boxTop + 2, width: pitch, height: boxBot - boxTop - 4, class: 'lhit' }, g1);
    S('rect', { x: x + pitch * 0.22, y: boxTop + 4, width: Math.max(1.5, pitch * 0.56), height: boxBot - boxTop - 8, rx: 1, class: 'dlayer' + (layerCls ? layerCls(i) : '') }, g1);
  }
  const gS = S('g', { 'data-h': 'seq' }, svg);
  R.forEach((r, i) => {
    const g1 = act(S('g', {}, gS), stepSeq, r.kind === 'media' ? 'media' : `tok:${i}`, r.label);
    if (r.kind === 'media') {
      S('rect', { x: COL.e, y: r.y, width: COL.ew, height: r.h, rx: 4, class: 'imgblock' }, g1);
      T(g1, COL.e + 8, r.y + 15, label, 'tok');
      T(g1, COL.e + 8, r.y + 31, `× ${fmt(tokens)}`, 'tok');
      if (note && r.h >= 60) T(g1, COL.e + 8, r.y + 47, note, 'tiny');
      if (r.capped) T(g1, COL.e + COL.ew - 8, r.y + r.h - 7, 'not to scale', 'tiny', 'end');
    } else {
      S('rect', { x: COL.e, y: r.y, width: COL.ew, height: r.h, rx: 4, class: 'chip' + (r.kind === 'mark' ? ' special' : r.kind === 'stamp' ? ' stamp' : '') }, g1);
      T(g1, COL.e + 8, r.y + 14, short(r.label, 25), 'tok');
    }
  });
  if (next) {
    const last = R[R.length - 1], ly = last.y + last.h / 2, gO = act(S('g', { 'data-h': 'dec' }, svg), stepDec, 'next', 'Next token');
    line(gO, [[COL.l + COL.lw + 4, ly], [COL.out + 4, ly]]);
    S('rect', { x: COL.out + 8, y: ly - 11, width: 72, height: 22, rx: 11, class: 'pillbox' }, gO);
    T(gO, COL.out + 44, ly + 4, 'next token', 'pill-l', 'middle');
  }
  const media = R.find(r => r.kind === 'media');
  return { R, bottom, boxBot, mediaY: media ? media.y + Math.min(media.h / 2, 26) : Y0 };
}
function headers(svg, cols) { cols.forEach(([x, a, b]) => { T(svg, x, 30, a, 'st'); T(svg, x, 47, b, 'st-s'); }); }
function encoderBox(svg, { x, w = 150, layers, label, sub, stepId = 'encode', taps = [], tapCls = 'tap', top = Y0 - 135, h = 270, below = [] }) {
  const g = act(S('g', { 'data-h': 'enc' }, svg), stepId, 'enc', label);
  S('rect', { x, y: top, width: w, height: h, rx: 8, class: 'subbox' }, g);
  T(g, x + w / 2, top + 18, `${layers} layers`, 'st-s', 'middle');
  if (sub) T(g, x + w / 2, top + 33, sub, 'dim', 'middle');
  const x0 = x + 16, pitch = (w - 32) / layers, colH = h - 96;
  for (let i = 0; i < layers; i++) {
    const g1 = S('g', { class: 'lay', 'data-pop': `elayer:${i}` }, g), cx = x0 + (i + 0.5) * pitch;
    S('rect', { x: cx - pitch / 2, y: top + 44, width: pitch, height: colH + 12, class: 'lhit' }, g1);
    S('rect', { x: cx - Math.max(1.6, pitch * 0.28), y: top + 50, width: Math.max(3.2, pitch * 0.56), height: colH, rx: 1, class: 'elayer' }, g1);
  }
  taps.forEach(t => S('circle', { cx: x0 + (t + 0.5) * pitch, cy: top + 43, r: 3.2, class: tapCls, 'data-h': 'enc' }, svg));
  below.forEach(([txt, cls], i) => T(svg, x + w / 2, top + h + 22 + 16 * i, txt, cls, 'middle', 'enc'));
  return { top, bot: top + h };
}
function smallBox(svg, { x, y, w, h, stepId, pop, label, lines = [], cls = 'subbox', hcls }) {
  const g = act(S('g', { 'data-h': hcls || stepId }, svg), stepId, pop || `step:${stepId}`, label);
  S('rect', { x, y, width: w, height: h, rx: 8, class: cls }, g);
  lines.forEach(([t, c], i) => T(g, x + w / 2, y + 20 + 16 * i, t, c, 'middle'));
  return g;
}
function costLine(svg, x, y, title, value, note) {
  const g = act(S('g', { 'data-h': 'cost' }, svg), 'cost', 'step:cost', title);
  T(g, x, y, title, 'st-s'); T(g, x, y + 26, value, 'cost-v'); T(g, x, y + 46, note, 'note');
  return y + 52;
}
function finish(svg, bottom) { svg.setAttribute('viewBox', `0 0 1400 ${Math.ceil(bottom + 14)}`); applyFocus(); }
function frameThumb(p, x, y, w, h, t, total, src) {
  const cust = st.media.video, href = src === 'video' && cust && cust.frames.length ? nearestFrame(t) : null;
  const [H, W] = cust && src === 'video' ? [cust.h, cust.w] : [720, 1280];
  scene(p, x, y, w, h, [0, 0, W, H], { size: [H, W], phase: total ? Math.min(1, t / total) : 0.5, href });
  S('rect', { x, y, width: w, height: h, class: 'frame' }, p);
}
function nearestFrame(t) {
  const f = st.media.video.frames; let best = f[0];
  for (const c of f) if (Math.abs(c.t - t) < Math.abs(best.t - t)) best = c;
  return best.url;
}

/* ---------- audio ---------- */
function drawAudio() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = mmodel(), k = st.key, v = val(), p = X().audioPlan(k, v), q = k === 'qwen2audio', d = m.dec, secs = X().secs;
  svg.setAttribute('aria-label', `${m.name}: ${secs(v)} of audio becomes ${fmt(p.tokens)} tokens.`);
  headers(svg, [[COL.a, 'Sound', `${secs(v)} · 16 kHz`], [COL.b, 'Spectrogram', q ? 'one 30-s window, 100 frames/s' : `${p.chunks} × 30-s chunk${p.chunks > 1 ? 's' : ''}`],
    [COL.c, 'Audio encoder', 'Whisper large-v3'], [COL.d, 'Adapter', m.conn.name], [COL.e, 'Language model', `${d.name} · ${d.layers} layers`]]);
  // A: the waveform
  const gA = act(S('g', { 'data-h': 'wave' }, svg), 'sound', 'step:sound', 'Waveform');
  S('rect', { x: COL.a, y: Y0 - 55, width: COL.aw, height: 110, rx: 6, class: 'subbox' }, gA);
  waveform(gA, COL.a + 8, Y0 - 45, COL.aw - 16, 90, wavePeaks(v, 120));
  if (q && v > 30) {
    const cut = COL.a + 8 + (COL.aw - 16) * 30 / v;
    S('rect', { x: cut, y: Y0 - 52, width: COL.a + COL.aw - 8 - cut, height: 104, class: 'shade' }, gA);
    T(gA, (cut + COL.a + COL.aw - 8) / 2, Y0 + 4, 'cut off', 'tiny', 'middle');
  }
  T(svg, COL.a + COL.aw / 2, Y0 + 77, `${fmt(Math.round(v * 16000))} numbers`, 'dim', 'middle', 'wave');
  T(svg, COL.a + COL.aw / 2, Y0 + 93, 'air pressure, 16,000 per second', 'note', 'middle', 'wave');
  line(svg, [[COL.a + COL.aw + 8, Y0], [COL.b - 8, Y0]], 'arr', 'mel');
  // B: the 30-s windows
  const gB = act(S('g', { 'data-h': 'mel' }, svg), 'mel', 'step:mel', 'Spectrogram');
  const wins = q ? 1 : Math.min(p.chunks, 4), gap = 6, ww = (COL.bw - gap * (wins - 1)) / wins, wy = Y0 - 60, wh = 120;
  for (let i = 0; i < wins; i++) {
    const idx = !q && p.chunks > 4 && i >= 2 ? (i === 2 ? -1 : p.chunks - 1) : i, wx = COL.b + i * (ww + gap);
    if (idx === -1) { S('rect', { x: wx, y: wy, width: ww, height: wh, rx: 4, class: 'tile' }, gB); T(gB, wx + ww / 2, Y0 + 4, '…', 'st', 'middle'); continue; }
    const t0 = idx * 30, filled = Math.max(0, Math.min(30, v - t0)) / 30;
    S('rect', { x: wx, y: wy, width: ww, height: wh, rx: 4, class: 'specbg' }, gB);
    spectrogram(gB, wx, wy, ww * filled, wh, t0, t0 + 30 * filled, Math.max(4, Math.round((q ? 90 : 60 / wins) * filled)));
    if (filled < 1) { S('rect', { x: wx + ww * filled, y: wy, width: ww * (1 - filled), height: wh, class: 'pad' }, gB); if (ww * (1 - filled) > 60) T(gB, wx + ww * (1 + filled) / 2, Y0 + 4, 'padding', 'tiny', 'middle'); }
    S('rect', { x: wx, y: wy, width: ww, height: wh, rx: 4, class: 'frame' }, gB);
    if (!q) T(gB, wx + 4, wy - 5, `chunk ${idx + 1}`, 'tiny');
  }
  T(gB, COL.b - 4, wy + 10, '7.5k', 'tiny', 'end'); T(gB, COL.b - 4, wy + wh, '100 Hz', 'tiny', 'end');
  T(svg, COL.b + COL.bw / 2, Y0 + 77, q ? `3,000 frames × 128 mel bands` : `${p.chunks} × 3,000 frames × 128 mel bands`, 'dim', 'middle', 'mel');
  T(svg, COL.b + COL.bw / 2, Y0 + 93, q ? (v > 30 ? 'only the first 30 s' : `${fmt(p.melFrames)} frames of sound, the rest silence`) : 'each chunk padded to 30 s and encoded alone', 'note', 'middle', 'mel');
  line(svg, [[COL.b + COL.bw + 8, Y0], [COL.c - 8, Y0]], 'arr', 'enc');
  // C, D
  encoderBox(svg, { x: COL.c, layers: 32, label: 'Whisper large-v3 encoder', sub: 'd 1,280 · 20 heads',
    below: [[`in: ${q ? '' : `${p.chunks} × `}3,000 → conv → 1,500`, 'dim'], ['no causal mask · 637M parameters', 'note']] });
  line(svg, [[COL.c + COL.cw + 8, Y0], [COL.d - 8, Y0]], 'arr', 'conn');
  const gD = smallBox(svg, { x: COL.d, y: Y0 - 50, w: COL.dw, h: 100, stepId: 'adapt', pop: 'step:adapt', label: m.conn.name, hcls: 'conn' });
  const cx = COL.d + COL.dw / 2;
  if (q) {
    [0, 1].forEach(i => S('rect', { x: cx - 40 + i * 12, y: Y0 - 32, width: 9, height: 30, class: 'vec' }, gD));
    line(gD, [[cx - 12, Y0 - 17], [cx + 12, Y0 - 17]]); S('rect', { x: cx + 18, y: Y0 - 32, width: 12, height: 30, class: 'vec lat' }, gD);
    T(gD, cx, Y0 + 22, 'average 2 → 1', 'tiny', 'middle'); T(gD, cx, Y0 + 38, 'LN · linear 1,280 → 4,096', 'tiny', 'middle');
  } else {
    [0, 1, 2, 3].forEach(i => S('rect', { x: cx - 48 + i * 9, y: Y0 - 32, width: 7, height: 30, class: 'vec' }, gD));
    line(gD, [[cx - 10, Y0 - 17], [cx + 12, Y0 - 17]]); S('rect', { x: cx + 18, y: Y0 - 32, width: 24, height: 30, class: 'vec lat' }, gD);
    T(gD, cx, Y0 + 22, '4 side by side → 1', 'tiny', 'middle'); T(gD, cx, Y0 + 38, 'MLP 5,120 → 3,072', 'tiny', 'middle');
  }
  T(svg, cx, Y0 + 70, `out: ${fmt(p.tokens)} × ${fmt(d.d)}`, 'dim', 'middle', 'conn');
  T(svg, cx, Y0 + 86, `${(m.params[1] / 1e6).toFixed(1)}M parameters`, 'note', 'middle', 'conn');
  // E
  const L = lmColumn(svg, { rows: m.template, tokens: p.tokens, label: q ? '<|AUDIO|>' : '[AUDIO]', note: q ? '25 per second' : '12.5 per second', dec: d });
  line(svg, L.mediaY === Y0 ? [[COL.d + COL.dw + 8, Y0], [COL.e - 6, Y0]] : [[COL.d + COL.dw + 8, Y0], [COL.e - 22, Y0], [COL.e - 22, L.mediaY], [COL.e - 6, L.mediaY]], 'arr', 'conn');
  const cache = p.tokens * X().kvPerToken(d);
  const yb = costLine(svg, COL.e, Math.max(Y0 + 157, L.boxBot + 40), 'The sound in the KV cache', MM.fmtMiB(cache),
    `${fmt(p.tokens)} tokens × ${d.layers} layers · ${(100 * p.tokens / d.ctx).toFixed(p.tokens / d.ctx < 0.01 ? 2 : 1)}% of the ${fmt(d.ctx)}-token context`);
  finish(svg, Math.max(yb, Y0 + 100));
}

/* ---------- video ---------- */
function drawVideo() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = mmodel(), k = st.key, v = val(), cust = customVideo(), [H, W] = cust ? [cust.h, cust.w] : [720, 1280];
  const p = X().videoPlan(k, v, H, W), q = k === 'qwen3vl', d = m.dec, secs = X().secs, total = Math.max(1, Math.round(v * 30));
  svg.setAttribute('aria-label', `${m.name}: a ${secs(v)} video becomes ${fmt(p.tokens)} tokens.`);
  headers(svg, [[COL.a, 'Video', `${secs(v)} · ${W} × ${H} · 30 fps${cust ? ' (assumed)' : ''}`], [COL.b, 'Frames read', q ? `${fmt(p.frames)} frames, in ${fmt(p.groups)} pairs` : '32 frames, evenly spaced'],
    [COL.c, 'Vision encoder', m.enc.name], [COL.d, 'Connector', m.conn.name], [COL.e, 'Language model', `${d.name} · ${d.layers} layers`]]);
  // A: a stack of frames
  const gA = act(S('g', { 'data-h': 'film' }, svg), 'video', 'step:video', 'Video');
  const fw = 160, fh = fw * H / W;
  for (let i = 3; i >= 0; i--) frameThumb(gA, COL.a + 6 + i * 9, Y0 - fh / 2 - 18 + i * 9, fw, fh, v * (3 - i) / 3, v, 'video');
  T(svg, COL.a + COL.aw / 2, Y0 + 77, `${fmt(total)} frames`, 'dim', 'middle', 'film');
  T(svg, COL.a + COL.aw / 2, Y0 + 93, 'each one a full image', 'note', 'middle', 'film');
  line(svg, [[COL.a + COL.aw + 8, Y0], [COL.b - 8, Y0]], 'arr', 'pick');
  // B: the time axis and the frames kept
  const gB = act(S('g', { 'data-h': 'pick' }, svg), 'sample', 'step:sample', 'Sampled frames');
  const ax = COL.b, aw = COL.bw, ay = Y0 - 112;
  S('rect', { x: ax, y: ay, width: aw, height: 8, rx: 2, class: 'specbg' }, gB);
  const times = q ? p.indices.map(i => i / 30) : Array.from({ length: 32 }, (_, i) => (v * (i + 0.5)) / 32);
  if (times.length <= 140) times.forEach(t => S('rect', { x: ax + aw * t / v - 0.6, y: ay - 3, width: 1.2, height: 14, class: 'tick' }, gB));
  else S('rect', { x: ax, y: ay - 3, width: aw, height: 14, class: 'tick dense' }, gB);
  T(gB, ax, ay - 8, '0 s', 'tiny'); T(gB, ax + aw, ay - 8, secs(v), 'tiny', 'end');
  if (q) {
    const show = Math.min(6, p.groups), cw = 82, th = 26, tw = th * p.frameSize[1] / p.frameSize[0], ww = Math.min(cw - 12, tw);
    for (let g = 0; g < show; g++) {
      const gi = g < 5 || p.groups <= 6 ? g : p.groups - 1, col = g % 3, row = Math.floor(g / 3);
      const x = ax + col * (cw + 12) + 4, y = Y0 - 82 + row * 78;
      for (let f = 1; f >= 0; f--) { const idx = Math.min(gi * 2 + f, p.frames - 1); frameThumb(gB, x + f * 6, y + f * 6, ww, ww * p.frameSize[0] / p.frameSize[1], p.indices[idx] / 30, v, 'video'); }
      T(gB, x, y + 50, `<${p.stamps[gi].toFixed(1)} seconds>`, 'tiny');
      if (g === 4 && p.groups > 6) T(gB, x + cw + 4, y + 20, '…', 'st', 'middle');
    }
  } else {
    const c = 15, gp = 2;
    for (let i = 0; i < 32; i++) frameThumb(gB, ax + 4 + (i % 8) * (c + gp) * 2, Y0 - 82 + Math.floor(i / 8) * 34, c * 2 - 2, c * 2 - 2, times[i], v, 'video');
  }
  T(svg, COL.b + COL.bw / 2, Y0 + 77, q ? `${p.frameSize[1]} × ${p.frameSize[0]} per frame` : '384 × 384 per frame (squashed)', 'dim', 'middle', 'pick');
  T(svg, COL.b + COL.bw / 2, Y0 + 93, q ? p.frames === 768 ? 'fewer than one pair per second: at most 768 frames' : 'one pair per second' : `one frame every ${(v / 32).toFixed(1)} s`, 'note', 'middle', 'pick');
  line(svg, [[COL.b + COL.bw + 8, Y0], [COL.c - 8, Y0]], 'arr', 'enc');
  encoderBox(svg, { x: COL.c, layers: m.enc.layers, label: m.enc.name, sub: `d ${fmt(m.enc.d)} · ${m.enc.heads} heads`,
    below: [[q ? `in: ${fmt(p.groups)} × ${fmt(p.perGroup * 4)} patches` : 'in: 32 × 729 patches', 'dim'], [`${q ? 'patch: 2 frames × 16 × 16' : 'each frame alone'} · ${(m.params[0] / 1e6).toFixed(0)}M`, 'note']] });
  line(svg, [[COL.c + COL.cw + 8, Y0], [COL.d - 8, Y0]], 'arr', 'conn');
  const gD = smallBox(svg, { x: COL.d, y: Y0 - 50, w: COL.dw, h: 100, stepId: 'connect', pop: 'step:connect', label: m.conn.name, hcls: 'conn' }), cx = COL.d + COL.dw / 2;
  if (q) {
    for (let i = 0; i < 4; i++) S('rect', { x: cx - 42 + (i % 2) * 15, y: Y0 - 28 + Math.floor(i / 2) * 15, width: 13, height: 13, class: 'vec' }, gD);
    line(gD, [[cx - 8, Y0 - 14], [cx + 14, Y0 - 14]]); S('rect', { x: cx + 20, y: Y0 - 21, width: 14, height: 14, class: 'vec lat' }, gD);
    T(gD, cx, Y0 + 22, '2 × 2 → 1', 'tiny', 'middle'); T(gD, cx, Y0 + 38, 'MLP 4,608 → 4,096', 'tiny', 'middle');
  } else {
    for (let i = 0; i < 9; i++) S('rect', { x: cx - 46 + (i % 3) * 9, y: Y0 - 32 + Math.floor(i / 3) * 9, width: 8, height: 8, class: 'vec' }, gD);
    line(gD, [[cx - 14, Y0 - 18], [cx + 8, Y0 - 18]]); for (let i = 0; i < 4; i++) S('rect', { x: cx + 14 + (i % 2) * 12, y: Y0 - 30 + Math.floor(i / 2) * 12, width: 11, height: 11, class: 'vec lat' }, gD);
    T(gD, cx, Y0 + 22, 'MLP, then 27 → 14', 'tiny', 'middle'); T(gD, cx, Y0 + 38, 'bilinear, per frame', 'tiny', 'middle');
  }
  T(svg, cx, Y0 + 70, `out: ${fmt(p.tokens)} × ${fmt(d.d)}`, 'dim', 'middle', 'conn');
  T(svg, cx, Y0 + 86, `${(m.params[1] / 1e6).toFixed(1)}M parameters`, 'note', 'middle', 'conn');
  let nStamp = 0;
  const rows = m.template.map(r => (r[1] === 'stamp' ? [nStamp++ === 0 ? `<${p.stamps[0].toFixed(1)} seconds>` : `<${(p.stamps[1] ?? p.stamps[0]).toFixed(1)} seconds> …`, 'stamp'] : r));
  const L = lmColumn(svg, { rows, tokens: p.tokens, label: q ? '<|video_pad|>' : '<video>', note: q ? `${fmt(p.perGroup)} per pair` : '196 per frame + 1', dec: d });
  line(svg, L.mediaY === Y0 ? [[COL.d + COL.dw + 8, Y0], [COL.e - 6, Y0]] : [[COL.d + COL.dw + 8, Y0], [COL.e - 22, Y0], [COL.e - 22, L.mediaY], [COL.e - 6, L.mediaY]], 'arr', 'conn');
  const cache = p.tokens * X().kvPerToken(d);
  const yb = costLine(svg, COL.e, Math.max(Y0 + 157, L.boxBot + 40), 'The video in the KV cache', MM.fmtMiB(cache),
    `${fmt(p.tokens)} tokens × ${d.layers} layers · ${(100 * p.tokens / d.ctx).toFixed(0)}% of the ${fmt(d.ctx)}-token context`);
  finish(svg, Math.max(yb, Y0 + 100));
}

/* ---------- speech out ---------- */
const SP = { a: 20, aw: 290, b: 350, bw: 190, c: 580, cw: 370, d: 990, dw: 150, e: 1180, ew: 200 };
function codeGrid(p, x, y, w, h, rows, cols, cls, delays) {
  const cw = w / cols, rh = h / rows, g = S('g', {}, p);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const d = delays ? delays[r] : 0; if (c < d) continue;
    S('rect', { x: x + c * cw + 0.8, y: y + r * rh + 0.8, width: cw - 1.6, height: rh - 1.6, rx: 1.5, class: 'code ' + cls(r, c) }, g);
  }
  return g;
}
function drawSpeech() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = mmodel(), k = st.key, v = val(), p = X().speechPlan(k, v), q = k === 'qwen3omni', secs = X().secs;
  svg.setAttribute('aria-label', `${m.name}: ${secs(v)} of speech is ${fmt(p.frames)} frames of ${m.codebooks} codes.`);
  headers(svg, [[SP.a, q ? 'Thinker' : 'Helium', q ? 'the language model, writing text' : 'every 80 ms: 17 streams in'], [SP.b, q ? 'Talker + code predictor' : 'Depth transformer', q ? 'one frame at a time' : '8 codes per step'],
    [SP.c, 'Codec tokens', `${m.codebooks} codebooks × ${fmt(p.frames)} frames`], [SP.d, q ? 'Code2Wav' : 'Mimi decoder', 'codes → sound'], [SP.e, 'Speech', `${secs(v)} · 24 kHz`]]);
  // A: the big model
  const gA = act(S('g', { 'data-h': 'lm' }, svg), q ? 'think' : 'main', q ? 'step:think' : 'step:main', m.lm.name);
  const top = Y0 - 110, h = 220;
  S('rect', { x: SP.a, y: top, width: SP.aw, height: h, rx: 8, class: 'subbox' }, gA);
  T(gA, SP.a + SP.aw / 2, top + 18, `${m.lm.layers} layers · ${m.lm.note}`, 'st-s', 'middle');
  const lx0 = SP.a + 100, lw = SP.aw - 112, N = m.lm.layers, pitch = lw / N;
  for (let i = 0; i < N; i++) S('rect', { x: lx0 + i * pitch + pitch * 0.22, y: top + 30, width: Math.max(1.4, pitch * 0.56), height: h - 76, rx: 1, class: 'dlayer' }, gA);
  if (q) {
    ['user:', 'Tell me', 'about the', 'sun.'].forEach((t, i) => { S('rect', { x: SP.a + 10, y: top + 32 + i * 26, width: 80, height: 20, rx: 4, class: 'chip' }, gA); T(gA, SP.a + 16, top + 46 + i * 26, t, 'tok'); });
    const words = ['The', 'sun', 'is', 'a', 'star', '…'];
    words.forEach((t, i) => { S('rect', { x: SP.a + 10 + i * 46, y: top + h - 36, width: 42, height: 20, rx: 4, class: 'chip out' }, gA); T(gA, SP.a + 31 + i * 46, top + h - 22, t, 'tok', 'middle'); });
  } else {
    [['text', 1, 'txt'], ['Moshi 1–8', 8, 'moshi'], ['user 1–8', 8, 'user']].reduce((y, [lab, n, cls]) => {
      T(gA, SP.a + 10, y + 10, lab, 'tiny');
      for (let i = 0; i < n; i++) S('rect', { x: SP.a + 62, y: y + i * 6, width: 32, height: 4.4, rx: 1, class: 'code ' + cls }, gA);
      return y + n * 6 + 14;
    }, top + 30);
    T(gA, SP.a + 10, top + h - 22, 'summed into one vector per step', 'tiny');
  }
  // the user's voice into Moshi
  if (!q) {
    const gU = act(S('g', { 'data-h': 'lm' }, svg), 'streams', 'step:streams', 'User audio');
    waveform(gU, SP.a + 20, Y0 + 128, 120, 26, wavePeaks(4, 50, 1.3), 'wave user');
    T(gU, SP.a + 150, Y0 + 145, '← the user, through Mimi’s encoder', 'tiny');
  }
  line(svg, [[SP.a + SP.aw + 8, Y0], [SP.b - 8, Y0]], 'arr', q ? 'main' : 'inner');
  T(svg, (SP.a + SP.aw + SP.b) / 2, Y0 - 8, q ? 'states' : 'zₜ', 'tiny', 'middle');
  // B
  if (q) {
    smallBox(svg, { x: SP.b, y: Y0 - 100, w: SP.bw, h: 86, stepId: 'talk', label: 'Talker', hcls: 'main', lines: [['Talker', 'st-s'], ['20 layers · MoE, 6 of 128 experts', 'tiny'], ['→ codebook 1, once per frame', 'tiny']] });
    smallBox(svg, { x: SP.b, y: Y0 + 14, w: SP.bw, h: 86, stepId: 'fill', label: 'Code predictor', hcls: 'inner', lines: [['code predictor', 'st-s'], ['5 layers · 1,024 wide', 'tiny'], ['→ codebooks 2–16, one by one', 'tiny']] });
    line(svg, [[SP.b + SP.bw / 2, Y0 - 12], [SP.b + SP.bw / 2, Y0 + 12]], 'arr', 'inner');
  } else {
    smallBox(svg, { x: SP.b, y: Y0 - 60, w: SP.bw, h: 120, stepId: 'depth', label: 'Depth transformer', hcls: 'inner',
      lines: [['depth transformer', 'st-s'], ['6 layers · 1,024 wide', 'tiny'], ['along the codebooks:', 'tiny'], ['text token, then codes 1 … 8', 'tiny'], ['(Helium: along time)', 'tiny']] });
  }
  line(svg, [[SP.b + SP.bw + 8, Y0], [SP.c - 8, Y0]], 'arr', 'grid');
  // C: the code grid
  const cols = Math.min(p.frames + (q ? 0 : 1), 24), rows = m.codebooks, gh = q ? 210 : 130, gy = Y0 - gh / 2;
  const gC = act(S('g', { 'data-h': 'grid' }, svg), q ? 'fill' : 'depth', 'grid', 'Code grid');
  codeGrid(gC, SP.c + 40, gy, SP.cw - 52, gh, rows, cols, r => (r === 0 ? 'main' : 'inner'), q ? null : X().moshiDelays(rows));
  for (let r = 0; r < rows; r += q ? 3 : 1) T(gC, SP.c + 34, gy + (r + 0.75) * gh / rows, String(r + 1), 'tiny', 'end');
  T(gC, SP.c + 34, gy - 6, 'codebook', 'tiny', 'end');
  T(gC, SP.c + 40, gy + gh + 14, '80 ms per column', 'tiny');
  if (p.frames + (q ? 0 : 1) > cols) T(gC, SP.c + SP.cw - 12, gy + gh + 14, `… ${fmt(p.frames)} frames`, 'tiny', 'end');
  T(svg, SP.c + SP.cw / 2, Y0 + 135, `${fmt(p.codes)} codes · 2,048 values each`, 'dim', 'middle', 'grid');
  T(svg, SP.c + SP.cw / 2, Y0 + 151, q ? 'highlighted: the Talker’s codebook 1 · the rest: code predictor' : 'highlighted: codebook 1 (meaning) · 2–8 (sound) come one step later', 'note', 'middle', 'grid');
  line(svg, [[SP.c + SP.cw + 8, Y0], [SP.d - 8, Y0]], 'arr', 'dec');
  smallBox(svg, { x: SP.d, y: Y0 - 50, w: SP.dw, h: 100, stepId: q ? 'decode' : 'codec', label: m.decoder, hcls: 'dec',
    lines: q ? [['Code2Wav', 'st-s'], ['8 layers, then', 'tiny'], ['upsampling × 1,920', 'tiny'], ['216M parameters', 'tiny']] : [['Mimi', 'st-s'], ['codec decoder', 'tiny'], ['× 1,920 per frame', 'tiny'], ['79M (with encoder)', 'tiny']] });
  line(svg, [[SP.d + SP.dw + 8, Y0], [SP.e - 8, Y0]], 'arr', 'wave');
  const gE = act(S('g', { 'data-h': 'wave' }, svg), q ? 'decode' : 'codec', q ? 'step:decode' : 'step:codec', 'Speech out');
  S('rect', { x: SP.e, y: Y0 - 45, width: SP.ew, height: 90, rx: 6, class: 'subbox' }, gE);
  waveform(gE, SP.e + 8, Y0 - 37, SP.ew - 16, 74, wavePeaks(Math.min(v, 6), 100));
  T(svg, SP.e + SP.ew / 2, Y0 + 67, `${fmt(p.samples)} samples`, 'dim', 'middle', 'wave');
  const yb = costLine(svg, SP.c, Y0 + 186, `What ${secs(v)} of speech takes`, `${fmt(p.mainSteps)} + ${fmt(p.innerSteps)} passes`,
    `${q ? 'Talker' : 'Helium'} once per frame + ${q ? 'code predictor' : 'depth transformer'} per code · ${MM.fmtMiB(p.kv)} cached`);
  finish(svg, yb);
}

/* ---------- image out ---------- */
const GN = { a: 20, aw: 300, b: 360, bw: 180, c: 580, cw: 300, d: 920, dw: 150, e: 1110, ew: 270 };
function drawGen() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = mmodel(), k = st.key, ratio = val(), p = X().genPlan(k, ratio), j = k === 'janus', d = m.dec;
  svg.setAttribute('aria-label', `${m.name}: an image is ${fmt(p.codes)} codes written one by one.`);
  headers(svg, [[GN.a, 'Language model', `${d.name} · ${d.layers} layers`], [GN.b, j ? 'Image head + guidance' : 'The usual head', j ? '4,096 → 16,384 codes' : '184,622 tokens, codes only'],
    [GN.c, 'Codes', `${p.grid[1]} × ${p.grid[0]}${j ? '' : ' + end of line'} · ${fmt(m.codebook)} in the codebook`], [GN.d, 'VQ decoder', `× ${m.down} per side`], [GN.e, 'Image', `${p.pixels[1]} × ${p.pixels[0]}`]]);
  const top = Y0 - 110, h = 220, gA = act(S('g', { 'data-h': 'lm' }, svg), 'prompt', 'step:prompt', 'Language model');
  S('rect', { x: GN.a, y: top, width: GN.aw, height: h, rx: 8, class: 'subbox' }, gA);
  ['A house on a', 'hill, at noon.', j ? '<begin_of_image>' : `${p.grid[0]}*${p.grid[1]}`].forEach((t, i) => { S('rect', { x: GN.a + 10, y: top + 34 + i * 26, width: 110, height: 20, rx: 4, class: i === 2 ? 'chip special' : 'chip' }, gA); T(gA, GN.a + 16, top + 48 + i * 26, t, 'tok'); });
  const lx0 = GN.a + 132, lw = GN.aw - 144, pitch = lw / d.layers;
  for (let i = 0; i < d.layers; i++) S('rect', { x: lx0 + i * pitch + pitch * 0.22, y: top + 30, width: Math.max(1.4, pitch * 0.56), height: h - 76, rx: 1, class: 'dlayer' }, gA);
  T(gA, GN.a + GN.aw / 2, top + 18, `${d.layers} layers`, 'st-s', 'middle');
  T(gA, GN.a + GN.aw / 2, top + h - 18, j ? '× 2: with the prompt, and without' : '× 2: prompt, and a negative prompt', 'tiny', 'middle');
  line(svg, [[GN.a + GN.aw + 8, Y0], [GN.b - 8, Y0]], 'arr', 'main');
  smallBox(svg, { x: GN.b, y: Y0 - 60, w: GN.bw, h: 120, stepId: 'step', label: 'Head', hcls: 'main',
    lines: j ? [['image head', 'st-s'], ['4,096 → 4,096 → 16,384', 'tiny'], ['guidance:', 'tiny'], ['ℓu + 5 (ℓc − ℓu)', 'mlt'], ['sample one code', 'tiny']] : [['language-model head', 'st-s'], ['4,096 → 184,622', 'tiny'], ['only codes allowed,', 'tiny'], ['or end of line', 'tiny'], ['sample one code', 'tiny']] });
  line(svg, [[GN.b + GN.bw + 8, Y0], [GN.c - 8, Y0]], 'arr', 'grid');
  // C: the code grid as a mosaic of the example image
  const [rows, cols] = p.grid, cell = Math.min((GN.cw - 10) / (cols + (j ? 0 : 1)), 250 / rows), gw = cell * cols, gh = cell * rows, gx = GN.c, gy = Y0 - gh / 2;
  const gC = act(S('g', { 'data-h': 'grid' }, svg), 'grid', 'grid', 'Code grid');
  mosaic(gC, gx, gy, cell, sampleScene(p.pixels, cols, rows));
  if (!j) { S('rect', { x: gx + gw, y: gy, width: cell, height: gh, class: 'eol' }, gC); T(gC, gx + gw + cell / 2, gy - 5, 'EOL', 'tiny', 'middle'); }
  line(gC, [[gx + 4, gy + cell * 0.5], [gx + gw - 4, gy + cell * 0.5]], 'raster', null, 'mm-arrow');
  T(svg, GN.c + GN.cw / 2, Y0 + 145, `${fmt(p.steps)} steps, left to right, top to bottom`, 'dim', 'middle', 'grid');
  // the feedback path: each code becomes the next input
  const gF = act(S('g', { 'data-h': 'inner' }, svg), 'feed', 'step:feed', 'Feed back');
  line(gF, [[gx + gw / 2, gy + gh + 4], [gx + gw / 2, Y0 + 168], [GN.a + GN.aw / 2, Y0 + 168], [GN.a + GN.aw / 2, top + h + 4]], 'feedback', null, 'mm-arrow');
  T(gF, (GN.a + GN.aw / 2 + gx + gw / 2) / 2, Y0 + 162, j ? 'the code goes back in: 8-number embedding → aligner MLP → 4,096' : 'the code goes back in as an ordinary token', 'tiny', 'middle');
  line(svg, [[GN.c + GN.cw + 8, Y0], [GN.d - 8, Y0]], 'arr', 'dec');
  smallBox(svg, { x: GN.d, y: Y0 - 50, w: GN.dw, h: 100, stepId: 'decode', label: 'VQ decoder', hcls: 'dec',
    lines: [['VQ decoder', 'st-s'], [`code → ${m.codeDim} numbers`, 'tiny'], [`conv. upsampling × ${m.down}`, 'tiny'], [`${(m.params[0] / 1e6).toFixed(0)}M (with encoder)`, 'tiny']] });
  line(svg, [[GN.d + GN.dw + 8, Y0], [GN.e - 8, Y0]], 'arr', 'out');
  const gE = act(S('g', { 'data-h': 'out' }, svg), 'decode', 'step:decode', 'Image out');
  const sc = Math.min(GN.ew / p.pixels[1], 240 / p.pixels[0]), iw = p.pixels[1] * sc, ih = p.pixels[0] * sc;
  scene(gE, GN.e + (GN.ew - iw) / 2, Y0 - ih / 2, iw, ih, [0, 0, p.pixels[1], p.pixels[0]], { size: p.pixels, href: null });
  S('rect', { x: GN.e + (GN.ew - iw) / 2, y: Y0 - ih / 2, width: iw, height: ih, class: 'frame' }, gE);
  T(svg, GN.e + GN.ew / 2, Y0 + 145, 'the page’s example image, as a stand-in', 'note', 'middle', 'out');
  const yb = costLine(svg, GN.c, Y0 + 200, 'What one image takes', `${fmt(p.steps)} × 2 passes`, `${MM.fmtMiB(p.kv)} of keys and values by the last code`);
  finish(svg, yb);
}
function sampleScene([H, W], nx, ny) {     // the example scene drawn at the image's own shape, one colour per code
  const g = geom(W, H), out = [];
  for (let r = 0; r < ny; r++) { const row = []; for (let c = 0; c < nx; c++) row.push(sceneRGB((c + 0.5) * W / nx, (r + 0.5) * H / ny, W, H, g)); out.push(row); }
  return out;
}

/* ---------- all together ---------- */
const OM = { a: 20, aw: 210, b: 270, bw: 170, c: 480, cw: 440, d: 960, dw: 200, e: 1200, ew: 180 };
function drawOmni() {
  const svg = $('#mm'); svg.replaceChildren(); defs(svg);
  const m = mmodel(), v = val(), p = X().omniPlan(v), d = m.dec, vid = p.grids * p.perGrid;
  svg.setAttribute('aria-label', `Qwen3-Omni: a ${v} s clip becomes ${fmt(vid)} video and ${fmt(p.audioTokens)} audio tokens, interleaved by time.`);
  headers(svg, [[OM.a, 'Inputs', `a ${v}-s clip and a question`], [OM.b, 'Encoders', 'one per kind of input'], [OM.c, 'One sequence', `${fmt(p.order.length)} tokens on one clock`],
    [OM.d, 'Thinker', '48 layers · 30B MoE, 3B active'], [OM.e, 'Answer', 'text, and speech']]);
  const gI = act(S('g', { 'data-h': 'in' }, svg), 'inputs', 'step:inputs', 'Inputs');
  S('rect', { x: OM.a, y: Y0 - 120, width: OM.aw, height: 22, rx: 4, class: 'chip' }, gI); T(gI, OM.a + 8, Y0 - 105, 'What do you see and hear?', 'tok');
  for (let i = 0; i < 3; i++) frameThumb(gI, OM.a + i * 70, Y0 - 78, 66, 37, v * i / 2, v, 'omni');
  T(gI, OM.a, Y0 - 28, `video: 2 frames per second`, 'tiny');
  S('rect', { x: OM.a, y: Y0, width: OM.aw, height: 60, rx: 6, class: 'subbox' }, gI);
  waveform(gI, OM.a + 6, Y0 + 6, OM.aw - 12, 48, wavePeaks(v, 80));
  T(gI, OM.a, Y0 + 76, 'its sound: 16,000 samples per second', 'tiny');
  line(svg, [[OM.a + OM.aw + 8, Y0 - 60], [OM.b - 8, Y0 - 60]], 'arr', 'enc');
  line(svg, [[OM.a + OM.aw + 8, Y0 + 30], [OM.b - 8, Y0 + 30]], 'arr', 'enc');
  smallBox(svg, { x: OM.b, y: Y0 - 108, w: OM.bw, h: 96, stepId: 'encoders', label: 'Vision encoder', hcls: 'enc', lines: [['vision encoder', 'st-s'], ['27 layers (Qwen3-VL)', 'tiny'], [`${p.perGrid} tokens per pair`, 'tiny'], ['539M', 'tiny']] });
  smallBox(svg, { x: OM.b, y: Y0 - 2, w: OM.bw, h: 96, stepId: 'encoders', label: 'Audio encoder', hcls: 'enc', lines: [['audio encoder (AuT)', 'st-s'], ['32 layers, from scratch', 'tiny'], ['13 tokens per second', 'tiny'], ['648M', 'tiny']] });
  line(svg, [[OM.b + OM.bw + 8, Y0 - 60], [OM.c - 8, Y0 - 20]], 'arr', 'seq');
  line(svg, [[OM.b + OM.bw + 8, Y0 + 46], [OM.c - 8, Y0 + 10]], 'arr', 'seq');
  // C: the interleaved sequence, one block per second
  const gC = act(S('g', { 'data-h': 'seq' }, svg), 'clock', 'step:clock', 'Interleaved sequence');
  const secsShown = Math.min(v, 10), bw = (OM.cw - 10) / secsShown, sy = Y0 - 22;
  for (let s2 = 0; s2 < secsShown; s2++) {
    const x = OM.c + s2 * bw;
    S('rect', { x: x + 1, y: sy, width: bw * 0.62, height: 40, rx: 3, class: 'imgblock' }, gC);
    S('rect', { x: x + bw * 0.62 + 2, y: sy, width: bw * 0.38 - 4, height: 40, rx: 3, class: 'audblock' }, gC);
    if (bw > 36) { T(gC, x + bw * 0.31, sy + 25, 'V', 'tiny', 'middle'); T(gC, x + bw * 0.81, sy + 25, 'A', 'tiny', 'middle'); }
    T(gC, x + 1, sy + 58, String(s2 * 13), 'tiny');
  }
  if (v > secsShown) T(gC, OM.c + OM.cw, sy + 25, '…', 'st', 'end');
  T(gC, OM.c, sy - 10, `each second: a video grid (${p.perGrid} tokens), then 13 audio tokens`, 'tiny');
  T(gC, OM.c, sy + 74, 'time positions (13 per second)', 'tiny');
  T(svg, OM.c + OM.cw / 2, Y0 + 100, `${fmt(vid)} video + ${fmt(p.audioTokens)} audio tokens`, 'dim', 'middle', 'seq');
  line(svg, [[OM.c + OM.cw + 8, Y0], [OM.d - 8, Y0]], 'arr', 'lm');
  const gD = act(S('g', { 'data-h': 'lm' }, svg), 'think', 'step:think', 'Thinker');
  S('rect', { x: OM.d, y: Y0 - 80, width: OM.dw, height: 160, rx: 8, class: 'subbox' }, gD);
  for (let i = 0; i < 48; i++) S('rect', { x: OM.d + 10 + i * (OM.dw - 20) / 48, y: Y0 - 66, width: 2, height: 132, rx: 1, class: 'dlayer' }, gD);
  line(svg, [[OM.d + OM.dw + 8, Y0 - 30], [OM.e - 8, Y0 - 50]], 'arr', 'out');
  line(svg, [[OM.d + OM.dw + 8, Y0 + 30], [OM.e - 8, Y0 + 40]], 'arr', 'out');
  const gE = act(S('g', { 'data-h': 'out' }, svg), 'speak', 'step:speak', 'Answer');
  S('rect', { x: OM.e, y: Y0 - 64, width: OM.ew, height: 24, rx: 4, class: 'chip out' }, gE); T(gE, OM.e + 8, Y0 - 48, 'A sunny hill; birds sing.', 'tok');
  S('rect', { x: OM.e, y: Y0 + 14, width: OM.ew, height: 56, rx: 6, class: 'subbox' }, gE);
  waveform(gE, OM.e + 6, Y0 + 18, OM.ew - 12, 48, wavePeaks(3, 60, 2.1));
  T(gE, OM.e, Y0 + 86, 'Talker → Code2Wav (Speech out tab)', 'tiny');
  const cache = p.order.length * X().kvPerToken(d);
  const yb = costLine(svg, OM.c, Y0 + 140, 'The clip in the Thinker’s cache', MM.fmtMiB(cache), `${fmt(p.order.length)} tokens × 48 layers · ${(100 * p.order.length / d.ctx).toFixed(1)}% of the 65,536-token context`);
  finish(svg, yb);
}

/* ---------- pop-ups for the new views ---------- */
function popMedia(key) {
  const [kind, a] = key.split(':'), S2 = steps();
  if (kind === 'step') { const s = S2.find(x => x.id === a) || S2[0]; return [s.title, s.shape, s.text, s.tex]; }
  const m = mmodel(), i = Number(a);
  switch (kind) {
    case 'enc': { const s = S2.find(x => x.id === 'encode'); return [s.title, s.shape, s.text, s.tex]; }
    case 'elayer': return [`Encoder layer ${i + 1}`, `of ${st.mode === 'audio' ? 32 : m.enc.layers}`, 'Self-attention over all positions of the window (no causal mask), then a feed-forward network.', []];
    case 'dlayer': return [`Layer ${i + 1}`, `of ${m.dec.layers}`, 'Causal self-attention over text and media tokens, then a feed-forward network.', []];
    case 'tok': { const r = (m.template || [])[i] || ['', 'text']; return [r[0], r[1] === 'stamp' ? 'timestamp, as text' : r[1] === 'mark' ? 'special token' : 'text', r[1] === 'stamp' ? 'The time of the next grid, written out as ordinary text tokens.' : r[1] === 'mark' ? 'Marks where the media starts or ends.' : 'Ordinary text tokens (shortened here).', []]; }
    case 'media': { const s = S2.find(x => x.id === 'sequence'); return [s.title, s.shape, s.text, s.tex]; }
    case 'next': return ['Next token', 'text out', 'The model writes text, one token at a time; the media is read, never generated.', []];
    case 'grid': { const s = S2.find(x => x.id === (st.mode === 'imagegen' ? 'grid' : st.key === 'moshi' ? 'depth' : 'fill')); return [s.title, s.shape, s.text, s.tex]; }
  }
  return null;
}

/* ---------- up close ---------- */
function closeAudio(svg) {
  const q = st.key === 'qwen2audio', x0 = 170, cw = 54, y0 = 40, v = val(), p = X().audioPlan(st.key, v);
  const rows = [['spectrogram frames', 16, '10 ms each', 'mel'], ['encoder outputs', 8, '20 ms each (stride 2)', 'enc'], [q ? 'tokens (average of 2)' : 'tokens (4 stacked)', q ? 4 : 2, q ? '40 ms each' : '80 ms each', 'tokc']];
  rows.forEach(([lab, n, note, cls], r) => {
    const y = y0 + r * 52, w = 16 * cw / n;
    T(svg, x0 - 12, y + 18, lab, 'st-s', 'end');
    for (let i = 0; i < n; i++) S('rect', { x: x0 + i * w + 1, y, width: w - 2, height: 26, rx: 3, class: 'cell ' + cls }, svg);
    T(svg, x0 + 16 * cw + 12, y + 18, note, 'note');
    if (r > 0) for (let i = 0; i < n; i++) {
      const prev = rows[r - 1][1], per = prev / n, xa = x0 + i * w;
      S('path', { d: `M${xa + 2} ${y - 4} L${xa + w / 2} ${y - 1} L${xa + w - 2} ${y - 4}`, class: 'merge' }, svg);
      if (per > 1 && i === 0) { /* brackets show which cells combine */ }
    }
  });
  T(svg, x0, y0 + 3 * 52 + 6, '0 s', 'tiny'); T(svg, x0 + 16 * cw, y0 + 3 * 52 + 6, '0.16 s', 'tiny', 'end');
  if (!q) {        // the chunks of this recording
    const cy = y0 + 3 * 52 + 30, n = p.chunks, w = 16 * cw;
    T(svg, x0 - 12, cy + 14, `${n} chunk${n > 1 ? 's' : ''} of 30 s`, 'st-s', 'end');
    for (let i = 0; i < n; i++) S('rect', { x: x0 + i * w / n + 0.5, y: cy, width: Math.max(0.6, w / n - 1), height: 20, rx: n > 30 ? 0 : 3, class: 'cell enc' }, svg);
    const fill = (v % 30 || 30) / 30;
    if (fill < 1) S('rect', { x: x0 + (n - 1 + fill) * w / n, y: cy, width: (1 - fill) * w / n, height: 20, class: 'pad' }, svg);
    T(svg, x0 + w + 12, cy + 14, 'attention stays inside a chunk', 'note');
  }
}
function closeVideo(svg) {
  const q = st.key === 'qwen3vl', v = val(), cust = customVideo(), [H, W] = cust ? [cust.h, cust.w] : [720, 1280], p = X().videoPlan(st.key, v, H, W);
  const x0 = 60, w = 1280, y0 = 50;
  if (q) {
    const span = Math.min(v, 8), sc = w / span;
    T(svg, x0, y0 - 18, `the first ${X().secs(span)}`, 'st-s');
    S('line', { x1: x0, y1: y0, x2: x0 + w, y2: y0, class: 'axis' }, svg);
    for (let s2 = 0; s2 <= span; s2++) { S('line', { x1: x0 + s2 * sc, y1: y0 - 4, x2: x0 + s2 * sc, y2: y0 + 4, class: 'axis' }, svg); T(svg, x0 + s2 * sc, y0 + 18, `${s2} s`, 'tiny', 'middle'); }
    const fw = Math.min(70, sc * 0.4), fh = fw * p.frameSize[0] / p.frameSize[1];
    p.indices.forEach((idx, i) => {
      const t = idx / 30; if (t > span) return;
      frameThumb(svg, x0 + t * sc - fw / 2, y0 + 34 + (i % 2) * 8, fw, fh, t, v, 'video');
    });
    p.stamps.forEach((t, g) => {
      if (t > span) return;
      const xa = x0 + (p.indices[2 * g] / 30) * sc, xb = x0 + (p.indices[Math.min(2 * g + 1, p.frames - 1)] / 30) * sc;
      S('path', { d: `M${xa - fw / 2} ${y0 + 50 + fh} V${y0 + 58 + fh} H${xb + fw / 2} V${y0 + 50 + fh}`, class: 'merge' }, svg);
      T(svg, (xa + xb) / 2, y0 + 76 + fh, `<${t.toFixed(1)} seconds>`, 'tok', 'middle');
      T(svg, (xa + xb) / 2, y0 + 92 + fh, `${fmt(p.perGroup)} tokens`, 'tiny', 'middle');
    });
  } else {
    const sc = w / v;
    T(svg, x0, y0 - 18, `32 frames over ${X().secs(v)}`, 'st-s');
    S('line', { x1: x0, y1: y0, x2: x0 + w, y2: y0, class: 'axis' }, svg);
    for (let i = 0; i < 32; i++) S('rect', { x: x0 + (v * (i + 0.5) / 32) * sc - 1, y: y0 - 6, width: 2, height: 12, class: 'tick' }, svg);
    T(svg, x0, y0 + 20, '0 s', 'tiny'); T(svg, x0 + w, y0 + 20, X().secs(v), 'tiny', 'end');
    const g1 = 6, gy = y0 + 50;
    T(svg, x0, gy - 8, 'one frame: 27 × 27 vectors', 'tiny');
    for (let r = 0; r < 27; r++) for (let c = 0; c < 27; c++) S('rect', { x: x0 + c * g1, y: gy + r * g1, width: g1 - 0.8, height: g1 - 0.8, class: 'cell enc' }, svg);
    line(svg, [[x0 + 27 * g1 + 14, gy + 80], [x0 + 27 * g1 + 70, gy + 80]]); T(svg, x0 + 27 * g1 + 42, gy + 70, 'bilinear', 'tiny', 'middle');
    const x2 = x0 + 27 * g1 + 84, g2 = 27 * g1 / 14;
    T(svg, x2, gy - 8, '14 × 14 = 196 tokens', 'tiny');
    for (let r = 0; r < 14; r++) for (let c = 0; c < 14; c++) S('rect', { x: x2 + c * g2, y: gy + r * g2, width: g2 - 1, height: g2 - 1, class: 'cell tokc' }, svg);
    T(svg, x2 + 14 * g2 + 24, gy + 80, '× 32 frames, + 1 newline vector = 6,273 tokens', 'note');
  }
}
function closeSpeech(svg) {
  const q = st.key === 'qwen3omni', x0 = 150, y0 = 40;
  if (q) {
    const cols = 14, cw = 52, rh = 13;
    T(svg, x0 - 12, y0 - 10, 'codebook', 'tiny', 'end');
    T(svg, x0 - 40, y0 + 10, 'Talker', 'st-s', 'end');
    S('path', { d: `M${x0 - 34} ${y0 + rh + 2} H${x0 - 30} V${y0 + 16 * rh - 2} H${x0 - 34}`, class: 'merge' }, svg);
    T(svg, x0 - 40, y0 + 8.5 * rh + 4, 'code predictor', 'st-s', 'end');
    for (let r = 0; r < 16; r++) {
      T(svg, x0 - 12, y0 + r * rh + 10, String(r + 1), 'tiny', 'end');
      for (let c = 0; c < cols; c++) {
        S('rect', { x: x0 + c * cw + 1, y: y0 + r * rh + 1, width: cw - 2, height: rh - 2, rx: 2, class: 'code ' + (r === 0 ? 'main' : 'inner') }, svg);
      }
    }
    for (let c = 0; c < cols; c++) T(svg, x0 + c * cw + cw / 2, y0 + 16 * rh + 14, `${(c * 80)} ms`, 'tiny', 'middle');
    T(svg, x0 + cols * cw + 20, y0 + 30, 'per frame: 1 Talker pass,', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 46, 'then 15 code-predictor passes', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 80, 'all 16 codes → Code2Wav →', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 96, '1,920 samples (80 ms at 24 kHz)', 'note');
  } else {
    const cols = 12, cw = 52, rh = 11, del = X().moshiDelays(), groups = [['Moshi text', 1, 'txt', [0]], ['Moshi audio', 8, 'moshi', del], ['user audio', 8, 'user', del]];
    let y = y0;
    groups.forEach(([lab, n, cls, ds]) => {
      T(svg, x0 - 12, y + n * rh / 2 + 4, lab, 'st-s', 'end');
      for (let r = 0; r < n; r++) for (let c = 0; c < cols; c++) {
        if (c < ds[r]) continue;
        S('rect', { x: x0 + c * cw + 1, y: y + r * rh + 1, width: cw - 2, height: rh - 2, rx: 2, class: 'code ' + cls + (cls === 'moshi' && r === 0 ? ' sem' : '') + (c === 6 && cls !== 'user' ? ' now' : '') }, svg);
      }
      y += n * rh + 10;
    });
    for (let c = 0; c < cols; c++) T(svg, x0 + c * cw + cw / 2, y + 6, `step ${c + 1}`, 'tiny', 'middle');
    S('rect', { x: x0 + 6 * cw - 2, y: y0 - 6, width: cw + 4, height: y - y0 + 2, rx: 4, class: 'stepcol' }, svg);
    T(svg, x0 + cols * cw + 20, y0 + 20, 'step 7 (outlined): Helium reads all 17 streams', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 36, 'of steps 1–6, then the depth transformer writes', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 52, 'the text token and Moshi’s 8 codes, top to bottom', 'note');
    T(svg, x0 + cols * cw + 20, y0 + 84, 'codebook 1 (highlighted) is on time; 2–8 are one step late', 'note');
  }
}
function closeGen(svg) {
  const j = st.key === 'janus', p = X().genPlan(st.key, val()), [rows, cols] = p.grid;
  const cell = Math.min((j ? 216 : 270) / rows, 600 / (cols + 1)), x0 = 60, y0 = 30;
  mosaic(svg, x0, y0, cell, sampleScene(p.pixels, cols, rows));
  if (!j) { S('rect', { x: x0 + cols * cell, y: y0, width: cell, height: rows * cell, class: 'eol' }, svg); T(svg, x0 + cols * cell + cell / 2 + 4, y0 - 6, 'EOL', 'tiny', 'start'); }
  if (j) for (let c = 0; c <= cols; c += 1) S('line', { x1: x0 + c * cell, y1: y0, x2: x0 + c * cell, y2: y0 + rows * cell, class: 'pgrid' }, svg);
  const tx = x0 + (cols + (j ? 0 : 1)) * cell + 40;
  T(svg, tx, y0 + 16, `${rows} × ${cols} = ${fmt(p.codes)} codes${j ? '' : `, + ${rows} end-of-line tokens`}`, 'st-s');
  T(svg, tx, y0 + 40, `each code: one of ${fmt(MM.MEDIA.MODELS[st.key].codebook)} entries, ${MM.MEDIA.MODELS[st.key].down} × ${MM.MEDIA.MODELS[st.key].down} pixels`, 'note');
  T(svg, tx, y0 + 64, j ? 'guidance at every code: ℓ = ℓu + 5 (ℓc − ℓu)' : 'guidance at every code, against a negative prompt', 'note');
  T(svg, tx, y0 + 88, `${fmt(p.steps)} steps × 2 sequences`, 'note');
}
function closeOmni(svg) {
  const p = X().omniPlan(3), x0 = 60, y0 = 50;
  let x = x0; const runs = [];
  for (const ch of p.order) { if (runs.length && runs[runs.length - 1][0] === ch) runs[runs.length - 1][1]++; else runs.push([ch, 1]); }
  let pos = 0, vgrid = 0;
  runs.forEach(([ch, n]) => {
    const w = ch === 'v' ? 230 : 13 * 12;
    S('rect', { x, y: y0, width: w - 4, height: 40, rx: 4, class: ch === 'v' ? 'imgblock' : 'audblock' }, svg);
    T(svg, x + 8, y0 + 17, ch === 'v' ? `video grid ${vgrid + 1}` : 'audio', 'tok');
    T(svg, x + 8, y0 + 33, `× ${n}`, 'tiny');
    T(svg, x + 2, y0 + 58, ch === 'v' ? `t = ${vgrid * 13}` : `t = ${pos} … ${pos + n - 1}`, 'tiny');
    if (ch === 'v') vgrid++; else pos += n;
    x += w;
  });
  T(svg, x0, y0 - 14, 'three seconds of a clip, in sequence order', 'st-s');
  T(svg, x0, y0 + 90, 'Each audio token takes one time position (13 per second); a video grid starting at second s takes position 13 s. Rows and columns of a grid get their own positions too (3D, as in Qwen3-VL).', 'note');
}

/* ---------- cost across the models of a modality ---------- */
function renderCostMedia() {
  const mode = st.mode, models = X().MODE[mode].models, v = val(), cust = customVideo(), secs = X().secs;
  const rows = models.map(k => {
    const m = X().MODELS[k];
    if (mode === 'audio') { const p = X().audioPlan(k, v), c = p.tokens * X().kvPerToken(m.dec); return [m.name, `${fmt(p.tokens)} tokens${p.dropped > 0 ? ' (first 30 s)' : ''}`, c, MM.fmtMiB(c), k]; }
    if (mode === 'video') { const [H, W] = cust ? [cust.h, cust.w] : [720, 1280], p = X().videoPlan(k, v, H, W), c = p.tokens * X().kvPerToken(m.dec); return [m.name, `${fmt(p.tokens)} tokens`, c, MM.fmtMiB(c), k]; }
    if (mode === 'speech') { const p = X().speechPlan(k, v); return [m.name, `${fmt(p.mainSteps)} big + ${fmt(p.innerSteps)} small passes`, p.kv, MM.fmtMiB(p.kv), k]; }
    if (mode === 'imagegen') { const p = X().genPlan(k, v); return [m.name, `${fmt(p.steps)} steps × 2`, p.kv, MM.fmtMiB(p.kv), k]; }
    return null;
  }).filter(Boolean);
  if (mode === 'omni') {
    const p = X().omniPlan(v), per = X().kvPerToken(X().MODELS.omni.dec);
    rows.push(['Video', `${fmt(p.grids * p.perGrid)} tokens`, p.grids * p.perGrid * per, MM.fmtMiB(p.grids * p.perGrid * per), 'omni'], ['Audio', `${fmt(p.audioTokens)} tokens`, p.audioTokens * per, MM.fmtMiB(p.audioTokens * per), 'omni']);
  }
  const max = Math.max(...rows.map(r => r[2]));
  $('#mm-cost-sub').textContent = { audio: `For ${secs(v)} of audio, at 16 bits: keys and values of every audio token in every layer.`, video: `For a ${secs(v)} video, at 16 bits: keys and values of every video token in every layer.`,
    speech: `For ${secs(v)} of speech: passes of the network that runs once per frame (“big”) and of the one that fills in each code (“small”); the cache is the big network’s.`,
    imagegen: 'For one image: forward passes of the language model, and the keys and values held when the last code is written.', omni: `For a ${secs(v)} clip with sound, in the Thinker's cache.` }[mode];
  $('#mm-cost-rows').innerHTML = rows.map(([name, f, val2, txt, k]) => `<button type="button" class="crow mm${k === st.key && mode !== 'omni' ? ' active' : ''}" data-k="${k}" aria-label="${esc(name)}: ${txt}.">
    <span class="c-l">${esc(name)}</span><span class="c-f">${esc(f)}</span><span class="c-bar"><i style="width:${(100 * val2 / max).toFixed(2)}%"></i></span><span class="c-v">${txt}</span></button>`).join('');
  document.querySelectorAll('#mm-cost-rows .crow').forEach(b => b.addEventListener('click', () => setModel(b.dataset.k)));
}

/* ---------- header, steps, controls ---------- */
const MEDIA_NOTE = { images: 'Hover any part of the diagram for details; try another image size, or your own image.',
  audio: 'Hover any part of the diagram for details; try another length, or your own recording.',
  video: 'Hover any part of the diagram for details; try another length, or your own video.',
  speech: 'Hover any part of the diagram for details; try a longer answer.', imagegen: 'Hover any part of the diagram for details; try another aspect ratio.',
  omni: 'Hover any part of the diagram for details; try a longer clip.' };
function tabsFor(mode) {
  const keys = X().MODE[mode].models;
  return keys.map(k => (mode === 'images' ? { k, b: { llava: 'Projector', qwen: 'Native size', llama: 'Cross-attention', gemma: 'Pooling' }[k], s: { llava: 'LLaVA-1.5', qwen: 'Qwen3-VL', llama: 'Llama 3.2 Vision', gemma: 'Gemma 3' }[k] }
    : { k, b: X().MODELS[k].tab, s: X().MODELS[k].name.replace('-30B-A3B', '').replace(' (moshiko)', '') }));
}
function renderTabs() {
  const tabs = $('#mm-tabs');
  tabs.innerHTML = tabsFor(st.mode).map(t => `<button class="tab" role="tab" data-k="${t.k}" type="button"><b>${esc(t.b)}</b><span>${esc(t.s)}</span></button>`).join('');
  tabs.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setModel(t.dataset.k)));
  tabs.setAttribute('aria-label', X().MODE[st.mode].title);
  document.querySelectorAll('#mm-modes button').forEach(b => { const on = b.dataset.mode === st.mode; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
}
function renderControls() {
  const box = $('#mm-size'), mode = st.mode;
  if (mode === 'images') {
    $('#mm-size-l').textContent = 'Image size (width × height)';
    box.setAttribute('aria-label', 'Image size (width × height)');
    box.innerHTML = MM.SIZES.map(([h, w], i) => `<button type="button" data-i="${i}">${w} × ${h}</button>`).join('') +
      '<button type="button" id="mm-mine">Your image…</button><input type="file" id="mm-file" accept="image/*" hidden>';
    box.querySelectorAll('button[data-i]').forEach(b => b.addEventListener('click', () => setSize(Number(b.dataset.i))));
    $('#mm-mine').addEventListener('click', () => (st.custom && st.sel !== 'custom' ? setSize('custom') : $('#mm-file').click()));
  } else {
    const c = X().CONTROLS[mode];
    $('#mm-size-l').textContent = c.label;
    box.setAttribute('aria-label', c.label);
    box.innerHTML = c.values.map((v, i) => `<button type="button" data-i="${i}">${esc(c.fmt(v))}</button>`).join('') +
      (c.file ? `<button type="button" id="mm-mine">${esc(c.fileLabel)}</button><input type="file" id="mm-file" accept="${c.file}" hidden>` : '');
    box.querySelectorAll('button[data-i]').forEach(b => b.addEventListener('click', () => { st.vals[mode] = Number(b.dataset.i); render(); }));
    if (c.file) $('#mm-mine').addEventListener('click', () => (st.media[mode] && st.vals[mode] !== 'custom' ? (st.vals[mode] = 'custom', render()) : $('#mm-file').click()));
  }
  if ($('#mm-file')) $('#mm-file').addEventListener('change', e => loadFile(e.target.files[0]));
}
function renderHeader() {
  const mode = st.mode;
  document.querySelectorAll('#mm-tabs .tab').forEach(t => { const on = t.dataset.k === st.key; t.classList.toggle('active', on); t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; });
  const [wt, ws] = X().WHOLE[mode];
  $('#mm-whole-title').textContent = wt; $('#mm-whole-sub').textContent = ws;
  $('#mm-cost-title').textContent = X().COST_TITLE[mode];
  $('#mm-why').innerHTML = X().WHY[mode].map(t => `<p>${t}</p>`).join('');
  const models = X().MODE[mode].models.map(k => (mode === 'images' ? MM.VLMS[k] : X().MODELS[k]));
  $('#mm-foot').innerHTML = X().FOOT[mode].map(t => `<p>${t}</p>`).join('') +
    `<p>Configs: ${[...new Map(models.map(m => [m.hf, m])).values()].map(m => `<a href="https://huggingface.co/${m.hf}" target="_blank" rel="noopener noreferrer">${esc(m.name)}</a>`).join(', ')}.</p>`;
  const src = mode === 'images' ? MM.SOURCES : X().SOURCES[mode];
  $('#mm-sources').innerHTML = src.map(r => `<li>${esc(r.by)} (${r.year}). <a href="${r.url}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>. <i>${esc(r.in)}</i>.</li>`).join('');
  let stats;
  if (mode === 'images') {
    const m = model(), p = plan();
    $('#mm-title').textContent = { llava: 'A projector into the token stream', qwen: 'Native resolution, 3D positions', llama: 'Gated cross-attention', gemma: 'A fixed budget of 256 tokens' }[st.key];
    $('#mm-lead').textContent = MM.LEAD[st.key];
    stats = [['Model', `<a href="https://huggingface.co/${m.hf}" target="_blank" rel="noopener noreferrer">${esc(m.name)}</a>`],
      ['Parameters', `${(m.total / 1e9).toFixed(2)}B`], ['Vision encoder', `${(m.enc.params / 1e6).toFixed(0)}M`], ['Connector', `${(m.conn.params / 1e6).toFixed(1)}M`],
      ['Language model', `${(m.dec.params / 1e9).toFixed(2)}B`],
      ['Image tokens here', st.key === 'llama' ? '1 (+ 6,404 beside)' : fmt(p.tokens)], ['Context', `${fmt(m.dec.ctx)} tokens`]];
  } else {
    const m = mmodel(), sm = X().summary(st.key, val(), customVideo()), total = m.params.reduce((a, b) => a + b, 0);
    const size = n => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : n === 0 ? '0 (shared)' : fmt(n));
    $('#mm-title').textContent = X().TITLE[st.key];
    $('#mm-lead').textContent = X().LEAD[st.key];
    stats = [['Model', `<a href="https://huggingface.co/${m.hf}" target="_blank" rel="noopener noreferrer">${esc(m.name)}</a>`], ['Parameters', size(total)],
      ...m.partNames.map((n, i) => [n, size(m.params[i])]), [sm.tokensLabel, fmt(sm.tokens)]];
    if (sm.ctx) stats.push(['Context', `${fmt(sm.ctx)} tokens`]);
  }
  $('#mm-stats').innerHTML = stats.map(([a, b]) => `<div><dt>${esc(a)}</dt><dd>${b}</dd></div>`).join('');
  const sel = mode === 'images' ? st.sel : st.vals[mode];
  document.querySelectorAll('#mm-size button[data-i]').forEach(b => { const on = String(sel) === b.dataset.i; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
  const mine = $('#mm-mine');
  if (mine) {
    const c = mode === 'images' ? st.custom : st.media[mode];
    mine.textContent = !c ? (mode === 'images' ? 'Your image…' : X().CONTROLS[mode].fileLabel)
      : mode === 'images' ? `Yours · ${c.w} × ${c.h}` : mode === 'audio' ? `Yours · ${X().secs(c.seconds)}` : `Yours · ${X().secs(c.seconds)}, ${c.w} × ${c.h}`;
    mine.classList.toggle('active', sel === 'custom'); mine.setAttribute('aria-pressed', String(sel === 'custom'));
  }
}
function renderSteps() {
  const S2 = steps(), chips = $('#mm-chips');
  chips.innerHTML = [{ chip: 'Overview' }, ...S2].map((s, idx) => {
    const i = idx - 1, on = i === st.step;
    return `<li><button type="button" class="chipbtn${on ? ' active' : ''}" data-i="${i}" aria-pressed="${on}">${i < 0 ? '' : `<span class="n">${idx}</span>`}${esc(s.chip)}</button></li>`;
  }).join('');
  chips.querySelectorAll('.chipbtn').forEach(b => b.addEventListener('click', () => setStep(Number(b.dataset.i), true)));
  const s = S2[st.step], card = $('#mm-card');
  if (!s) {
    const ov = st.mode === 'images' ? MM.overview(st.key, ...hw()) : { tex: S2.slice(0, 4).map(x => x.tex[0]), text: X().LEAD[st.key] };
    const head = st.mode === 'images' ? 'From pixels to tokens in four lines' : `${X().WHOLE[st.mode][0]}, step by step`;
    card.innerHTML = `<div class="card-math">${ov.tex.map(math).join('')}</div>
      <div class="card-text"><h3>${esc(head)}</h3><p>${fmtText(ov.text)}</p>
      <p class="hint-l">${esc(MEDIA_NOTE[st.mode])}</p>
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
function draw() { ({ images: drawImages, audio: drawAudio, video: drawVideo, speech: drawSpeech, imagegen: drawGen, omni: drawOmni })[st.mode](); }
function popContent(key) { return st.mode === 'images' ? popImages(key) : popMedia(key); }
function renderCost() { if (st.mode === 'images') renderCostImages(); else renderCostMedia(); }
function render() { renderHeader(); draw(); renderSteps(); drawClose(); renderCost(); }
const path = () => `${st.mode}/${st.key}`;
function setMode(mode, fromHash) {
  if (!X().MODE[mode]) return;
  st.mode = mode; st.key = st.keys[mode]; st.step = -1;
  renderTabs(); renderControls();
  if (!fromHash) { try { history.replaceState(null, '', `#multimodal/${path()}`); } catch (e) { /* sandboxed */ } }
  render();
}
function setModel(k, fromHash) {
  if (!X().MODE[st.mode].models.includes(k)) return;
  st.key = st.keys[st.mode] = k;
  if (!fromHash) { try { history.replaceState(null, '', `#multimodal/${path()}`); } catch (e) { /* sandboxed */ } }
  render();
}
function setSize(sel) { st.sel = sel; render(); }
function loadFile(file) {
  if (!file) return;
  if (st.mode === 'images' && /^image\//.test(file.type)) {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => { st.custom = { url: reader.result, img, w: img.naturalWidth, h: img.naturalHeight, name: file.name }; setSize('custom'); };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  } else if (st.mode === 'audio' && /^audio\/|^video\//.test(file.type)) {
    file.arrayBuffer().then(buf => {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      return new Ctx().decodeAudioData(buf);
    }).then(audio => {
      st.media.audio = { seconds: audio.duration, rate: audio.sampleRate, data: audio.getChannelData(0), name: file.name };
      st.vals.audio = 'custom'; render();
    }).catch(() => { $('#mm-size-l').textContent = 'That file could not be decoded as audio.'; });
  } else if (st.mode === 'video' && /^video\//.test(file.type)) {
    const url = URL.createObjectURL(file), el = document.createElement('video');
    el.muted = true; el.preload = 'auto'; el.src = url;
    el.addEventListener('loadedmetadata', async () => {
      const media = { seconds: el.duration, w: el.videoWidth, h: el.videoHeight, frames: [], name: file.name };
      const c = document.createElement('canvas'); c.width = 160; c.height = Math.max(1, Math.round(160 * el.videoHeight / el.videoWidth));
      for (let i = 0; i < 12; i++) {             // a few frames for the pictures; everything stays in this tab
        const t = el.duration * (i + 0.5) / 12;
        await new Promise(res => { el.addEventListener('seeked', res, { once: true }); el.currentTime = t; });
        c.getContext('2d').drawImage(el, 0, 0, c.width, c.height);
        media.frames.push({ t, url: c.toDataURL('image/jpeg', 0.7) });
      }
      URL.revokeObjectURL(url);
      st.media.video = media; st.vals.video = 'custom'; render();
    }, { once: true });
    el.addEventListener('error', () => { $('#mm-size-l').textContent = 'That file could not be read as a video.'; }, { once: true });
  }
}

function init() {
  U = window.LABUI; ({ S, esc, fmtText, math } = U);
  document.querySelectorAll('#mm-modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  const fig = $('#mm-fig');
  fig.addEventListener('dragover', e => { e.preventDefault(); });
  fig.addEventListener('drop', e => { e.preventDefault(); loadFile(e.dataTransfer.files[0]); });
  $('#mm-tabs').addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    const keys = X().MODE[st.mode].models, i = keys.indexOf(st.key), k = keys[(i + (e.key === 'ArrowRight' ? 1 : keys.length - 1)) % keys.length];
    setModel(k); document.querySelector(`#mm-tabs .tab[data-k="${k}"]`).focus();
  });
  document.addEventListener('keydown', e => {
    if ($('.page[data-page="multimodal"]').hidden) return;
    if (e.target.closest && e.target.closest('#mm-tabs, #mm-modes, input, select, textarea')) return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); setStep(st.step + 1, true); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setStep(st.step - 1, true); }
    else if (e.key === 'Escape') setStep(-1, false);
    else if (/^[1-9]$/.test(e.key)) { const k = X().MODE[st.mode].models[Number(e.key) - 1]; if (k) setModel(k); }
  });
  U.attachPopups(fig, $('#mm-pop'), popContent);
  st.ready = true;
}
window.MMAPP = {
  key: path,
  show(sub) {
    const first = !st.ready;
    if (first) init();
    const [a, b] = (sub || '').split('/');
    if (X().MODE[a]) { st.mode = a; st.key = st.keys[a] = X().MODE[a].models.includes(b) ? b : st.keys[a]; }
    else if (MM.VLMS[a]) { st.mode = 'images'; st.key = st.keys.images = a; }     // older links: #multimodal/qwen
    else st.key = st.keys[st.mode];
    renderTabs(); renderControls(); render();
  },
};
})();
