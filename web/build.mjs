// Build the single-file Attention Explorer.
//   node build.mjs <out.html>
// 1. Renders every TeX string from src/content.js with Temml, to check it and to collect the glyphs it uses.
// 2. Subsets Latin Modern Math to those glyphs plus what the pop-ups can need (scripts/subset_font.py).
// 3. Inlines content.js, mm-content.js, Temml (which typesets all math at runtime), mm-app.js, app.js and the font
//    into src/template.html.
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const temml = require('temml');
const LAB = require(join(here, 'src/content.js'));
const MM = require(join(here, 'src/mm-content.js'));
const out = process.argv[2] || join(here, 'dist/attention.html');
// Any OpenType font with a MATH table works; Latin Modern Math ships with TeX Live
// (or download latinmodern-math.otf from GUST) — point MATH_FONT at it.
const fontPath = process.env.MATH_FONT || '/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf';

const table = {};
const chars = new Set();
for (const tex of [...LAB.allTex(), ...MM.allTex()]) {
  const html = temml.renderToString(tex, { displayMode: false, wrap: 'tex', throwOnError: true });
  if (/merror/.test(html)) throw new Error('TeX error in: ' + tex);
  table[tex] = html;
  const text = html.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  for (const c of text) chars.add(c);
}
// Glyphs used outside MathML: diagram labels and <var> in prose use math italic letters.
for (let i = 0; i < 26; i++) { chars.add(String.fromCodePoint(0x1D434 + i)); chars.add(String.fromCodePoint(0x1D44E + i)); }
chars.add('\u210E');
for (let c = 0x20; c < 0x7f; c++) chars.add(String.fromCharCode(c));
for (const c of '⊤·×÷√−∞≤⇒←ℝ⌈⌉⋯∈\u0302') chars.add(c);
// Pop-ups typeset TeX at runtime (values change with the controls), so keep everything they can use:
// upright and italic Greek (single-letter <mi> is drawn in math italic), operators, arrows, accents.
for (let c = 0x391; c <= 0x3C9; c++) chars.add(String.fromCodePoint(c));
for (let c = 0x1D6E2; c <= 0x1D71B; c++) chars.add(String.fromCodePoint(c));
for (const c of '−±∓·⋅×÷√∞≤≥≈≠∈∉∑∏⊙⊗∼∝←→⇒↦⌈⌉⌊⌋‖∣⊤′″…⋯ℝℓ∂∇∅∪∩∘∗°ˆ˜¯‾~⁡⁢⁣{}[]()|%–—“”‘’') chars.add(c);

const tmp = mkdtempSync(join(tmpdir(), 'lab-font-'));
const charFile = join(tmp, 'chars.txt');
writeFileSync(charFile, [...chars].join(''), 'utf8');
if (process.env.LAB_FONT_CHARS) writeFileSync(process.env.LAB_FONT_CHARS, [...chars].join(''), 'utf8');  // for a glyph-coverage check
const fontB64 = execFileSync('python3', [join(here, 'scripts/subset_font.py'), fontPath, charFile], { maxBuffer: 1 << 26 }).toString().trim();

const fontCss = `@font-face { font-family: "LabMath"; src: url(data:font/woff2;base64,${fontB64}) format("woff2"); font-display: block; }
/* Subset of Latin Modern Math (GUST Font License), renamed. MathML by Temml (MIT). */
math { math-style: normal; font-style: normal; font-weight: normal; line-height: normal; text-indent: 0; letter-spacing: normal; direction: ltr; font-feature-settings: "dtls" off; display: inline-flex; flex-wrap: wrap; align-items: baseline; row-gap: .3em; }
math * { border-color: currentColor; }
math > mrow { padding: .3ex 0; }
mfrac > :nth-child(2), msqrt, mover > :first-child { math-shift: compact; }
@supports (not (-moz-appearance: none)) { .tml-sml-pad { padding-left: .05em; } .tml-med-pad { padding-left: .1em; } .tml-lrg-pad { padding-left: .15em; } }
.tml-right { text-align: right; } .tml-left { text-align: left; }
@-moz-document url-prefix() { math { display: inline; } math > mrow { padding: 0; } }`;

const guard = s => s.replace(/<\/script/gi, '<\\/script');
let html = readFileSync(join(here, 'src/template.html'), 'utf8');
html = html
  .replace('/*__MATH_FONT__*/', () => fontCss)
  .replace('/*__CONTENT_JS__*/', () => guard(readFileSync(join(here, 'src/content.js'), 'utf8')))
  .replace('/*__MM_CONTENT_JS__*/', () => guard(readFileSync(join(here, 'src/mm-content.js'), 'utf8')))
  .replace('/*__MM_APP_JS__*/', () => guard(readFileSync(join(here, 'src/mm-app.js'), 'utf8')))
  .replace('/*__APP_JS__*/', () => guard(readFileSync(join(here, 'src/app.js'), 'utf8')))
  .replace('/*__RETRIEVAL_CONTENT_JS__*/', () => guard(readFileSync(join(here, 'src/retrieval-content.js'), 'utf8')))
  .replace('/*__RETRIEVAL_APP_JS__*/', () => guard(readFileSync(join(here, 'src/retrieval-app.js'), 'utf8')))
  // Temml renders the pop-up equations at runtime; its MIT license header is kept in the minified file.
  .replace('/*__TEMML_JS__*/', () => guard(readFileSync(require.resolve('temml/dist/temml.min.js'), 'utf8')))
  .replace('/*__TOKENS__*/null', () => readFileSync(join(here, 'src/tokens.json'), 'utf8').trim().replace(/</g, '\\u003c'))
  // Machine-readable copy of the dimensions, checked against architecture_specs.py by the tests.
  .replace('__SPECS_JSON__', () => JSON.stringify(Object.fromEntries(LAB.ORDER.map(k => {
    const m = LAB.MODELS[k];
    return [k, { model: m.model, hf: m.hf, d: m.d, H: m.H, G: m.G, dh: m.dh, dr: m.dr ?? null, dv: m.dv ?? null,
      dc: m.dc ?? null, L: m.L, ctx: m.ctx, cache_per_token_per_layer: LAB.cacheValues(m),
      V: m.V, dff: m.dff, tied: m.tied, experts: m.experts ?? null, active: m.active ?? null,
      shared: m.shared ?? null, dExpert: m.dExpert ?? null, firstDense: m.firstDense ?? null,
      dq: m.dq ?? null, index: m.index ?? null, lin: m.lin ?? null, interval: m.interval ?? null,
      ropeDim: m.ropeDim ?? null, thetaN: m.thetaN ?? null, cache_layers: LAB.cacheLayers(m) }];
  }))).replace(/</g, '\\u003c'));
for (const marker of ['__MATH_FONT__', '__CONTENT_JS__', '__MM_CONTENT_JS__', '__MM_APP_JS__', '__APP_JS__', '__RETRIEVAL_CONTENT_JS__', '__RETRIEVAL_APP_JS__', '__SPECS_JSON__', '__TOKENS__', '__TEMML_JS__'])
  if (html.includes(marker)) throw new Error('unfilled marker ' + marker);
writeFileSync(out, html, 'utf8');
console.log(`wrote ${out}: ${(html.length / 1024).toFixed(0)} KiB, ${Object.keys(table).length} equations, font ${(fontB64.length * 3 / 4 / 1024).toFixed(0)} KiB`);
