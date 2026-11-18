/* Multimodal page — data, arithmetic and text. No DOM.
 * Every number comes from the models' config.json / preprocessor_config.json; the resize rules follow
 * the transformers image processors. llm_lab/reference/vision.py has the same rules in Python, and
 * tests/test_explorer.py checks that both give the same plans and costs. */
(function (root) {
'use strict';

const fmt = n => Math.round(n).toLocaleString('en-US');
const mib = values => values * 2 / 2 ** 20;                         // 16-bit values → MiB
const fmtMiB = v => { const m = mib(v); return m >= 100 ? `${fmt(m)} MiB` : `${m.toFixed(m >= 10 ? 1 : 2).replace(/\.?0+$/, '')} MiB`; };

const VLMS = {
  llava: {
    key: 'llava', tab: 'Projector', name: 'LLaVA-1.5-7B', hf: 'llava-hf/llava-1.5-7b-hf', kind: 'projector', year: 2023,
    enc: { name: 'CLIP ViT-L/14-336', layers: 24, d: 1024, heads: 16, ff: 4096, patch: 14, size: 336, params: 303507456,
      taps: [22], tapNote: 'layer 23 of 24', positions: 'learned, one per patch and one for the class token' },
    conn: { name: 'MLP projector', params: 20979712 },
    dec: { name: 'Vicuna-7B v1.5', layers: 32, d: 4096, heads: 32, kv: 32, dh: 128, ff: 11008, V: 32064, ctx: 4096, params: 6738939904, cross: [] },
    total: 7063427072, imageToken: '<image>',
    template: [['USER:', 'text'], ['<image>', 'img'], ['Describe the image.', 'text'], ['ASSISTANT:', 'text']],
  },
  qwen: {
    key: 'qwen', tab: 'Native size', name: 'Qwen3-VL-8B', hf: 'Qwen/Qwen3-VL-8B-Instruct', kind: 'native', year: 2025,
    enc: { name: 'Qwen3-VL ViT', layers: 27, d: 1152, heads: 16, ff: 4304, patch: 16, size: null, params: 415891440,
      taps: [8, 16, 24, 26], tapNote: 'last layer, plus layers 9, 17 and 25 for DeepStack', positions: '2D rotary, plus learned positions interpolated from a 48 × 48 grid' },
    conn: { name: '2 × 2 merger', params: 160496896 },
    dec: { name: 'Qwen3-8B', layers: 36, d: 4096, heads: 32, kv: 8, dh: 128, ff: 12288, V: 151936, ctx: 262144, params: 8190735360, cross: [], deepstack: [0, 1, 2] },
    total: 8767123696, imageToken: '<|image_pad|>',
    template: [['<|im_start|>user', 'text'], ['<|vision_start|>', 'mark'], ['<|image_pad|>', 'img'], ['<|vision_end|>', 'mark'], ['Describe the image.', 'text'], ['…<|im_start|>assistant', 'text']],
  },
  llama: {
    key: 'llama', tab: 'Cross-attention', name: 'Llama 3.2 11B Vision', hf: 'meta-llama/Llama-3.2-11B-Vision-Instruct', kind: 'cross', year: 2024,
    enc: { name: 'Llama 3.2 ViT-H/14', layers: 32, global: 8, d: 1280, heads: 16, ff: 5120, patch: 14, size: 560, params: 863567379,
      taps: [3, 7, 15, 23, 30, 39], tapNote: 'layers 4, 8, 16, 24, 31 and the last global layer, side by side', positions: 'learned per patch, plus learned per tile layout' },
    conn: { name: 'linear projector', params: 31461376 },
    dec: { name: 'Llama 3.1 8B + 8 cross-attention layers', layers: 40, d: 4096, heads: 32, kv: 8, dh: 128, ff: 14336, V: 128256, ctx: 131072, params: 9775192080,
      cross: [3, 8, 13, 18, 23, 28, 33, 38] },
    total: 10670220835, imageToken: '<|image|>',
    template: [['<|begin_of_text|> …user…', 'text'], ['<|image|>', 'mark'], ['Describe the image.', 'text'], ['<|eot_id|> …assistant…', 'text']],
  },
  gemma: {
    key: 'gemma', tab: 'Pooling', name: 'Gemma 3 4B', hf: 'google/gemma-3-4b-it', kind: 'pooled', year: 2025,
    enc: { name: 'SigLIP So400m/14', layers: 27, d: 1152, heads: 16, ff: 4304, patch: 14, size: 896, params: 416866032,
      taps: [26], tapNote: 'last layer', positions: 'learned, one per patch' },
    conn: { name: 'pool + linear', params: 2950272 },
    dec: { name: 'Gemma 3 4B', layers: 34, d: 2560, heads: 8, kv: 4, dh: 256, ff: 10240, V: 262208, ctx: 131072, params: 3880263168, cross: [],
      window: 1024, globalEvery: 6 },
    total: 4300079472, imageToken: '<image_soft_token>',
    template: [['<start_of_turn>user', 'text'], ['<start_of_image>', 'mark'], ['<image_soft_token>', 'img'], ['<end_of_image>', 'mark'], ['Describe the image.', 'text'], ['…<start_of_turn>model', 'text']],
  },
};
const ORDER = ['llava', 'qwen', 'llama', 'gemma'];
const SIZES = [[336, 336], [480, 640], [768, 1024], [1080, 1920], [1200, 800]];   // [height, width]
const QWEN = { merge: 2, minPixels: 65536, maxPixels: 16777216 };
const LLAMA_TILES = 4;

/* ---- the resize rules (same as llm_lab/reference/vision.py) ---- */
function pyRound(x) {           // Python's round(): halves go to the even neighbour
  const f = Math.floor(x), d = x - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}
function smartResize(h, w, factor = 32, minPixels = QWEN.minPixels, maxPixels = QWEN.maxPixels) {
  let hb = pyRound(h / factor) * factor, wb = pyRound(w / factor) * factor;
  if (hb * wb > maxPixels) {
    const beta = Math.sqrt(h * w / maxPixels);
    hb = Math.max(factor, Math.floor(h / beta / factor) * factor);
    wb = Math.max(factor, Math.floor(w / beta / factor) * factor);
  } else if (hb * wb < minPixels) {
    const beta = Math.sqrt(minPixels / (h * w));
    hb = Math.ceil(h * beta / factor) * factor;
    wb = Math.ceil(w * beta / factor) * factor;
  }
  return [hb, wb];
}
function tileArrangements(max = LLAMA_TILES) {
  const out = [];
  for (let a = 1; a <= max; a++) for (let b = 1; b <= max; b++) if (a * b <= max) out.push([a, b]);
  return out;
}
function tiledCanvas(h, w, max = LLAMA_TILES, tile = 560) {
  const canvases = tileArrangements(max).map(([a, b]) => [a * tile, b * tile]);
  const scales = canvases.map(([ch, cw]) => Math.min(ch / h, cw / w));
  const up = scales.filter(s => s >= 1);
  const best = up.length ? Math.min(...up) : Math.max(...scales.filter(s => s < 1));
  return canvases.filter((c, i) => scales[i] === best).sort((a, b) => a[0] * a[1] - b[0] * b[1])[0];
}
function fitToCanvas(h, w, ch, cw, tile = 560) {
  const th = Math.min(Math.max(h, tile), ch), tw = Math.min(Math.max(w, tile), cw);
  if (tw / w < th / h) return [Math.min(Math.floor(h * tw / w) || 1, th), tw];
  return [th, Math.min(Math.floor(w * th / h) || 1, tw)];
}
/* What one image becomes: the pixels the encoder sees, its patch grid, and what the decoder reads. */
function imagePlan(key, h, w) {
  const m = VLMS[key], p = m.enc.patch;
  if (key === 'llava') {
    const s = m.enc.size, g = s / p;
    const side = Math.min(h, w);                                   // the centre square that survives the crop
    return { resized: [s, s], canvas: [s, s], crop: [side, side], patches: [g, g], tiles: 1, encoderPositions: g * g + 1, tokens: 576, cross: 0 };
  }
  if (key === 'qwen') {
    const [rh, rw] = smartResize(h, w, p * QWEN.merge);
    const gh = rh / p, gw = rw / p;
    return { resized: [rh, rw], canvas: [rh, rw], patches: [gh, gw], tiles: 1, encoderPositions: gh * gw, tokens: gh * gw / 4, cross: 0 };
  }
  if (key === 'llama') {
    const tile = m.enc.size, [ch, cw] = tiledCanvas(h, w, LLAMA_TILES, tile), [rh, rw] = fitToCanvas(h, w, ch, cw, tile);
    const g = tile / p, perTile = g * g + 1;
    return { resized: [rh, rw], canvas: [ch, cw], patches: [g, g], tiles: (ch / tile) * (cw / tile), tileGrid: [ch / tile, cw / tile],
      encoderPositions: LLAMA_TILES * perTile, tokens: 1, cross: LLAMA_TILES * perTile, perTile };
  }
  const s = m.enc.size, g = s / p;
  return { resized: [s, s], canvas: [s, s], patches: [g, g], tiles: 1, encoderPositions: g * g, tokens: 256, cross: 0, pool: g / 16 };
}
const kvPerToken = m => 2 * m.dec.kv * m.dec.dh * (m.dec.layers - m.dec.cross.length);
const crossPerPosition = m => 2 * m.dec.kv * m.dec.dh * m.dec.cross.length;
function cacheValues(key, h, w) {
  const m = VLMS[key], plan = imagePlan(key, h, w);
  return plan.tokens * kvPerToken(m) + plan.cross * crossPerPosition(m);
}

/* ---- Qwen3-VL: 3D positions (same as reference.vision.mrope_positions) ---- */
function mropePositions(segments, merge = 2) {
  const out = [[], [], []];
  let pos = 0;
  for (const [kind, size] of segments) {
    if (kind === 'text') { for (let i = 0; i < size; i++) out.forEach(a => a.push(pos + i)); pos += size; continue; }
    const [t, h, w] = size, rows = h / merge, cols = w / merge;
    for (let f = 0; f < t; f++) for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { out[0].push(pos + f); out[1].push(pos + r); out[2].push(pos + c); }
    pos += Math.max(rows, cols);
  }
  return out;
}
function mropePairAxes(headDim = 128, section = [24, 20, 20]) {
  const pairs = Array(headDim / 2).fill('T');
  [['H', 1, section[1]], ['W', 2, section[2]]].forEach(([a, off, n]) => { for (let i = off; i < 3 * n; i += 3) pairs[i] = a; });
  return pairs.join('');
}
/* ---- Gemma 3: causal, but one image's tokens see each other ---- */
function gemmaMask(isImage) {
  let cur = -1;
  const block = isImage.map((img, i) => (img && (i === 0 || !isImage[i - 1]) ? ++cur : img ? cur : -1));
  return isImage.map((_, i) => isImage.map((__, j) => j <= i || (block[i] >= 0 && block[i] === block[j])));
}

/* ---- text ---- */
const LEAD = {
  llava: 'The simplest connection: a vision encoder turns a fixed 336 × 336 crop into 576 patch vectors, and a small MLP maps each into the language model’s embedding space. To the decoder they are 576 more tokens.',
  qwen: 'The image keeps its own size and shape: one token per 32 × 32 pixels, so a larger picture costs more tokens. Each image token gets a 3D position (time, row, column), and features from three inner encoder layers are added into the first decoder layers.',
  llama: 'The image never enters the token sequence. A vision encoder reads up to four 560-pixel tiles, and eight cross-attention layers, inserted into a frozen Llama 3.1 8B, let the text read those image vectors.',
  gemma: 'Every image costs the same: it is squashed to 896 × 896, encoded into 64 × 64 patches, and averaged in 4 × 4 blocks down to 256 tokens. Inside the decoder, an image’s tokens see each other in both directions.',
};

function overview(key, h, w) {
  const m = VLMS[key], plan = imagePlan(key, h, w);
  return {
    tex: [
      String.raw`\text{image} \in \mathbb{R}^{${h} \times ${w} \times 3}`,
      String.raw`\text{patches} = \operatorname{split}_{${m.enc.patch}\times${m.enc.patch}}(\operatorname{resize}(\text{image}))`,
      String.raw`V = \operatorname{ViT}(\text{patches})\,,\quad X_{\text{img}} = \operatorname{connect}(V)`,
      key === 'llama' ? String.raw`h \mathrel{+}= \tanh(g)\cdot\operatorname{CrossAttn}(h, X_{\text{img}})` : String.raw`\text{tokens} = [\,\text{text},\ X_{\text{img}},\ \text{text}\,] \to \text{decoder}`,
    ],
    text: `${m.name} reads this ${w} × ${h} image as ${key === 'llama' ? `${fmt(plan.cross)} image vectors in its cross-attention layers, and one <|image|> token in the text` : `${fmt(plan.tokens)} tokens in the text sequence`}. Its cache for the image: ${fmtMiB(cacheValues(key, h, w))}.`,
  };
}

function steps(key, h, w) {
  const m = VLMS[key], e = m.enc, d = m.dec, plan = imagePlan(key, h, w);
  const [rh, rw] = plan.resized, [gh, gw] = plan.patches, P = e.patch;
  const perPatch = P * P * 3;
  const S = [];
  S.push({ id: 'image', chip: 'Image', title: 'An image is a grid of numbers', on: ['img'],
    tex: [String.raw`${h} \times ${w} \times 3 = ${fmt(h * w * 3)}\ \text{numbers}`],
    shape: `${w} × ${h} pixels, 3 colors`,
    text: `To a model, a picture is ${fmt(h * w * 3)} numbers: red, green and blue for each of its ${fmt(h * w)} pixels, scaled to around −1 to 1. Nothing in them says “sun” or “hill”; the encoder has to find that.` });
  const resize = ({
    llava: () => `The processor scales the short side to 336 and cuts out the centre square, so a wide image loses its sides (the dashed square). LLaVA-1.5 always sees 336 × 336 pixels.`,
    qwen: () => `The processor keeps the aspect ratio and rounds both sides to multiples of 32 (patch 16 × merge 2): ${w} × ${h} → ${rw} × ${rh}. Very small or very large images are scaled into 65,536 – 16.8 million pixels.`,
    llama: () => `The processor picks the grid of up to four 560 × 560 tiles that needs the least scaling (here ${plan.tileGrid[1]} wide × ${plan.tileGrid[0]} high), scales the image into it keeping its shape (${rw} × ${rh}), and pads the rest.`,
    gemma: () => `The processor stretches every image to 896 × 896, whatever its shape: the round sun of a wide picture becomes an oval. (Pan & scan, off by default, can add zoomed crops.)`,
  })[key]();
  S.push({ id: 'resize', chip: 'Resize', title: key === 'llama' ? 'Fit the image into tiles' : key === 'qwen' ? 'Round to the patch grid' : key === 'llava' ? 'Crop to a square' : 'Squash to a square', on: ['img', 'resized'],
    tex: [key === 'llama' ? String.raw`${w} \times ${h} \to ${plan.canvas[1]} \times ${plan.canvas[0]}\ (${plan.tiles}\ \text{tile${plan.tiles > 1 ? 's' : ''}})` : String.raw`${w} \times ${h} \to ${rw} \times ${rh}`],
    shape: key === 'llama' ? `canvas ${plan.canvas[1]} × ${plan.canvas[0]}` : `${rw} × ${rh} pixels`, text: resize });
  const patchText = {
    llava: `The square is cut into 24 × 24 = 576 patches of 14 × 14 pixels. Each patch's ${perPatch} numbers are mapped by one learned matrix to a 1,024-dim vector, and a class token is added in front: 577 vectors.`,
    qwen: `${rw} × ${rh} pixels make ${gw} × ${gh} = ${fmt(gh * gw)} patches of 16 × 16. The image is copied into two frames, so each patch is 2 × 16 × 16 × 3 = 1,536 numbers, mapped to 1,152 dims. The processor orders patches so that each 2 × 2 block (thick lines) is consecutive.`,
    llama: `Each tile is cut into 40 × 40 = 1,600 patches of 14 × 14, plus a class token: 1,601 vectors per tile. The encoder always runs on four tiles; the ${4 - plan.tiles} empty one${4 - plan.tiles === 1 ? ' is' : 's are'} masked.`,
    gemma: `896 × 896 pixels make 64 × 64 = 4,096 patches of 14 × 14, each mapped to a 1,152-dim vector. The thick lines mark the 4 × 4 blocks that will be averaged into one token.`,
  }[key];
  S.push({ id: 'patches', chip: 'Patches', title: 'Cut into patches, one vector each', on: ['resized', 'grid', 'enc-in'],
    tex: [String.raw`x_{i} = W_{\text{patch}}\,\operatorname{flatten}(\text{patch}_i) + p_i`, String.raw`${perPatch * (key === 'qwen' ? 2 : 1)} \to ${e.d}`],
    shape: key === 'llama' ? `4 × 1,601 × ${fmt(e.d)}` : `${fmt(plan.encoderPositions)} × ${fmt(e.d)}`, text: patchText });
  const encText = {
    llava: `24 transformer layers, like the decoder's but without a causal mask: every patch attends to every other. CLIP was trained to match images with their captions, so its vectors already describe content. LLaVA takes the output of layer 23, not the last, and drops the class token: 576 vectors.`,
    qwen: `27 layers in which all ${fmt(gh * gw)} patches attend to each other, with 2D rotary positions so a patch knows its row and column at any image size. Besides the last layer, the outputs of layers 9, 17 and 25 are kept for DeepStack.`,
    llama: `32 layers run on each tile separately, then 8 “global” layers see all four tiles together. The last global layer's output and those of local layers 4, 8, 16, 24 and 31 are set side by side: 6 × 1,280 = 7,680 dims per position.`,
    gemma: `SigLIP's 27 layers, all patches attending to each other. SigLIP is CLIP trained with a sigmoid loss; Gemma 3 keeps it frozen and shares it across the 4B, 12B and 27B models.`,
  }[key];
  S.push({ id: 'encode', chip: 'Encode', title: `Vision encoder: ${e.name}`, on: ['enc'],
    tex: [String.raw`V = \operatorname{ViT}_{${e.layers}${e.global ? '+' + e.global : ''}}(X),\ \text{no causal mask}`],
    shape: `${e.layers}${e.global ? ` + ${e.global}` : ''} layers · d ${fmt(e.d)} · ${e.heads} heads · ${(e.params / 1e6).toFixed(0)}M parameters`, text: encText });
  const connText = {
    llava: `Two linear layers with a GELU between them map each 1,024-dim patch vector to the decoder's 4,096 dims. That is the whole bridge: 21M parameters, trained first alone on image–caption pairs, then together with the language model.`,
    qwen: `Each 2 × 2 block of neighbouring patches is concatenated (4 × 1,152 = 4,608 dims) and passed through a two-layer MLP to 4,096: ${fmt(gh * gw)} patches become ${fmt(plan.tokens)} tokens. Three more mergers turn the layer 9, 17 and 25 features into tokens of their own for DeepStack.`,
    llama: `One linear map takes each 7,680-dim position to the decoder's 4,096 dims. Nothing is merged or dropped: ${fmt(plan.cross)} vectors per image, the class tokens and the empty tiles included.`,
    gemma: `The 64 × 64 patch vectors are averaged in 4 × 4 blocks (16 × 16 = 256 vectors), RMS-normalized, and multiplied by one 1,152 × 2,560 matrix into the decoder's embedding space.`,
  }[key];
  S.push({ id: 'connect', chip: 'Connect', title: `Connector: ${m.conn.name}`, on: ['conn'],
    tex: [{
      llava: String.raw`x_{\text{img}} = W_2\,\operatorname{GELU}(W_1 v + b_1) + b_2`,
      qwen: String.raw`x_{\text{img}} = \operatorname{MLP}([v_{1}; v_{2}; v_{3}; v_{4}])\,,\ 4608 \to 4096`,
      llama: String.raw`x_{\text{img}} = W\,[v^{(\text{last})}; v^{(4)}; v^{(8)}; v^{(16)}; v^{(24)}; v^{(31)}] + b`,
      gemma: String.raw`x_{\text{img}} = \operatorname{RMSNorm}\big(\tfrac{1}{16}\textstyle\sum_{4\times4} v\big)\,W`,
    }[key]],
    shape: { llava: '576 × 4,096', qwen: `${fmt(plan.tokens)} × 4,096`, llama: `${fmt(plan.cross)} × 4,096`, gemma: '256 × 2,560' }[key], text: connText });
  const seqText = {
    llava: `The prompt contains one <image> placeholder; the processor repeats it 576 times and the model swaps those embeddings for the 576 image vectors. From here on they are ordinary tokens with positions 1 to 576 in a row, as if the image were a 576-word sentence.`,
    qwen: `<|vision_start|>, ${fmt(plan.tokens)} × <|image_pad|>, <|vision_end|>. The image tokens get 3D positions (time, row, column) instead of 1, 2, 3 …: the decoder's rotary embedding rotates some frequency pairs by the row and others by the column. Text after the image continues at the image's start + its longer side (${Math.max(gh, gw) / 2}), not + ${fmt(plan.tokens)}.`,
    llama: `The text holds a single <|image|> token. The ${fmt(plan.cross)} image vectors stay outside the sequence, as keys and values for the cross-attention layers. Each text token may read the images that come before it.`,
    gemma: `"\\n\\n<start_of_image>", 256 × <image_soft_token>, "<end_of_image>\\n\\n". The 256 placeholders are swapped for the image vectors.`,
  }[key];
  S.push({ id: 'sequence', chip: 'Into the text', title: key === 'llama' ? 'One token in the text, the image beside it' : 'Image tokens join the text', on: ['seq'],
    tex: [key === 'llama' ? String.raw`\text{text: } 1\ \text{token}\,,\quad \text{memory: } 4 \times 1601 = 6404` : String.raw`T = T_{\text{text}} + ${fmt(plan.tokens)}`],
    shape: key === 'llama' ? '1 token + 6,404 vectors' : `${fmt(plan.tokens)} image tokens`, text: seqText });
  const decText = {
    llava: `Vicuna-7B reads image and text tokens alike: 32 layers of causal self-attention with 32 query and 32 key/value heads (MHA). Text tokens after the image attend to its 576 tokens like to earlier words.`,
    qwen: `36 layers of causal self-attention (32 query, 8 key/value heads). DeepStack adds the layer 9, 17 and 25 features onto the image tokens' hidden states after decoder layers 1, 2 and 3, so finer visual detail reaches the decoder without extra tokens.`,
    llama: `40 layers: the 32 of Llama 3.1 8B, unchanged and frozen during image training, plus 8 cross-attention layers (every 5th, from layer 4). In those, queries come from the text and keys and values from the image, and the result is added through gates tanh(g) that start at 0, so the model starts out as plain Llama.`,
    gemma: `34 layers with 8 query and 4 key/value heads. Image tokens attend to each other in both directions; everything else stays causal. Five of every six layers only look back 1,024 tokens, so a 256-token image fits inside that window.`,
  }[key];
  S.push({ id: 'decode', chip: 'Decode', title: `Decoder: ${d.name}`, on: ['dec'],
    tex: [{
      llava: String.raw`\text{causal self-attention over } [\text{text}, x_{\text{img}}, \text{text}]`,
      qwen: String.raw`h_{\text{img}}^{(\ell)} \mathrel{+}= \operatorname{merge}_{\ell}(v^{(8 \ell + 1)})\,,\ \ell = 1,2,3`,
      llama: String.raw`h \mathrel{+}= \tanh(g_{\text{attn}})\,\operatorname{softmax}\!\big(\tfrac{q_{\text{text}} K_{\text{img}}^{\top}}{\sqrt{128}}\big) V_{\text{img}}`,
      gemma: String.raw`\text{mask}_{ij} = [\,j \le i\,] \lor [\,i, j \in \text{same image}\,]`,
    }[key]],
    shape: `${d.layers} layers · d ${fmt(d.d)} · ${d.heads} q / ${d.kv} kv heads`, text: decText });
  const cost = cacheValues(key, h, w);
  S.push({ id: 'cost', chip: 'Cost', title: 'What the image costs', on: ['cost', 'seq'],
    tex: [key === 'llama' ? String.raw`6404 \times 2 \cdot 8 \cdot 128 \cdot 8 + 1 \times 2 \cdot 8 \cdot 128 \cdot 32` : String.raw`${fmt(plan.tokens)} \times 2 \cdot ${d.kv} \cdot ${d.dh} \cdot ${d.layers}`],
    shape: `${fmtMiB(cost)} at 16 bits`,
    text: key === 'llama'
      ? `The cache stores keys and values for all 6,404 image positions in each of the 8 cross-attention layers: ${fmtMiB(cost)}, the same for every image, since empty tiles are cached too. The one <|image|> token adds 128 KiB.`
      : `Each image token stores keys and values in every layer, like a word: ${fmt(plan.tokens)} × ${fmt(kvPerToken(m))} numbers = ${fmtMiB(cost)} for this image. ${key === 'qwen' ? 'A larger image costs proportionally more.' : 'Every image costs the same, whatever its size.'}` });
  return S;
}

const CLOSE = {
  llava: { title: 'From a patch to a token', sub: 'what happens to one 14 × 14 patch',
    text: 'A patch is flattened into 588 numbers and mapped by one learned matrix (a convolution with stride 14) to 1,024 dims; a learned position vector is added. After 23 of CLIP\'s 24 layers, the projector maps it to 4,096 dims. The class token, CLIP\'s summary of the whole image, is dropped: LLaVA wants one vector per region.' },
  qwen: { title: 'Positions in three dimensions (M-RoPE)', sub: 'a short prompt with a 3 × 5-token image',
    text: 'Text tokens count up as usual, with the same number on all three axes. Image tokens share the image\'s start on the time axis and count rows and columns on the other two. The rotary embedding uses each axis for a third of its frequency pairs, interleaved so that every axis gets fast and slow frequencies.' },
  llama: { title: 'Gated cross-attention', sub: 'which image positions each text token may read',
    text: 'Rows are text tokens, columns the 4 tiles × 1,601 image positions (drawn as 4 blocks). Tokens before <|image|> have nothing to read; after it, they read the tiles the image fills, never the empty ones. The layer\'s output is scaled by tanh(g), with g learned from 0.' },
  gemma: { title: 'Pooling, and an image that sees itself', sub: '64 × 64 patches → 16 × 16 tokens, and the attention mask',
    text: 'Each token is the average of a 4 × 4 block of patch vectors. In the decoder, text stays causal, but the 256 tokens of one image attend to each other in both directions, as they did in the encoder: an image has no reading order.' },
};

const SOURCES = [
  { by: 'Liu, Li, Li & Lee', year: 2024, title: 'Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)', in: 'CVPR 2024', url: 'https://arxiv.org/abs/2310.03744' },
  { by: 'Radford et al.', year: 2021, title: 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)', in: 'ICML 2021', url: 'https://arxiv.org/abs/2103.00020' },
  { by: 'Dosovitskiy et al.', year: 2021, title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', in: 'ICLR 2021', url: 'https://arxiv.org/abs/2010.11929' },
  { by: 'Qwen Team', year: 2025, title: 'Qwen3-VL Technical Report', in: 'arXiv:2511.21631', url: 'https://arxiv.org/abs/2511.21631' },
  { by: 'Wang et al.', year: 2024, title: 'Qwen2-VL: Enhancing Vision-Language Model\'s Perception of the World at Any Resolution (M-RoPE)', in: 'arXiv:2409.12191', url: 'https://arxiv.org/abs/2409.12191' },
  { by: 'Meng et al.', year: 2024, title: 'DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs', in: 'NeurIPS 2024', url: 'https://arxiv.org/abs/2406.04334' },
  { by: 'Llama Team, AI @ Meta', year: 2024, title: 'The Llama 3 Herd of Models', in: 'arXiv:2407.21783', url: 'https://arxiv.org/abs/2407.21783' },
  { by: 'Alayrac et al.', year: 2022, title: 'Flamingo: a Visual Language Model for Few-Shot Learning (gated cross-attention)', in: 'NeurIPS 2022', url: 'https://arxiv.org/abs/2204.14198' },
  { by: 'Gemma Team', year: 2025, title: 'Gemma 3 Technical Report', in: 'arXiv:2503.19786', url: 'https://arxiv.org/abs/2503.19786' },
  { by: 'Zhai, Mustafa, Kolesnikov & Beyer', year: 2023, title: 'Sigmoid Loss for Language Image Pre-Training (SigLIP)', in: 'ICCV 2023', url: 'https://arxiv.org/abs/2303.15343' },
];

function allTex() {
  const out = [];
  for (const k of ORDER) for (const [h, w] of SIZES) {
    overview(k, h, w).tex.forEach(t => out.push(t));
    steps(k, h, w).forEach(s => s.tex.forEach(t => out.push(t)));
  }
  return [...new Set(out)];
}

const MM = { VLMS, ORDER, SIZES, QWEN, LLAMA_TILES, fmt, fmtMiB, mib, pyRound, smartResize, tileArrangements, tiledCanvas, fitToCanvas,
  imagePlan, kvPerToken, crossPerPosition, cacheValues, mropePositions, mropePairAxes, gemmaMask, LEAD, overview, steps, CLOSE, SOURCES, allTex };
if (typeof module !== 'undefined' && module.exports) module.exports = MM; else root.MM = MM;
})(typeof window !== 'undefined' ? window : globalThis);
