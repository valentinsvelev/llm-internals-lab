/* Multimodal page, beyond images: audio in, video in, speech out, image out, and all together.
 * Data, arithmetic and text; no DOM. The rules mirror llm_lab/reference/{audio,video,speech,imagegen}.py,
 * which tests/test_media_models.py checks against transformers; tests/test_explorer.py checks this file
 * against those modules. */
(function (root) {
'use strict';
const MM = root.MM || (typeof require !== 'undefined' ? require('./mm-content.js') : null);
const { fmt, pyRound, smartResize } = MM;
const kvPerToken = d => 2 * d.kv * d.dh * d.layers;
const fmtMiB = MM.fmtMiB;
const secs = s => { const r = Math.round(s); return s < 60 ? `${+s.toFixed(s < 10 ? 1 : 0)} s` : r % 60 ? `${Math.floor(r / 60)} min ${r % 60} s` : `${r / 60} min`; };

const MODES = [
  { key: 'images', label: 'Images', title: 'Vision-language models', models: ['llava', 'qwen', 'llama', 'gemma'] },
  { key: 'audio', label: 'Audio', title: 'Audio-language models', models: ['qwen2audio', 'voxtral'] },
  { key: 'video', label: 'Video', title: 'Video-language models', models: ['qwen3vl', 'llava_ov'] },
  { key: 'speech', label: 'Speech out', title: 'Models that speak', models: ['qwen3omni', 'moshi'] },
  { key: 'imagegen', label: 'Image out', title: 'Models that draw', models: ['janus', 'emu3'] },
  { key: 'omni', label: 'All together', title: 'One model for everything', models: ['omni'] },
];
const MODE = Object.fromEntries(MODES.map(m => [m.key, m]));

/* ---------------------------------------------------------------- models ---------------------------------------------------------------- */
const whisper = { name: 'Whisper large-v3 encoder', layers: 32, d: 1280, heads: 20, ff: 5120, mels: 128 };
const MODELS = {
  qwen2audio: { key: 'qwen2audio', mode: 'audio', tab: 'Pooling', name: 'Qwen2-Audio-7B', hf: 'Qwen/Qwen2-Audio-7B-Instruct', year: 2024,
    enc: whisper, conn: { name: 'average pool + linear' },
    dec: { name: 'Qwen-7B', layers: 32, d: 4096, heads: 32, kv: 32, dh: 128, ff: 11008, V: 156032, ctx: 8192 },
    params: [636968960, 5246976, 7754878976], partNames: ['Audio encoder', 'Adapter', 'Language model'],
    template: [['<|im_start|>user', 'text'], ['<|audio_bos|>', 'mark'], ['<|AUDIO|>', 'media'], ['<|audio_eos|>', 'mark'], ['What is said?', 'text'], ['…<|im_start|>assistant', 'text']] },
  voxtral: { key: 'voxtral', mode: 'audio', tab: 'Stacking', name: 'Voxtral Mini 3B', hf: 'mistralai/Voxtral-Mini-3B-2507', year: 2025,
    enc: whisper, conn: { name: '4 frames stacked + MLP' },
    dec: { name: 'Ministral 3B', layers: 30, d: 3072, heads: 32, kv: 8, dh: 128, ff: 8192, V: 131072, ctx: 32768 },
    params: [636968960, 25165824, 4014136320], partNames: ['Audio encoder', 'Adapter', 'Language model'],
    template: [['<s>[INST]', 'text'], ['[AUDIO]', 'media'], ['What is said?', 'text'], ['[/INST]', 'text']] },
  qwen3vl: { key: 'qwen3vl', mode: 'video', tab: '2 frames per second', name: 'Qwen3-VL-8B', hf: 'Qwen/Qwen3-VL-8B-Instruct', year: 2025,
    enc: { name: 'Qwen3-VL ViT', layers: 27, d: 1152, heads: 16, ff: 4304, patch: 16 }, conn: { name: '2 × 2 merger' },
    dec: MM.VLMS.qwen.dec, params: [415891440, 160496896, 8190735360], partNames: ['Vision encoder', 'Connector', 'Language model'],
    template: [['<|im_start|>user', 'text'], ['<0.3 seconds>', 'stamp'], ['<|video_pad|>', 'media'], ['<1.3 seconds> …', 'stamp'], ['What happens?', 'text'], ['…<|im_start|>assistant', 'text']] },
  llava_ov: { key: 'llava_ov', mode: 'video', tab: '32 frames', name: 'LLaVA-OneVision-7B', hf: 'llava-hf/llava-onevision-qwen2-7b-ov-hf', year: 2024,
    enc: { name: 'SigLIP So400m/14-384', layers: 26, d: 1152, heads: 16, ff: 4304, patch: 14 }, conn: { name: 'MLP + 2 × 2 pooling' },
    dec: { name: 'Qwen2-7B', layers: 28, d: 3584, heads: 28, kv: 4, dh: 128, ff: 18944, V: 152128, ctx: 32768 },
    params: [397747744, 16980992, 7616078848], partNames: ['Vision encoder', 'Connector', 'Language model'],
    template: [['<|im_start|>user', 'text'], ['<video>', 'media'], ['What happens?', 'text'], ['…<|im_start|>assistant', 'text']] },
  qwen3omni: { key: 'qwen3omni', mode: 'speech', tab: 'Thinker–Talker', name: 'Qwen3-Omni-30B-A3B', hf: 'Qwen/Qwen3-Omni-30B-A3B-Instruct', year: 2025,
    params: [31719205488, 3324596480, 216016577], partNames: ['Thinker', 'Talker', 'Code2Wav'],
    codebooks: 16, codebook: 2048, main: { name: 'Talker', layers: 20, d: 1024, kv: 2, dh: 128, note: 'MoE, 128 experts, 6 active' },
    inner: { name: 'code predictor', layers: 5, d: 1024 }, decoder: 'Code2Wav', lm: { name: 'Thinker', layers: 48, d: 2048, note: '30B MoE, 3B active' } },
  moshi: { key: 'moshi', mode: 'speech', tab: 'Full duplex', name: 'Moshi', hf: 'kmhf/hf-moshiko', year: 2024,
    params: [79308609, 6973366272, 714362880], partNames: ['Mimi codec', 'Helium', 'Depth transformer'],
    codebooks: 8, codebook: 2048, main: { name: 'Helium', layers: 32, d: 4096, kv: 32, dh: 128, note: 'one pass per 80 ms' },
    inner: { name: 'depth transformer', layers: 6, d: 1024 }, decoder: 'Mimi', lm: { name: 'Helium', layers: 32, d: 4096, note: '7B' } },
  janus: { key: 'janus', mode: 'imagegen', tab: 'Own image head', name: 'Janus-Pro-7B', hf: 'deepseek-community/Janus-Pro-7B', year: 2025,
    params: [71883403, 425524224, 6910365696], partNames: ['VQ tokenizer', 'Image layers', 'Language model'],
    dec: { name: 'DeepSeek-LLM 7B', layers: 30, d: 4096, heads: 32, kv: 32, dh: 128, ff: 11008, V: 102400, ctx: 16384 },
    codebook: 16384, down: 16, codeDim: 8 },
  emu3: { key: 'emu3', mode: 'imagegen', tab: 'Shared vocabulary', name: 'Emu3-Gen', hf: 'BAAI/Emu3-Gen', year: 2024,
    params: [270740287, 0, 8492011520], partNames: ['VQ tokenizer', 'Image layers', 'Language model'],
    dec: { name: 'Emu3 (Llama-style)', layers: 32, d: 4096, heads: 32, kv: 8, dh: 128, ff: 14336, V: 184622, ctx: 9216 },
    codebook: 32768, down: 8, codeDim: 4 },
  omni: { key: 'omni', mode: 'omni', tab: 'Qwen3-Omni', name: 'Qwen3-Omni-30B-A3B', hf: 'Qwen/Qwen3-Omni-30B-A3B-Instruct', year: 2025,
    params: [647927168, 538631408, 30532646912, 3540613057], partNames: ['Audio encoder', 'Vision encoder', 'Thinker LM', 'Talker + Code2Wav'],
    dec: { name: 'Thinker', layers: 48, d: 2048, heads: 32, kv: 4, dh: 128, V: 152064, ctx: 65536 } },
};

/* ---------------------------------------------------------------- audio ---------------------------------------------------------------- */
const SR = 16000, HOP = 160, CHUNK_SAMPLES = 480000, CHUNK_FRAMES = 3000, ENC_POS = 1500;
function audioPlan(key, seconds) {
  const samples = Math.max(1, pyRound(seconds * SR));
  if (key === 'qwen2audio') {
    const kept = Math.min(samples, CHUNK_SAMPLES), frames = Math.ceil(kept / HOP);
    const conv = Math.floor((frames - 1) / 2) + 1, tokens = Math.floor((conv - 2) / 2) + 1;
    return { samples: kept, chunks: 1, melFrames: frames, encoderPositions: ENC_POS, tokens, dropped: (samples - kept) / SR };
  }
  const chunks = Math.ceil(samples / CHUNK_SAMPLES);
  return { samples, chunks, melFrames: Math.ceil(samples / HOP), encoderPositions: chunks * ENC_POS, tokens: chunks * 375, dropped: 0 };
}

/* ---------------------------------------------------------------- video ---------------------------------------------------------------- */
const VID = { fps: 2, minFrames: 4, maxFrames: 768, temporal: 2, factor: 32, minPixels: 4096, maxPixels: 25165824, maxTokens: 768, ovFrames: 32 };
function sampleFrames(seconds, videoFps = 30) {
  const total = Math.max(1, pyRound(seconds * videoFps));
  let n = Math.trunc(total / videoFps * VID.fps);
  n = Math.min(Math.max(n, VID.minFrames), VID.maxFrames, total);
  if (n === 1) return [0];
  const step = (total - 1) / (n - 1), out = [];
  for (let i = 0; i < n - 1; i++) out.push(pyRound(i * step));
  out.push(total - 1);
  return out;
}
function videoSmartResize(frames, height, width, maxPixels, minPixels = VID.minPixels, factor = VID.factor) {
  if (height < factor || width < factor) { const s = Math.max(factor / height, factor / width); height = Math.trunc(height * s); width = Math.trunc(width * s); }
  let h = pyRound(height / factor) * factor, w = pyRound(width / factor) * factor;
  const t = pyRound(frames / VID.temporal) * VID.temporal;
  if (t * h * w > maxPixels) {
    const beta = Math.sqrt(frames * height * width / maxPixels);
    h = Math.max(factor, Math.floor(height / beta / factor) * factor); w = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (t * h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (frames * height * width));
    h = Math.ceil(height * beta / factor) * factor; w = Math.ceil(width * beta / factor) * factor;
  }
  return [h, w];
}
function videoPlan(key, seconds, height = 720, width = 1280, videoFps = 30) {
  if (key === 'qwen3vl') {
    const idx = sampleFrames(seconds, videoFps), n = idx.length;
    const perFrame = Math.max(Math.min(VID.maxTokens * VID.factor * VID.factor, Math.floor(VID.maxPixels / n)), Math.trunc(VID.minPixels * 1.05));
    const [rh, rw] = videoSmartResize(n, height, width, perFrame * n);
    const groups = Math.ceil(n / VID.temporal), perGroup = (rh / VID.factor) * (rw / VID.factor);
    const times = idx.map(i => i / videoFps), stamps = [];
    for (let i = 0; i < n; i += VID.temporal) stamps.push((times[i] + times[Math.min(i + VID.temporal - 1, n - 1)]) / 2);
    return { frames: n, indices: idx, frameSize: [rh, rw], groups, perGroup, tokens: groups * perGroup, stamps };
  }
  const side = Math.ceil(Math.floor(384 / 14) / 2), per = side * side;
  return { frames: VID.ovFrames, indices: null, frameSize: [384, 384], groups: VID.ovFrames, perGroup: per, tokens: VID.ovFrames * per + 1, stamps: [] };
}

/* ---------------------------------------------------------------- speech ---------------------------------------------------------------- */
const SPEECH = { frameRate: 12.5, samplesPerFrame: 1920, outRate: 24000, moshiStreams: 17, positionsPerSecond: 13 };
function speechPlan(key, seconds) {
  const m = MODELS[key], frames = Math.ceil(seconds * SPEECH.frameRate - 1e-9);
  const kv = frames * 2 * m.main.kv * m.main.dh * m.main.layers;
  const inner = key === 'qwen3omni' ? frames * (m.codebooks - 1) : frames * m.codebooks;
  return { frames, codes: frames * m.codebooks, mainSteps: frames, innerSteps: inner, samples: frames * SPEECH.samplesPerFrame, kv };
}
const moshiDelays = (n = 8) => [0, ...Array(n - 1).fill(1)];
function omniAudioTokens(mel, nWindow = 50) {
  const chunk = 2 * nWindow, rest = mel % chunk, feat = Math.floor((rest - 1) / 2) + 1;
  return Math.floor((Math.floor((feat - 1) / 2) + 1 - 1) / 2) + 1 + Math.floor(mel / chunk) * 13;
}
function omniInterleave(grids, perGrid, audio, secondsPerGrid = 1, pps = SPEECH.positionsPerSecond) {
  const vt = []; for (let g = 0; g < grids; g++) for (let k = 0; k < perGrid; k++) vt.push(g * secondsPerGrid * pps);
  let out = '', i = 0, j = 0;
  while (i < vt.length && j < audio) { if (vt[i] <= j) { out += 'v'; i++; } else { out += 'a'; j++; } }
  return out + 'v'.repeat(vt.length - i) + 'a'.repeat(audio - j);
}
function omniPlan(seconds, height = 360, width = 640) {
  const [rh, rw] = smartResize(height, width), per = (rh / 32) * (rw / 32), audio = omniAudioTokens(seconds * 100);
  return { grids: seconds, perGrid: per, frameSize: [rh, rw], audioTokens: audio, order: omniInterleave(seconds, per, audio), positions: seconds * SPEECH.positionsPerSecond };
}

/* ---------------------------------------------------------------- image out ---------------------------------------------------------------- */
const GEN = { janusSide: 384, cfg: 5, emu3Area: 720 * 720, ratios: ['1:1', '4:3', '16:9', '9:16'] };
function emu3Grid(ratio = '1:1', area = GEN.emu3Area, factor = 8) {
  const [w, h] = ratio.split(':').map(Number), s = Math.sqrt(area / (w * h));
  return [pyRound(h * s / factor), pyRound(w * s / factor)];
}
function genPlan(key, ratio = '1:1') {
  const m = MODELS[key], per = kvPerToken(m.dec);
  if (key === 'janus') { const g = GEN.janusSide / m.down; return { grid: [g, g], pixels: [GEN.janusSide, GEN.janusSide], codes: g * g, steps: g * g, sequences: 2, kv: 2 * g * g * per }; }
  const [r, c] = emu3Grid(ratio), steps = r * (c + 1);
  return { grid: [r, c], pixels: [r * m.down, c * m.down], codes: r * c, steps, sequences: 2, kv: 2 * steps * per };
}

/* ---------------------------------------------------------------- controls ---------------------------------------------------------------- */
const CONTROLS = {
  audio: { label: 'Audio length', values: [5, 30, 120, 600, 1800], fmt: secs, def: 1, file: 'audio/*', fileLabel: 'Your audio…' },
  video: { label: 'Video length (1280 × 720, 30 fps)', values: [10, 60, 300, 1200], fmt: secs, def: 1, file: 'video/*', fileLabel: 'Your video…' },
  speech: { label: 'Length of the spoken answer', values: [2, 10, 30, 120], fmt: secs, def: 1 },
  imagegen: { label: 'Aspect ratio (Emu3; Janus is always square)', values: GEN.ratios, fmt: r => r, def: 0 },
  omni: { label: 'Clip length (640 × 360 video with sound)', values: [3, 10, 30], fmt: secs, def: 0 },
};

/* ---------------------------------------------------------------- per-model numbers shown as stats ---------------------------------------------------------------- */
function summary(key, v, custom) {
  const m = MODELS[key];
  if (m.mode === 'audio') { const p = audioPlan(key, v); return { tokens: p.tokens, tokensLabel: 'Audio tokens here', cache: p.tokens * kvPerToken(m.dec), ctx: m.dec.ctx }; }
  if (m.mode === 'video') {
    const [h, w] = custom ? [custom.h, custom.w] : [720, 1280], p = videoPlan(key, v, h, w);
    return { tokens: p.tokens, tokensLabel: 'Video tokens here', cache: p.tokens * kvPerToken(m.dec), ctx: m.dec.ctx };
  }
  if (m.mode === 'speech') { const p = speechPlan(key, v); return { tokens: p.codes, tokensLabel: 'Codes written', cache: p.kv, ctx: null }; }
  if (m.mode === 'imagegen') { const p = genPlan(key, v); return { tokens: p.steps, tokensLabel: 'Tokens written', cache: p.kv, ctx: m.dec.ctx }; }
  const p = omniPlan(v); return { tokens: p.grids * p.perGrid + p.audioTokens, tokensLabel: 'Clip tokens', cache: (p.grids * p.perGrid + p.audioTokens) * kvPerToken(m.dec), ctx: m.dec.ctx };
}

/* ---------------------------------------------------------------- text ---------------------------------------------------------------- */
const TITLE = {
  qwen2audio: 'Whisper features, pooled', voxtral: 'Whisper features, stacked', qwen3vl: 'Frames in pairs, with timestamps', llava_ov: 'A fixed 32 frames, pooled',
  qwen3omni: 'A Talker beside the Thinker', moshi: 'Listening and speaking at once', janus: 'Image codes with their own head', emu3: 'Image codes as words', omni: 'Sound and video on one clock',
};
const LEAD = {
  qwen2audio: 'Sound becomes a spectrogram, Whisper’s encoder turns 30 seconds of it into 1,500 vectors, and averaging pairs leaves 25 tokens per second. One clip, at most 30 seconds.',
  voxtral: 'The same Whisper encoder, run on each 30-second chunk separately; four neighbouring vectors are stacked into one token, so a second of sound costs 12.5 tokens and 40 minutes fit in a 32k context.',
  qwen3vl: 'Two frames per second, every two frames merged into one grid of tokens with a text timestamp before it. Long videos keep more frames, each with fewer pixels.',
  llava_ov: 'However long the video, 32 frames (its usual setting): each read by the image encoder as a 384 × 384 picture, then shrunk from 27 × 27 to 14 × 14 vectors. Always 6,273 tokens.',
  qwen3omni: 'The language model (Thinker) writes text; a smaller Talker reads its hidden states and writes speech as codec tokens, 16 per 80 ms, which Code2Wav turns into sound while the answer is still being written.',
  moshi: 'One 7B model runs every 80 ms over its own text, its own voice and the user’s voice, so it can listen while it talks. A small depth transformer writes the 8 codes of each step.',
  janus: 'The language model writes an image as 576 codes, one per 16 × 16 pixels, through its own embedding and head for image codes. Reading images goes through a different encoder.',
  emu3: 'Image codes are just more words in the vocabulary: 90 rows of 90 codes, each row ended by a newline token, written by the ordinary language-model head. 8,190 tokens for one 720 × 720 image.',
  omni: 'Qwen3-Omni reads text, images, sound and video in one sequence. A clip’s frames and its sound are interleaved by time, 13 positions per second, and the answer can come back as speech.',
};

function steps(key, v, custom) {
  const m = MODELS[key], S = [];
  if (m.mode === 'audio') {
    const p = audioPlan(key, v), sec = +v.toFixed(2), q = key === 'qwen2audio', d = m.dec;
    S.push({ id: 'sound', chip: 'Sound', title: 'Sound is a list of numbers', on: ['wave'], tex: [String.raw`${fmt(Math.round(sec * SR)).replace(/,/g, '{,}')}\ \text{samples} = ${sec}\,\text{s} \times 16{,}000`],
      shape: `${secs(sec)} · 16 kHz · mono`, text: `Speech or music is air pressure measured 16,000 times a second: ${fmt(Math.round(sec * SR))} numbers for ${secs(sec)}. ${q && p.dropped > 0 ? `Qwen2-Audio reads only the first 30 s; the other ${secs(Math.round(p.dropped))} are cut off.` : ''}` });
    S.push({ id: 'mel', chip: 'Spectrogram', title: 'A picture of the sound', on: ['wave', 'mel'], tex: [String.raw`25\,\text{ms windows every } 10\,\text{ms} \to 100\ \text{frames/s} \times 128\ \text{mel bands}`],
      shape: q ? `3,000 frames (${fmt(p.melFrames)} with sound)` : `${p.chunks} × 3,000 frames`,
      text: q ? `Every 10 ms, the strength of 128 frequency bands (spaced like human pitch perception, on a log scale) over the last 25 ms. Whisper's encoder always reads 30 s, so ${p.melFrames < 3000 ? 'the rest is padded with silence' : 'nothing is padded'}: 3,000 frames.`
        : `The same spectrogram, but the audio is padded to a whole number of 30-s chunks: ${p.chunks} chunk${p.chunks > 1 ? 's' : ''} of 3,000 frames, each encoded on its own.` });
    S.push({ id: 'encode', chip: 'Encode', title: 'Whisper large-v3’s encoder', on: ['enc'], tex: [String.raw`3000 \xrightarrow{\text{conv, stride } 2} 1500 \times 1280 \xrightarrow{32\ \text{layers}} 1500 \times 1280`],
      shape: `${q ? '1,500' : `${p.chunks} × 1,500`} × 1,280 · 32 layers · 20 heads`,
      text: `Two convolutions turn the 128 bands into 1,280-dim vectors; the second has stride 2, so 50 vectors per second. Then 32 transformer layers in which every position sees the whole 30-s window (no causal mask)${q ? '' : ', but never another chunk'}. Whisper learned these features from 5 million hours of transcription.` });
    S.push({ id: 'adapt', chip: 'Adapt', title: q ? 'Average pairs, project' : 'Stack four, project', on: ['conn'],
      tex: [q ? String.raw`x = W\,\operatorname{LN}\big(\tfrac{1}{2}(v_{2i} + v_{2i+1})\big),\ 1280 \to 4096` : String.raw`x = W_2\,\operatorname{GELU}\big(W_1\,[v_{4i}; v_{4i+1}; v_{4i+2}; v_{4i+3}]\big),\ 5120 \to 3072`],
      shape: q ? `${fmt(p.tokens)} × 4,096` : `${fmt(p.tokens)} × 3,072`,
      text: q ? `Neighbouring vectors are averaged in pairs (750 per 30 s, 25 per second), normalized and mapped to the decoder's 4,096 dims by one linear layer. Only the ${fmt(p.tokens)} that cover sound are kept.`
        : `Four neighbouring vectors are set side by side (4 × 1,280 = 5,120) and mapped through a two-layer MLP to 3,072: 375 tokens per chunk, 12.5 per second. Padding at the end of the last chunk is kept.` });
    S.push({ id: 'sequence', chip: 'Into the text', title: 'Audio tokens join the text', on: ['seq'], tex: [String.raw`T = T_{\text{text}} + ${fmt(p.tokens).replace(/,/g, '{,}')}`],
      shape: `${fmt(p.tokens)} audio tokens`, text: `The prompt holds one ${q ? '<|AUDIO|>' : '[AUDIO]'} placeholder; the processor repeats it ${fmt(p.tokens)} times and the model swaps in the audio vectors. From here on they are ordinary tokens with ordinary positions.` });
    S.push({ id: 'decode', chip: 'Decode', title: `Decoder: ${d.name}`, on: ['dec'], tex: [String.raw`\text{causal self-attention over } [\text{text}, x_{\text{audio}}, \text{text}]`],
      shape: `${d.layers} layers · d ${fmt(d.d)} · ${d.heads} q / ${d.kv} kv heads`, text: `${d.name} reads the audio tokens like words. Transcribing, translating or answering questions about the sound are all just text it writes next.` });
    const cache = p.tokens * kvPerToken(d);
    S.push({ id: 'cost', chip: 'Cost', title: 'What the sound costs', on: ['cost', 'seq'], tex: [String.raw`${fmt(p.tokens).replace(/,/g, '{,}')} \times 2 \cdot ${d.kv} \cdot ${d.dh} \cdot ${d.layers}`],
      shape: `${fmtMiB(cache)} at 16 bits`, text: `Each audio token stores keys and values in every layer: ${fmtMiB(cache)} for ${secs(v)}${q ? '' : `, ${(100 * p.tokens / d.ctx).toFixed(p.tokens / d.ctx < 0.1 ? 1 : 0)}% of the 32k context`}. ${q ? 'A minute of audio would need two separate clips.' : 'At 12.5 tokens per second, 40 minutes is 30,000 tokens.'}` });
    return S;
  }
  if (m.mode === 'video') {
    const [H, W] = custom ? [custom.h, custom.w] : [720, 1280], p = videoPlan(key, v, H, W), q = key === 'qwen3vl', d = m.dec;
    const total = Math.max(1, pyRound(v * 30));
    S.push({ id: 'video', chip: 'Video', title: 'A video is many images', on: ['film'], tex: [String.raw`${fmt(total).replace(/,/g, '{,}')}\ \text{frames} \times ${H} \times ${W} \times 3`],
      shape: `${secs(v)} · ${fmt(total)} frames`, text: `${secs(v)} at 30 frames per second is ${fmt(total)} images of ${W} × ${H}. No model reads them all.` });
    S.push({ id: 'sample', chip: 'Sample', title: q ? 'Two frames per second' : '32 frames, evenly spaced', on: ['film', 'pick'], tex: [q ? String.raw`n = \min(\max(\lfloor 2t \rfloor, 4), 768) = ${p.frames}` : String.raw`n = 32,\ \text{one every } ${(v / 32).toFixed(v / 32 < 10 ? 2 : 1)}\,\text{s}`],
      shape: `${fmt(p.frames)} frames`, text: q ? `Qwen3-VL keeps 2 frames per second (at least 4, at most 768): ${fmt(p.frames)} frames${p.frames === 768 ? ', so here frames are further apart than half a second' : ''}.`
        : `LLaVA-OneVision's usual setting is 32 frames spread over the whole video: one every ${(v / 32).toFixed(1)} s here. A long video is sampled sparsely, a short one densely.` });
    S.push({ id: 'resize', chip: 'Resize', title: q ? 'Share a pixel budget' : 'Squash each frame', on: ['pick'], tex: [q ? String.raw`${W} \times ${H} \to ${p.frameSize[1]} \times ${p.frameSize[0]}` : String.raw`${W} \times ${H} \to 384 \times 384`],
      shape: `${p.frameSize[1]} × ${p.frameSize[0]} per frame`,
      text: q ? `All sampled frames together get at most 25.2 million pixels, and each frame at most 786,432 (768 tokens): ${fmt(p.frames)} frames of ${p.frameSize[1]} × ${p.frameSize[0]}. The longer the video, the smaller each frame.`
        : 'Each frame is resized to 384 × 384 whatever its shape, as for a single image.' });
    S.push({ id: 'encode', chip: 'Encode', title: q ? 'Patches through time' : 'Each frame alone', on: ['enc'],
      tex: [q ? String.raw`\text{patch} = 2 \times 16 \times 16 \times 3\ \text{(two frames)}` : String.raw`384 / 14 = 27 \Rightarrow 27 \times 27 = 729\ \text{vectors per frame}`],
      shape: q ? `${p.groups} grids × ${fmt(p.perGroup * 4)} patches` : '32 × 729 × 1,152',
      text: q ? `The image encoder of the Images tab, with its 2-frame patches finally used: each patch covers 16 × 16 pixels in two consecutive frames, so ${fmt(p.frames)} frames make ${fmt(p.groups)} grids.`
        : `SigLIP reads every frame as an independent 384 × 384 image: 729 vectors per frame, 23,328 for the video. The frames never see each other in the encoder.` });
    S.push({ id: 'connect', chip: 'Connect', title: q ? 'Merge 2 × 2' : 'Project, then pool 2 × 2', on: ['conn'],
      tex: [q ? String.raw`${fmt(p.perGroup * 4).replace(/,/g, '{,}')} \to ${fmt(p.perGroup).replace(/,/g, '{,}')}\ \text{per grid}` : String.raw`27 \times 27 \xrightarrow{\text{bilinear}} 14 \times 14 = 196`],
      shape: `${fmt(p.tokens)} × ${fmt(d.d)}`, text: q ? `As for images, 2 × 2 neighbouring patches are merged into one 4,096-dim token: ${fmt(p.perGroup)} tokens per grid, ${fmt(p.tokens)} in all.`
        : `An MLP maps each vector to the decoder's 3,584 dims, then every frame's 27 × 27 grid is resized to 14 × 14 by bilinear interpolation: 196 tokens per frame, plus one learned “newline” vector at the end. 32 × 196 + 1 = 6,273.` });
    S.push({ id: 'sequence', chip: 'Into the text', title: q ? 'Timestamps between grids' : 'One long block', on: ['seq'],
      tex: [q ? String.raw`\langle t\ \text{seconds}\rangle\ \texttt{<|vision\_start|>}\ ${p.perGroup} \times \texttt{<|video\_pad|>}\ \texttt{<|vision\_end|>}` : String.raw`6273 \times \texttt{<video>}`],
      shape: `${fmt(p.tokens)} video tokens`, text: q ? `Each grid is preceded by its time as text, such as “<${(p.stamps[0] || 0).toFixed(1)} seconds>”, the average time of its two frames. Each grid gets positions as if it were a separate image; time is given only as text.`
        : 'The frames follow each other in one block; their order is the only sign of time. Nothing says how far apart two frames are.' });
    const cache = p.tokens * kvPerToken(d);
    S.push({ id: 'cost', chip: 'Cost', title: 'What the video costs', on: ['cost', 'seq'], tex: [String.raw`${fmt(p.tokens).replace(/,/g, '{,}')} \times 2 \cdot ${d.kv} \cdot ${d.dh} \cdot ${d.layers}`],
      shape: `${fmtMiB(cache)} at 16 bits`, text: `${fmt(p.tokens)} tokens (${q ? 'plus a few for each timestamp' : 'whatever the length'}): ${fmtMiB(cache)} of cache, ${(100 * p.tokens / d.ctx).toFixed(0)}% of the ${fmt(d.ctx)}-token context.` });
    return S;
  }
  if (m.mode === 'speech') {
    const p = speechPlan(key, v), q = key === 'qwen3omni';
    if (q) {
      S.push({ id: 'think', chip: 'Think', title: 'The Thinker writes text', on: ['lm'], tex: [String.raw`\text{Thinker: } 48\ \text{layers},\ 128\ \text{experts},\ 8\ \text{active}`],
        shape: '30B parameters, 3B active per token', text: 'The language model (the Thinker) reads the conversation and writes its answer as text tokens, as any LLM. It never produces sound itself.' });
      S.push({ id: 'talk', chip: 'Talk', title: 'The Talker writes the first codebook', on: ['main'], tex: [String.raw`c^{(1)}_t = \operatorname{Talker}(h^{\text{Thinker}}_{\le t},\ c_{<t})`],
        shape: `${fmt(p.frames)} frames · one pass each`, text: 'A smaller MoE transformer (20 layers, 1,024 wide) reads the Thinker’s hidden states and the text as it is written, and writes one token per 80 ms: codebook 1, the coarsest description of that slice of sound.' });
      S.push({ id: 'fill', chip: 'Fill in', title: 'The code predictor adds 15 codebooks', on: ['inner', 'grid'], tex: [String.raw`c^{(k)}_t = \operatorname{MTP}(c^{(1)}_t, \ldots, c^{(k-1)}_t),\ k = 2 \ldots 16`],
        shape: `${fmt(p.innerSteps)} small passes`, text: 'A 5-layer transformer predicts codebooks 2 to 16 of the same frame, one after another, each refining the sound the earlier ones describe (residual quantization). 16 codes of 2,048 values: 176 bits per 80 ms.' });
      S.push({ id: 'decode', chip: 'To sound', title: 'Code2Wav', on: ['dec', 'wave'], tex: [String.raw`16\ \text{codes} \to 1920\ \text{samples}\ (8 \cdot 5 \cdot 4 \cdot 3 \cdot 2 \cdot 2)`],
        shape: `${fmt(p.samples)} samples at 24 kHz`, text: 'The 16 codes of a frame are embedded and averaged, 8 transformer layers look back over recent frames, and transposed convolutions upsample 1,920×: 24,000 samples per second. Frames can be played as soon as they are written.' });
    } else {
      S.push({ id: 'codec', chip: 'Codec', title: 'Mimi: sound to codes and back', on: ['dec', 'grid'], tex: [String.raw`24{,}000\ \text{samples/s} \to 12.5\ \text{frames/s} \times 8\ \text{codes}`],
        shape: '1,920 samples per frame', text: 'Mimi compresses 80 ms of 24 kHz sound into codes of 8 codebooks (of its 32) with 2,048 values each. Codebook 1 was trained to match a speech model’s features, so it carries what is said; 2–8 carry how it sounds.' });
      S.push({ id: 'streams', chip: '17 streams', title: 'Every step reads 17 streams', on: ['lm', 'grid'], tex: [String.raw`x_t = e^{\text{text}}(w_t) + \textstyle\sum_{k=1}^{8} e_k(c^{\text{moshi}}_{k,t}) + \sum_{k=1}^{8} e_k(c^{\text{user}}_{k,t})`],
        shape: '1 + 8 + 8 embeddings, summed', text: 'Moshi’s own text (an “inner monologue”, aligned in time with its voice), its 8 codes and the user’s 8 codes are embedded and added into one vector per 80 ms step. There are no turns: both voices are always there.' });
      S.push({ id: 'main', chip: 'Helium', title: 'Helium, once per step', on: ['lm'], tex: [String.raw`z_t = \operatorname{Helium}(x_1, \ldots, x_t)`],
        shape: `${fmt(p.frames)} passes · 32 layers · 4,096 wide`, text: 'A 7B transformer runs along time, one pass per 80 ms, and predicts Moshi’s next text token. Its context is 3,000 steps: 4 minutes of conversation.' });
      S.push({ id: 'depth', chip: 'Depth', title: 'The depth transformer writes 8 codes', on: ['inner', 'grid'], tex: [String.raw`c_{k,t} = \operatorname{Depth}(z_t, w_t, c_{1,t}, \ldots, c_{k-1,t}),\ k = 1 \ldots 8`],
        shape: `${fmt(p.innerSteps)} small passes`, text: 'A 6-layer transformer, 1,024 wide, runs along the codebooks instead of time: from Helium’s output and the text token it writes codebook 1, then 2, … 8. Codebooks 2–8 lag one step behind codebook 1, so each step’s fine detail can depend on the next step’s content.' });
    }
    const m2 = MODELS[key];
    S.push({ id: 'cost', chip: 'Cost', title: 'What speaking costs', on: ['cost'], tex: [String.raw`${fmt(p.frames)}\ \text{frames} \times ${m2.codebooks}\ \text{codes} = ${fmt(p.codes).replace(/,/g, '{,}')}`],
      shape: `${fmt(p.mainSteps)} big + ${fmt(p.innerSteps)} small passes`, text: `${secs(v)} of speech is ${fmt(p.frames)} frames. The per-frame network runs ${fmt(p.mainSteps)} times and caches ${fmtMiB(p.kv)} of keys and values; the small one runs ${fmt(p.innerSteps)} times. ${q ? 'Speech can start after the first frame: Code2Wav plays each frame as it arrives.' : 'To stay real-time, every pass of a step must finish within 80 ms.'}` });
    return S;
  }
  if (m.mode === 'imagegen') {
    const p = genPlan(key, v), j = key === 'janus', d = m.dec;
    S.push({ id: 'prompt', chip: 'Prompt', title: 'Start from a prompt', on: ['lm'], tex: [j ? String.raw`\text{prompt} \to \text{LM},\ \text{twice: with and without the text}` : String.raw`\text{prompt} + \texttt{<|image start|>}\,${p.grid[0]}{*}${p.grid[1]}`],
      shape: `${p.sequences} sequences`, text: j ? 'The text prompt goes into the language model twice: as written, and with its text replaced by padding. The second run says what the model would draw anyway, for classifier-free guidance.'
        : `The prompt ends with the image's size in codes, “${p.grid[0]}*${p.grid[1]}”, so the model knows when rows end. Emu3's own example also runs a second, negative prompt for guidance.` });
    S.push({ id: 'step', chip: 'One code', title: j ? 'Its own head picks a code' : 'The usual head picks a code', on: ['main'],
      tex: [j ? String.raw`\ell = \operatorname{head}(h_t) \in \mathbb{R}^{16384},\ \ \ell_{\text{cfg}} = \ell_u + 5\,(\ell_c - \ell_u)` : String.raw`\ell = W_{\text{LM}}\,h_t \in \mathbb{R}^{184622},\ \text{restricted to } 32768\ \text{codes} + \text{EOL}`],
      shape: j ? '4,096 → 4,096 → 16,384' : '4,096 → 184,622', text: j ? 'The last hidden state goes through a separate image head (two linear layers) to scores over the 16,384 codes. Guidance pushes the scores away from the unprompted run, by a factor of 5; then a code is sampled.'
        : 'Codes are vocabulary entries 151,854 onward, so the ordinary language-model head scores them. At each step only codes (or, at a row’s end, the newline token) are allowed.' });
    S.push({ id: 'feed', chip: 'Feed back', title: j ? 'Back in through an aligner' : 'Back in as a word', on: ['inner'], tex: [j ? String.raw`x_{t+1} = \operatorname{MLP}\big(E_{\text{gen}}[c_t]\big),\ 8 \to 4096` : String.raw`x_{t+1} = E[c_t]`],
      shape: j ? '8 → 4,096 → 4,096' : 'one embedding row', text: j ? 'The chosen code is looked up in a small table (8 numbers per code) and an MLP maps it to the language model’s 4,096 dims: the next input. Reading images uses a different encoder (SigLIP); Janus keeps understanding and drawing apart.'
        : 'The chosen code is the next input token, looked up in the same embedding table as words.' });
    S.push({ id: 'grid', chip: `${fmt(p.codes)} codes`, title: 'Row by row', on: ['grid'], tex: [j ? String.raw`24 \times 24 = 576\ \text{steps}` : String.raw`${p.grid[0]} \times (${p.grid[1]} + 1) = ${fmt(p.steps).replace(/,/g, '{,}')}\ \text{steps}`],
      shape: `${fmt(p.steps)} forward passes`, text: j ? 'Codes are written left to right, top to bottom: 576 passes of the language model for one image, each conditioned on all codes above and to the left.'
        : `${p.grid[0]} rows of ${p.grid[1]} codes, each row closed by an end-of-line token: ${fmt(p.steps)} passes, as many as a 6,000-word essay.` });
    S.push({ id: 'decode', chip: 'To pixels', title: 'The VQ decoder', on: ['dec', 'out'], tex: [String.raw`${p.grid[0]} \times ${p.grid[1]} \times ${m.codeDim} \to ${p.pixels[0]} \times ${p.pixels[1]} \times 3`],
      shape: `${p.pixels[1]} × ${p.pixels[0]} pixels`, text: `Each code is replaced by its codebook vector (${m.codeDim} numbers) and a convolutional decoder upsamples ${m.down}× in each direction to pixels. Fine detail the codebook cannot express is lost: this decoder sets the image quality ceiling.` });
    S.push({ id: 'cost', chip: 'Cost', title: 'What drawing costs', on: ['cost'], tex: [String.raw`2 \times ${fmt(p.steps).replace(/,/g, '{,}')} \times 2 \cdot ${d.kv} \cdot ${d.dh} \cdot ${d.layers}`],
      shape: `${fmtMiB(p.kv)} of cache`, text: `${fmt(p.steps)} steps for each of the 2 sequences; their caches hold ${fmtMiB(p.kv)} by the end. A diffusion model would instead refine all pixels at once over a few dozen passes.` });
    return S;
  }
  const p = omniPlan(v), d = m.dec, vid = p.grids * p.perGrid;
  S.push({ id: 'inputs', chip: 'Inputs', title: 'Text, image, sound, video', on: ['in'], tex: [String.raw`\text{a } ${v}\,\text{s clip: } ${2 * v}\ \text{frames} + ${fmt(v * SR).replace(/,/g, '{,}')}\ \text{samples}`],
    shape: `${v} s · 640 × 360 · with sound`, text: `A ${v}-second clip and a question. Frames are sampled at 2 per second (as Qwen's own utilities do), the sound at 16 kHz.` });
  S.push({ id: 'encoders', chip: 'Encoders', title: 'One encoder per kind of input', on: ['enc'], tex: [String.raw`\text{video: } ${p.grids} \times ${p.perGrid},\quad \text{audio: } 13/\text{s} \to ${p.audioTokens}`],
    shape: `${fmt(vid)} + ${fmt(p.audioTokens)} tokens`, text: `The vision encoder of Qwen3-VL turns each pair of frames into ${p.perGrid} tokens; an audio encoder trained from scratch (AuT, 32 layers) turns each second of sound into 13 tokens. Both map into the Thinker's 2,048 dims.` });
  S.push({ id: 'clock', chip: 'One clock', title: 'Interleaved by time', on: ['seq'], tex: [String.raw`t_{\text{pos}} = 13 \times \text{seconds}`],
    shape: `${fmt(p.order.length)} tokens · ${p.positions} time positions`, text: 'Each second, the video grid comes first, then that second’s 13 audio tokens. Every audio token is one position; a video grid starting at second s gets time position 13 s. So what is seen and what is heard at the same moment sit next to each other.' });
  S.push({ id: 'think', chip: 'Think', title: 'The Thinker answers', on: ['lm'], tex: [String.raw`48\ \text{layers},\ 128\ \text{experts},\ 8\ \text{active}`],
    shape: `${fmt(d.layers)} layers · 30B (3B active)`, text: 'The Thinker reads the whole interleaved sequence with 3D positions (time, row, column) and writes the answer as text.' });
  S.push({ id: 'speak', chip: 'Speak', title: 'The Talker speaks it', on: ['out'], tex: [String.raw`12.5\ \text{frames/s} \times 16\ \text{codes} \to 24\,\text{kHz}`], shape: 'see Speech out',
    text: 'The Talker, code predictor and Code2Wav of the Speech out tab turn the answer into sound while it is being written.' });
  const cache = p.order.length * kvPerToken(d);
  S.push({ id: 'cost', chip: 'Cost', title: 'What the clip costs', on: ['cost'], tex: [String.raw`${fmt(p.order.length).replace(/,/g, '{,}')} \times 2 \cdot ${d.kv} \cdot ${d.dh} \cdot ${d.layers}`],
    shape: `${fmtMiB(cache)} at 16 bits`, text: `${fmt(vid)} video and ${fmt(p.audioTokens)} audio tokens: ${fmtMiB(cache)} of the Thinker's cache. Video dominates: one second of 640 × 360 video costs ${p.perGrid} tokens, one second of sound 13.` });
  return S;
}

const CLOSE = {
  qwen2audio: { title: 'From frames to tokens', sub: 'the first 0.16 s of sound, at each stage', text: 'Every 10 ms gives one spectrogram frame; the stride-2 convolution halves that to 50 vectors per second; averaging pairs halves it again: 25 tokens per second, one per 40 ms.' },
  voxtral: { title: 'Chunks, then stacks of four', sub: 'how the audio is cut, and the first 0.16 s', text: 'Each 30-s chunk is encoded alone: attention never crosses a chunk boundary, which keeps the cost linear in length. Within a chunk, four consecutive 20-ms vectors become one token: 12.5 tokens per second, one per 80 ms.' },
  qwen3vl: { title: 'The timeline Qwen3-VL sees', sub: 'sampled frames, the pairs they form, and each pair’s timestamp', text: 'Frames are sampled 0.5 s apart (further when the 768-frame cap applies). Each pair shares one grid of tokens and one timestamp, the average of its two frames’ times.' },
  llava_ov: { title: 'Sparse frames, pooled', sub: '32 frames over the video, and one frame’s 27 × 27 → 14 × 14', text: 'The 32 frames are spread evenly, whatever the length; a 20-minute video gets one frame every 37.5 seconds. Pooling keeps a quarter of each frame’s vectors.' },
  qwen3omni: { title: 'Who writes which code', sub: '16 codebooks × the first frames of the answer', text: 'Codebook 1 of each frame comes from the Talker (one pass per frame); codebooks 2–16 from the code predictor (one small pass each). Code2Wav needs all 16 of a frame to make its 80 ms of sound.' },
  moshi: { title: 'The 17 streams, step by step', sub: 'text, Moshi’s 8 codebooks (delayed) and the user’s 8', text: 'Each column is one 80-ms step. Helium reads a whole column; the depth transformer writes Moshi’s 8 codes of a column from top to bottom. Codebooks 2–8 sit one step to the right of codebook 1.' },
  janus: { title: 'An image as 576 codes', sub: 'the example image as Janus would write it: 24 × 24 codes, in raster order', text: 'Each cell is one code: one entry of the 16,384-entry codebook, standing for 16 × 16 pixels. The order of writing is left to right, top to bottom; guidance mixes two runs at every cell.' },
  emu3: { title: 'An image as a page of text', sub: 'codes in rows, each row ended by a newline token', text: 'Emu3 writes codes like characters on a page: a row of codes, an end-of-line token, the next row. The size line “90*90” before the image tells it how long the rows are.' },
  omni: { title: 'Three seconds on one clock', sub: 'the order of video (V) and audio (A) tokens, and their time positions', text: 'The first grid of video starts at time position 0, the next at 13, then 26; the audio tokens of each second fill the positions between. Ordering by time lets attention match a sound to the frame it belongs to.' },
};

const SOURCES = {
  audio: [
    { by: 'Chu et al.', year: 2024, title: 'Qwen2-Audio Technical Report', in: 'arXiv:2407.10759', url: 'https://arxiv.org/abs/2407.10759' },
    { by: 'Liu et al. (Mistral AI)', year: 2025, title: 'Voxtral', in: 'arXiv:2507.13264', url: 'https://arxiv.org/abs/2507.13264' },
    { by: 'Radford et al.', year: 2023, title: 'Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)', in: 'ICML 2023', url: 'https://arxiv.org/abs/2212.04356' },
  ],
  video: [
    { by: 'Qwen Team', year: 2025, title: 'Qwen3-VL Technical Report', in: 'arXiv:2511.21631', url: 'https://arxiv.org/abs/2511.21631' },
    { by: 'Li et al.', year: 2024, title: 'LLaVA-OneVision: Easy Visual Task Transfer', in: 'TMLR', url: 'https://arxiv.org/abs/2408.03326' },
    { by: 'Zhai, Mustafa, Kolesnikov & Beyer', year: 2023, title: 'Sigmoid Loss for Language Image Pre-Training (SigLIP)', in: 'ICCV 2023', url: 'https://arxiv.org/abs/2303.15343' },
  ],
  speech: [
    { by: 'Qwen Team', year: 2025, title: 'Qwen3-Omni Technical Report', in: 'arXiv:2509.17765', url: 'https://arxiv.org/abs/2509.17765' },
    { by: 'Défossez et al.', year: 2024, title: 'Moshi: a speech-text foundation model for real-time dialogue', in: 'arXiv:2410.00037', url: 'https://arxiv.org/abs/2410.00037' },
    { by: 'Zeghidour et al.', year: 2021, title: 'SoundStream: An End-to-End Neural Audio Codec (residual vector quantization)', in: 'IEEE/ACM TASLP', url: 'https://arxiv.org/abs/2107.03312' },
  ],
  imagegen: [
    { by: 'Chen et al.', year: 2025, title: 'Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling', in: 'arXiv:2501.17811', url: 'https://arxiv.org/abs/2501.17811' },
    { by: 'Wang et al.', year: 2024, title: 'Emu3: Next-Token Prediction is All You Need', in: 'arXiv:2409.18869', url: 'https://arxiv.org/abs/2409.18869' },
    { by: 'Ho & Salimans', year: 2022, title: 'Classifier-Free Diffusion Guidance', in: 'NeurIPS 2021 Workshop', url: 'https://arxiv.org/abs/2207.12598' },
    { by: 'Esser, Rombach & Ommer', year: 2021, title: 'Taming Transformers for High-Resolution Image Synthesis (VQGAN)', in: 'CVPR 2021', url: 'https://arxiv.org/abs/2012.09841' },
  ],
  omni: [
    { by: 'Qwen Team', year: 2025, title: 'Qwen3-Omni Technical Report', in: 'arXiv:2509.17765', url: 'https://arxiv.org/abs/2509.17765' },
    { by: 'Xu et al.', year: 2025, title: 'Qwen2.5-Omni Technical Report (Thinker–Talker, TMRoPE)', in: 'arXiv:2503.20215', url: 'https://arxiv.org/abs/2503.20215' },
  ],
};


/* ---------------------------------------------------------------- section text per modality ---------------------------------------------------------------- */
const WHOLE = {
  images: ['From pixels to tokens', 'One image on its way into the language model. Click any part, or step through it.'],
  audio: ['From sound to tokens', 'A recording on its way into the language model. Click any part, or step through it.'],
  video: ['From frames to tokens', 'A video on its way into the language model. Click any part, or step through it.'],
  speech: ['From text to speech', 'How the answer becomes sound. Click any part, or step through it.'],
  imagegen: ['From text to pixels', 'How the language model writes an image. Click any part, or step through it.'],
  omni: ['Everything on one clock', 'A video with sound, a question, and a spoken answer, in one model. Click any part, or step through it.'],
};
const COST_TITLE = { images: 'What one image costs', audio: 'What the sound costs', video: 'What the video costs', speech: 'What speaking costs',
  imagegen: 'What drawing one image costs', omni: 'What the clip costs' };
const WHY = {
  images: ['Image tokens are paid for twice: they take up context, and every one of them stores keys and values in every layer, like a word. A 576-token image is about 14% of LLaVA-1.5’s 4,096-token context.',
    'The four designs trade detail for cost differently. LLaVA and Gemma spend the same on every image, so a small image is overpaid and a large one loses detail (LLaVA crops it, Gemma squashes it). Qwen3-VL pays in proportion to the pixels, so detail is kept but a large image is expensive. Llama 3.2 keeps the text short — one token per image — but caches 6,404 image positions in each cross-attention layer, empty tiles included.'],
  audio: ['Both models start from the same Whisper encoder; the difference is what comes after. Averaging pairs keeps 25 tokens per second, stacking four keeps 12.5: half the cost for the same seconds, with the detail folded into wider vectors instead of dropped.',
    'Qwen2-Audio reads one 30-second window, like Whisper itself. Voxtral encodes each 30-second chunk separately, so its encoder cost grows only linearly and a 40-minute recording fits in a 32k context.'],
  video: ['Video multiplies the cost of an image by the number of frames read. Qwen3-VL keeps two frames per second and shrinks each frame as the video grows, so long videos keep their timing but lose sharpness; the timestamps tell the model when each grid happened.',
    'LLaVA-OneVision always reads 32 frames, so the cost is fixed but a long video is seen through gaps of half a minute; nothing in its input says how much time passed between two frames.'],
  speech: ['Speech is written as a codec’s tokens: 12.5 frames per second, each frame a stack of codes from coarse to fine. Writing every code with the big model would mean 200 big passes per second for Qwen3-Omni; instead the big network runs once per frame and a small one fills in the rest.',
    'Qwen3-Omni keeps thinking and speaking apart: the Thinker writes text, the Talker turns it into speech. Moshi has no turns at all: it reads the user’s voice and writes its own at every step, so it can be interrupted, and can interrupt.'],
  imagegen: ['Writing an image token by token costs one forward pass per code: 576 for Janus-Pro, 8,190 for Emu3, each doubled by classifier-free guidance. The codebook decides how much detail survives: 16 × 16 pixels per code at 16,384 entries, or 8 × 8 at 32,768.',
    'Janus keeps drawing and reading apart: SigLIP features to understand an image, VQ codes with their own head to write one. Emu3 uses one vocabulary for everything, so images and text are written by the same head, at the price of long sequences.'],
  omni: ['Time is the common axis. Audio comes at 13 tokens per second; video at one grid per second, a grid costing as many tokens as the frame’s size requires. The model sees each second’s frames next to that second’s sound.',
    'The same Thinker answers questions about text, images, audio and video; the Talker of the Speech out tab can speak the answer while it is written.'],
};
const FOOT = {
  images: ['Dimensions come from the official config and preprocessor files of the four models. Image sizes, tiles, patch grids and token counts follow the transformers image processors, and parameter counts are those of the transformers models. No weights are loaded and nothing is computed from your image except its size and, for the mosaics, its colours; it never leaves this tab. Cache sizes assume 16-bit values.',
    'The resize rules, the connectors, Qwen’s 3D positions, Gemma’s image mask and Llama’s gated cross-attention are also runnable PyTorch in <code>llm_lab/reference/vision.py</code>, tested against Hugging Face transformers.'],
  audio: ['Dimensions come from the official config files; token counts follow the Whisper feature extractor and the Qwen2-Audio and Voxtral processors. Your audio stays in this tab: only its length and, for the pictures, its waveform are used. Cache sizes assume 16-bit values.',
    'The token counts and both adapters are runnable PyTorch in <code>llm_lab/reference/audio.py</code>, tested against Hugging Face transformers.'],
  video: ['Frame sampling, frame sizes and timestamps follow transformers’ Qwen3VLVideoProcessor with the per-frame pixel cap of Qwen’s own utilities; LLaVA-OneVision’s pooling follows its transformers model. Your video stays in this tab: only its length, size and a few frames for the pictures are used.',
    'The rules and the pooling are runnable PyTorch in <code>llm_lab/reference/video.py</code>, tested against Hugging Face transformers.'],
  speech: ['Dimensions come from the official config files. Frame rates, codebooks, the upsampling of Code2Wav and Moshi’s delays follow the transformers models; the sounds on this page are drawn, not generated.',
    'The plans and delays are in <code>llm_lab/reference/speech.py</code>, tested against Hugging Face transformers, including the parameter counts of every part.'],
  imagegen: ['Dimensions come from the official config files. Code grids follow Janus’s configuration and Emu3’s processor; classifier-free guidance is transformers’ own. The mosaics show the page’s example image as codes would tile it, not a generated image.',
    'Janus’s generation layers and the grid rules are runnable PyTorch in <code>llm_lab/reference/imagegen.py</code>, tested against Hugging Face transformers.'],
  omni: ['The interleaving, Qwen3-Omni’s audio token count (13 per second) and its time positions follow its transformers processor. Video is sampled at 2 frames per second, as Qwen’s own utilities do; frames of 640 × 360 stay that size.',
    'The plan is in <code>llm_lab/reference/speech.py</code>, tested against Hugging Face transformers.'],
};

function allTex() {
  const out = [];
  for (const mode of MODES) {
    if (mode.key === 'images') continue;
    const c = CONTROLS[mode.key];
    for (const k of mode.models) for (const v of c.values) steps(k, v).forEach(s => s.tex.forEach(t => out.push(t)));
  }
  return [...new Set(out)];
}

const MEDIA = { MODES, MODE, MODELS, CONTROLS, SR, HOP, CHUNK_SAMPLES, VID, SPEECH, GEN, kvPerToken, secs, audioPlan, sampleFrames, videoSmartResize,
  videoPlan, speechPlan, moshiDelays, omniAudioTokens, omniInterleave, omniPlan, emu3Grid, genPlan, summary, TITLE, LEAD, steps, CLOSE, SOURCES, WHOLE, COST_TITLE, WHY, FOOT, allTex };
MM.MEDIA = MEDIA;
if (typeof module !== 'undefined' && module.exports) module.exports = MEDIA;
})(typeof window !== 'undefined' ? window : globalThis);
