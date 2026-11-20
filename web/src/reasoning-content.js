/* Reasoning page: one diagram of how a reasoning model thinks, with DeepSeek-R1 as the example. No DOM.
 *
 * The model reasons by writing: the prompt ends in <think>, and each thinking token is one more forward
 * pass of the same network, appended to the context it reads. The architecture numbers are DeepSeek-R1's
 * config.json (the DeepSeek-V3 architecture); parameters and the KV cache per token are computed from it,
 * as llm_lab/reference/reasoning.py does, and the expert router is DeepSeek-V3's rule on demo scores
 * (tests/test_reasoning.py checks all three). The thinking trace is written for the page.
 */
(function (root) {
'use strict';

const PAPER = {by: 'Guo et al.', year: 2025, title: 'DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning',
  in: 'Nature 645, 633–638', url: 'https://www.nature.com/articles/s41586-025-09422-z'};

// How long a reply may get, thinking included: DeepSeek evaluates R1 with at most 32,768 generated tokens (its README);
// R1-Zero's RL capped each output at 32,768 tokens, and at 65,536 after step 8.2k (the paper's Methods).
const MAX_GENERATION = 32768;
const RL_MAX_LENGTH = [32768, 65536];

// The prompt R1 was trained with, paraphrased (the exact wording is the paper's Table 1): reason inside <think>,
// answer inside <answer>, nothing about how to reason.
const TEMPLATE = 'A conversation between User and Assistant. The User asks a question and the Assistant solves it. '
  + 'The Assistant first thinks about the reasoning process and then gives the answer. The reasoning goes inside '
  + '<think> </think> tags and the answer inside <answer> </answer> tags. User: {question} Assistant:';

/* ---------- the base model: DeepSeek-V3's architecture, as in DeepSeek-R1's config.json ---------- */
const ARCH = {name: 'DeepSeek-R1', base: 'DeepSeek-V3', source: 'https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json',
  vocab: 129280, hidden: 7168, layers: 61, denseLayers: 3, heads: 128, qRank: 1536, kvRank: 512, nopeDim: 128, ropeDim: 64, vDim: 128,
  intermediate: 18432, experts: 256, topK: 8, expertDim: 2048, shared: 1, groups: 8, topGroups: 4, scaling: 2.5,
  maxPositions: 163840, mtp: 1, cacheBytesPerValue: 2};
// Parameters by part, as transformers' DeepseekV3ForCausalLM holds them (the multi-token-prediction module aside).
function parameters(a = ARCH) {
  const H = a.hidden, moeLayers = a.layers - a.denseLayers, glu = (i, o) => 3 * i * o;
  const attnPerLayer = H * a.qRank + a.qRank + a.qRank * a.heads * (a.nopeDim + a.ropeDim)      // queries, through a 1,536 latent
    + H * (a.kvRank + a.ropeDim) + a.kvRank + a.kvRank * a.heads * (a.nopeDim + a.vDim)          // keys and values, through a 512 latent
    + a.heads * a.vDim * H;                                                                       // output
  const parts = {
    embedding: a.vocab * H,
    attention: a.layers * attnPerLayer,
    denseFfn: a.denseLayers * glu(H, a.intermediate),
    router: moeLayers * a.experts * H,
    sharedExperts: moeLayers * a.shared * glu(H, a.expertDim),
    routedExperts: moeLayers * a.experts * glu(H, a.expertDim),
    norms: a.layers * 2 * H + H,
    head: a.vocab * H,
  };
  const total = Object.values(parts).reduce((x, y) => x + y, 0);
  const active = total - parts.routedExperts + parts.routedExperts * a.topK / a.experts;
  return {parts, total, active, perExpert: glu(H, a.expertDim), attnPerLayer};
}
// MLA caches, per layer and token, the 512-number latent of keys and values and the 64-number rotary key.
const cacheValuesPerToken = (a = ARCH) => a.layers * (a.kvRank + a.ropeDim);
const cacheBytesPerToken = (a = ARCH) => cacheValuesPerToken(a) * a.cacheBytesPerValue;

/* DeepSeek-V3's router for one token: sigmoid scores; the experts come in 8 groups of 32, a group
 * scores the sum of its 2 best experts, only the best 4 groups may be used, and the 8 best experts
 * among them run. Their weights are the scores renormalized to 1, times 2.5. (The load-balancing bias
 * that also shifts the choice is left at 0 here.) */
function route(scores, a = ARCH) {
  const per = a.experts / a.groups;
  const groupScore = Array.from({length: a.groups}, (_, g) => {
    const top = scores.slice(g * per, (g + 1) * per).slice().sort((x, y) => y - x);
    return top[0] + top[1];
  });
  const groups = groupScore.map((s, g) => [s, g]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, a.topGroups).map(x => x[1]).sort((x, y) => x - y);
  const allowed = new Set(groups);
  const experts = scores.map((s, e) => [s, e]).filter(([, e]) => allowed.has(Math.floor(e / per)))
    .sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, a.topK).map(x => x[1]);
  const sum = experts.reduce((t, e) => t + scores[e], 0);
  return {groupScore, groups, experts, weights: experts.map(e => a.scaling * scores[e] / sum)};
}
// Demo router scores: sigmoid of normal logits from a seeded generator (not a model's).
function demoScores(seed, a = ARCH) {
  let x = seed >>> 0 || 1;
  const u = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return (x + 0.5) / 4294967296; };
  return Array.from({length: a.experts}, () => {
    const z = Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
    return 1 / (1 + Math.exp(-z));
  });
}

/* ---------- thinking: a trace, written for the page ----------
 * The pieces are words and signs, about one token each (the real tokenizer splits differently). */
const TRACE = {
  question: 'If a > 1, what is the sum of the real solutions of √(a − √(a + x)) = x?',
  thinking: 'Square both sides: a − √(a + x) = x², so √(a + x) = a − x². Square again: a + x = (a − x²)², a quartic in x. Two roots look possible. Wait, let me check that step again: √(a + x) = a − x² needs a − x² ≥ 0. The quartic factors as (x² + x + 1 − a)(x² − x − a) = 0. The second factor gives a − x² = −x < 0, so it fails. The first gives x = (√(4a − 3) − 1)/2, which works. One real solution, so the sum is that value.',
  answer: '(√(4a − 3) − 1)/2',
};
const pieces = text => text.match(/[\p{L}\p{N}²]+|[^\s\p{L}\p{N}]/gu) || [];

const API = {PAPER, MAX_GENERATION, RL_MAX_LENGTH, TEMPLATE, ARCH, parameters, cacheValuesPerToken, cacheBytesPerToken, route, demoScores, TRACE, pieces};
if (typeof module !== 'undefined' && module.exports) module.exports = API;
root.REASON = API;
})(typeof window === 'undefined' ? globalThis : window);
