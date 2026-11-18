# PyTorch reference implementations

`src/llm_lab/reference/` implements every mechanism the explorer draws, as plain PyTorch that is meant
to be read next to the page. Each of the six tabs is a complete, runnable model built from its
published config; Qwen's block-sparse QSA is a single layer, since no released checkpoint uses it yet.

| tab | model | where to read it |
|---|---|---|
| MHA | GPT-2 small | `attention.py` — `GroupedQueryAttention` with as many K/V heads as query heads |
| MQA | Gemma 2B | the same class with one K/V head |
| GQA | Qwen2.5-0.5B | the same class, `repeat_interleave` of 2 K/V heads to 14 query heads |
| MLA | DeepSeek-V2-Lite | `mla.py` — `MultiHeadLatentAttention`, `_expanded` and `_absorbed` |
| GDN | Qwen3-Next-80B-A3B | `gated_deltanet.py` — `delta_rule_recurrent`, `delta_rule_chunked`, `GatedDeltaNet`; gated attention in `attention.py` |
| DSA | DeepSeek-V3.2 | `mla.py` — `LightningIndexer` feeding a top-k mask into MLA |
| QSA | Qwen3.8-Flash-Next | `qsa.py` — `QSAIndexer` (block pooling) and `QwenSparseAttention` |

Shared pieces are in `layers.py` (RMSNorm with the `1 + w` variant, RoPE in half-split and interleaved
layouts, partial rotary, YaRN, GELU/SiLU MLPs, mixture of experts with softmax, sigmoid and group-limited
routers), the model in `model.py` (`DecoderLM`, `Cache`), the eight decoding methods of the page in
`decoding.py`, and the Hugging Face bridge in `hf.py`.

```bash
pip install -e ".[dev]"
python -m llm_lab.reference                       # the six models: parameters, active parameters, cache
python -m pytest tests/test_reference_models.py   # about 5 seconds on a CPU
python -m llm_lab.reference --run openai-community/gpt2   # a real checkpoint (downloads it)
```

```python
import torch
from llm_lab.reference import CONFIGS, DecoderLM

cfg = CONFIGS['mla'].tiny()          # DeepSeek-V2-Lite's architecture at toy size
model = DecoderLM(cfg)
cache = model.new_cache()
logits = model(torch.tensor([[1, 2, 3]]), cache)
print(cache.numbers_per_layer())     # one latent (dc) and one rotary key (dR) per token and layer
```

## How it is verified

`tests/test_reference_models.py` builds each architecture twice, once with this code and once with
transformers (`AutoModelForCausalLM.from_config` with the equivalent config), gives the transformers
model random weights, copies them across with `load_hf_weights` and compares:

- **logits** for all six models, max difference ≤ 6 · 10⁻⁶; small configs keep every mechanism
  (K/V sharing, latent compression, YaRN, an indexer budget smaller than the sequence, the 3 : 1
  GDN/attention pattern, every router type), and every parameter is random, norms included;
- **greedy generation**, token for token against `generate()`, for GQA, GDN and DSA;
- **parameter counts at full size**, on the meta device: GPT-2 124,439,808 · Gemma 2B 2,506,172,416 ·
  Qwen2.5-0.5B 494,032,768 · DeepSeek-V2-Lite 15,706,484,224 · Qwen3-Next 79,674,391,296 ·
  DeepSeek-V3.2 671,877,929,216, identical to transformers;
- **QSA** against `Qwen4ExpTextAttention` from transformers, 23 tokens so the block selection matters
  (max difference 0).

and, without transformers:

- the **KV cache**: prefill plus one-token steps equals one full forward pass, and the numbers stored
  per token equal `cache_values_per_token` (MLA 576, DSA 704, …; GDN a fixed 548,864 per layer),
  which `tests/test_explorer.py` checks against the page;
- **absorbed MLA equals expanded MLA**, the identity that lets DeepSeek cache only the latent;
- **chunked Gated DeltaNet equals the recurrent form**, including a padded last chunk and the carried
  state, and the delta rule stores a written key exactly (S k = v);
- **DSA with an unlimited budget is plain MLA**, and a small budget changes the output;
- **YaRN**: frequencies and attention factor equal transformers' `ROPE_INIT_FUNCTIONS['yarn']`, the
  softmax scale is the page's 1.59 (V2-Lite) and 1.87 (V3.2), pairs 0–10 keep their speed and 23–31
  are slowed 40×;
- **decoding** on the page's illustrative logits: top-k 3, top-p 0.6 → 2 tokens, min-p 0.2 → 4,
  typical 0.5 → 4, top-k 5 at τ = 0.8 gives “smaller” 46%; 40,000 speculative-sampling draws are
  distributed as the target p, with acceptance rate Σ min(p, q); beam search finds a sequence greedy
  decoding misses.

The tests use random weights, so they need no downloads. `python -m llm_lab.reference --run <model id>`
loads a real checkpoint through the same `load_hf_weights` and prints the logit difference to
transformers.

## Vision-language models

`vision.py` covers the Multimodal page: how LLaVA-1.5-7B, Qwen3-VL-8B, Llama 3.2 11B Vision and
Gemma 3 4B turn an image into what their language model reads. It holds the four configs, the
processors' resize rules (`smart_resize`, `tiled_canvas`, `fit_to_canvas`, `image_plan`), the cache an
image costs (`image_cache_values`), Qwen's 3D positions (`mrope_positions`, `mrope_pair_axes`),
Gemma's bidirectional image mask (`gemma_attention_mask`) and the connectors as modules with
transformers' weight names (`llava_projector`, `qwen_merger`, `gemma_projector`,
`gated_cross_attention_block`). The vision encoders and decoders themselves are standard ViTs and
decoders, so they are not repeated here.

`tests/test_vision_models.py` checks, against transformers:

- the **resize rules** on 2,000 random sizes (`smart_resize`, `get_optimal_tiled_canvas`,
  `get_image_size_fit_to_canvas`), and the **token counts** against the image processors themselves
  (CLIP, Qwen2-VL, Mllama, Gemma 3) on eight sizes from 100 × 100 to 4032 × 3024;
- the **connectors**, with random weights, against `LlavaMultiModalProjector`,
  `Qwen3VLVisionPatchMerger` (both norm placements), `Gemma3MultiModalProjector` and
  `MllamaCrossAttentionDecoderLayer` with its cross-attention mask, including rows that see no image;
- **M-RoPE** positions against `Qwen3VLModel.get_rope_index` and the interleaved frequency layout
  against the rotary embedding; **Gemma's mask** against its image-block ids;
- **parameter counts at full size**, per part: LLaVA-1.5-7B 7,063,427,072 · Qwen3-VL-8B
  8,767,123,696 · Llama 3.2 11B Vision 10,670,220,835 · Gemma 3 4B 4,300,079,472;
- that **Llama 3.2 caches four tiles** of 1,601 positions per cross-attention layer, used or not.

`tests/test_explorer.py` then runs the page's own arithmetic (`web/src/mm-content.js`) in node on 400
image sizes and checks it gives the same plans, costs, positions and masks as `vision.py`.

## Choices

- **Readable over fast**: float32, eager attention with an explicit mask, no fused kernels or FP8.
  DSA's indexer runs in FP8 after a Hadamard rotation in DeepSeek's code; both preserve the dot products
  up to rounding, so this float version selects the same tokens.
- **One `ModelConfig`** for all six models, with the published values as constants in `config.py` and
  a `tiny()` method that shrinks widths while keeping every structural choice.
- **Weights stored the way the math is written** (for example, GDN's fused `in_proj_qkvz` is split into
  `q_proj`, `k_proj`, `v_proj`, `z_proj`; experts are stacked `[E, I, H]` tensors); `hf.py` does the
  conversion, so the model code never mentions a checkpoint layout.
