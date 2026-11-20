# LLM Internals Lab

An interactive, offline explorer of how language models work, each idea shown in a real model with its
published dimensions: six attention mechanisms, from text to the next token, and how models read
images, sound and video, and speak and draw. Plus readable PyTorch implementations, tested against
Hugging Face `transformers`.

**Live app:** <https://valentinsvelev.github.io/llm-internals-lab/>

![The explorer](docs/screenshot.png)

**Architectures**

| Tab | Mechanism | Model |
|---|---|---|
| MHA | multi-head attention | GPT-2 small |
| MQA | multi-query attention | Gemma 2B |
| GQA | grouped-query attention | Qwen2.5-0.5B |
| MLA | multi-head latent attention | DeepSeek-V2-Lite |
| GDN | Gated DeltaNet + gated attention | Qwen3-Next-80B-A3B |
| Sparse | DeepSeek Sparse Attention (and Qwen's QSA) | DeepSeek-V3.2 |

**Multimodal**, by modality (try your own image, recording or video; it never leaves the browser)

| Modality | Models | What the page shows |
|---|---|---|
| Images | LLaVA-1.5, Qwen3-VL, Llama 3.2 Vision, Gemma 3 | projector, native size with 3D positions, gated cross-attention, pooling |
| Audio | Qwen2-Audio, Voxtral | Whisper features, pooled to 25 or stacked to 12.5 tokens per second |
| Video | Qwen3-VL, LLaVA-OneVision | 2 frames per second with timestamps, or a fixed 32 frames pooled |
| Speech out | Qwen3-Omni, Moshi | codec tokens: a Talker beside the Thinker, or full duplex with a depth transformer |
| Image out | Janus-Pro, Emu3 | images written as VQ codes, with guidance, through their own head or as ordinary words |
| All together | Qwen3-Omni | video and its sound interleaved on one clock, answered in speech |

**Retrieval**: sparse, learned sparse, dense and hybrid retrieval; the first three as a dot product
q · d, hybrid as the fusion of the BM25 and the dense ranking (reciprocal rank fusion, or a weighted sum
of normalized scores). Sparse: five
short documents become a vocabulary of unique words; each document is a vector of BM25 or TF-IDF
weights, the query a vector with a 1 at each of its terms. Learned sparse: a SPLADE-style model
(opensearch-neural-sparse-encoding-v2-distill) weights the 30,522 WordPiece tokens, adding related
terms the text does not contain (expansion). Dense: all-MiniLM-L6-v2 turns every text into 384
numbers of length 1 (computed in advance by `web/scripts/make_embeddings.py`), and a side-by-side
ranking shows where each one wins. The repository currently ships illustrative placeholder vectors and weights
(random, noted in the page's footer): run `python3 web/scripts/make_embeddings.py` (needs huggingface.co) and
rebuild to get MiniLM's. The boxes on the arrows open each step: the tokenizer, the
weighting or the encoder, and the sum behind each score. The top-k documents go into the RAG prompt
with citations (no model runs on this page).

**Reasoning**: one diagram of how a reasoning model thinks, with DeepSeek-R1 as the example. The model
reasons by writing: after `<think>`, every token is one forward pass (embedding, 61 layers of MLA
attention and 8 of 256 experts, LM head, sampling) whose output joins the context the next pass reads.
Step or play through a thought (written for the page) from prefill past a “Wait” to `</think>` and the
answer, and watch attention read one more position and the KV cache grow with every pass. A check after
every token shows how thinking stops: the model draws `</think>` (and later an end token), or the
length limit or a thinking budget cuts it short. Click any part for its details. Parameters and the cache are computed from R1's config and the expert router runs
DeepSeek-V3's rule.

An Agents page is planned.

## Run the explorer

It is one self-contained HTML file: open `src/llm_lab/assets/explorer.html` in a browser.
Or serve it (no dependencies beyond Python):

```bash
pip install -e .
llm-lab --serve                  # http://127.0.0.1:8767
```

## PyTorch reference implementations

`llm_lab.reference` builds all six models from their configs (attention, MLA, YaRN, the lightning
indexer, Gated DeltaNet, mixture of experts, KV caches) and implements the eight decoding methods the
page shows; `llm_lab.reference.vision`, `audio`, `video`, `speech` and `imagegen` hold the multimodal
pipelines (resizing and sampling rules, adapters, positions, masks, codec layouts, generation layers);
`llm_lab.reference.retrieval` holds the Retrieval page's tokenizer, BM25 and TF-IDF vectors and fusion;
and `llm_lab.reference.reasoning` the Reasoning page's numbers and expert routing.
How they are verified:
[docs/reference-implementations.md](docs/reference-implementations.md).

```bash
pip install -e ".[pytorch]"
python -m llm_lab.reference      # parameters, active parameters, cache per token
```

## Tests

```bash
pip install -e ".[dev]"
python -m playwright install chromium
python -m pytest
```

The page is built from [web/](web/README.md). `python -m llm_lab.next_token` replaces its
illustrative next-token logits with real ones.

## Deployment

Pushes to `main` run the tests and then publish the page to GitHub Pages
(`.github/workflows/deploy.yml`). Enable **Settings → Pages → Source: GitHub Actions** once.

## License

MIT. The page bundles Temml (MIT) and a subset of Latin Modern Math (GUST Font License);
see [docs/attribution.md](docs/attribution.md).
