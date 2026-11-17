# LLM Internals Lab

An interactive, offline explorer of how decoder-only language models work: six attention mechanisms,
each shown in a real model with its published dimensions, from text to the next token. Plus readable
PyTorch implementations of every mechanism, tested to give the same logits as Hugging Face `transformers`.

**Live app:** <https://valentinsvelev.github.io/llm-internals-lab/>

![The explorer](docs/screenshot.png)

| Tab | Mechanism | Model |
|---|---|---|
| MHA | multi-head attention | GPT-2 small |
| MQA | multi-query attention | Gemma 2B |
| GQA | grouped-query attention | Qwen2.5-0.5B |
| MLA | multi-head latent attention | DeepSeek-V2-Lite |
| GDN | Gated DeltaNet + gated attention | Qwen3-Next-80B-A3B |
| Sparse | DeepSeek Sparse Attention (and Qwen's QSA) | DeepSeek-V3.2 |

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
page shows. How they are verified: [docs/reference-implementations.md](docs/reference-implementations.md).

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
