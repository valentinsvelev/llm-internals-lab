# Contributing

Issues and pull requests are welcome.

- **A new mechanism or model:** implement it in `src/llm_lab/reference` with a test that compares it
  with the `transformers` implementation, then add it to the page (`web/src/content.js` for the
  data and text, `web/src/app.js` for the diagram).
- **Numbers on the page** come from published `config.json` files; `tests/test_explorer.py` checks
  them against `src/llm_lab/reference/config.py`.
- **Anything illustrative** (logits, attention weights, expert choices) is labeled as such.

Before submitting: rebuild the page (`web/README.md`), then `python -m pytest` and `ruff check src tests`.
