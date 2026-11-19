# Retrieval Lab — first interactive implementation

This adds the first complete interactive slice of the **Retrieval** page to `llm-internals-lab`, consistent with the existing architecture and multimodal explorers.

## Included

- A **five-step tour** from question to retrieved evidence and citation-ready model context.
- Five small **fictional Moonlight Aquarium documents**, indexed as either 10 shorter passages or 5 whole documents.
- Real **BM25 scoring** (k₁ = 1.2, b = 0.75, smoothed non-negative IDF): term breakdowns, an inverted-index view, and ranked results with score bars.
- Editable query, six example questions, and a **top-k** slider to show how evidence selection affects context.
- An assembled **RAG prompt** with source identifiers and citations; a clearly labeled *predefined* answer is shown only when the passage used for that example is retrieved. **No LLM, embeddings, vector DB, or external API is running.**
- Consistent light/dark styling, responsive mobile layout, keyboard-operable controls, and tests.

The page leaves Embeddings, Contrastive Learning, Approximate Nearest Neighbors, and Reranking as the next modules rather than pretending to simulate real neural embeddings.

## Install

Extract the ZIP into the root of the GitHub repository (alongside `web/`, `src/`, and `tests/`) and run:

```bash
python apply_llm_lab_retrieval_v1.py
python -m pytest tests/test_retrieval_browser.py
python -m pytest tests/test_explorer_browser.py
git diff --check
```

Commit and push the changes from your usual workflow. The script modifies both the source templates and the checked-in single-file HTML deployed by GitHub Pages. It also updates `web/build.mjs`, so future builds preserve the Retrieval page. Run `cd web && npm run build` after installing or modifying it **if** your usual build dependencies and font are installed.

The installer is designed for the published `main` layout and for the earlier v1–v7 accessibility changes. It refuses to overwrite an existing retrieval module with different content, and running the installer twice makes no further modifications.

## Technical boundary

The page implements the *lexical retrieval* part of a typical retrieval-augmented generation pipeline and illustrates how retrieved context can be passed to a generator. It does not implement a trained embedding encoder, HNSW, reranking models, or generative inference. These are intentionally reserved for subsequent iterations.

**Sources:** Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* (doi:10.1561/1500000019); Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv:2005.11401).

## Checks already performed

- JavaScript and Python syntax checks.
- Chromium UI smoke tests on **1280px** and **420px** widths: preset switching, custom query, zero results, top-k effect, document granularity, guided tour.
- Installer applied **twice** to a source-layout fixture: second run had no changes.

The actual repository's full test suite still needs to run on your checkout or CI.
