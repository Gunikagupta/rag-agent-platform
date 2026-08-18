# Production RAG Platform — Veritas-RAG

## Status: Day 6/15 — Hybrid Search & Metric Baseline Verification Complete

## Corpus & Architecture
* **Corpus**: ~300 arXiv paper abstracts focused on LLM evaluation, benchmark reliability, and causal discovery in ML.
* **Chunking Strategy**: Sentence & boundary-aware sliding window chunking (~300 tokens, 50-token overlap) to preserve contextual boundaries across chunk seams.
* **Embeddings**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
* **Vector Database**: Local Qdrant instance.
* **Inference Engine**: Groq API (`llama-3.1-8b-instant` for dev/test iterations; `llama-3.3-70b-versatile` for evaluation scoring).

---

## Metric Tracking & Benchmark Milestones

| Pipeline Iteration | Factual Recall@5 | Chunk Precision@5 | Correct Refusal Rate | Notes / Target |
| :--- | :--- | :--- | :--- | :--- |
| **Milestone 1: Dense Retrieval Only** | **92.31%** (12/13) | **76.92%** | **100.00%** (10/10) | Baseline setup with `bge-small-en-v1.5` + Qdrant |
| **Milestone 2: Hybrid Search (BM25 + Dense)** | *Pending* | *Pending* | *Pending* | Keyword + vector score fusion |
| **Milestone 3: Hybrid + Cross-Encoder Rerank** | *Pending* | *Pending* | *Pending* | Top-15 candidates reranked to Top-5 via `ms-marco-MiniLM-L-6-v2` |

---

## Known Limitations & Edge Cases Handled

* **Generation Sampling Bounds**: Resolved infinite repetition loops on mathematical formulas during greedy decoding (`temperature=0`) by setting `temperature=0.2` and introducing a mild `frequency_penalty`.
* **Guardrail Evaluation**: Refusal detection initially relied on exact string matching, leading to undercounted true refusals; updated string matching logic to handle broader phrasing variations.
* **Eval Harness Isolation**: Confirmed system separates **retrieval recall** (vector hit inside Top-K context) from **generation errors** (LLM context misalignment or repetition loops), preventing evaluation script contamination.
* **Test Case Hardening**: Out-of-scope and false-premise test sets currently feature explicit non-domain queries. Next iteration will test near-miss out-of-scope cases within the AI/ML domain.