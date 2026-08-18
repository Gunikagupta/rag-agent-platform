# 🧠 RoachMind: Autonomous RAG Agent with 4-Layer Memory Architecture

**RoachMind** is a production-grade, distributed AI Agent platform powered by **CockroachDB Native Vector Search (`VECTOR(384)`)**, **Hybrid Dense-Sparse Search (BM25 + BGE)**, and a **4-Layer Memory Architecture**. Built for high-concurrency multi-turn reasoning, RoachMind goes beyond simple conversation logging by actively retrieving, synthesizing, and persisting agentic state across sessions.

---

## 🌟 Strategic Key Differentiators

* **Active Agentic Memory Design:** Rather than statically appending chat history, RoachMind treats memory as an active retrieval target (`recall_memory`). The agent queries past conversational turns from CockroachDB to resolve cross-turn dependencies dynamically.
* **CockroachDB Native Distributed Vector Indexing:** Utilizes CockroachDB's native `VECTOR(384)` data type and L2/Cosine Distance operators (`<->`) for resilient, zero-downtime similarity search across distributed nodes.
* **Hybrid Search + Cross-Encoder Reranking:** Combines CockroachDB dense embeddings (`BAAI/bge-small-en-v1.5`) with sparse BM25 retrieval, followed by `ms-marco-MiniLM-L-6-v2` cross-encoder reranking for maximum precision.
* **4-Layer Memory Taxonomy:** Structured prompt engineering and state management enforcing Episodic, Semantic, Procedural, and State Memory separation.

---

## 🏗️ System Architecture
