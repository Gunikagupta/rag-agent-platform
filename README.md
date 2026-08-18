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


```

```
                              +-----------------------+
                              |     Streamlit UI      |
                              +-----------+-----------+
                                          |
                                          v
                               +---------------------+
                               |  RoachMind Agent    |
                               |    (src/agent.py)   |
                               +----------+----------+
                                          |
                 +------------------------+------------------------+
                 |                                                 |
                 v                                                 v
    +-------------------------+                       +-------------------------+
    |     Hybrid Searcher     |                       |    Agentic Memory Tool  |
    |     (src/search.py)     |                       |    (src/memory.py)      |
    +------------+------------+                       +------------+------------+
                 |                                                 |
     +-----------+-----------+                                     |
     |                       |                                     |
     v                       v                                     v

```

+------------------+   +-------------------+              +-------------------------+
|  CockroachDB     |   |   BM25 Index      |              |   CockroachDB           |
|  `paper_embeds`  |   |   (Local Sparse)  |              |   `conversations`       |
|  VECTOR(384)     |   +---------+---------+              |   (Episodic History)    |
+--------+---------+             |                        +------------+------------+
|                       |                                     |
+-----------+-----------+                                     |
|                                                 |
v                                                 v
+-------------------------+                       +-------------------------+
| Cross-Encoder Reranker  |                       | 4-Layer Memory Prompt   |
| (ms-marco-MiniLM-L-6-v2)|                       | Synthesis Engine        |
+------------+------------+                       +------------+------------+
|                                                 |
+------------------------+------------------------+
|
v
+-----------------------+
| Multi-Turn Synthesis  |
| & CockroachDB Persist |
+-----------------------+

```

---

## 🧬 The 4-Layer Memory Model

RoachMind organizes agent cognitive processing into four explicit operational tiers:

| Layer | Type | Mechanism & Storage | Function |
| :--- | :--- | :--- | :--- |
| **1. Episodic Memory** | Session-level Turn History | CockroachDB `conversations` table | Stores and queries prior interaction turns within the active session key. |
| **2. Semantic Memory** | Fact & Domain Knowledge | CockroachDB `paper_embeddings` (`VECTOR(384)`) | Retrieves corpus chunks via hybrid vector + BM25 similarity search. |
| **3. Procedural Memory** | Actionable Reasoning | Dynamic Hybrid Fusion Pipeline | Reranks candidates and enforces step-by-step reasoning logic. |
| **4. State Memory** | Environment State | CockroachDB persistent transaction key | Tracks active session flags, execution latency, and system audit logs. |

---

## 🛠️ Database Schema

CockroachDB tables are initialized via `src/db.py`:

```sql
-- 1. Paper Vector Embeddings Table
CREATE TABLE IF NOT EXISTS paper_embeddings (
    chunk_id STRING PRIMARY KEY,
    paper_id STRING,
    title STRING,
    text STRING,
    embedding VECTOR(384)
);

-- 2. Agent Conversational Memory Table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id STRING,
    question STRING,
    answer STRING,
    retrieved_chunk_ids STRING[],
    created_at TIMESTAMP DEFAULT now()
);

```

---

## 🚀 Getting Started

### Prerequisite Environment Variables

Export your CockroachDB connection string:

```bash
export COCKROACH_CONN_STRING="postgresql://<username>:<password>@<host>:26257/defaultdb?sslmode=require"

```

### Installation

1. Clone the repository and navigate to the project root:
```bash
git clone [https://github.com/your-org/roachmind.git](https://github.com/your-org/roachmind.git)
cd roachmind

```


2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. Install required dependencies:
```bash
pip install -r requirements.txt

```



### Database Setup & Verification

1. **Initialize Database Tables:**
```bash
python -m src.db

```


2. **Test Agent Backend Execution:**
```bash
python -m src.agent

```


3. **Launch Interactive Streamlit Dashboard:**
```bash
streamlit run src/app.py

```



---

## 🔎 Example Memory Synthesis Output

```text
🧠 ROACHMIND 4-LAYER MEMORY SYNTHESIS

1. EPISODIC MEMORY:
Retrieved active interaction history for session inc_investigation_session_001 from CockroachDB.

2. SEMANTIC MEMORY:
Matched top vector chunk from "MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning" via CockroachDB VECTOR(384) cosine search:
> "Agents debate in sequence using previous round answers..."

3. PROCEDURAL MEMORY:
Executed hybrid reranking strategy (BM25 + Dense Vector + Cross-Encoder).

4. STATE MEMORY:
Interaction state successfully persisted to CockroachDB conversations table.

```

---

## 📁 Repository Structure

```text
.
├── src/
│   ├── agent.py            # Main Agent Loop & 4-Memory Prompt Formatter
│   ├── db.py               # CockroachDB Schema & Connection Factory
│   ├── memory.py           # Active Memory Persistence & Recall Tools
│   ├── search.py           # Hybrid Dense-Sparse Searcher & Reranker
│   └── skills.py           # RoachMind Operational Skills
├── data/                   # Local Corpus Chunks & Embeddings
├── requirements.txt        # System Dependencies
└── README.md               # Architecture Documentation

```

```

---

<ElicitationsGroup message="What would you like to do next?">
  <Elicitation label="Draft 2-minute video script" query="Write out the exact word-for-word 2-minute video script for me to read during my recording."/>
</ElicitationsGroup>

```