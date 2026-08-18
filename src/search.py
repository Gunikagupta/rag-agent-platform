import numpy as np
import psycopg2
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# 1. Configuration
CONN_STR = "postgresql://gunika:rpEVIqirX3d0gYZOjk-cTQ@joyous-runner-32378.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=require"

# 2. Initialize Models
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-Max normalization to scale scores into [0, 1]."""
    scores = np.array(scores, dtype=float)
    if len(scores) == 0 or scores.max() == scores.min():
        return np.zeros_like(scores)
    denom = scores.max() - scores.min()
    if denom == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / denom


class HybridSearcher:
    def __init__(self, corpus_chunks: list[dict], conn_str: str = CONN_STR):
        self.conn_str = conn_str
        self.corpus_chunks = corpus_chunks

        # Mapping by chunk ID for BM25/Dense candidate lookup
        self.chunk_lookup = {str(c["chunk_id"]): c for c in corpus_chunks}

        print("Building BM25 index over corpus...")
        tokenized_corpus = [c["text"].lower().split() for c in corpus_chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully.")

    def search(
        self,
        query: str,
        top_k_retrieval: int = 15,
        top_k_final: int = 5,
        alpha: float = 0.5
    ) -> list[dict]:
        # --- A. DENSE RETRIEVAL (CockroachDB Vector Search) ---
        query_vector = embedder.encode(query).tolist()

        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()

        # CockroachDB nearest-neighbor query using L2 vector distance (<->)
        cur.execute(
            """
            SELECT chunk_id, paper_id, title, text, (embedding <-> %s::VECTOR) AS distance
            FROM paper_embeddings
            ORDER BY distance ASC
            LIMIT %s;
            """,
            (str(query_vector), top_k_retrieval * 2)
        )
        dense_rows = cur.fetchall()
        cur.close()
        conn.close()

        # Convert vector distances to similarity scores (1 / (1 + distance))
        dense_scores_raw = {}
        dense_payloads = {}
        for row in dense_rows:
            cid, paper_id, title, text, dist = row
            dense_scores_raw[str(cid)] = 1.0 / (1.0 + float(dist))
            dense_payloads[str(cid)] = {
                "chunk_id": cid,
                "paper_id": paper_id,
                "title": title,
                "text": text
            }

        # --- B. SPARSE RETRIEVAL (BM25) ---
        tokenized_query = query.lower().split()
        bm25_scores_raw = self.bm25_index.get_scores(tokenized_query)

        bm25_scores_dict = {
            str(self.corpus_chunks[i]["chunk_id"]): bm25_scores_raw[i]
            for i in range(len(self.corpus_chunks))
        }

        top_bm25_indices = np.argsort(bm25_scores_raw)[::-1][:top_k_retrieval * 2]
        candidate_ids = set(dense_scores_raw.keys()).union(
            {str(self.corpus_chunks[i]["chunk_id"]) for i in top_bm25_indices}
        )

        # --- C. SCORE NORMALIZATION & HYBRID FUSION ---
        cand_ids_list = list(candidate_ids)

        raw_dense_vals = np.array([dense_scores_raw.get(cid, 0.0) for cid in cand_ids_list])
        raw_bm25_vals = np.array([bm25_scores_dict.get(cid, 0.0) for cid in cand_ids_list])

        norm_dense = normalize_scores(raw_dense_vals)
        norm_bm25 = normalize_scores(raw_bm25_vals)

        hybrid_candidates = []
        for i, cid in enumerate(cand_ids_list):
            hybrid_score = (alpha * norm_dense[i]) + ((1.0 - alpha) * norm_bm25[i])

            payload = self.chunk_lookup.get(cid, {})
            if cid in dense_payloads:
                payload.update(dense_payloads[cid])

            if payload:
                hybrid_candidates.append({
                    "id": cid,
                    "payload": payload,
                    "hybrid_score": hybrid_score
                })

        top_hybrid_candidates = sorted(
            hybrid_candidates,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:top_k_retrieval]

        # --- D. CROSS-ENCODER RERANKING ---
        pairs = [[query, item["payload"].get("text", "")] for item in top_hybrid_candidates]
        rerank_scores = reranker.predict(pairs)

        for i, item in enumerate(top_hybrid_candidates):
            item["rerank_score"] = float(rerank_scores[i])

        final_top_k = sorted(
            top_hybrid_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k_final]

        results = []
        for item in final_top_k:
            p = item["payload"]
            results.append({
                "paper_id": str(p.get("paper_id", "")),
                "title": p.get("title", ""),
                "text": p.get("text", "")
            })

        return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    json_matches = list(Path(".").rglob("fulltext_chunks.json"))
    if not json_matches:
        raise FileNotFoundError("Could not find fulltext_chunks.json")

    with open(json_matches[0], "r") as f:
        corpus = json.load(f)

    searcher = HybridSearcher(corpus_chunks=corpus)

    results = searcher.search(
        query="Why did Transformers replace RNNs?",
        top_k_retrieval=15,
        top_k_final=5,
        alpha=0.5
    )

    print("\n--- TOP RERANKED RESULTS ---")
    for idx, payload in enumerate(results, 1):
        print(f"{idx}. [{payload['paper_id']}] {payload['title']}\n   {payload['text'][:200]}...\n")