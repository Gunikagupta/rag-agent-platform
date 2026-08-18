import os
import json
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from src.search import HybridSearcher

load_dotenv()
client = Groq()

qdrant_client = QdrantClient("localhost", port=6333)

with open("data/papers.json", "r") as f:
    papers_data = json.load(f)

# Structure chunks so paper_id is unequivocally present
corpus_chunks = []
for idx, paper in enumerate(papers_data):
    # Ensure paper_id matches expected format in eval_set.json (e.g. "2607.18232v1")
    p_id = str(paper.get("paper_id") or paper.get("expected_id") or paper.get("id") or (idx + 1))
    
    corpus_chunks.append({
        "id": idx + 1,  # Numeric ID for BM25 mapping
        "paper_id": p_id,
        "title": paper.get("title", ""),
        "text": f"{paper.get('title', '')}\n{paper.get('abstract', paper.get('text', ''))}".strip()
    })

# Instantiate searcher globally
searcher = HybridSearcher(
    qdrant_client=qdrant_client,
    collection_name="papers",
    corpus_chunks=corpus_chunks
)

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[{i+1}] {chunk.get('title', 'Untitled')}\n{chunk.get('text', '')}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    
    if len(context) > 10000:
        context = context[:10000] + "\n\n[Context truncated]"

    return f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so explicitly — do not make anything up.

Context:
{context}

Question: {query}

Answer (cite sources by their [number]):"""

def answer(query, top_k=5):
    retrieved = searcher.search(
        query=query, 
        top_k_retrieval=15, 
        top_k_final=top_k, 
        alpha=0.5
    )
    prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        frequency_penalty=0.5
    )
    return response.choices[0].message.content, retrieved

if __name__ == "__main__":
    while True:
        q = input("\nQuery (or 'quit'): ")
        if q.lower() == "quit":
            break
        result, sources = answer(q)
        print("\n--- Answer ---")
        print(result)
        print("\n--- Sources used ---")
        for s in sources:
            print(f"- {s.get('title', 'Untitled Source')} (ID: {s.get('paper_id')})")