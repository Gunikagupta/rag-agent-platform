import json
import os
import psycopg2
from pathlib import Path
from sentence_transformers import SentenceTransformer

CONN_STR = "postgresql://gunika:rpEVIqirX3d0gYZOjk-cTQ@joyous-runner-32378.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=require"

def find_json_file(filename="fulltext_chunks.json"):
    matches = list(Path(".").rglob(filename))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Could not find {filename} anywhere in the project.")

def init_vector_db():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(CONN_STR)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_embeddings (
            chunk_id STRING PRIMARY KEY,
            paper_id STRING,
            title STRING,
            text STRING,
            embedding VECTOR(384)
        );
    """)
    conn.commit()

    json_path = find_json_file("fulltext_chunks.json")
    print(f"Loading data from: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Generating embeddings and inserting {len(chunks)} chunks into CockroachDB...")
    for i, chunk in enumerate(chunks, 1):
        # Clean null bytes (0x00) from text/title strings
        title = chunk.get("title", "").replace("\x00", "")
        text = chunk.get("text", "").replace("\x00", "")
        paper_id = chunk.get("paper_id", "").replace("\x00", "")

        text_to_embed = f"{title} {text}"
        embedding = model.encode(text_to_embed).tolist()

        cur.execute("""
            INSERT INTO paper_embeddings (chunk_id, paper_id, title, text, embedding)
            VALUES (%s, %s, %s, %s, %s::VECTOR)
            ON CONFLICT (chunk_id) DO NOTHING;
        """, (
            chunk["chunk_id"],
            paper_id,
            title,
            text,
            str(embedding)
        ))

        if i % 50 == 0 or i == len(chunks):
            conn.commit()
            print(f"Processed {i}/{len(chunks)} chunks")

    cur.close()
    conn.close()
    print("Vector database populated successfully!")

if __name__ == "__main__":
    init_vector_db()