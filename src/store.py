import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


with open("data/fulltext_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("data/embeddings.npy")

assert len(chunks) == embeddings.shape[0], f"Mismatch! Chunks: {len(chunks)}, Embeddings: {embeddings.shape[0]}"

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "fulltext_papers"  

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "chunk_id": chunks[i]["chunk_id"],
            "paper_id": chunks[i]["paper_id"],
            "text": chunks[i]["text"],
            "title": chunks[i]["title"],
            "chunk_index": chunks[i]["chunk_index"]
        }
    )
    for i in range(len(chunks))
]

BATCH_SIZE = 250
print(f"Upserting {len(points)} points into Qdrant in batches of {BATCH_SIZE}...")

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i : i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"Indexed batch {i // BATCH_SIZE + 1} / {(len(points) + BATCH_SIZE - 1) // BATCH_SIZE}")

print(f"\nSuccessfully indexed {len(points)} points into collection '{COLLECTION_NAME}'!")