from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("data/fulltext_chunks.json","r",encoding = "utf-8") as f:
    all_chunks = json.load(f)

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

print(embeddings.shape)  # should be (num_chunks, 384) for bge-small
np.save("data/embeddings.npy", embeddings)
print(f"Saved embeddings with shape {embeddings.shape} to data/embeddings.npy")