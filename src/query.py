from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
client = QdrantClient(url="http://localhost:6333")

def search(query_text, top_k=5):
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name="fulltext_papers",
        query=query_vector,
        limit=top_k
    )
    return results.points

if __name__ == "__main__":
    query = "Speech conversion system based on a three stage based architecture"
    results = search(query)
    for point in results:
        print(f"{point.score:.4f}  {point.payload['chunk_id']}")