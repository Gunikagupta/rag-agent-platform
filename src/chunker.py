import json

with open("data/papers.json") as f:
    papers = json.load(f)

chunks = []
for p in papers:
    chunks.append({
        "chunk_id": p["id"],
        "text": p["summary"],
        "title": p["title"],
        "source_url": p["pdf_url"]
    })
with open("data/chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)
print(f"{len(chunks)} chunks ready")