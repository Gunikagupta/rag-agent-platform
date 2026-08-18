import os
import json
from pathlib import Path

# Paths
FULLTEXT_DIR = Path("data/fulltext")
METADATA_FILE = Path("data/papers.json")  
OUTPUT_FILE = Path("data/fulltext_chunks.json")

# Chunking Configuration
CHUNK_SIZE = 400   # Target words per chunk
OVERLAP = 50       # Word overlap between consecutive chunks
STEP = CHUNK_SIZE - OVERLAP  # Advance 350 words each step


def load_title_map(metadata_path: Path) -> dict:
    """Loads paper metadata and builds a lookup dict: {paper_id: title}."""
    title_map = {}
    if not metadata_path.exists():
        print(f"⚠️ Warning: Metadata file {metadata_path} not found. Defaulting titles to 'Unknown Title'.")
        return title_map

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        for paper in data:
            p_id = paper.get("id") or paper.get("paper_id")
            title = paper.get("title", "Unknown Title")
            if p_id:
                title_map[str(p_id)] = title
    elif isinstance(data, dict):
        for p_id, info in data.items():
            if isinstance(info, dict):
                title_map[str(p_id)] = info.get("title", "Unknown Title")
            else:
                title_map[str(p_id)] = str(info)

    return title_map


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Splits a single document into overlapping chunks based on word count."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        
        if len(chunk_words) < 30 and chunks:
            break
            
        chunks.append(" ".join(chunk_words))

    return chunks


def main():
    print(f"Loading paper titles from {METADATA_FILE}...")
    title_map = load_title_map(METADATA_FILE)

    if not FULLTEXT_DIR.exists():
        raise FileNotFoundError(f"Directory {FULLTEXT_DIR} does not exist!")

    all_txt_files = list(FULLTEXT_DIR.glob("*.txt"))
    print(f"Found {len(all_txt_files)} full-text files in {FULLTEXT_DIR}.")

    all_chunks = []
    total_chunks_created = 0

    for txt_path in all_txt_files:
        paper_id = txt_path.stem

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            print(f"⚠️ Skipping empty file: {txt_path.name}")
            continue

        title = title_map.get(paper_id, "Unknown Title")
        chunks = chunk_text(content)

        for index, chunk_str in enumerate(chunks):
            chunk_obj = {
                "chunk_id": f"{paper_id}_chunk_{index}",
                "paper_id": paper_id,
                "title": title,
                "chunk_index": index,
                "text": chunk_str
            }
            all_chunks.append(chunk_obj)

        total_chunks_created += len(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("CHUNKING COMPLETE")
    print("="*50)
    print(f"Total Papers Processed : {len(all_txt_files)}")
    print(f"Total Chunks Created   : {total_chunks_created}")
    print(f"Average Chunks / Paper : {total_chunks_created / len(all_txt_files):.1f}")
    print(f"Output saved to        : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()