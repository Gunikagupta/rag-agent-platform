import json
import requests
import fitz
import time
import os

with open("data/papers.json") as f:
    papers = json.load(f)

failed_ids = ['2607.18237v1', '2607.18235v1', '2607.18232v1', '2607.18231v1', '2607.18226v1',
              '2607.18225v1', '2607.18223v1', '2607.18222v1', '2607.18213v1', '2607.18091v1',
              '2607.18080v1', '2607.17893v1', '2607.17890v1']
subset = [p for p in papers if p["id"] in failed_ids]
os.makedirs("data/fulltext", exist_ok=True)

failed = []

def fetch_one(paper, retries=2):
    for attempt in range(retries):
        try:
            response = requests.get(
                paper["pdf_url"],
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (research script)"}
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt < retries - 1:
                print(f"  retry {attempt+1} for {paper['id']}: {e}")
                time.sleep(3)
            else:
                raise

for i, p in enumerate(subset):
    out_path = f"data/fulltext/{p['id']}.txt"
    if os.path.exists(out_path):
        continue

    try:
        pdf_bytes = fetch_one(p)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        with open(out_path, "w") as f:
            f.write(text)

        print(f"[{i+1}/{len(subset)}] OK: {p['id']}")

    except Exception as e:
        print(f"[{i+1}/{len(subset)}] FAILED: {p['id']} - {e}")
        failed.append(p["id"])

    time.sleep(1.5)  # slower pace to avoid triggering rate limits

print(f"\nDone. {len(failed)} failures: {failed}")