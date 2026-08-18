import json

with open("data/papers.json") as f:
    papers = json.load(f)

with open("data/eval_set.json") as f:
    eval_set = json.load(f)

# Build a set of all ids actually present in your corpus, for fast lookup
corpus_ids = {p["id"] for p in papers}

print(f"Corpus has {len(corpus_ids)} papers\n")

for item in eval_set:
    expected = item.get("expected_id") or item.get("expected_source_id")
    if expected is None:
        continue  # out_of_scope / false_premise questions won't have one
    if expected not in corpus_ids:
        print(f"[MISSING] q{item['id']}: expected_id '{expected}' NOT found in corpus")
    else:
        print(f"[OK]      q{item['id']}: expected_id '{expected}' found")