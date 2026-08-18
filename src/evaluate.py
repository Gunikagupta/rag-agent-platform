import json
import time 
import os
from src.generate import answer

CACHE_PATH = "data/answer_cache.json"

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def check_retrieval(sources, expected_id):
    str_sources = [str(s) for s in sources]
    return str(expected_id).strip().lower() in [s.strip().lower() for s in str_sources]

def compute_chunk_precision(sources, expected_id):
    if not sources:
        return 0.0
    target = str(expected_id).strip().lower()
    matches = sum(1 for s in sources if str(s).strip().lower() == target)
    return matches / len(sources)

def check_refusal(phrases, result):
    if not result:
        return False
    result_lower = result.lower()
    return any(phrase.lower() in result_lower for phrase in phrases)

phrases = [
    "does not contain enough information",
    "cannot answer",
    "can't answer",
    "not enough information",
    "no information about",
    "there is no information",
    "cannot find any information",
    "do not have enough information",
    "cannot verify",
    "cannot provide an answer",
    "don't have enough information",
    "cannot provide information",
    "cannot answer that question",
    "not supported by provided context",
    "not supported by the provided context"
]

with open("data/eval_set.json") as f:
    eval_set = json.load(f)

cache = load_cache()
results = []

print(f"Starting evaluation on {len(eval_set)} questions...\n")

for idx, item in enumerate(eval_set):
    question = item["question"]
    
    if question in cache:
        generated = cache[question]["answer"]
        sources = cache[question]["sources_id"]
    else:
        success = False
        attempts = 0
        while not success and attempts < 3:
            attempts += 1
            try:
                print(f"[{idx + 1}/{len(eval_set)}] Processing: {question[:50]}...")
                generated, sources = answer(question)
                
                # --- FIX 1: Safe extraction handling both Dict and ScoredPoint objects ---
                sources_id = [
                    s.payload["paper_id"] if hasattr(s, "payload") else s.get("paper_id", s.get("id"))
                    for s in sources
                ]
                
                cache[question] = {"answer": generated, "sources_id": sources_id}
                sources = sources_id
                save_cache(cache)
                
                time.sleep(2)
                success = True
            except Exception as e:
                print(f"⚠️ Error on attempt {attempts}: {e}")
                print("Waiting 20 seconds before retrying...")
                time.sleep(20)

        if not success:
            print(f"❌ Failed question {idx + 1} after 3 attempts.")
            generated = "ERROR: Failed to generate answer due to API limits."
            sources = []

    row = {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "generated_answer": generated
    }

    if item["category"] == "factual":
        row["retrieval_hit"] = check_retrieval(sources, item["expected_id"])
        row["chunk_precision"] = compute_chunk_precision(sources, item["expected_id"])
    elif item["category"] in ("out_of_scope", "false_premise"):
        row["correctly_refused"] = check_refusal(phrases, generated)

    results.append(row)

# --- Summary Metrics ---

factual = [r for r in results if r["category"] == "factual"]
refusal_cats = [r for r in results if r["category"] in ("out_of_scope", "false_premise")]

print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)

if factual:
    recall = sum(r.get("retrieval_hit", False) for r in factual) / len(factual)
    precision = sum(r.get("chunk_precision", 0.0) for r in factual) / len(factual)
    print(f"Factual Recall@5 (Paper Hit Rate): {recall:.2%} ({len(factual)} questions)")
    print(f"Chunk Precision@5 (Signal-to-Noise): {precision:.2%}")

if refusal_cats:
    for r in refusal_cats:
        r["correctly_refused"] = check_refusal(phrases, r["generated_answer"])
        
    correct_count = sum(1 for r in refusal_cats if r["correctly_refused"])
    refusal_rate = correct_count / len(refusal_cats)
    
    print(f"Correct Refusal Rate (Guardrails):  {refusal_rate:.2%} ({correct_count}/{len(refusal_cats)} questions)")

with open("data/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFull results saved to data/eval_results.json")