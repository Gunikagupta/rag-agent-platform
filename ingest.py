import json
import os
import arxiv

def fetch_arxiv_papers(query="LLM evaluation", max_results=300):
    print(f"Fetching {max_results} papers for query: '{query}'...")
    
    # Configure client and search query
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in client.results(search):
        clean_summary = result.summary.replace("\n", " ")
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title.strip(),
            "summary": clean_summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d")
        })
    
    # Define relative output path
    output_path = os.path.join("data", "papers.json")
    
    # Ensure directory exists before writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
        
    print(f"Success! Saved {len(papers)} papers to {output_path}")

if __name__ == "__main__":
    fetch_arxiv_papers()