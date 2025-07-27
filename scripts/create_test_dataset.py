#!/usr/bin/env python3
"""Create test dataset from seed papers for pipeline testing."""

import json

import pandas as pd

# Load seed papers
with open("data/seed_papers.json") as f:
    data = json.load(f)

# Convert to DataFrame
papers = []
for p in data["seed_papers"]:
    papers.append(
        {
            "title": p["title"],
            "authors": "; ".join(p["authors"]),
            "year": p["year"],
            "abstract": p["abstract"],
            "doi": p.get("doi", ""),
            "url": p["url"],
            "source": "seed_data",
            "arxiv_id": p.get("arxiv_id", ""),
            "venue": p.get("venue", ""),
        }
    )

df = pd.DataFrame(papers)
df.to_csv("data/raw/test_papers.csv", index=False)
print(f"Created test dataset with {len(df)} papers")
print("\nPapers:")
for _, row in df.iterrows():
    print(f"- {row.title[:60]}...")
