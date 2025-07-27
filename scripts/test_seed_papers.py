#!/usr/bin/env python3
"""Test script to process seed papers through the pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.harvesters import ArxivHarvester
from src.lit_review.processing import PDFFetcher
from src.lit_review.extraction import LLMExtractor
from src.lit_review.utils import load_config


def main():
    """Process seed papers through the pipeline."""
    print("🧪 Testing pipeline with seed papers\n")
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Load seed papers
    with open("data/seed_papers.json", "r") as f:
        seed_data = json.load(f)
    
    papers = seed_data["seed_papers"]
    print(f"📚 Loaded {len(papers)} seed papers")
    
    # Convert to DataFrame
    df = pd.DataFrame(papers)
    
    # Add required columns
    df["source_db"] = "arxiv"
    df["pdf_path"] = ""
    df["pdf_status"] = ""
    df["screening_id"] = [f"SEED_{i:04d}" for i in range(len(df))]
    
    # Save raw papers
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    raw_path = output_dir / "seed_papers_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n✓ Saved raw papers to: {raw_path}")
    
    # Fetch PDFs for papers with arxiv_id
    print("\n📥 Fetching PDFs from arXiv...")
    pdf_fetcher = PDFFetcher(config)
    df_with_pdfs = pdf_fetcher.fetch_pdfs(df, parallel=True)
    
    # Show PDF status
    pdf_stats = df_with_pdfs["pdf_status"].value_counts()
    print("\nPDF fetch results:")
    for status, count in pdf_stats.items():
        print(f"  {status}: {count}")
    
    # Save with PDFs
    pdf_path = output_dir / "seed_papers_with_pdfs.csv" 
    df_with_pdfs.to_csv(pdf_path, index=False)
    print(f"\n✓ Saved papers with PDFs to: {pdf_path}")
    
    # Extract information with LLM
    print("\n🤖 Extracting information with LLM...")
    extractor = LLMExtractor(config)
    
    # Filter to papers with PDFs
    papers_with_pdfs = df_with_pdfs[
        df_with_pdfs["pdf_status"].str.startswith("downloaded")
    ].copy()
    
    if len(papers_with_pdfs) > 0:
        print(f"Processing {len(papers_with_pdfs)} papers with PDFs...")
        extraction_df = extractor.extract_all(papers_with_pdfs, parallel=False)
        
        # Save extraction results
        extraction_path = output_dir / "seed_papers_extracted.csv"
        extraction_df.to_csv(extraction_path, index=False)
        print(f"\n✓ Saved extraction results to: {extraction_path}")
        
        # Show extraction statistics
        if "extraction_status" in extraction_df.columns:
            extraction_stats = extraction_df["extraction_status"].value_counts()
            print("\nExtraction results:")
            for status, count in extraction_stats.items():
                print(f"  {status}: {count}")
        
        # Show some extracted fields
        if "venue_type" in extraction_df.columns:
            print("\nExtracted venue types:")
            print(extraction_df["venue_type"].value_counts())
        
        if "llm_family" in extraction_df.columns:
            print("\nExtracted LLM families:")
            print(extraction_df["llm_family"].value_counts())
    else:
        print("⚠️  No papers with PDFs found to extract from")
    
    print("\n✅ Pipeline test complete!")


if __name__ == "__main__":
    main()