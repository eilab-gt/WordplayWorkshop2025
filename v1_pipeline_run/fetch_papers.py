#!/usr/bin/env python3
"""Fetch PDFs and alternative formats for harvested papers."""

import logging
import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.lit_review.processing import PDFFetcher
from src.lit_review.harvesters import ArxivHarvester
from src.lit_review.utils import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def fix_arxiv_id(arxiv_id):
    """Fix arxiv ID that was stored as float and lost its decimal."""
    if pd.isna(arxiv_id):
        return None
    
    # Convert to string and remove any .0 from float conversion
    arxiv_str = str(arxiv_id).replace('.0', '')
    
    # If it looks like a mangled arxiv ID (e.g., 231103220 instead of 2311.03220)
    # The pattern is usually YYMM.XXXXX where YY=year, MM=month
    if re.match(r'^\d{7,}$', arxiv_str) and len(arxiv_str) >= 9:
        # Most arxiv IDs have format YYMM.XXXXX
        # So for 2311032204, it should be 2311.03220
        return arxiv_str[:4] + '.' + arxiv_str[4:9]
    
    # If it already has a dot, return as is
    if '.' in arxiv_str:
        return arxiv_str
    
    return None

def main():
    """Fetch PDFs and alternative formats."""
    # Load configuration
    config = load_config('config.yaml')
    
    # Load harvested papers
    papers_df = pd.read_csv('v1_pipeline_run/harvested/papers_raw.csv')
    print(f"📚 Loaded {len(papers_df)} papers")
    
    # Create output directories
    pdf_dir = Path('v1_pipeline_run/pdfs')
    tex_dir = Path('v1_pipeline_run/tex')
    html_dir = Path('v1_pipeline_run/html')
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fetchers
    pdf_fetcher = PDFFetcher(config)
    arxiv_harvester = ArxivHarvester(config)
    
    # Track statistics
    stats = {
        'total': len(papers_df),
        'pdf_success': 0,
        'tex_success': 0,
        'html_success': 0,
        'failed': 0
    }
    
    # Process each paper
    for idx, row in papers_df.iterrows():
        # Fix arxiv ID
        arxiv_id = fix_arxiv_id(row.get('arxiv_id'))
        paper_id = arxiv_id if arxiv_id else f"paper_{idx}"
        title = row.get('title', 'Unknown')
        print(f"\n📄 Processing [{idx+1}/{len(papers_df)}]: {title[:60]}...")
        
        success = False
        
        # Try to fetch TeX source first (for arXiv papers)
        if arxiv_id and row['source_db'] == 'arxiv':
            print(f"  🔍 Checking for TeX source (arxiv:{arxiv_id})...")
            tex_content = arxiv_harvester.fetch_tex_source(arxiv_id)
            if tex_content:
                tex_file = tex_dir / f"{paper_id}.tex"
                tex_file.write_text(tex_content, encoding='utf-8')
                print(f"  ✅ TeX source saved: {tex_file.name}")
                stats['tex_success'] += 1
                success = True
            
            # Try HTML as well
            print(f"  🔍 Checking for HTML version...")
            html_content = arxiv_harvester.fetch_html_source(arxiv_id)
            if html_content:
                html_file = html_dir / f"{paper_id}.html"
                html_file.write_text(html_content, encoding='utf-8')
                print(f"  ✅ HTML version saved: {html_file.name}")
                stats['html_success'] += 1
                success = True
        
        # Try to fetch PDF using direct download
        pdf_url = row.get('pdf_url')
        if pdf_url and not pd.isna(pdf_url):
            pdf_url_str = str(pdf_url)
            print(f"  📥 Downloading PDF from: {pdf_url_str[:50]}...")
            try:
                import requests
                response = requests.get(pdf_url_str, timeout=30, headers={
                    'User-Agent': 'LiteratureReviewPipeline/1.0 (Research Tool)'
                })
                if response.status_code == 200:
                    pdf_file = pdf_dir / f"{paper_id}.pdf"
                    pdf_file.write_bytes(response.content)
                    print(f"  ✅ PDF saved: {pdf_file.name}")
                    stats['pdf_success'] += 1
                    success = True
                else:
                    print(f"  ⚠️ PDF download failed with status: {response.status_code}")
            except Exception as e:
                print(f"  ⚠️ PDF download failed: {e}")
        
        if not success:
            print(f"  ❌ Failed to fetch any format")
            stats['failed'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("📊 FETCH SUMMARY")
    print("="*60)
    print(f"Total papers: {stats['total']}")
    print(f"PDFs fetched: {stats['pdf_success']}")
    print(f"TeX sources fetched: {stats['tex_success']}")
    print(f"HTML versions fetched: {stats['html_success']}")
    print(f"Failed to fetch: {stats['failed']}")
    print(f"\nSuccess rate: {((stats['total'] - stats['failed']) / stats['total'] * 100):.1f}%")
    
    # Save fetch metadata
    metadata = {
        'paper_id': [],
        'has_pdf': [],
        'has_tex': [],
        'has_html': [],
        'arxiv_id': [],
        'title': []
    }
    
    for idx, row in papers_df.iterrows():
        arxiv_id = fix_arxiv_id(row.get('arxiv_id'))
        paper_id = arxiv_id if arxiv_id else f"paper_{idx}"
        metadata['paper_id'].append(paper_id)
        metadata['arxiv_id'].append(arxiv_id)
        metadata['title'].append(row.get('title', ''))
        metadata['has_pdf'].append((pdf_dir / f"{paper_id}.pdf").exists())
        metadata['has_tex'].append((tex_dir / f"{paper_id}.tex").exists())
        metadata['has_html'].append((html_dir / f"{paper_id}.html").exists())
    
    fetch_df = pd.DataFrame(metadata)
    fetch_df.to_csv('v1_pipeline_run/fetch_metadata.csv', index=False)
    print(f"\n💾 Fetch metadata saved to: v1_pipeline_run/fetch_metadata.csv")

if __name__ == "__main__":
    main()