#!/usr/bin/env python3
"""Test script for enhanced pipeline features."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.lit_review.extraction.enhanced_llm_extractor import EnhancedLLMExtractor
from src.lit_review.harvesters.arxiv_harvester import ArxivHarvester
from src.lit_review.harvesters.base import BaseHarvester
from src.lit_review.utils import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_tex_html_extraction():
    """Test TeX and HTML extraction from arXiv."""
    print("\n=== Testing TeX/HTML Extraction ===")

    # Test papers with known arXiv IDs
    test_ids = [
        "2301.00810",  # Recent AI paper
        "2312.17617",  # Another recent paper
    ]

    config = type(
        "Config", (), {"rate_limits": {"arxiv": {"delay_milliseconds": 1000}}}
    )()
    harvester = ArxivHarvester(config)

    for arxiv_id in test_ids:
        print(f"\nTesting arXiv:{arxiv_id}")

        # Test TeX extraction
        tex_content = harvester.fetch_tex_source(arxiv_id)
        if tex_content:
            print(f"✓ TeX source fetched: {len(tex_content)} chars")
            print(f"  Preview: {tex_content[:200]}...")
        else:
            print("✗ TeX source not available")

        # Test HTML extraction
        html_content = harvester.fetch_html_source(arxiv_id)
        if html_content:
            print(f"✓ HTML version fetched: {len(html_content)} chars")
        else:
            print("✗ HTML version not available")

        time.sleep(1)  # Rate limiting


def test_keyword_filtering():
    """Test abstract keyword filtering."""
    print("\n=== Testing Keyword Filtering ===")

    # Create test papers
    from src.lit_review.harvesters.base import Paper

    test_papers = [
        Paper(
            title="LLM-powered Wargaming Simulation",
            authors=["Author A"],
            year=2024,
            abstract="This paper explores using GPT-4 for military wargaming simulations with human players.",
            source_db="test",
        ),
        Paper(
            title="Medical AI Diagnosis",
            authors=["Author B"],
            year=2024,
            abstract="Using machine learning for medical diagnosis in hospitals.",
            source_db="test",
        ),
        Paper(
            title="Autonomous Agent Warfare",
            authors=["Author C"],
            year=2024,
            abstract="Autonomous agents in warfare simulation using LLMs and reinforcement learning.",
            source_db="test",
        ),
    ]

    # Test different filter configurations
    config = type("Config", (), {})()
    harvester = BaseHarvester(config)

    # Test 1: Include keywords
    print("\nTest 1: Include 'wargaming' OR 'LLM'")
    filtered = harvester.filter_by_keywords(
        test_papers, include_keywords=["wargaming", "LLM"], min_matches=1
    )
    print(f"Result: {len(filtered)}/{len(test_papers)} papers passed")
    for p in filtered:
        print(f"  ✓ {p.title}")

    # Test 2: Exclude keywords
    print("\nTest 2: Exclude 'medical'")
    filtered = harvester.filter_by_keywords(test_papers, exclude_keywords=["medical"])
    print(f"Result: {len(filtered)}/{len(test_papers)} papers passed")
    for p in filtered:
        print(f"  ✓ {p.title}")

    # Test 3: Multiple criteria
    print("\nTest 3: Include 'LLM' AND 'simulation', exclude 'medical'")
    filtered = harvester.filter_by_keywords(
        test_papers,
        include_keywords=["LLM", "simulation"],
        exclude_keywords=["medical"],
        min_matches=2,
    )
    print(f"Result: {len(filtered)}/{len(test_papers)} papers passed")
    for p in filtered:
        print(f"  ✓ {p.title}")


def test_llm_service():
    """Test LLM service connectivity."""
    print("\n=== Testing LLM Service ===")

    import requests

    service_url = "http://localhost:8000"

    # Check health
    try:
        response = requests.get(f"{service_url}/health", timeout=2)
        if response.status_code == 200:
            print("✓ LLM service is healthy")
        else:
            print("✗ LLM service returned non-200 status")
            return False
    except Exception as e:
        print(f"✗ LLM service not reachable: {e}")
        print("\nTo start the service:")
        print("  python -m src.lit_review.llm_service")
        return False

    # Check available models
    try:
        response = requests.get(f"{service_url}/models", timeout=2)
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable models:")
            for model, info in models.items():
                status = "✓" if info["available"] else "✗"
                print(f"  {status} {model}: {info['description']}")

        # Test extraction endpoint
        print("\nTesting extraction endpoint...")
        test_text = "This is a test paper about LLM-powered wargaming simulations."

        response = requests.post(
            f"{service_url}/extract",
            json={"text": test_text, "model": "gemini/gemini-pro", "temperature": 0.1},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print("✓ Extraction test successful")
            else:
                print(f"✗ Extraction failed: {result.get('error')}")
        else:
            print(f"✗ Extraction returned status {response.status_code}")

    except Exception as e:
        print(f"✗ Error testing service: {e}")

    return True


def test_enhanced_extractor():
    """Test enhanced LLM extractor."""
    print("\n=== Testing Enhanced Extractor ===")

    # Load config
    config = load_config("config/test_config.yaml")

    # Create test DataFrame
    test_data = {
        "title": ["Test Paper 1", "Test Paper 2"],
        "authors": ["Author A", "Author B"],
        "year": [2024, 2024],
        "abstract": [
            "LLM-powered wargaming simulation research",
            "Another paper about AI in simulations",
        ],
        "arxiv_id": ["2301.00810", None],
        "pdf_path": [None, "test_output/cached_pdfs/test.pdf"],
        "pdf_status": ["", "cached"],
        "source_db": ["arxiv", "test"],
    }

    df = pd.DataFrame(test_data)

    # Initialize extractor
    extractor = EnhancedLLMExtractor(config)

    # Check service
    if not extractor.check_service_health():
        print("✗ LLM service not available")
        print("  Start with: python -m src.lit_review.llm_service")
        return

    print("✓ Enhanced extractor initialized")

    # Show available models
    models = extractor.get_available_models()
    print(f"\n{len(models)} models available")


def run_integration_test():
    """Run a complete integration test."""
    print("\n=== Integration Test: 10 Papers ===")

    config = load_config("config/test_config.yaml")

    # Step 1: Harvest with filtering
    print("\n1. Harvesting papers...")
    from src.lit_review.harvesters import SearchHarvester

    harvester = SearchHarvester(config)
    df = harvester.search_all(sources=["arxiv"], max_results_per_source=10)
    print(f"   Found {len(df)} papers")

    # Apply keyword filter
    if len(df) > 0:
        print("\n2. Applying keyword filters...")
        from src.lit_review.harvesters.base import Paper

        papers = []
        for _, row in df.iterrows():
            paper = Paper(
                title=row.get("title", ""),
                authors=(
                    row.get("authors", "").split("; ")
                    if isinstance(row.get("authors"), str)
                    else []
                ),
                year=row.get("year", 0),
                abstract=row.get("abstract", ""),
                source_db=row.get("source_db", ""),
                url=row.get("url"),
                doi=row.get("doi"),
                arxiv_id=row.get("arxiv_id"),
                venue=row.get("venue"),
                citations=row.get("citations"),
                pdf_url=row.get("pdf_url"),
                keywords=(
                    row.get("keywords", "").split("; ")
                    if isinstance(row.get("keywords"), str)
                    else []
                ),
            )
            papers.append(paper)

        # Filter for wargaming/simulation papers
        temp_harvester = BaseHarvester(config)
        filtered_papers = temp_harvester.filter_by_keywords(
            papers,
            include_keywords=["simulation", "game", "agent", "LLM", "GPT"],
            min_matches=1,
        )

        if filtered_papers:
            df_filtered = pd.DataFrame([p.to_dict() for p in filtered_papers])
            print(f"   Filtered to {len(df_filtered)} papers")

            # Save for inspection
            output_path = Path("test_output/integration_test_papers.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_filtered.to_csv(output_path, index=False)
            print(f"   Saved to {output_path}")
        else:
            print("   No papers matched filters")

    print("\n✅ Integration test complete!")


def main():
    """Run all tests."""
    print("=== Enhanced Pipeline Test Suite ===")

    # Test individual components
    test_tex_html_extraction()
    test_keyword_filtering()

    # Test LLM service (if running)
    if test_llm_service():
        test_enhanced_extractor()

    # Run integration test
    run_integration_test()

    print("\n=== All tests complete! ===")


if __name__ == "__main__":
    main()
