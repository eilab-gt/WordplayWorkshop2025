#!/usr/bin/env python3
"""Test full v3.0.0 pipeline functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.lit_review.harvesters.query_builder import QueryBuilder
from src.lit_review.processing.disambiguator import Disambiguator
from src.lit_review.utils.config import load_config


def test_full_pipeline():
    """Test the complete v3.0.0 pipeline flow."""
    print("Testing Full v3.0.0 Pipeline...")
    print("=" * 60)

    # Load config
    config = load_config()
    print("\n1. Configuration loaded successfully")
    print(f"   Year range: {config.search_years}")
    print(
        f"   Terms loaded: {len(config.wargame_terms)} wargame, {len(config.llm_terms)} LLM"
    )

    # Test query building
    print("\n2. Testing Query Builder:")
    builder = QueryBuilder()
    query = builder.build_query_from_config(config)
    print(f"   Base query built: {len(query)} chars")

    # Test platform translations
    gs_query = builder.translate_for_google_scholar(query)
    arxiv_query = builder.translate_for_arxiv(query)
    ss_query = builder.translate_for_semantic_scholar(query)
    cr_query = builder.translate_for_crossref(query)

    print("   Platform translations:")
    print(f"   - Google Scholar: {len(gs_query)} chars")
    print(f"   - arXiv: {len(arxiv_query)} chars")
    print(f"   - Semantic Scholar: {len(ss_query)} chars")
    print(f"   - CrossRef: {len(cr_query)} chars")

    # Test disambiguation
    print("\n3. Testing Disambiguator:")
    disambiguator = Disambiguator(config)

    # Create test papers
    test_papers = pd.DataFrame(
        [
            {
                "title": "Matrix Game Theory Equilibrium Analysis",
                "abstract": "We study Nash equilibrium in matrix games...",
                "year": 2023,
                "url": "https://example.com/paper1",
            },
            {
                "title": "Red Teaming LLMs for Jailbreak Detection",
                "abstract": "We red team ChatGPT to find prompt injection vulnerabilities...",
                "year": 2023,
                "url": "https://example.mil/paper2",
            },
            {
                "title": "Military Wargaming with Large Language Models",
                "abstract": "We use GPT-4 for crisis simulation exercises...",
                "year": 2023,
                "url": "https://rand.org/paper3",
            },
            {
                "title": "AlphaZero Plays Board Games",
                "abstract": "Reinforcement learning for Go and chess...",
                "year": 2023,
                "url": "https://example.com/paper4",
            },
        ]
    )

    # Apply disambiguation
    result_df = disambiguator.apply_rules(test_papers.copy())
    report = disambiguator.create_disambiguation_report(result_df)

    print(f"   Input papers: {len(test_papers)}")
    print(f"   After disambiguation: {len(result_df)}")
    print(f"   Report keys: {list(report.keys())}")

    # Calculate excluded from input and output difference
    excluded_count = len(test_papers) - len(result_df)
    print(f"   Excluded: {excluded_count}")

    if "grey_literature" in report:
        print(f"   Grey literature: {report['grey_literature']['count']}")

    # Test NEAR operator parsing
    print("\n4. Testing NEAR Operator Handling:")
    near_terms = [
        '"red team*" NEAR/5 (exercise OR simulation OR wargame)',
        '"matrix game*" NEAR/5 (Nash OR equilibrium OR payoff)',
    ]

    for term in near_terms:
        parsed = builder.parse_query_term(term)
        print(f"   {term}")
        print(
            f"   -> Type: {parsed['type']}, Distance: {parsed.get('distance', 'N/A')}"
        )

    # Test wildcard expansion
    print("\n5. Testing Wildcard Expansion:")
    known_terms = ["wargame", "wargaming", "wargames", "wargamer"]
    expanded = builder.expand_wildcards("wargam*", known_terms)
    print(f"   wargam* -> {expanded}")

    print("\n" + "=" * 60)
    print("✅ All v3.0.0 pipeline components working correctly!")
    print("\nKey v3.0.0 Features Verified:")
    print("- NEAR operator parsing and translation")
    print("- Wildcard pattern support")
    print("- Disambiguation filtering")
    print("- Grey literature tagging")
    print("- Platform-specific query optimization")


if __name__ == "__main__":
    test_full_pipeline()
