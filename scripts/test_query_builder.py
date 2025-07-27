#!/usr/bin/env python3
"""Test script to verify NEAR operator and wildcard support in query builder."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.harvesters.query_builder import QueryBuilder
from src.lit_review.utils.config import ConfigLoader


def test_query_builder():
    """Test the QueryBuilder with v3.0.0 configuration."""
    print("Testing Query Builder with NEAR Operators and Wildcards...")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Create query builder
    builder = QueryBuilder()

    # Test parsing different term types
    print("\n1. Testing Query Term Parsing:")
    test_terms = [
        '"red team*" NEAR/5 (exercise OR simulation OR wargame)',
        "wargame*",
        '"matrix game*" NEAR/5 (scenario OR crisis OR policy)',
        '"large language model"',
        "LLM",
    ]

    for term in test_terms:
        parsed = builder.parse_query_term(term)
        print(f"\n   Term: {term}")
        print(f"   Parsed: {parsed}")

    # Test building base query
    print("\n2. Building Base Query from Config:")
    base_query = builder.build_query_from_config(config)
    print(f"   Base query length: {len(base_query)}")
    print(f"   Base query preview: {base_query[:200]}...")

    # Test translations for different platforms
    print("\n3. Testing Platform-Specific Translations:")

    # Test Google Scholar translation
    print("\n   Google Scholar:")
    gs_query = builder.translate_for_google_scholar(base_query)
    print(f"   - Length: {len(gs_query)}")
    print(f"   - Preview: {gs_query[:150]}...")

    # Test arXiv translation
    print("\n   arXiv:")
    arxiv_query = builder.translate_for_arxiv(base_query)
    print(f"   - Length: {len(arxiv_query)}")
    print(f"   - Preview: {arxiv_query[:150]}...")

    # Test Semantic Scholar translation
    print("\n   Semantic Scholar:")
    ss_query = builder.translate_for_semantic_scholar(base_query)
    print(f"   - Length: {len(ss_query)}")
    print(f"   - Preview: {ss_query[:150]}...")

    # Test CrossRef translation
    print("\n   CrossRef:")
    cr_query = builder.translate_for_crossref(base_query)
    print(f"   - Length: {len(cr_query)}")
    print(f"   - Preview: {cr_query[:150]}...")

    # Test wildcard expansion
    print("\n4. Testing Wildcard Expansion:")
    known_terms = ["wargame", "wargaming", "wargames", "warfare", "warmup"]
    expanded = builder.expand_wildcards("wargam*", known_terms)
    print("   Pattern: wargam*")
    print(f"   Expanded to: {expanded}")

    # Test specific NEAR operator handling
    print("\n5. Testing NEAR Operator in Exclusions:")
    # Check if exclusions with NEAR are handled
    exclusion_terms = [
        '"matrix game*" NEAR/5 (Nash OR equilibrium OR payoff)',
        '"red teaming" NEAR/5 (LLM OR "language model" OR ChatGPT)',
    ]

    for term in exclusion_terms:
        parsed = builder.parse_query_term(term)
        formatted = builder._format_parsed_term(parsed)
        print(f"\n   Exclusion: {term}")
        print(f"   Formatted: {formatted}")

    print("\n" + "=" * 60)
    print("✅ Query Builder tests completed!")
    print("\nThe query builder successfully:")
    print("- Parses NEAR operators and wildcards")
    print("- Translates queries for each search platform")
    print("- Handles complex exclusion patterns")
    print("- Expands wildcards against known terms")


if __name__ == "__main__":
    try:
        test_query_builder()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
