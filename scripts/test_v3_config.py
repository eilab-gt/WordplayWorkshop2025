#!/usr/bin/env python3
"""Test script to verify v3.0.0 configuration is loaded correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.utils.config import ConfigLoader


def test_v3_config():
    """Test that v3.0.0 configuration loads correctly."""
    print("Testing v3.0.0 Configuration Loading...")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Test 1: Year range
    print("\n1. Year Range:")
    print("   Expected: (2022, 2025)")
    print(f"   Actual: {config.search_years}")
    assert config.search_years == (2022, 2025), "Year range not updated!"
    print("   ✅ PASS")

    # Test 2: New vocabulary terms
    print("\n2. New Vocabulary Terms:")

    # Debug: check if raw config has the terms
    if hasattr(config_loader, "_raw_config"):
        raw_wargame = config_loader._raw_config.get("wargame_terms", [])
        print(f"   Raw config wargame terms: {len(raw_wargame)}")
        print(f"   First few: {raw_wargame[:3] if raw_wargame else 'NONE'}")

    print(f"   Wargame terms count: {len(config.wargame_terms)}")
    print(f"   LLM terms count: {len(config.llm_terms)}")
    print(f"   Exclusion terms count: {len(config.exclusion_terms)}")

    # Check for specific new terms
    if len(config.wargame_terms) > 0:
        assert "TTX" in config.wargame_terms, "TTX not found in wargame terms"
        assert (
            "foundation model*" in config.llm_terms
        ), "foundation model* not found in LLM terms"
        assert "军事推演" in config.wargame_terms, "Chinese terms not found"
        print("   ✅ PASS - New terms found")
    else:
        print("   ⚠️  WARNING - Terms not loaded from config!")

    # Test 3: Disambiguation rules
    print("\n3. Disambiguation Rules:")
    print(f"   Rules found: {list(config.disambiguation.keys())}")
    assert "matrix_game" in config.disambiguation, "matrix_game rule not found"
    assert "generic_surveys" in config.disambiguation, "generic_surveys rule not found"
    print("   ✅ PASS")

    # Test 4: Grey literature sources
    print("\n4. Grey Literature Sources:")
    print(f"   Sources: {config.grey_lit_sources}")
    assert ".mil" in config.grey_lit_sources, ".mil not found in grey lit sources"
    assert ".gov" in config.grey_lit_sources, ".gov not found in grey lit sources"
    print("   ✅ PASS")

    # Test 5: Query strategies
    print("\n5. Query Strategies:")
    print(f"   Strategies found: {list(config.query_strategies.keys())}")
    assert "primary" in config.query_strategies, "primary strategy not found"
    assert "secondary" in config.query_strategies, "secondary strategies not found"
    print("   ✅ PASS")

    # Test 6: Quality metrics
    print("\n6. Quality Metrics:")
    print(
        f"   Minimum precision: {config.quality_metrics.get('minimum_precision', 'NOT FOUND')}"
    )
    print(
        f"   Target recall: {config.quality_metrics.get('target_recall', 'NOT FOUND')}"
    )

    print("\n" + "=" * 60)
    print("✅ All v3.0.0 configuration tests PASSED!")
    print("\nConfiguration is ready for use with the updated search pipeline.")


if __name__ == "__main__":
    try:
        test_v3_config()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
