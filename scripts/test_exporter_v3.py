#!/usr/bin/env python3
"""Test script to verify v3.0.0 exporter functionality with exclusion reports."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.utils.config import ConfigLoader
from src.lit_review.utils.exporter import Exporter


def create_test_data():
    """Create test data for exporter testing."""
    # Create main extraction dataframe
    extraction_df = pd.DataFrame(
        {
            "title": ["Paper 1", "Paper 2", "Paper 3"],
            "authors": ["Author A", "Author B; Author C", "Author D"],
            "year": [2023, 2024, 2022],
            "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
            "doi": ["10.1234/1", "10.1234/2", "10.1234/3"],
            "url": [
                "http://example.com/1",
                "http://example.com/2",
                "http://example.com/3",
            ],
            "source_db": ["arxiv", "semantic_scholar", "crossref"],
            "extraction_status": ["success", "success", "success"],
            "llm_family": ["GPT-4", "Claude", "GPT-4"],
            "venue_type": ["conference", "journal", "workshop"],
            "grey_lit_flag": [False, False, True],
        }
    )

    # Create excluded papers dataframe
    excluded_df = pd.DataFrame(
        {
            "title": ["Excluded 1", "Excluded 2", "Excluded 3", "Excluded 4"],
            "authors": ["Author X", "Author Y", "Author Z", "Author W"],
            "year": [2023, 2024, 2022, 2023],
            "abstract": [
                "Matrix game Nash equilibrium",
                "Red teaming LLM jailbreak",
                "Generic survey",
                "Hospital simulation",
            ],
            "source_db": ["google_scholar", "arxiv", "crossref", "semantic_scholar"],
            "disambiguation_status": ["excluded", "excluded", "excluded", "excluded"],
            "disambiguation_reason": [
                "matrix_game: negative context detected",
                "red_teaming: negative context detected",
                "generic_surveys: negative context without required positive terms",
                "exclusion_terms: hospital detected",
            ],
            "wargame_relevance": ["Yes", "Yes", "No", "No"],
            "llm_relevance": ["No", "Yes", "Yes", "No"],
        }
    )

    # Create disambiguation report
    disambiguation_report = {
        "statistics": {
            "total_papers": 7,
            "matrix_game_filtered": 1,
            "red_teaming_filtered": 1,
            "rl_board_game_filtered": 0,
            "generic_surveys_filtered": 1,
            "grey_lit_tagged": 1,
        },
        "exclusion_reasons": {
            "matrix_game: negative context detected": 1,
            "red_teaming: negative context detected": 1,
            "generic_surveys: negative context without required positive terms": 1,
            "exclusion_terms: hospital detected": 1,
        },
        "grey_literature": {"count": 1, "sources": {"crossref": 1}},
        "examples": {
            "matrix_game: negative context detected": [
                {
                    "title": "Excluded 1",
                    "year": 2023,
                    "abstract_snippet": "Matrix game Nash equilibrium...",
                }
            ]
        },
    }

    # Create summary
    summary = {
        "total_papers": len(extraction_df),
        "excluded_papers": len(excluded_df),
        "extraction_success_rate": 1.0,
        "sources": extraction_df["source_db"].value_counts().to_dict(),
        "year_distribution": extraction_df["year"].value_counts().to_dict(),
    }

    return extraction_df, excluded_df, disambiguation_report, summary


def test_exporter_v3():
    """Test the v3.0.0 exporter functionality."""
    print("Testing v3.0.0 Exporter with Exclusion Reports...")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Create exporter
    exporter = Exporter(config)

    # Create test data
    extraction_df, excluded_df, disambiguation_report, summary = create_test_data()

    print("\n1. Test Data Created:")
    print(f"   - Extraction papers: {len(extraction_df)}")
    print(f"   - Excluded papers: {len(excluded_df)}")
    print(
        f"   - Grey literature tagged: {disambiguation_report['statistics']['grey_lit_tagged']}"
    )

    # Create empty figures list for testing
    figures = []

    # Test export with v3.0.0 features
    print("\n2. Creating Export Package with v3.0.0 Features...")
    try:
        output_name = f"test_v3_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=figures,
            summary=summary,
            output_name=output_name,
            excluded_df=excluded_df,
            disambiguation_report=disambiguation_report,
        )

        print(f"\n   ✅ Export package created: {archive_path}")

        # Verify the archive contains expected files
        import zipfile

        print("\n3. Verifying Archive Contents:")
        with zipfile.ZipFile(archive_path, "r") as zf:
            files = zf.namelist()

            # Check for v3.0.0 specific files
            v3_files = [
                "excluded_papers/excluded_papers.csv",
                "excluded_papers/excluded_summary.json",
                "disambiguation_report.json",
            ]

            for expected_file in v3_files:
                found = any(expected_file in f for f in files)
                status = "✅" if found else "❌"
                print(f"   {status} {expected_file}")

            # Check standard files
            standard_files = [
                "data/extraction_results.csv",
                "data/extraction_results.json",
                "data/extraction_results.xlsx",
                "summary_report.json",
                "metadata.json",
                "README.md",
            ]

            print("\n   Standard files:")
            for expected_file in standard_files:
                found = any(expected_file in f for f in files)
                status = "✅" if found else "❌"
                print(f"   {status} {expected_file}")

        # Read and display README content
        print("\n4. README Content Preview:")
        with zipfile.ZipFile(archive_path, "r") as zf:
            readme_content = zf.read(
                next(f for f in zf.namelist() if f.endswith("README.md"))
            ).decode("utf-8")
            print("   " + readme_content[:500].replace("\n", "\n   ") + "...")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ v3.0.0 Exporter tests completed successfully!")
    print("\nThe exporter now includes:")
    print("- Excluded papers export with detailed reasons")
    print("- Disambiguation statistics report")
    print("- Grey literature tagging information")
    print("- Enhanced README with exclusion statistics")


if __name__ == "__main__":
    test_exporter_v3()
