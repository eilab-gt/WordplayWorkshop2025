#!/usr/bin/env python3
"""
Demo script for the literature review pipeline.
Shows a quick end-to-end example with sample data.
"""

import time
from pathlib import Path

import click
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()


def create_demo_config():
    """Create a demo configuration file."""
    config = {
        "search": {
            "queries": {"demo": '"LLM" AND "wargaming" AND "simulation"'},
            "sources": {
                "google_scholar": {"enabled": False},  # Disabled for demo
                "arxiv": {"enabled": True, "max_results": 5},
                "semantic_scholar": {"enabled": False},
                "crossref": {"enabled": False},
            },
        },
        "api_keys": {"openai": "demo-key-not-real"},  # Demo mode
        "paths": {
            "data_dir": "./demo_data",
            "pdf_cache": "./demo_pdfs",
            "output_dir": "./demo_outputs",
            "log_db": "./demo_logs/logging.db",
        },
        "extraction": {"model": "gpt-3.5-turbo", "temperature": 0.3},
        "failure_vocabularies": {
            "escalation": ["escalation", "nuclear"],
            "bias": ["bias", "unfair"],
            "hallucination": ["hallucination", "confabulate"],
        },
        "viz": {
            "charts": {
                "timeline": {"enabled": True},
                "venue_dist": {"enabled": True},
                "failure_modes": {"enabled": True},
            }
        },
    }

    config_path = Path("demo_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def create_sample_papers():
    """Create sample paper data for demonstration."""
    papers = pd.DataFrame(
        {
            "title": [
                "Large Language Models in Strategic Wargaming: A Novel Approach",
                "GPT-4 as Autonomous Agents in Military Simulations",
                "Evaluating LLM Performance in Crisis Management Games",
                "Bias and Hallucination in AI-Powered Conflict Simulations",
                "Open-Ended Wargaming with Transformer Models",
            ],
            "authors": [
                "Smith, John; Doe, Jane",
                "Johnson, Alice; Brown, Bob",
                "Wilson, Carol; Davis, Dan",
                "Miller, Eve; Taylor, Frank",
                "Anderson, Grace; Lee, Henry",
            ],
            "year": [2024, 2023, 2024, 2023, 2024],
            "abstract": [
                "This paper explores the application of large language models (LLMs) in strategic wargaming scenarios. We demonstrate how GPT-4 can be used to generate realistic military strategies and assess potential escalation risks.",
                "We investigate the use of GPT-4 as autonomous agents in military simulation exercises. Our results show promising capabilities but also reveal concerning bias patterns in decision-making.",
                "A comprehensive evaluation of various LLMs in crisis management wargames. We find that while models excel at generating plausible scenarios, they often exhibit hallucination when faced with complex multi-party conflicts.",
                "This study examines failure modes in AI-powered conflict simulations, focusing on bias and hallucination. We identify key patterns where LLMs tend to escalate conflicts unnecessarily.",
                "We present a framework for open-ended wargaming using transformer models. Unlike traditional quantitative approaches, our method allows for natural language interaction and emergent strategies.",
            ],
            "source_db": [
                "arxiv",
                "arxiv",
                "google_scholar",
                "semantic_scholar",
                "crossref",
            ],
            "doi": [
                "10.1234/fake.2024.001",
                "",
                "10.1234/fake.2024.002",
                "10.1234/fake.2023.001",
                "",
            ],
            "url": [
                "https://arxiv.org/abs/2401.12345",
                "https://arxiv.org/abs/2301.23456",
                "https://example.com/paper3",
                "https://example.com/paper4",
                "https://example.com/paper5",
            ],
            "arxiv_id": ["2401.12345", "2301.23456", "", "", ""],
            "venue": [
                "International Conference on AI and Defense",
                "",
                "Journal of Military AI",
                "Workshop on AI Safety",
                "Strategic Studies Quarterly",
            ],
            "citations": [15, 8, 22, 12, 5],
            "pdf_url": [
                "https://arxiv.org/pdf/2401.12345.pdf",
                "https://arxiv.org/pdf/2301.23456.pdf",
                "",
                "",
                "",
            ],
        }
    )

    return papers


def create_sample_extraction_results():
    """Create sample extraction results for demonstration."""
    results = pd.DataFrame(
        {
            "screening_id": [
                "SCREEN_0001",
                "SCREEN_0002",
                "SCREEN_0003",
                "SCREEN_0004",
                "SCREEN_0005",
            ],
            "title": [
                "Large Language Models in Strategic Wargaming: A Novel Approach",
                "GPT-4 as Autonomous Agents in Military Simulations",
                "Evaluating LLM Performance in Crisis Management Games",
                "Bias and Hallucination in AI-Powered Conflict Simulations",
                "Open-Ended Wargaming with Transformer Models",
            ],
            "venue_type": [
                "conference",
                "tech-report",
                "journal",
                "workshop",
                "journal",
            ],
            "game_type": ["matrix", "digital", "seminar", "hybrid", "matrix"],
            "open_ended": ["yes", "yes", "no", "yes", "yes"],
            "quantitative": ["yes", "no", "yes", "yes", "no"],
            "llm_family": [
                "GPT-4",
                "GPT-4",
                "Multiple",
                "GPT-3.5",
                "Custom Transformer",
            ],
            "llm_role": ["player", "player", "analyst", "generator", "player"],
            "eval_metrics": [
                "Win rate; Human evaluation",
                "Decision quality; Realism score",
                "Accuracy; Response time",
                "Bias detection rate",
                "Creativity score; Plausibility",
            ],
            "failure_modes": [
                "escalation",
                "bias",
                "hallucination",
                "bias|hallucination|escalation",
                "",
            ],
            "awscale": [3, 4, 2, 3, 5],
            "code_release": [
                "github.com/fake/repo1",
                "none",
                "none",
                "github.com/fake/repo4",
                "none",
            ],
            "extraction_status": [
                "success",
                "success",
                "success",
                "success",
                "success",
            ],
        }
    )

    return results


@click.command()
@click.option("--full", is_flag=True, help="Run full demo including visualizations")
def demo(full):
    """Run a demonstration of the literature review pipeline."""
    console.print(
        Panel.fit(
            "[bold blue]Literature Review Pipeline Demo[/bold blue]\n\n"
            "This demo shows the pipeline working with sample data.\n"
            "No API calls are made - everything runs locally.",
            box="double",
        )
    )

    # Setup
    console.print("\n[bold]1. Setting up demo environment...[/bold]")
    create_demo_config()

    # Create directories
    for dir_name in [
        "demo_data/raw",
        "demo_data/processed",
        "demo_data/extracted",
        "demo_outputs",
        "demo_pdfs",
        "demo_logs",
    ]:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    console.print("   ✅ Created demo directories")
    console.print("   ✅ Created demo configuration")

    # Step 1: Paper Discovery
    console.print("\n[bold]2. Simulating paper discovery...[/bold]")
    papers = create_sample_papers()

    for _ in track(range(5), description="   Searching databases"):
        time.sleep(0.2)  # Simulate search time

    papers_path = Path("demo_data/raw/papers_demo.csv")
    papers.to_csv(papers_path, index=False)
    console.print(f"   ✅ Found {len(papers)} papers")

    # Step 2: Normalization
    console.print("\n[bold]3. Processing and normalizing papers...[/bold]")

    # Add screening IDs
    papers["screening_id"] = [f"SCREEN_{i:04d}" for i in range(1, len(papers) + 1)]
    normalized_path = Path("demo_data/processed/normalized_demo.csv")
    papers.to_csv(normalized_path, index=False)

    console.print("   ✅ Removed duplicates")
    console.print("   ✅ Generated screening IDs")

    # Step 3: Show screening
    console.print("\n[bold]4. Screening papers (simulated)...[/bold]")
    console.print("   📋 In real usage, you would:")
    console.print("      - Export to Excel for manual review")
    console.print("      - Mark papers for inclusion/exclusion")
    console.print("      - Add screening notes")
    console.print("   ✅ All papers marked for inclusion (demo)")

    # Step 4: Extraction
    console.print("\n[bold]5. Extracting information with LLM (simulated)...[/bold]")

    extraction_results = create_sample_extraction_results()

    for _ in track(range(5), description="   Processing papers"):
        time.sleep(0.3)  # Simulate extraction time

    extraction_path = Path("demo_data/extracted/extraction_demo.csv")
    extraction_results.to_csv(extraction_path, index=False)

    console.print("   ✅ Extracted venue types and game types")
    console.print("   ✅ Identified LLM families and roles")
    console.print("   ✅ Tagged failure modes")

    # Step 5: Analysis
    console.print("\n[bold]6. Analysis results:[/bold]")

    # Show statistics
    stats = {
        "Total papers": len(extraction_results),
        "Conference papers": len(
            extraction_results[extraction_results["venue_type"] == "conference"]
        ),
        "Journal papers": len(
            extraction_results[extraction_results["venue_type"] == "journal"]
        ),
        "Open-ended games": len(
            extraction_results[extraction_results["open_ended"] == "yes"]
        ),
        "Papers with code": len(
            extraction_results[extraction_results["code_release"] != "none"]
        ),
    }

    for key, value in stats.items():
        console.print(f"   • {key}: [bold]{value}[/bold]")

    # Show failure modes
    console.print("\n   [bold]Failure modes detected:[/bold]")
    failure_counts = {}
    for modes in extraction_results["failure_modes"]:
        if modes:
            for mode in modes.split("|"):
                failure_counts[mode] = failure_counts.get(mode, 0) + 1

    for mode, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
        console.print(f"   • {mode}: {count} papers")

    if full:
        # Step 6: Visualizations
        console.print("\n[bold]7. Generating visualizations...[/bold]")
        console.print("   📊 In real usage, this would create:")
        console.print("      - Publication timeline")
        console.print("      - Venue distribution charts")
        console.print("      - Failure mode analysis")
        console.print("      - LLM family breakdown")
        console.print("   ✅ Visualizations saved to demo_outputs/")

    # Step 7: Export
    console.print("\n[bold]8. Creating export package...[/bold]")
    console.print("   📦 Package would contain:")
    console.print("      - All data files (CSV format)")
    console.print("      - Visualizations (if generated)")
    console.print("      - README and metadata")
    console.print("   ✅ Ready for sharing or Zenodo upload")

    # Cleanup options
    console.print("\n[bold green]Demo complete![/bold green]")
    console.print("\nDemo files created in:")
    console.print("  • Data: demo_data/")
    console.print("  • Config: demo_config.yaml")

    console.print("\n[yellow]To run the real pipeline:[/yellow]")
    console.print("1. Add your OpenAI API key to config.yaml")
    console.print("2. Run: python run.py harvest")
    console.print("3. Follow the Quick Start Guide")

    # Offer cleanup
    if click.confirm("\nClean up demo files?", default=True):
        import shutil

        shutil.rmtree("demo_data", ignore_errors=True)
        shutil.rmtree("demo_outputs", ignore_errors=True)
        shutil.rmtree("demo_pdfs", ignore_errors=True)
        shutil.rmtree("demo_logs", ignore_errors=True)
        Path("demo_config.yaml").unlink(missing_ok=True)
        console.print("✅ Demo files cleaned up")


if __name__ == "__main__":
    demo()
