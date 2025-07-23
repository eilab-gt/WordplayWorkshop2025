#!/usr/bin/env python3
"""
Generate statistics and reports from literature review data.
"""

import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def find_latest_file(directory: Path, pattern: str) -> Path:
    """Find the most recent file matching pattern in directory."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def generate_paper_stats(df: pd.DataFrame) -> dict:
    """Generate statistics from papers dataframe."""
    stats = {
        "total_papers": len(df),
        "date_range": {
            "min_year": (
                int(df["year"].min())
                if "year" in df and not df["year"].isna().all()
                else None
            ),
            "max_year": (
                int(df["year"].max())
                if "year" in df and not df["year"].isna().all()
                else None
            ),
        },
        "sources": {},
        "venues": {},
        "top_authors": {},
    }

    # Source distribution
    if "source_db" in df:
        stats["sources"] = df["source_db"].value_counts().to_dict()

    # Venue analysis
    if "venue" in df:
        venue_counts = df["venue"].value_counts().head(10)
        stats["venues"] = venue_counts.to_dict()

    # Author analysis
    if "authors" in df:
        all_authors = []
        for author_list in df["authors"].dropna():
            authors = [a.strip() for a in str(author_list).split(";")]
            all_authors.extend(authors)

        author_counts = pd.Series(all_authors).value_counts().head(10)
        stats["top_authors"] = author_counts.to_dict()

    return stats


def generate_extraction_stats(df: pd.DataFrame) -> dict:
    """Generate statistics from extraction results."""
    stats = {
        "total_extracted": len(df),
        "extraction_success_rate": 0,
        "venue_types": {},
        "game_types": {},
        "llm_distribution": {},
        "failure_modes": {},
        "awscale_distribution": {},
        "code_availability": {"available": 0, "not_available": 0},
    }

    # Success rate
    if "extraction_status" in df:
        success_count = len(df[df["extraction_status"] == "success"])
        stats["extraction_success_rate"] = (
            (success_count / len(df) * 100) if len(df) > 0 else 0
        )

    # Venue types
    if "venue_type" in df:
        stats["venue_types"] = df["venue_type"].value_counts().to_dict()

    # Game types
    if "game_type" in df:
        stats["game_types"] = df["game_type"].value_counts().to_dict()

    # LLM distribution
    if "llm_family" in df:
        stats["llm_distribution"] = df["llm_family"].value_counts().head(10).to_dict()

    # Failure modes
    if "failure_modes" in df:
        all_failures = []
        for modes in df["failure_modes"].dropna():
            if "|" in str(modes):
                all_failures.extend(modes.split("|"))
            elif modes and str(modes) != "nan":
                all_failures.append(str(modes))

        if all_failures:
            failure_counts = pd.Series(all_failures).value_counts()
            stats["failure_modes"] = failure_counts.to_dict()

    # AWScale distribution
    if "awscale" in df:
        awscale_counts = df["awscale"].value_counts().sort_index()
        stats["awscale_distribution"] = awscale_counts.to_dict()

    # Code availability
    if "code_release" in df:
        has_code = (
            df["code_release"].notna()
            & (df["code_release"] != "none")
            & (df["code_release"] != "")
        )
        stats["code_availability"]["available"] = int(has_code.sum())
        stats["code_availability"]["not_available"] = len(df) - int(has_code.sum())

    return stats


@click.command()
@click.option("--papers", type=click.Path(exists=True), help="Papers CSV file")
@click.option(
    "--extraction", type=click.Path(exists=True), help="Extraction results CSV"
)
@click.option("--output", type=click.Path(), help="Output JSON file for stats")
@click.option(
    "--format",
    type=click.Choice(["console", "json", "both"]),
    default="console",
    help="Output format",
)
def generate_stats(papers, extraction, output, format):
    """Generate comprehensive statistics from literature review data."""

    all_stats = {
        "generated_at": datetime.now().isoformat(),
        "papers": {},
        "extraction": {},
    }

    # Find files if not specified
    if not papers:
        papers = find_latest_file(Path("data/raw"), "*.csv")
        if papers:
            console.print(f"[blue]Using papers file: {papers}[/blue]")

    if not extraction:
        extraction = find_latest_file(Path("data/extracted"), "*.csv")
        if extraction:
            console.print(f"[blue]Using extraction file: {extraction}[/blue]")

    # Generate paper statistics
    if papers:
        df_papers = pd.read_csv(papers)
        paper_stats = generate_paper_stats(df_papers)
        all_stats["papers"] = paper_stats

        if format in ["console", "both"]:
            # Display paper statistics
            panel = Panel(
                f"Total papers: [bold]{paper_stats['total_papers']}[/bold]\n"
                f"Year range: [bold]{paper_stats['date_range']['min_year']} - {paper_stats['date_range']['max_year']}[/bold]",
                title="Paper Collection Statistics",
                box=box.ROUNDED,
            )
            console.print(panel)

            # Source distribution table
            if paper_stats["sources"]:
                table = Table(title="Papers by Source")
                table.add_column("Source", style="cyan")
                table.add_column("Count", style="yellow")
                table.add_column("Percentage", style="green")

                total = sum(paper_stats["sources"].values())
                for source, count in sorted(
                    paper_stats["sources"].items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / total * 100) if total > 0 else 0
                    table.add_row(source, str(count), f"{percentage:.1f}%")

                console.print(table)

            # Top authors
            if paper_stats["top_authors"]:
                console.print("\n[bold]Top 10 Authors:[/bold]")
                for i, (author, count) in enumerate(
                    list(paper_stats["top_authors"].items())[:10], 1
                ):
                    console.print(f"  {i}. {author}: {count} papers")

    # Generate extraction statistics
    if extraction:
        df_extraction = pd.read_csv(extraction)
        extraction_stats = generate_extraction_stats(df_extraction)
        all_stats["extraction"] = extraction_stats

        if format in ["console", "both"]:
            console.print("\n")

            # Extraction overview
            panel = Panel(
                f"Total extracted: [bold]{extraction_stats['total_extracted']}[/bold]\n"
                f"Success rate: [bold]{extraction_stats['extraction_success_rate']:.1f}%[/bold]\n"
                f"Papers with code: [bold]{extraction_stats['code_availability']['available']}[/bold]",
                title="Extraction Statistics",
                box=box.ROUNDED,
            )
            console.print(panel)

            # Game types and venue types
            if extraction_stats["game_types"] or extraction_stats["venue_types"]:
                table = Table(title="Classification Results")
                table.add_column("Category", style="cyan")
                table.add_column("Type", style="yellow")
                table.add_column("Count", style="green")

                if extraction_stats["venue_types"]:
                    for venue, count in extraction_stats["venue_types"].items():
                        table.add_row("Venue", str(venue), str(count))

                if extraction_stats["game_types"]:
                    for game, count in extraction_stats["game_types"].items():
                        table.add_row("Game Type", str(game), str(count))

                console.print(table)

            # Failure modes
            if extraction_stats["failure_modes"]:
                console.print("\n[bold]Failure Modes Detected:[/bold]")
                for mode, count in sorted(
                    extraction_stats["failure_modes"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    console.print(f"  • {mode}: {count} occurrences")

            # AWScale distribution
            if extraction_stats["awscale_distribution"]:
                console.print("\n[bold]AWScale Distribution:[/bold]")
                for scale, count in sorted(
                    extraction_stats["awscale_distribution"].items()
                ):
                    bar = "█" * int(count * 5)  # Simple bar chart
                    console.print(f"  {scale}: {bar} ({count})")

    # Save to JSON if requested
    if format in ["json", "both"] and output:
        with open(output, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)
        console.print(f"\n[green]Statistics saved to {output}[/green]")

    # Summary
    if format in ["console", "both"]:
        console.print("\n[bold]Summary:[/bold]")
        if all_stats["papers"]:
            console.print(
                f"  • Collected {all_stats['papers']['total_papers']} papers from {len(all_stats['papers']['sources'])} sources"
            )
        if all_stats["extraction"]:
            console.print(
                f"  • Successfully extracted {all_stats['extraction']['extraction_success_rate']:.0f}% of papers"
            )
            if all_stats["extraction"]["failure_modes"]:
                console.print(
                    f"  • Identified {len(all_stats['extraction']['failure_modes'])} types of failure modes"
                )


if __name__ == "__main__":
    generate_stats()
