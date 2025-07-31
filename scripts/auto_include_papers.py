#!/usr/bin/env python3
"""
Automatically mark papers for pipeline inclusion based on automated criteria.
This reduces human workload by pre-filtering clearly relevant papers.
"""

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def check_wargame_relevance(title, abstract):
    """Check if paper mentions wargaming concepts."""
    text = f"{title} {abstract}".lower()

    # Strong indicators
    strong_terms = [
        "war game",
        "wargame",
        "wargaming",
        "war-game",
        "military simulation",
        "defense simulation",
        "crisis simulation",
        "strategic simulation",
        "red team",
        "blue team",
        "tabletop exercise",
    ]

    # Weak indicators (need combination)
    weak_terms = [
        "strategic",
        "military",
        "defense",
        "crisis",
        "simulation",
        "game theory",
        "nash equilibrium",
    ]

    strong_matches = sum(1 for term in strong_terms if term in text)
    weak_matches = sum(1 for term in weak_terms if term in text)

    return strong_matches > 0 or weak_matches >= 3


def check_llm_relevance(title, abstract):
    """Check if paper mentions LLM/AI concepts."""
    text = f"{title} {abstract}".lower()

    llm_terms = [
        "large language model",
        "llm",
        "gpt",
        "chatgpt",
        "neural network",
        "deep learning",
        "transformer",
        "ai agent",
        "multi-agent",
        "reinforcement learning",
        "machine learning",
        "artificial intelligence",
    ]

    return sum(1 for term in llm_terms if term in text) >= 2


def calculate_relevance_score(row):
    """Calculate automated relevance score (0-10)."""
    score = 5  # Base score

    title = str(row.get("title", "")).lower()
    abstract = str(row.get("abstract", "")).lower()

    # Check both domains
    has_wargame = check_wargame_relevance(title, abstract)
    has_llm = check_llm_relevance(title, abstract)

    if has_wargame and has_llm:
        score = 9  # High relevance
    elif has_wargame or has_llm:
        score = 7  # Medium relevance
    else:
        score = 3  # Low relevance

    # Bonus for specific mentions
    if "war game" in title or "wargame" in title:
        score = min(10, score + 1)

    return score


def auto_include_papers(input_csv, output_csv, threshold=6):
    """Automatically mark papers for pipeline inclusion."""
    console.print("\n[bold]Auto-Including Papers for Pipeline[/bold]")
    console.print(f"Threshold: {threshold}/10\n")

    # Load papers
    df = pd.read_csv(input_csv)
    total_papers = len(df)

    # Calculate relevance scores
    df["auto_relevance_score"] = df.apply(calculate_relevance_score, axis=1)

    # Auto-include based on threshold
    df["auto_include"] = df["auto_relevance_score"] >= threshold
    df["auto_include_reason"] = df.apply(
        lambda row: (
            f"Automated: relevance score {row['auto_relevance_score']}/10"
            if row["auto_include"]
            else "Below threshold"
        ),
        axis=1,
    )

    # For backward compatibility, also set include_ta
    df.loc[df["auto_include"], "include_ta"] = 1
    df.loc[df["auto_include"], "reason_ta"] = df.loc[
        df["auto_include"], "auto_include_reason"
    ]
    df.loc[df["auto_include"], "screener_ta"] = "automated"
    df.loc[df["auto_include"], "screened_ta_date"] = pd.Timestamp.now().strftime(
        "%Y-%m-%d"
    )

    # Statistics
    included = df["auto_include"].sum()
    excluded = total_papers - included

    # Display results
    table = Table(title="Auto-Inclusion Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    table.add_row(
        "✅ Auto-Included", str(included), f"{included/total_papers*100:.1f}%"
    )
    table.add_row(
        "❌ Below Threshold", str(excluded), f"{excluded/total_papers*100:.1f}%"
    )
    table.add_row("Total Papers", str(total_papers), "100%", style="bold")

    console.print(table)

    # Show included papers
    if included > 0:
        console.print("\n[bold green]Auto-Included Papers:[/bold green]")
        for _, row in df[df["auto_include"]].iterrows():
            console.print(f"• [{row['auto_relevance_score']}] {row['title'][:80]}...")

    # Show excluded papers for review
    if excluded > 0:
        console.print(
            "\n[bold yellow]Papers Below Threshold (for human review):[/bold yellow]"
        )
        for _, row in df[~df["auto_include"]].head(5).iterrows():
            console.print(f"• [{row['auto_relevance_score']}] {row['title'][:80]}...")
        if excluded > 5:
            console.print(f"  ... and {excluded - 5} more")

    # Save results
    df.to_csv(output_csv, index=False)
    console.print(f"\n✅ Saved results to: {output_csv}")

    # Save excluded papers separately for human review
    excluded_csv = Path(output_csv).parent / "papers_excluded_for_review.csv"
    df_excluded = df[~df["auto_include"]].copy()

    # Add additional context for excluded papers
    df_excluded["wargame_relevance"] = df_excluded.apply(
        lambda row: (
            "Yes" if check_wargame_relevance(row["title"], row["abstract"]) else "No"
        ),
        axis=1,
    )
    df_excluded["llm_relevance"] = df_excluded.apply(
        lambda row: (
            "Yes" if check_llm_relevance(row["title"], row["abstract"]) else "No"
        ),
        axis=1,
    )

    # Save excluded papers with context
    df_excluded.to_csv(excluded_csv, index=False)
    console.print(f"📄 Saved excluded papers for review to: {excluded_csv}")

    # Create exclusion report
    report_path = Path(output_csv).parent / "auto_inclusion_report.txt"
    with open(report_path, "w") as f:
        f.write("AUTO-INCLUSION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total papers: {total_papers}\n")
        f.write(f"Auto-included: {included} ({included/total_papers*100:.1f}%)\n")
        f.write(f"Below threshold: {excluded} ({excluded/total_papers*100:.1f}%)\n")
        f.write(f"Threshold used: {threshold}/10\n\n")

        f.write("SCORING BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        score_dist = df["auto_relevance_score"].value_counts().sort_index()
        for score, count in score_dist.items():
            f.write(f"Score {score}: {count} papers\n")

        f.write("\nEXCLUDED PAPERS ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Papers with wargame terms: {len(df_excluded[df_excluded['wargame_relevance'] == 'Yes'])}\n"
        )
        f.write(
            f"Papers with LLM terms: {len(df_excluded[df_excluded['llm_relevance'] == 'Yes'])}\n"
        )
        f.write(
            f"Papers with both: {len(df_excluded[(df_excluded['wargame_relevance'] == 'Yes') & (df_excluded['llm_relevance'] == 'Yes')])}\n"
        )
        f.write(f"\nExcluded papers saved to: {excluded_csv.name}\n")

    console.print(f"📄 Report saved to: {report_path}")

    return df


@click.command()
@click.option("--input", "-i", required=True, help="Input CSV file")
@click.option("--output", "-o", help="Output CSV file (default: overwrite input)")
@click.option(
    "--threshold", "-t", default=6, type=int, help="Inclusion threshold (0-10)"
)
def main(input, output, threshold):
    """Auto-include papers based on relevance scoring."""
    if not output:
        output = input

    auto_include_papers(input, output, threshold)


if __name__ == "__main__":
    main()
