#!/usr/bin/env python3
"""
Database inspection utility for the literature review pipeline.
Provides tools to examine papers, logs, and extraction results.
"""

import sqlite3
from pathlib import Path

import click
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Database inspection utility."""
    pass


@cli.command()
@click.option("--csv", type=click.Path(exists=True), help="CSV file to inspect")
@click.option("--limit", default=10, help="Number of rows to display")
@click.option("--columns", help="Comma-separated list of columns to show")
def papers(csv, limit, columns):
    """Inspect papers data."""
    if not csv:
        # Try to find the most recent papers file
        data_dir = Path("data/raw")
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            console.print("[red]No papers CSV files found in data/raw/[/red]")
            return
        csv = str(max(csv_files, key=lambda x: x.stat().st_mtime))
        console.print(f"[blue]Using most recent file: {csv}[/blue]")

    df = pd.read_csv(csv)

    # Show summary
    panel = Panel(
        f"Total papers: [bold]{len(df)}[/bold]\n"
        f"Date range: [bold]{df['year'].min() if 'year' in df else 'N/A'} - {df['year'].max() if 'year' in df else 'N/A'}[/bold]\n"
        f"Sources: [bold]{', '.join(df['source_db'].unique()) if 'source_db' in df else 'N/A'}[/bold]",
        title="Papers Summary",
        box=box.ROUNDED,
    )
    console.print(panel)

    # Show sample data
    table = Table(title=f"Sample Papers (showing {min(limit, len(df))} of {len(df)})")

    if columns:
        cols = [c.strip() for c in columns.split(",")]
    else:
        cols = ["title", "authors", "year", "source_db"]

    # Add columns
    for col in cols:
        if col in df.columns:
            table.add_column(col, style="cyan", no_wrap=col != "title")

    # Add rows
    for _idx, row in df.head(limit).iterrows():
        values = []
        for col in cols:
            if col in df.columns:
                val = str(row[col])
                if col == "title" and len(val) > 60:
                    val = val[:57] + "..."
                values.append(val)
        table.add_row(*values)

    console.print(table)


@cli.command()
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="logs/logging.db",
    help="SQLite log database",
)
@click.option("--level", help="Filter by log level (INFO, WARNING, ERROR)")
@click.option("--limit", default=20, help="Number of logs to display")
@click.option("--errors-only", is_flag=True, help="Show only errors")
def logs(db, level, limit, errors_only):
    """Inspect log database."""
    if not Path(db).exists():
        console.print(f"[red]Log database not found: {db}[/red]")
        return

    conn = sqlite3.connect(db)

    # Build query
    query = "SELECT timestamp, level, logger, message FROM logs"
    conditions = []

    if errors_only:
        conditions.append("level = 'ERROR'")
    elif level:
        conditions.append(f"level = '{level.upper()}'")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += f" ORDER BY timestamp DESC LIMIT {limit}"

    df = pd.read_sql_query(query, conn)

    # Get summary stats
    stats_query = """
    SELECT
        level,
        COUNT(*) as count
    FROM logs
    GROUP BY level
    """
    stats_df = pd.read_sql_query(stats_query, conn)

    # Show summary
    stats_text = "\n".join(
        [
            f"{row['level']}: [bold]{row['count']}[/bold]"
            for _, row in stats_df.iterrows()
        ]
    )
    panel = Panel(
        f"Total logs: [bold]{stats_df['count'].sum()}[/bold]\n\n{stats_text}",
        title="Log Summary",
        box=box.ROUNDED,
    )
    console.print(panel)

    # Show logs
    table = Table(title=f"Recent Logs (showing {len(df)} entries)")
    table.add_column("Time", style="blue", no_wrap=True)
    table.add_column("Level", style="yellow")
    table.add_column("Logger", style="green")
    table.add_column("Message", style="cyan")

    for _, row in df.iterrows():
        level_style = {
            "ERROR": "red bold",
            "WARNING": "yellow",
            "INFO": "green",
            "DEBUG": "blue",
        }.get(row["level"], "white")

        table.add_row(
            row["timestamp"][:19],  # Remove microseconds
            f"[{level_style}]{row['level']}[/{level_style}]",
            row["logger"].split(".")[-1],  # Short logger name
            row["message"][:80] + "..." if len(row["message"]) > 80 else row["message"],
        )

    console.print(table)
    conn.close()


@cli.command()
@click.option("--csv", type=click.Path(exists=True), help="Extraction results CSV")
@click.option("--stats", is_flag=True, help="Show statistics only")
def extraction(csv, stats):
    """Inspect extraction results."""
    if not csv:
        # Try to find extraction results
        data_dir = Path("data/extracted")
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            console.print("[red]No extraction CSV files found in data/extracted/[/red]")
            return
        csv = str(max(csv_files, key=lambda x: x.stat().st_mtime))
        console.print(f"[blue]Using most recent file: {csv}[/blue]")

    df = pd.read_csv(csv)

    # Calculate statistics
    total = len(df)
    successful = (
        len(df[df["extraction_status"] == "success"])
        if "extraction_status" in df
        else 0
    )

    stats_data = {
        "Total papers": total,
        "Successfully extracted": successful,
        "Extraction rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
    }

    if "venue_type" in df:
        stats_data["Venue types"] = dict(df["venue_type"].value_counts())

    if "game_type" in df:
        stats_data["Game types"] = dict(df["game_type"].value_counts())

    if "llm_family" in df:
        stats_data["LLM families"] = dict(df["llm_family"].value_counts().head(5))

    if "failure_modes" in df:
        # Count failure modes
        all_failures = []
        for modes in df["failure_modes"].dropna():
            if "|" in str(modes):
                all_failures.extend(modes.split("|"))
            elif modes and str(modes) != "nan":
                all_failures.append(str(modes))

        if all_failures:
            failure_counts = pd.Series(all_failures).value_counts().head(5)
            stats_data["Top failure modes"] = dict(failure_counts)

    # Show statistics
    panel_content = []
    for key, value in stats_data.items():
        if isinstance(value, dict):
            sub_items = "\n".join([f"  • {k}: {v}" for k, v in value.items()])
            panel_content.append(f"[bold]{key}:[/bold]\n{sub_items}")
        else:
            panel_content.append(f"[bold]{key}:[/bold] {value}")

    panel = Panel(
        "\n\n".join(panel_content), title="Extraction Statistics", box=box.ROUNDED
    )
    console.print(panel)

    if not stats:
        # Show sample extractions
        table = Table(title="Sample Extractions")
        table.add_column("ID", style="blue")
        table.add_column("Venue", style="green")
        table.add_column("Game Type", style="yellow")
        table.add_column("LLM", style="cyan")
        table.add_column("AWScale", style="magenta")
        table.add_column("Status", style="red")

        for _, row in df.head(10).iterrows():
            table.add_row(
                str(row.get("screening_id", "N/A")),
                str(row.get("venue_type", "N/A")),
                str(row.get("game_type", "N/A")),
                str(row.get("llm_family", "N/A")),
                str(row.get("awscale", "N/A")),
                str(row.get("extraction_status", "N/A")),
            )

        console.print(table)


@cli.command()
@click.argument("screening_id")
@click.option(
    "--csv", type=click.Path(exists=True), help="CSV file containing the paper"
)
def paper(screening_id, csv):
    """Show detailed information about a specific paper."""
    if not csv:
        # Search in multiple locations
        locations = [Path("data/raw"), Path("data/processed"), Path("data/extracted")]

        found = False
        for loc in locations:
            for csv_file in loc.glob("*.csv"):
                df = pd.read_csv(csv_file)
                if (
                    "screening_id" in df.columns
                    and screening_id in df["screening_id"].values
                ):
                    csv = str(csv_file)
                    found = True
                    break
            if found:
                break

        if not found:
            console.print(f"[red]Paper {screening_id} not found[/red]")
            return

    df = pd.read_csv(csv)

    # Find the paper
    if "screening_id" in df.columns:
        paper_data = df[df["screening_id"] == screening_id]
    else:
        console.print(f"[red]No screening_id column in {csv}[/red]")
        return

    if paper_data.empty:
        console.print(f"[red]Paper {screening_id} not found in {csv}[/red]")
        return

    paper = paper_data.iloc[0]

    # Display paper details
    console.print(f"\n[bold blue]Paper Details: {screening_id}[/bold blue]\n")

    # Key fields
    key_fields = ["title", "authors", "year", "venue", "doi", "url"]
    for field in key_fields:
        if field in paper and pd.notna(paper[field]):
            console.print(f"[bold]{field.title()}:[/bold] {paper[field]}")

    # Abstract
    if "abstract" in paper and pd.notna(paper["abstract"]):
        console.print(f"\n[bold]Abstract:[/bold]\n{paper['abstract'][:500]}...")

    # Extraction results
    extraction_fields = [
        "venue_type",
        "game_type",
        "open_ended",
        "quantitative",
        "llm_family",
        "llm_role",
        "awscale",
        "failure_modes",
    ]

    has_extraction = any(field in paper for field in extraction_fields)
    if has_extraction:
        console.print("\n[bold]Extraction Results:[/bold]")
        for field in extraction_fields:
            if field in paper and pd.notna(paper[field]):
                console.print(f"  • {field}: {paper[field]}")

    # Status information
    if "pdf_status" in paper:
        console.print(f"\n[bold]PDF Status:[/bold] {paper['pdf_status']}")
    if "extraction_status" in paper:
        console.print(f"[bold]Extraction Status:[/bold] {paper['extraction_status']}")


if __name__ == "__main__":
    cli()
