#!/usr/bin/env python3
"""Main CLI interface for the literature review pipeline."""

import logging
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.lit_review.extraction import LLMExtractor, Tagger
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher, ScreenUI
from src.lit_review.utils import Exporter, load_config
from src.lit_review.visualization import Visualizer

# Initialize rich console
console = Console()


# Configure logging
def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/pipeline.log"), logging.StreamHandler()],
    )


@click.group()
@click.option("--config", default="config.yaml", help="Path to configuration file")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config, log_level):
    """Literature Review Pipeline - Automated systematic review for LLM wargaming papers."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Set up logging
    setup_logging(log_level)

    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["console"] = console

    console.print("[bold green]Literature Review Pipeline[/bold green]")
    console.print(f"Configuration: {config}")


@cli.command()
@click.option("--query", default="preset1", help="Query preset or custom query")
@click.option("--sources", multiple=True, help="Sources to search (default: all)")
@click.option("--max-results", default=100, help="Maximum results per source")
@click.option("--parallel/--sequential", default=True, help="Parallel search")
@click.option("--output", help="Output file path")
@click.pass_context
def harvest(ctx, query, sources, max_results, parallel, output):
    """Harvest papers from configured sources."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    console.print("\n[bold]Starting harvest...[/bold]")

    # Initialize harvester
    harvester = SearchHarvester(config)

    # Convert sources tuple to list
    sources_list = list(sources) if sources else None

    # Perform search
    with console.status("[bold green]Searching databases..."):
        df = harvester.search_all(
            sources=sources_list, max_results_per_source=max_results, parallel=parallel
        )

    # Show results
    console.print(f"\n[green]✓[/green] Found {len(df)} papers")

    # Show source breakdown
    if len(df) > 0:
        table = Table(title="Papers by Source")
        table.add_column("Source", style="cyan")
        table.add_column("Count", style="magenta")

        for source, count in df["source_db"].value_counts().items():
            table.add_row(source, str(count))

        console.print(table)

    # Normalize and deduplicate
    console.print("\n[bold]Normalizing and deduplicating...[/bold]")
    normalizer = Normalizer(config)
    df = normalizer.normalize_dataframe(df)

    console.print(f"[green]✓[/green] After deduplication: {len(df)} papers")

    # Save results
    output_path = Path(output) if output else config.raw_papers_path

    harvester.save_results(df, output_path)
    console.print(f"\n[green]✓[/green] Results saved to: {output_path}")


@cli.command("prepare-screen")
@click.option("--input", "input_file", help="Input CSV file")
@click.option("--output", help="Output screening file")
@click.option("--asreview", is_flag=True, help="Include ASReview format")
@click.pass_context
def prepare_screen(ctx, input_file, output, asreview):
    """Prepare screening sheet for manual review."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    # Load data
    input_path = Path(input_file) if input_file else config.raw_papers_path

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_path}")
        return

    console.print(f"\n[bold]Loading papers from {input_path}...[/bold]")
    df = pd.read_csv(input_path)
    console.print(f"Loaded {len(df)} papers")

    # Fetch PDFs
    console.print("\n[bold]Fetching PDFs...[/bold]")
    pdf_fetcher = PDFFetcher(config)
    df = pdf_fetcher.fetch_pdfs(df, parallel=True)

    # Prepare screening sheet
    console.print("\n[bold]Preparing screening sheet...[/bold]")
    screen_ui = ScreenUI(config)

    output_path = Path(output) if output else config.screening_progress_path

    screening_df = screen_ui.prepare_screening_sheet(
        df, output_path=output_path, include_asreview=asreview
    )

    # Show statistics
    stats = screen_ui.get_screening_statistics(screening_df)
    console.print(f"\n[green]✓[/green] Screening sheet prepared: {output_path}")
    console.print(f"Total papers: {stats['total_papers']}")

    # Show PDF statistics
    pdf_stats = pdf_fetcher.get_cache_statistics()
    console.print(
        f"\nPDF cache: {pdf_stats['total_files']} files ({pdf_stats['total_size_mb']:.1f} MB)"
    )


@cli.command()
@click.option("--input", "input_file", help="Screening progress CSV")
@click.option("--output", help="Output extraction file")
@click.option("--parallel/--sequential", default=True, help="Parallel extraction")
@click.option("--skip-llm", is_flag=True, help="Skip LLM extraction")
@click.option("--skip-tagging", is_flag=True, help="Skip regex tagging")
@click.pass_context
def extract(ctx, input_file, output, parallel, skip_llm, skip_tagging):
    """Extract structured information from screened papers."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    # Load screening data
    input_path = Path(input_file) if input_file else config.screening_progress_path

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Screening file not found: {input_path}")
        console.print("Run 'prepare-screen' first to create screening sheet")
        return

    console.print(f"\n[bold]Loading screening data from {input_path}...[/bold]")
    df = pd.read_csv(input_path)

    # Filter to included papers
    if "include_ft" in df.columns:
        included_df = df[df["include_ft"] == "yes"].copy()
        console.print(f"Processing {len(included_df)} papers marked for inclusion")
    else:
        included_df = df.copy()
        console.print(f"Processing all {len(included_df)} papers (no screening data)")

    if len(included_df) == 0:
        console.print("[yellow]No papers to extract![/yellow]")
        return

    # LLM extraction
    if not skip_llm:
        console.print("\n[bold]Extracting with LLM...[/bold]")
        extractor = LLMExtractor(config)
        included_df = extractor.extract_all(included_df, parallel=parallel)

    # Regex tagging
    if not skip_tagging:
        console.print("\n[bold]Applying regex tagging...[/bold]")
        tagger = Tagger(config)
        included_df = tagger.tag_papers(included_df, use_llm_results=not skip_llm)

        # Show failure mode summary
        failure_summary = tagger.get_failure_mode_summary(included_df)
        if len(failure_summary) > 0:
            console.print("\n[bold]Top failure modes detected:[/bold]")
            for _, row in failure_summary.head(5).iterrows():
                console.print(
                    f"  {row['failure_mode']}: {row['count']} papers ({row['percentage']:.1f}%)"
                )

    # Save results
    output_path = Path(output) if output else config.extraction_results_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    included_df.to_csv(output_path, index=False)
    console.print(f"\n[green]✓[/green] Extraction results saved to: {output_path}")


@cli.command()
@click.option("--input", "input_file", help="Extraction results CSV")
@click.option("--output-dir", help="Output directory for figures")
@click.option("--show", is_flag=True, help="Show plots interactively")
@click.pass_context
def visualise(ctx, input_file, output_dir, show):
    """Create visualizations from extraction results."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    # Load extraction data
    input_path = Path(input_file) if input_file else config.extraction_results_path

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Extraction file not found: {input_path}")
        console.print("Run 'extract' first to create extraction results")
        return

    console.print(f"\n[bold]Loading extraction data from {input_path}...[/bold]")
    df = pd.read_csv(input_path)
    console.print(f"Loaded {len(df)} papers")

    # Create visualizations
    console.print("\n[bold]Creating visualizations...[/bold]")
    visualizer = Visualizer(config)

    if output_dir:
        visualizer.output_dir = Path(output_dir)
        visualizer.output_dir.mkdir(parents=True, exist_ok=True)

    # Create all visualizations
    figures = []
    save = not show  # Save if not showing

    with console.status("[bold green]Generating plots..."):
        figures = visualizer.create_all_visualizations(df, save=save)

    if save:
        console.print(
            f"\n[green]✓[/green] Created {len(figures)} visualizations in: {visualizer.output_dir}"
        )
        for fig in figures:
            console.print(f"  - {fig.name}")

    # Create summary report
    summary = visualizer.create_summary_report(df)
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Total papers: {summary['total_papers']}")
    console.print(f"  Year range: {summary['year_range']}")

    if "awscale" in summary:
        console.print(f"  AWScale mean: {summary['awscale']['mean']:.2f}")

    if "top_failure_modes" in summary:
        console.print("\n  Top failure modes:")
        for mode, count in list(summary["top_failure_modes"].items())[:5]:
            console.print(f"    - {mode}: {count}")


@cli.command()
@click.option("--input", "input_file", help="Extraction results CSV")
@click.option("--output", help="Output package name")
@click.option("--include-pdfs", is_flag=True, help="Include PDFs in export")
@click.option("--include-logs", is_flag=True, help="Include logs in export")
@click.option("--zenodo", is_flag=True, help="Upload to Zenodo")
@click.pass_context
def export(ctx, input_file, output, include_pdfs, include_logs, zenodo):
    """Export results in various formats."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    # Load extraction data
    input_path = Path(input_file) if input_file else config.extraction_results_path

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Extraction file not found: {input_path}")
        return

    console.print(f"\n[bold]Loading extraction data from {input_path}...[/bold]")
    df = pd.read_csv(input_path)

    # Create visualizations first
    console.print("\n[bold]Creating visualizations for export...[/bold]")
    visualizer = Visualizer(config)
    figures = visualizer.create_all_visualizations(df, save=True)

    # Create summary
    summary = visualizer.create_summary_report(df)

    # Export package
    console.print("\n[bold]Creating export package...[/bold]")
    exporter = Exporter(config)

    # Override settings from command line
    if include_pdfs:
        exporter.include_pdfs = True
    if include_logs:
        exporter.include_logs = True
    if zenodo:
        exporter.zenodo_enabled = True

    archive_path = exporter.export_full_package(
        extraction_df=df, figures=figures, summary=summary, output_name=output
    )

    console.print(f"\n[green]✓[/green] Export package created: {archive_path}")

    # Also export BibTeX
    bibtex_path = exporter.export_bibtex(df)
    console.print(f"[green]✓[/green] BibTeX file created: {bibtex_path}")


@cli.command()
@click.option("--query", help="Search query for specific papers")
@click.option("--max-results", default=5, help="Maximum results")
@click.pass_context
def test(ctx, query, max_results):
    """Test the pipeline with a small search."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    console.print("\n[bold]Running test search...[/bold]")

    # Test search
    harvester = SearchHarvester(config)

    if not query:
        query = "LLM wargame"

    console.print(f"Query: {query}")

    # Search only Google Scholar for testing
    df = harvester.search_all(
        sources=["google_scholar"], max_results_per_source=max_results, parallel=False
    )

    if len(df) > 0:
        console.print(f"\n[green]✓[/green] Found {len(df)} papers")

        # Show sample results
        console.print("\n[bold]Sample results:[/bold]")
        for _idx, row in df.head(3).iterrows():
            console.print(f"\n[cyan]{row['title']}[/cyan]")
            console.print(f"  Authors: {row['authors']}")
            console.print(f"  Year: {row['year']}")
            console.print(f"  Source: {row['source_db']}")
    else:
        console.print("[red]No results found![/red]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status and statistics."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    console.print("\n[bold]Pipeline Status[/bold]\n")

    # Check for existing files
    files_to_check = [
        ("Raw papers", config.raw_papers_path),
        ("Screening sheet", config.screening_progress_path),
        ("Extraction results", config.extraction_results_path),
        ("Log database", config.logging_db_path),
    ]

    table = Table(title="Data Files")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Records", style="magenta")

    for name, path in files_to_check:
        if path.exists():
            status = "✓ Exists"
            if path.suffix == ".csv":
                try:
                    df = pd.read_csv(path)
                    records = str(len(df))
                except:
                    records = "Error"
            else:
                records = "-"
        else:
            status = "✗ Not found"
            records = "-"

        table.add_row(name, status, records)

    console.print(table)

    # Show PDF cache statistics
    pdf_fetcher = PDFFetcher(config)
    cache_stats = pdf_fetcher.get_cache_statistics()

    console.print("\n[bold]PDF Cache:[/bold]")
    console.print(f"  Location: {cache_stats['cache_dir']}")
    console.print(f"  Files: {cache_stats['total_files']}")
    console.print(f"  Size: {cache_stats['total_size_mb']:.1f} MB")

    # Show configuration summary
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Search years: {config.search_years[0]}-{config.search_years[1]}")
    console.print(f"  LLM model: {config.llm_model}")
    console.print(f"  Parallel workers: {config.parallel_workers}")


@cli.command("clean-cache")
@click.option("--pdfs", is_flag=True, help="Clean PDF cache")
@click.option("--logs", is_flag=True, help="Clean logs")
@click.option("--all", "clean_all", is_flag=True, help="Clean everything")
@click.option("--days", default=30, help="Keep files newer than N days")
@click.pass_context
def clean_cache(ctx, pdfs, logs, clean_all, days):
    """Clean up cache and temporary files."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    if clean_all:
        pdfs = logs = True

    if not (pdfs or logs):
        console.print("[yellow]Nothing to clean! Use --pdfs, --logs, or --all[/yellow]")
        return

    console.print(f"\n[bold]Cleaning up files older than {days} days...[/bold]")

    if pdfs:
        pdf_fetcher = PDFFetcher(config)
        pdf_fetcher.cleanup_cache(keep_days=days)
        console.print("[green]✓[/green] Cleaned PDF cache")

    if logs:
        # Clean old log files
        log_dir = Path(config.log_dir)
        if log_dir.exists():
            import time

            cutoff_time = time.time() - (days * 24 * 60 * 60)
            removed = 0

            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed += 1

            if removed > 0:
                console.print(f"[green]✓[/green] Removed {removed} old log files")


if __name__ == "__main__":
    cli()
