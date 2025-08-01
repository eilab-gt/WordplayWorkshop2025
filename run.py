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
from src.lit_review.extraction.enhanced_llm_extractor import EnhancedLLMExtractor
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher, ScreenUI
from src.lit_review.utils import Exporter, load_config
from src.lit_review.utils.content_cache import ContentCache
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
@click.option(
    "--config", default="config/config.yaml", help="Path to configuration file"
)
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
@click.option("--filter-keywords", help="Include keywords (comma-separated)")
@click.option("--exclude-keywords", help="Exclude keywords (comma-separated)")
@click.option(
    "--min-keyword-matches", default=1, help="Minimum keyword matches required"
)
@click.pass_context
def harvest(
    ctx,
    query,
    sources,
    max_results,
    parallel,
    output,
    filter_keywords,
    exclude_keywords,
    min_keyword_matches,
):
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

    # Apply keyword filtering if specified
    if filter_keywords or exclude_keywords:
        console.print("\n[bold]Applying keyword filters...[/bold]")
        original_count = len(df)

        # Parse keywords
        include_list = (
            [k.strip() for k in filter_keywords.split(",")] if filter_keywords else None
        )
        exclude_list = (
            [k.strip() for k in exclude_keywords.split(",")]
            if exclude_keywords
            else None
        )

        # Apply filtering using SearchHarvester
        temp_harvester = SearchHarvester(config)

        # Convert DataFrame to Paper objects for filtering
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

        # Filter papers by keywords
        filtered_papers = []
        for paper in papers:
            if not paper.abstract:
                continue

            abstract_lower = paper.abstract.lower()

            # Check exclusions first
            if exclude_list:
                excluded = any(
                    keyword.lower() in abstract_lower for keyword in exclude_list
                )
                if excluded:
                    continue

            # Check inclusions
            if include_list:
                matches = sum(
                    1 for keyword in include_list if keyword.lower() in abstract_lower
                )
                if matches >= min_keyword_matches:
                    filtered_papers.append(paper)
            else:
                # No inclusion filter, add if not excluded
                filtered_papers.append(paper)

        # Convert back to DataFrame
        if filtered_papers:
            df = pd.DataFrame([p.to_dict() for p in filtered_papers])
            console.print(
                f"[green]✓[/green] Filtered from {original_count} to {len(df)} papers"
            )
        else:
            console.print(
                "[yellow]Warning:[/yellow] No papers matched the keyword filters"
            )
            df = pd.DataFrame()

    # Save results
    output_path = Path(output) if output else config.raw_papers_path
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    harvester.save_results(df, output_path)
    console.print(f"\n[green]✓[/green] Results saved to: {output_path}")
    
    # Save config snapshot alongside results for reproducibility
    from datetime import datetime
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_backup = output_path.parent / f"config_used_{timestamp}.yaml"
    shutil.copy2("config/config.yaml", config_backup)
    console.print(f"[green]✓[/green] Config snapshot saved to: {config_backup}")


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
@click.option(
    "--use-enhanced", is_flag=True, help="Use enhanced LLM extractor with LiteLLM"
)
@click.option(
    "--llm-service-url", default="http://localhost:8000", help="LLM service URL"
)
@click.option(
    "--prefer-tex", is_flag=True, help="Prefer TeX/HTML over PDFs when available"
)
@click.pass_context
def extract(
    ctx,
    input_file,
    output,
    parallel,
    skip_llm,
    skip_tagging,
    use_enhanced,
    llm_service_url,
    prefer_tex,
):
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
    # Handle both CSV and Excel files
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    # Filter to included papers
    if "include_ft" in df.columns and df["include_ft"].notna().any():
        included_df = df[df["include_ft"] == "yes"].copy()
        console.print(
            f"Processing {len(included_df)} papers marked for full-text inclusion"
        )
    elif "include_ta" in df.columns and df["include_ta"].notna().any():
        # Use title-abstract screening if no full-text screening done
        included_df = df[df["include_ta"] == 1].copy()
        console.print(
            f"Processing {len(included_df)} papers marked for inclusion (title/abstract)"
        )
    elif "auto_include" in df.columns:
        # Use automated inclusion if available
        included_df = df[df["auto_include"] == True].copy()
        console.print(
            f"Processing {len(included_df)} papers marked by automated inclusion"
        )
    else:
        included_df = df.copy()
        console.print(f"Processing all {len(included_df)} papers (no screening data)")

    if len(included_df) == 0:
        console.print("[yellow]No papers to extract![/yellow]")
        return

    # LLM extraction
    if not skip_llm:
        if use_enhanced:
            console.print("\n[bold]Extracting with Enhanced LLM (LiteLLM)...[/bold]")
            extractor = EnhancedLLMExtractor(config, llm_service_url)

            # Check service health
            if not extractor.check_service_health():
                console.print("[red]Error:[/red] LLM service is not running")
                console.print(
                    "Start the service with: python -m src.lit_review.llm_service"
                )
                console.print(f"Expected URL: {llm_service_url}")
                return

            # Show available models
            models = extractor.get_available_models()
            console.print("\n[bold]Available models:[/bold]")
            for model, info in models.items():
                status = "[green]✓[/green]" if info["available"] else "[red]✗[/red]"
                console.print(f"  {status} {model}: {info['description']}")

            included_df = extractor.extract_all(included_df, parallel=parallel)
        else:
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


@cli.command("cache-stats")
@click.pass_context
def cache_stats(ctx):
    """Show cache statistics and usage."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    console.print("\n[bold]Content Cache Statistics[/bold]")

    # Initialize content cache
    content_cache = ContentCache(config)
    stats = content_cache.get_statistics()

    # Create summary table
    table = Table(title="Cache Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entries", str(stats["total_entries"]))
    table.add_row("Total Size", f"{stats['total_size_mb']:.1f} MB")
    table.add_row("Cache Hits", str(stats["cache_hits"]))
    table.add_row("Cache Misses", str(stats["cache_misses"]))
    table.add_row("Hit Rate", f"{stats['hit_rate']:.1f}%")
    table.add_row("Time Saved", f"{stats['time_saved_seconds']/60:.1f} minutes")
    table.add_row("Bandwidth Saved", f"{stats['bytes_saved']/1024/1024:.1f} MB")

    console.print(table)

    # Show breakdown by type
    if stats.get("by_type"):
        type_table = Table(title="Cache by Content Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")
        type_table.add_column("Size (MB)", style="green")

        for content_type, type_stats in stats["by_type"].items():
            type_table.add_row(
                content_type.upper(),
                str(type_stats["count"]),
                f"{type_stats['total_size_mb']:.1f}",
            )

        console.print("\n")
        console.print(type_table)


@cli.command("cache-clean")
@click.option("--type", help="Content type to clean (pdf, html, tex, or all)")
@click.option("--days", default=90, help="Keep files newer than N days")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def cache_clean(ctx, type, days, force):
    """Clean content cache."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    content_cache = ContentCache(config)

    # Get current stats before cleaning
    stats = content_cache.get_statistics()

    if type and type != "all":
        # Clean specific type
        if type not in ["pdf", "html", "tex"]:
            console.print(
                f"[red]Error:[/red] Invalid type '{type}'. Use pdf, html, tex, or all"
            )
            return

        type_stats = stats.get("by_type", {}).get(type, {})
        if not force and type_stats.get("count", 0) > 0:
            if not click.confirm(
                f"Remove {type_stats['count']} {type.upper()} files ({type_stats.get('total_size_mb', 0):.1f} MB)?"
            ):
                return

        removed = content_cache.clear_cache(content_type=type)
        console.print(
            f"[green]✓[/green] Removed {removed} {type.upper()} cache entries"
        )

    elif days:
        # Clean old entries
        if not force:
            if not click.confirm(f"Remove cache entries older than {days} days?"):
                return

        removed = content_cache.cleanup_old_entries(days=days)
        console.print(f"[green]✓[/green] Removed {removed} old cache entries")

    else:
        # Clean all
        if not force:
            if not click.confirm(
                f"Remove ALL {stats['total_entries']} cache entries ({stats['total_size_mb']:.1f} MB)?"
            ):
                return

        removed = content_cache.clear_cache()
        console.print(f"[green]✓[/green] Removed {removed} cache entries")


@cli.command("clean-cache")
@click.option("--pdfs", is_flag=True, help="Clean PDF cache")
@click.option("--logs", is_flag=True, help="Clean logs")
@click.option("--all", "clean_all", is_flag=True, help="Clean everything")
@click.option("--days", default=30, help="Keep files newer than N days")
@click.pass_context
def clean_cache(ctx, pdfs, logs, clean_all, days):
    """Clean up cache and temporary files (deprecated - use cache-clean)."""
    config = ctx.obj["config"]
    console = ctx.obj["console"]

    console.print(
        "[yellow]Note: This command is deprecated. Use 'cache-clean' for content cache management.[/yellow]"
    )

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
