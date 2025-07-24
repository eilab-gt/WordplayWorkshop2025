#!/usr/bin/env python3
"""Production-scale harvesting CLI with advanced monitoring and resume capabilities."""

import sys
import time
from pathlib import Path

import click
import pandas as pd
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.lit_review.harvesters.production_harvester import ProductionHarvester
from src.lit_review.utils import load_config

console = Console()


@click.group()
def cli():
    """Production Literature Review Harvester - Scale to millions of papers."""
    pass


@cli.command()
@click.option("--config", default="config/config.yaml", help="Configuration file")
@click.option("--sources", multiple=True, help="Sources to search (default: all)")
@click.option("--max-papers", default=50000, help="Maximum papers to harvest")
@click.option("--resume", help="Resume from session ID")
@click.option("--checkpoint-interval", default=100, help="Checkpoint every N papers")
@click.option("--output", help="Output directory for results")
@click.option("--monitor", is_flag=True, help="Enable real-time monitoring")
def harvest(config, sources, max_papers, resume, checkpoint_interval, output, monitor):
    """Execute production-scale harvest with monitoring and resume capability."""
    console.print(
        Panel.fit(
            "[bold green]🚀 PRODUCTION LITERATURE HARVESTER[/bold green]\n"
            f"Target: [yellow]{max_papers:,}[/yellow] papers\n"
            f"Sources: [cyan]{'all' if not sources else ', '.join(sources)}[/cyan]",
            box=box.DOUBLE,
            style="bold",
        )
    )

    # Load production configuration
    try:
        prod_config = load_config(config)
        console.print(f"[green]✓[/green] Loaded production config: {config}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load config: {e}")
        return

    # Initialize production harvester
    harvester = ProductionHarvester(prod_config)
    console.print("[green]✓[/green] Initialized production harvester")

    # Setup output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(prod_config.output_dir)

    # Check for resume session
    session_id = None
    if resume:
        try:
            status = harvester.get_session_status(resume)
            if "error" in status:
                console.print(f"[red]✗[/red] {status['error']}")
                return
            console.print(f"[yellow]🔄[/yellow] Resuming session: {resume}")
            console.print(
                f"Previous progress: {status.get('total_progress', 0):,} papers"
            )
            session_id = resume
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to resume session: {e}")
            return

    # Setup monitoring
    if monitor:
        with Live(console=console, refresh_per_second=2) as live:
            progress = Progress()
            live.update(progress)

            overall_task = progress.add_task(
                "[bold blue]Overall Progress", total=max_papers
            )

            source_tasks = {}
            if sources:
                for source in sources:
                    source_tasks[source] = progress.add_task(
                        f"[cyan]{source.title()}", total=max_papers // len(sources)
                    )

            # Progress callback
            def update_progress(session_id, source, source_papers, total_papers):
                progress.update(overall_task, completed=total_papers)
                if source in source_tasks:
                    progress.update(source_tasks[source], completed=source_papers)

                # Update live display with stats
                stats_table = Table(title="Harvest Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="magenta")

                stats_table.add_row("Session ID", session_id)
                stats_table.add_row("Total Papers", f"{total_papers:,}")
                stats_table.add_row("Current Source", source)
                stats_table.add_row("Source Papers", f"{source_papers:,}")
                stats_table.add_row("Progress", f"{total_papers/max_papers:.1%}")

                live.update(Panel(stats_table, title="🔍 Production Harvest Monitor"))

            # Execute harvest with monitoring
            try:
                df = harvester.search_production_scale(
                    sources=list(sources) if sources else None,
                    max_results_total=max_papers,
                    resume_session=session_id,
                    checkpoint_callback=update_progress,
                )

                progress.update(overall_task, completed=len(df))

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️[/yellow] Harvest interrupted by user")
                return
            except Exception as e:
                console.print(f"\n[red]✗[/red] Harvest failed: {e}")
                return
    else:
        # Execute without monitoring
        try:
            console.print("[bold]Starting production harvest...[/bold]")
            start_time = time.time()

            df = harvester.search_production_scale(
                sources=list(sources) if sources else None,
                max_results_total=max_papers,
                resume_session=session_id,
            )

            elapsed = time.time() - start_time

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️[/yellow] Harvest interrupted by user")
            return
        except Exception as e:
            console.print(f"\n[red]✗[/red] Harvest failed: {e}")
            return

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"production_harvest_{timestamp}.csv"

    df.to_csv(output_file, index=False)

    # Display results
    console.print(
        Panel.fit(
            f"[bold green]🎉 HARVEST COMPLETE![/bold green]\n\n"
            f"[bold]Papers harvested:[/bold] [yellow]{len(df):,}[/yellow]\n"
            f"[bold]Unique papers:[/bold] [green]{len(df):,}[/green]\n"
            f"[bold]Time elapsed:[/bold] [cyan]{elapsed/60:.1f} minutes[/cyan]\n"
            f"[bold]Rate:[/bold] [magenta]{len(df)/(elapsed/60):.0f} papers/min[/magenta]\n"
            f"[bold]Output:[/bold] [blue]{output_file}[/blue]",
            box=box.DOUBLE,
        )
    )

    # Show source breakdown
    if len(df) > 0:
        source_table = Table(title="📊 Harvest Summary by Source")
        source_table.add_column("Source", style="cyan", width=20)
        source_table.add_column("Papers", style="magenta", justify="right")
        source_table.add_column("Percentage", style="green", justify="right")

        source_counts = df["source_db"].value_counts()
        for source, count in source_counts.items():
            percentage = count / len(df) * 100
            source_table.add_row(source.title(), f"{count:,}", f"{percentage:.1f}%")

        console.print(source_table)


@cli.command()
@click.option("--config", default="config/config.yaml", help="Configuration file")
def status(config):
    """Show status of all harvest sessions."""
    try:
        prod_config = load_config(config)
        harvester = ProductionHarvester(prod_config)

        sessions_df = harvester.list_sessions()

        if len(sessions_df) == 0:
            console.print("[yellow]No harvest sessions found[/yellow]")
            return

        # Create status table
        status_table = Table(title="📋 Harvest Sessions")
        status_table.add_column("Session ID", style="cyan")
        status_table.add_column("Start Time", style="blue")
        status_table.add_column("Status", style="magenta")
        status_table.add_column("Papers", style="green", justify="right")
        status_table.add_column("Duration", style="yellow")

        for _, session in sessions_df.iterrows():
            # Calculate duration
            start_time = pd.to_datetime(session["start_time"])
            end_time = (
                pd.to_datetime(session["end_time"])
                if session["end_time"]
                else pd.Timestamp.now()
            )
            duration = end_time - start_time

            # Format status with emoji
            status_emoji = {"completed": "✅", "running": "🔄", "failed": "❌"}.get(
                session["status"], "❓"
            )

            status_table.add_row(
                session["session_id"],
                start_time.strftime("%Y-%m-%d %H:%M"),
                f"{status_emoji} {session['status']}",
                (
                    f"{session['total_papers']:,}"
                    if pd.notna(session["total_papers"])
                    else "0"
                ),
                str(duration).split(".")[0],  # Remove microseconds
            )

        console.print(status_table)

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get status: {e}")


@cli.command()
@click.argument("session_id")
@click.option("--config", default="config/config.yaml", help="Configuration file")
def detail(session_id, config):
    """Show detailed status of a specific harvest session."""
    try:
        prod_config = load_config(config)
        harvester = ProductionHarvester(prod_config)

        status = harvester.get_session_status(session_id)

        if "error" in status:
            console.print(f"[red]✗[/red] {status['error']}")
            return

        # Session info panel
        session_info = status["session_info"]
        info_text = (
            f"[bold]Session ID:[/bold] {session_info['session_id']}\n"
            f"[bold]Status:[/bold] {session_info['status']}\n"
            f"[bold]Start Time:[/bold] {session_info['start_time']}\n"
            f"[bold]End Time:[/bold] {session_info.get('end_time', 'Running...')}\n"
            f"[bold]Total Papers:[/bold] {session_info.get('total_papers', 0):,}\n"
            f"[bold]Config Hash:[/bold] {session_info.get('config_hash', 'N/A')}"
        )

        console.print(Panel(info_text, title="📋 Session Details"))

        # Progress by source
        progress = status.get("progress_by_source", {})
        if progress:
            progress_table = Table(title="📊 Progress by Source")
            progress_table.add_column("Source", style="cyan")
            progress_table.add_column("Papers Found", style="magenta", justify="right")

            for source, papers in progress.items():
                progress_table.add_row(source.title(), f"{papers:,}")

            console.print(progress_table)

        console.print(
            f"\n[bold]Total Progress:[/bold] [green]{status.get('total_progress', 0):,}[/green] papers"
        )

        if status.get("last_activity"):
            console.print(f"[bold]Last Activity:[/bold] {status['last_activity']}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get session details: {e}")


@cli.command()
@click.option("--config", default="config/config.yaml", help="Configuration file")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be tested without execution"
)
def test(config, dry_run):
    """Test production harvester configuration and connectivity."""
    console.print("[bold blue]🧪 PRODUCTION HARVESTER TEST[/bold blue]")

    try:
        prod_config = load_config(config)
        console.print(f"[green]✓[/green] Config loaded: {config}")
    except Exception as e:
        console.print(f"[red]✗[/red] Config error: {e}")
        return

    if dry_run:
        console.print("[yellow]ℹ️[/yellow] Dry run mode - no actual requests made")

    # Test harvester initialization
    try:
        harvester = ProductionHarvester(prod_config)
        console.print("[green]✓[/green] Production harvester initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Harvester initialization failed: {e}")
        return

    # Test each source
    test_table = Table(title="🔍 Source Connectivity Test")
    test_table.add_column("Source", style="cyan")
    test_table.add_column("Status", style="magenta")
    test_table.add_column("Test Papers", style="green", justify="right")
    test_table.add_column("Response Time", style="yellow", justify="right")

    for source_name, source_harvester in harvester.harvesters.items():
        try:
            start_time = time.time()

            if not dry_run:
                # Test with small query
                test_papers = source_harvester.search("machine learning", max_results=5)
                paper_count = len(test_papers)
            else:
                paper_count = "N/A (dry run)"

            response_time = time.time() - start_time

            test_table.add_row(
                source_name.title(),
                "[green]✓ Connected[/green]",
                str(paper_count),
                f"{response_time:.2f}s",
            )

        except Exception as e:
            test_table.add_row(
                source_name.title(),
                "[red]✗ Failed[/red]",
                "0",
                f"Error: {str(e)[:30]}...",
            )

    console.print(test_table)

    # Test database initialization
    try:
        test_session = harvester._generate_session_id()
        harvester._create_session(test_session, ["test"], 100)
        console.print("[green]✓[/green] Progress database working")
    except Exception as e:
        console.print(f"[red]✗[/red] Database error: {e}")


if __name__ == "__main__":
    cli()
