#!/usr/bin/env python3
"""
Check if all required dependencies are installed and working.
"""

import os
import sys
from importlib import import_module
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Required packages and their import names
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "click": "click",
    "rich": "rich",
    "pyyaml": "yaml",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "scholarly": "scholarly",
    "arxiv": "arxiv",
    "openai": "openai",
    "python-dotenv": "dotenv",
    "openpyxl": "openpyxl",
    "PyPDF2": "PyPDF2",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "pre-commit": "pre_commit",
}

# Optional packages
OPTIONAL_PACKAGES = {"jupyter": "jupyter", "notebook": "notebook", "ipython": "IPython"}


def check_package(import_name: str) -> tuple[bool, str]:
    """Check if a package can be imported."""
    try:
        version = getattr(import_module(import_name), "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"
    except Exception as e:
        return False, f"error: {e!s}"


def check_api_setup():
    """Check if API keys are configured."""

    api_status = {}

    # Check environment variables
    api_status["OPENAI_API_KEY"] = "set" if os.getenv("OPENAI_API_KEY") else "not set"
    api_status["SEMANTIC_SCHOLAR_API_KEY"] = (
        "set" if os.getenv("SEMANTIC_SCHOLAR_API_KEY") else "not set"
    )

    # Check config file
    config_exists = Path("config.yaml").exists()
    api_status["config.yaml"] = "exists" if config_exists else "not found"

    return api_status


def main():
    """Run dependency checks."""
    console.print(
        Panel.fit(
            "[bold blue]Literature Review Pipeline - Dependency Check[/bold blue]",
            box="double",
        )
    )

    # Check required packages
    console.print("\n[bold]Required Packages:[/bold]")
    table = Table()
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")

    all_good = True
    for package, import_name in REQUIRED_PACKAGES.items():
        installed, version = check_package(import_name)
        status = "✅" if installed else "❌"
        if not installed:
            all_good = False
        table.add_row(package, status, version)

    console.print(table)

    # Check optional packages
    console.print("\n[bold]Optional Packages:[/bold]")
    opt_table = Table()
    opt_table.add_column("Package", style="cyan")
    opt_table.add_column("Status", style="green")
    opt_table.add_column("Version", style="yellow")

    for package, import_name in OPTIONAL_PACKAGES.items():
        installed, version = check_package(import_name)
        status = "✅" if installed else "⚠️"
        opt_table.add_row(package, status, version)

    console.print(opt_table)

    # Check Python version
    console.print("\n[bold]Python Version:[/bold]")
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    console.print(f"  ✅ Python {python_version}")

    # Check API setup
    console.print("\n[bold]API Configuration:[/bold]")
    api_status = check_api_setup()
    for key, status in api_status.items():
        icon = "✅" if status in ["set", "exists"] else "⚠️"
        console.print(f"  {icon} {key}: {status}")

    # Check directories
    console.print("\n[bold]Directory Structure:[/bold]")

    required_dirs = ["data", "outputs", "pdf_cache", "logs", "src/lit_review"]
    for dir_name in required_dirs:
        exists = Path(dir_name).exists()
        icon = "✅" if exists else "⚠️"
        console.print(f"  {icon} {dir_name}")

    # Final status
    console.print("\n" + "=" * 50)
    if all_good:
        console.print(
            "[bold green]✅ All required dependencies are installed![/bold green]"
        )
        console.print("\nYou're ready to use the pipeline. Try:")
        console.print("  python run.py --help")
    else:
        console.print("[bold red]❌ Some required dependencies are missing![/bold red]")
        console.print("\nTo fix:")
        console.print("  1. Activate your virtual environment")
        console.print("  2. Run: uv pip install -e .")
        console.print("  3. Add API keys to config.yaml or .env")

    # Return exit code
    return 0 if all_good else 1


if __name__ == "__main__":
    exit(main())
