#!/usr/bin/env python3
"""
Literature Review Pipeline - Module Entry Point

This module entry point redirects to the main run.py CLI.
Use either:
  - python run.py [command] [options]
  - python -m src.lit_review [command] [options]

Both are equivalent.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Redirect to the main run.py CLI."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    run_py = project_root / "run.py"
    
    # Execute run.py with all passed arguments
    cmd = [sys.executable, str(run_py)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()