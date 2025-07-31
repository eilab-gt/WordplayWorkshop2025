"""Common utilities for scripts."""

from pathlib import Path
from typing import Optional

import pandas as pd


def find_latest_file(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    """Find the most recent file matching pattern in directory.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files

    Returns:
        Path to most recent file or None if no files found
    """
    files = list(directory.glob(pattern))
    return max(files, key=lambda x: x.stat().st_mtime) if files else None


def parse_failure_modes(series: pd.Series) -> list[str]:
    """Parse failure modes from pipe-separated values.

    Args:
        series: Pandas series containing failure modes

    Returns:
        List of individual failure mode strings
    """
    modes = []
    for value in series.dropna():
        value_str = str(value)
        if value_str and value_str != "nan":
            if "|" in value_str:
                modes.extend(mode.strip() for mode in value_str.split("|"))
            else:
                modes.append(value_str.strip())
    return modes


def safe_column_value(df: pd.DataFrame, column: str, operation: str, default=None):
    """Safely get a column value with an operation.

    Args:
        df: DataFrame to operate on
        column: Column name
        operation: Operation to perform (e.g., 'min', 'max', 'mean')
        default: Default value if column missing or all NaN

    Returns:
        Result of operation or default value
    """
    if column in df and not df[column].isna().all():
        return getattr(df[column], operation)()
    return default


# Constants for common values
BAR_CHART_SCALE = 5
DEMO_SLEEP_TIME = 0.2
EXTRACTION_SLEEP_TIME = 0.3
