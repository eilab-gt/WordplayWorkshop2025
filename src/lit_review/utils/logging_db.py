"""SQLite logging handler for the pipeline."""

import builtins
import contextlib
import json
import logging
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class SQLiteHandler(logging.Handler):
    """Custom logging handler that writes to SQLite database."""

    def __init__(self, db_path: Path, table_name: str = "pipeline_logs"):
        """Initialize SQLite logging handler.

        Args:
            db_path: Path to SQLite database file
            table_name: Name of the logging table
        """
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the database table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module TEXT,
                function TEXT,
                level TEXT,
                status TEXT,
                message TEXT,
                error_trace TEXT,
                metadata TEXT
            )
        """
        )

        # Create indices
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON {self.table_name} (timestamp)
        """
        )

        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_module
            ON {self.table_name} (module)
        """
        )

        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_level
            ON {self.table_name} (level)
        """
        )

        conn.commit()
        conn.close()

    def emit(self, record: logging.LogRecord):
        """Emit a log record to the database.

        Args:
            record: Log record to emit
        """
        try:
            # Extract information
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            module = record.name
            function = record.funcName
            level = record.levelname
            message = self.format(record)

            # Extract status from message if present
            status = "info"
            if "success" in message.lower():
                status = "success"
            elif "error" in message.lower() or level == "ERROR":
                status = "error"
            elif "warning" in message.lower() or level == "WARNING":
                status = "warning"

            # Get error trace if exception
            error_trace = None
            if record.exc_info:
                error_trace = "".join(traceback.format_exception(*record.exc_info))

            # Extract any extra metadata
            metadata = {}
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    metadata[key] = str(value)

            metadata_json = json.dumps(metadata) if metadata else None

            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                f"""
                INSERT INTO {self.table_name}
                (timestamp, module, function, level, status, message, error_trace, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    module,
                    function,
                    level,
                    status,
                    message,
                    error_trace,
                    metadata_json,
                ),
            )

            conn.commit()
            conn.close()

        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)


class LoggingDatabase:
    """Interface for querying the logging database."""

    def __init__(self, db_path: Path, table_name: str = "pipeline_logs"):
        """Initialize logging database interface.

        Args:
            db_path: Path to SQLite database
            table_name: Name of the logging table
        """
        self.db_path = db_path
        self.table_name = table_name

    def query_logs(
        self,
        module: str | None = None,
        level: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query logs with filters.

        Args:
            module: Filter by module name
            level: Filter by log level
            status: Filter by status
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results

        Returns:
            List of log entries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = f"SELECT * FROM {self.table_name} WHERE 1=1"
        params = []

        if module:
            query += " AND module LIKE ?"
            params.append(f"%{module}%")

        if level:
            query += " AND level = ?"
            params.append(level)

        if status:
            query += " AND status = ?"
            params.append(status)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        # Convert to dictionaries
        logs = []
        for row in cursor.fetchall():
            log_entry = dict(row)
            # Parse metadata JSON if present
            if log_entry.get("metadata"):
                with contextlib.suppress(builtins.BaseException):
                    log_entry["metadata"] = json.loads(log_entry["metadata"])
            logs.append(log_entry)

        conn.close()
        return logs

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics from logs.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total logs
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        stats["total_logs"] = cursor.fetchone()[0]

        # Logs by level
        cursor.execute(
            f"""
            SELECT level, COUNT(*) as count
            FROM {self.table_name}
            GROUP BY level
        """
        )
        stats["by_level"] = dict(cursor.fetchall())

        # Logs by status
        cursor.execute(
            f"""
            SELECT status, COUNT(*) as count
            FROM {self.table_name}
            GROUP BY status
        """
        )
        stats["by_status"] = dict(cursor.fetchall())

        # Logs by module (top 10)
        cursor.execute(
            f"""
            SELECT module, COUNT(*) as count
            FROM {self.table_name}
            GROUP BY module
            ORDER BY count DESC
            LIMIT 10
        """
        )
        stats["top_modules"] = dict(cursor.fetchall())

        # Time range
        cursor.execute(
            f"""
            SELECT MIN(timestamp) as first_log, MAX(timestamp) as last_log
            FROM {self.table_name}
        """
        )
        row = cursor.fetchone()
        if row and row[0]:
            stats["time_range"] = {"first": row[0], "last": row[1]}

        # Error count
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {self.table_name}
            WHERE level = 'ERROR' OR status = 'error'
        """
        )
        stats["error_count"] = cursor.fetchone()[0]

        conn.close()
        return stats

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent error logs.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error log entries
        """
        return self.query_logs(level="ERROR", limit=limit)

    def export_to_csv(self, output_path: Path, filters: dict[str, Any] | None = None):
        """Export logs to CSV file.

        Args:
            output_path: Path for CSV output
            filters: Optional filters to apply
        """
        import pandas as pd

        # Query logs with filters
        logs = self.query_logs(**(filters or {}), limit=10000)

        # Convert to DataFrame
        df = pd.DataFrame(logs)

        # Save to CSV
        df.to_csv(output_path, index=False)

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Remove logs older than specified days.

        Args:
            days_to_keep: Number of days to keep logs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        cursor.execute(
            f"""
            DELETE FROM {self.table_name}
            WHERE timestamp < ?
        """,
            (cutoff_date.isoformat(),),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        
        # Vacuum to reclaim space (must be done outside a transaction)
        conn.execute("VACUUM")
        
        conn.close()

        return deleted_count


def setup_db_logging(config) -> SQLiteHandler:
    """Set up database logging for the pipeline.

    Args:
        config: Configuration object

    Returns:
        SQLiteHandler instance
    """
    # Create handler
    db_handler = SQLiteHandler(config.logging_db_path)

    # Set formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    db_handler.setFormatter(formatter)

    # Add to root logger
    logger = logging.getLogger()
    logger.addHandler(db_handler)

    return db_handler


# Add to imports if needed
from datetime import timedelta
