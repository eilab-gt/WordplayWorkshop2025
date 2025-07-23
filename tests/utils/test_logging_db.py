"""Tests for the SQLite logging module."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from src.lit_review.utils import LoggingDatabase, SQLiteHandler, setup_db_logging


class TestSQLiteHandler:
    """Test cases for SQLiteHandler class."""

    def test_handler_creation(self, temp_dir):
        """Test SQLiteHandler creation."""
        db_path = Path(temp_dir) / "test_logs.db"
        handler = SQLiteHandler(db_path)

        assert handler.db_path == db_path
        assert db_path.exists()

        # Check that table was created
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_logs'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_emit_log_record(self, temp_dir):
        """Test emitting log records."""
        db_path = Path(temp_dir) / "test_emit.db"
        handler = SQLiteHandler(db_path)

        # Create a test logger
        logger = logging.getLogger("test_logger")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log some messages
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Check logs were written
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pipeline_logs")
        count = cursor.fetchone()[0]
        assert count == 3

        # Check log content
        cursor.execute("SELECT level, message FROM pipeline_logs ORDER BY timestamp")
        logs = cursor.fetchall()
        assert logs[0][0] == "INFO"
        assert logs[0][1] == "Test info message"
        assert logs[1][0] == "WARNING"
        assert logs[2][0] == "ERROR"
        conn.close()

    def test_extra_fields(self, temp_dir):
        """Test logging with extra fields."""
        db_path = Path(temp_dir) / "test_extra.db"
        handler = SQLiteHandler(db_path)

        logger = logging.getLogger("test_extra")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log with extra data
        logger.info(
            "Processing paper",
            extra={"paper_id": "SCREEN_0001", "source": "arxiv", "status": "success"},
        )

        # Check extra data was stored
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM pipeline_logs WHERE message LIKE '%Processing paper%'")
        metadata_json = cursor.fetchone()[0]
        extra_data = json.loads(metadata_json) if metadata_json else {}

        assert extra_data["paper_id"] == "SCREEN_0001"
        assert extra_data["source"] == "arxiv"
        assert extra_data["status"] == "success"
        conn.close()

    def test_exception_logging(self, temp_dir):
        """Test logging exceptions."""
        db_path = Path(temp_dir) / "test_exception.db"
        handler = SQLiteHandler(db_path)

        logger = logging.getLogger("test_exception")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        # Log an exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        # Check exception was logged
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT message, error_trace FROM pipeline_logs")
        message, error_trace = cursor.fetchone()

        assert "An error occurred" in message
        assert error_trace is not None
        assert "ValueError: Test exception" in error_trace
        conn.close()


class TestLoggingDatabase:
    """Test cases for LoggingDatabase class."""

    def test_database_creation(self, temp_dir):
        """Test LoggingDatabase creation."""
        db_path = Path(temp_dir) / "test_db.db"
        # First create the database with SQLiteHandler
        handler = SQLiteHandler(db_path)
        db = LoggingDatabase(db_path)

        assert db.db_path == db_path
        assert db_path.exists()

    def test_query_logs(self, temp_dir):
        """Test querying logs."""
        db_path = Path(temp_dir) / "test_query.db"
        handler = SQLiteHandler(db_path)

        # Add some logs
        logger = logging.getLogger("test_query")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Query logs
        db = LoggingDatabase(db_path)

        # Get all logs
        all_logs = db.query_logs()
        assert len(all_logs) == 4

        # Filter by level
        errors = db.query_logs(level="ERROR")
        assert len(errors) == 1
        assert errors[0]["message"] == "Error message"

        # Filter by multiple levels - the API doesn't support lists, so test each
        warnings = db.query_logs(level="WARNING")
        assert len(warnings) == 1

    def test_query_by_time_range(self, temp_dir):
        """Test querying logs by time range."""
        db_path = Path(temp_dir) / "test_time.db"
        # First create the database with SQLiteHandler
        handler = SQLiteHandler(db_path)
        db = LoggingDatabase(db_path)

        # Manually insert logs with specific timestamps
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # The table is already created by SQLiteHandler, just use it
        # Table name is pipeline_logs

        # Insert logs at different times
        base_time = datetime.now()
        for i in range(5):
            timestamp = base_time.replace(hour=i)
            cursor.execute(
                "INSERT INTO pipeline_logs (timestamp, module, function, level, status, message, error_trace, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp.isoformat(), "test", "test_func", "INFO", "info", f"Message {i}", None, "{}"),
            )

        conn.commit()
        conn.close()

        # Query specific time range
        start_time = base_time.replace(hour=1)
        end_time = base_time.replace(hour=3)

        filtered_logs = db.query_logs(start_time=start_time, end_time=end_time)
        # Should get messages 1, 2 (hour=1 and hour=2)
        assert len(filtered_logs) >= 2

    def test_query_by_logger(self, temp_dir):
        """Test querying logs by logger name."""
        db_path = Path(temp_dir) / "test_logger.db"
        handler = SQLiteHandler(db_path)

        # Create multiple loggers
        logger1 = logging.getLogger("harvester")
        logger1.addHandler(handler)
        logger1.setLevel(logging.INFO)

        logger2 = logging.getLogger("extractor")
        logger2.addHandler(handler)
        logger2.setLevel(logging.INFO)

        logger1.info("Harvester message 1")
        logger1.info("Harvester message 2")
        logger2.info("Extractor message 1")

        # Query by logger (module in the actual implementation)
        db = LoggingDatabase(db_path)

        harvester_logs = db.query_logs(module="harvester")
        assert len(harvester_logs) == 2

        extractor_logs = db.query_logs(module="extractor")
        assert len(extractor_logs) == 1

    def test_get_summary(self, temp_dir):
        """Test getting log summary."""
        db_path = Path(temp_dir) / "test_summary.db"
        handler = SQLiteHandler(db_path)

        logger = logging.getLogger("test_summary")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Add various logs
        for _ in range(5):
            logger.debug("Debug")
        for _ in range(3):
            logger.info("Info")
        for _ in range(2):
            logger.warning("Warning")
        logger.error("Error")

        # Get summary
        db = LoggingDatabase(db_path)
        summary = db.get_summary_statistics()

        assert summary["total_logs"] == 11
        assert summary["by_level"]["DEBUG"] == 5
        assert summary["by_level"]["INFO"] == 3
        assert summary["by_level"]["WARNING"] == 2
        assert summary["by_level"]["ERROR"] == 1
        assert "test_summary" in summary["top_modules"]

    def test_clear_logs(self, temp_dir):
        """Test clearing logs."""
        db_path = Path(temp_dir) / "test_clear.db"
        handler = SQLiteHandler(db_path)

        logger = logging.getLogger("test_clear")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("Message 1")
        logger.info("Message 2")

        db = LoggingDatabase(db_path)

        # Verify logs exist
        logs = db.query_logs()
        assert len(logs) == 2

        # Clear logs (using cleanup_old_logs with 0 days)
        deleted = db.cleanup_old_logs(days_to_keep=0)

        # Verify logs are cleared
        logs = db.query_logs()
        assert len(logs) == 0
        assert deleted == 2


class TestSetupDBLogging:
    """Test cases for setup_db_logging function."""

    def test_setup_basic(self, temp_dir):
        """Test basic logging setup."""
        db_path = Path(temp_dir) / "test_setup.db"

        # Create a mock config object
        class MockConfig:
            def __init__(self, path):
                self.logging_db_path = path
        
        config = MockConfig(db_path)
        
        # Setup logging
        db_handler = setup_db_logging(config)

        assert isinstance(db_handler, SQLiteHandler)
        assert db_path.exists()

        # Test that root logger has the handler
        root_logger = logging.getLogger()
        assert db_handler in root_logger.handlers

    def test_setup_with_logging(self, temp_dir):
        """Test setup and actual logging."""
        db_path = Path(temp_dir) / "test_logging.db"

        # Create a mock config object
        class MockConfig:
            def __init__(self, path):
                self.logging_db_path = path
        
        config = MockConfig(db_path)
        
        # Setup logging
        db_handler = setup_db_logging(config)
        
        # Log a message
        logger = logging.getLogger("test_module")
        logger.setLevel(logging.INFO)
        logger.info("Test message")

        # Check the log was written
        db = LoggingDatabase(db_path)
        logs = db.query_logs()
        
        assert len(logs) >= 1
        assert any("Test message" in log["message"] for log in logs)

    def test_setup_with_formatter(self, temp_dir):
        """Test that formatter is applied."""
        db_path = Path(temp_dir) / "test_formatter.db"

        # Create a mock config object
        class MockConfig:
            def __init__(self, path):
                self.logging_db_path = path
        
        config = MockConfig(db_path)
        
        # Setup logging
        db_handler = setup_db_logging(config)
        
        # The formatter should be set
        assert db_handler.formatter is not None
        
        # Log a message to test formatting
        logger = logging.getLogger("formatter_test")
        logger.setLevel(logging.INFO)
        logger.info("Formatted message")

        # Check that the message was formatted
        db = LoggingDatabase(db_path)
        logs = db.query_logs()
        
        # The message should contain timestamp prefix from formatter
        assert len(logs) >= 1
        # Message should have been formatted with asctime, name, levelname
        assert any("formatter_test" in log["message"] and "INFO" in log["message"] for log in logs)
