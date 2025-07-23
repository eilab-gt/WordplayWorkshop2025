"""Tests for the SQLite logging module."""
import pytest
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
import json

from src.lit_review.utils import SQLiteHandler, LoggingDatabase, setup_db_logging


class TestSQLiteHandler:
    """Test cases for SQLiteHandler class."""
    
    def test_handler_creation(self, temp_dir):
        """Test SQLiteHandler creation."""
        db_path = Path(temp_dir) / 'test_logs.db'
        handler = SQLiteHandler(str(db_path))
        
        assert handler.db_path == str(db_path)
        assert db_path.exists()
        
        # Check that table was created
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_emit_log_record(self, temp_dir):
        """Test emitting log records."""
        db_path = Path(temp_dir) / 'test_emit.db'
        handler = SQLiteHandler(str(db_path))
        
        # Create a test logger
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log some messages
        logger.info('Test info message')
        logger.warning('Test warning message')
        logger.error('Test error message')
        
        # Check logs were written
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM logs")
        count = cursor.fetchone()[0]
        assert count == 3
        
        # Check log content
        cursor.execute("SELECT level, message FROM logs ORDER BY timestamp")
        logs = cursor.fetchall()
        assert logs[0][0] == 'INFO'
        assert logs[0][1] == 'Test info message'
        assert logs[1][0] == 'WARNING'
        assert logs[2][0] == 'ERROR'
        conn.close()
    
    def test_extra_fields(self, temp_dir):
        """Test logging with extra fields."""
        db_path = Path(temp_dir) / 'test_extra.db'
        handler = SQLiteHandler(str(db_path))
        
        logger = logging.getLogger('test_extra')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log with extra data
        logger.info('Processing paper', extra={
            'paper_id': 'SCREEN_0001',
            'source': 'arxiv',
            'status': 'success'
        })
        
        # Check extra data was stored
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT extra FROM logs WHERE message='Processing paper'")
        extra_json = cursor.fetchone()[0]
        extra_data = json.loads(extra_json)
        
        assert extra_data['paper_id'] == 'SCREEN_0001'
        assert extra_data['source'] == 'arxiv'
        assert extra_data['status'] == 'success'
        conn.close()
    
    def test_exception_logging(self, temp_dir):
        """Test logging exceptions."""
        db_path = Path(temp_dir) / 'test_exception.db'
        handler = SQLiteHandler(str(db_path))
        
        logger = logging.getLogger('test_exception')
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
        cursor.execute("SELECT message, extra FROM logs")
        message, extra_json = cursor.fetchone()
        extra_data = json.loads(extra_json)
        
        assert "An error occurred" in message
        assert 'exc_info' in extra_data
        assert 'ValueError: Test exception' in extra_data['exc_info']
        conn.close()


class TestLoggingDatabase:
    """Test cases for LoggingDatabase class."""
    
    def test_database_creation(self, temp_dir):
        """Test LoggingDatabase creation."""
        db_path = Path(temp_dir) / 'test_db.db'
        db = LoggingDatabase(str(db_path))
        
        assert db.db_path == str(db_path)
        assert db_path.exists()
    
    def test_query_logs(self, temp_dir):
        """Test querying logs."""
        db_path = Path(temp_dir) / 'test_query.db'
        handler = SQLiteHandler(str(db_path))
        
        # Add some logs
        logger = logging.getLogger('test_query')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        
        # Query logs
        db = LoggingDatabase(str(db_path))
        
        # Get all logs
        all_logs = db.query_logs()
        assert len(all_logs) == 4
        
        # Filter by level
        errors = db.query_logs(level='ERROR')
        assert len(errors) == 1
        assert errors[0]['message'] == 'Error message'
        
        # Filter by multiple levels
        important = db.query_logs(level=['WARNING', 'ERROR'])
        assert len(important) == 2
    
    def test_query_by_time_range(self, temp_dir):
        """Test querying logs by time range."""
        db_path = Path(temp_dir) / 'test_time.db'
        db = LoggingDatabase(str(db_path))
        
        # Manually insert logs with specific timestamps
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                logger TEXT,
                message TEXT,
                extra TEXT
            )
        ''')
        
        # Insert logs at different times
        base_time = datetime.now()
        for i in range(5):
            timestamp = base_time.replace(hour=i)
            cursor.execute(
                "INSERT INTO logs (timestamp, level, logger, message, extra) VALUES (?, ?, ?, ?, ?)",
                (timestamp.isoformat(), 'INFO', 'test', f'Message {i}', '{}')
            )
        
        conn.commit()
        conn.close()
        
        # Query specific time range
        start_time = base_time.replace(hour=1).isoformat()
        end_time = base_time.replace(hour=3).isoformat()
        
        filtered_logs = db.query_logs(start_time=start_time, end_time=end_time)
        # Should get messages 1, 2 (hour=1 and hour=2)
        assert len(filtered_logs) >= 2
    
    def test_query_by_logger(self, temp_dir):
        """Test querying logs by logger name."""
        db_path = Path(temp_dir) / 'test_logger.db'
        handler = SQLiteHandler(str(db_path))
        
        # Create multiple loggers
        logger1 = logging.getLogger('harvester')
        logger1.addHandler(handler)
        logger1.setLevel(logging.INFO)
        
        logger2 = logging.getLogger('extractor')
        logger2.addHandler(handler)
        logger2.setLevel(logging.INFO)
        
        logger1.info('Harvester message 1')
        logger1.info('Harvester message 2')
        logger2.info('Extractor message 1')
        
        # Query by logger
        db = LoggingDatabase(str(db_path))
        
        harvester_logs = db.query_logs(logger='harvester')
        assert len(harvester_logs) == 2
        
        extractor_logs = db.query_logs(logger='extractor')
        assert len(extractor_logs) == 1
    
    def test_get_summary(self, temp_dir):
        """Test getting log summary."""
        db_path = Path(temp_dir) / 'test_summary.db'
        handler = SQLiteHandler(str(db_path))
        
        logger = logging.getLogger('test_summary')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Add various logs
        for _ in range(5):
            logger.debug('Debug')
        for _ in range(3):
            logger.info('Info')
        for _ in range(2):
            logger.warning('Warning')
        logger.error('Error')
        
        # Get summary
        db = LoggingDatabase(str(db_path))
        summary = db.get_summary()
        
        assert summary['total_logs'] == 11
        assert summary['by_level']['DEBUG'] == 5
        assert summary['by_level']['INFO'] == 3
        assert summary['by_level']['WARNING'] == 2
        assert summary['by_level']['ERROR'] == 1
        assert 'test_summary' in summary['by_logger']
    
    def test_clear_logs(self, temp_dir):
        """Test clearing logs."""
        db_path = Path(temp_dir) / 'test_clear.db'
        handler = SQLiteHandler(str(db_path))
        
        logger = logging.getLogger('test_clear')
        logger.addHandler(handler)
        logger.info('Message 1')
        logger.info('Message 2')
        
        db = LoggingDatabase(str(db_path))
        
        # Verify logs exist
        logs = db.query_logs()
        assert len(logs) == 2
        
        # Clear logs
        db.clear_logs()
        
        # Verify logs are cleared
        logs = db.query_logs()
        assert len(logs) == 0


class TestSetupDBLogging:
    """Test cases for setup_db_logging function."""
    
    def test_setup_basic(self, temp_dir):
        """Test basic logging setup."""
        db_path = Path(temp_dir) / 'test_setup.db'
        
        # Setup logging
        db_handler = setup_db_logging(str(db_path))
        
        assert isinstance(db_handler, SQLiteHandler)
        assert db_path.exists()
        
        # Test that root logger has the handler
        root_logger = logging.getLogger()
        assert db_handler in root_logger.handlers
    
    def test_setup_with_logger_name(self, temp_dir):
        """Test setup with specific logger."""
        db_path = Path(temp_dir) / 'test_named.db'
        
        # Setup for specific logger
        db_handler = setup_db_logging(str(db_path), logger_name='lit_review')
        
        # Check specific logger has handler
        logger = logging.getLogger('lit_review')
        assert db_handler in logger.handlers
        
        # Root logger should not have it
        root_logger = logging.getLogger()
        assert db_handler not in root_logger.handlers
    
    def test_setup_with_level(self, temp_dir):
        """Test setup with specific log level."""
        db_path = Path(temp_dir) / 'test_level.db'
        
        # Setup with WARNING level
        db_handler = setup_db_logging(str(db_path), level=logging.WARNING)
        
        # Test logging at different levels
        logger = logging.getLogger()
        logger.info('Should not be logged')
        logger.warning('Should be logged')
        
        # Check what was logged
        db = LoggingDatabase(str(db_path))
        logs = db.query_logs()
        
        assert len(logs) == 1
        assert logs[0]['level'] == 'WARNING'