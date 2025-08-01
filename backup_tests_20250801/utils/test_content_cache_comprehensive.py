"""Comprehensive tests for the ContentCache module to improve coverage."""

import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from src.lit_review.utils.content_cache import ContentCache


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration object."""
    config = Mock()
    config.cache_dir = tmp_path / "test_cache"
    config.cache_max_age_days = 90
    config.use_cache = True
    return config


@pytest.fixture
def content_cache(mock_config):
    """Create ContentCache instance for testing."""
    return ContentCache(mock_config)


class TestContentCache:
    """Test cases for the ContentCache class."""

    def test_init(self, mock_config):
        """Test ContentCache initialization."""
        cache = ContentCache(mock_config)

        assert cache.config == mock_config
        assert cache.cache_dir == mock_config.cache_dir
        assert cache.pdf_dir == cache.cache_dir / "pdfs"
        assert cache.html_dir == cache.cache_dir / "html"
        assert cache.tex_dir == cache.cache_dir / "tex"
        assert cache.max_age_days == 90
        assert cache.enabled is True
        assert cache.cache_db == cache.cache_dir / "content_cache.db"

        # Check directories were created
        assert cache.pdf_dir.exists()
        assert cache.html_dir.exists()
        assert cache.tex_dir.exists()

        # Check database was initialized
        assert cache.cache_db.exists()

    def test_init_cache_db(self, mock_config):
        """Test cache database initialization."""
        cache = ContentCache(mock_config)

        # Check tables were created
        conn = sqlite3.connect(str(cache.cache_db))
        cursor = conn.cursor()

        # Check cache_entries table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cache_entries'"
        )
        assert cursor.fetchone() is not None

        # Check indexes
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_paper_content'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_created'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_get_or_fetch_cache_disabled(self, mock_config):
        """Test get_or_fetch when cache is disabled."""
        mock_config.use_cache = False
        cache = ContentCache(mock_config)

        # Mock fetch function
        pdf_content = b"%PDF-1.4\nTest PDF content"
        fetch_func = Mock(return_value=pdf_content)

        path, was_cached = cache.get_or_fetch("test_paper", "pdf", fetch_func)

        assert path is not None
        assert was_cached is False
        assert fetch_func.called
        assert path.read_bytes() == pdf_content

    def test_get_or_fetch_cache_miss(self, content_cache):
        """Test get_or_fetch with cache miss."""
        # Mock fetch function
        pdf_content = b"%PDF-1.4\nTest PDF content"
        fetch_func = Mock(return_value=pdf_content)

        path, was_cached = content_cache.get_or_fetch(
            "test_paper_123", "pdf", fetch_func, "http://example.com/paper.pdf"
        )

        assert path is not None
        assert was_cached is False
        assert fetch_func.called
        assert path.exists()
        assert path.read_bytes() == pdf_content
        assert content_cache.stats["cache_misses"] == 1
        assert content_cache.stats["cache_hits"] == 0

    def test_get_or_fetch_cache_hit(self, content_cache):
        """Test get_or_fetch with cache hit."""
        # First, populate cache
        pdf_content = b"%PDF-1.4\nTest PDF content"
        fetch_func = Mock(return_value=pdf_content)

        # First call - cache miss
        path1, was_cached1 = content_cache.get_or_fetch("test_paper", "pdf", fetch_func)

        # Reset fetch function
        fetch_func.reset_mock()

        # Second call - cache hit
        path2, was_cached2 = content_cache.get_or_fetch("test_paper", "pdf", fetch_func)

        assert path2 == path1
        assert was_cached2 is True
        assert not fetch_func.called  # Should not fetch again
        assert content_cache.stats["cache_hits"] == 1
        assert content_cache.stats["cache_misses"] == 1

    def test_get_or_fetch_invalid_cache(self, content_cache):
        """Test get_or_fetch with invalid cache entry."""
        # Create a cache entry manually
        paper_id = "test_paper_invalid"
        cache_path = content_cache._get_cache_path(paper_id, "pdf")
        cache_path.write_bytes(b"corrupted content")

        # Save cache entry with wrong hash
        content_cache._save_cache_entry(
            paper_id, "pdf", cache_path, "wrong_hash", 100, "http://example.com"
        )

        # Mock fetch function
        pdf_content = b"%PDF-1.4\nCorrect content"
        fetch_func = Mock(return_value=pdf_content)

        path, was_cached = content_cache.get_or_fetch(paper_id, "pdf", fetch_func)

        assert path is not None
        assert was_cached is False
        assert fetch_func.called
        assert path.read_bytes() == pdf_content

    def test_get_or_fetch_concurrent_access(self, content_cache):
        """Test concurrent access to same paper."""
        results = []
        fetch_count = [0]

        def slow_fetch():
            time.sleep(0.1)  # Simulate slow download
            fetch_count[0] += 1
            return b"%PDF-1.4\nTest content"

        def worker():
            path, was_cached = content_cache.get_or_fetch(
                "concurrent_paper", "pdf", slow_fetch
            )
            results.append((path, was_cached))

        # Start multiple threads trying to fetch same paper
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one fetch should have occurred
        assert fetch_count[0] == 1
        assert len(results) == 5

        # All should get same path
        paths = [r[0] for r in results]
        assert all(p == paths[0] for p in paths)

        # Only one should have fetched
        was_cached_count = sum(1 for _, cached in results if not cached)
        assert was_cached_count == 1

    def test_get_cache_path(self, content_cache):
        """Test cache path generation."""
        # Test PDF path
        pdf_path = content_cache._get_cache_path("paper123", "pdf")
        assert pdf_path == content_cache.pdf_dir / "paper123.pdf"

        # Test HTML path
        html_path = content_cache._get_cache_path("paper123", "html")
        assert html_path == content_cache.html_dir / "paper123.html"

        # Test TeX path
        tex_path = content_cache._get_cache_path("paper123", "tex")
        assert tex_path == content_cache.tex_dir / "paper123.tex"

        # Test sanitization of paper ID
        safe_path = content_cache._get_cache_path("paper/with:special*chars", "pdf")
        assert safe_path.name == "paper_with_special_chars.pdf"

        # Test invalid content type
        with pytest.raises(ValueError):
            content_cache._get_cache_path("paper123", "invalid")

    def test_save_content(self, content_cache, tmp_path):
        """Test content saving."""
        # Test PDF (binary)
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"%PDF-1.4\nBinary content"
        content_cache._save_content(pdf_path, pdf_content, "pdf")
        assert pdf_path.read_bytes() == pdf_content

        # Test HTML (text)
        html_path = tmp_path / "test.html"
        html_content = "<html><body>Test</body></html>"
        content_cache._save_content(html_path, html_content, "html")
        assert html_path.read_text() == html_content

        # Test HTML from bytes
        html_bytes = b"<html><body>Test bytes</body></html>"
        html_path2 = tmp_path / "test2.html"
        content_cache._save_content(html_path2, html_bytes, "html")
        assert html_path2.read_text() == html_bytes.decode("utf-8")

        # Test TeX (text)
        tex_path = tmp_path / "test.tex"
        tex_content = (
            "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
        )
        content_cache._save_content(tex_path, tex_content, "tex")
        assert tex_path.read_text() == tex_content

        # Test invalid PDF content (string instead of bytes)
        with pytest.raises(ValueError):
            content_cache._save_content(tmp_path / "bad.pdf", "string content", "pdf")

    def test_calculate_hash(self, content_cache, tmp_path):
        """Test file hash calculation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = b"Test content for hashing"
        test_file.write_bytes(test_content)

        # Calculate hash
        file_hash = content_cache._calculate_hash(test_file)

        # Verify hash
        expected_hash = hashlib.md5(test_content).hexdigest()
        assert file_hash == expected_hash

    def test_cache_entry_operations(self, content_cache, tmp_path):
        """Test cache entry database operations."""
        paper_id = "test_paper_db"
        content_type = "pdf"
        file_path = tmp_path / "test.pdf"
        file_path.write_bytes(b"test content")
        file_hash = "test_hash"
        file_size = 1024
        source_url = "http://example.com/paper.pdf"

        # Save entry
        content_cache._save_cache_entry(
            paper_id, content_type, file_path, file_hash, file_size, source_url
        )

        # Get entry
        entry = content_cache._get_cache_entry(paper_id, content_type)
        assert entry is not None
        assert entry["paper_id"] == paper_id
        assert entry["content_type"] == content_type
        assert entry["file_hash"] == file_hash
        assert entry["file_size"] == file_size
        assert entry["source_url"] == source_url
        assert entry["access_count"] == 1

        # Update access stats
        content_cache._update_access_stats(entry["id"])

        # Get updated entry
        updated_entry = content_cache._get_cache_entry(paper_id, content_type)
        assert updated_entry["access_count"] == 2

        # Remove entry
        content_cache._remove_cache_entry(entry["id"])

        # Verify removed
        removed_entry = content_cache._get_cache_entry(paper_id, content_type)
        assert removed_entry is None

    def test_is_cache_valid(self, content_cache, tmp_path):
        """Test cache validation."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_content = b"test content"
        test_file.write_bytes(test_content)
        file_hash = content_cache._calculate_hash(test_file)

        # Test valid cache
        valid_entry = {"created_at": datetime.now().isoformat(), "file_hash": file_hash}
        assert content_cache._is_cache_valid(valid_entry, test_file) is True

        # Test missing file
        missing_file = tmp_path / "missing.pdf"
        assert content_cache._is_cache_valid(valid_entry, missing_file) is False

        # Test old cache
        old_entry = {
            "created_at": (datetime.now() - timedelta(days=100)).isoformat(),
            "file_hash": file_hash,
        }
        assert content_cache._is_cache_valid(old_entry, test_file) is False

        # Test wrong hash
        wrong_hash_entry = {
            "created_at": datetime.now().isoformat(),
            "file_hash": "wrong_hash",
        }
        assert content_cache._is_cache_valid(wrong_hash_entry, test_file) is False

    def test_get_statistics(self, content_cache):
        """Test statistics retrieval."""
        # Add some cache entries
        for i in range(3):
            pdf_content = f"PDF content {i}".encode()
            content_cache.get_or_fetch(f"paper_{i}", "pdf", lambda c=pdf_content: c)

        for i in range(2):
            html_content = f"<html>Content {i}</html>"
            content_cache.get_or_fetch(f"paper_{i}", "html", lambda c=html_content: c)

        # Get statistics
        stats = content_cache.get_statistics()

        assert stats["total_entries"] == 5
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 5
        assert stats["hit_rate"] == 0.0
        assert "by_type" in stats
        assert stats["by_type"]["pdf"]["count"] == 3
        assert stats["by_type"]["html"]["count"] == 2

    def test_cleanup_old_entries(self, content_cache):
        """Test cleanup of old cache entries."""
        # Create old entry
        paper_id = "old_paper"
        cache_path = content_cache._get_cache_path(paper_id, "pdf")
        cache_path.write_bytes(b"old content")

        # Manually create old cache entry
        conn = sqlite3.connect(str(content_cache.cache_db))
        cursor = conn.cursor()
        old_date = datetime.now() - timedelta(days=100)
        cursor.execute(
            """
            INSERT INTO cache_entries
            (paper_id, content_type, file_path, file_hash, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (paper_id, "pdf", str(cache_path), "hash", 100, old_date),
        )
        conn.commit()
        conn.close()

        # Create recent entry
        recent_content = b"recent content"
        content_cache.get_or_fetch("recent_paper", "pdf", lambda: recent_content)

        # Clean up old entries
        removed_count = content_cache.cleanup_old_entries(days=50)

        assert removed_count == 1
        assert not cache_path.exists()

        # Verify recent entry still exists
        recent_entry = content_cache._get_cache_entry("recent_paper", "pdf")
        assert recent_entry is not None

    def test_clear_cache(self, content_cache):
        """Test cache clearing."""
        # Add entries of different types
        content_cache.get_or_fetch("paper1", "pdf", lambda: b"pdf content")
        content_cache.get_or_fetch("paper2", "html", lambda: "html content")
        content_cache.get_or_fetch("paper3", "tex", lambda: "tex content")

        # Clear specific type
        removed_count = content_cache.clear_cache("pdf")
        assert removed_count == 1

        # Verify PDF is gone but others remain
        assert content_cache._get_cache_entry("paper1", "pdf") is None
        assert content_cache._get_cache_entry("paper2", "html") is not None
        assert content_cache._get_cache_entry("paper3", "tex") is not None

        # Clear all
        removed_count = content_cache.clear_cache()
        assert removed_count == 2

        # Verify all gone
        assert content_cache._get_cache_entry("paper2", "html") is None
        assert content_cache._get_cache_entry("paper3", "tex") is None

    def test_fetch_returns_none(self, content_cache):
        """Test handling when fetch function returns None."""
        fetch_func = Mock(return_value=None)

        path, was_cached = content_cache.get_or_fetch("no_content", "pdf", fetch_func)

        assert path is None
        assert was_cached is False
        assert fetch_func.called
        assert content_cache.stats["cache_misses"] == 1

    def test_different_content_types(self, content_cache):
        """Test caching different content types for same paper."""
        paper_id = "multi_format_paper"

        # Cache PDF
        pdf_path, _ = content_cache.get_or_fetch(
            paper_id, "pdf", lambda: b"%PDF-1.4\nPDF content"
        )

        # Cache HTML
        html_path, _ = content_cache.get_or_fetch(
            paper_id, "html", lambda: "<html>HTML content</html>"
        )

        # Cache TeX
        tex_path, _ = content_cache.get_or_fetch(
            paper_id, "tex", lambda: "\\documentclass{article}"
        )

        # All should be different files
        assert pdf_path != html_path != tex_path
        assert pdf_path.suffix == ".pdf"
        assert html_path.suffix == ".html"
        assert tex_path.suffix == ".tex"

        # All should exist in database
        assert content_cache._get_cache_entry(paper_id, "pdf") is not None
        assert content_cache._get_cache_entry(paper_id, "html") is not None
        assert content_cache._get_cache_entry(paper_id, "tex") is not None
