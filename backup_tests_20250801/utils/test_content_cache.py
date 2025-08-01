"""Tests for the ContentCache class."""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from src.lit_review.utils.content_cache import ContentCache


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, cache_dir="/tmp/test_cache"):
        self.cache_dir = cache_dir
        self.cache_max_age_days = 90
        self.use_cache = True


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return MockConfig()


@pytest.fixture
def content_cache(mock_config, tmp_path):
    """Create a ContentCache instance with temporary directory."""
    mock_config.cache_dir = str(tmp_path / "cache")
    return ContentCache(mock_config)


def test_init_creates_directories(content_cache, tmp_path):
    """Test that initialization creates necessary directories."""
    cache_dir = tmp_path / "cache"
    assert (cache_dir / "pdfs").exists()
    assert (cache_dir / "html").exists()
    assert (cache_dir / "tex").exists()
    assert (cache_dir / "content_cache.db").exists()


def test_cache_disabled(mock_config, tmp_path):
    """Test behavior when cache is disabled."""
    mock_config.cache_dir = str(tmp_path / "cache")
    mock_config.use_cache = False
    cache = ContentCache(mock_config)

    # Mock fetch function
    fetch_func = Mock(return_value=b"test content")

    # Get content - should always fetch
    path, was_cached = cache.get_or_fetch("test_id", "pdf", fetch_func)

    assert fetch_func.called
    assert not was_cached
    assert path.exists()
    assert path.read_bytes() == b"test content"


def test_get_or_fetch_cache_miss(content_cache):
    """Test get_or_fetch with cache miss."""
    # Mock fetch function
    test_content = b"PDF content here"
    fetch_func = Mock(return_value=test_content)

    # First call - cache miss
    path, was_cached = content_cache.get_or_fetch(
        "test_paper_123", "pdf", fetch_func, source_url="http://example.com/paper.pdf"
    )

    assert fetch_func.called
    assert not was_cached
    assert path is not None
    assert path.exists()
    assert path.read_bytes() == test_content
    assert content_cache.stats["cache_misses"] == 1
    assert content_cache.stats["cache_hits"] == 0


def test_get_or_fetch_cache_hit(content_cache):
    """Test get_or_fetch with cache hit."""
    # First, populate cache
    test_content = b"PDF content here"
    fetch_func = Mock(return_value=test_content)

    path1, was_cached1 = content_cache.get_or_fetch("test_paper_123", "pdf", fetch_func)

    # Reset mock
    fetch_func.reset_mock()

    # Second call - cache hit
    path2, was_cached2 = content_cache.get_or_fetch("test_paper_123", "pdf", fetch_func)

    assert not fetch_func.called  # Should not fetch again
    assert was_cached2
    assert path1 == path2
    assert content_cache.stats["cache_hits"] == 1
    assert content_cache.stats["cache_misses"] == 1


def test_different_content_types(content_cache):
    """Test caching different content types."""
    # Test PDF
    pdf_content = b"PDF content"
    pdf_path, _ = content_cache.get_or_fetch("paper1", "pdf", lambda: pdf_content)
    assert pdf_path.suffix == ".pdf"
    assert pdf_path.parent.name == "pdfs"

    # Test HTML
    html_content = "<html>content</html>"
    html_path, _ = content_cache.get_or_fetch("paper1", "html", lambda: html_content)
    assert html_path.suffix == ".html"
    assert html_path.parent.name == "html"

    # Test TeX
    tex_content = "\\documentclass{article}"
    tex_path, _ = content_cache.get_or_fetch("paper1", "tex", lambda: tex_content)
    assert tex_path.suffix == ".tex"
    assert tex_path.parent.name == "tex"


def test_invalid_content_type(content_cache):
    """Test handling of invalid content type."""
    with pytest.raises(ValueError, match="Unknown content type"):
        content_cache._get_cache_path("paper1", "invalid")


def test_cache_expiry(content_cache):
    """Test that old cache entries are detected as invalid."""
    # Create a cache entry
    test_content = b"Old content"
    path, _ = content_cache.get_or_fetch("old_paper", "pdf", lambda: test_content)

    # Manually modify the database to set old creation time
    import sqlite3

    conn = sqlite3.connect(str(content_cache.cache_db))
    cursor = conn.cursor()

    old_date = datetime.now() - timedelta(days=100)
    cursor.execute(
        """
        UPDATE cache_entries
        SET created_at = ?
        WHERE paper_id = ?
    """,
        (old_date.isoformat(), "old_paper"),
    )
    conn.commit()
    conn.close()

    # Try to get again - should be invalid
    new_content = b"New content"
    fetch_func = Mock(return_value=new_content)

    path2, was_cached = content_cache.get_or_fetch("old_paper", "pdf", fetch_func)

    assert fetch_func.called
    assert not was_cached
    assert path2.read_bytes() == new_content


def test_file_hash_validation(content_cache):
    """Test that modified files are detected."""
    # Create a cache entry
    test_content = b"Original content"
    path, _ = content_cache.get_or_fetch("paper_hash", "pdf", lambda: test_content)

    # Modify the file directly
    path.write_bytes(b"Modified content")

    # Try to get again - should detect invalid hash
    new_content = b"Fresh content"
    fetch_func = Mock(return_value=new_content)

    path2, was_cached = content_cache.get_or_fetch("paper_hash", "pdf", fetch_func)

    assert fetch_func.called
    assert not was_cached
    assert path2.read_bytes() == new_content


def test_get_statistics(content_cache):
    """Test statistics retrieval."""
    # Add some cache entries
    content_cache.get_or_fetch("paper1", "pdf", lambda: b"pdf1")
    content_cache.get_or_fetch("paper2", "pdf", lambda: b"pdf2")
    content_cache.get_or_fetch("paper3", "html", lambda: "html content")
    content_cache.get_or_fetch("paper4", "tex", lambda: "tex content")

    # Get from cache (hits)
    content_cache.get_or_fetch("paper1", "pdf", lambda: b"pdf1")
    content_cache.get_or_fetch("paper3", "html", lambda: "html content")

    stats = content_cache.get_statistics()

    assert stats["total_entries"] == 4
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 4
    assert stats["hit_rate"] == pytest.approx(33.33, 0.1)
    assert "by_type" in stats
    assert stats["by_type"]["pdf"]["count"] == 2
    assert stats["by_type"]["html"]["count"] == 1
    assert stats["by_type"]["tex"]["count"] == 1


def test_cleanup_old_entries(content_cache):
    """Test cleaning up old cache entries."""
    # Create entries
    content_cache.get_or_fetch("paper1", "pdf", lambda: b"content1")
    content_cache.get_or_fetch("paper2", "pdf", lambda: b"content2")

    # Manually set one as old
    import sqlite3

    conn = sqlite3.connect(str(content_cache.cache_db))
    cursor = conn.cursor()

    old_date = datetime.now() - timedelta(days=100)
    cursor.execute(
        """
        UPDATE cache_entries
        SET created_at = ?
        WHERE paper_id = ?
    """,
        (old_date.isoformat(), "paper1"),
    )
    conn.commit()
    conn.close()

    # Clean up entries older than 90 days
    removed = content_cache.cleanup_old_entries(days=90)

    assert removed == 1

    # Verify paper1 is gone but paper2 remains
    stats = content_cache.get_statistics()
    assert stats["total_entries"] == 1


def test_clear_cache_by_type(content_cache):
    """Test clearing cache by content type."""
    # Create entries of different types
    content_cache.get_or_fetch("paper1", "pdf", lambda: b"pdf1")
    content_cache.get_or_fetch("paper2", "pdf", lambda: b"pdf2")
    content_cache.get_or_fetch("paper3", "html", lambda: "html")
    content_cache.get_or_fetch("paper4", "tex", lambda: "tex")

    # Clear only PDFs
    removed = content_cache.clear_cache(content_type="pdf")

    assert removed == 2

    # Verify only PDFs were removed
    stats = content_cache.get_statistics()
    assert stats["total_entries"] == 2
    assert "pdf" not in stats["by_type"]
    assert stats["by_type"]["html"]["count"] == 1
    assert stats["by_type"]["tex"]["count"] == 1


def test_clear_all_cache(content_cache):
    """Test clearing all cache."""
    # Create entries
    content_cache.get_or_fetch("paper1", "pdf", lambda: b"pdf1")
    content_cache.get_or_fetch("paper2", "html", lambda: "html")
    content_cache.get_or_fetch("paper3", "tex", lambda: "tex")

    # Clear all
    removed = content_cache.clear_cache()

    assert removed == 3

    # Verify all are removed
    stats = content_cache.get_statistics()
    assert stats["total_entries"] == 0


def test_paper_id_sanitization(content_cache):
    """Test that paper IDs are properly sanitized for filesystem."""
    # Test with problematic characters
    paper_id = "10.1234/journal.2024.12345"

    path = content_cache._get_cache_path(paper_id, "pdf")

    # Should replace dots and slashes
    assert "/" not in path.stem
    assert "." not in path.stem  # except for extension

    # Should still be able to cache and retrieve
    content = b"test content"
    path, was_cached = content_cache.get_or_fetch(paper_id, "pdf", lambda: content)

    assert path.exists()
    assert not was_cached

    # Second call should hit cache
    path2, was_cached2 = content_cache.get_or_fetch(paper_id, "pdf", lambda: content)

    assert was_cached2
    assert path == path2


def test_concurrent_access(content_cache):
    """Test that concurrent access is handled properly."""
    import threading

    results = []
    fetch_count = 0
    lock = threading.Lock()

    def fetch_func():
        nonlocal fetch_count
        with lock:
            fetch_count += 1
        time.sleep(0.1)  # Simulate slow fetch
        return b"concurrent content"

    def worker():
        path, was_cached = content_cache.get_or_fetch(
            "concurrent_paper", "pdf", fetch_func
        )
        results.append((path, was_cached))

    # Start multiple threads trying to get the same content
    threads = []
    for _ in range(5):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    # Wait for all to complete
    for t in threads:
        t.join()

    # Should have fetched only once
    assert fetch_count == 1

    # All should have gotten the same path
    paths = [r[0] for r in results]
    assert all(p == paths[0] for p in paths)

    # Only first should have was_cached=False
    cached_flags = [r[1] for r in results]
    assert cached_flags.count(False) == 1


def test_fetch_function_returns_none(content_cache):
    """Test handling when fetch function returns None."""
    fetch_func = Mock(return_value=None)

    path, was_cached = content_cache.get_or_fetch("failed_paper", "pdf", fetch_func)

    assert path is None
    assert not was_cached
    assert content_cache.stats["cache_misses"] == 1

    # Should not create any cache entry
    stats = content_cache.get_statistics()
    assert stats["total_entries"] == 0


def test_binary_vs_text_content(content_cache):
    """Test handling of binary vs text content."""
    # Binary content (PDF)
    binary_content = b"\x00\x01\x02\x03PDF content"
    pdf_path, _ = content_cache.get_or_fetch(
        "binary_paper", "pdf", lambda: binary_content
    )
    assert pdf_path.read_bytes() == binary_content

    # Text content (HTML) as string
    text_content = "<html>Text content</html>"
    html_path, _ = content_cache.get_or_fetch(
        "text_paper", "html", lambda: text_content
    )
    assert html_path.read_text() == text_content

    # Text content (HTML) as bytes
    text_bytes = b"<html>Bytes content</html>"
    html_path2, _ = content_cache.get_or_fetch(
        "bytes_paper", "html", lambda: text_bytes
    )
    assert html_path2.read_text() == text_bytes.decode("utf-8")
