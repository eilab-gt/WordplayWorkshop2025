"""Unified content caching system for PDFs, HTML, and LaTeX files."""

import hashlib
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ContentCache:
    """Unified caching system for all content types (PDF, HTML, TeX)."""

    def __init__(self, config):
        """Initialize the content cache.

        Args:
            config: Configuration object with cache settings
        """
        self.config = config
        self.cache_dir = Path(getattr(config, "cache_dir", "./content_cache"))
        self.pdf_dir = self.cache_dir / "pdfs"
        self.html_dir = self.cache_dir / "html"
        self.tex_dir = self.cache_dir / "tex"

        # Cache settings
        self.max_age_days = getattr(config, "cache_max_age_days", 90)
        self.enabled = getattr(config, "use_cache", True)

        # Create all directories
        for directory in [self.pdf_dir, self.html_dir, self.tex_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Cache metadata database
        self.cache_db = self.cache_dir / "content_cache.db"
        self._init_cache_db()

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_saved": 0,
            "time_saved_seconds": 0,
        }

        # Lock for concurrent access
        self._locks = {}
        self._lock_mutex = threading.Lock()

    def _init_cache_db(self):
        """Initialize the cache metadata database."""
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                content_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_size INTEGER,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                UNIQUE(paper_id, content_type)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_content
            ON cache_entries(paper_id, content_type)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created
            ON cache_entries(created_at)
        """
        )

        conn.commit()
        conn.close()

    def get_or_fetch(
        self,
        paper_id: str,
        content_type: str,
        fetch_func: Callable[[], Optional[Any]],
        source_url: Optional[str] = None,
    ) -> tuple[Optional[Path], bool]:
        """Get content from cache or fetch if not present.

        Args:
            paper_id: Unique identifier for the paper
            content_type: Type of content ('pdf', 'html', 'tex')
            fetch_func: Function to fetch content if not cached
            source_url: Optional URL where content was fetched from

        Returns:
            Tuple of (path_to_content, was_cached)
        """
        if not self.enabled:
            # Cache disabled, always fetch
            content = fetch_func()
            if content:
                path = self._get_cache_path(paper_id, content_type)
                self._save_content(path, content, content_type)
                return path, False
            return None, False

        # Get lock for this specific paper+type combination
        lock_key = f"{paper_id}:{content_type}"
        with self._lock_mutex:
            if lock_key not in self._locks:
                self._locks[lock_key] = threading.Lock()
            lock = self._locks[lock_key]

        # Use the lock for the rest of the operation
        with lock:
            # Check cache first
            cache_path = self._get_cache_path(paper_id, content_type)
            cache_entry = self._get_cache_entry(paper_id, content_type)

            if cache_entry and cache_path.exists():
                # Validate cache
                if self._is_cache_valid(cache_entry, cache_path):
                    self._update_access_stats(cache_entry["id"])
                    self.stats["cache_hits"] += 1
                    self.stats["bytes_saved"] += cache_entry["file_size"]
                    self.stats["time_saved_seconds"] += 2  # Estimate 2s per download
                    logger.info(f"Cache hit for {content_type} content: {paper_id}")
                    return cache_path, True
                else:
                    # Invalid cache, remove it
                    logger.warning(f"Invalid cache entry for {paper_id}, removing")
                    self._remove_cache_entry(cache_entry["id"])
                    cache_path.unlink(missing_ok=True)

            # Cache miss - fetch content
            self.stats["cache_misses"] += 1
            logger.info(
                f"Cache miss for {content_type} content: {paper_id}, fetching..."
            )

            start_time = time.time()
            content = fetch_func()
            fetch_time = time.time() - start_time

            if content:
                # Save to cache
                self._save_content(cache_path, content, content_type)
                file_size = cache_path.stat().st_size
                file_hash = self._calculate_hash(cache_path)

                # Update metadata
                self._save_cache_entry(
                    paper_id, content_type, cache_path, file_hash, file_size, source_url
                )

                logger.info(
                    f"Cached {content_type} content for {paper_id} "
                    f"({file_size/1024:.1f} KB in {fetch_time:.1f}s)"
                )
                return cache_path, False

            return None, False

    def _get_cache_path(self, paper_id: str, content_type: str) -> Path:
        """Generate cache file path for given paper and content type."""
        # Clean paper_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper_id)

        if content_type == "pdf":
            return self.pdf_dir / f"{safe_id}.pdf"
        elif content_type == "html":
            return self.html_dir / f"{safe_id}.html"
        elif content_type == "tex":
            return self.tex_dir / f"{safe_id}.tex"
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _save_content(self, path: Path, content: Any, content_type: str):
        """Save content to cache file."""
        if content_type in ["html", "tex"]:
            # Text content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            path.write_text(content, encoding="utf-8")
        else:
            # Binary content (PDF)
            if isinstance(content, str):
                raise ValueError("PDF content must be bytes")
            path.write_bytes(content)

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _get_cache_entry(self, paper_id: str, content_type: str) -> Optional[dict]:
        """Get cache entry from database."""
        conn = sqlite3.connect(str(self.cache_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM cache_entries
            WHERE paper_id = ? AND content_type = ?
        """,
            (paper_id, content_type),
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def _save_cache_entry(
        self,
        paper_id: str,
        content_type: str,
        file_path: Path,
        file_hash: str,
        file_size: int,
        source_url: Optional[str],
    ):
        """Save or update cache entry in database."""
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO cache_entries
            (paper_id, content_type, file_path, file_hash,
             file_size, source_url, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
        """,
            (paper_id, content_type, str(file_path), file_hash, file_size, source_url),
        )

        conn.commit()
        conn.close()

    def _update_access_stats(self, entry_id: int):
        """Update access statistics for cache entry."""
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE cache_entries
            SET accessed_at = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id = ?
        """,
            (entry_id,),
        )

        conn.commit()
        conn.close()

    def _is_cache_valid(self, cache_entry: dict, file_path: Path) -> bool:
        """Validate cache entry."""
        # Check file exists
        if not file_path.exists():
            return False

        # Check age
        created = datetime.fromisoformat(cache_entry["created_at"])
        if datetime.now() - created > timedelta(days=self.max_age_days):
            return False

        # Check file integrity
        current_hash = self._calculate_hash(file_path)
        if current_hash != cache_entry["file_hash"]:
            return False

        return True

    def _remove_cache_entry(self, entry_id: int):
        """Remove cache entry from database."""
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        # Get counts by type
        cursor.execute(
            """
            SELECT content_type, COUNT(*) as count,
                   SUM(file_size) as total_size
            FROM cache_entries
            GROUP BY content_type
        """
        )

        stats = dict(self.stats)
        stats["by_type"] = {}

        for row in cursor.fetchall():
            stats["by_type"][row[0]] = {
                "count": row[1],
                "total_size_mb": row[2] / (1024 * 1024) if row[2] else 0,
            }

        # Get total stats
        cursor.execute(
            """
            SELECT COUNT(*) as total_entries,
                   SUM(file_size) as total_size,
                   SUM(access_count) as total_accesses
            FROM cache_entries
        """
        )

        row = cursor.fetchone()
        stats["total_entries"] = row[0] or 0
        stats["total_size_mb"] = row[1] / (1024 * 1024) if row[1] else 0
        stats["total_accesses"] = row[2] or 0

        # Calculate hit rate
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        stats["hit_rate"] = (
            stats["cache_hits"] / total_requests * 100 if total_requests > 0 else 0
        )

        conn.close()
        return stats

    def cleanup_old_entries(self, days: Optional[int] = None) -> int:
        """Remove cache entries older than specified days.

        Args:
            days: Number of days to keep (default: max_age_days)

        Returns:
            Number of entries removed
        """
        if days is None:
            days = self.max_age_days

        cutoff_date = datetime.now() - timedelta(days=days)

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        # Get entries to remove
        cursor.execute(
            """
            SELECT id, file_path FROM cache_entries
            WHERE created_at < ?
        """,
            (cutoff_date,),
        )

        entries_to_remove = cursor.fetchall()

        # Remove files and database entries
        for entry_id, file_path in entries_to_remove:
            path = Path(file_path)
            if path.exists():
                path.unlink()
            cursor.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))

        conn.commit()
        conn.close()

        logger.info(f"Removed {len(entries_to_remove)} old cache entries")
        return len(entries_to_remove)

    def clear_cache(self, content_type: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            content_type: Optional type to clear (None = all types)

        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()

        if content_type:
            cursor.execute(
                """
                SELECT id, file_path FROM cache_entries
                WHERE content_type = ?
            """,
                (content_type,),
            )
        else:
            cursor.execute("SELECT id, file_path FROM cache_entries")

        entries = cursor.fetchall()

        # Remove files
        for _, file_path in entries:
            path = Path(file_path)
            if path.exists():
                path.unlink()

        # Clear database
        if content_type:
            cursor.execute(
                "DELETE FROM cache_entries WHERE content_type = ?", (content_type,)
            )
        else:
            cursor.execute("DELETE FROM cache_entries")

        conn.commit()
        conn.close()

        logger.info(
            f"Cleared {len(entries)} cache entries"
            f"{f' of type {content_type}' if content_type else ''}"
        )
        return len(entries)
