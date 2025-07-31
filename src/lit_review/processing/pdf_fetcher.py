"""PDF fetcher for downloading papers from various sources."""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..utils.content_cache import ContentCache

logger = logging.getLogger(__name__)


class PDFFetcher:
    """Downloads PDFs from various sources using DOIs and direct URLs."""

    def __init__(self, config):
        """Initialize PDF fetcher.

        Args:
            config: Configuration object
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.email = config.unpaywall_email
        self.timeout = config.pdf_timeout_seconds
        self.max_size_mb = config.pdf_max_size_mb

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize content cache
        self.content_cache = ContentCache(config)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    f"LitReviewPipeline/1.0 (mailto:{self.email})"
                    if self.email
                    else "LitReviewPipeline/1.0"
                )
            }
        )

        # Track statistics
        self.stats = {
            "total_attempted": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "already_cached": 0,
            "unpaywall_success": 0,
            "arxiv_success": 0,
            "direct_success": 0,
        }

    def fetch_pdfs(self, df: pd.DataFrame, parallel: bool = True) -> pd.DataFrame:
        """Fetch PDFs for all papers in DataFrame.

        Args:
            df: DataFrame with paper information
            parallel: Whether to download in parallel

        Returns:
            DataFrame with added pdf_path and pdf_status columns
        """
        logger.info(f"Starting PDF fetch for {len(df)} papers")
        self.stats["total_attempted"] = len(df)

        # Add columns for results
        df["pdf_path"] = ""
        df["pdf_status"] = ""
        df["pdf_hash"] = ""

        df = self._fetch_parallel(df) if parallel else self._fetch_sequential(df)

        self._log_statistics()
        return df

    def _fetch_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch PDFs sequentially.

        Args:
            df: DataFrame with paper information

        Returns:
            Updated DataFrame
        """
        for idx, row in df.iterrows():
            result = self._fetch_single_pdf(row)
            df.at[idx, "pdf_path"] = result["path"]
            df.at[idx, "pdf_status"] = result["status"]
            df.at[idx, "pdf_hash"] = result["hash"]

            # Small delay to be polite
            time.sleep(0.5)

        return df

    def _fetch_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch PDFs in parallel.

        Args:
            df: DataFrame with paper information

        Returns:
            Updated DataFrame
        """
        batch_size = self.config.batch_size_pdf
        max_workers = min(batch_size, self.config.parallel_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_idx = {
                executor.submit(self._fetch_single_pdf, row): idx
                for idx, row in df.iterrows()
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    df.at[idx, "pdf_path"] = result["path"]
                    df.at[idx, "pdf_status"] = result["status"]
                    df.at[idx, "pdf_hash"] = result["hash"]
                except Exception as e:
                    logger.error(f"Error fetching PDF for index {idx}: {e}")
                    df.at[idx, "pdf_status"] = "error"

        return df

    def _fetch_single_pdf(self, paper: pd.Series) -> dict[str, str]:
        """Fetch PDF for a single paper using content cache.

        Args:
            paper: Paper data as pandas Series

        Returns:
            Dictionary with path, status, and hash
        """
        # Generate paper ID for caching
        paper_id = self._generate_paper_id(paper)

        # Try different sources in order
        pdf_url = None
        source = None

        # 1. Try existing PDF URL (e.g., from arXiv)
        if paper.get("pdf_url"):
            pdf_url = paper["pdf_url"]
            source = "direct"

        # 2. Try arXiv if we have an ID
        elif paper.get("arxiv_id") and isinstance(paper.get("arxiv_id"), str):
            pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
            source = "arxiv"

        # 3. Try Unpaywall if we have a DOI
        elif paper.get("doi") and self.email and isinstance(paper.get("doi"), str):
            unpaywall_url = self._get_unpaywall_url(paper["doi"])
            if unpaywall_url:
                pdf_url = unpaywall_url
                source = "unpaywall"

        # 4. Try URL if it looks like a PDF
        elif paper.get("url") and paper["url"].endswith(".pdf"):
            pdf_url = paper["url"]
            source = "direct"

        if not pdf_url:
            self.stats["failed_downloads"] += 1
            return {"path": "", "status": "not_found", "hash": ""}

        # Use content cache to get or fetch the PDF
        def fetch_func():
            """Fetcher function for the cache."""
            content = self._download_pdf_content(pdf_url)
            if content:
                # Verify it's a PDF
                if content[:5] == b"%PDF-":
                    return content
                else:
                    logger.warning(f"Downloaded content is not a PDF for {pdf_url}")
                    return None
            return None

        # Get from cache or fetch
        cache_path, was_cached = self.content_cache.get_or_fetch(
            paper_id, "pdf", fetch_func, source_url=pdf_url
        )

        if cache_path:
            if was_cached:
                self.stats["already_cached"] += 1
                status = "cached"
            else:
                self.stats["successful_downloads"] += 1
                if source == "unpaywall":
                    self.stats["unpaywall_success"] += 1
                elif source == "arxiv":
                    self.stats["arxiv_success"] += 1
                else:
                    self.stats["direct_success"] += 1
                status = f"downloaded_{source}"

            # Calculate hash
            file_hash = self._calculate_file_hash(cache_path)
            return {
                "path": str(cache_path),
                "status": status,
                "hash": file_hash,
            }

        # Failed to download
        self.stats["failed_downloads"] += 1
        return {"path": "", "status": "not_found", "hash": ""}

    def _generate_paper_id(self, paper: pd.Series) -> str:
        """Generate a unique paper ID for caching.

        Args:
            paper: Paper data

        Returns:
            Unique ID string
        """
        # Prefer DOI, then arXiv ID, then generate from title+authors
        if paper.get("doi") and isinstance(paper.get("doi"), str):
            return paper["doi"]
        elif paper.get("arxiv_id") and isinstance(paper.get("arxiv_id"), str):
            return f"arxiv:{paper['arxiv_id']}"
        else:
            # Generate from title and first author
            title = paper.get("title", "Unknown")
            authors = paper.get("authors", "")
            first_author = authors.split(";")[0].strip() if authors else "Unknown"
            year = paper.get("year", "XXXX")
            # Create a simple hash
            content = f"{title}:{first_author}:{year}"
            return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_filename(self, paper: pd.Series) -> str:
        """Generate a filename for the PDF.

        Args:
            paper: Paper data

        Returns:
            Filename string
        """
        # Extract first author's last name
        authors = paper.get("authors", "")
        if authors:
            # Take first author
            first_author = authors.split(";")[0].strip()
            # Extract last name
            parts = first_author.split()
            author_name = parts[-1] if parts else "Unknown"
        else:
            author_name = "Unknown"

        # Clean author name
        author_name = "".join(c for c in author_name if c.isalnum())

        # Get year
        year = paper.get("year", "XXXX")

        # Create base filename
        base_name = f"{author_name}_{year}"

        # Add title slug to make unique
        title = paper.get("title", "")
        if title:
            # Take first few words
            title_words = title.split()[:3]
            title_slug = "_".join(
                "".join(c for c in word if c.isalnum()) for word in title_words
            )
            base_name = f"{base_name}_{title_slug}"

        # Limit length
        base_name = base_name[:100]

        return f"{base_name}.pdf"

    def _get_unpaywall_url(self, doi: str) -> str | None:
        """Get PDF URL from Unpaywall.

        Args:
            doi: Digital Object Identifier

        Returns:
            PDF URL or None
        """
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": self.email}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            # Check for OA location
            best_location = data.get("best_oa_location")
            if best_location and best_location.get("url_for_pdf"):
                return best_location["url_for_pdf"]

            # Check other OA locations
            for location in data.get("oa_locations", []):
                if location.get("url_for_pdf"):
                    return location["url_for_pdf"]

            return None

        except Exception as e:
            logger.debug(f"Unpaywall error for DOI {doi}: {e}")
            return None

    def _download_pdf_content(self, url: str) -> bytes | None:
        """Download PDF content from URL.

        Args:
            url: PDF URL

        Returns:
            PDF content as bytes or None if failed
        """
        try:
            # Stream download to handle large files
            response = self.session.get(
                url, stream=True, timeout=self.timeout, allow_redirects=True
            )

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.warning(f"Non-PDF content type: {content_type} for {url}")
                return None

            # Check file size
            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    logger.warning(f"PDF too large: {size_mb:.1f}MB for {url}")
                    return None

            # Download content
            chunks = []
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)

                    # Check size during download
                    if downloaded > self.max_size_mb * 1024 * 1024:
                        logger.warning(f"Download exceeded size limit for {url}")
                        return None

            content = b"".join(chunks)
            logger.debug(
                f"Successfully downloaded {len(content)/1024:.1f}KB from {url}"
            )
            return content

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error downloading {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")

        return None

    def _download_pdf(self, url: str, filepath: Path) -> bool:
        """Download PDF from URL.

        Args:
            url: PDF URL
            filepath: Path to save file

        Returns:
            True if successful
        """
        try:
            # Stream download to handle large files
            response = self.session.get(
                url, stream=True, timeout=self.timeout, allow_redirects=True
            )

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.warning(f"Non-PDF content type: {content_type} for {url}")
                return False

            # Check file size
            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    logger.warning(f"PDF too large: {size_mb:.1f}MB for {url}")
                    return False

            # Download
            with open(filepath, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Check size during download
                        if downloaded > self.max_size_mb * 1024 * 1024:
                            logger.warning(f"Download exceeded size limit for {url}")
                            filepath.unlink()  # Remove partial file
                            return False

            # Verify it's a PDF
            if not self._verify_pdf(filepath):
                filepath.unlink()
                return False

            logger.debug(f"Successfully downloaded: {filepath.name}")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error downloading {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")

        return False

    def _verify_pdf(self, filepath: Path) -> bool:
        """Verify that a file is a valid PDF.

        Args:
            filepath: Path to file

        Returns:
            True if valid PDF
        """
        try:
            with open(filepath, "rb") as f:
                header = f.read(5)
                return header == b"%PDF-"
        except Exception:
            return False

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file.

        Args:
            filepath: Path to file

        Returns:
            Hash string
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""

    def _log_statistics(self):
        """Log download statistics."""
        logger.info("PDF fetching statistics:")
        logger.info(f"  Total attempted: {self.stats['total_attempted']}")
        logger.info(f"  Already cached: {self.stats['already_cached']}")
        logger.info(f"  Successful downloads: {self.stats['successful_downloads']}")
        logger.info(f"    - Unpaywall: {self.stats['unpaywall_success']}")
        logger.info(f"    - arXiv: {self.stats['arxiv_success']}")
        logger.info(f"    - Direct: {self.stats['direct_success']}")
        logger.info(f"  Failed downloads: {self.stats['failed_downloads']}")

    def cleanup_cache(self, keep_days: int = 30):
        """Clean up old PDFs from cache.

        Args:
            keep_days: Number of days to keep files
        """
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        removed_size = 0

        for filepath in self.cache_dir.glob("*.pdf"):
            if filepath.stat().st_mtime < cutoff_time:
                size = filepath.stat().st_size
                filepath.unlink()
                removed_count += 1
                removed_size += size

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} PDFs ({removed_size / 1024 / 1024:.1f}MB)"
            )

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get statistics about the PDF cache.

        Returns:
            Dictionary with cache statistics
        """
        pdf_files = list(self.cache_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(pdf_files),
            "total_size_mb": total_size / 1024 / 1024,
            "average_size_mb": (
                (total_size / len(pdf_files) / 1024 / 1024) if pdf_files else 0
            ),
        }
