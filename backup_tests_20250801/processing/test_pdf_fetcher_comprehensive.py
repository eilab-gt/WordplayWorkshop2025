"""Comprehensive tests for the PDFFetcher module to improve coverage."""

import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.lit_review.processing.pdf_fetcher import PDFFetcher


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.cache_dir = Path("test_cache")
    config.unpaywall_email = "test@example.com"
    config.pdf_timeout_seconds = 30
    config.pdf_max_size_mb = 50
    config.batch_size_pdf = 10
    config.parallel_workers = 3
    return config


@pytest.fixture
def sample_papers_df():
    """Create sample paper DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Deep Learning for Natural Language Processing",
                "Machine Learning Applications in Healthcare",
                "AI in Education: A Systematic Review",
                "Neural Networks for Image Recognition",
                "Transformers in NLP: BERT and Beyond",
            ],
            "authors": [
                "Smith, John; Doe, Jane",
                "Johnson, Alice; Brown, Bob",
                "Williams, Carol",
                "Davis, David; Miller, Emily",
                "Anderson, Frank; Taylor, Grace",
            ],
            "year": [2023, 2024, 2023, 2022, 2024],
            "doi": [
                "10.1234/test1",
                "10.1234/test2",
                "",
                "10.1234/test4",
                "10.1234/test5",
            ],
            "arxiv_id": ["2301.00001", "", "", "", "2401.00001"],
            "pdf_url": [
                "https://arxiv.org/pdf/2301.00001.pdf",
                "",
                "",
                "",
                "https://arxiv.org/pdf/2401.00001.pdf",
            ],
            "url": [
                "https://arxiv.org/abs/2301.00001",
                "https://doi.org/10.1234/test2",
                "https://semanticscholar.org/paper/12345",
                "https://example.com/paper.pdf",
                "https://arxiv.org/abs/2401.00001",
            ],
        }
    )


@pytest.fixture
def mock_content_cache():
    """Create a mock ContentCache object."""
    cache = Mock()
    cache.get_or_fetch = Mock()
    return cache


class TestPDFFetcher:
    """Test cases for the PDFFetcher class."""

    def test_init(self, mock_config, tmp_path):
        """Test PDFFetcher initialization."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = Mock()

            fetcher = PDFFetcher(mock_config)

            assert fetcher.config == mock_config
            assert fetcher.cache_dir == tmp_path
            assert fetcher.email == "test@example.com"
            assert fetcher.timeout == 30
            assert fetcher.max_size_mb == 50
            assert tmp_path.exists()
            assert fetcher.stats["total_attempted"] == 0
            assert "User-Agent" in fetcher.session.headers

    def test_fetch_pdfs_sequential(
        self, mock_config, sample_papers_df, mock_content_cache, tmp_path
    ):
        """Test sequential PDF fetching."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            # Mock _fetch_single_pdf
            with patch.object(fetcher, "_fetch_single_pdf") as mock_fetch:
                mock_fetch.side_effect = [
                    {
                        "path": "/cache/paper1.pdf",
                        "status": "downloaded_direct",
                        "hash": "hash1",
                    },
                    {"path": "/cache/paper2.pdf", "status": "cached", "hash": "hash2"},
                    {"path": "", "status": "not_found", "hash": ""},
                    {
                        "path": "/cache/paper4.pdf",
                        "status": "downloaded_unpaywall",
                        "hash": "hash4",
                    },
                    {
                        "path": "/cache/paper5.pdf",
                        "status": "downloaded_arxiv",
                        "hash": "hash5",
                    },
                ]

                with patch("time.sleep"):  # Skip delays
                    result_df = fetcher.fetch_pdfs(
                        sample_papers_df.copy(), parallel=False
                    )

                assert len(result_df) == 5
                assert "pdf_path" in result_df.columns
                assert "pdf_status" in result_df.columns
                assert "pdf_hash" in result_df.columns
                assert result_df.iloc[0]["pdf_path"] == "/cache/paper1.pdf"
                assert result_df.iloc[2]["pdf_status"] == "not_found"
                assert mock_fetch.call_count == 5

    def test_fetch_pdfs_parallel(
        self, mock_config, sample_papers_df, mock_content_cache, tmp_path
    ):
        """Test parallel PDF fetching."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            # Mock _fetch_single_pdf
            def fetch_side_effect(paper):
                idx = paper.name  # Get the index from Series
                results = [
                    {
                        "path": f"/cache/paper{idx}.pdf",
                        "status": "downloaded_direct",
                        "hash": f"hash{idx}",
                    },
                    {
                        "path": f"/cache/paper{idx}.pdf",
                        "status": "cached",
                        "hash": f"hash{idx}",
                    },
                    {"path": "", "status": "not_found", "hash": ""},
                    {
                        "path": f"/cache/paper{idx}.pdf",
                        "status": "downloaded_unpaywall",
                        "hash": f"hash{idx}",
                    },
                    {
                        "path": f"/cache/paper{idx}.pdf",
                        "status": "downloaded_arxiv",
                        "hash": f"hash{idx}",
                    },
                ]
                return results[idx % 5]

            with patch.object(
                fetcher, "_fetch_single_pdf", side_effect=fetch_side_effect
            ):
                result_df = fetcher.fetch_pdfs(sample_papers_df.copy(), parallel=True)

                assert len(result_df) == 5
                assert "pdf_path" in result_df.columns
                assert "pdf_status" in result_df.columns
                assert "pdf_hash" in result_df.columns

    def test_fetch_single_pdf_direct_url(
        self, mock_config, mock_content_cache, tmp_path
    ):
        """Test fetching PDF from direct URL."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Test Paper",
                    "pdf_url": "https://example.com/paper.pdf",
                    "doi": "10.1234/test",
                    "arxiv_id": "",
                    "url": "https://example.com/paper.html",
                }
            )

            # Mock content cache to return successful download
            mock_path = tmp_path / "cached_paper.pdf"
            mock_content_cache.get_or_fetch.return_value = (mock_path, False)

            with patch.object(fetcher, "_calculate_file_hash") as mock_hash:
                mock_hash.return_value = "test_hash_123"

                result = fetcher._fetch_single_pdf(paper)

                assert result["path"] == str(mock_path)
                assert result["status"] == "downloaded_direct"
                assert result["hash"] == "test_hash_123"
                assert fetcher.stats["successful_downloads"] == 1
                assert fetcher.stats["direct_success"] == 1

    def test_fetch_single_pdf_arxiv(self, mock_config, mock_content_cache, tmp_path):
        """Test fetching PDF from arXiv."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Test Paper",
                    "pdf_url": "",
                    "doi": "",
                    "arxiv_id": "2301.00001",
                    "url": "https://arxiv.org/abs/2301.00001",
                }
            )

            mock_path = tmp_path / "arxiv_paper.pdf"
            mock_content_cache.get_or_fetch.return_value = (mock_path, False)

            with patch.object(fetcher, "_calculate_file_hash") as mock_hash:
                mock_hash.return_value = "arxiv_hash_123"

                result = fetcher._fetch_single_pdf(paper)

                # Check that arXiv URL was constructed correctly
                call_args = mock_content_cache.get_or_fetch.call_args
                assert (
                    call_args[1]["source_url"] == "https://arxiv.org/pdf/2301.00001.pdf"
                )

                assert result["path"] == str(mock_path)
                assert result["status"] == "downloaded_arxiv"
                assert result["hash"] == "arxiv_hash_123"
                assert fetcher.stats["arxiv_success"] == 1

    def test_fetch_single_pdf_unpaywall(
        self, mock_config, mock_content_cache, tmp_path
    ):
        """Test fetching PDF from Unpaywall."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Test Paper",
                    "pdf_url": "",
                    "doi": "10.1234/test",
                    "arxiv_id": "",
                    "url": "https://doi.org/10.1234/test",
                }
            )

            with patch.object(fetcher, "_get_unpaywall_url") as mock_unpaywall:
                mock_unpaywall.return_value = "https://unpaywall.org/paper.pdf"

                mock_path = tmp_path / "unpaywall_paper.pdf"
                mock_content_cache.get_or_fetch.return_value = (mock_path, False)

                with patch.object(fetcher, "_calculate_file_hash") as mock_hash:
                    mock_hash.return_value = "unpaywall_hash_123"

                    result = fetcher._fetch_single_pdf(paper)

                    assert result["path"] == str(mock_path)
                    assert result["status"] == "downloaded_unpaywall"
                    assert result["hash"] == "unpaywall_hash_123"
                    assert fetcher.stats["unpaywall_success"] == 1

    def test_fetch_single_pdf_cached(self, mock_config, mock_content_cache, tmp_path):
        """Test fetching already cached PDF."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Test Paper",
                    "pdf_url": "https://example.com/paper.pdf",
                    "doi": "10.1234/test",
                }
            )

            # Mock content cache to indicate file was already cached
            mock_path = tmp_path / "cached_paper.pdf"
            mock_content_cache.get_or_fetch.return_value = (mock_path, True)

            with patch.object(fetcher, "_calculate_file_hash") as mock_hash:
                mock_hash.return_value = "cached_hash_123"

                result = fetcher._fetch_single_pdf(paper)

                assert result["path"] == str(mock_path)
                assert result["status"] == "cached"
                assert result["hash"] == "cached_hash_123"
                assert fetcher.stats["already_cached"] == 1
                assert fetcher.stats["successful_downloads"] == 0

    def test_fetch_single_pdf_not_found(
        self, mock_config, mock_content_cache, tmp_path
    ):
        """Test handling when PDF cannot be found."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Test Paper",
                    "pdf_url": "",
                    "doi": "",
                    "arxiv_id": "",
                    "url": "https://example.com/paper.html",
                }
            )

            result = fetcher._fetch_single_pdf(paper)

            assert result["path"] == ""
            assert result["status"] == "not_found"
            assert result["hash"] == ""
            assert fetcher.stats["failed_downloads"] == 1

    def test_generate_paper_id_with_doi(self, mock_config, tmp_path):
        """Test paper ID generation with DOI."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "doi": "10.1234/test",
                    "arxiv_id": "2301.00001",
                    "title": "Test Paper",
                    "authors": "Smith, John",
                    "year": 2023,
                }
            )

            paper_id = fetcher._generate_paper_id(paper)
            assert paper_id == "10.1234/test"

    def test_generate_paper_id_with_arxiv(self, mock_config, tmp_path):
        """Test paper ID generation with arXiv ID."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "doi": "",
                    "arxiv_id": "2301.00001",
                    "title": "Test Paper",
                    "authors": "Smith, John",
                    "year": 2023,
                }
            )

            paper_id = fetcher._generate_paper_id(paper)
            assert paper_id == "arxiv:2301.00001"

    def test_generate_paper_id_fallback(self, mock_config, tmp_path):
        """Test paper ID generation fallback to title/author hash."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "doi": "",
                    "arxiv_id": "",
                    "title": "Test Paper Title",
                    "authors": "Smith, John; Doe, Jane",
                    "year": 2023,
                }
            )

            paper_id = fetcher._generate_paper_id(paper)
            # Should be a 16-character hash
            assert len(paper_id) == 16
            assert paper_id.isalnum()

    def test_generate_filename(self, mock_config, tmp_path):
        """Test filename generation."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            paper = pd.Series(
                {
                    "title": "Deep Learning for Natural Language Processing",
                    "authors": "Smith, John; Doe, Jane",
                    "year": 2023,
                }
            )

            filename = fetcher._generate_filename(paper)
            assert filename == "John_2023_Deep_Learning_for.pdf"

    def test_get_unpaywall_url_success(self, mock_config, tmp_path):
        """Test successful Unpaywall URL retrieval."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}
            }

            with patch.object(fetcher.session, "get", return_value=mock_response):
                url = fetcher._get_unpaywall_url("10.1234/test")

                assert url == "https://example.com/paper.pdf"

    def test_get_unpaywall_url_not_found(self, mock_config, tmp_path):
        """Test Unpaywall URL when paper not found."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            mock_response = Mock()
            mock_response.status_code = 404

            with patch.object(fetcher.session, "get", return_value=mock_response):
                url = fetcher._get_unpaywall_url("10.1234/test")

                assert url is None

    def test_download_pdf_content_success(self, mock_config, tmp_path):
        """Test successful PDF content download."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Content-Type": "application/pdf",
                "Content-Length": "1000000",  # 1MB
            }
            mock_response.iter_content.return_value = [b"%PDF-", b"content"]

            with patch.object(fetcher.session, "get", return_value=mock_response):
                content = fetcher._download_pdf_content("https://example.com/paper.pdf")

                assert content == b"%PDF-content"

    def test_download_pdf_content_too_large(self, mock_config, tmp_path):
        """Test handling of PDFs that are too large."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Content-Type": "application/pdf",
                "Content-Length": str(100 * 1024 * 1024),  # 100MB
            }

            with patch.object(fetcher.session, "get", return_value=mock_response):
                content = fetcher._download_pdf_content("https://example.com/paper.pdf")

                assert content is None

    def test_download_pdf_content_wrong_type(self, mock_config, tmp_path):
        """Test handling of non-PDF content types."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "Content-Type": "text/html",
                "Content-Length": "10000",
            }

            with patch.object(fetcher.session, "get", return_value=mock_response):
                content = fetcher._download_pdf_content(
                    "https://example.com/paper.html"
                )

                assert content is None

    def test_verify_pdf_valid(self, mock_config, tmp_path):
        """Test PDF verification with valid PDF."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Create a mock PDF file
            pdf_file = tmp_path / "test.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n%rest of PDF content")

            assert fetcher._verify_pdf(pdf_file) is True

    def test_verify_pdf_invalid(self, mock_config, tmp_path):
        """Test PDF verification with invalid file."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Create a non-PDF file
            non_pdf_file = tmp_path / "test.txt"
            non_pdf_file.write_text("Not a PDF file")

            assert fetcher._verify_pdf(non_pdf_file) is False

    def test_calculate_file_hash(self, mock_config, tmp_path):
        """Test file hash calculation."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Create a test file
            test_file = tmp_path / "test.pdf"
            test_content = b"Test PDF content"
            test_file.write_bytes(test_content)

            file_hash = fetcher._calculate_file_hash(test_file)

            # Calculate expected hash
            expected_hash = hashlib.sha256(test_content).hexdigest()

            assert file_hash == expected_hash

    def test_calculate_file_hash_error(self, mock_config, tmp_path):
        """Test file hash calculation with error."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Try to hash non-existent file
            non_existent = tmp_path / "non_existent.pdf"

            file_hash = fetcher._calculate_file_hash(non_existent)

            assert file_hash == ""

    def test_cleanup_cache(self, mock_config, tmp_path):
        """Test cache cleanup functionality."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Create some test PDFs with different ages
            old_pdf = tmp_path / "old.pdf"
            old_pdf.write_bytes(b"%PDF-old")
            # Set modification time to 40 days ago
            old_time = time.time() - (40 * 24 * 60 * 60)
            import os

            os.utime(old_pdf, (old_time, old_time))

            new_pdf = tmp_path / "new.pdf"
            new_pdf.write_bytes(b"%PDF-new")

            # Clean up cache with 30 day limit
            fetcher.cleanup_cache(keep_days=30)

            # Old file should be removed, new file should remain
            assert not old_pdf.exists()
            assert new_pdf.exists()

    def test_get_cache_statistics(self, mock_config, tmp_path):
        """Test cache statistics retrieval."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Create some test PDFs
            pdf1 = tmp_path / "paper1.pdf"
            pdf1.write_bytes(b"%PDF-" + b"A" * 1000)

            pdf2 = tmp_path / "paper2.pdf"
            pdf2.write_bytes(b"%PDF-" + b"B" * 2000)

            stats = fetcher.get_cache_statistics()

            assert stats["cache_dir"] == str(tmp_path)
            assert stats["total_files"] == 2
            assert stats["total_size_mb"] > 0
            assert stats["average_size_mb"] > 0

    def test_parallel_error_handling(
        self, mock_config, sample_papers_df, mock_content_cache, tmp_path
    ):
        """Test error handling in parallel fetching."""
        mock_config.cache_dir = tmp_path

        with patch(
            "src.lit_review.processing.pdf_fetcher.ContentCache"
        ) as mock_cache_cls:
            mock_cache_cls.return_value = mock_content_cache

            fetcher = PDFFetcher(mock_config)

            # Mock _fetch_single_pdf to raise exception for one paper
            def fetch_side_effect(paper):
                if paper.name == 2:  # Third paper
                    raise Exception("Test error")
                return {
                    "path": f"/cache/paper{paper.name}.pdf",
                    "status": "downloaded",
                    "hash": f"hash{paper.name}",
                }

            with patch.object(
                fetcher, "_fetch_single_pdf", side_effect=fetch_side_effect
            ):
                result_df = fetcher.fetch_pdfs(sample_papers_df.copy(), parallel=True)

                assert len(result_df) == 5
                assert result_df.iloc[2]["pdf_status"] == "error"
                assert result_df.iloc[2]["pdf_path"] == ""

    def test_log_statistics(self, mock_config, tmp_path, caplog):
        """Test statistics logging."""
        mock_config.cache_dir = tmp_path

        with patch("src.lit_review.processing.pdf_fetcher.ContentCache"):
            fetcher = PDFFetcher(mock_config)

            # Set some statistics
            fetcher.stats = {
                "total_attempted": 100,
                "already_cached": 20,
                "successful_downloads": 60,
                "unpaywall_success": 10,
                "arxiv_success": 30,
                "direct_success": 20,
                "failed_downloads": 20,
            }

            with caplog.at_level("INFO"):
                fetcher._log_statistics()

            log_text = caplog.text
            assert "PDF fetching statistics:" in log_text
            assert "Total attempted: 100" in log_text
            assert "Already cached: 20" in log_text
            assert "Successful downloads: 60" in log_text
            assert "Failed downloads: 20" in log_text
