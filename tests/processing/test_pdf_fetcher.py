"""Tests for the PDFFetcher module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.lit_review.processing import PDFFetcher


class TestPDFFetcher:
    """Test cases for PDFFetcher class."""

    def test_init(self, sample_config, temp_dir):
        """Test PDFFetcher initialization."""
        fetcher = PDFFetcher(sample_config)
        assert fetcher.config is not None
        assert fetcher.cache_dir.exists()

    def test_generate_filename(self, sample_config):
        """Test filename generation."""
        fetcher = PDFFetcher(sample_config)

        # Test filename generation from paper data
        paper = pd.Series(
            {
                "title": "Test: Paper? With* Special/Characters|2024",
                "authors": "John Doe; Jane Smith",
                "year": 2024,
            }
        )
        filename = fetcher._generate_filename(paper)
        assert filename.endswith(".pdf")
        # Check that special characters are removed
        assert ":" not in filename
        assert "?" not in filename
        assert "*" not in filename
        assert "/" not in filename
        assert "|" not in filename

        # Test with missing authors
        paper_no_author = pd.Series({"title": "Test Paper", "year": 2024})
        filename = fetcher._generate_filename(paper_no_author)
        assert "Unknown" in filename

        # Test filename length limit
        long_title = "A" * 300
        paper_long = pd.Series(
            {"title": long_title, "authors": "Author Name", "year": 2024}
        )
        filename = fetcher._generate_filename(paper_long)
        # Base name is limited to 100 chars + .pdf
        assert len(filename) <= 104

    def test_download_pdf_direct(self, sample_config, temp_dir):
        """Test direct PDF download."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher.session, "get") as mock_get:
            # Mock successful PDF download
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/pdf"}
            mock_response.iter_content = lambda chunk_size: [
                b"%PDF-1.4 fake pdf content"
            ]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Test download
            filepath = fetcher.cache_dir / "test_paper.pdf"
            success = fetcher._download_pdf("https://example.com/paper.pdf", filepath)

            assert success is True
            assert filepath.exists()
            assert filepath.suffix == ".pdf"

    def test_fetch_single_pdf_arxiv(self, sample_config):
        """Test arXiv PDF fetching."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher, "_download_pdf_content") as mock_download:
            # Mock successful PDF download
            mock_download.return_value = b"%PDF-1.4 fake pdf content"

            # Create test paper with arXiv ID
            paper = pd.Series(
                {
                    "title": "Test arXiv Paper",
                    "arxiv_id": "2301.12345",
                    "authors": "Test Author",
                    "year": 2023,
                }
            )

            result = fetcher._fetch_single_pdf(paper)

            assert result["status"] == "downloaded_arxiv"
            assert result["path"]
            # Check that arxiv URL was used
            mock_download.assert_called_once()
            call_args = mock_download.call_args[0]
            assert "arxiv.org/pdf/2301.12345.pdf" in call_args[0]

    def test_download_pdf_failure(self, sample_config):
        """Test handling of download failures."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher.session, "get") as mock_get:
            # Mock failed download
            mock_get.side_effect = requests.RequestException("Connection error")

            # Test download failure handling
            filepath = fetcher.cache_dir / "test_paper.pdf"
            success = fetcher._download_pdf("https://example.com/paper.pdf", filepath)

            assert success is False

    def test_fetch_pdfs(self, sample_config, sample_screening_df):
        """Test batch PDF fetching."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher, "_fetch_single_pdf") as mock_fetch:
            # Setup mock to return different results
            mock_fetch.side_effect = [
                {
                    "path": "pdf_cache/arxiv_paper.pdf",
                    "status": "downloaded_arxiv",
                    "hash": "abc123",
                },
                {"path": "", "status": "not_found", "hash": ""},
                {
                    "path": "pdf_cache/direct_paper.pdf",
                    "status": "downloaded_direct",
                    "hash": "def456",
                },
            ]

            # Run batch fetch
            updated_df = fetcher.fetch_pdfs(sample_screening_df, parallel=False)

            assert isinstance(updated_df, pd.DataFrame)
            assert "pdf_path" in updated_df.columns
            assert "pdf_status" in updated_df.columns
            assert "pdf_hash" in updated_df.columns

            # Check that fetch was called for each row
            assert mock_fetch.call_count == len(sample_screening_df)

    def test_fetch_unpaywall(self, sample_config):
        """Test Unpaywall DOI fetching."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher.session, "get") as mock_get:
            # Mock Unpaywall API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "best_oa_location": {"url_for_pdf": "https://example.com/oa_paper.pdf"}
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            pdf_url = fetcher._get_unpaywall_url("10.1234/test.2024.001")

            assert pdf_url == "https://example.com/oa_paper.pdf"
            # Verify Unpaywall API was called correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "unpaywall.org" in call_args[0][0]
            assert call_args[1]["params"]["email"] == fetcher.email

    def test_pdf_status_tracking(self, sample_config):
        """Test PDF download status tracking."""
        fetcher = PDFFetcher(sample_config)

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
                "title": ["Paper 1", "Paper 2", "Paper 3"],
                "arxiv_id": ["2301.12345", "", ""],
                "doi": ["", "10.1234/test", ""],
                "url": ["", "", "https://example.com/paper3.pdf"],
                "authors": ["Author One", "Author Two", "Author Three"],
                "year": [2023, 2024, 2024],
            }
        )

        with patch.object(fetcher, "_fetch_single_pdf") as mock_fetch:
            # Setup different outcomes
            mock_fetch.side_effect = [
                {
                    "path": "pdf_cache/paper1.pdf",
                    "status": "downloaded_arxiv",
                    "hash": "hash1",
                },
                {"path": "", "status": "not_found", "hash": ""},
                {
                    "path": "pdf_cache/paper3.pdf",
                    "status": "downloaded_direct",
                    "hash": "hash3",
                },
            ]

            updated_df = fetcher.fetch_pdfs(df, parallel=False)

            # Check status tracking
            assert updated_df.loc[0, "pdf_status"] == "downloaded_arxiv"
            assert updated_df.loc[1, "pdf_status"] == "not_found"
            assert updated_df.loc[2, "pdf_status"] == "downloaded_direct"

    def test_skip_existing_pdfs(self, sample_config, temp_dir):
        """Test that existing PDFs are not re-downloaded."""
        fetcher = PDFFetcher(sample_config)

        # Create a fake existing PDF in the proper cache location
        paper = pd.Series(
            {
                "title": "Existing Paper",
                "authors": "Test Author",
                "year": 2024,
                "arxiv_id": "2301.12345",
            }
        )

        # Mock the content cache to return cached result
        fake_path = Path(temp_dir) / "pdfs" / "cached_paper.pdf"
        fake_path.parent.mkdir(exist_ok=True)
        fake_path.write_bytes(b"%PDF-1.4 existing content")

        with patch.object(fetcher.content_cache, "get_or_fetch") as mock_cache:
            # Return cached result (path, was_cached=True)
            mock_cache.return_value = (fake_path, True)

            # Create DataFrame with paper that should map to existing PDF
            df = pd.DataFrame(
                {
                    "screening_id": ["SCREEN_0001"],
                    "title": ["Existing Paper"],
                    "authors": ["Test Author"],
                    "year": [2024],
                    "arxiv_id": ["2301.12345"],
                }
            )

            updated_df = fetcher.fetch_pdfs(df, parallel=False)

            # Should have used the cache
            assert updated_df.loc[0, "pdf_status"] == "cached"
            assert updated_df.loc[0, "pdf_path"] == str(fake_path)

    def test_parallel_download(self, sample_config):
        """Test parallel PDF downloading."""
        fetcher = PDFFetcher(sample_config)

        # Create test DataFrame with multiple papers
        df = pd.DataFrame(
            {
                "screening_id": [f"SCREEN_{i:04d}" for i in range(5)],
                "title": [f"Paper {i}" for i in range(5)],
                "arxiv_id": ["2301.12345" if i % 2 == 0 else "" for i in range(5)],
                "doi": ["" if i % 2 == 0 else f"10.1234/test.{i}" for i in range(5)],
                "authors": [f"Author {i}" for i in range(5)],
                "year": [2024 for i in range(5)],
            }
        )

        with patch.object(fetcher, "_fetch_single_pdf") as mock_fetch:
            mock_fetch.return_value = {
                "path": "test.pdf",
                "status": "downloaded_arxiv",
                "hash": "testhash",
            }

            # Test with parallel execution
            updated_df = fetcher.fetch_pdfs(df, parallel=True)

            assert len(updated_df) == 5
            assert all(updated_df["pdf_path"] == "test.pdf")
            assert mock_fetch.call_count == 5

    def test_url_validation(self, sample_config):
        """Test URL validation in PDF fetching."""
        fetcher = PDFFetcher(sample_config)

        # Test paper with non-PDF URL
        paper_non_pdf = pd.Series(
            {
                "title": "Test Paper",
                "url": "https://example.com/page.html",
                "authors": "Test Author",
                "year": 2024,
            }
        )

        # Test paper with PDF URL
        paper_pdf = pd.Series(
            {
                "title": "Test Paper",
                "url": "https://example.com/paper.pdf",
                "authors": "Test Author",
                "year": 2024,
            }
        )

        with patch.object(fetcher, "_download_pdf_content") as mock_download:
            # Mock successful PDF download
            mock_download.return_value = b"%PDF-1.4 fake pdf content"

            # Non-PDF URL should not trigger download from URL field alone
            result1 = fetcher._fetch_single_pdf(paper_non_pdf)
            assert result1["status"] == "not_found"

            # PDF URL should trigger download
            result2 = fetcher._fetch_single_pdf(paper_pdf)
            assert result2["status"] == "downloaded_direct"
