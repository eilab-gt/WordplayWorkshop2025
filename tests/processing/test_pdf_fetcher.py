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
        assert fetcher.pdf_dir.exists()

    def test_clean_filename(self, sample_config):
        """Test filename cleaning."""
        fetcher = PDFFetcher(sample_config)

        # Test various problematic characters
        filename = fetcher._clean_filename("Test: Paper? With* Special/Characters|2024")
        assert ":" not in filename
        assert "?" not in filename
        assert "*" not in filename
        assert "/" not in filename
        assert "|" not in filename

        # Test truncation of long filenames
        long_title = "A" * 300
        filename = fetcher._clean_filename(long_title)
        assert len(filename) <= 255

    @patch("requests.get")
    def test_download_pdf_direct(self, mock_get, sample_config, temp_dir):
        """Test direct PDF download."""
        # Mock successful PDF download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_get.return_value = mock_response

        fetcher = PDFFetcher(sample_config)

        # Test download
        pdf_path = fetcher._download_pdf(
            "https://example.com/paper.pdf", "test_paper.pdf"
        )

        assert pdf_path is not None
        assert Path(pdf_path).exists()
        assert Path(pdf_path).suffix == ".pdf"

    @patch("requests.get")
    def test_download_pdf_arxiv(self, mock_get, sample_config):
        """Test arXiv PDF download."""
        # Mock successful PDF download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4 fake arxiv pdf"
        mock_get.return_value = mock_response

        fetcher = PDFFetcher(sample_config)

        # Create test paper with arXiv ID
        paper = pd.Series(
            {
                "title": "Test arXiv Paper",
                "arxiv_id": "2301.12345",
                "pdf_url": "https://arxiv.org/pdf/2301.12345.pdf",
            }
        )

        pdf_path = fetcher._fetch_arxiv(paper)

        assert pdf_path is not None
        mock_get.assert_called_with(
            "https://arxiv.org/pdf/2301.12345.pdf",
            headers={"User-Agent": fetcher.headers["User-Agent"]},
            timeout=30,
        )

    @patch("requests.get")
    def test_download_pdf_failure(self, mock_get, sample_config):
        """Test handling of download failures."""
        # Mock failed download
        mock_get.side_effect = requests.RequestException("Connection error")

        fetcher = PDFFetcher(sample_config)

        # Test download failure handling
        pdf_path = fetcher._download_pdf(
            "https://example.com/paper.pdf", "test_paper.pdf"
        )

        assert pdf_path is None

    def test_fetch_batch(self, sample_config, sample_screening_df):
        """Test batch PDF fetching."""
        fetcher = PDFFetcher(sample_config)

        with patch.object(fetcher, "_fetch_arxiv") as mock_arxiv:
            with patch.object(fetcher, "_fetch_doi") as mock_doi:
                with patch.object(fetcher, "_fetch_direct_url") as mock_direct:
                    # Setup mocks
                    mock_arxiv.return_value = "pdf_cache/arxiv_paper.pdf"
                    mock_doi.return_value = None
                    mock_direct.return_value = "pdf_cache/direct_paper.pdf"

                    # Run batch fetch
                    updated_df = fetcher.fetch_batch(sample_screening_df, max_workers=2)

                    assert isinstance(updated_df, pd.DataFrame)
                    assert "pdf_path" in updated_df.columns
                    assert "pdf_status" in updated_df.columns

                    # Check that appropriate methods were called
                    assert mock_arxiv.called or mock_doi.called or mock_direct.called

    @patch("requests.get")
    def test_fetch_doi_scihub(self, mock_get, sample_config):
        """Test Sci-Hub DOI fetching."""
        # Mock redirect and PDF download
        mock_redirect = Mock()
        mock_redirect.status_code = 302
        mock_redirect.headers = {"Location": "https://sci-hub.se/downloads/paper.pdf"}

        mock_pdf = Mock()
        mock_pdf.status_code = 200
        mock_pdf.content = b"%PDF-1.4 fake pdf"

        mock_get.side_effect = [mock_redirect, mock_pdf]

        fetcher = PDFFetcher(sample_config)

        paper = pd.Series({"title": "Test DOI Paper", "doi": "10.1234/test.2024.001"})

        fetcher._fetch_doi(paper)

        # Should attempt Sci-Hub
        assert any("sci-hub" in str(call) for call in mock_get.call_args_list)

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
            }
        )

        with patch.object(fetcher, "_fetch_arxiv") as mock_arxiv:
            with patch.object(fetcher, "_fetch_doi") as mock_doi:
                with patch.object(fetcher, "_fetch_direct_url") as mock_direct:
                    # Setup different outcomes
                    mock_arxiv.return_value = "pdf_cache/paper1.pdf"
                    mock_doi.return_value = None  # Failed
                    mock_direct.return_value = "pdf_cache/paper3.pdf"

                    updated_df = fetcher.fetch_batch(df)

                    # Check status tracking
                    assert updated_df.loc[0, "pdf_status"] == "downloaded_arxiv"
                    assert updated_df.loc[1, "pdf_status"] in [
                        "not_found",
                        "download_failed",
                    ]
                    assert updated_df.loc[2, "pdf_status"] == "downloaded_direct"

    def test_skip_existing_pdfs(self, sample_config, temp_dir):
        """Test that existing PDFs are not re-downloaded."""
        fetcher = PDFFetcher(sample_config)

        # Create a fake existing PDF
        existing_pdf = fetcher.pdf_dir / "existing_paper.pdf"
        existing_pdf.write_bytes(b"%PDF-1.4 existing content")

        # Create DataFrame with paper that already has PDF
        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "title": ["Existing Paper"],
                "pdf_path": [str(existing_pdf)],
                "pdf_status": ["downloaded_direct"],
            }
        )

        with patch.object(fetcher, "_download_pdf") as mock_download:
            updated_df = fetcher.fetch_batch(df)

            # Should not attempt to download
            mock_download.assert_not_called()
            assert updated_df.loc[0, "pdf_path"] == str(existing_pdf)

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
            }
        )

        with patch.object(fetcher, "_fetch_arxiv", return_value="arxiv.pdf"):
            with patch.object(fetcher, "_fetch_doi", return_value="doi.pdf"):
                # Test with parallel execution
                updated_df = fetcher.fetch_batch(df, max_workers=3)

                assert len(updated_df) == 5
                assert all(updated_df["pdf_path"].notna())

    def test_url_validation(self, sample_config):
        """Test URL validation before download attempts."""
        fetcher = PDFFetcher(sample_config)

        # Test various invalid URLs
        paper_invalid = pd.Series({"title": "Test Paper", "url": "not-a-valid-url"})

        paper_empty = pd.Series({"title": "Test Paper", "url": ""})

        assert fetcher._fetch_direct_url(paper_invalid) is None
        assert fetcher._fetch_direct_url(paper_empty) is None
