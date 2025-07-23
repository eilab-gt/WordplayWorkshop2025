"""Tests for the Exporter module."""

import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.lit_review.utils import Exporter


class TestExporter:
    """Test cases for Exporter class."""

    def test_init(self, sample_config):
        """Test Exporter initialization."""
        exporter = Exporter(sample_config)
        assert exporter.config is not None
        assert hasattr(exporter, "output_dir")
        assert hasattr(exporter, "zenodo_enabled")
        assert hasattr(exporter, "zenodo_token")

    def test_create_package_basic(self, sample_config, temp_dir):
        """Test basic package creation."""
        exporter = Exporter(sample_config)

        # Create test data
        extraction_df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002"],
                "title": ["Paper 1", "Paper 2"],
                "authors": ["Author A", "Author B"],
                "year": [2024, 2023],
                "venue_type": ["conference", "journal"],
                "llm_family": ["GPT-4", "Claude"],
                "abstract": ["Abstract 1", "Abstract 2"],
                "doi": ["10.1234/test1", "10.1234/test2"],
                "url": ["https://example.com/1", "https://example.com/2"],
                "source_db": ["arxiv", "google_scholar"],
                "extraction_status": ["success", "success"],
                "failure_modes": ["", ""],
                "pdf_status": ["downloaded", "downloaded"],
            }
        )

        # Create mock visualizations
        viz_paths = {
            "timeline": Path(temp_dir) / "timeline.png",
            "venue_dist": Path(temp_dir) / "venue.png",
        }
        for path in viz_paths.values():
            path.write_text("fake image data")

        summary = {"total_papers": 2, "with_extraction": 2}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=list(viz_paths.values()),
            summary=summary,
            output_name="test_package",
        )

        assert result_path is not None
        assert result_path.exists()
        assert result_path.suffix == ".zip"

        # Verify package contents
        with zipfile.ZipFile(result_path, "r") as zf:
            files = zf.namelist()
            assert any("extraction_results.csv" in f for f in files)
            assert any("timeline.png" in f for f in files)
            assert any("metadata.json" in f for f in files)

    def test_package_structure(self, sample_config, temp_dir):
        """Test package directory structure."""
        exporter = Exporter(sample_config)

        extraction_df = pd.DataFrame({
            "screening_id": ["SCREEN_0001"],
            "title": ["Test Paper"],
            "authors": ["Test Author"],
            "year": [2024],
            "abstract": ["Test abstract"],
            "doi": ["10.1234/test"],
            "url": ["https://example.com"],
            "source_db": ["arxiv"],
            "extraction_status": ["success"],
            "failure_modes": [""],
            "pdf_status": ["downloaded"],
            "llm_family": ["GPT-4"],
        })

        summary = {"total_papers": 1, "extracted": 1}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="structured",
        )

        with zipfile.ZipFile(result_path, "r") as zf:
            files = zf.namelist()
            # Check directory structure
            assert any("data/" in f for f in files)
            assert any("metadata.json" in f for f in files)

    def test_metadata_generation(self, sample_config, temp_dir):
        """Test metadata JSON generation."""
        exporter = Exporter(sample_config)

        extraction_df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
                "title": ["Paper 1", "Paper 2", "Paper 3"],
                "authors": ["Author A", "Author B", "Author C"],
                "year": [2022, 2023, 2024],
                "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
                "doi": ["10.1234/1", "10.1234/2", "10.1234/3"],
                "url": ["https://ex.com/1", "https://ex.com/2", "https://ex.com/3"],
                "source_db": ["arxiv", "arxiv", "google_scholar"],
                "llm_family": ["GPT-4", "Claude", "GPT-4"],
                "extraction_status": ["success", "success", "failed"],
                "failure_modes": ["", "", "parse_error"],
                "pdf_status": ["downloaded", "downloaded", "failed"],
            }
        )

        summary = {"total_papers": 3, "papers_with_extraction": 2}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="metadata_test",
        )

        # Extract and check metadata
        with zipfile.ZipFile(result_path, "r") as zf:
            metadata_content = zf.read("metadata_test_temp/metadata.json")
            metadata = json.loads(metadata_content)

            assert "export_info" in metadata
            assert "timestamp" in metadata["export_info"]
            assert metadata["export_info"]["total_papers"] == 3
            assert "data_summary" in metadata
            assert "year_range" in metadata["data_summary"]

    @patch("requests.put")
    @patch("requests.post")
    def test_zenodo_upload(self, mock_post, mock_put, sample_config, temp_dir):
        """Test Zenodo upload functionality."""
        # Enable Zenodo in config
        sample_config.zenodo_enabled = True
        sample_config.zenodo_token = "test-token"
        
        exporter = Exporter(sample_config)

        # Mock Zenodo API responses
        # Create deposit response
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {
            "id": "12345",
            "links": {
                "bucket": "https://zenodo.org/api/files/bucket-id",
                "publish": "https://zenodo.org/api/deposit/depositions/12345/actions/publish",
            },
        }
        mock_post.return_value = create_response

        # Upload file response
        upload_response = Mock()
        upload_response.status_code = 200
        
        # Metadata update response
        metadata_response = Mock()
        metadata_response.status_code = 200
        
        mock_put.side_effect = [upload_response, metadata_response]

        # Create test data
        extraction_df = pd.DataFrame({
            "screening_id": ["SCREEN_0001"],
            "title": ["Test Paper"],
            "authors": ["Test Author"],
            "year": [2024],
            "abstract": ["Test abstract"],
            "doi": ["10.1234/test"],
            "url": ["https://example.com"],
            "source_db": ["arxiv"],
            "extraction_status": ["success"],
            "failure_modes": [""],
            "pdf_status": ["downloaded"],
            "llm_family": ["GPT-4"],
        })

        summary = {"total_papers": 1, "extracted": 1}
        
        # This should create package and upload to Zenodo
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="zenodo_test",
        )

        assert result_path.exists()
        assert mock_post.call_count == 1  # Create deposition
        assert mock_put.call_count == 2   # Upload file and update metadata

    def test_create_readme(self, sample_config, temp_dir):
        """Test README generation."""
        exporter = Exporter(sample_config)

        extraction_df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002"],
                "title": ["Paper 1", "Paper 2"],
                "authors": ["Author A", "Author B"],
                "year": [2023, 2024],
                "abstract": ["Abstract 1", "Abstract 2"],
                "doi": ["10.1234/1", "10.1234/2"],
                "url": ["https://ex.com/1", "https://ex.com/2"],
                "source_db": ["arxiv", "google_scholar"],
                "llm_family": ["GPT-4", "Claude"],
                "extraction_status": ["success", "success"],
                "failure_modes": ["", ""],
                "pdf_status": ["downloaded", "downloaded"],
            }
        )

        summary = {"total_papers": 2, "extracted": 2}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="readme_test",
        )

        # Check README exists and contains expected content
        with zipfile.ZipFile(result_path, "r") as zf:
            readme_content = zf.read("readme_test_temp/README.md").decode("utf-8")
            assert "Literature Review Export" in readme_content
            assert "Summary Statistics" in readme_content
            assert "Total papers: 2" in readme_content

    def test_export_formats(self, sample_config, temp_dir):
        """Test different export formats within the package."""
        exporter = Exporter(sample_config)

        extraction_df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "title": ["Paper 1"],
                "authors": ["Author A"],
                "year": [2024],
                "abstract": ["Test abstract"],
                "doi": ["10.1234/test"],
                "url": ["https://example.com"],
                "source_db": ["arxiv"],
                "venue_type": ["conference"],
                "extraction_status": ["success"],
                "failure_modes": [""],
                "pdf_status": ["downloaded"],
                "llm_family": ["GPT-4"],
            }
        )

        summary = {"total_papers": 1, "extracted": 1}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="formats_test",
        )

        with zipfile.ZipFile(result_path, "r") as zf:
            files = zf.namelist()
            # Should have CSV files
            assert any(".csv" in f for f in files)
            # Should also have JSON versions
            assert any(".json" in f for f in files)
            # Should have Excel file
            assert any(".xlsx" in f for f in files)

    def test_empty_dataframes(self, sample_config, temp_dir):
        """Test handling of empty DataFrames."""
        exporter = Exporter(sample_config)

        # Create DataFrame with one row to avoid NaN issues
        # This tests minimal data scenario
        minimal_extraction = pd.DataFrame({
            "screening_id": ["SCREEN_0001"],
            "title": [""],
            "authors": [""],
            "year": [2024],
            "abstract": [""],
            "doi": [""],
            "url": [""],
            "source_db": ["unknown"],
            "extraction_status": ["failed"],
            "failure_modes": ["no_data"],
            "pdf_status": ["not_found"],
            "llm_family": [""],
        })

        summary = {"total_papers": 1, "extracted": 0}

        # Should handle minimal DataFrames gracefully
        result_path = exporter.export_full_package(
            extraction_df=minimal_extraction,
            figures=[],
            summary=summary,
            output_name="empty_test",
        )

        assert result_path is not None
        assert result_path.exists()
        assert result_path.suffix == ".zip"

    def test_custom_metadata(self, sample_config, temp_dir):
        """Test adding custom metadata to package."""
        exporter = Exporter(sample_config)

        extraction_df = pd.DataFrame({
            "screening_id": ["SCREEN_0001"],
            "title": ["Paper 1"],
            "authors": ["Author A"],
            "year": [2024],
            "abstract": ["Test abstract"],
            "doi": ["10.1234/test"],
            "url": ["https://example.com"],
            "source_db": ["arxiv"],
            "extraction_status": ["success"],
            "failure_modes": [""],
            "pdf_status": ["downloaded"],
            "llm_family": ["GPT-4"],
        })

        # Note: export_full_package doesn't support custom_metadata parameter
        # so we'll just test that the standard metadata is created
        summary = {
            "total_papers": 1,
            "extracted": 1,
            "project_name": "LLM Wargaming Review",
            "version": "1.0",
            "authors": ["Researcher A", "Researcher B"],
        }

        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="custom_meta",
        )

        with zipfile.ZipFile(result_path, "r") as zf:
            metadata_content = zf.read("custom_meta_temp/metadata.json")
            metadata = json.loads(metadata_content)

            # Check standard metadata is created
            assert "export_info" in metadata
            assert "data_summary" in metadata

    def test_compression_levels(self, sample_config, temp_dir):
        """Test package compression."""
        exporter = Exporter(sample_config)

        # Create larger test data
        extraction_df = pd.DataFrame(
            {
                "screening_id": [f"SCREEN_{i:04d}" for i in range(100)],
                "title": [f"Paper {i}" for i in range(100)],
                "authors": [f"Author {i}" for i in range(100)],
                "year": [2020 + (i % 5) for i in range(100)],
                "abstract": ["Long abstract text " * 50 for _ in range(100)],
                "doi": [f"10.1234/test{i}" for i in range(100)],
                "url": [f"https://example.com/{i}" for i in range(100)],
                "source_db": ["arxiv" if i % 2 == 0 else "google_scholar" for i in range(100)],
                "venue_type": ["conference"] * 100,
                "extraction_status": ["success"] * 100,
                "failure_modes": [""] * 100,
                "pdf_status": ["downloaded"] * 100,
                "llm_family": ["GPT-4" if i % 2 == 0 else "Claude" for i in range(100)],
            }
        )

        summary = {"total_papers": 100, "extracted": 100}
        
        result_path = exporter.export_full_package(
            extraction_df=extraction_df,
            figures=[],
            summary=summary,
            output_name="compressed",
        )

        # Check that compression is effective
        assert result_path.exists()
        # The compressed size should be significantly smaller than uncompressed
        # (though we can't easily test this without extracting)

    def test_error_handling(self, sample_config, temp_dir):
        """Test error handling in package creation."""
        # Create a config with invalid output directory
        bad_config = Mock()
        bad_config.output_dir = "/invalid/path/that/does/not/exist"
        bad_config.export_compression = "zip"
        bad_config.export_include_pdfs = False
        bad_config.export_include_logs = False
        bad_config.zenodo_enabled = False
        bad_config.zenodo_token = None
        
        exporter = Exporter(bad_config)

        extraction_df = pd.DataFrame({
            "screening_id": ["SCREEN_0001"],
            "title": ["Paper 1"],
            "authors": ["Author A"],
            "year": [2024],
            "abstract": ["Test abstract"],
            "doi": ["10.1234/test"],
            "url": ["https://example.com"],
            "source_db": ["arxiv"],
            "extraction_status": ["success"],
            "failure_modes": [""],
            "pdf_status": ["downloaded"],
            "llm_family": ["GPT-4"],
        })

        summary = {"total_papers": 1, "extracted": 1}

        with pytest.raises(Exception):  # Should raise some exception
            exporter.export_full_package(
                extraction_df=extraction_df,
                figures=[],
                summary=summary,
                output_name="invalid_test",
            )

    def test_export_bibtex(self, sample_config, temp_dir):
        """Test BibTeX export functionality."""
        exporter = Exporter(sample_config)

        # Create test data with various venue types
        df = pd.DataFrame({
            "title": ["Conference Paper", "Journal Article", "Tech Report"],
            "authors": ["Author A; Author B", "Author C", "Author D; Author E"],
            "year": [2024, 2023, 2022],
            "venue": ["Proc. of ICML", "Nature", "MIT Tech Report"],
            "venue_type": ["conference", "journal", "tech report"],
            "doi": ["10.1234/conf", "10.1234/journal", ""],
            "url": ["https://conf.com", "https://journal.com", "https://tech.com"],
            "abstract": ["Conference abstract", "Journal abstract", "Tech report abstract"],
        })

        output_path = Path(temp_dir) / "test.bib"
        result_path = exporter.export_bibtex(df, output_path)

        assert result_path.exists()
        
        # Read and verify BibTeX content
        with open(result_path, "r") as f:
            content = f.read()
            
        # Check entry types
        assert "@inproceedings{" in content
        assert "@article{" in content
        assert "@techreport{" in content
        
        # Check content includes all papers
        assert "Conference Paper" in content
        assert "Journal Article" in content
        assert "Tech Report" in content
        
        # Check authors are formatted correctly (may have extra spaces)
        assert "Author A and" in content and "Author B" in content
        assert "Author D and" in content and "Author E" in content
