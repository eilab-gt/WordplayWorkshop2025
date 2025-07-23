"""Tests for enhanced LLM extractor."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import requests

from src.lit_review.extraction.enhanced_llm_extractor import EnhancedLLMExtractor


class TestEnhancedLLMExtractor:
    """Test suite for enhanced LLM extractor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock()
        config.parallel_workers = 3
        return config

    @pytest.fixture
    def extractor(self, config):
        """Create extractor instance."""
        return EnhancedLLMExtractor(config, "http://localhost:8000")

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "title": ["Paper 1", "Paper 2", "Paper 3"],
                "authors": ["Author A", "Author B", "Author C"],
                "year": [2024, 2024, 2023],
                "abstract": [
                    "LLM wargaming simulation",
                    "AI agent research",
                    "Machine learning study",
                ],
                "arxiv_id": ["2401.00001", None, "2301.00002"],
                "pdf_path": [None, "test.pdf", "test2.pdf"],
                "pdf_status": ["", "cached", "downloaded_success"],
                "source_db": ["arxiv", "other", "arxiv"],
            }
        )

    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.llm_service_url == "http://localhost:8000"
        assert len(extractor.model_preferences) >= 3
        assert "gemini/gemini-pro" in extractor.model_preferences
        assert extractor.stats["total_attempted"] == 0

    @patch("requests.get")
    def test_check_service_health_success(self, mock_get, extractor):
        """Test service health check when healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert extractor.check_service_health() is True
        mock_get.assert_called_with("http://localhost:8000/health", timeout=5)

    @patch("requests.get")
    def test_check_service_health_failure(self, mock_get, extractor):
        """Test service health check when unhealthy."""
        mock_get.side_effect = requests.RequestException("Connection error")

        assert extractor.check_service_health() is False

    @patch("requests.get")
    def test_get_available_models(self, mock_get, extractor):
        """Test getting available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "gemini/gemini-pro": {"available": True},
            "gpt-3.5-turbo": {"available": False},
        }
        mock_get.return_value = mock_response

        models = extractor.get_available_models()
        assert len(models) == 2
        assert models["gemini/gemini-pro"]["available"] is True

    def test_filter_papers_for_extraction(self, extractor, sample_df):
        """Test filtering papers for extraction."""
        filtered = extractor._filter_papers_for_extraction(sample_df)

        # Should include papers with PDFs and arxiv IDs
        assert len(filtered) == 3
        assert all(idx in filtered.index for idx in [0, 1, 2])

    @patch("requests.get")
    @patch("requests.post")
    def test_extract_all_no_service(self, mock_post, mock_get, extractor, sample_df):
        """Test extraction when service is not available."""
        # Service health check fails
        mock_get.side_effect = requests.RequestException()

        result = extractor.extract_all(sample_df)

        # Should return original DataFrame
        assert len(result) == len(sample_df)
        assert mock_post.call_count == 0

    @patch.object(EnhancedLLMExtractor, "check_service_health", return_value=True)
    @patch.object(EnhancedLLMExtractor, "get_available_models", return_value={})
    @patch.object(EnhancedLLMExtractor, "_extract_single_paper")
    def test_extract_all_sequential(
        self, mock_extract, mock_models, mock_health, extractor, sample_df
    ):
        """Test sequential extraction."""
        mock_extract.return_value = {"extraction_status": "success"}

        result = extractor.extract_all(sample_df, parallel=False)

        assert mock_extract.call_count == 3
        assert "extraction_status" in result.columns

    def test_get_paper_content_pdf_only(self, extractor, tmp_path):
        """Test getting content from PDF."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("PDF content")

        paper = pd.Series({"pdf_path": str(pdf_path), "arxiv_id": None})

        with patch.object(
            extractor, "_extract_pdf_text", return_value="Extracted text"
        ):
            content, content_type = extractor._get_paper_content(paper)

            assert content == "Extracted text"
            assert content_type == "pdf"

    @patch.object(EnhancedLLMExtractor, "_extract_tex_content")
    def test_get_paper_content_tex_priority(self, mock_tex, extractor):
        """Test TeX content is prioritized over PDF."""
        mock_tex.return_value = ("TeX content", True)

        paper = pd.Series({"arxiv_id": "2401.00001", "pdf_path": "test.pdf"})

        content, content_type = extractor._get_paper_content(paper)

        assert content == "TeX content"
        assert content_type == "tex"
        mock_tex.assert_called_once()

    @patch.object(
        EnhancedLLMExtractor, "_extract_tex_content", return_value=("", False)
    )
    @patch.object(EnhancedLLMExtractor, "_extract_html_content")
    def test_get_paper_content_html_fallback(self, mock_html, mock_tex, extractor):
        """Test HTML content as fallback."""
        mock_html.return_value = ("HTML content", True)

        paper = pd.Series({"arxiv_id": "2401.00001", "pdf_path": None})

        content, content_type = extractor._get_paper_content(paper)

        assert content == "HTML content"
        assert content_type == "html"

    def test_clean_tex_content(self, extractor):
        """Test TeX content cleaning."""
        tex_input = r"""
        \documentclass{article}
        \begin{document}
        \section{Introduction}
        This is \textbf{bold} and \textit{italic} text.
        \cite{ref1} shows that \ref{fig1} is important.
        \begin{equation}
        E = mc^2
        \end{equation}
        \end{document}
        """

        cleaned = extractor._clean_tex_content(tex_input)

        assert "## Introduction" in cleaned
        assert "This is bold and italic text." in cleaned
        assert "[citation]" in cleaned
        assert "[equation]" in cleaned

    @patch("requests.post")
    def test_llm_service_extract_success(self, mock_post, extractor):
        """Test successful LLM service extraction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "extracted_data": {
                "research_questions": "What are the questions?",
                "key_contributions": "Main contributions",
            },
        }
        mock_post.return_value = mock_response

        paper = pd.Series(
            {
                "title": "Test Paper",
                "authors": "Author A",
                "year": 2024,
                "abstract": "Test abstract",
            }
        )

        result = extractor._llm_service_extract("Test content", paper)

        assert result is not None
        assert result["extraction_status"] == "success"
        assert result["extraction_model"] == "gemini/gemini-pro"
        assert "research_questions" in result

    @patch("requests.post")
    def test_llm_service_extract_fallback(self, mock_post, extractor):
        """Test fallback to other models."""
        # First model fails with 401
        response1 = Mock()
        response1.status_code = 401

        # Second model succeeds
        response2 = Mock()
        response2.status_code = 200
        response2.json.return_value = {
            "success": True,
            "extracted_data": {"field": "value"},
        }

        mock_post.side_effect = [response1, response2]

        paper = pd.Series({"title": "Test"})
        result = extractor._llm_service_extract("Content", paper)

        assert result is not None
        assert mock_post.call_count == 2

    def test_assign_awscale(self, extractor):
        """Test AWScale assignment."""
        # Test various combinations
        test_cases = [
            ({"simulation_approach": "matrix game"}, 4),
            ({"simulation_approach": "digital"}, 2),
            (
                {
                    "simulation_approach": "unknown",
                    "human_llm_comparison": "human vs llm",
                },
                4,
            ),
            ({}, 3),  # Default
        ]

        for extracted, expected in test_cases:
            score = extractor._assign_awscale(extracted)
            assert 1 <= score <= 5
            assert score == expected

    def test_calculate_confidence(self, extractor):
        """Test confidence calculation."""
        # Full extraction
        extracted = {
            "research_questions": "Questions",
            "key_contributions": "Contributions",
            "simulation_approach": "Approach",
            "llm_usage": "Usage",
            "evaluation_metrics": "Metrics",
        }

        confidence = extractor._calculate_confidence(extracted)
        assert 0.5 <= confidence <= 1.0
        assert confidence > 0.9  # Should be high with all fields

        # Minimal extraction
        minimal = {"research_questions": "Questions"}
        min_confidence = extractor._calculate_confidence(minimal)
        assert min_confidence < confidence

    def test_extract_single_paper_insufficient_content(self, extractor):
        """Test extraction with insufficient content."""
        paper = pd.Series({"pdf_path": None, "arxiv_id": None})

        with patch.object(extractor, "_get_paper_content", return_value=("", "none")):
            result = extractor._extract_single_paper(paper)

            assert result["extraction_status"] == "insufficient_content"

    def test_extract_single_paper_success(self, extractor):
        """Test successful single paper extraction."""
        paper = pd.Series({"title": "Test Paper", "arxiv_id": "2401.00001"})

        # Mock the LLM service extract to return a valid extraction
        mock_extraction = {
            "extraction_status": "success",
            "research_questions": "What are the questions?",
            "key_contributions": "Main contributions",
        }

        with patch.object(
            extractor, "_get_paper_content", return_value=("Long content text", "tex")
        ):
            with patch.object(
                extractor, "_llm_service_extract", return_value=mock_extraction
            ):
                result = extractor._extract_single_paper(paper)

                assert result["extraction_status"] == "success"
                assert result["content_type"] == "tex"
                assert extractor.stats["tex_processed"] == 1

    def test_log_statistics(self, extractor, caplog):
        """Test statistics logging."""
        import logging

        extractor.stats = {
            "total_attempted": 10,
            "successful_extractions": 7,
            "failed_extractions": 3,
            "pdf_processed": 5,
            "tex_processed": 2,
            "html_processed": 0,
            "llm_errors": 1,
            "awscale_assignments": 7,
        }

        with caplog.at_level(logging.INFO):
            extractor._log_statistics()

        assert "Enhanced LLM extraction statistics:" in caplog.text
        assert "Total attempted: 10" in caplog.text
        assert "TeX processed: 2" in caplog.text


class TestTeXHTMLExtraction:
    """Test TeX and HTML extraction methods."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = MagicMock()
        return EnhancedLLMExtractor(config)

    @patch("src.lit_review.extraction.enhanced_llm_extractor.ArxivHarvester")
    def test_extract_tex_content_success(self, mock_harvester_class, extractor):
        """Test successful TeX extraction."""
        mock_harvester = Mock()
        mock_harvester.fetch_tex_source.return_value = (
            r"\documentclass{article}\begin{document}Content\end{document}"
        )
        mock_harvester_class.return_value = mock_harvester

        content, success = extractor._extract_tex_content("2401.00001")

        assert success is True
        assert len(content) > 0

    @patch("src.lit_review.extraction.enhanced_llm_extractor.ArxivHarvester")
    def test_extract_html_content_success(self, mock_harvester_class, extractor):
        """Test successful HTML extraction."""
        mock_harvester = Mock()
        mock_harvester.fetch_html_source.return_value = (
            "<html><body>Content</body></html>"
        )
        mock_harvester_class.return_value = mock_harvester

        content, success = extractor._extract_html_content("2401.00001")

        assert success is True
        assert "Content" in content

    def test_extract_pdf_text(self, extractor, tmp_path):
        """Test PDF text extraction."""
        # This will fail without a real PDF, but tests the method exists
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        with patch("pdfminer.high_level.extract_text", return_value="Extracted text"):
            text = extractor._extract_pdf_text(pdf_path)
            assert text == "Extracted text"
