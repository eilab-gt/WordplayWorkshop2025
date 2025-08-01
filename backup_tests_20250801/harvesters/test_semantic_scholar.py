"""Tests for Semantic Scholar harvester."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from src.lit_review.harvesters.semantic_scholar import SemanticScholarHarvester
from tests.test_fixtures import (
    create_mock_response,
    create_semantic_scholar_response,
)
from tests.test_utils import (
    assert_rate_limiting_applied,
    mock_time_sleep,
)


class TestSemanticScholarHarvester:
    """Test suite for Semantic Scholar harvester."""

    @pytest.fixture
    def harvester(self, mock_config):
        """Create Semantic Scholar harvester instance."""
        # Add API key to config
        mock_config.semantic_scholar_key = "test-api-key"
        return SemanticScholarHarvester(mock_config)

    def test_init(self, mock_config):
        """Test harvester initialization."""
        mock_config.semantic_scholar_key = "test-api-key"
        harvester = SemanticScholarHarvester(mock_config)

        assert harvester.api_key == "test-api-key"
        assert harvester.rate_limits == mock_config.rate_limits["semantic_scholar"]
        assert harvester.delay_milliseconds == 10
        assert harvester.headers == {"x-api-key": "test-api-key"}

    def test_init_no_api_key(self, mock_config):
        """Test initialization without API key."""
        mock_config.semantic_scholar_key = None
        harvester = SemanticScholarHarvester(mock_config)

        assert harvester.api_key is None
        assert harvester.headers == {}

    @patch("requests.get")
    def test_search_success(self, mock_get, harvester):
        """Test successful search."""
        # Mock API response
        response_data = create_semantic_scholar_response(
            [
                {
                    "paper_id": "abc123",
                    "title": "LLM-powered Wargaming Systems",
                    "authors": ["John Doe", "Jane Smith"],
                    "year": 2024,
                    "abstract": "This paper presents an LLM-based approach to wargaming...",
                    "venue": "AI Conference 2024",
                    "citations": 10,
                    "doi": "10.1234/test1",
                },
                {
                    "paper_id": "def456",
                    "title": "GPT-4 in Military Simulations",
                    "authors": ["Alice Johnson"],
                    "year": 2023,
                    "abstract": "We explore the use of GPT-4 in military simulations...",
                    "venue": "Defense AI Journal",
                    "citations": 5,
                    "arxiv_id": "2301.12345",
                },
            ]
        )

        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        with mock_time_sleep() as mock_sleep:
            papers = harvester.search("LLM wargaming", max_results=10)

        # Verify results
        assert len(papers) == 2

        # Check first paper
        assert papers[0].title == "LLM-powered Wargaming Systems"
        assert papers[0].authors == ["John Doe", "Jane Smith"]
        assert papers[0].year == 2024
        assert papers[0].source_db == "semantic_scholar"
        assert papers[0].citations == 10
        assert papers[0].venue == "AI Conference 2024"
        assert papers[0].doi == "10.1234/test1"

        # Check second paper (with arXiv)
        assert papers[1].title == "GPT-4 in Military Simulations"
        assert papers[1].authors == ["Alice Johnson"]
        assert papers[1].year == 2023
        assert papers[1].arxiv_id == "2301.12345"
        assert papers[1].pdf_url == "https://arxiv.org/pdf/2301.12345.pdf"

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "paper/search" in call_args[0][0]
        assert call_args[1]["params"]["query"] == "LLM wargaming"
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["headers"] == {"x-api-key": "test-api-key"}

        # No rate limiting for single batch
        assert mock_sleep.call_count == 0

    @patch("requests.get")
    def test_search_pagination(self, mock_get, harvester):
        """Test search with pagination."""
        # Create responses for pagination
        responses = []
        for i in range(3):
            papers = [
                {
                    "paper_id": f"id{i}_{j}",
                    "title": f"Paper {i}-{j}",
                    "authors": [f"Author {i}-{j}"],
                    "year": 2024,
                }
                for j in range(100)
            ]
            response_data = create_semantic_scholar_response(papers)
            responses.append(create_mock_response(200, response_data))

        mock_get.side_effect = responses

        with mock_time_sleep() as mock_sleep:
            papers = harvester.search("test query", max_results=250)

        assert len(papers) == 250
        assert mock_get.call_count == 3
        # Rate limiting between batches
        assert_rate_limiting_applied(mock_sleep, expected_calls=3, min_delay_ms=10)

    @patch("requests.get")
    def test_search_api_error(self, mock_get, harvester):
        """Test search with API error."""
        mock_get.side_effect = requests.RequestException("API Error")

        papers = harvester.search("test query")

        assert papers == []

    @patch("requests.get")
    def test_search_invalid_json(self, mock_get, harvester):
        """Test search with invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        assert papers == []

    @patch("requests.get")
    def test_search_filters_by_year(self, mock_get, harvester):
        """Test that search filters results by year."""
        response_data = create_semantic_scholar_response(
            [
                {
                    "paper_id": "old123",
                    "title": "Old Paper",
                    "authors": ["Author 1"],
                    "year": 2015,  # Too old
                },
                {
                    "paper_id": "recent456",
                    "title": "Recent Paper",
                    "authors": ["Author 2"],
                    "year": 2023,  # Within range
                },
                {
                    "paper_id": "future789",
                    "title": "Future Paper",
                    "authors": ["Author 3"],
                    "year": 2026,  # Too new
                },
            ]
        )

        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        # Verify year filter was included in API call
        call_args = mock_get.call_args
        assert call_args[1]["params"]["year"] == "2018-2025"

        # Additional filtering after retrieval
        assert len(papers) == 1
        assert papers[0].title == "Recent Paper"
        assert papers[0].year == 2023

    def test_extract_paper_missing_fields(self, harvester):
        """Test paper extraction with missing fields."""
        result = {
            "title": "Minimal Paper",
            "year": 2024,
            # Missing authors, abstract, etc.
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.title == "Minimal Paper"
        assert paper.authors == []
        assert paper.abstract == ""
        assert paper.year == 2024

    def test_extract_paper_no_title(self, harvester):
        """Test paper extraction with no title."""
        result = {
            "authors": [{"name": "Test Author"}],
            "year": 2024,
        }

        paper = harvester._extract_paper(result)

        assert paper is None

    def test_extract_paper_with_external_ids(self, harvester):
        """Test extraction of DOI and arXiv ID."""
        result = {
            "title": "Test Paper",
            "authors": [{"name": "Test Author"}],
            "year": 2024,
            "externalIds": {
                "DOI": "10.1234/test",
                "ArXiv": "2401.12345",
                "PubMed": "12345678",  # Should be ignored
            },
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.doi == "10.1234/test"
        assert paper.arxiv_id == "2401.12345"
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_extract_paper_with_publication_types(self, harvester):
        """Test extraction of publication types as keywords."""
        result = {
            "title": "Test Paper",
            "authors": [{"name": "Test Author"}],
            "year": 2024,
            "publicationTypes": ["Conference", "Journal"],
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.keywords == ["Conference", "Journal"]

    @patch("requests.get")
    def test_get_paper_by_doi(self, mock_get, harvester):
        """Test getting paper by DOI."""
        response_data = {
            "paperId": "abc123",
            "title": "Paper Retrieved by DOI",
            "authors": [{"name": "Test Author"}],
            "year": 2024,
            "externalIds": {"DOI": "10.1234/test"},
        }

        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        paper = harvester.get_paper_by_id("10.1234/test")

        assert paper is not None
        assert paper.title == "Paper Retrieved by DOI"
        assert paper.doi == "10.1234/test"

        # Verify API call
        mock_get.assert_called_once()
        assert "paper/DOI:10.1234/test" in mock_get.call_args[0][0]

    @patch("requests.get")
    def test_get_paper_by_semantic_scholar_id(self, mock_get, harvester):
        """Test getting paper by Semantic Scholar ID."""
        response_data = {
            "paperId": "abc123",
            "title": "Paper Retrieved by SS ID",
            "authors": [{"name": "Test Author"}],
            "year": 2024,
        }

        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        paper = harvester.get_paper_by_id("abc123")

        assert paper is not None
        assert paper.title == "Paper Retrieved by SS ID"

        # Verify API call
        mock_get.assert_called_once()
        assert "paper/abc123" in mock_get.call_args[0][0]

    @patch("requests.get")
    def test_get_paper_by_id_not_found(self, mock_get, harvester):
        """Test getting paper when not found."""
        mock_response = create_mock_response(404)
        mock_get.return_value = mock_response

        paper = harvester.get_paper_by_id("notfound")

        assert paper is None

    @patch("requests.get")
    def test_get_recommendations(self, mock_get, harvester):
        """Test getting paper recommendations."""
        response_data = {
            "recommendedPapers": [
                {
                    "paperId": "rec1",
                    "title": "Recommended Paper 1",
                    "authors": [{"name": "Author 1"}],
                    "year": 2024,
                },
                {
                    "paperId": "rec2",
                    "title": "Recommended Paper 2",
                    "authors": [{"name": "Author 2"}],
                    "year": 2023,
                },
            ]
        }

        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        papers = harvester.get_recommendations("seed123", max_results=10)

        assert len(papers) == 2
        assert papers[0].title == "Recommended Paper 1"
        assert papers[1].title == "Recommended Paper 2"

        # Verify API call
        mock_get.assert_called_once()
        assert "recommendations/v1/papers/forpaper/seed123" in mock_get.call_args[0][0]
        assert mock_get.call_args[1]["params"]["limit"] == 10

    @patch("requests.get")
    def test_get_recommendations_error(self, mock_get, harvester):
        """Test recommendations with API error."""
        mock_get.side_effect = requests.RequestException("API Error")

        papers = harvester.get_recommendations("seed123")

        assert papers == []

    def test_clean_text_functionality(self, harvester):
        """Test clean_text method inherited from base class."""
        assert harvester.clean_text("  Extra   spaces  ") == "Extra spaces"
        assert harvester.clean_text(None) == ""
        assert harvester.clean_text("") == ""
        assert harvester.clean_text("Normal text") == "Normal text"

    @patch("requests.get")
    def test_rate_limiting_between_batches(self, mock_get, harvester):
        """Test that rate limiting is applied between batches."""
        # Create two full batches
        responses = []
        for i in range(2):
            papers = [
                {
                    "paper_id": f"id{i}_{j}",
                    "title": f"Paper {i}-{j}",
                    "authors": [{"name": f"Author {i}-{j}"}],
                    "year": 2024,
                }
                for j in range(100)
            ]
            response_data = create_semantic_scholar_response(papers)
            responses.append(create_mock_response(200, response_data))

        mock_get.side_effect = responses

        with mock_time_sleep() as mock_sleep:
            papers = harvester.search("test query", max_results=150)

        assert len(papers) == 150
        assert mock_get.call_count == 2
        # Should sleep after each batch
        # With 150 results and 100 per batch, we need 2 batches, so 2 sleeps
        assert mock_sleep.call_count == 2
        assert_rate_limiting_applied(mock_sleep, expected_calls=2, min_delay_ms=10)


class TestSemanticScholarEdgeCases:
    """Test edge cases for Semantic Scholar harvester."""

    @pytest.fixture
    def harvester(self, mock_config):
        """Create harvester with minimal config."""
        mock_config.rate_limits = {}  # No rate limits defined
        mock_config.semantic_scholar_key = None  # No API key
        return SemanticScholarHarvester(mock_config)

    def test_default_rate_limits(self, harvester):
        """Test default rate limits when not configured."""
        assert harvester.delay_milliseconds == 100  # Default value

    @patch("requests.get")
    def test_search_empty_response(self, mock_get, harvester):
        """Test search with empty response."""
        response_data = {"data": [], "total": 0}
        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        assert papers == []

    @patch("requests.get")
    def test_search_malformed_response(self, mock_get, harvester):
        """Test search with malformed response structure."""
        response_data = {"unexpected": "structure"}
        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        assert papers == []

    def test_extract_paper_exception_handling(self, harvester):
        """Test extraction when exception occurs."""
        # Create a result that will cause an exception
        result = {
            "title": "Test Paper",
            "authors": "not a list",  # This will cause issues
            "year": "not a number",
        }

        paper = harvester._extract_paper(result)

        # Should handle exception and return None
        assert paper is None

    @patch("requests.get")
    def test_batch_with_partial_results(self, mock_get, harvester):
        """Test handling of partial results in a batch."""
        # Return less than requested
        papers = [
            {
                "paper_id": f"id{i}",
                "title": f"Paper {i}",
                "authors": [{"name": f"Author {i}"}],
                "year": 2024,
            }
            for i in range(50)  # Only 50 results instead of 100
        ]
        response_data = create_semantic_scholar_response(papers)
        mock_response = create_mock_response(200, response_data)
        mock_get.return_value = mock_response

        with mock_time_sleep():
            papers = harvester.search("test query", max_results=100)

        # Should stop after receiving partial results
        assert len(papers) == 50
        assert mock_get.call_count == 1
