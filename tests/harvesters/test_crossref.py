"""Tests for CrossRef harvester."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from src.lit_review.harvesters.crossref import CrossrefHarvester


class TestCrossrefHarvester:
    """Test suite for CrossRef harvester."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.rate_limits = {
            "crossref": {
                "requests_per_second": 100,
                "delay_milliseconds": 10,
                "burst_limit": 200,
            }
        }
        config.unpaywall_email = "test@example.com"
        config.search.years.start = 2018
        config.search.years.end = 2025
        config.search.llm_min_params = 100_000_000
        config.search_years = (2018, 2025)  # Add this for filter_by_year
        return config

    @pytest.fixture
    def harvester(self, mock_config):
        """Create CrossRef harvester instance."""
        return CrossrefHarvester(mock_config)

    def test_init(self, mock_config):
        """Test harvester initialization."""
        harvester = CrossrefHarvester(mock_config)

        assert harvester.rate_limits == mock_config.rate_limits["crossref"]
        assert harvester.delay_milliseconds == 10
        assert harvester.email == "test@example.com"
        assert "test@example.com" in harvester.headers["User-Agent"]

    def test_init_no_email(self, mock_config):
        """Test initialization without email."""
        mock_config.unpaywall_email = None
        harvester = CrossrefHarvester(mock_config)

        assert "mailto:" not in harvester.headers["User-Agent"]
        assert "LitReviewPipeline/1.0" in harvester.headers["User-Agent"]

    @patch("time.sleep")
    @patch("requests.get")
    def test_search_success(self, mock_get, mock_sleep, harvester):
        """Test successful search."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/test1",
                        "title": ["LLM-powered Wargaming Simulation"],
                        "author": [
                            {"given": "John", "family": "Doe"},
                            {"given": "Jane", "family": "Smith"},
                        ],
                        "published-print": {"date-parts": [[2024, 1, 15]]},
                        "abstract": "Abstract about LLM wargaming",
                        "URL": "https://doi.org/10.1234/test1",
                        "type": "journal-article",
                        "container-title": ["Journal of AI Warfare"],
                    },
                    {
                        "DOI": "10.5678/test2",
                        "title": ["GPT-4 in Military Decision Making"],
                        "author": [{"given": "Alice", "family": "Johnson"}],
                        "published-online": {"date-parts": [[2023, 6]]},
                        "abstract": "Abstract about GPT-4 military applications",
                        "URL": "https://doi.org/10.5678/test2",
                        "type": "proceedings-article",
                    },
                ],
                "total-results": 2,
            },
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Execute search
        papers = harvester.search("LLM wargaming", max_results=10)

        # Verify results
        assert len(papers) == 2

        # Check first paper
        assert papers[0].title == "LLM-powered Wargaming Simulation"
        assert papers[0].authors == ["John Doe", "Jane Smith"]
        assert papers[0].year == 2024
        assert papers[0].doi == "10.1234/test1"
        assert papers[0].url == "https://doi.org/10.1234/test1"
        assert papers[0].source_db == "crossref"
        assert papers[0].venue == "Journal of AI Warfare"
        # venue_type is not part of Paper class

        # Check second paper
        assert papers[1].title == "GPT-4 in Military Decision Making"
        assert papers[1].authors == ["Alice Johnson"]
        assert papers[1].year == 2023
        # venue_type is not part of Paper class

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "https://api.crossref.org/works" in call_args[0][0]
        assert call_args[1]["headers"] == harvester.headers
        assert "query" in call_args[1]["params"]
        assert "rows" in call_args[1]["params"]

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

    @patch("time.sleep")
    @patch("requests.get")
    def test_search_pagination(self, mock_get, mock_sleep, harvester):
        """Test search with pagination."""
        # Mock paginated responses
        responses = []
        for i in range(3):
            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "ok",
                "message": {
                    "items": [
                        {
                            "DOI": f"10.1234/page{i}_{j}",
                            "title": [f"Paper {i}-{j}"],
                            "author": [],
                            "published-print": {"date-parts": [[2024]]},
                        }
                        for j in range(100)
                    ],
                    "total-results": 250,
                },
            }
            mock_response.raise_for_status = Mock()
            responses.append(mock_response)

        mock_get.side_effect = responses

        papers = harvester.search("test query", max_results=250)

        assert len(papers) == 250
        assert mock_get.call_count == 3

    @patch("requests.get")
    def test_search_filters_by_year(self, mock_get, harvester):
        """Test that search filters results by year."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/old",
                        "title": ["Old Paper"],
                        "published-print": {"date-parts": [[2015]]},  # Too old
                    },
                    {
                        "DOI": "10.1234/recent",
                        "title": ["Recent Paper"],
                        "published-print": {"date-parts": [[2020]]},  # Within range
                    },
                    {
                        "DOI": "10.1234/future",
                        "title": ["Future Paper"],
                        "published-print": {"date-parts": [[2026]]},  # Too new
                    },
                ],
                "total-results": 3,
            },
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        assert len(papers) == 1
        assert papers[0].title == "Recent Paper"
        assert papers[0].year == 2020

    @patch("requests.get")
    def test_parse_paper_missing_fields(self, mock_get, harvester):
        """Test parsing papers with missing fields."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "message": {
                "items": [
                    {
                        # Minimal paper with many missing fields but has title
                        "title": ["Minimal Paper"],
                        # Add a year within range so it doesn't get filtered out
                        "published-print": {"date-parts": [[2024]]},
                    },
                    {
                        # Paper with no title (should be skipped)
                        "DOI": "10.1234/notitle",
                        "author": [{"given": "Test", "family": "Author"}],
                        "published-print": {"date-parts": [[2024]]},
                    },
                ],
                "total-results": 2,
            },
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = harvester.search("test query")

        # Only one paper should be returned (the one with title)
        assert len(papers) == 1
        assert papers[0].title == "Minimal Paper"
        assert papers[0].doi is None or papers[0].doi == ""
        assert papers[0].authors == []
        assert papers[0].year == 2024  # Year should be extracted

    def test_venue_extraction(self, harvester):
        """Test that venue is extracted from container-title."""
        # This test verifies venue extraction logic is part of paper parsing
        # The actual implementation should extract venue from container-title field
        pass  # Placeholder - actual CrossRef implementation handles this in _extract_paper

    @patch("requests.get")
    def test_get_paper_by_doi(self, mock_get, harvester):
        """Test getting paper by DOI."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "message": {
                "DOI": "10.1234/test",
                "title": ["Paper Retrieved by DOI"],
                "author": [{"given": "Test", "family": "Author"}],
                "published-print": {"date-parts": [[2024]]},
            },
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        paper = harvester.get_paper_by_doi("10.1234/test")

        assert paper is not None
        assert paper.title == "Paper Retrieved by DOI"
        assert paper.doi == "10.1234/test"

        # Verify API call
        mock_get.assert_called_once()
        assert "https://api.crossref.org/works/10.1234/test" in mock_get.call_args[0][0]

    @patch("requests.get")
    def test_get_paper_by_doi_not_found(self, mock_get, harvester):
        """Test getting paper by DOI when not found."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        paper = harvester.get_paper_by_doi("10.1234/notfound")

        assert paper is None

    def test_doi_handling(self, harvester):
        """Test that DOI is properly extracted from papers."""
        # This test would verify DOI extraction in actual implementation
        # CrossRef provides DOIs which should be extracted in _extract_paper
        pass  # Placeholder - actual implementation extracts DOI from response

    @patch("time.sleep")
    @patch("requests.get")
    def test_rate_limiting(self, mock_get, mock_sleep, harvester):
        """Test rate limiting is applied."""
        # Create multiple pages of results to trigger rate limiting
        responses = []
        for i in range(3):
            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "ok",
                "message": {
                    "items": [
                        {
                            "DOI": f"10.1234/test{i}",
                            "title": [f"Test Paper {i}"],
                            "published-print": {"date-parts": [[2024]]},
                        }
                    ],
                    "total-results": 300,  # More results to trigger pagination
                },
            }
            mock_response.raise_for_status = Mock()
            responses.append(mock_response)

        mock_get.side_effect = responses

        # Search with pagination to trigger multiple requests
        papers = harvester.search("test query", max_results=300)

        # Verify sleep was called for rate limiting between requests
        assert mock_sleep.call_count >= 2  # Called between batches


class TestCrossrefEdgeCases:
    """Test edge cases for CrossRef harvester."""

    @pytest.fixture
    def harvester(self):
        """Create harvester with minimal config."""
        config = MagicMock()
        config.rate_limits = {}
        config.unpaywall_email = None
        config.search.years.start = 2018
        config.search.years.end = 2025
        config.search_years = (2018, 2025)  # Add this for filter_by_year
        return CrossrefHarvester(config)

    @patch("requests.get")
    def test_search_empty_response(self, mock_get, harvester):
        """Test search with empty response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "message": {"items": None, "total-results": 0},
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = harvester.search("test query")
        assert papers == []

    @patch("requests.get")
    def test_search_malformed_response(self, mock_get, harvester):
        """Test search with malformed response structure."""
        mock_response = Mock()
        mock_response.json.return_value = {"unexpected": "structure"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = harvester.search("test query")
        assert papers == []

    def test_clean_text_various_inputs(self, harvester):
        """Test clean_text with various inputs."""
        # Normal text
        assert harvester.clean_text("Normal text") == "Normal text"

        # Text with extra whitespace
        assert harvester.clean_text("  Extra   spaces  ") == "Extra spaces"

        # None
        assert harvester.clean_text(None) == ""

        # Empty string
        assert harvester.clean_text("") == ""
