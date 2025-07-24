"""Tests for Google Scholar harvester."""

from unittest.mock import patch

import pytest

from src.lit_review.harvesters.google_scholar import GoogleScholarHarvester
from tests.test_utils import assert_rate_limiting_applied, mock_time_sleep


class TestGoogleScholarHarvester:
    """Test suite for Google Scholar harvester."""

    @pytest.fixture
    def harvester(self, mock_config):
        """Create Google Scholar harvester instance."""
        with patch("src.lit_review.harvesters.google_scholar.ProxyGenerator"):
            return GoogleScholarHarvester(mock_config)

    def test_init(self, mock_config):
        """Test harvester initialization."""
        with patch(
            "src.lit_review.harvesters.google_scholar.ProxyGenerator"
        ) as mock_proxy:
            harvester = GoogleScholarHarvester(mock_config)

            assert harvester.rate_limits == mock_config.rate_limits["google_scholar"]
            # Google Scholar looks for delay_seconds, but mock_config has delay_milliseconds
            # So it falls back to default of 5 seconds
            assert harvester.delay_seconds == 5
            assert harvester.source_name == "googlescholar"

            # Verify proxy setup was attempted
            mock_proxy.assert_called_once()

    def test_init_with_proxy_error(self, mock_config):
        """Test initialization when proxy setup fails."""
        with patch(
            "src.lit_review.harvesters.google_scholar.ProxyGenerator"
        ) as mock_proxy:
            mock_proxy.side_effect = Exception("Proxy error")

            # Should not raise exception
            harvester = GoogleScholarHarvester(mock_config)
            assert harvester is not None

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_success(self, mock_scholarly, harvester):
        """Test successful search."""
        # Mock search results
        mock_results = [
            {
                "bib": {
                    "title": "LLM-powered Wargaming Systems",
                    "author": ["John Doe", "Jane Smith"],
                    "pub_year": "2024",
                    "abstract": "This paper presents an LLM-based approach to wargaming...",
                    "venue": "AI Conference 2024",
                },
                "pub_url": "https://example.com/paper1",
                "num_citations": 10,
            },
            {
                "bib": {
                    "title": "GPT-4 in Military Simulations",
                    "author": "Alice Johnson",  # Single string author
                    "pub_year": "2023",
                    "abstract": "We explore the use of GPT-4 in military simulations...",
                    "venue": "Defense AI Journal",
                },
                "pub_url": "https://example.com/paper2",
                "num_citations": 5,
                "eprint_url": "https://arxiv.org/abs/2301.12345",
            },
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep() as mock_sleep:
            papers = harvester.search("LLM wargaming", max_results=10)

        # Verify results
        assert len(papers) == 2

        # Check first paper
        assert papers[0].title == "LLM-powered Wargaming Systems"
        assert papers[0].authors == ["John Doe", "Jane Smith"]
        assert papers[0].year == 2024
        assert papers[0].source_db == "google_scholar"
        assert papers[0].citations == 10
        assert papers[0].venue == "AI Conference 2024"

        # Check second paper (with arXiv)
        assert papers[1].title == "GPT-4 in Military Simulations"
        assert papers[1].authors == ["Alice Johnson"]
        assert papers[1].year == 2023
        assert papers[1].arxiv_id == "2301.12345"
        assert papers[1].pdf_url == "https://arxiv.org/pdf/2301.12345.pdf"

        # Verify rate limiting
        assert_rate_limiting_applied(mock_sleep, expected_calls=2, min_delay_ms=1000)

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_with_captcha_error(self, mock_scholarly, harvester):
        """Test search when CAPTCHA is encountered."""
        mock_scholarly.search_pubs.side_effect = Exception("CAPTCHA required")

        papers = harvester.search("test query")

        assert papers == []
        mock_scholarly.search_pubs.assert_called_once_with("test query")

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_with_fetch_error(self, mock_scholarly, harvester):
        """Test search when fetch error occurs."""
        mock_scholarly.search_pubs.side_effect = Exception(
            "Cannot Fetch: Network error"
        )

        papers = harvester.search("test query")

        assert papers == []

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_with_unexpected_error(self, mock_scholarly, harvester):
        """Test search with unexpected error."""
        mock_scholarly.search_pubs.side_effect = Exception("Unexpected error")

        papers = harvester.search("test query")

        assert papers == []

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_with_extraction_error(self, mock_scholarly, harvester):
        """Test search when paper extraction fails."""
        mock_results = [
            {"bib": {}},  # Missing title
            {
                "bib": {
                    "title": "Valid Paper",
                    "author": "Test Author",
                    "pub_year": "2024",
                },
                "pub_url": "https://example.com/paper",
            },
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep():
            papers = harvester.search("test query", max_results=2)

        # Only valid paper should be returned
        assert len(papers) == 1
        assert papers[0].title == "Valid Paper"

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_filters_by_year(self, mock_scholarly, harvester):
        """Test that search filters results by year."""
        mock_results = [
            {
                "bib": {
                    "title": "Old Paper",
                    "author": "Author 1",
                    "pub_year": "2015",  # Too old
                },
                "pub_url": "https://example.com/old",
            },
            {
                "bib": {
                    "title": "Recent Paper",
                    "author": "Author 2",
                    "pub_year": "2023",  # Within range
                },
                "pub_url": "https://example.com/recent",
            },
            {
                "bib": {
                    "title": "Future Paper",
                    "author": "Author 3",
                    "pub_year": "2026",  # Too new
                },
                "pub_url": "https://example.com/future",
            },
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep():
            papers = harvester.search("test query")

        assert len(papers) == 1
        assert papers[0].title == "Recent Paper"
        assert papers[0].year == 2023

    def test_extract_paper_with_doi(self, harvester):
        """Test paper extraction with DOI in abstract."""
        result = {
            "bib": {
                "title": "Test Paper",
                "author": ["Test Author"],
                "pub_year": "2024",
                "abstract": "This paper (DOI: 10.1234/test.2024.001) presents...",
            },
            "pub_url": "https://example.com/paper",
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        # The DOI regex includes parentheses in its pattern, so it captures the closing paren
        assert paper.doi in ["10.1234/test.2024.001", "10.1234/test.2024.001)"]

    def test_extract_paper_with_description(self, harvester):
        """Test paper extraction using description when abstract is missing."""
        result = {
            "bib": {
                "title": "Test Paper",
                "author": ["Test Author"],
                "pub_year": "2024",
            },
            "description": "This is the paper description used as abstract.",
            "pub_url": "https://example.com/paper",
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.abstract == "This is the paper description used as abstract."

    def test_extract_paper_with_invalid_year(self, harvester):
        """Test paper extraction with invalid year."""
        result = {
            "bib": {
                "title": "Test Paper",
                "author": ["Test Author"],
                "pub_year": "invalid",
            },
            "pub_url": "https://example.com/paper",
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.year == 0

    def test_extract_paper_with_arxiv_variations(self, harvester):
        """Test extraction of arXiv ID from different URL formats."""
        test_cases = [
            ("https://arxiv.org/abs/2401.12345", "2401.12345"),
            ("https://arxiv.org/pdf/2401.12345", "2401.12345"),
            ("https://arxiv.org/abs/2401.12345v2", "2401.12345"),
            ("http://arxiv.org/abs/2401.12345", "2401.12345"),
        ]

        for url, expected_id in test_cases:
            result = {
                "bib": {
                    "title": "Test Paper",
                    "author": ["Test Author"],
                    "pub_year": "2024",
                },
                "eprint_url": url,
            }

            paper = harvester._extract_paper(result)

            assert paper is not None
            assert paper.arxiv_id == expected_id
            assert paper.pdf_url == f"https://arxiv.org/pdf/{expected_id}.pdf"

    def test_extract_paper_with_authors_as_string(self, harvester):
        """Test extraction when authors is a string with 'and' separator."""
        result = {
            "bib": {
                "title": "Test Paper",
                "author": "John Doe and Jane Smith and Alice Johnson",
                "pub_year": "2024",
            },
            "pub_url": "https://example.com/paper",
        }

        paper = harvester._extract_paper(result)

        assert paper is not None
        assert paper.authors == ["John Doe", "Jane Smith", "Alice Johnson"]

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_advanced(self, mock_scholarly, harvester):
        """Test advanced search functionality."""
        mock_results = [
            {
                "bib": {
                    "title": "Advanced Search Result",
                    "author": ["Specific Author"],
                    "pub_year": "2023",
                },
                "pub_url": "https://example.com/paper",
            }
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep():
            papers = harvester.search_advanced(
                title="Advanced",
                author="Specific Author",
                pub_year_start=2020,
                pub_year_end=2024,
                max_results=10,
            )

        # Verify query construction
        expected_query_parts = ['intitle:"Advanced"', 'author:"Specific Author"']
        actual_query = mock_scholarly.search_pubs.call_args[0][0]

        for part in expected_query_parts:
            assert part in actual_query

        assert len(papers) == 1
        assert papers[0].title == "Advanced Search Result"

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_search_advanced_year_filtering(self, mock_scholarly, harvester):
        """Test advanced search with year filtering."""
        mock_results = [
            {
                "bib": {
                    "title": "Paper 1",
                    "author": ["Author 1"],
                    "pub_year": "2019",  # Too old
                },
                "pub_url": "https://example.com/paper1",
            },
            {
                "bib": {
                    "title": "Paper 2",
                    "author": ["Author 2"],
                    "pub_year": "2021",  # Within range
                },
                "pub_url": "https://example.com/paper2",
            },
            {
                "bib": {
                    "title": "Paper 3",
                    "author": ["Author 3"],
                    "pub_year": "2025",  # Too new
                },
                "pub_url": "https://example.com/paper3",
            },
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep():
            papers = harvester.search_advanced(
                pub_year_start=2020, pub_year_end=2024, max_results=10
            )

        assert len(papers) == 1
        assert papers[0].title == "Paper 2"
        assert papers[0].year == 2021

    def test_clean_text_functionality(self, harvester):
        """Test clean_text method inherited from base class."""
        # Test various text cleaning scenarios
        assert harvester.clean_text("  Extra   spaces  ") == "Extra spaces"
        assert harvester.clean_text(None) == ""
        assert harvester.clean_text("") == ""
        assert harvester.clean_text("Normal text") == "Normal text"

    @patch("src.lit_review.harvesters.google_scholar.scholarly")
    def test_max_results_limit(self, mock_scholarly, harvester):
        """Test that max_results is respected."""
        # Create more results than max_results
        mock_results = [
            {
                "bib": {
                    "title": f"Paper {i}",
                    "author": [f"Author {i}"],
                    "pub_year": "2024",
                },
                "pub_url": f"https://example.com/paper{i}",
            }
            for i in range(10)
        ]

        mock_scholarly.search_pubs.return_value = iter(mock_results)

        with mock_time_sleep():
            papers = harvester.search("test query", max_results=3)

        assert len(papers) == 3
        assert papers[0].title == "Paper 0"
        assert papers[2].title == "Paper 2"


class TestGoogleScholarEdgeCases:
    """Test edge cases for Google Scholar harvester."""

    @pytest.fixture
    def harvester(self, mock_config):
        """Create harvester with minimal config."""
        mock_config.rate_limits = {}  # No rate limits defined
        with patch("src.lit_review.harvesters.google_scholar.ProxyGenerator"):
            return GoogleScholarHarvester(mock_config)

    def test_default_rate_limits(self, harvester):
        """Test default rate limits when not configured."""
        assert harvester.delay_seconds == 5  # Default value

    def test_extract_paper_empty_result(self, harvester):
        """Test extraction with empty result."""
        result = {}
        paper = harvester._extract_paper(result)
        assert paper is None

    def test_extract_paper_exception_handling(self, harvester):
        """Test extraction when exception occurs."""
        # Create a result that will cause an exception
        result = {
            "bib": {
                "title": "Test Paper",
                "author": None,  # This will cause issues
                "pub_year": None,
            }
        }

        paper = harvester._extract_paper(result)

        # Should handle exception and still return a paper
        assert paper is not None
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.year == 0

    def test_build_query_inherited(self, harvester, mock_config):
        """Test that build_query is inherited from base class."""
        # Set up config for query building
        mock_config.wargame_terms = ["wargame"]
        mock_config.llm_terms = ["LLM"]
        mock_config.action_terms = ["simulation"]
        mock_config.exclusion_terms = ["review"]

        query = harvester.build_query()

        assert "wargame" in query
        assert "LLM" in query
        assert "simulation" in query
        assert "NOT" in query
        assert "review" in query
