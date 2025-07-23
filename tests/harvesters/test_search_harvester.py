"""Tests for the SearchHarvester module."""

from unittest.mock import Mock, patch

import pandas as pd

from src.lit_review.harvesters import SearchHarvester


class TestSearchHarvester:
    """Test cases for SearchHarvester class."""

    def test_init(self, sample_config):
        """Test SearchHarvester initialization."""
        harvester = SearchHarvester(sample_config)
        assert harvester.config is not None
        assert "google_scholar" in harvester.harvesters
        assert "arxiv" in harvester.harvesters
        assert "semantic_scholar" in harvester.harvesters
        assert "crossref" in harvester.harvesters

    @patch("time.sleep", return_value=None)  # Mock time.sleep to avoid delays
    def test_search_google_scholar(self, mock_sleep, sample_config, mock_scholarly):
        """Test Google Scholar search functionality."""
        harvester = SearchHarvester(sample_config)

        # Test with valid query
        results_df = harvester.search_all(sources=["google_scholar"], max_results_per_source=10)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert "title" in results_df.columns
        assert "authors" in results_df.columns
        assert "source_db" in results_df.columns
        assert all(results_df["source_db"] == "google_scholar")

    @patch("time.sleep", return_value=None)  # Mock time.sleep to avoid delays
    def test_search_arxiv(self, mock_sleep, sample_config, mock_arxiv):
        """Test arXiv search functionality."""
        harvester = SearchHarvester(sample_config)

        # Test with valid query
        results_df = harvester.search_all(sources=["arxiv"], max_results_per_source=10)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert "title" in results_df.columns
        assert "arxiv_id" in results_df.columns
        assert "pdf_url" in results_df.columns
        assert all(results_df["source_db"] == "arxiv")

    @patch("requests.get")
    def test_search_semantic_scholar(self, mock_get, sample_config):
        """Test Semantic Scholar search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "paperId": "12345",
                    "title": "Test Paper",
                    "authors": [{"name": "Test Author"}],
                    "year": 2024,
                    "abstract": "Test abstract",
                    "venue": "Test Conference",
                    "citationCount": 10,
                    "externalIds": {"DOI": "10.1234/test"},
                    "url": "https://test.com",
                    "isOpenAccess": True,
                    "openAccessPdf": {"url": "https://test.com/pdf"},
                }
            ]
        }
        mock_get.return_value = mock_response

        harvester = SearchHarvester(sample_config)
        results_df = harvester.search_all(sources=["semantic_scholar"], max_results_per_source=10)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert all(results_df["source_db"] == "semantic_scholar")

    @patch("requests.get")
    def test_search_crossref(self, mock_get, sample_config):
        """Test Crossref search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Test Paper"],
                        "author": [{"given": "Test", "family": "Author"}],
                        "published-print": {"date-parts": [[2024]]},
                        "abstract": "Test abstract",
                        "container-title": ["Test Journal"],
                        "DOI": "10.1234/test",
                        "URL": "https://test.com",
                        "is-referenced-by-count": 5,
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        harvester = SearchHarvester(sample_config)
        results_df = harvester.search_all(sources=["crossref"], max_results_per_source=10)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert all(results_df["source_db"] == "crossref")

    @patch("time.sleep", return_value=None)  # Mock time.sleep to avoid delays
    def test_search_all_sequential(self, mock_sleep, sample_config, mock_scholarly, mock_arxiv):
        """Test searching all sources sequentially."""
        # Mock the semantic scholar and crossref responses
        with patch("requests.get") as mock_get:
            # Setup mock responses for semantic scholar and crossref
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"paperId": "123", "title": "SS Paper", "authors": [], "year": 2024}]
            }
            mock_get.return_value = mock_response

            harvester = SearchHarvester(sample_config)
            results_df = harvester.search_all(parallel=False)

            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) > 0
            # Check that we got results from different sources
            unique_sources = results_df["source_db"].unique()
            assert len(unique_sources) >= 2  # At least 2 sources returned results

    @patch("time.sleep", return_value=None)  # Mock time.sleep to avoid delays
    def test_search_all_parallel(self, mock_sleep, sample_config, mock_scholarly, mock_arxiv):
        """Test searching all sources in parallel."""
        # Mock the semantic scholar and crossref responses
        with patch("requests.get") as mock_get:
            # Setup mock responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"paperId": "123", "title": "SS Paper", "authors": [], "year": 2024}]
            }
            mock_get.return_value = mock_response

            harvester = SearchHarvester(sample_config)
            results_df = harvester.search_all(parallel=True)

            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) > 0

    @patch("time.sleep", return_value=None)  # Mock time.sleep to avoid delays
    def test_search_with_specific_sources(self, mock_sleep, sample_config, mock_scholarly, mock_arxiv):
        """Test searching specific sources only."""
        harvester = SearchHarvester(sample_config)
        results_df = harvester.search_all(sources=["google_scholar", "arxiv"])

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        # Verify only specified sources are in results
        unique_sources = set(results_df["source_db"].unique())
        assert unique_sources.issubset({"google_scholar", "arxiv"})

    def test_search_error_handling(self, sample_config):
        """Test error handling in search methods."""
        harvester = SearchHarvester(sample_config)

        # Test with Google Scholar error
        with patch(
            "scholarly.scholarly.search_pubs", side_effect=Exception("Search error")
        ):
            results_df = harvester.search_all(sources=["google_scholar"])
            assert isinstance(results_df, pd.DataFrame)
            # Should return empty dataframe on error

        # Test with arXiv error
        with patch("arxiv.Search", side_effect=Exception("Search error")):
            results_df = harvester.search_all(sources=["arxiv"])
            assert isinstance(results_df, pd.DataFrame)
            # Should return empty dataframe on error

    def test_empty_query(self, sample_config):
        """Test behavior with empty query."""
        harvester = SearchHarvester(sample_config)

        # Should use default preset query from config
        with patch.object(harvester.harvesters["google_scholar"], "search") as mock_search:
            mock_search.return_value = []
            harvester.search_all(sources=["google_scholar"])
            mock_search.assert_called_once()
            # Check that a query was used (not empty)
            args, kwargs = mock_search.call_args
            assert len(args[0]) > 0  # Query should not be empty
