"""Tests for the SearchHarvester module."""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.lit_review.harvesters import SearchHarvester


class TestSearchHarvester:
    """Test cases for SearchHarvester class."""
    
    def test_init(self, sample_config):
        """Test SearchHarvester initialization."""
        harvester = SearchHarvester(sample_config)
        assert harvester.config is not None
        assert harvester.google_scholar is not None
        assert harvester.arxiv is not None
        assert harvester.semantic_scholar is not None
        assert harvester.crossref is not None
    
    def test_search_google_scholar(self, sample_config, mock_scholarly):
        """Test Google Scholar search functionality."""
        harvester = SearchHarvester(sample_config)
        
        # Test with valid query
        results_df = harvester.search_google_scholar("LLM wargaming", max_results=10)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert 'title' in results_df.columns
        assert 'authors' in results_df.columns
        assert 'source_db' in results_df.columns
        assert all(results_df['source_db'] == 'google_scholar')
    
    def test_search_arxiv(self, sample_config, mock_arxiv):
        """Test arXiv search functionality."""
        harvester = SearchHarvester(sample_config)
        
        # Test with valid query
        results_df = harvester.search_arxiv("LLM wargaming", max_results=10)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert 'title' in results_df.columns
        assert 'arxiv_id' in results_df.columns
        assert 'pdf_url' in results_df.columns
        assert all(results_df['source_db'] == 'arxiv')
    
    @patch('requests.get')
    def test_search_semantic_scholar(self, mock_get, sample_config):
        """Test Semantic Scholar search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'paperId': '12345',
                    'title': 'Test Paper',
                    'authors': [{'name': 'Test Author'}],
                    'year': 2024,
                    'abstract': 'Test abstract',
                    'venue': 'Test Conference',
                    'citationCount': 10,
                    'externalIds': {'DOI': '10.1234/test'},
                    'url': 'https://test.com',
                    'isOpenAccess': True,
                    'openAccessPdf': {'url': 'https://test.com/pdf'}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        harvester = SearchHarvester(sample_config)
        results_df = harvester.search_semantic_scholar("LLM wargaming", max_results=10)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert all(results_df['source_db'] == 'semantic_scholar')
    
    @patch('requests.get')
    def test_search_crossref(self, mock_get, sample_config):
        """Test Crossref search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': {
                'items': [
                    {
                        'title': ['Test Paper'],
                        'author': [{'given': 'Test', 'family': 'Author'}],
                        'published-print': {'date-parts': [[2024]]},
                        'abstract': 'Test abstract',
                        'container-title': ['Test Journal'],
                        'DOI': '10.1234/test',
                        'URL': 'https://test.com',
                        'is-referenced-by-count': 5
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        harvester = SearchHarvester(sample_config)
        results_df = harvester.search_crossref("LLM wargaming", max_results=10)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert all(results_df['source_db'] == 'crossref')
    
    def test_search_all_sequential(self, sample_config, mock_scholarly, mock_arxiv):
        """Test searching all sources sequentially."""
        with patch.object(SearchHarvester, 'search_semantic_scholar') as mock_ss:
            with patch.object(SearchHarvester, 'search_crossref') as mock_cr:
                # Setup mocks
                mock_ss.return_value = pd.DataFrame({'title': ['SS Paper'], 'source_db': ['semantic_scholar']})
                mock_cr.return_value = pd.DataFrame({'title': ['CR Paper'], 'source_db': ['crossref']})
                
                harvester = SearchHarvester(sample_config)
                results_df = harvester.search_all(parallel=False)
                
                assert isinstance(results_df, pd.DataFrame)
                assert len(results_df) > 0
                assert 'google_scholar' in results_df['source_db'].values
                assert 'arxiv' in results_df['source_db'].values
                assert 'semantic_scholar' in results_df['source_db'].values
                assert 'crossref' in results_df['source_db'].values
    
    def test_search_all_parallel(self, sample_config):
        """Test searching all sources in parallel."""
        with patch.object(SearchHarvester, 'search_google_scholar') as mock_gs:
            with patch.object(SearchHarvester, 'search_arxiv') as mock_ax:
                with patch.object(SearchHarvester, 'search_semantic_scholar') as mock_ss:
                    with patch.object(SearchHarvester, 'search_crossref') as mock_cr:
                        # Setup mocks
                        mock_gs.return_value = pd.DataFrame({'title': ['GS Paper'], 'source_db': ['google_scholar']})
                        mock_ax.return_value = pd.DataFrame({'title': ['AX Paper'], 'source_db': ['arxiv']})
                        mock_ss.return_value = pd.DataFrame({'title': ['SS Paper'], 'source_db': ['semantic_scholar']})
                        mock_cr.return_value = pd.DataFrame({'title': ['CR Paper'], 'source_db': ['crossref']})
                        
                        harvester = SearchHarvester(sample_config)
                        results_df = harvester.search_all(parallel=True)
                        
                        assert isinstance(results_df, pd.DataFrame)
                        assert len(results_df) == 4
    
    def test_search_with_specific_sources(self, sample_config):
        """Test searching specific sources only."""
        with patch.object(SearchHarvester, 'search_google_scholar') as mock_gs:
            with patch.object(SearchHarvester, 'search_arxiv') as mock_ax:
                # Setup mocks
                mock_gs.return_value = pd.DataFrame({'title': ['GS Paper'], 'source_db': ['google_scholar']})
                mock_ax.return_value = pd.DataFrame({'title': ['AX Paper'], 'source_db': ['arxiv']})
                
                harvester = SearchHarvester(sample_config)
                results_df = harvester.search_all(sources=['google_scholar', 'arxiv'])
                
                assert isinstance(results_df, pd.DataFrame)
                assert len(results_df) == 2
                assert set(results_df['source_db'].unique()) == {'google_scholar', 'arxiv'}
    
    def test_search_error_handling(self, sample_config):
        """Test error handling in search methods."""
        harvester = SearchHarvester(sample_config)
        
        # Test with Google Scholar error
        with patch('scholarly.scholarly.search_pubs', side_effect=Exception("Search error")):
            results_df = harvester.search_google_scholar("test query")
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 0
        
        # Test with arXiv error
        with patch('arxiv.Search', side_effect=Exception("Search error")):
            results_df = harvester.search_arxiv("test query")
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 0
    
    def test_empty_query(self, sample_config):
        """Test behavior with empty query."""
        harvester = SearchHarvester(sample_config)
        
        # Should use default preset query
        with patch.object(harvester, 'search_google_scholar') as mock_search:
            mock_search.return_value = pd.DataFrame()
            harvester.search_all(sources=['google_scholar'])
            mock_search.assert_called_once()
            # Check that a query was used (not empty)
            args, kwargs = mock_search.call_args
            assert len(args[0]) > 0  # Query should not be empty