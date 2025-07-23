"""Tests for the Normalizer module."""
import pytest
import pandas as pd
from pathlib import Path

from src.lit_review.processing import Normalizer


class TestNormalizer:
    """Test cases for Normalizer class."""
    
    def test_init(self, sample_config):
        """Test Normalizer initialization."""
        normalizer = Normalizer(sample_config)
        assert normalizer.config is not None
    
    def test_normalize_basic(self, sample_config, sample_papers_df):
        """Test basic normalization functionality."""
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(sample_papers_df)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert len(normalized_df) == len(sample_papers_df)
        assert 'dedupe_key' in normalized_df.columns
        assert 'title_normalized' in normalized_df.columns
        assert 'screening_id' in normalized_df.columns
    
    def test_duplicate_removal_by_doi(self, sample_config):
        """Test duplicate removal based on DOI."""
        # Create DataFrame with duplicate DOIs
        df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 1 Duplicate', 'Paper 2'],
            'authors': ['Author A', 'Author A', 'Author B'],
            'year': [2024, 2024, 2023],
            'doi': ['10.1234/test', '10.1234/test', '10.5678/test'],
            'source_db': ['google_scholar', 'arxiv', 'crossref']
        })
        
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        # Should keep only 2 papers (duplicates removed)
        assert len(normalized_df) == 2
        assert normalized_df['doi'].nunique() == 2
    
    def test_duplicate_removal_by_title(self, sample_config):
        """Test duplicate removal based on fuzzy title matching."""
        # Create DataFrame with similar titles
        df = pd.DataFrame({
            'title': [
                'Using LLMs in Strategic Wargaming',
                'Using LLMs in Strategic Wargaming: A Study',  # Similar title
                'Completely Different Paper'
            ],
            'authors': ['Author A', 'Author A', 'Author B'],
            'year': [2024, 2024, 2023],
            'doi': ['', '', ''],  # No DOIs
            'source_db': ['google_scholar', 'arxiv', 'crossref']
        })
        
        normalizer = Normalizer(sample_config)
        normalizer.fuzzy_threshold = 85  # Set threshold for testing
        normalized_df = normalizer.normalize(df)
        
        # Should remove the similar title
        assert len(normalized_df) < 3
    
    def test_author_normalization(self, sample_config):
        """Test author name normalization."""
        df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 2'],
            'authors': [
                'Smith, John A.; Doe, Jane B.',
                'Johnson, Alice'
            ],
            'year': [2024, 2023],
            'doi': ['10.1234/test1', '10.1234/test2'],
            'source_db': ['google_scholar', 'arxiv']
        })
        
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        assert 'authors_normalized' in normalized_df.columns
        # Check that authors are properly formatted
        assert ';' in normalized_df.iloc[0]['authors_normalized']
    
    def test_screening_id_generation(self, sample_config, sample_papers_df):
        """Test screening ID generation."""
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(sample_papers_df)
        
        # Check that screening IDs are generated
        assert all(normalized_df['screening_id'].str.startswith('SCREEN_'))
        # Check that IDs are unique
        assert normalized_df['screening_id'].nunique() == len(normalized_df)
        # Check format (SCREEN_XXXX)
        assert all(normalized_df['screening_id'].str.match(r'SCREEN_\d{4}'))
    
    def test_missing_data_handling(self, sample_config):
        """Test handling of missing data."""
        df = pd.DataFrame({
            'title': ['Paper 1', None, 'Paper 3'],
            'authors': ['Author A', 'Author B', None],
            'year': [2024, None, 2023],
            'doi': [None, '10.1234/test', ''],
            'source_db': ['google_scholar', 'arxiv', 'crossref']
        })
        
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        # Should handle missing data gracefully
        assert len(normalized_df) > 0
        # Papers without titles should be removed
        assert normalized_df['title'].notna().all()
    
    def test_empty_dataframe(self, sample_config):
        """Test normalization of empty DataFrame."""
        df = pd.DataFrame()
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert len(normalized_df) == 0
    
    def test_preserve_metadata(self, sample_config):
        """Test that metadata is preserved during normalization."""
        df = pd.DataFrame({
            'title': ['Paper 1'],
            'authors': ['Author A'],
            'year': [2024],
            'doi': ['10.1234/test'],
            'source_db': ['google_scholar'],
            'venue': ['Test Conference'],
            'abstract': ['Test abstract'],
            'url': ['https://test.com'],
            'citations': [10],
            'pdf_url': ['https://test.com/pdf']
        })
        
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        # Check that all original columns are preserved
        for col in df.columns:
            assert col in normalized_df.columns
        
        # Check that values are preserved
        assert normalized_df.iloc[0]['venue'] == 'Test Conference'
        assert normalized_df.iloc[0]['citations'] == 10
    
    def test_duplicate_stats_reporting(self, sample_config, caplog):
        """Test that duplicate statistics are properly reported."""
        # Create DataFrame with duplicates
        df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 1', 'Paper 2', 'Paper 2'],
            'authors': ['Author A', 'Author A', 'Author B', 'Author B'],
            'year': [2024, 2024, 2023, 2023],
            'doi': ['10.1234/test', '10.1234/test', '', ''],
            'source_db': ['google_scholar', 'arxiv', 'crossref', 'semantic_scholar']
        })
        
        normalizer = Normalizer(sample_config)
        with caplog.at_level('INFO'):
            normalized_df = normalizer.normalize(df)
        
        # Check that deduplication stats are logged
        assert 'Removed' in caplog.text or 'duplicate' in caplog.text.lower()
    
    def test_source_priority_in_deduplication(self, sample_config):
        """Test that source priority is respected during deduplication."""
        # Create DataFrame with duplicates from different sources
        df = pd.DataFrame({
            'title': ['Same Paper', 'Same Paper'],
            'authors': ['Author A', 'Author A'],
            'year': [2024, 2024],
            'doi': ['10.1234/test', '10.1234/test'],
            'source_db': ['arxiv', 'google_scholar'],  # arXiv should be preferred
            'pdf_url': ['https://arxiv.org/pdf', '']
        })
        
        normalizer = Normalizer(sample_config)
        normalized_df = normalizer.normalize(df)
        
        # Should keep only one paper
        assert len(normalized_df) == 1
        # Should prefer the one with PDF URL (arxiv)
        assert normalized_df.iloc[0]['source_db'] == 'arxiv'
        assert normalized_df.iloc[0]['pdf_url'] == 'https://arxiv.org/pdf'