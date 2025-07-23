"""Tests for the LLMExtractor module."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import PyPDF2

from src.lit_review.extraction import LLMExtractor


class TestLLMExtractor:
    """Test cases for LLMExtractor class."""
    
    def test_init(self, sample_config, mock_openai_client):
        """Test LLMExtractor initialization."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            assert extractor.config is not None
            assert extractor.client is not None
            assert extractor.model == 'gpt-4'
            assert extractor.temperature == 0.3
    
    def test_extract_pdf_text(self, sample_config, temp_dir, mock_openai_client):
        """Test PDF text extraction."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Create a mock PDF file
            pdf_path = Path(temp_dir) / 'test.pdf'
            
            # Mock PyPDF2 reader
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = "This is a test PDF about LLM wargaming."
                mock_reader.return_value.pages = [mock_page]
                
                text = extractor._extract_pdf_text(str(pdf_path))
                
                assert text is not None
                assert "LLM wargaming" in text
    
    def test_llm_extraction(self, sample_config, mock_openai_client):
        """Test LLM-based information extraction."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Test context
            context = "This paper presents a matrix wargame using GPT-4 as a player agent."
            
            result = extractor._llm_extract(context)
            
            assert isinstance(result, dict)
            assert 'venue_type' in result
            assert 'game_type' in result
            assert result['game_type'] == 'matrix'
            assert result['llm_family'] == 'GPT-4'
            assert result['llm_role'] == 'player'
    
    def test_extract_single_paper(self, sample_config, temp_dir, mock_openai_client):
        """Test extraction for a single paper."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Create test paper
            paper = pd.Series({
                'screening_id': 'SCREEN_0001',
                'title': 'Test Paper',
                'abstract': 'This paper explores LLM wargaming.',
                'pdf_path': str(Path(temp_dir) / 'test.pdf')
            })
            
            # Mock PDF text extraction
            with patch.object(extractor, '_extract_pdf_text', return_value="Full paper text about GPT-4 in wargaming"):
                result = extractor._extract_single(paper)
                
                assert isinstance(result, dict)
                assert result['screening_id'] == 'SCREEN_0001'
                assert result['extraction_status'] == 'success'
                assert 'venue_type' in result
                assert 'llm_family' in result
    
    def test_extract_batch(self, sample_config, sample_screening_df, mock_openai_client):
        """Test batch extraction."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Filter to papers with PDFs
            df_with_pdfs = sample_screening_df[sample_screening_df['pdf_path'].notna()].copy()
            
            # Mock PDF text extraction
            with patch.object(extractor, '_extract_pdf_text', return_value="Paper text"):
                with patch.object(extractor, '_llm_extract') as mock_llm:
                    mock_llm.return_value = {
                        'venue_type': 'conference',
                        'game_type': 'matrix',
                        'open_ended': 'yes',
                        'quantitative': 'yes',
                        'llm_family': 'GPT-4',
                        'llm_role': 'player'
                    }
                    
                    results_df = extractor.extract_batch(df_with_pdfs, max_workers=2)
                    
                    assert isinstance(results_df, pd.DataFrame)
                    assert len(results_df) == len(df_with_pdfs)
                    assert 'extraction_status' in results_df.columns
                    assert 'venue_type' in results_df.columns
    
    def test_extraction_error_handling(self, sample_config, mock_openai_client):
        """Test error handling during extraction."""
        # Mock client that raises an error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('openai.OpenAI', return_value=mock_client):
            extractor = LLMExtractor(sample_config)
            
            paper = pd.Series({
                'screening_id': 'SCREEN_0001',
                'title': 'Test Paper',
                'abstract': 'Test abstract',
                'pdf_path': 'fake_path.pdf'
            })
            
            with patch.object(extractor, '_extract_pdf_text', return_value="Text"):
                result = extractor._extract_single(paper)
                
                assert result['extraction_status'] == 'error'
                assert result['screening_id'] == 'SCREEN_0001'
    
    def test_missing_pdf_handling(self, sample_config, mock_openai_client):
        """Test handling of papers without PDFs."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Paper without PDF path
            paper = pd.Series({
                'screening_id': 'SCREEN_0001',
                'title': 'Test Paper',
                'abstract': 'Test abstract',
                'pdf_path': None
            })
            
            result = extractor._extract_single(paper)
            
            # Should still attempt extraction using abstract
            assert result['screening_id'] == 'SCREEN_0001'
            assert 'venue_type' in result  # Should have attempted extraction
    
    def test_awscale_calculation(self, sample_config, mock_openai_client):
        """Test AWScale calculation."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Test different combinations
            # Open-ended + Quantitative = middle of scale (3)
            result1 = extractor._calculate_awscale('yes', 'yes')
            assert result1 == 3
            
            # Open-ended only = more wild (4-5)
            result2 = extractor._calculate_awscale('yes', 'no')
            assert result2 >= 4
            
            # Quantitative only = more analytic (1-2)
            result3 = extractor._calculate_awscale('no', 'yes')
            assert result3 <= 2
            
            # Neither = undefined (3)
            result4 = extractor._calculate_awscale('no', 'no')
            assert result4 == 3
    
    def test_json_parsing_fallback(self, sample_config):
        """Test JSON parsing with fallback for malformed responses."""
        # Mock client with malformed JSON response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='Not valid JSON but contains venue_type: conference'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            extractor = LLMExtractor(sample_config)
            
            result = extractor._llm_extract("Test context")
            
            # Should return None or handle gracefully
            assert result is None or isinstance(result, dict)
    
    def test_confidence_scoring(self, sample_config, mock_openai_client):
        """Test extraction confidence scoring."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Create papers with different amounts of text
            paper_full = pd.Series({
                'screening_id': 'SCREEN_0001',
                'title': 'Full Paper',
                'abstract': 'Detailed abstract ' * 50,  # Long abstract
                'pdf_path': 'test.pdf'
            })
            
            paper_minimal = pd.Series({
                'screening_id': 'SCREEN_0002',
                'title': 'Minimal Paper',
                'abstract': 'Short.',
                'pdf_path': None
            })
            
            with patch.object(extractor, '_extract_pdf_text', return_value="Full text " * 100):
                result_full = extractor._extract_single(paper_full)
                result_minimal = extractor._extract_single(paper_minimal)
                
                # Full paper should have higher confidence
                assert result_full.get('extraction_confidence', 0) > result_minimal.get('extraction_confidence', 0)
    
    def test_parallel_extraction(self, sample_config, mock_openai_client):
        """Test parallel extraction performance."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            extractor = LLMExtractor(sample_config)
            
            # Create multiple papers
            papers = []
            for i in range(5):
                papers.append({
                    'screening_id': f'SCREEN_{i:04d}',
                    'title': f'Paper {i}',
                    'abstract': f'Abstract for paper {i}',
                    'pdf_path': f'paper_{i}.pdf'
                })
            
            df = pd.DataFrame(papers)
            
            with patch.object(extractor, '_extract_pdf_text', return_value="Text"):
                # Test with different worker counts
                results_df = extractor.extract_batch(df, max_workers=3)
                
                assert len(results_df) == 5
                assert all(results_df['extraction_status'] == 'success')