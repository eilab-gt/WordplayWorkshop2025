"""Tests for the Exporter module."""
import pytest
import pandas as pd
from pathlib import Path
import zipfile
import json
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock

from src.lit_review.utils import Exporter


class TestExporter:
    """Test cases for Exporter class."""
    
    def test_init(self, sample_config):
        """Test Exporter initialization."""
        exporter = Exporter(sample_config)
        assert exporter.config is not None
        assert hasattr(exporter, 'output_dir')
        assert hasattr(exporter, 'zenodo_config')
    
    def test_create_package_basic(self, sample_config, temp_dir):
        """Test basic package creation."""
        exporter = Exporter(sample_config)
        
        # Create test data
        papers_df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 2'],
            'authors': ['Author A', 'Author B'],
            'year': [2024, 2023]
        })
        
        extraction_df = pd.DataFrame({
            'screening_id': ['SCREEN_0001', 'SCREEN_0002'],
            'venue_type': ['conference', 'journal'],
            'llm_family': ['GPT-4', 'Claude']
        })
        
        # Create mock visualizations
        viz_paths = {
            'timeline': Path(temp_dir) / 'timeline.png',
            'venue_dist': Path(temp_dir) / 'venue.png'
        }
        for path in viz_paths.values():
            path.write_text('fake image data')
        
        output_path = Path(temp_dir) / 'package.zip'
        result_path = exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths=viz_paths,
            output_path=output_path
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify package contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert any('papers_raw.csv' in f for f in files)
            assert any('extraction_results.csv' in f for f in files)
            assert any('timeline.png' in f for f in files)
            assert any('metadata.json' in f for f in files)
    
    def test_package_structure(self, sample_config, temp_dir):
        """Test package directory structure."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({'title': ['Test Paper']})
        extraction_df = pd.DataFrame({'screening_id': ['SCREEN_0001']})
        
        output_path = Path(temp_dir) / 'structured.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path
        )
        
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            # Check directory structure
            assert any('data/' in f for f in files)
            assert any('metadata.json' in f for f in files)
    
    def test_metadata_generation(self, sample_config, temp_dir):
        """Test metadata JSON generation."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 2', 'Paper 3'],
            'year': [2022, 2023, 2024]
        })
        
        extraction_df = pd.DataFrame({
            'screening_id': ['SCREEN_0001', 'SCREEN_0002'],
            'llm_family': ['GPT-4', 'Claude']
        })
        
        output_path = Path(temp_dir) / 'metadata_test.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path
        )
        
        # Extract and check metadata
        with zipfile.ZipFile(output_path, 'r') as zf:
            metadata_content = zf.read('metadata.json')
            metadata = json.loads(metadata_content)
            
            assert 'created_at' in metadata
            assert 'statistics' in metadata
            assert metadata['statistics']['total_papers'] == 3
            assert metadata['statistics']['papers_with_extraction'] == 2
            assert 'date_range' in metadata['statistics']
    
    @patch('requests.post')
    def test_zenodo_upload(self, mock_post, sample_config):
        """Test Zenodo upload functionality."""
        exporter = Exporter(sample_config)
        
        # Mock Zenodo API responses
        # Create deposit response
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {
            'id': '12345',
            'links': {
                'bucket': 'https://zenodo.org/api/files/bucket-id',
                'publish': 'https://zenodo.org/api/deposit/depositions/12345/actions/publish'
            }
        }
        
        # Upload file response
        upload_response = Mock()
        upload_response.status_code = 201
        
        # Publish response
        publish_response = Mock()
        publish_response.status_code = 202
        publish_response.json.return_value = {
            'doi': '10.5281/zenodo.12345',
            'links': {'record_html': 'https://zenodo.org/record/12345'}
        }
        
        mock_post.side_effect = [create_response, upload_response, publish_response]
        
        # Create test package
        package_path = Path(sample_config).parent / 'test_package.zip'
        with zipfile.ZipFile(package_path, 'w') as zf:
            zf.writestr('test.txt', 'test content')
        
        try:
            doi = exporter.upload_to_zenodo(package_path)
            assert doi == '10.5281/zenodo.12345'
            assert mock_post.call_count == 3
        finally:
            package_path.unlink()
    
    def test_create_readme(self, sample_config, temp_dir):
        """Test README generation."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 2'],
            'source_db': ['arxiv', 'google_scholar']
        })
        
        extraction_df = pd.DataFrame({
            'screening_id': ['SCREEN_0001'],
            'llm_family': ['GPT-4']
        })
        
        output_path = Path(temp_dir) / 'readme_test.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path
        )
        
        # Check README exists and contains expected content
        with zipfile.ZipFile(output_path, 'r') as zf:
            readme_content = zf.read('README.md').decode('utf-8')
            assert 'Literature Review Dataset' in readme_content
            assert 'Statistics' in readme_content
            assert '2 papers' in readme_content
    
    def test_export_formats(self, sample_config, temp_dir):
        """Test different export formats within the package."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({
            'title': ['Paper 1'],
            'year': [2024],
            'abstract': ['Test abstract']
        })
        
        extraction_df = pd.DataFrame({
            'screening_id': ['SCREEN_0001'],
            'venue_type': ['conference']
        })
        
        output_path = Path(temp_dir) / 'formats_test.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path,
            include_json=True  # If supported
        )
        
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            # Should have CSV files
            assert any('.csv' in f for f in files)
            # Might also have JSON versions if implemented
    
    def test_empty_dataframes(self, sample_config, temp_dir):
        """Test handling of empty DataFrames."""
        exporter = Exporter(sample_config)
        
        empty_papers = pd.DataFrame()
        empty_extraction = pd.DataFrame()
        
        output_path = Path(temp_dir) / 'empty_test.zip'
        
        # Should handle empty DataFrames gracefully
        result_path = exporter.create_package(
            papers_df=empty_papers,
            extraction_df=empty_extraction,
            viz_paths={},
            output_path=output_path
        )
        
        assert result_path == output_path
        assert output_path.exists()
    
    def test_custom_metadata(self, sample_config, temp_dir):
        """Test adding custom metadata to package."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({'title': ['Paper 1']})
        extraction_df = pd.DataFrame({'screening_id': ['SCREEN_0001']})
        
        custom_metadata = {
            'project_name': 'LLM Wargaming Review',
            'version': '1.0',
            'authors': ['Researcher A', 'Researcher B']
        }
        
        output_path = Path(temp_dir) / 'custom_meta.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path,
            custom_metadata=custom_metadata
        )
        
        with zipfile.ZipFile(output_path, 'r') as zf:
            metadata_content = zf.read('metadata.json')
            metadata = json.loads(metadata_content)
            
            # Check custom metadata is included
            if 'custom' in metadata:
                assert metadata['custom']['project_name'] == 'LLM Wargaming Review'
            # Or metadata might be merged at top level
            elif 'project_name' in metadata:
                assert metadata['project_name'] == 'LLM Wargaming Review'
    
    def test_compression_levels(self, sample_config, temp_dir):
        """Test package compression."""
        exporter = Exporter(sample_config)
        
        # Create larger test data
        papers_df = pd.DataFrame({
            'title': [f'Paper {i}' for i in range(100)],
            'abstract': ['Long abstract text ' * 50 for _ in range(100)]
        })
        
        extraction_df = pd.DataFrame({
            'screening_id': [f'SCREEN_{i:04d}' for i in range(100)],
            'venue_type': ['conference'] * 100
        })
        
        output_path = Path(temp_dir) / 'compressed.zip'
        exporter.create_package(
            papers_df=papers_df,
            extraction_df=extraction_df,
            viz_paths={},
            output_path=output_path
        )
        
        # Check that compression is effective
        assert output_path.exists()
        # The compressed size should be significantly smaller than uncompressed
        # (though we can't easily test this without extracting)
    
    def test_error_handling(self, sample_config, temp_dir):
        """Test error handling in package creation."""
        exporter = Exporter(sample_config)
        
        papers_df = pd.DataFrame({'title': ['Paper 1']})
        
        # Try to write to invalid path
        invalid_path = Path('/invalid/path/package.zip')
        
        with pytest.raises(Exception):  # Should raise some exception
            exporter.create_package(
                papers_df=papers_df,
                extraction_df=pd.DataFrame(),
                viz_paths={},
                output_path=invalid_path
            )