"""Tests for the CLI interface."""
import pytest
from click.testing import CliRunner
from pathlib import Path
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
import json

from run import cli


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Literature review pipeline CLI' in result.output
        assert 'harvest' in result.output
        assert 'extract' in result.output
    
    @patch('run.SearchHarvester')
    def test_harvest_command(self, mock_harvester, sample_config):
        """Test harvest command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.search_all.return_value = pd.DataFrame({
            'title': ['Test Paper'],
            'authors': ['Test Author'],
            'year': [2024]
        })
        mock_harvester.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create config file
            Path('config.yaml').write_text('search:\n  queries:\n    preset1: "test"')
            
            result = runner.invoke(cli, ['harvest', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Harvesting papers' in result.output
            assert 'Found 1 papers' in result.output
    
    @patch('run.SearchHarvester')
    @patch('run.Normalizer')
    @patch('run.ScreenUI')
    def test_prepare_screen_command(self, mock_screen, mock_normalizer, mock_harvester, sample_config):
        """Test prepare-screen command."""
        # Setup mocks
        test_df = pd.DataFrame({
            'title': ['Test Paper'],
            'screening_id': ['SCREEN_0001']
        })
        
        mock_normalizer.return_value.normalize.return_value = test_df
        mock_screen.return_value.generate_sheet.return_value = 'screening.xlsx'
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            test_df.to_csv('papers.csv', index=False)
            Path('config.yaml').write_text('paths:\n  output_dir: outputs')
            
            result = runner.invoke(cli, ['prepare-screen', '--input', 'papers.csv', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Preparing screening sheet' in result.output
    
    @patch('run.LLMExtractor')
    def test_extract_command(self, mock_extractor):
        """Test extract command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.extract_batch.return_value = pd.DataFrame({
            'screening_id': ['SCREEN_0001'],
            'venue_type': ['conference'],
            'extraction_status': ['success']
        })
        mock_extractor.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            pd.DataFrame({
                'screening_id': ['SCREEN_0001'],
                'title': ['Test Paper'],
                'pdf_path': ['test.pdf']
            }).to_csv('screening.csv', index=False)
            
            Path('config.yaml').write_text('api_keys:\n  openai: test-key')
            
            result = runner.invoke(cli, ['extract', '--input', 'screening.csv', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Extracting information' in result.output
    
    @patch('run.Visualizer')
    def test_visualise_command(self, mock_visualizer):
        """Test visualise command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate_all_charts.return_value = {
            'timeline': 'timeline.png',
            'venue_dist': 'venue.png'
        }
        mock_visualizer.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            pd.DataFrame({
                'year': [2023, 2024],
                'venue_type': ['conference', 'journal']
            }).to_csv('extraction.csv', index=False)
            
            Path('config.yaml').write_text('viz:\n  charts:\n    timeline:\n      enabled: true')
            
            result = runner.invoke(cli, ['visualise', '--input', 'extraction.csv', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Creating visualizations' in result.output
    
    @patch('run.Exporter')
    def test_export_command(self, mock_exporter):
        """Test export command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.create_package.return_value = 'package.zip'
        mock_exporter.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input files
            pd.DataFrame({'title': ['Paper 1']}).to_csv('papers.csv', index=False)
            pd.DataFrame({'screening_id': ['SCREEN_0001']}).to_csv('extraction.csv', index=False)
            
            Path('config.yaml').write_text('export:\n  zenodo:\n    enabled: false')
            
            result = runner.invoke(cli, ['export', 
                                         '--papers', 'papers.csv',
                                         '--extraction', 'extraction.csv',
                                         '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Creating export package' in result.output
    
    def test_test_command(self):
        """Test the test command."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yaml').write_text('search:\n  queries:\n    preset1: "test"')
            
            result = runner.invoke(cli, ['test', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Testing pipeline components' in result.output
            assert 'Config' in result.output
    
    @patch('run.LoggingDatabase')
    def test_status_command(self, mock_db):
        """Test status command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_summary.return_value = {
            'total_logs': 100,
            'by_level': {'INFO': 80, 'WARNING': 15, 'ERROR': 5}
        }
        mock_instance.query_logs.return_value = [
            {'timestamp': '2024-01-01T10:00:00', 'level': 'INFO', 'message': 'Test log'}
        ]
        mock_db.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yaml').write_text('paths:\n  log_db: logs.db')
            
            result = runner.invoke(cli, ['status', '--config', 'config.yaml'])
            
            assert result.exit_code == 0
            assert 'Pipeline Status' in result.output
            assert '100' in result.output
    
    def test_invalid_config_path(self):
        """Test handling of invalid config path."""
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest', '--config', 'nonexistent.yaml'])
        
        assert result.exit_code != 0
        assert 'Error' in result.output or 'not found' in result.output
    
    def test_command_with_all_options(self):
        """Test command with all available options."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('config.yaml').write_text('search:\n  queries:\n    custom: "test query"')
            
            with patch('run.SearchHarvester') as mock_harvester:
                mock_instance = Mock()
                mock_instance.search_all.return_value = pd.DataFrame({'title': ['Test']})
                mock_harvester.return_value = mock_instance
                
                result = runner.invoke(cli, [
                    'harvest',
                    '--config', 'config.yaml',
                    '--query', 'custom',
                    '--sources', 'arxiv,crossref',
                    '--max-results', '50',
                    '--no-parallel',
                    '--output', 'custom_output.csv'
                ])
                
                # Check that options were parsed
                mock_instance.search_all.assert_called_once()
                call_args = mock_instance.search_all.call_args
                assert call_args[1]['sources'] == ['arxiv', 'crossref']
                assert call_args[1]['max_results_per_source'] == 50
                assert call_args[1]['parallel'] == False