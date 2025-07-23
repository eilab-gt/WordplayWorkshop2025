"""Tests for the CLI interface."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from click.testing import CliRunner

from run import cli


class TestCLI:
    """Test cases for CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Literature Review Pipeline" in result.output
        assert "harvest" in result.output
        assert "extract" in result.output

    @patch("run.Normalizer")
    @patch("run.SearchHarvester")
    def test_harvest_command(self, mock_harvester, mock_normalizer, sample_config):
        """Test harvest command."""
        # Setup mock
        mock_instance = Mock()
        test_df = pd.DataFrame(
            {
                "title": ["Test Paper"],
                "authors": ["Test Author"],
                "year": [2024],
                "source_db": ["test_source"],
            }
        )
        mock_instance.search_all.return_value = test_df
        mock_instance.save_results = Mock()
        mock_harvester.return_value = mock_instance
        
        # Mock normalizer
        mock_norm_instance = Mock()
        mock_norm_instance.normalize_dataframe.return_value = test_df
        mock_normalizer.return_value = mock_norm_instance

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create config file
            Path("config.yaml").write_text('search:\n  queries:\n    preset1: "test"\npaths:\n  output_dir: outputs')

            result = runner.invoke(cli, ["--config", "config.yaml", "harvest"])

            assert result.exit_code == 0
            assert "Starting harvest" in result.output
            assert "Found 1 papers" in result.output

    @patch("run.SearchHarvester")
    @patch("run.Normalizer")
    @patch("run.ScreenUI")
    def test_prepare_screen_command(
        self, mock_screen, mock_normalizer, mock_harvester, sample_config
    ):
        """Test prepare-screen command."""
        # Setup mocks
        test_df = pd.DataFrame(
            {"title": ["Test Paper"], "screening_id": ["SCREEN_0001"]}
        )

        mock_normalizer.return_value.normalize.return_value = test_df
        mock_screen.return_value.generate_sheet.return_value = "screening.xlsx"

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            test_df.to_csv("papers.csv", index=False)
            Path("config.yaml").write_text("paths:\n  output_dir: outputs")

            result = runner.invoke(
                cli,
                ["--config", "config.yaml", "prepare-screen", "--input", "papers.csv"],
            )

            assert result.exit_code == 0
            assert "Preparing screening sheet" in result.output or "Loading papers" in result.output

    @patch("run.Tagger")
    @patch("run.LLMExtractor")
    def test_extract_command(self, mock_extractor, mock_tagger):
        """Test extract command."""
        # Setup mock
        mock_instance = Mock()
        test_df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "venue_type": ["conference"],
                "extraction_status": ["success"],
                "title": ["Test Paper"],
                "pdf_path": ["test.pdf"],
            }
        )
        mock_instance.extract_all.return_value = test_df
        mock_extractor.return_value = mock_instance
        
        # Mock tagger
        mock_tag_instance = Mock()
        mock_tag_instance.tag_papers.return_value = test_df
        mock_tag_instance.get_failure_mode_summary.return_value = pd.DataFrame()
        mock_tagger.return_value = mock_tag_instance

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            pd.DataFrame(
                {
                    "screening_id": ["SCREEN_0001"],
                    "title": ["Test Paper"],
                    "pdf_path": ["test.pdf"],
                }
            ).to_csv("screening.csv", index=False)

            Path("config.yaml").write_text("api_keys:\n  openai: test-key\npaths:\n  output_dir: outputs")

            result = runner.invoke(
                cli, ["--config", "config.yaml", "extract", "--input", "screening.csv"]
            )

            assert result.exit_code == 0
            assert "Loading screening data" in result.output or "Extracting with LLM" in result.output

    @patch("run.Visualizer")
    def test_visualise_command(self, mock_visualizer):
        """Test visualise command."""
        # Setup mock
        mock_instance = Mock()
        # Mock the output directory
        mock_instance.output_dir = Path("outputs/figures")
        # Mock figure objects
        mock_fig1 = Mock()
        mock_fig1.name = "timeline.png"
        mock_fig2 = Mock()
        mock_fig2.name = "venue_dist.png"
        mock_instance.create_all_visualizations.return_value = [mock_fig1, mock_fig2]
        mock_instance.create_summary_report.return_value = {
            "total_papers": 2,
            "year_range": "2023-2024",
        }
        mock_visualizer.return_value = mock_instance

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            pd.DataFrame(
                {"year": [2023, 2024], "venue_type": ["conference", "journal"]}
            ).to_csv("extraction.csv", index=False)

            Path("config.yaml").write_text(
                "viz:\n  charts:\n    timeline:\n      enabled: true"
            )

            result = runner.invoke(
                cli,
                ["--config", "config.yaml", "visualise", "--input", "extraction.csv"],
            )

            assert result.exit_code == 0
            assert "Creating visualizations" in result.output or "Loading extraction data" in result.output

    @patch("run.Visualizer")
    @patch("run.Exporter")
    def test_export_command(self, mock_exporter, mock_visualizer):
        """Test export command."""
        # Setup mock exporter
        mock_exp_instance = Mock()
        mock_exp_instance.export_full_package.return_value = Path("package.zip")
        mock_exp_instance.export_bibtex.return_value = Path("refs.bib")
        mock_exporter.return_value = mock_exp_instance
        
        # Setup mock visualizer
        mock_viz_instance = Mock()
        mock_viz_instance.create_all_visualizations.return_value = []
        mock_viz_instance.create_summary_report.return_value = {}
        mock_visualizer.return_value = mock_viz_instance

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input file
            pd.DataFrame({"screening_id": ["SCREEN_0001"], "title": ["Paper 1"]}).to_csv(
                "extraction.csv", index=False
            )

            Path("config.yaml").write_text("export:\n  zenodo:\n    enabled: false")

            result = runner.invoke(
                cli,
                [
                    "--config",
                    "config.yaml",
                    "export",
                    "--input",
                    "extraction.csv",
                ],
            )

            assert result.exit_code == 0
            assert "Creating export package" in result.output or "Loading extraction data" in result.output

    @patch("run.SearchHarvester")
    def test_test_command(self, mock_harvester):
        """Test the test command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.search_all.return_value = pd.DataFrame(
            {
                "title": ["Test Paper"],
                "authors": ["Test Author"],
                "year": [2024],
                "source_db": ["google_scholar"],
            }
        )
        mock_harvester.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("config.yaml").write_text('search:\n  queries:\n    preset1: "test"')

            result = runner.invoke(cli, ["--config", "config.yaml", "test"])

            assert result.exit_code == 0
            assert "Running test search" in result.output or "Query:" in result.output

    @patch("run.PDFFetcher")
    def test_status_command(self, mock_pdf_fetcher):
        """Test status command."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_cache_statistics.return_value = {
            "cache_dir": Path("cache/pdfs"),
            "total_files": 10,
            "total_size_mb": 50.5,
        }
        mock_pdf_fetcher.return_value = mock_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("config.yaml").write_text("paths:\n  output_dir: outputs")

            result = runner.invoke(cli, ["--config", "config.yaml", "status"])

            assert result.exit_code == 0
            assert "Pipeline Status" in result.output

    def test_invalid_config_path(self):
        """Test handling of invalid config path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", "nonexistent.yaml", "harvest"])

        assert result.exit_code != 0
        assert result.exception is not None
        assert "not found" in str(result.exception) or "Configuration file" in str(result.exception)

    def test_command_with_all_options(self):
        """Test command with all available options."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("config.yaml").write_text(
                'search:\n  queries:\n    custom: "test query"'
            )

            with patch("run.SearchHarvester") as mock_harvester, patch("run.Normalizer") as mock_normalizer:
                mock_instance = Mock()
                test_df = pd.DataFrame(
                    {"title": ["Test"], "source_db": ["test_source"]}
                )
                mock_instance.search_all.return_value = test_df
                mock_instance.save_results = Mock()
                mock_harvester.return_value = mock_instance
                
                # Mock normalizer
                mock_norm = Mock()
                mock_norm.normalize_dataframe.return_value = test_df
                mock_normalizer.return_value = mock_norm

                result = runner.invoke(
                    cli,
                    [
                        "--config",
                        "config.yaml",
                        "harvest",
                        "--query",
                        "custom",
                        "--sources",
                        "arxiv",
                        "--sources",
                        "crossref",
                        "--max-results",
                        "50",
                        "--sequential",
                        "--output",
                        "custom_output.csv",
                    ],
                )

                # Check that the command executed successfully
                assert result.exit_code == 0
                
                # Check that options were parsed
                mock_instance.search_all.assert_called_once()
                call_args = mock_instance.search_all.call_args
                assert call_args[1]["sources"] == ["arxiv", "crossref"]
                assert call_args[1]["max_results_per_source"] == 50
                assert not call_args[1]["parallel"]
