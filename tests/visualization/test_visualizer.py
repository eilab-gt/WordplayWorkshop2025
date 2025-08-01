"""Comprehensive tests for the Visualizer module to improve coverage."""

from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.lit_review.visualization.visualizer import Visualizer


@pytest.fixture
def sample_papers_data():
    """Create sample paper data for testing."""
    return pd.DataFrame(
        {
            "title": ["Paper A", "Paper B", "Paper C", "Paper D", "Paper E"],
            "year": [2021, 2022, 2023, 2023, 2024],
            "source_db": ["arxiv", "semantic_scholar", "arxiv", "crossref", "arxiv"],
            "game_type": ["seminar", "matrix", "digital", "seminar", "hybrid"],
            "open_ended": ["yes", "yes", "no", "yes", "no"],
            "quantitative": ["yes", "no", "yes", "yes", "no"],
            "awscale": [1, 3, 2, 4, 5],
            "llm_family": ["GPT-4", "Claude", "GPT-3", "GPT-4", "Llama"],
            "llm_role": ["player", "generator", "analyst", "player", "player"],
            "failure_modes": [
                "escalation,bias",
                "hallucination",
                "escalation",
                "bias",
                "deception,escalation",
            ],
            "venue": [
                "Conference A",
                "Journal B",
                "Conference A",
                "Journal C",
                "Workshop D",
            ],
            "citations": [10, 5, 15, 3, 0],
        }
    )


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.output_dir = Path("test_output")
    config.viz_format = "png"
    config.viz_dpi = 150
    config.viz_style = "seaborn-v0_8"
    config.viz_figsize = (10, 6)
    config.viz_colors = {
        "game_types": {
            "seminar": "#1f77b4",
            "matrix": "#ff7f0e",
            "digital": "#2ca02c",
            "hybrid": "#d62728",
        },
        "awscale": {
            "1": "#d62728",
            "2": "#ff7f0e",
            "3": "#ffbb78",
            "4": "#2ca02c",
            "5": "#1f77b4",
        },
    }
    config.failure_vocab = {
        "content": ["bias", "hallucination"],
        "interactive": ["escalation", "deception"],
        "security": ["jailbreak", "prompt_injection"],
    }
    return config


class TestVisualizer:
    """Test cases for the Visualizer class."""

    def test_init(self, mock_config, tmp_path):
        """Test Visualizer initialization."""
        mock_config.output_dir = tmp_path

        visualizer = Visualizer(mock_config)

        assert visualizer.config == mock_config
        assert visualizer.output_dir == tmp_path / "figures"
        assert visualizer.output_dir.exists()
        assert visualizer.format == "png"
        assert visualizer.dpi == 150

    def test_init_invalid_style(self, mock_config, tmp_path):
        """Test initialization with invalid matplotlib style."""
        mock_config.output_dir = tmp_path
        mock_config.viz_style = "nonexistent-style"

        with patch("matplotlib.pyplot.style.use") as mock_style:
            mock_style.side_effect = [
                OSError(),
                None,
            ]  # First call fails, second succeeds
            visualizer = Visualizer(mock_config)

        assert mock_style.call_count == 2
        mock_style.assert_called_with("default")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_create_all_visualizations(
        self, mock_show, mock_savefig, mock_config, sample_papers_data, tmp_path
    ):
        """Test creating all visualizations."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        # Mock individual visualization methods
        with patch.object(
            visualizer, "plot_time_series", return_value=tmp_path / "timeline.png"
        ):
            with patch.object(
                visualizer,
                "plot_source_distribution",
                return_value=tmp_path / "sources.png",
            ):
                with patch.object(
                    visualizer,
                    "plot_game_types",
                    return_value=tmp_path / "game_types.png",
                ):
                    with patch.object(
                        visualizer,
                        "plot_awscale_distribution",
                        return_value=tmp_path / "awscale.png",
                    ):
                        with patch.object(
                            visualizer,
                            "plot_failure_modes",
                            return_value=tmp_path / "failures.png",
                        ):
                            with patch.object(
                                visualizer,
                                "plot_llm_families",
                                return_value=tmp_path / "llms.png",
                            ):
                                with patch.object(
                                    visualizer,
                                    "plot_venue_types",
                                    return_value=tmp_path / "venue_types.png",
                                ):
                                    with patch.object(
                                        visualizer,
                                        "plot_game_characteristics",
                                        return_value=tmp_path
                                        / "game_characteristics.png",
                                    ):
                                        figures = visualizer.create_all_visualizations(
                                            sample_papers_data, save=True
                                        )

                                        assert len(figures) == 8
                                        assert all(
                                            isinstance(fig, Path) for fig in figures
                                        )

    def test_plot_time_series(self, mock_config, sample_papers_data, tmp_path):
        """Test time series plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_time_series(sample_papers_data, save=True)

        assert fig_path == tmp_path / "figures" / "time_series.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_time_series_empty_data(self, mock_config, tmp_path):
        """Test time series plotting with empty data."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        empty_df = pd.DataFrame()

        result = visualizer.plot_time_series(empty_df, save=False)
        assert result is None

    def test_plot_source_distribution(self, mock_config, sample_papers_data, tmp_path):
        """Test source distribution plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_source_distribution(
                sample_papers_data, save=True
            )

        assert fig_path == tmp_path / "figures" / "source_distribution.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_game_types(self, mock_config, sample_papers_data, tmp_path):
        """Test game types plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_game_types(sample_papers_data, save=True)

        assert fig_path == tmp_path / "figures" / "game_types.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_awscale_distribution(self, mock_config, sample_papers_data, tmp_path):
        """Test AWScale distribution plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_awscale_distribution(
                sample_papers_data, save=True
            )

        assert fig_path == tmp_path / "figures" / "awscale_distribution.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_failure_modes(self, mock_config, sample_papers_data, tmp_path):
        """Test failure modes plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_failure_modes(sample_papers_data, save=True)

        assert fig_path == tmp_path / "figures" / "failure_modes.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_llm_families(self, mock_config, sample_papers_data, tmp_path):
        """Test LLM families plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_llm_families(sample_papers_data, save=True)

        assert fig_path == tmp_path / "figures" / "llm_families.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_plot_venue_types(self, mock_config, sample_papers_data, tmp_path):
        """Test venue types plotting."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        # Add venue_type data
        sample_papers_data["venue_type"] = [
            "conference",
            "journal",
            "conference",
            "journal",
            "workshop",
        ]

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig_path = visualizer.plot_venue_types(sample_papers_data, save=True)

        assert fig_path == tmp_path / "figures" / "venue_types.png"
        mock_savefig.assert_called_once()
        plt.close("all")

    def test_create_summary_report(self, mock_config, sample_papers_data):
        """Test summary report generation."""
        visualizer = Visualizer(mock_config)

        report = visualizer.create_summary_report(sample_papers_data)

        assert isinstance(report, dict)
        assert "total_papers" in report
        assert report["total_papers"] == 5
        assert "year_range" in report
        assert report["year_range"] == "2021-2024"
        assert "sources" in report
        assert "game_types" in report

    def test_plot_with_missing_columns(self, mock_config, tmp_path):
        """Test plotting with missing required columns."""
        mock_config.output_dir = tmp_path
        visualizer = Visualizer(mock_config)

        # DataFrame missing 'year' column
        incomplete_df = pd.DataFrame(
            {"title": ["Paper A", "Paper B"], "source_db": ["arxiv", "crossref"]}
        )

        result = visualizer.plot_time_series(incomplete_df, save=False)
        assert result is None

    def test_custom_color_scheme(self, mock_config, sample_papers_data, tmp_path):
        """Test visualization with custom color scheme."""
        mock_config.output_dir = tmp_path

        # Custom colors
        mock_config.viz_colors = {
            "game_types": {
                "seminar": "#FF0000",
                "matrix": "#00FF00",
                "digital": "#0000FF",
                "hybrid": "#FFFF00",
            }
        }

        visualizer = Visualizer(mock_config)

        with patch("matplotlib.pyplot.savefig"):
            visualizer.plot_game_types(sample_papers_data, save=True)

        # Verify custom colors were used (would need to inspect the plot)
        plt.close("all")
