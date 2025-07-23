"""Tests for the Visualizer module."""

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from src.lit_review.analysis import Visualizer


class TestVisualizer:
    """Test cases for Visualizer class."""

    def test_init(self, sample_config):
        """Test Visualizer initialization."""
        visualizer = Visualizer(sample_config)
        assert visualizer.config is not None
        assert hasattr(visualizer, "viz_config")
        assert hasattr(visualizer, "output_dir")

    @patch("matplotlib.pyplot.savefig")
    def test_create_timeline(self, mock_savefig, sample_config, temp_dir):
        """Test timeline visualization creation."""
        visualizer = Visualizer(sample_config)

        # Create test data
        df = pd.DataFrame(
            {
                "year": [2020, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2024],
                "title": [f"Paper {i}" for i in range(9)],
            }
        )

        output_path = Path(temp_dir) / "timeline.png"
        result_path = visualizer.create_timeline(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")  # Clean up

    @patch("matplotlib.pyplot.savefig")
    def test_create_venue_distribution(self, mock_savefig, sample_config, temp_dir):
        """Test venue distribution visualization."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame(
            {
                "venue_type": [
                    "conference",
                    "conference",
                    "journal",
                    "workshop",
                    "journal",
                    "conference",
                ]
            }
        )

        output_path = Path(temp_dir) / "venue_dist.png"
        result_path = visualizer.create_venue_distribution(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_create_failure_modes_chart(self, mock_savefig, sample_config, temp_dir):
        """Test failure modes visualization."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame(
            {
                "failure_modes": [
                    "escalation|bias",
                    "hallucination",
                    "bias|prompt_sensitivity",
                    "escalation|hallucination|bias",
                    "",
                ]
            }
        )

        output_path = Path(temp_dir) / "failure_modes.png"
        result_path = visualizer.create_failure_modes_chart(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_create_llm_families_chart(self, mock_savefig, sample_config, temp_dir):
        """Test LLM families visualization."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame(
            {
                "llm_family": [
                    "GPT-4",
                    "GPT-4",
                    "Claude",
                    "Llama-70B",
                    "GPT-3.5",
                    "Claude",
                ]
            }
        )

        output_path = Path(temp_dir) / "llm_families.png"
        result_path = visualizer.create_llm_families_chart(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_create_game_types_chart(self, mock_savefig, sample_config, temp_dir):
        """Test game types visualization."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame(
            {
                "game_type": [
                    "matrix",
                    "seminar",
                    "digital",
                    "matrix",
                    "hybrid",
                    "seminar",
                ]
            }
        )

        output_path = Path(temp_dir) / "game_types.png"
        result_path = visualizer.create_game_types_chart(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_create_awscale_distribution(self, mock_savefig, sample_config, temp_dir):
        """Test AWScale distribution visualization."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame({"awscale": [1, 2, 2, 3, 3, 3, 4, 4, 5]})

        output_path = Path(temp_dir) / "awscale.png"
        result_path = visualizer.create_awscale_distribution(df, output_path)

        assert result_path == output_path
        mock_savefig.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.savefig")
    def test_generate_all_charts(self, mock_savefig, sample_config, temp_dir):
        """Test generating all charts at once."""
        visualizer = Visualizer(sample_config)

        # Create comprehensive test data
        df = pd.DataFrame(
            {
                "year": [2022, 2023, 2024] * 3,
                "venue_type": ["conference", "journal", "workshop"] * 3,
                "failure_modes": ["bias", "escalation|bias", ""] * 3,
                "llm_family": ["GPT-4", "Claude", "Llama"] * 3,
                "game_type": ["matrix", "seminar", "digital"] * 3,
                "awscale": [1, 3, 5] * 3,
            }
        )

        chart_paths = visualizer.generate_all_charts(df)

        assert isinstance(chart_paths, dict)
        assert "timeline" in chart_paths
        assert "venue_dist" in chart_paths
        assert "failure_modes" in chart_paths
        assert "llm_families" in chart_paths
        assert "game_types" in chart_paths
        assert "awscale" in chart_paths

        # Check that savefig was called for each enabled chart
        assert mock_savefig.call_count >= 6
        plt.close("all")

    def test_empty_dataframe_handling(self, sample_config, temp_dir):
        """Test handling of empty DataFrames."""
        visualizer = Visualizer(sample_config)

        empty_df = pd.DataFrame()

        # Should handle empty DataFrame gracefully
        with patch("matplotlib.pyplot.savefig"):
            output_path = Path(temp_dir) / "empty_test.png"
            # Different methods might handle empty data differently
            # Testing that they don't crash
            try:
                visualizer.create_timeline(empty_df, output_path)
            except Exception as e:
                # Should handle gracefully, not crash
                raise AssertionError(f"Failed to handle empty DataFrame: {e}")

        plt.close("all")

    def test_missing_columns_handling(self, sample_config, temp_dir):
        """Test handling of DataFrames with missing expected columns."""
        visualizer = Visualizer(sample_config)

        # DataFrame missing expected columns
        df = pd.DataFrame({"title": ["Paper 1", "Paper 2"]})

        with patch("matplotlib.pyplot.savefig"):
            output_path = Path(temp_dir) / "missing_cols.png"
            # Should handle missing columns gracefully
            try:
                result = visualizer.create_timeline(df, output_path)
                # Might return None or create empty chart
                assert result is None or Path(result).exists()
            except KeyError:
                # Should not raise KeyError
                raise AssertionError("Failed to handle missing columns")

        plt.close("all")

    def test_custom_figsize(self, sample_config):
        """Test that custom figure sizes from config are applied."""
        visualizer = Visualizer(sample_config)

        # Check that config figsize is loaded
        assert "timeline" in visualizer.viz_config["charts"]
        assert "figsize" in visualizer.viz_config["charts"]["timeline"]
        assert len(visualizer.viz_config["charts"]["timeline"]["figsize"]) == 2

    def test_chart_styling(self, sample_config, temp_dir):
        """Test that charts have proper styling applied."""
        visualizer = Visualizer(sample_config)

        df = pd.DataFrame({"year": [2022, 2023, 2024], "count": [5, 10, 15]})

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.title") as mock_title:
                with patch("matplotlib.pyplot.xlabel") as mock_xlabel:
                    with patch("matplotlib.pyplot.ylabel") as mock_ylabel:
                        output_path = Path(temp_dir) / "styled_chart.png"
                        visualizer.create_timeline(df, output_path)

                        # Check that styling methods were called
                        mock_title.assert_called()
                        mock_xlabel.assert_called()
                        mock_ylabel.assert_called()

        plt.close("all")

    def test_disabled_charts(self, sample_config):
        """Test that disabled charts are not generated."""
        # Modify config to disable some charts
        visualizer = Visualizer(sample_config)
        visualizer.viz_config["charts"]["timeline"]["enabled"] = False

        df = pd.DataFrame({"year": [2022, 2023, 2024]})

        with patch("matplotlib.pyplot.savefig"):
            chart_paths = visualizer.generate_all_charts(df)

            # Timeline should not be in results
            assert "timeline" not in chart_paths

            # Other charts should still be generated
            assert len(chart_paths) > 0

        plt.close("all")
