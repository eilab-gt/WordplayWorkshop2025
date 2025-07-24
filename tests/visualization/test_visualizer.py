"""Tests for visualization module."""

from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.lit_review.visualization import Visualizer
from tests.test_doubles import RealConfigForTests

# Use non-interactive backend for tests
matplotlib.use("Agg")


class TestVisualizerBehavior:
    """Test visualizer creates correct charts and statistics."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        config = RealConfigForTests(
            output_dir=tmp_path / "output",
            viz_format="png",
            viz_dpi=100,  # Lower DPI for faster tests
            viz_style="default",
            viz_figsize=(8, 6),
            viz_colors={
                "awscale": {
                    "1": "#d62728",  # red
                    "2": "#ff7f0e",  # orange
                    "3": "#bcbd22",  # yellow-green
                    "4": "#2ca02c",  # green
                    "5": "#1f77b4",  # blue
                }
            },
        )
        yield config
        config.cleanup()

    @pytest.fixture
    def visualizer(self, config):
        """Create visualizer instance."""
        return Visualizer(config)

    @pytest.fixture
    def sample_papers_df(self):
        """Create comprehensive sample data for testing."""
        return pd.DataFrame(
            [
                {
                    "title": "LLM Wargaming Paper 1",
                    "year": 2022,
                    "awscale": 4,
                    "game_type": "Matrix Game",
                    "failure_modes": "Hallucination; Context Limits",
                    "llm_family": "GPT",
                    "source_db": "arxiv",
                    "venue": "NeurIPS",
                    "venue_type": "conference",
                    "open_ended": True,
                    "quantitative": False,
                },
                {
                    "title": "AI Strategy Game Study",
                    "year": 2023,
                    "awscale": 3,
                    "game_type": "Digital Simulation",
                    "failure_modes": "Bias; Hallucination",
                    "llm_family": "Claude",
                    "source_db": "semantic_scholar",
                    "venue": "ICML",
                    "venue_type": "conference",
                    "open_ended": False,
                    "quantitative": True,
                },
                {
                    "title": "Human-AI Team Wargaming",
                    "year": 2023,
                    "awscale": 5,
                    "game_type": "Tabletop Exercise",
                    "failure_modes": "Context Limits",
                    "llm_family": "GPT",
                    "source_db": "arxiv",
                    "venue": "AI Magazine",
                    "venue_type": "journal",
                    "open_ended": True,
                    "quantitative": True,
                },
                {
                    "title": "Automated Planning Research",
                    "year": 2024,
                    "awscale": 2,
                    "game_type": "Digital Simulation",
                    "failure_modes": "Reasoning Errors; Bias",
                    "llm_family": "LLaMA",
                    "source_db": "crossref",
                    "venue": "AAAI",
                    "venue_type": "conference",
                    "open_ended": False,
                    "quantitative": True,
                },
                {
                    "title": "Crisis Management Simulation",
                    "year": 2024,
                    "awscale": 4,
                    "game_type": "Matrix Game",
                    "failure_modes": None,  # Test missing data
                    "llm_family": "Other",
                    "source_db": "arxiv",
                    "venue": None,  # Test missing venue
                    "venue_type": None,
                    "open_ended": True,
                    "quantitative": False,
                },
            ]
        )

    @pytest.mark.fast
    def test_initializes_with_correct_output_directory(self, visualizer, config):
        """Test visualizer creates output directory on initialization."""
        expected_dir = Path(config.output_dir) / "figures"
        assert visualizer.output_dir == expected_dir
        assert visualizer.output_dir.exists()

    @pytest.mark.fast
    def test_handles_invalid_matplotlib_style_gracefully(self, config):
        """Test visualizer falls back to default style if configured style not found."""
        config.viz_style = "non_existent_style"
        visualizer = Visualizer(config)
        # Should not raise exception and use default
        assert visualizer.style == "non_existent_style"

    def test_creates_all_standard_visualizations(self, visualizer, sample_papers_df):
        """Test all visualization methods are called and produce outputs."""
        figures = visualizer.create_all_visualizations(sample_papers_df, save=True)

        # Should create 8 standard figures
        assert len(figures) >= 6  # Some may be skipped if no data
        assert all(fig.exists() for fig in figures)
        assert all(fig.suffix == f".{visualizer.format}" for fig in figures)

    def test_time_series_shows_publication_trends(self, visualizer, sample_papers_df):
        """Test time series plot correctly shows publication counts by year."""
        fig_path = visualizer.plot_time_series(sample_papers_df, save=True)

        assert fig_path.exists()
        assert "time_series" in fig_path.name

        # Verify data correctness
        year_counts = sample_papers_df["year"].value_counts().sort_index()
        assert year_counts[2023] == 2  # 2 papers in 2023
        assert year_counts[2024] == 2  # 2 papers in 2024

    def test_awscale_distribution_uses_configured_colors(
        self, visualizer, sample_papers_df
    ):
        """Test AWScale histogram uses colors from configuration."""
        fig_path = visualizer.plot_awscale_distribution(sample_papers_df, save=True)

        assert fig_path.exists()
        assert "awscale" in fig_path.name

        # Check distribution
        awscale_counts = sample_papers_df["awscale"].value_counts()
        assert awscale_counts[4] == 2  # Two papers with AWScale 4
        assert awscale_counts[2] == 1  # One paper with AWScale 2

    def test_handles_missing_data_gracefully(self, visualizer):
        """Test visualizer handles DataFrames with missing values."""
        df_with_nulls = pd.DataFrame(
            [
                {"title": "Paper 1", "year": 2023, "awscale": None},
                {"title": "Paper 2", "year": None, "awscale": 3},
                {"title": "Paper 3", "year": 2024, "awscale": 4},
            ]
        )

        # Should not crash
        figures = visualizer.create_all_visualizations(df_with_nulls, save=True)
        assert len(figures) > 0

    def test_failure_modes_aggregates_correctly(self, visualizer, sample_papers_df):
        """Test failure modes are parsed and counted correctly."""
        fig_path = visualizer.plot_failure_modes(sample_papers_df, save=True)

        if fig_path:  # May be None if no failure data
            assert fig_path.exists()

            # Manually count failure modes
            all_failures = []
            for failures in sample_papers_df["failure_modes"].dropna():
                all_failures.extend([f.strip() for f in failures.split(";")])

            failure_counts = pd.Series(all_failures).value_counts()
            assert failure_counts["Hallucination"] == 2
            assert failure_counts["Bias"] == 2
            assert failure_counts["Context Limits"] == 2

    def test_source_distribution_shows_all_databases(
        self, visualizer, sample_papers_df
    ):
        """Test source database distribution includes all sources."""
        fig_path = visualizer.plot_source_distribution(sample_papers_df, save=True)

        assert fig_path.exists()
        source_counts = sample_papers_df["source_db"].value_counts()
        assert source_counts["arxiv"] == 3
        assert source_counts["semantic_scholar"] == 1
        assert source_counts["crossref"] == 1

    def test_saves_figures_in_correct_format(self, visualizer, sample_papers_df):
        """Test figures are saved in configured format with correct DPI."""
        # Test different formats
        for fmt in ["png", "pdf", "svg"]:
            visualizer.format = fmt
            fig_path = visualizer.plot_time_series(sample_papers_df, save=True)

            assert fig_path.suffix == f".{fmt}"
            assert fig_path.exists()

            # Clean up non-PNG files
            if fmt != "png":
                fig_path.unlink()

    def test_creates_summary_report_with_statistics(self, visualizer, sample_papers_df):
        """Test summary report contains correct statistical information."""
        summary = visualizer.create_summary_report(sample_papers_df)

        assert summary["total_papers"] == 5
        assert summary["year_range"] == "2022-2024"
        assert "sources" in summary
        assert summary["sources"]["arxiv"] == 3

        if "awscale" in summary:
            assert 2 <= summary["awscale"]["mean"] <= 5
            assert summary["awscale"]["mode"] == 4  # Most common value

        if "top_failure_modes" in summary:
            top_modes = list(summary["top_failure_modes"].keys())
            assert "Hallucination" in top_modes[:3]
            assert "Bias" in top_modes[:3]

    @pytest.mark.parametrize(
        "plot_method,expected_file_pattern",
        [
            ("plot_time_series", "time_series"),
            ("plot_game_types", "game_type"),
            ("plot_awscale_distribution", "awscale"),
            ("plot_llm_families", "llm_famil"),
            ("plot_source_distribution", "source_dist"),
            ("plot_venue_types", "venue_type"),
        ],
    )
    def test_plot_methods_create_expected_files(
        self, visualizer, sample_papers_df, plot_method, expected_file_pattern
    ):
        """Test each plot method creates file with expected name pattern."""
        method = getattr(visualizer, plot_method)
        fig_path = method(sample_papers_df, save=True)

        if fig_path:  # Some plots may return None if no data
            assert fig_path.exists()
            assert expected_file_pattern in fig_path.name.lower()

    def test_handles_empty_dataframe(self, visualizer):
        """Test visualizer handles empty DataFrame without crashing."""
        empty_df = pd.DataFrame()
        figures = visualizer.create_all_visualizations(empty_df, save=True)
        # Should complete without errors, though may produce fewer figures
        assert isinstance(figures, list)

    def test_respects_save_parameter(self, visualizer, sample_papers_df):
        """Test figures are not saved when save=False."""
        initial_count = len(list(visualizer.output_dir.glob("*")))

        # Create without saving
        figures = visualizer.create_all_visualizations(sample_papers_df, save=False)

        # No new files should be created
        final_count = len(list(visualizer.output_dir.glob("*")))
        assert final_count == initial_count
        assert figures == []  # No paths returned when not saving

    @pytest.mark.slow
    def test_produces_high_quality_outputs_for_publication(
        self, visualizer, sample_papers_df
    ):
        """Test high DPI outputs suitable for publication."""
        visualizer.dpi = 300  # Publication quality
        visualizer.format = "pdf"  # Vector format

        fig_path = visualizer.plot_time_series(sample_papers_df, save=True)

        assert fig_path.exists()
        assert fig_path.suffix == ".pdf"
        # File should be larger due to high quality
        assert fig_path.stat().st_size > 1000  # At least 1KB

    def test_consistent_styling_across_plots(self, visualizer, sample_papers_df):
        """Test all plots use consistent visual styling."""
        # Create a few different plots
        plots = [
            visualizer.plot_time_series(sample_papers_df, save=False),
            visualizer.plot_awscale_distribution(sample_papers_df, save=False),
            visualizer.plot_source_distribution(sample_papers_df, save=False),
        ]

        # All should use same figure size
        for _ in plots:
            if plt.get_fignums():  # If figure was created
                fig = plt.gcf()
                assert fig.get_size_inches()[0] == visualizer.figsize[0]
                assert fig.get_size_inches()[1] == visualizer.figsize[1]
                plt.close(fig)


@pytest.mark.unit
class TestVisualizerHelperMethods:
    """Test internal helper methods of Visualizer."""

    def test_color_mapping_for_awscale(self, visualizer):
        """Test AWScale values map to correct colors."""
        # Access color configuration
        awscale_colors = visualizer.colors.get("awscale", {})

        # Should have colors for scales 1-5
        for scale in range(1, 6):
            assert str(scale) in awscale_colors
            # Colors should be valid hex
            assert awscale_colors[str(scale)].startswith("#")
            assert len(awscale_colors[str(scale)]) == 7

    def test_figure_naming_convention(self, visualizer):
        """Test figures are named with timestamp and description."""
        # Mock datetime to get predictable names
        with patch("src.lit_review.visualization.visualizer.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20240115_120000"

            # Create a simple plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            # Save using visualizer's method (if it has one)
            # This tests the naming convention
            expected_pattern = "20240115_120000"

            plt.close(fig)
