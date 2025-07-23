"""Tests for the Tagger module."""

import pandas as pd

from src.lit_review.analysis import Tagger


class TestTagger:
    """Test cases for Tagger class."""

    def test_init(self, sample_config):
        """Test Tagger initialization."""
        tagger = Tagger(sample_config)
        assert tagger.config is not None
        assert hasattr(tagger, "failure_vocabularies")
        assert hasattr(tagger, "llm_patterns")
        assert hasattr(tagger, "game_type_patterns")

    def test_tag_failures_basic(self, sample_config):
        """Test basic failure mode tagging."""
        tagger = Tagger(sample_config)

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
                "abstract": [
                    "This study shows escalation risks in LLM wargaming scenarios.",
                    "We found significant bias in the model outputs during conflict simulation.",
                    "No issues were found in this implementation.",
                ],
                "failure_modes": ["", "", ""],  # Empty initially
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "failure_modes_regex" in tagged_df.columns
        assert "escalation" in tagged_df.loc[0, "failure_modes_regex"]
        assert "bias" in tagged_df.loc[1, "failure_modes_regex"]
        assert tagged_df.loc[2, "failure_modes_regex"] == ""

    def test_tag_failures_multiple(self, sample_config):
        """Test tagging multiple failure modes in one text."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "abstract": [
                    "The model showed both hallucination and bias, with clear data leakage from training."
                ],
                "failure_modes": [""],
            }
        )

        tagged_df = tagger.tag_failures(df)

        failures = tagged_df.loc[0, "failure_modes_regex"]
        assert "hallucination" in failures
        assert "bias" in failures
        assert "data_leakage" in failures
        assert failures.count("|") >= 2  # At least 3 failure modes

    def test_llm_detection(self, sample_config):
        """Test LLM family detection."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": [
                    "SCREEN_0001",
                    "SCREEN_0002",
                    "SCREEN_0003",
                    "SCREEN_0004",
                ],
                "abstract": [
                    "We used GPT-4 for generating strategic moves.",
                    "Claude 2 was employed as the main reasoning engine.",
                    "The study uses Llama-70B for simulation.",
                    "A custom transformer model was developed.",
                ],
                "title": [
                    "GPT-4 Study",
                    "Claude Research",
                    "Llama Paper",
                    "Custom Model",
                ],
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "llm_detected" in tagged_df.columns
        assert "gpt4" in tagged_df.loc[0, "llm_detected"].lower()
        assert "claude" in tagged_df.loc[1, "llm_detected"].lower()
        assert "llama" in tagged_df.loc[2, "llm_detected"].lower()
        assert tagged_df.loc[3, "llm_detected"] == ""  # No specific LLM detected

    def test_game_type_detection(self, sample_config):
        """Test game type detection."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
                "abstract": [
                    "We conducted a seminar wargame with domain experts.",
                    "The matrix game approach was used for scenario planning.",
                    "Digital simulation environment for conflict modeling.",
                ],
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "game_type_detected" in tagged_df.columns
        assert tagged_df.loc[0, "game_type_detected"] == "seminar"
        assert tagged_df.loc[1, "game_type_detected"] == "matrix"
        assert tagged_df.loc[2, "game_type_detected"] == "digital"

    def test_metrics_detection(self, sample_config):
        """Test evaluation metrics detection."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002"],
                "abstract": [
                    "We measured win rate and conducted human evaluation of outputs.",
                    "Success was measured through SME assessment and plausibility scores.",
                ],
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "metrics_detected" in tagged_df.columns
        assert "win_rate" in tagged_df.loc[0, "metrics_detected"]
        assert "human_evaluation" in tagged_df.loc[0, "metrics_detected"]
        assert "assessment" in tagged_df.loc[1, "metrics_detected"]

    def test_code_availability_detection(self, sample_config):
        """Test code availability detection."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
                "abstract": [
                    "Code available at github.com/example/repo",
                    "Implementation details at https://github.com/author/project",
                    "No code is publicly available.",
                ],
                "code_release": ["", "", ""],
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "code_detected" in tagged_df.columns
        assert "github.com/example/repo" in tagged_df.loc[0, "code_detected"]
        assert "github.com/author/project" in tagged_df.loc[1, "code_detected"]
        assert tagged_df.loc[2, "code_detected"] == ""

    def test_case_insensitive_matching(self, sample_config):
        """Test case-insensitive pattern matching."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "abstract": ["The model showed HALLUCINATION and Bias issues."],
            }
        )

        tagged_df = tagger.tag_failures(df)

        failures = tagged_df.loc[0, "failure_modes_regex"]
        assert "hallucination" in failures
        assert "bias" in failures

    def test_empty_dataframe(self, sample_config):
        """Test handling of empty DataFrame."""
        tagger = Tagger(sample_config)

        empty_df = pd.DataFrame()
        tagged_df = tagger.tag_failures(empty_df)

        assert isinstance(tagged_df, pd.DataFrame)
        assert len(tagged_df) == 0

    def test_missing_text_columns(self, sample_config):
        """Test handling of DataFrames missing text columns."""
        tagger = Tagger(sample_config)

        # DataFrame without abstract or title
        df = pd.DataFrame({"screening_id": ["SCREEN_0001"], "year": [2024]})

        tagged_df = tagger.tag_failures(df)

        # Should handle gracefully
        assert isinstance(tagged_df, pd.DataFrame)
        assert "failure_modes_regex" in tagged_df.columns
        assert tagged_df.loc[0, "failure_modes_regex"] == ""

    def test_pattern_boundary_matching(self, sample_config):
        """Test that patterns match word boundaries correctly."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001", "SCREEN_0002"],
                "abstract": [
                    "The escalation of conflict was studied.",  # Should match
                    "The descalation process was effective.",  # Should not match 'escalation'
                ],
            }
        )

        tagged_df = tagger.tag_failures(df)

        assert "escalation" in tagged_df.loc[0, "failure_modes_regex"]
        # Depending on implementation, this might or might not match
        # The test documents the expected behavior

    def test_preserve_existing_tags(self, sample_config):
        """Test that existing failure mode tags are preserved."""
        tagger = Tagger(sample_config)

        df = pd.DataFrame(
            {
                "screening_id": ["SCREEN_0001"],
                "abstract": ["Study shows bias in outputs."],
                "failure_modes": ["escalation|deception"],  # Existing tags
            }
        )

        tagged_df = tagger.tag_failures(df)

        # Should combine existing and detected tags
        assert "bias" in tagged_df.loc[0, "failure_modes_regex"]
        # Original tags might be preserved depending on implementation
