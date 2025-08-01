"""Refactored tests for enhanced LLM extractor using behavioral testing."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from src.lit_review.extraction.enhanced_llm_extractor import EnhancedLLMExtractor
from tests.test_doubles import FakeLLMService, RealConfigForTests


class TestEnhancedLLMExtractorBehavior:
    """Test enhanced LLM extractor behavior (not implementation)."""

    @pytest.fixture
    def config(self):
        """Create real config for testing."""
        config = RealConfigForTests()
        yield config
        config.cleanup()

    @pytest.fixture
    def fake_llm_service(self):
        """Create fake LLM service."""
        return FakeLLMService(healthy=True)

    @pytest.fixture
    def extractor_with_fake_service(self, config, fake_llm_service, monkeypatch):
        """Create extractor with fake service backend."""
        extractor = EnhancedLLMExtractor(config, "http://localhost:8000")

        # Patch HTTP calls to use fake service
        def fake_get(url, **kwargs):
            if url.endswith("/health"):
                response = requests.Response()
                response.status_code = 200 if fake_llm_service.check_health() else 503
                return response
            elif url.endswith("/models"):
                response = requests.Response()
                response.status_code = 200
                response._content = json.dumps(fake_llm_service.get_models()).encode()
                return response
            raise ValueError(f"Unexpected URL: {url}")

        def fake_post(url, json_data=None, **kwargs):
            if url.endswith("/extract"):
                result = fake_llm_service.extract(
                    json_data["text"],
                    json_data["model"],
                    temperature=json_data.get("temperature", 0.1),
                )
                response = requests.Response()
                response.status_code = 200
                response._content = json.dumps(result).encode()
                return response
            raise ValueError(f"Unexpected URL: {url}")

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("requests.Session.get", fake_get)
        monkeypatch.setattr("requests.Session.post", fake_post)

        extractor._fake_service = fake_llm_service  # For assertions
        return extractor

    @pytest.fixture
    def sample_papers_df(self):
        """Create realistic sample papers DataFrame."""
        return pd.DataFrame(
            [
                {
                    "title": "LLM Wargaming: A Novel Framework for Strategic Simulation",
                    "authors": "Smith, A.; Jones, B.",
                    "year": 2024,
                    "abstract": "We present a novel framework for integrating LLMs into wargaming...",
                    "arxiv_id": "2401.00001",
                    "doi": "10.1234/test.2024.001",
                    "pdf_path": None,
                    "pdf_status": "",
                    "source_db": "arxiv",
                },
                {
                    "title": "Evaluating Human-AI Teams in Complex Decision Making",
                    "authors": "Brown, C.",
                    "year": 2024,
                    "abstract": "This study evaluates human-AI collaboration in strategic scenarios...",
                    "arxiv_id": None,
                    "doi": "10.1234/test.2024.002",
                    "pdf_path": "/tmp/brown_2024.pdf",
                    "pdf_status": "downloaded_success",
                    "source_db": "semantic_scholar",
                },
                {
                    "title": "Multi-Agent LLM Systems for Automated Planning",
                    "authors": "Davis, D.; Wilson, E.",
                    "year": 2023,
                    "abstract": "We develop a multi-agent system using LLMs for automated planning...",
                    "arxiv_id": "2312.00999",
                    "doi": None,
                    "pdf_path": "/tmp/davis_2023.pdf",
                    "pdf_status": "cached",
                    "source_db": "arxiv",
                },
            ]
        )

    def test_service_health_check_behavior(self, extractor_with_fake_service):
        """Test that extractor correctly handles service health status."""
        extractor = extractor_with_fake_service

        # Healthy service
        assert extractor.check_service_health() is True

        # Unhealthy service
        extractor._fake_service.healthy = False
        assert extractor.check_service_health() is False

    def test_model_availability_affects_extraction(self, extractor_with_fake_service):
        """Test that model availability determines extraction behavior."""
        extractor = extractor_with_fake_service

        models = extractor.get_available_models()
        assert "gemini/gemini-pro" in models
        assert models["gemini/gemini-pro"]["available"] is True
        assert models["claude-3-haiku-20240307"]["available"] is False

    def test_extraction_filters_papers_correctly(
        self, extractor_with_fake_service, sample_papers_df
    ):
        """Test that only papers with content sources are processed."""
        extractor = extractor_with_fake_service

        # Filter papers - should include those with PDFs or arxiv IDs
        filtered = extractor._filter_papers_for_extraction(sample_papers_df)

        # All 3 papers should be included (2 have PDFs, 2 have arxiv IDs)
        assert len(filtered) == 3
        assert set(filtered.index) == {0, 1, 2}

    def test_extraction_uses_preferred_models(
        self, extractor_with_fake_service, sample_papers_df
    ):
        """Test that extraction tries models in preference order."""
        extractor = extractor_with_fake_service
        fake_service = extractor._fake_service

        # Make preferred model unavailable
        fake_service.available_models["gemini/gemini-pro"]["available"] = False

        # Create minimal PDF content
        pdf_path = Path(sample_papers_df.iloc[1]["pdf_path"])
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("Test PDF content for extraction testing")

        # Extract single paper
        paper = sample_papers_df.iloc[1]
        result = extractor._extract_single_paper(paper)

        # Should have tried next model (gpt-3.5-turbo)
        assert len(fake_service.call_history) > 0
        last_call = fake_service.call_history[-1]
        assert last_call["model"] == "gpt-3.5-turbo"

    def test_extraction_enriches_papers_with_insights(
        self, extractor_with_fake_service, sample_papers_df
    ):
        """Test that extraction adds meaningful insights to papers."""
        extractor = extractor_with_fake_service

        # Create PDF files
        for _, paper in sample_papers_df.iterrows():
            if paper["pdf_path"]:
                pdf_path = Path(paper["pdf_path"])
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_text(f"Content about {paper['title']}")

        # Run extraction
        enriched_df = extractor.extract_all(sample_papers_df, parallel=False)

        # Verify insights were added
        for idx, row in enriched_df.iterrows():
            if row["pdf_path"] or row["arxiv_id"]:
                assert row["extraction_status"] == "success"
                assert pd.notna(row["research_questions"])
                assert pd.notna(row["key_contributions"])
                assert pd.notna(row["awscale"])  # AWScale should be assigned

    def test_extraction_handles_service_failures_gracefully(
        self, extractor_with_fake_service, sample_papers_df
    ):
        """Test graceful handling of service failures."""
        extractor = extractor_with_fake_service

        # Make service unhealthy
        extractor._fake_service.healthy = False

        # Should handle gracefully
        result_df = extractor.extract_all(sample_papers_df)

        # Papers should be unchanged (no extraction)
        assert len(result_df) == len(sample_papers_df)
        assert all(result_df["extraction_status"] == "")

    def test_awscale_assignment_logic(self, extractor_with_fake_service):
        """Test AWScale assignment based on paper characteristics."""
        extractor = extractor_with_fake_service

        # Test different extraction results
        test_cases = [
            {
                "simulation_approach": "Matrix game with human facilitators",
                "human_llm_comparison": "Humans and LLMs showed complementary strengths",
                "expected_awscale": 4,  # Higher for human-in-loop
            },
            {
                "simulation_approach": "Fully automated digital simulation",
                "human_llm_comparison": "LLM-only system",
                "expected_awscale": 2,  # Lower for pure automation
            },
            {
                "simulation_approach": "Tabletop exercise with AI support",
                "human_llm_comparison": "Human decision makers with LLM advisors",
                "expected_awscale": 4,  # Higher for human+AI
            },
        ]

        for case in test_cases:
            awscale = extractor._assign_awscale(case)
            assert (
                awscale == case["expected_awscale"]
            ), f"Failed for: {case['simulation_approach']}"

    def test_parallel_extraction_maintains_consistency(
        self, extractor_with_fake_service, sample_papers_df
    ):
        """Test that parallel and sequential extraction produce same results."""
        extractor = extractor_with_fake_service

        # Create PDF files
        for _, paper in sample_papers_df.iterrows():
            if paper["pdf_path"]:
                pdf_path = Path(paper["pdf_path"])
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_text("Test content")

        # Sequential extraction
        seq_df = extractor.extract_all(sample_papers_df.copy(), parallel=False)

        # Parallel extraction
        par_df = extractor.extract_all(sample_papers_df.copy(), parallel=True)

        # Results should be equivalent
        for col in ["extraction_status", "research_questions", "key_contributions"]:
            if col in seq_df.columns and col in par_df.columns:
                assert seq_df[col].equals(par_df[col])

    def test_content_type_tracking(self, extractor_with_fake_service, sample_papers_df):
        """Test that content type (PDF/TeX/HTML) is properly tracked."""
        extractor = extractor_with_fake_service

        # Create test content
        pdf_path = Path(sample_papers_df.iloc[1]["pdf_path"])
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("PDF content")

        # Mock TeX extraction
        with patch.object(
            extractor, "_extract_tex_content", return_value=("TeX content", True)
        ):
            # Extract papers
            result_df = extractor.extract_all(sample_papers_df)

            # Check content types
            assert result_df.iloc[0]["content_type"] == "tex"  # Has arxiv_id
            assert result_df.iloc[1]["content_type"] == "pdf"  # Has PDF
            assert result_df.iloc[2]["content_type"] in ["tex", "pdf"]  # Has both
