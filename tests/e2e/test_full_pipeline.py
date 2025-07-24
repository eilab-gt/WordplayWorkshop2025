"""End-to-end tests for the complete literature review pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.lit_review.extraction import EnhancedLLMExtractor
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher
from src.lit_review.utils import Exporter
from src.lit_review.visualization import Visualizer
from tests.test_doubles import (
    FakeArxivAPI,
    FakeLLMService,
    FakePDFServer,
    RealConfigForTests,
)


@pytest.mark.e2e
@pytest.mark.slow
class TestFullPipelineIntegration:
    """Test complete pipeline from search to export."""

    @pytest.fixture(scope="class")
    def e2e_config(self):
        """Create configuration for E2E tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RealConfigForTests(
                cache_dir=Path(tmpdir) / "cache",
                output_dir=Path(tmpdir) / "output",
                data_dir=Path(tmpdir) / "data",
                log_dir=Path(tmpdir) / "logs",
                # Smaller limits for E2E tests
                search_years=(2023, 2024),
                sample_size=5,  # Process only 5 papers
                pdf_timeout_seconds=5,
                parallel_workers=2,
            )
            yield config

    @pytest.fixture
    def fake_services(self):
        """Set up all fake services for E2E test."""
        return {
            "arxiv": FakeArxivAPI(),
            "llm": FakeLLMService(healthy=True),
            "pdf": FakePDFServer(),
        }

    def test_complete_pipeline_produces_expected_outputs(
        self, e2e_config, fake_services, monkeypatch
    ):
        """
        Test complete pipeline flow:
        1. Search for papers
        2. Normalize and deduplicate
        3. Fetch PDFs
        4. Extract insights with LLM
        5. Create visualizations
        6. Export results
        """
        # Patch external services to use fakes
        self._patch_external_services(monkeypatch, fake_services)

        # Step 1: Search for papers
        harvester = SearchHarvester(e2e_config)
        search_results = self._mock_search(harvester, fake_services["arxiv"])

        assert len(search_results) > 0, "Search should find papers"
        assert all(
            col in search_results.columns for col in ["title", "authors", "year"]
        )

        # Save raw results
        raw_path = e2e_config.data_dir / "raw" / "papers_raw.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        search_results.to_csv(raw_path, index=False)

        # Step 2: Normalize and deduplicate
        normalizer = Normalizer(e2e_config)
        normalized_df = normalizer.normalize(search_results)
        deduped_df = normalizer.deduplicate(normalized_df)

        assert len(deduped_df) <= len(
            search_results
        ), "Deduplication should not add papers"
        assert "paper_id" in deduped_df.columns, "Should assign paper IDs"

        # Step 3: Fetch PDFs
        pdf_fetcher = PDFFetcher(e2e_config)
        pdf_df = pdf_fetcher.fetch_pdfs(deduped_df)

        successful_pdfs = pdf_df[pdf_df["pdf_status"].str.contains("downloaded|cached")]
        assert len(successful_pdfs) > 0, "Should download at least some PDFs"

        # Step 4: Extract insights
        extractor = EnhancedLLMExtractor(e2e_config)
        extracted_df = extractor.extract_all(pdf_df, parallel=False)

        successful_extractions = extracted_df[
            extracted_df["extraction_status"] == "success"
        ]
        assert (
            len(successful_extractions) > 0
        ), "Should extract from at least some papers"
        assert all(
            col in extracted_df.columns
            for col in ["research_questions", "key_contributions", "awscale"]
        )

        # Save extraction results
        extraction_path = e2e_config.data_dir / "extracted" / "extraction.csv"
        extraction_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_df.to_csv(extraction_path, index=False)

        # Step 5: Create visualizations
        visualizer = Visualizer(e2e_config)
        figures = visualizer.create_all_visualizations(extracted_df)

        assert len(figures) > 0, "Should create at least some visualizations"
        assert all(fig.exists() for fig in figures), "All figures should be saved"

        # Step 6: Export results
        exporter = Exporter(e2e_config)

        # Create export package
        summary = visualizer.create_summary_report(extracted_df)
        archive_path = exporter.export_full_package(
            extraction_df=extracted_df, figures=figures, summary=summary
        )

        assert archive_path.exists(), "Export archive should be created"
        assert archive_path.suffix == ".zip", "Should be a zip file"

        # Verify archive contents
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            namelist = zf.namelist()
            assert any("extraction.csv" in name for name in namelist)
            assert any("README.md" in name for name in namelist)
            assert any(".png" in name or ".pdf" in name for name in namelist)

        # Create BibTeX export
        bibtex_path = exporter.export_bibtex(extracted_df)
        assert bibtex_path.exists(), "BibTeX file should be created"
        assert bibtex_path.read_text().count("@article") >= len(successful_extractions)

    def test_pipeline_handles_partial_failures_gracefully(
        self, e2e_config, fake_services, monkeypatch
    ):
        """Test pipeline continues when some operations fail."""
        self._patch_external_services(monkeypatch, fake_services)

        # Make some services fail partially
        fake_services["pdf"].rate_limit_enabled = True
        fake_services["llm"].available_models["gemini/gemini-pro"]["available"] = False

        # Run simplified pipeline
        harvester = SearchHarvester(e2e_config)
        search_results = self._mock_search(harvester, fake_services["arxiv"])

        # Process with failures
        normalizer = Normalizer(e2e_config)
        normalized_df = normalizer.normalize(search_results)

        pdf_fetcher = PDFFetcher(e2e_config)
        pdf_df = pdf_fetcher.fetch_pdfs(normalized_df)

        # Some PDFs should fail due to rate limiting
        failed_pdfs = pdf_df[pdf_df["pdf_status"] == "not_found"]
        assert len(failed_pdfs) > 0, "Some PDFs should fail"

        # But some should succeed
        successful_pdfs = pdf_df[pdf_df["pdf_status"].str.contains("downloaded|cached")]
        assert len(successful_pdfs) > 0, "Some PDFs should succeed"

        # Extraction should use fallback model
        extractor = EnhancedLLMExtractor(e2e_config)
        extracted_df = extractor.extract_all(pdf_df)

        # Check that fallback model was used
        successful = extracted_df[extracted_df["extraction_status"] == "success"]
        if len(successful) > 0:
            assert any(successful["extraction_model"] == "gpt-3.5-turbo")

    def test_pipeline_checkpoint_and_resume(
        self, e2e_config, fake_services, monkeypatch
    ):
        """Test pipeline can be interrupted and resumed."""
        self._patch_external_services(monkeypatch, fake_services)

        # Start pipeline
        harvester = SearchHarvester(e2e_config)
        search_results = self._mock_search(harvester, fake_services["arxiv"])

        # Save intermediate state
        checkpoint_path = e2e_config.data_dir / "checkpoint.csv"
        search_results.to_csv(checkpoint_path, index=False)

        # "Interrupt" and resume
        resumed_df = pd.read_csv(checkpoint_path)
        assert resumed_df.equals(search_results), "Should restore exact state"

        # Continue processing
        normalizer = Normalizer(e2e_config)
        normalized_df = normalizer.normalize(resumed_df)
        assert len(normalized_df) == len(search_results)

    def test_pipeline_produces_reproducible_results(
        self, e2e_config, fake_services, monkeypatch
    ):
        """Test pipeline produces same results when run twice."""
        self._patch_external_services(monkeypatch, fake_services)

        # Run pipeline twice
        results = []
        for run in range(2):
            harvester = SearchHarvester(e2e_config)
            search_df = self._mock_search(harvester, fake_services["arxiv"])

            normalizer = Normalizer(e2e_config)
            normalized_df = normalizer.normalize(search_df)
            deduped_df = normalizer.deduplicate(normalized_df)

            # Sort for comparison
            deduped_df = deduped_df.sort_values("title").reset_index(drop=True)
            results.append(deduped_df)

        # Results should be identical
        pd.testing.assert_frame_equal(results[0], results[1])

    def test_pipeline_performance_metrics(self, e2e_config, fake_services, monkeypatch):
        """Test pipeline completes within reasonable time and resource limits."""
        import os
        import time

        import psutil

        self._patch_external_services(monkeypatch, fake_services)

        # Record start metrics
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run mini pipeline
        harvester = SearchHarvester(e2e_config)
        search_df = self._mock_search(harvester, fake_services["arxiv"], max_results=10)

        normalizer = Normalizer(e2e_config)
        normalized_df = normalizer.normalize(search_df)

        pdf_fetcher = PDFFetcher(e2e_config)
        pdf_df = pdf_fetcher.fetch_pdfs(normalized_df)

        # Record end metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        duration = end_time - start_time
        memory_used = end_memory - start_memory

        # Performance assertions
        assert duration < 30, f"Pipeline took too long: {duration:.1f}s"
        assert memory_used < 500, f"Pipeline used too much memory: {memory_used:.1f}MB"

        # Log performance
        print("\nPerformance metrics:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Memory: {memory_used:.1f}MB")
        print(f"  Papers/sec: {len(search_df) / duration:.1f}")

    def _patch_external_services(self, monkeypatch, fake_services):
        """Patch all external service calls to use fakes."""

        # Patch arXiv
        def mock_arxiv_search(*args, **kwargs):
            class MockSearch:
                def results(self):
                    query = args[0] if args else kwargs.get("query", "")
                    return fake_services["arxiv"].search(query)

            return MockSearch()

        monkeypatch.setattr("arxiv.Search", mock_arxiv_search)

        # Patch HTTP requests for PDFs and LLM service
        def mock_get(url, *args, **kwargs):
            if "arxiv.org/pdf" in url:
                arxiv_id = url.split("/")[-1].replace(".pdf", "")
                content, status = fake_services["pdf"].serve_pdf(arxiv_id)

                class MockResponse:
                    status_code = status
                    headers = {
                        "Content-Type": (
                            "application/pdf" if status == 200 else "text/plain"
                        )
                    }
                    content = content

                    def iter_content(self, chunk_size):
                        return [
                            content[i : i + chunk_size]
                            for i in range(0, len(content), chunk_size)
                        ]

                    def raise_for_status(self):
                        if self.status_code >= 400:
                            raise Exception(f"HTTP {self.status_code}")

                return MockResponse()
            elif "localhost:8000/health" in url:

                class MockResponse:
                    status_code = 200 if fake_services["llm"].healthy else 503

                return MockResponse()
            elif "localhost:8000/models" in url:

                class MockResponse:
                    status_code = 200

                    def json(self):
                        return fake_services["llm"].get_models()

                return MockResponse()
            else:
                raise ValueError(f"Unexpected URL: {url}")

        def mock_post(url, json=None, *args, **kwargs):
            if "localhost:8000/extract" in url:
                result = fake_services["llm"].extract(
                    json["text"], json["model"], **json
                )

                class MockResponse:
                    status_code = 200

                    def json(self):
                        return result

                return MockResponse()
            else:
                raise ValueError(f"Unexpected POST URL: {url}")

        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.post", mock_post)
        monkeypatch.setattr("requests.Session.get", mock_get)
        monkeypatch.setattr("requests.Session.post", mock_post)

    def _mock_search(self, harvester, fake_arxiv, max_results=5):
        """Perform a mock search using the harvester."""
        # The fake_arxiv already has realistic papers pre-generated
        # Just run the search with appropriate query terms
        results = harvester.search_all(
            sources=["arxiv"],
            max_results_per_source=max_results,
            custom_query="LLM wargaming strategic simulation multi-agent",
        )
        return results
