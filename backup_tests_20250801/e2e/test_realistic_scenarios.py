"""E2E tests with realistic scenarios and data."""

import time

import pandas as pd
import pytest

from src.lit_review.extraction import EnhancedLLMExtractor
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher
from src.lit_review.utils import Exporter
from src.lit_review.visualization import Visualizer
from tests.test_data_generators import RealisticTestDataGenerator
from tests.test_doubles import (
    FakeArxivAPI,
    FakeLLMService,
    FakePDFServer,
    RealConfigForTests,
)


@pytest.mark.e2e
@pytest.mark.realistic
class TestRealisticScenarios:
    """Test realistic end-to-end scenarios with quality data."""

    @pytest.fixture
    def realistic_config(self, tmp_path):
        """Create configuration for realistic tests."""
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            search_years=(2022, 2024),
            sample_size=20,  # More papers for realistic test
            pdf_timeout_seconds=10,
            parallel_workers=4,
            # Realistic search terms
            wargame_terms=[
                "wargame",
                "wargaming",
                "war game",
                "strategic simulation",
                "crisis simulation",
                "conflict simulation",
            ],
            llm_terms=[
                "LLM",
                "large language model",
                "GPT",
                "Claude",
                "language model",
                "foundation model",
                "AI agent",
                "artificial intelligence",
            ],
        )
        yield config
        config.cleanup()

    @pytest.fixture
    def realistic_services(self):
        """Set up realistic fake services."""
        # Use consistent seed for reproducibility
        generator = RealisticTestDataGenerator(seed=12345)

        # Create services with realistic data
        arxiv = FakeArxivAPI(seed=12345)
        llm = FakeLLMService(healthy=True)
        pdf = FakePDFServer()

        # Generate PDFs for papers
        for paper in arxiv.papers[:20]:  # First 20 papers
            arxiv_id = paper["id"]
            # Generate unique PDF content for each paper
            pdf_content = f"""
%PDF-1.5
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}

{paper['abstract']}

1. Introduction
This research investigates the intersection of large language models and strategic simulations...

2. Related Work
Previous studies have explored AI in gaming contexts [1,2,3]...

3. Methodology
We employed {generator.generate_simulation_approach(paper)} with careful evaluation...

4. Results
{generator._generate_comparison()}

5. Discussion
Our findings suggest important implications for the field...

6. Conclusion
This work contributes to understanding LLM capabilities in strategic contexts.

References:
[1] Prior work on AI gaming
[2] LLM evaluation studies
[3] Strategic simulation frameworks
            """.encode()
            pdf.add_pdf(arxiv_id, pdf_content)

        return {"arxiv": arxiv, "llm": llm, "pdf": pdf, "generator": generator}

    def test_realistic_research_workflow(
        self, realistic_config, realistic_services, monkeypatch
    ):
        """Test a realistic research workflow from search to analysis."""
        self._patch_external_services(monkeypatch, realistic_services)

        # Step 1: Researcher performs targeted search
        harvester = SearchHarvester(realistic_config)

        # Search with realistic query combinations
        search_queries = [
            "LLM wargaming decision support",
            "large language model strategic simulation",
            "GPT military planning AI",
            "multi-agent crisis management LLM",
        ]

        all_results = []
        for query in search_queries:
            results = harvester.search_all(query, max_results=5)
            all_results.append(results)

        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)

        assert len(combined_df) > 0, "Should find papers for realistic queries"
        assert combined_df["title"].nunique() > 5, "Should find diverse papers"

        # Step 2: Process and deduplicate
        normalizer = Normalizer(realistic_config)
        normalized_df = normalizer.normalize_dataframe(combined_df)
        deduped_df = normalizer.deduplicate(normalized_df)

        print(
            f"\nFound {len(deduped_df)} unique papers from {len(combined_df)} results"
        )

        # Verify deduplication worked
        assert len(deduped_df) <= len(combined_df)
        assert deduped_df["paper_id"].nunique() == len(deduped_df)

        # Step 3: Fetch PDFs with realistic success/failure rates
        pdf_fetcher = PDFFetcher(realistic_config)

        # Set some papers to fail (simulate real-world issues)
        realistic_services["pdf"].failure_rate = 0.1  # 10% random failures

        pdf_df = pdf_fetcher.fetch_pdfs(deduped_df)

        # Check realistic distribution of results
        status_counts = pdf_df["pdf_status"].value_counts()
        print(f"\nPDF fetch results: {status_counts.to_dict()}")

        successful = pdf_df[pdf_df["pdf_status"].str.contains("Union[downloaded, cached]")]
        failed = pdf_df[pdf_df["pdf_status"].isin(["not_found", "error"])]

        assert len(successful) > len(failed), "Most PDFs should succeed"
        assert len(failed) > 0, "Some failures expected in realistic scenario"

        # Step 4: Extract insights with varying quality
        extractor = EnhancedLLMExtractor(realistic_config)

        # Make extraction more realistic - some papers harder to process
        def realistic_extract(text, model, **kwargs):
            # Simulate varying extraction quality
            if len(text) < 1000:  # Short PDFs might fail
                return {
                    "success": False,
                    "error": "Insufficient content for extraction",
                }

            # Use the generator for realistic extraction
            paper_dict = {"title": "Unknown", "abstract": text[:500]}
            extracted = realistic_services["generator"].generate_extraction_results(
                paper_dict
            )

            return {
                "success": True,
                "extracted_data": extracted,
                "model_used": model,
                "tokens_used": len(text) // 4,
                "extraction_confidence": extracted["extraction_confidence"],
            }

        realistic_services["llm"].extract = realistic_extract

        extracted_df = extractor.extract_all(pdf_df, parallel=True)

        # Verify extraction results
        extraction_success = extracted_df["extraction_status"] == "success"
        assert extraction_success.sum() > 0, "Should extract from some papers"

        # Check extraction quality indicators
        successful_extractions = extracted_df[extraction_success]
        if "extraction_confidence" in successful_extractions.columns:
            avg_confidence = successful_extractions["extraction_confidence"].mean()
            assert 0.5 < avg_confidence < 1.0, "Confidence should be realistic"

        # Step 5: Generate comprehensive visualizations
        visualizer = Visualizer(realistic_config)
        figures = visualizer.create_all_visualizations(extracted_df)

        assert len(figures) >= 5, "Should create multiple visualizations"

        # Verify visualizations capture realistic patterns
        summary = visualizer.create_summary_report(extracted_df)

        # Check for realistic distribution in summary
        if "year_range" in summary:
            assert "2022" in summary["year_range"] or "2023" in summary["year_range"]

        if "llm_families" in summary:
            # Should see variety of LLM families
            assert len(summary["llm_families"]) > 2

        # Step 6: Export comprehensive results
        exporter = Exporter(realistic_config)

        # Create research package
        archive_path = exporter.export_full_package(
            extraction_df=extracted_df,
            figures=figures,
            summary=summary,
            include_failed=True,  # Include failed extractions for transparency
        )

        assert archive_path.exists()
        assert archive_path.stat().st_size > 10000  # Should be substantial

        # Verify package contents
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            files = zf.namelist()

            # Should have various file types
            assert any(".csv" in f for f in files), "Should include data"
            assert any(
                ".png" in f or ".pdf" in f for f in files
            ), "Should include figures"
            assert any("README" in f for f in files), "Should include documentation"

    def test_incremental_research_sessions(
        self, realistic_config, realistic_services, monkeypatch
    ):
        """Test realistic incremental research over multiple sessions."""
        self._patch_external_services(monkeypatch, realistic_services)

        # Session 1: Initial exploration
        harvester = SearchHarvester(realistic_config)
        session1_results = harvester.search_all("LLM wargaming", max_results=10)

        # Save session 1 results
        checkpoint_file = realistic_config.data_dir / "session1_checkpoint.csv"
        session1_results.to_csv(checkpoint_file, index=False)

        # Session 2: Refined search based on initial findings
        session1_df = pd.read_csv(checkpoint_file)

        # Analyze what we found
        keywords = []
        for abstract in session1_df["abstract"].dropna()[:5]:
            if "multi-agent" in abstract.lower():
                keywords.append("multi-agent")
            if "strategic" in abstract.lower():
                keywords.append("strategic planning")

        # Refined search
        if keywords:
            refined_query = f"LLM {' OR '.join(set(keywords))}"
            session2_results = harvester.search_all(refined_query, max_results=10)

            # Combine sessions
            combined = pd.concat([session1_df, session2_results], ignore_index=True)

            # Deduplicate across sessions
            normalizer = Normalizer(realistic_config)
            final_df = normalizer.deduplicate(normalizer.normalize_dataframe(combined))

            assert len(final_df) >= len(
                session1_df
            ), "Should preserve or expand dataset"
            assert final_df["paper_id"].nunique() == len(
                final_df
            ), "Should have unique papers"

    def test_collaborative_annotation_workflow(
        self, realistic_config, realistic_services, monkeypatch
    ):
        """Test realistic collaborative research workflow."""
        self._patch_external_services(monkeypatch, realistic_services)

        # Multiple researchers working on same dataset
        harvester = SearchHarvester(realistic_config)
        base_results = harvester.search_all("LLM strategic simulation", max_results=15)

        # Researcher 1: Focus on technical aspects
        researcher1_annotations = []
        for idx, paper in base_results.iterrows():
            if any(
                term in paper.get("title", "").lower()
                for term in ["framework", "architecture", "system"]
            ):
                researcher1_annotations.append(
                    {
                        "paper_id": idx,
                        "technical_relevance": "high",
                        "annotator": "researcher1",
                        "notes": "Strong technical contribution",
                    }
                )

        # Researcher 2: Focus on applications
        researcher2_annotations = []
        for idx, paper in base_results.iterrows():
            if any(
                term in paper.get("abstract", "").lower()
                for term in ["application", "case study", "evaluation"]
            ):
                researcher2_annotations.append(
                    {
                        "paper_id": idx,
                        "application_relevance": "high",
                        "annotator": "researcher2",
                        "notes": "Practical application demonstrated",
                    }
                )

        # Merge annotations
        annotations_df = pd.DataFrame(researcher1_annotations + researcher2_annotations)

        if len(annotations_df) > 0:
            # Check for papers annotated by both
            overlap = annotations_df.groupby("paper_id").size()
            multi_annotated = overlap[overlap > 1]

            print(f"\nPapers annotated by multiple researchers: {len(multi_annotated)}")

    def test_error_recovery_and_resilience(
        self, realistic_config, realistic_services, monkeypatch
    ):
        """Test system resilience to various failure modes."""
        self._patch_external_services(monkeypatch, realistic_services)

        # Introduce various failure modes
        harvester = SearchHarvester(realistic_config)

        # Test 1: Partial service failures
        realistic_services["pdf"].failure_rate = 0.3  # 30% PDF failures
        realistic_services["llm"].available_models["gemini/gemini-pro"]["available"] = (
            False
        )

        # Should still complete workflow
        results = harvester.search_all("LLM wargaming", max_results=10)
        normalizer = Normalizer(realistic_config)
        normalized = normalizer.normalize_dataframe(results)

        pdf_fetcher = PDFFetcher(realistic_config)
        pdf_results = pdf_fetcher.fetch_pdfs(normalized)

        # Check graceful degradation
        success_rate = (pdf_results["pdf_status"] == "downloaded").mean()
        assert 0.5 < success_rate < 0.8, "Should have partial success despite failures"

        # Test 2: Recovery after transient failures
        realistic_services["pdf"].failure_rate = 0.0  # Fix the issue

        # Retry failed PDFs
        failed_papers = pdf_results[pdf_results["pdf_status"] != "downloaded"]
        if len(failed_papers) > 0:
            retry_results = pdf_fetcher.fetch_pdfs(failed_papers)
            new_success_rate = (retry_results["pdf_status"] == "downloaded").mean()
            assert new_success_rate > success_rate, "Should improve after retry"

        # Test 3: Fallback extraction strategies
        extractor = EnhancedLLMExtractor(realistic_config)
        extracted = extractor.extract_all(pdf_results, parallel=False)

        # Should use fallback model
        if "extraction_model" in extracted.columns:
            models_used = extracted["extraction_model"].dropna().unique()
            assert "gpt-3.5-turbo" in models_used, "Should fallback to available model"

    def test_performance_with_realistic_load(
        self, realistic_config, realistic_services, monkeypatch
    ):
        """Test performance under realistic research loads."""
        import os

        import psutil

        self._patch_external_services(monkeypatch, realistic_services)

        # Simulate realistic research load
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple search queries in parallel
        harvester = SearchHarvester(realistic_config)
        queries = [
            "LLM military planning",
            "GPT strategic simulation",
            "multi-agent wargaming AI",
            "language model crisis management",
            "transformer conflict simulation",
        ]

        # Parallel search simulation
        all_results = []
        for query in queries:
            results = harvester.search_all(query, max_results=20)
            all_results.append(results)

        combined = pd.concat(all_results, ignore_index=True)

        # Full pipeline processing
        normalizer = Normalizer(realistic_config)
        normalized = normalizer.normalize_dataframe(combined)
        deduped = normalizer.deduplicate(normalized)

        # Measure performance
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        duration = end_time - start_time
        memory_used = end_memory - start_memory
        papers_per_second = len(deduped) / duration

        print("\nPerformance metrics:")
        print(f"  Processed {len(deduped)} unique papers from {len(combined)} results")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory used: {memory_used:.1f}MB")
        print(f"  Throughput: {papers_per_second:.1f} papers/second")

        # Performance assertions
        assert duration < 60, "Should complete in reasonable time"
        assert memory_used < 1000, "Should not use excessive memory"
        assert papers_per_second > 0.5, "Should maintain reasonable throughput"

    def _patch_external_services(self, monkeypatch, services):
        """Patch external services with realistic fakes."""

        # Patch arXiv with realistic API
        def mock_arxiv_search(*args, **kwargs):
            class MockSearch:
                def results(self):
                    query = args[0] if args else kwargs.get("query", "")
                    max_results = kwargs.get("max_results", 10)
                    return services["arxiv"].search(query, max_results)

            return MockSearch()

        monkeypatch.setattr("arxiv.Search", mock_arxiv_search)

        # Patch HTTP requests
        def mock_get(url, *args, **kwargs):
            if "arxiv.org/pdf" in url:
                arxiv_id = url.split("/")[-1].replace(".pdf", "")
                content, status = services["pdf"].serve_pdf(arxiv_id)

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
            elif "localhost:8000" in url:
                # LLM service endpoints
                if "/health" in url:

                    class MockResponse:
                        status_code = 200 if services["llm"].healthy else 503

                    return MockResponse()
                elif "/models" in url:

                    class MockResponse:
                        status_code = 200

                        def json(self):
                            return services["llm"].get_models()

                    return MockResponse()
            else:
                raise ValueError(f"Unexpected URL: {url}")

        def mock_post(url, json=None, *args, **kwargs):
            if "localhost:8000/extract" in url:
                result = services["llm"].extract(json["text"], json["model"], **json)

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
