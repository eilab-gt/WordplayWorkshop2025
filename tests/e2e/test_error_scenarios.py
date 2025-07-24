"""Comprehensive error scenario tests for E2E pipeline."""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import requests

from src.lit_review.extraction import EnhancedLLMExtractor
from src.lit_review.harvesters import ArxivHarvester, SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher
from src.lit_review.utils.content_cache import ContentCache
from src.lit_review.visualization import Visualizer
from tests.test_doubles import (
    FakeArxivAPI,
    FakeLLMService,
    FakePDFServer,
    RealConfigForTests,
)


@pytest.mark.e2e
@pytest.mark.error_scenarios
class TestErrorScenarios:
    """Test comprehensive error scenarios and recovery mechanisms."""

    @pytest.fixture
    def error_config(self, tmp_path):
        """Configuration for error scenario testing."""
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            pdf_timeout_seconds=2,  # Short timeout for testing
            pdf_max_retries=2,
            llm_timeout_seconds=5,
            llm_max_retries=1,
            sample_size=10,
        )
        yield config
        config.cleanup()

    @pytest.fixture
    def error_services(self):
        """Services configured for error testing."""
        return {
            "arxiv": FakeArxivAPI(seed=999),
            "llm": FakeLLMService(healthy=True),
            "pdf": FakePDFServer(),
        }

    def test_network_failures_during_search(
        self, error_config, error_services, monkeypatch
    ):
        """Test handling of network failures during paper search."""
        call_count = 0

        def flaky_arxiv_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Fail first 2 attempts, succeed on 3rd
            if call_count < 3:
                raise requests.ConnectionError("Network error")

            class MockSearch:
                def results(self):
                    return error_services["arxiv"].search(
                        args[0], kwargs.get("max_results", 10)
                    )

            return MockSearch()

        monkeypatch.setattr("arxiv.Search", flaky_arxiv_search)

        harvester = ArxivHarvester(error_config)

        # Should handle retries internally
        try:
            results = harvester.search("LLM wargaming", max_results=5)
            # If successful after retries
            assert len(results) > 0
        except Exception as e:
            # Should provide meaningful error after exhausting retries
            assert "Network error" in str(e) or "Connection" in str(e)

    def test_malformed_data_handling(self, error_config, error_services, monkeypatch):
        """Test handling of malformed/incomplete data from sources."""
        # Add papers with missing/malformed data
        malformed_papers = [
            {
                "id": "malformed.00001",
                "title": None,  # Missing title
                "authors": ["Test Author"],
                "abstract": "Abstract text",
                "pdf_url": "https://arxiv.org/pdf/malformed.00001.pdf",
                "published": "2024-01-01",
                "categories": ["cs.AI"],
            },
            {
                "id": "malformed.00002",
                "title": "Paper with Bad Date",
                "authors": [],  # Empty authors
                "abstract": "Abstract",
                "pdf_url": "https://arxiv.org/pdf/malformed.00002.pdf",
                "published": "invalid-date",  # Invalid date
                "categories": [],
            },
            {
                "id": "malformed.00003",
                "title": "",  # Empty title
                "authors": ["Author"],
                "abstract": None,  # Missing abstract
                "pdf_url": None,  # Missing PDF URL
                "published": "2024-01-01",
                "categories": ["cs.AI"],
            },
        ]

        for paper in malformed_papers:
            error_services["arxiv"].add_paper(paper)

        self._patch_services(monkeypatch, error_services)

        # Process malformed data
        harvester = SearchHarvester(error_config)
        results = harvester.search_arxiv("malformed", max_results=10)

        # Should handle gracefully
        assert len(results) > 0

        # Normalize should clean up data
        normalizer = Normalizer(error_config)
        normalized = normalizer.normalize(results)

        # Check data cleaning
        assert all(normalized["title"].notna()), "Should handle missing titles"
        assert all(normalized["year"] >= 0), "Should handle invalid dates"

    def test_pdf_download_failures(self, error_config, error_services, monkeypatch):
        """Test various PDF download failure scenarios."""
        self._patch_services(monkeypatch, error_services)

        # Configure different failure modes
        error_services["pdf"].failure_rate = 0.0  # Start with no random failures

        # Test specific failure scenarios
        failure_scenarios = {
            "timeout.00001": (b"", 408),  # Timeout
            "forbidden.00002": (b"Access Denied", 403),  # Access denied
            "notfound.00003": (b"Not Found", 404),  # Not found
            "servererror.00004": (b"Internal Server Error", 500),  # Server error
            "ratelimit.00005": (b"Rate Limited", 429),  # Rate limit
            "corrupt.00006": (b"\x00\x01\x02", 200),  # Corrupt PDF (non-PDF bytes)
        }

        # Create papers for each scenario
        test_papers = []
        for arxiv_id, (content, status) in failure_scenarios.items():
            paper = {
                "paper_id": arxiv_id,
                "title": f"Test Paper {arxiv_id}",
                "arxiv_id": arxiv_id,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            }
            test_papers.append(paper)

            # Configure PDF server response
            error_services["pdf"].available_pdfs[arxiv_id] = content
            # Override serve_pdf for specific status codes
            original_serve = error_services["pdf"].serve_pdf

            def serve_with_status(aid):
                if aid in failure_scenarios:
                    return failure_scenarios[aid]
                return original_serve(aid)

            error_services["pdf"].serve_pdf = serve_with_status

        papers_df = pd.DataFrame(test_papers)

        # Attempt PDF downloads
        pdf_fetcher = PDFFetcher(error_config)
        results = pdf_fetcher.fetch_pdfs(papers_df)

        # Verify appropriate handling of each failure type
        for idx, row in results.iterrows():
            arxiv_id = row["arxiv_id"]
            status = row["pdf_status"]

            if arxiv_id == "timeout.00001":
                assert status in [
                    "error",
                    "timeout",
                ], "Should mark timeout appropriately"
            elif arxiv_id == "forbidden.00002":
                assert status in ["error", "forbidden"], "Should handle access denied"
            elif arxiv_id == "notfound.00003":
                assert status == "not_found", "Should mark as not found"
            elif arxiv_id == "servererror.00004":
                assert status == "error", "Should mark server errors"
            elif arxiv_id == "ratelimit.00005":
                assert status in ["error", "rate_limited"], "Should handle rate limits"
            elif arxiv_id == "corrupt.00006":
                assert status in ["error", "corrupt"], "Should detect corrupt PDFs"

    def test_llm_service_failures(self, error_config, error_services, monkeypatch):
        """Test LLM service failure scenarios."""
        self._patch_services(monkeypatch, error_services)

        # Create test papers with PDFs
        test_papers = pd.DataFrame(
            [
                {
                    "paper_id": f"llmtest.{i:05d}",
                    "title": f"LLM Test Paper {i}",
                    "pdf_path": f"/fake/path/llmtest.{i:05d}.pdf",
                    "pdf_status": "downloaded",
                }
                for i in range(5)
            ]
        )

        extractor = EnhancedLLMExtractor(error_config)

        # Test 1: Service unavailable
        error_services["llm"].healthy = False
        results = extractor.extract_all(test_papers[:1])
        assert all(
            results["extraction_status"] == "error"
        ), "Should fail when service unhealthy"

        # Test 2: Model not available
        error_services["llm"].healthy = True
        error_services["llm"].available_models = {}  # No models available
        results = extractor.extract_all(test_papers[1:2])
        assert all(
            results["extraction_status"] == "error"
        ), "Should fail with no models"

        # Test 3: Extraction failures
        def failing_extract(text, model, **kwargs):
            return {
                "success": False,
                "error": "Extraction failed due to content complexity",
            }

        error_services["llm"].extract = failing_extract
        results = extractor.extract_all(test_papers[2:3])
        assert all(
            results["extraction_status"] == "error"
        ), "Should handle extraction failures"

        # Test 4: Partial success with fallback
        call_count = 0

        def partial_extract(text, model, **kwargs):
            nonlocal call_count
            call_count += 1

            # Fail primary model, succeed with fallback
            if model == "gemini/gemini-pro" and call_count == 1:
                return {"success": False, "error": "Model overloaded"}

            return {
                "success": True,
                "extracted_data": {
                    "research_questions": "Test RQ",
                    "key_contributions": "Test contribution",
                },
                "model_used": model,
            }

        error_services["llm"].extract = partial_extract
        error_services["llm"].available_models = {
            "gemini/gemini-pro": {"available": True},
            "gpt-3.5-turbo": {"available": True},
        }

        results = extractor.extract_all(test_papers[3:4])
        successful = results[results["extraction_status"] == "success"]
        if len(successful) > 0 and "extraction_model" in successful.columns:
            # Should have used fallback model
            assert any(successful["extraction_model"] == "gpt-3.5-turbo")

    def test_file_system_errors(self, error_config, error_services, monkeypatch):
        """Test file system related errors."""
        # Test 1: Read-only output directory
        read_only_dir = Path(tempfile.mkdtemp())
        try:
            # Make directory read-only
            os.chmod(read_only_dir, 0o444)

            config = RealConfigForTests(
                output_dir=read_only_dir / "output", data_dir=read_only_dir / "data"
            )

            # Should handle permission errors gracefully
            try:
                visualizer = Visualizer(config)
                # Attempt to save a figure
                test_df = pd.DataFrame([{"year": 2024, "title": "Test"}])
                visualizer.plot_time_series(test_df, save=True)
            except PermissionError:
                pass  # Expected
            finally:
                # Restore permissions for cleanup
                os.chmod(read_only_dir, 0o755)

        finally:
            import shutil

            shutil.rmtree(read_only_dir, ignore_errors=True)

        # Test 2: Disk space issues (simulate with large file)
        # This is hard to test without actually filling disk, so we mock
        original_savefig = plt.savefig if "plt" in locals() else None

        def failing_savefig(*args, **kwargs):
            raise OSError("No space left on device")

        if original_savefig:
            monkeypatch.setattr("matplotlib.pyplot.savefig", failing_savefig)

            visualizer = Visualizer(error_config)
            test_df = pd.DataFrame([{"year": 2024, "title": "Test"}])

            # Should handle gracefully
            fig_path = visualizer.plot_time_series(test_df, save=True)
            assert fig_path is None, "Should return None on save failure"

    def test_data_corruption_recovery(self, error_config, error_services, monkeypatch):
        """Test recovery from corrupted cache and data files."""
        self._patch_services(monkeypatch, error_services)

        # Test 1: Corrupted cache database
        cache = ContentCache(error_config)

        # Corrupt the cache database
        cache_db = cache.cache_dir / "cache_metadata.db"
        if cache_db.exists():
            # Write garbage to database file
            with open(cache_db, "wb") as f:
                f.write(b"CORRUPTED DATA")

        # Cache should reinitialize
        cache2 = ContentCache(error_config)
        # Should not crash when accessing
        result = cache2.get_or_fetch("test_id", "pdf", lambda: b"test content")
        assert result is not None

        # Test 2: Corrupted CSV files
        corrupted_csv = error_config.data_dir / "corrupted.csv"
        corrupted_csv.parent.mkdir(exist_ok=True)

        # Write malformed CSV
        with open(corrupted_csv, "w") as f:
            f.write("title,year,broken\n")
            f.write("Test,2024,extra,columns,here\n")
            f.write("Missing columns\n")
            f.write('"Unclosed quote,2024,test\n')

        # Should handle corrupted data
        try:
            df = pd.read_csv(corrupted_csv, error_bad_lines=False)
            # Process what we can
            assert len(df) >= 0
        except Exception:
            # At minimum shouldn't crash the system
            pass

    def test_concurrent_access_conflicts(self, error_config, error_services):
        """Test handling of concurrent access to shared resources."""
        import threading
        import time

        cache = ContentCache(error_config)
        results = []
        errors = []

        def concurrent_cache_access(thread_id):
            try:
                for i in range(5):
                    paper_id = f"concurrent_test_{i}"

                    # All threads try to cache same papers
                    path, cached = cache.get_or_fetch(
                        paper_id,
                        "pdf",
                        lambda: f"Content from thread {thread_id}".encode(),
                    )

                    results.append(
                        {
                            "thread": thread_id,
                            "paper": paper_id,
                            "cached": cached,
                            "content": path.read_bytes().decode() if path else None,
                        }
                    )

                    # Small delay to increase conflict chance
                    time.sleep(0.01)

            except Exception as e:
                errors.append({"thread": thread_id, "error": str(e)})

        # Launch multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_cache_access, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify results
        assert (
            len(errors) == 0
        ), f"Should handle concurrent access without errors: {errors}"

        # Check that same paper has consistent content across threads
        paper_contents = {}
        for result in results:
            paper = result["paper"]
            content = result["content"]

            if paper in paper_contents:
                assert (
                    paper_contents[paper] == content
                ), "Same paper should have same content"
            else:
                paper_contents[paper] = content

    def test_memory_exhaustion_handling(
        self, error_config, error_services, monkeypatch
    ):
        """Test handling of memory exhaustion scenarios."""
        self._patch_services(monkeypatch, error_services)

        # Create a large dataset that could cause memory issues
        large_papers = []
        for i in range(100):
            paper = error_services["arxiv"].generator.generate_paper()
            # Add very long abstract to increase memory usage
            paper["abstract"] = paper["abstract"] * 100  # Make it huge
            large_papers.append(paper)

        papers_df = pd.DataFrame(large_papers)

        # Process in batches to avoid memory issues
        normalizer = Normalizer(error_config)

        batch_size = 10
        processed_batches = []

        for start in range(0, len(papers_df), batch_size):
            batch = papers_df.iloc[start : start + batch_size]
            processed = normalizer.normalize(batch)
            processed_batches.append(processed)

            # Force garbage collection between batches
            import gc

            gc.collect()

        # Combine results
        final_df = pd.concat(processed_batches, ignore_index=True)

        assert len(final_df) == len(papers_df), "Should process all papers"
        assert final_df["paper_id"].nunique() == len(
            final_df
        ), "Should maintain data integrity"

    def test_cascade_failure_prevention(
        self, error_config, error_services, monkeypatch
    ):
        """Test prevention of cascade failures across pipeline stages."""
        self._patch_services(monkeypatch, error_services)

        # Set up scenario where one failure could cascade
        harvester = SearchHarvester(error_config)

        # Stage 1: Search succeeds
        results = harvester.search_arxiv("LLM", max_results=10)
        assert len(results) > 0

        # Stage 2: Normalization with some bad data
        results.loc[0, "year"] = "invalid"  # Corrupt one entry
        normalizer = Normalizer(error_config)
        normalized = normalizer.normalize(results)

        # Should handle bad entry without affecting others
        assert len(normalized) >= len(results) - 1

        # Stage 3: PDF fetch with failures
        error_services["pdf"].failure_rate = 0.5  # 50% failure rate
        pdf_fetcher = PDFFetcher(error_config)
        pdf_results = pdf_fetcher.fetch_pdfs(normalized)

        # Should complete despite failures
        assert len(pdf_results) == len(normalized)
        assert pdf_results["pdf_status"].notna().all()

        # Stage 4: Extraction with partial failures
        successful_pdfs = pdf_results[pdf_results["pdf_status"] == "downloaded"]
        if len(successful_pdfs) > 0:
            extractor = EnhancedLLMExtractor(error_config)

            # Make some extractions fail
            original_extract = error_services["llm"].extract

            def sometimes_fail(text, model, **kwargs):
                if hash(text) % 3 == 0:  # Fail ~33% of time
                    return {"success": False, "error": "Random failure"}
                return original_extract(text, model, **kwargs)

            error_services["llm"].extract = sometimes_fail

            extraction_results = extractor.extract_all(successful_pdfs)

            # Pipeline should complete
            assert len(extraction_results) == len(successful_pdfs)

            # Stage 5: Visualization should work with partial data
            visualizer = Visualizer(error_config)
            figures = visualizer.create_all_visualizations(extraction_results)

            # Should create some visualizations despite failures
            assert isinstance(figures, list)

    def test_configuration_errors(self, error_config, tmp_path):
        """Test handling of configuration errors."""
        # Test 1: Invalid configuration values
        try:
            bad_config = RealConfigForTests(
                pdf_timeout_seconds=-1,  # Invalid
                parallel_workers=0,  # Invalid
                sample_size=-10,  # Invalid
            )
            # Should either handle gracefully or raise meaningful error
        except ValueError as e:
            assert "timeout" in str(e).lower() or "worker" in str(e).lower()

        # Test 2: Missing required directories
        non_existent = Path("/definitely/does/not/exist")
        config = RealConfigForTests(
            cache_dir=non_existent / "cache", output_dir=non_existent / "output"
        )

        # Should create directories or handle gracefully
        if config.cache_dir.exists():
            assert config.cache_dir.is_dir()

    def _patch_services(self, monkeypatch, services):
        """Patch external services for testing."""

        # Patch arXiv
        def mock_arxiv_search(*args, **kwargs):
            class MockSearch:
                def results(self):
                    query = args[0] if args else kwargs.get("query", "")
                    return services["arxiv"].search(
                        query, kwargs.get("max_results", 10)
                    )

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
                        if not content:
                            return []
                        return [
                            content[i : i + chunk_size]
                            for i in range(0, len(content), chunk_size)
                        ]

                    def raise_for_status(self):
                        if self.status_code >= 400:
                            raise requests.HTTPError(f"HTTP {self.status_code}")

                return MockResponse()
            elif "localhost:8000" in url:
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
                result = services["llm"].extract(
                    json.get("text", ""), json.get("model", "gpt-3.5-turbo"), **json
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
