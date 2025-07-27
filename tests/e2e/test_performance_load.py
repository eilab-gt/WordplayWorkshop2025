"""Performance and load testing for production-scale operations."""

import gc
import json
import math
import os

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import pytest

from src.lit_review.extraction import EnhancedLLMExtractor
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher
from src.lit_review.processing.batch_processor import BatchProcessor
from src.lit_review.utils.content_cache import ContentCache
from src.lit_review.visualization import Visualizer
from tests.test_data_generators import RealisticTestDataGenerator
from tests.test_doubles import (
    FakeArxivAPI,
    FakeLLMService,
    FakePDFServer,
    RealConfigForTests,
)


@pytest.mark.e2e
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.skip(reason="Performance benchmarks should run in nightly suite, not CI")
class TestPerformanceAndLoad:
    """Test system performance under various load conditions."""

    @pytest.fixture
    def perf_config(self, tmp_path):
        """Configuration optimized for performance testing."""
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            parallel_workers=os.cpu_count() or 4,
            batch_size_pdf=100,
            pdf_timeout_seconds=5,
            llm_timeout_seconds=10,
            cache_max_age_days=30,
            memory_limit_gb=4,
        )
        yield config
        config.cleanup()

    @pytest.fixture
    def load_services(self):
        """Services configured for load testing."""
        generator = RealisticTestDataGenerator(seed=99999)
        arxiv = FakeArxivAPI(seed=99999)

        # Pre-generate many papers for load testing
        print("Generating test data...")
        for i in range(2000):
            paper = generator.generate_paper()
            paper["arxiv_id"] = f"load.{i:05d}"
            arxiv.add_paper(
                {
                    "id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "abstract": paper["abstract"],
                    "pdf_url": f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf",
                    "published": paper.get("published", "2024-01-01"),
                    "categories": paper.get("categories", ["cs.AI"]),
                }
            )

        return {
            "arxiv": arxiv,
            "llm": FakeLLMService(healthy=True),
            "pdf": FakePDFServer(),
            "generator": generator,
        }

    def test_search_performance_baseline(self, perf_config, load_services, monkeypatch):
        """Establish baseline search performance metrics."""
        self._patch_services(monkeypatch, load_services)

        harvester = SearchHarvester(perf_config)

        # Test various query complexities
        queries = [
            "LLM",  # Simple
            "LLM wargaming",  # Medium
            "large language model strategic simulation multi-agent",  # Complex
            "LLM OR GPT OR Claude AND wargaming OR simulation",  # Boolean
        ]

        results = {}

        for query in queries:
            # Warm up
            harvester.search_all(query, max_results=1)

            # Measure
            times = []
            for _ in range(5):  # 5 runs each
                start = time.time()
                results_df = harvester.search_all(query, max_results=100)
                duration = time.time() - start
                times.append(duration)

            avg_time = sum(times) / len(times)
            papers_per_second = len(results_df) / avg_time if avg_time > 0 else 0

            results[query] = {
                "avg_time": avg_time,
                "min_time": min(times),
                "max_time": max(times),
                "papers_found": len(results_df),
                "papers_per_second": papers_per_second,
            }

        # Print results
        print("\n=== Search Performance Baseline ===")
        for query, metrics in results.items():
            print(f"\nQuery: '{query}'")
            print(f"  Papers found: {metrics['papers_found']}")
            print(f"  Avg time: {metrics['avg_time']:.3f}s")
            print(f"  Throughput: {metrics['papers_per_second']:.1f} papers/s")

        # Assertions
        for metrics in results.values():
            assert metrics["avg_time"] < 2.0, "Search should complete within 2 seconds"
            assert metrics["papers_per_second"] > 50, "Should process >50 papers/second"

    def test_concurrent_search_load(self, perf_config, load_services, monkeypatch):
        """Test system under concurrent search load."""
        self._patch_services(monkeypatch, load_services)

        harvester = SearchHarvester(perf_config)

        # Concurrent search function
        def search_worker(worker_id: int, num_searches: int) -> dict[str, Any]:
            queries = [
                "LLM wargaming",
                "strategic simulation",
                "multi-agent systems",
                "GPT military",
                "Claude planning",
            ]

            start_time = time.time()
            results = []
            errors = 0

            for _i in range(num_searches):
                try:
                    query = random.choice(queries)
                    df = harvester.search_all(query, max_results=50)
                    results.append(len(df))
                except Exception as e:
                    errors += 1

            duration = time.time() - start_time

            return {
                "worker_id": worker_id,
                "searches": num_searches,
                "total_papers": sum(results),
                "errors": errors,
                "duration": duration,
                "searches_per_second": num_searches / duration if duration > 0 else 0,
            }

        # Run concurrent searches
        num_workers = 5
        searches_per_worker = 20

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(search_worker, i, searches_per_worker)
                for i in range(num_workers)
            ]

            results = [f.result() for f in futures]

        # Analyze results
        total_searches = sum(r["searches"] for r in results)
        total_papers = sum(r["total_papers"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        total_duration = max(r["duration"] for r in results)

        print("\n=== Concurrent Search Load Test ===")
        print(f"Workers: {num_workers}")
        print(f"Total searches: {total_searches}")
        print(f"Total papers: {total_papers}")
        print(f"Total errors: {total_errors}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Throughput: {total_searches/total_duration:.1f} searches/s")

        # Assertions
        assert total_errors == 0, "Should handle concurrent searches without errors"
        assert total_searches / total_duration > 5, "Should handle >5 searches/second"

    def test_processing_pipeline_throughput(
        self, perf_config, load_services, monkeypatch
    ):
        """Test full pipeline processing throughput."""
        self._patch_services(monkeypatch, load_services)

        # Generate test dataset
        papers = []
        for i in range(500):
            paper = load_services["generator"].generate_paper()
            paper["pdf_path"] = f"/fake/{i}.pdf"
            paper["pdf_status"] = "downloaded" if random.random() > 0.1 else "not_found"
            papers.append(paper)

        papers_df = pd.DataFrame(papers)

        # Measure each pipeline stage
        stage_metrics = {}

        # Stage 1: Normalization
        normalizer = Normalizer(perf_config)
        start = time.time()
        normalized_df = normalizer.normalize_dataframe(papers_df)
        stage_metrics["normalization"] = {
            "duration": time.time() - start,
            "papers_per_second": len(papers_df) / (time.time() - start),
        }

        # Stage 2: Deduplication
        start = time.time()
        deduped_df = normalizer.deduplicate(normalized_df)
        stage_metrics["deduplication"] = {
            "duration": time.time() - start,
            "papers_per_second": len(normalized_df) / (time.time() - start),
        }

        # Stage 3: PDF Processing (simulated)
        pdf_fetcher = PDFFetcher(perf_config)
        start = time.time()
        # Simulate PDF processing without actual downloads
        pdf_df = deduped_df.copy()
        pdf_df["pdf_processed"] = True
        stage_metrics["pdf_processing"] = {
            "duration": time.time() - start,
            "papers_per_second": len(deduped_df) / (time.time() - start),
        }

        # Stage 4: Extraction (simulated)
        extractor = EnhancedLLMExtractor(perf_config)
        successful_pdfs = pdf_df[pdf_df["pdf_status"] == "downloaded"]

        start = time.time()
        # Simulate extraction
        extracted_df = successful_pdfs.copy()
        extracted_df["extraction_status"] = "success"
        extracted_df["extraction_time"] = 0.1  # Simulated
        stage_metrics["extraction"] = {
            "duration": time.time() - start,
            "papers_per_second": (
                len(successful_pdfs) / (time.time() - start)
                if len(successful_pdfs) > 0
                else 0
            ),
        }

        # Print metrics
        print("\n=== Pipeline Throughput Metrics ===")
        total_duration = sum(m["duration"] for m in stage_metrics.values())
        print(f"Total papers: {len(papers_df)}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Overall throughput: {len(papers_df)/total_duration:.1f} papers/s")

        for stage, metrics in stage_metrics.items():
            print(f"\n{stage.capitalize()}:")
            print(f"  Duration: {metrics['duration']:.3f}s")
            print(f"  Throughput: {metrics['papers_per_second']:.1f} papers/s")

        # Assertions
        assert all(
            m["papers_per_second"] > 100 for m in stage_metrics.values()
        ), "Each stage should process >100 papers/second"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_memory_usage_under_load(self, perf_config, load_services):
        """Test memory usage patterns under various loads."""
        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_snapshots = []

        # Test 1: Large dataframe operations
        large_df = pd.DataFrame(
            [load_services["generator"].generate_paper() for _ in range(10000)]
        )

        memory_snapshots.append(
            {
                "operation": "10K papers loaded",
                "memory_mb": process.memory_info().rss / 1024 / 1024 - baseline_memory,
            }
        )

        # Test 2: Batch processing
        batch_processor = BatchProcessor(perf_config)
        processed_count = 0

        for batch in batch_processor.process_dataframe(
            large_df, lambda b: b.assign(processed=True), batch_size=1000
        ):
            processed_count += len(batch)

        memory_snapshots.append(
            {
                "operation": "Batch processing complete",
                "memory_mb": process.memory_info().rss / 1024 / 1024 - baseline_memory,
            }
        )

        # Test 3: Cleanup
        del large_df
        gc.collect()

        memory_snapshots.append(
            {
                "operation": "After cleanup",
                "memory_mb": process.memory_info().rss / 1024 / 1024 - baseline_memory,
            }
        )

        # Print memory profile
        print("\n=== Memory Usage Profile ===")
        print(f"Baseline: {baseline_memory:.1f} MB")
        for snapshot in memory_snapshots:
            print(f"{snapshot['operation']}: +{snapshot['memory_mb']:.1f} MB")

        # Assertions
        max_memory_increase = max(s["memory_mb"] for s in memory_snapshots)
        assert (
            max_memory_increase < 500
        ), f"Memory increase too high: {max_memory_increase:.1f} MB"

        # Memory should be mostly released after cleanup
        final_memory = memory_snapshots[-1]["memory_mb"]
        assert (
            final_memory < 100
        ), f"Memory not released properly: {final_memory:.1f} MB still used"

    def test_cache_performance_stress_test(self, perf_config, load_services):
        """Stress test cache with high concurrency and large data."""
        cache = ContentCache(perf_config)

        # Test parameters
        num_threads = 10
        items_per_thread = 100
        content_size = 10000  # 10KB per item

        results = {"cache_hits": 0, "cache_misses": 0, "errors": 0, "timings": []}
        results_lock = threading.Lock()

        def cache_worker(worker_id: int):
            """Worker that performs cache operations."""
            local_timings = []

            for i in range(items_per_thread):
                paper_id = f"stress_test_{i % 50}"  # Reuse some IDs

                try:
                    start = time.time()
                    path, was_cached = cache.get_or_fetch(
                        paper_id,
                        "pdf",
                        lambda i=i: f"Content {worker_id}-{i}".encode()
                        * (content_size // 20),
                    )
                    duration = time.time() - start
                    local_timings.append(duration)

                    with results_lock:
                        if was_cached:
                            results["cache_hits"] += 1
                        else:
                            results["cache_misses"] += 1

                except Exception as e:
                    with results_lock:
                        results["errors"] += 1

                # Small delay to simulate real usage
                time.sleep(0.001)

            with results_lock:
                results["timings"].extend(local_timings)

        # Run stress test
        start_time = time.time()
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=cache_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total_duration = time.time() - start_time

        # Calculate metrics
        total_operations = results["cache_hits"] + results["cache_misses"]
        hit_rate = (
            results["cache_hits"] / total_operations if total_operations > 0 else 0
        )
        avg_time = (
            sum(results["timings"]) / len(results["timings"])
            if results["timings"]
            else 0
        )

        print("\n=== Cache Stress Test Results ===")
        print(f"Threads: {num_threads}")
        print(f"Total operations: {total_operations}")
        print(f"Cache hits: {results['cache_hits']} ({hit_rate:.1%})")
        print(f"Cache misses: {results['cache_misses']}")
        print(f"Errors: {results['errors']}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Operations/second: {total_operations/total_duration:.1f}")
        print(f"Avg operation time: {avg_time*1000:.1f}ms")

        # Assertions
        assert (
            results["errors"] == 0
        ), "Cache should handle concurrent access without errors"
        assert (
            total_operations / total_duration > 100
        ), "Should handle >100 operations/second"
        assert avg_time < 0.1, "Average operation should be <100ms"

    def test_visualization_performance(self, perf_config, load_services):
        """Test visualization performance with large datasets."""
        visualizer = Visualizer(perf_config)

        # Generate large dataset with all required fields
        papers = []
        for _i in range(1000):
            paper = load_services["generator"].generate_paper()
            papers.append(paper)

        large_df = pd.DataFrame(papers)

        # Measure visualization generation time
        viz_metrics = {}

        # Test each visualization method
        viz_methods = [
            ("time_series", visualizer.plot_time_series),
            ("awscale", visualizer.plot_awscale_distribution),
            ("game_types", visualizer.plot_game_types),
            ("failure_modes", visualizer.plot_failure_modes),
            ("llm_families", visualizer.plot_llm_families),
            ("source_dist", visualizer.plot_source_distribution),
        ]

        for name, method in viz_methods:
            start = time.time()
            try:
                fig_path = method(large_df, save=True)
                duration = time.time() - start
                success = fig_path is not None
            except Exception as e:
                duration = time.time() - start
                success = False

            viz_metrics[name] = {"duration": duration, "success": success}

        # Test all visualizations
        start = time.time()
        all_figures = visualizer.create_all_visualizations(large_df, save=True)
        total_duration = time.time() - start

        # Print metrics
        print("\n=== Visualization Performance ===")
        print(f"Dataset size: {len(large_df)} papers")
        print(f"Total visualization time: {total_duration:.2f}s")

        for name, metrics in viz_metrics.items():
            status = "✓" if metrics["success"] else "✗"
            print(f"{name}: {metrics['duration']:.3f}s {status}")

        # Assertions
        assert (
            total_duration < 10
        ), "All visualizations should complete within 10 seconds"
        successful = sum(1 for m in viz_metrics.values() if m["success"])
        assert successful >= 4, "Most visualizations should succeed"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_scalability_limits(self, perf_config, load_services, monkeypatch):
        """Test system scalability limits and degradation patterns."""
        self._patch_services(monkeypatch, load_services)

        # Test with increasing dataset sizes
        sizes = [100, 500, 1000, 2000, 5000]
        metrics = []

        for size in sizes:
            # Generate dataset
            papers = []
            for _i in range(size):
                paper = load_services["generator"].generate_paper()
                papers.append(paper)

            papers_df = pd.DataFrame(papers)

            # Measure processing time
            start = time.time()

            # Full pipeline simulation
            normalizer = Normalizer(perf_config)
            normalized = normalizer.normalize_dataframe(papers_df)
            deduped = normalizer.deduplicate(normalized)

            duration = time.time() - start

            # Memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            metrics.append(
                {
                    "size": size,
                    "duration": duration,
                    "memory_mb": memory_mb,
                    "papers_per_second": size / duration,
                }
            )

            # Cleanup
            del papers_df, normalized, deduped
            gc.collect()

        # Analyze scalability
        print("\n=== Scalability Analysis ===")
        print("Size | Duration | Memory | Throughput")
        print("-" * 45)

        for m in metrics:
            print(
                f"{m['size']:5d} | {m['duration']:7.2f}s | {m['memory_mb']:6.0f}MB | {m['papers_per_second']:6.1f} p/s"
            )

        # Check for linear or better scalability
        if len(metrics) > 1:
            # Calculate scalability factor
            first = metrics[0]
            last = metrics[-1]

            size_increase = last["size"] / first["size"]
            time_increase = last["duration"] / first["duration"]

            scalability_factor = time_increase / size_increase

            print(f"\nScalability factor: {scalability_factor:.2f}")
            print("(1.0 = perfect linear, <1.0 = super-linear, >1.0 = sub-linear)")

            # Should maintain reasonable scalability
            assert scalability_factor < 1.5, "Scalability degradation too severe"

    def test_production_load_simulation(self, perf_config, load_services, monkeypatch):
        """Simulate realistic production load patterns."""
        self._patch_services(monkeypatch, load_services)

        # Simulate a day's worth of operations in compressed time
        simulation_duration = 30  # seconds
        start_time = time.time()

        # Metrics collection
        operations = {
            "searches": 0,
            "papers_processed": 0,
            "cache_operations": 0,
            "errors": 0,
        }

        # Components
        harvester = SearchHarvester(perf_config)
        normalizer = Normalizer(perf_config)
        cache = ContentCache(perf_config)

        # Simulate varying load patterns
        while time.time() - start_time < simulation_duration:
            # Vary load based on "time of day"
            elapsed = time.time() - start_time
            load_factor = 0.5 + 0.5 * abs(math.sin(elapsed / 10))  # Sine wave pattern

            # Search operations
            if random.random() < load_factor:
                try:
                    query = random.choice(
                        [
                            "LLM wargaming",
                            "strategic AI",
                            "multi-agent simulation",
                            "GPT military",
                        ]
                    )
                    results = harvester.search_all(
                        query, max_results=int(50 * load_factor)
                    )
                    operations["searches"] += 1
                    operations["papers_processed"] += len(results)

                    # Process some results
                    if len(results) > 0:
                        normalized = normalizer.normalize_dataframe(results[:10])

                        # Cache operations
                        for _, paper in normalized.iterrows():
                            if "paper_id" in paper:
                                cache.get_or_fetch(
                                    paper["paper_id"],
                                    "metadata",
                                    lambda: json.dumps(paper.to_dict()).encode(),
                                )
                                operations["cache_operations"] += 1

                except Exception as e:
                    operations["errors"] += 1

            # Small delay
            time.sleep(0.1)

        # Calculate metrics
        total_duration = time.time() - start_time

        print("\n=== Production Load Simulation Results ===")
        print(f"Simulation duration: {total_duration:.1f}s")
        print(f"Total searches: {operations['searches']}")
        print(f"Papers processed: {operations['papers_processed']}")
        print(f"Cache operations: {operations['cache_operations']}")
        print(f"Errors: {operations['errors']}")
        print(f"Searches/minute: {operations['searches'] * 60 / total_duration:.1f}")
        print(
            f"Papers/minute: {operations['papers_processed'] * 60 / total_duration:.1f}"
        )

        # Assertions
        assert (
            operations["errors"] < operations["searches"] * 0.01
        ), "Error rate too high"
        assert (
            operations["searches"] > 50
        ), "Should complete reasonable number of searches"
        assert operations["papers_processed"] > 500, "Should process many papers"

    def _patch_services(self, monkeypatch, services):
        """Patch external services for testing."""

        def mock_arxiv_search(*args, **kwargs):
            class MockSearch:
                def results(self):
                    query = args[0] if args else kwargs.get("query", "")
                    return services["arxiv"].search(
                        query, kwargs.get("max_results", 10)
                    )

            return MockSearch()

        monkeypatch.setattr("arxiv.Search", mock_arxiv_search)

        def mock_get(url, *args, **kwargs):
            if "arxiv.org/pdf" in url:
                arxiv_id = url.split("/")[-1].replace(".pdf", "")
                content, status = services["pdf"].serve_pdf(arxiv_id)

                class MockResponse:
                    status_code = status
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
                            raise Exception(f"HTTP {self.status_code}")

                return MockResponse()

        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.Session.get", mock_get)


# Math import is now at the top of the file
