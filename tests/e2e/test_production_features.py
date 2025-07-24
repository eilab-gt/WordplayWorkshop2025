"""Tests for production-specific features and integration."""

import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.lit_review.harvesters.production_harvester import ProductionHarvester
from src.lit_review.harvesters.query_optimizer import QueryOptimizer
from src.lit_review.processing.batch_processor import BatchProcessor
from src.lit_review.utils.content_cache import ContentCache
from tests.test_data_generators import RealisticTestDataGenerator
from tests.test_doubles import (
    FakeArxivAPI,
    FakeLLMService,
    FakePDFServer,
    RealConfigForTests,
)


@pytest.mark.e2e
@pytest.mark.production
class TestProductionFeatures:
    """Test production-specific features and scale."""

    @pytest.fixture
    def production_config(self, tmp_path):
        """Create production-ready configuration."""
        config_dict = {
            "paths": {
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "output"),
                "data_dir": str(tmp_path / "data"),
                "log_dir": str(tmp_path / "logs"),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            },
            "production": {
                "max_papers_per_session": 10000,
                "checkpoint_interval": 100,
                "batch_size": 50,
                "max_workers": 8,
                "memory_limit_gb": 4,
                "enable_monitoring": True,
                "enable_telemetry": False,
            },
            "search": {
                "wargame_terms": ["wargame", "wargaming", "strategic simulation"],
                "llm_terms": ["LLM", "large language model", "GPT", "Claude"],
                "years": [2020, 2024],
                "sample_size": 100,
            },
            "processing": {
                "pdf_timeout_seconds": 30,
                "pdf_max_size_mb": 100,
                "llm_timeout_seconds": 60,
                "retry_attempts": 3,
                "cache_max_age_days": 180,
            },
            "sources": {
                "arxiv": {"enabled": True, "rate_limit": {"delay_milliseconds": 3000}},
                "semantic_scholar": {
                    "enabled": True,
                    "rate_limit": {"requests_per_second": 10},
                },
                "crossref": {
                    "enabled": True,
                    "rate_limit": {"requests_per_second": 50},
                },
            },
        }

        # Write config file
        config_path = tmp_path / "production_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Also create a standard config object
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            checkpoint_dir=tmp_path / "checkpoints",
            max_papers_per_session=10000,
            checkpoint_interval=100,
            batch_size=50,
            parallel_workers=8,
        )

        return config, config_path

    @pytest.fixture
    def production_services(self):
        """Create services for production testing."""
        generator = RealisticTestDataGenerator(seed=54321)
        arxiv = FakeArxivAPI(seed=54321)

        # Generate a large number of realistic papers
        for i in range(500):
            paper = generator.generate_paper()
            paper["arxiv_id"] = f"prod.{i:05d}"
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

    def test_production_harvester_checkpointing(
        self, production_config, production_services, monkeypatch
    ):
        """Test production harvester checkpoint and resume functionality."""
        config, config_path = production_config
        self._patch_services(monkeypatch, production_services)

        # Initialize harvester
        harvester = ProductionHarvester(str(config_path))

        # Start a harvest session
        session_id = harvester.start_session(sources=["arxiv"], max_papers=50)

        assert session_id is not None
        assert harvester.current_session is not None

        # Harvest some papers
        results = []
        for _ in range(3):  # 3 batches
            batch = harvester.harvest_next_batch()
            if batch is not None and len(batch) > 0:
                results.append(batch)

        # Should have created checkpoints
        checkpoint_dir = Path(config.checkpoint_dir) / session_id
        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
        assert len(checkpoints) > 0

        # Simulate interruption
        papers_before = harvester.get_session_stats()["papers_harvested"]
        harvester.save_checkpoint()

        # Create new harvester and resume
        harvester2 = ProductionHarvester(str(config_path))
        resumed = harvester2.resume_session(session_id)

        assert resumed is True
        assert harvester2.current_session is not None

        # Should continue from where it left off
        stats = harvester2.get_session_stats()
        assert stats["papers_harvested"] == papers_before

        # Can continue harvesting
        batch = harvester2.harvest_next_batch()
        assert batch is not None

    def test_query_optimizer(self, production_config):
        """Test query optimization for better search results."""
        config, _ = production_config
        optimizer = QueryOptimizer(config)

        # Test basic optimization
        basic_query = "LLM wargaming"
        optimized = optimizer.optimize_query(basic_query, source="arxiv")

        # Should expand query
        assert len(optimized) > len(basic_query)
        assert "OR" in optimized or "AND" in optimized

        # Test with feedback
        optimizer.add_feedback(basic_query, relevance_score=0.3)  # Poor results

        # Should adjust strategy
        optimized2 = optimizer.optimize_query(basic_query, source="arxiv")
        assert optimized2 != optimized  # Should be different

        # Test source-specific optimization
        arxiv_query = optimizer.optimize_query("LLM agents", source="arxiv")
        scholar_query = optimizer.optimize_query(
            "LLM agents", source="semantic_scholar"
        )

        # May have different formats
        assert isinstance(arxiv_query, str)
        assert isinstance(scholar_query, str)

    def test_batch_processor_memory_management(
        self, production_config, production_services
    ):
        """Test batch processor memory management and efficiency."""
        config, _ = production_config
        processor = BatchProcessor(config)

        # Create large dataset
        large_dataset = []
        for i in range(1000):
            paper = production_services["generator"].generate_paper()
            paper["long_text"] = "X" * 10000  # 10KB per paper
            large_dataset.append(paper)

        papers_df = pd.DataFrame(large_dataset)

        # Process in batches
        processed_batches = []

        def process_func(batch):
            # Simulate processing
            batch["processed"] = True
            batch["processing_time"] = 0.01
            return batch

        # Should handle memory efficiently
        for batch in processor.process_dataframe(
            papers_df, process_func, batch_size=100
        ):
            processed_batches.append(batch)

            # Check memory isn't growing excessively
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Should stay under reasonable limit
            assert memory_mb < 2000, f"Memory usage too high: {memory_mb}MB"

        # Should process all papers
        total_processed = sum(len(batch) for batch in processed_batches)
        assert total_processed == len(papers_df)

    def test_production_monitoring_and_telemetry(
        self, production_config, production_services, monkeypatch
    ):
        """Test production monitoring and telemetry features."""
        config, config_path = production_config
        self._patch_services(monkeypatch, production_services)

        # Enable monitoring
        harvester = ProductionHarvester(str(config_path))

        # Mock telemetry endpoint
        telemetry_data = []

        def mock_send_telemetry(data):
            telemetry_data.append(data)

        if hasattr(harvester, "_send_telemetry"):
            harvester._send_telemetry = mock_send_telemetry

        # Start monitored session
        session_id = harvester.start_session(sources=["arxiv"], max_papers=20)

        # Harvest with monitoring
        start_time = time.time()
        total_papers = 0

        while total_papers < 20:
            batch = harvester.harvest_next_batch()
            if batch is None or len(batch) == 0:
                break
            total_papers += len(batch)

            # Get current stats
            stats = harvester.get_session_stats()

            # Should track metrics
            assert stats["papers_harvested"] == total_papers
            assert stats["start_time"] is not None
            assert "success_rate" in stats

        duration = time.time() - start_time

        # Final stats should be comprehensive
        final_stats = harvester.get_session_stats()
        assert final_stats["papers_harvested"] > 0
        assert final_stats["duration_seconds"] > 0
        assert final_stats["papers_per_second"] > 0

    def test_distributed_deduplication(self, production_config, production_services):
        """Test deduplication across large distributed datasets."""
        config, _ = production_config

        # Create papers with intentional duplicates
        papers = []

        # Original papers
        for i in range(100):
            paper = production_services["generator"].generate_paper()
            paper["source_batch"] = "batch1"
            papers.append(paper)

        # Add duplicates with variations
        for i in range(0, 50, 5):  # Every 5th paper
            dup = papers[i].copy()
            dup["source_batch"] = "batch2"
            # Slight variations
            dup["title"] = dup["title"].upper()  # Case difference
            dup["abstract"] = dup["abstract"] + " (Updated)"  # Minor addition
            papers.append(dup)

        # Shuffle to simulate distributed collection
        import random

        random.shuffle(papers)

        # Process through batch processor with deduplication
        from src.lit_review.processing import Normalizer

        normalizer = Normalizer(config)

        # Deduplicate in batches (simulating distributed processing)
        all_deduped = []
        batch_size = 30

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            batch_df = pd.DataFrame(batch)

            # Normalize and deduplicate within batch
            normalized = normalizer.normalize(batch_df)
            deduped = normalizer.deduplicate(normalized)
            all_deduped.append(deduped)

        # Global deduplication
        combined = pd.concat(all_deduped, ignore_index=True)
        final_deduped = normalizer.deduplicate(combined)

        # Should have removed duplicates
        assert len(final_deduped) < len(papers)
        assert len(final_deduped) <= 100  # Should be close to original count

        # Check no duplicate DOIs or arXiv IDs
        if "doi" in final_deduped.columns:
            doi_counts = final_deduped["doi"].dropna().value_counts()
            assert all(doi_counts == 1)

        if "arxiv_id" in final_deduped.columns:
            arxiv_counts = final_deduped["arxiv_id"].dropna().value_counts()
            assert all(arxiv_counts == 1)

    def test_cache_performance_at_scale(self, production_config, production_services):
        """Test cache performance with production-scale data."""
        config, _ = production_config
        cache = ContentCache(config)

        # Simulate production cache usage
        cache_times = []
        fetch_times = []

        # First pass - populate cache
        for i in range(100):
            paper_id = f"scale_test_{i}"

            start = time.time()
            path, was_cached = cache.get_or_fetch(
                paper_id,
                "pdf",
                lambda: f"Content for paper {i}".encode() * 1000,  # ~10KB each
            )
            duration = time.time() - start

            if was_cached:
                cache_times.append(duration)
            else:
                fetch_times.append(duration)

        # Second pass - all from cache
        for i in range(100):
            paper_id = f"scale_test_{i}"

            start = time.time()
            path, was_cached = cache.get_or_fetch(
                paper_id, "pdf", lambda: b"Should not be called"
            )
            duration = time.time() - start

            assert was_cached, "Should be cached on second pass"
            cache_times.append(duration)

        # Cache should be much faster
        if cache_times and fetch_times:
            avg_cache = sum(cache_times) / len(cache_times)
            avg_fetch = sum(fetch_times) / len(fetch_times)

            # Cache should be at least 10x faster
            assert (
                avg_cache < avg_fetch / 10
            ), f"Cache not fast enough: {avg_cache:.3f}s vs {avg_fetch:.3f}s"

        # Test cache size management
        stats = cache.get_stats()
        assert stats["total_items"] >= 100
        assert stats["cache_size_mb"] > 0

    def test_error_recovery_in_production(
        self, production_config, production_services, monkeypatch
    ):
        """Test error recovery mechanisms in production scenarios."""
        config, config_path = production_config
        self._patch_services(monkeypatch, production_services)

        # Configure services to fail intermittently
        production_services["pdf"].failure_rate = 0.2  # 20% failure
        production_services["llm"].available_models["gemini/gemini-pro"][
            "available"
        ] = False

        # Run production harvest
        harvester = ProductionHarvester(str(config_path))
        session_id = harvester.start_session(sources=["arxiv"], max_papers=50)

        errors = []
        successes = 0

        # Harvest with error tracking
        while harvester.get_session_stats()["papers_harvested"] < 50:
            try:
                batch = harvester.harvest_next_batch()
                if batch is None:
                    break

                # Process batch (this would normally include PDF fetch and extraction)
                for _, paper in batch.iterrows():
                    try:
                        # Simulate processing
                        if random.random() < 0.8:  # 80% success
                            successes += 1
                        else:
                            raise Exception("Simulated processing error")
                    except Exception as e:
                        errors.append(str(e))

            except Exception as e:
                errors.append(str(e))
                # Should recover and continue

        # Should complete despite errors
        stats = harvester.get_session_stats()
        assert stats["papers_harvested"] > 0
        assert successes > 0

        # Error rate should be reasonable
        total_attempts = successes + len(errors)
        error_rate = len(errors) / total_attempts if total_attempts > 0 else 0
        assert error_rate < 0.5, "Error rate too high"

    def test_production_export_formats(self, production_config, production_services):
        """Test production-ready export formats."""
        config, _ = production_config

        # Create realistic dataset
        papers = []
        for i in range(50):
            paper = production_services["generator"].generate_paper()
            extraction = production_services["generator"].generate_extraction_results(
                paper
            )
            papers.append({**paper, **extraction})

        papers_df = pd.DataFrame(papers)

        # Test various export formats
        from src.lit_review.utils import Exporter

        exporter = Exporter(config)

        # 1. Research-ready CSV
        csv_path = config.output_dir / "research_export.csv"
        papers_df.to_csv(csv_path, index=False)
        assert csv_path.exists()

        # 2. Machine-readable JSON
        json_path = config.output_dir / "research_export.json"
        papers_df.to_json(json_path, orient="records", indent=2)
        assert json_path.exists()

        # 3. BibTeX for citations
        bibtex_path = exporter.export_bibtex(papers_df)
        assert bibtex_path.exists()

        # Verify BibTeX format
        bibtex_content = bibtex_path.read_text()
        assert (
            bibtex_content.count("@article") >= len(papers_df) * 0.8
        )  # Most should export

        # 4. Analysis-ready Parquet
        parquet_path = config.output_dir / "research_export.parquet"
        papers_df.to_parquet(parquet_path)
        assert parquet_path.exists()

        # Parquet should be smaller than CSV
        assert parquet_path.stat().st_size < csv_path.stat().st_size

    def test_production_pipeline_end_to_end(
        self, production_config, production_services, monkeypatch
    ):
        """Test complete production pipeline from search to export."""
        config, config_path = production_config
        self._patch_services(monkeypatch, production_services)

        # Initialize all production components
        harvester = ProductionHarvester(str(config_path))
        batch_processor = BatchProcessor(config)

        # Start production run
        session_id = harvester.start_session(
            sources=["arxiv"], max_papers=100, checkpoint_interval=25
        )

        # Phase 1: Harvest papers
        all_papers = []
        while len(all_papers) < 100:
            batch = harvester.harvest_next_batch()
            if batch is None or len(batch) == 0:
                break
            all_papers.append(batch)

            # Should checkpoint periodically
            if len(all_papers) % 25 == 0:
                checkpoint_files = list(
                    (Path(config.checkpoint_dir) / session_id).glob("*.json")
                )
                assert len(checkpoint_files) > 0

        # Combine results
        if all_papers:
            combined_df = pd.concat(all_papers, ignore_index=True)

            # Phase 2: Process in batches
            from src.lit_review.processing import Normalizer

            normalizer = Normalizer(config)

            processed_batches = []
            for batch in batch_processor.process_dataframe(
                combined_df, lambda b: normalizer.normalize(b), batch_size=25
            ):
                processed_batches.append(batch)

            final_df = pd.concat(processed_batches, ignore_index=True)

            # Phase 3: Generate production report
            report = {
                "session_id": session_id,
                "total_papers": len(final_df),
                "sources": (
                    final_df["source_db"].value_counts().to_dict()
                    if "source_db" in final_df.columns
                    else {}
                ),
                "year_distribution": (
                    final_df["year"].value_counts().to_dict()
                    if "year" in final_df.columns
                    else {}
                ),
                "harvest_stats": harvester.get_session_stats(),
            }

            # Save report
            report_path = config.output_dir / f"production_report_{session_id}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            assert report_path.exists()
            assert report["total_papers"] > 0

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
                            raise Exception(f"HTTP {self.status_code}")

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

        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.post", mock_post)
        monkeypatch.setattr("requests.Session.get", mock_get)
        monkeypatch.setattr("requests.Session.post", mock_post)
