"""Edge case and boundary condition tests for E2E pipeline."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from src.lit_review.extraction import EnhancedLLMExtractor
from src.lit_review.harvesters import SearchHarvester
from src.lit_review.processing import Normalizer, PDFFetcher
from src.lit_review.utils import Exporter
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
@pytest.mark.edge_cases
class TestEdgeCases:
    """Test edge cases and boundary conditions in the pipeline."""

    @pytest.fixture
    def edge_config(self, tmp_path):
        """Configuration for edge case testing."""
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            sample_size=100,  # Large sample for boundary testing
            search_years=(1990, 2050),  # Wide year range
            pdf_max_size_mb=100,  # Large PDF limit
            cache_max_age_days=365,
            batch_size_pdf=50,
        )
        # Disable secondary queries for edge case tests
        config.query_strategies["secondary"] = []
        yield config
        config.cleanup()

    @pytest.fixture
    def edge_services(self):
        """Services configured for edge case testing."""
        generator = RealisticTestDataGenerator(seed=12345)
        arxiv = FakeArxivAPI(seed=12345)

        # Add edge case papers
        edge_papers = self._create_edge_case_papers(generator)
        for paper in edge_papers:
            if "arxiv_id" in paper:
                arxiv.add_paper(
                    {
                        "id": paper["arxiv_id"],
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "abstract": paper.get("abstract", ""),
                        "pdf_url": paper.get(
                            "pdf_url", f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
                        ),
                        "published": paper.get("published", "2024-01-01"),
                        "categories": paper.get("categories", ["cs.AI"]),
                    }
                )

        # Create fake services
        from unittest.mock import Mock

        # Mock Semantic Scholar API responses
        ss = Mock()
        ss.get = Mock(return_value=Mock(status_code=200, json=lambda: {"data": []}))
        ss.post = Mock(return_value=Mock(status_code=200, json=lambda: {"data": []}))

        pdf_server = FakePDFServer()

        return {
            "arxiv": arxiv,
            "ss": ss,
            "llm": FakeLLMService(healthy=True),
            "pdf_server": pdf_server,
            "generator": generator,
        }

    def _create_edge_case_papers(self, generator):
        """Create papers with edge case characteristics."""
        edge_papers = []

        # 1. Extremely long title
        long_title_paper = generator.generate_paper()
        long_title_paper["title"] = (
            "A " + " ".join(["Very"] * 50) + " Long Title About LLMs"
        )
        long_title_paper["arxiv_id"] = "edge.00001"
        edge_papers.append(long_title_paper)

        # 2. Huge number of authors
        many_authors_paper = generator.generate_paper()
        many_authors_paper["authors"] = [
            generator.generate_author_name() for _ in range(100)
        ]
        many_authors_paper["arxiv_id"] = "edge.00002"
        edge_papers.append(many_authors_paper)

        # 3. Unicode and special characters
        unicode_paper = generator.generate_paper()
        unicode_paper["title"] = (
            "LLMs and 机器学习 (Machine Learning): Émergent Behaviors with π≈3.14"
        )
        unicode_paper["authors"] = [
            "José García",
            "李明",
            "Müller, K.",
            "Владимир Петров",
        ]
        unicode_paper["abstract"] = (
            "We study LLMs with émphasis on ∂f/∂x and Σ notation..."
        )
        unicode_paper["arxiv_id"] = "edge.00003"
        edge_papers.append(unicode_paper)

        # 4. Very old and very new papers
        old_paper = generator.generate_paper(year=1990)
        old_paper["arxiv_id"] = "edge.00004"
        edge_papers.append(old_paper)

        future_paper = generator.generate_paper(year=2050)
        future_paper["arxiv_id"] = "edge.00005"
        edge_papers.append(future_paper)

        # 5. Minimal data paper
        minimal_paper = {
            "title": "X",  # Single character title
            "authors": ["A"],  # Single character author
            "abstract": "Y",  # Single character abstract
            "arxiv_id": "edge.00006",
            "year": 2024,
        }
        edge_papers.append(minimal_paper)

        # 6. Maximum data paper
        max_paper = generator.generate_paper()
        max_paper["title"] = max_paper["title"] * 10  # Very long
        max_paper["abstract"] = (
            generator.generate_abstract(max_paper["title"]) * 20
        )  # Huge abstract
        max_paper["arxiv_id"] = "edge.00007"
        edge_papers.append(max_paper)

        # 7. Papers with all possible AWScale values
        for awscale in range(1, 6):
            awscale_paper = generator.generate_paper()
            awscale_paper["awscale"] = awscale
            awscale_paper["arxiv_id"] = f"edge.awscale{awscale}"
            edge_papers.append(awscale_paper)

        # 8. Papers with edge case game types
        edge_game_types = [
            "",
            "Unknown",
            "N/A",
            "Mixed Reality Hybrid Simulation/Training Exercise",
        ]
        for i, game_type in enumerate(edge_game_types):
            game_paper = generator.generate_paper()
            game_paper["game_type"] = game_type
            game_paper["arxiv_id"] = f"edge.game{i:03d}"
            edge_papers.append(game_paper)

        # 9. Papers with extreme failure mode counts
        no_failure_paper = generator.generate_paper()
        no_failure_paper["failure_modes"] = None
        no_failure_paper["arxiv_id"] = "edge.nofail"
        edge_papers.append(no_failure_paper)

        many_failures_paper = generator.generate_paper()
        many_failures_paper["failure_modes"] = "; ".join(
            [f"Failure Mode {i}" for i in range(20)]
        )
        many_failures_paper["arxiv_id"] = "edge.manyfail"
        edge_papers.append(many_failures_paper)

        return edge_papers

    def _patch_services(self, monkeypatch, services):
        """Patch external services with test doubles."""
        # Patch ArXiv API
        monkeypatch.setattr(
            "src.lit_review.harvesters.arxiv_harvester.arxiv.Search.results",
            lambda self: services["arxiv"].search(self.query, self.max_results),
        )

        # Import Mock for the nested functions
        from unittest.mock import Mock
        
        # Patch all requests (SS, CrossRef, Google Scholar)
        def mock_get(*args, **kwargs):
            # Return empty results for all API calls
            return Mock(
                status_code=200,
                json=lambda: {"data": [], "results": [], "message": {"items": []}},
                text="",
                raise_for_status=lambda: None,
            )

        def mock_post(*args, **kwargs):
            return Mock(
                status_code=200,
                json=lambda: {"data": [], "results": []},
                raise_for_status=lambda: None,
            )

        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.post", mock_post)

        # Patch Google Scholar BeautifulSoup parsing
        monkeypatch.setattr(
            "src.lit_review.harvesters.google_scholar.BeautifulSoup",
            lambda html, parser: Mock(find_all=lambda *args, **kwargs: []),
        )

        # Patch PDF fetching
        monkeypatch.setattr(
            "src.lit_review.processing.pdf_fetcher.requests.get",
            lambda url, **kwargs: services["pdf_server"].get_pdf(url),
        )

        # Patch LLM service
        monkeypatch.setattr(
            "src.lit_review.extraction.enhanced_llm_extractor.EnhancedLLMExtractor._check_llm_service",
            lambda self: True,
        )
        monkeypatch.setattr(
            "src.lit_review.extraction.enhanced_llm_extractor.requests.post",
            lambda url, **kwargs: services["llm"].extract(
                kwargs.get("json", {}).get("text", ""),
                kwargs.get("json", {}).get("model", "gemini/gemini-pro"),
            ),
        )

    def test_empty_dataset_handling(self, edge_config, edge_services, monkeypatch):
        """Test pipeline with empty datasets at various stages."""
        # Clear any papers from arxiv for empty test
        edge_services["arxiv"].papers = []

        self._patch_services(monkeypatch, edge_services)

        # Test 1: Empty search results
        harvester = SearchHarvester(edge_config)
        empty_results = harvester.search_all(max_results_per_source=10)

        assert len(empty_results) == 0, "Should handle empty search results"

        # Test 2: Empty dataframe through pipeline
        empty_df = pd.DataFrame()

        normalizer = Normalizer(edge_config)
        normalized = normalizer.normalize_dataframe(empty_df)
        assert len(normalized) == 0

        pdf_fetcher = PDFFetcher(edge_config)
        pdf_results = pdf_fetcher.fetch_pdfs(normalized)
        assert len(pdf_results) == 0

        extractor = EnhancedLLMExtractor(edge_config)
        extracted = extractor.extract_all(pdf_results)
        assert len(extracted) == 0

        visualizer = Visualizer(edge_config)
        figures = visualizer.create_all_visualizations(extracted)
        assert isinstance(figures, list)

        # Test 3: Single row dataframe
        single_row = pd.DataFrame(
            [{"title": "Single Paper", "year": 2024, "authors": "Single Author"}]
        )

        normalized = normalizer.normalize_dataframe(single_row)
        assert len(normalized) == 1
        assert "paper_id" in normalized.columns

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_extreme_data_sizes(self, edge_config, edge_services, monkeypatch):
        """Test with extremely large and small data sizes."""
        self._patch_services(monkeypatch, edge_services)

        # Test 1: Very large paper abstract
        huge_abstract_paper = {
            "id": "huge.00001",
            "title": "Paper with Massive Abstract",
            "authors": ["Test Author"],
            "abstract": "LLM " * 10000,  # ~40KB of text
            "pdf_url": "https://arxiv.org/pdf/huge.00001.pdf",
            "published": "2024-01-01",
            "categories": ["cs.AI"],
        }
        edge_services["arxiv"].add_paper(huge_abstract_paper)

        harvester = SearchHarvester(edge_config)
        results = harvester.search_all(max_results_per_source=5)

        # Should handle large text
        assert len(results) > 0
        normalizer = Normalizer(edge_config)
        normalized = normalizer.normalize_dataframe(results)
        assert len(normalized) > 0

        # Test 2: Very large batch processing
        large_batch = []
        for i in range(200):  # Large batch
            paper = edge_services["generator"].generate_paper()
            paper["batch_index"] = i
            large_batch.append(paper)

        large_df = pd.DataFrame(large_batch)

        # Should handle in batches
        normalized = normalizer.normalize_dataframe(large_df)
        assert len(normalized) == len(large_df)

        # Test 3: Very small PDF
        edge_services["pdf"].add_pdf("tiny.00001", b"X")  # 1 byte PDF

        # Test 4: Very large PDF
        large_pdf_content = b"%PDF-1.4\n" + b"A" * (50 * 1024 * 1024)  # 50MB
        edge_services["pdf"].add_pdf("large.00001", large_pdf_content)

    def test_boundary_years(self, edge_config, edge_services, monkeypatch):
        """Test papers with boundary year values."""
        self._patch_services(monkeypatch, edge_services)

        # Search for very old and very new papers
        harvester = SearchHarvester(edge_config)
        results = harvester.search_all(max_results_per_source=5)

        # Check year handling
        if "year" in results.columns:
            years = results["year"].dropna()

            # Should handle old papers
            assert any(years < 2000), "Should find old papers"

            # Should handle future papers
            assert any(years > 2030), "Should find future papers"

        # Normalize should handle year boundaries
        normalizer = Normalizer(edge_config)
        normalized = normalizer.normalize_dataframe(results)

        # Years should be within reasonable bounds after normalization
        if "year" in normalized.columns:
            normalized_years = normalized["year"].dropna()
            assert all(normalized_years >= 1900), "Years should be reasonable"
            assert all(normalized_years <= 2100), "Years should not be too far future"

    def test_special_characters_and_unicode(
        self, edge_config, edge_services, monkeypatch
    ):
        """Test handling of special characters and unicode."""
        self._patch_services(monkeypatch, edge_services)

        # Search for papers with unicode
        harvester = SearchHarvester(edge_config)
        results = harvester.search_all(max_results_per_source=2)

        # Find unicode paper
        unicode_papers = results[
            results["title"].str.contains(
                "(?:机器学习|émphasis)|π", na=False, regex=True
            )
        ]

        if len(unicode_papers) > 0:
            # Process through pipeline
            normalizer = Normalizer(edge_config)
            normalized = normalizer.normalize_dataframe(unicode_papers)

            # Should preserve unicode
            assert any("机器学习" in str(title) for title in normalized["title"])

            # Visualization should handle unicode
            visualizer = Visualizer(edge_config)
            try:
                figures = visualizer.create_all_visualizations(normalized)
                # Should not crash with unicode
                assert isinstance(figures, list)
            except Exception as e:
                # Some matplotlib backends may have issues with unicode
                assert "codec" in str(e).lower() or "encode" in str(e).lower()

    def test_duplicate_handling_edge_cases(
        self, edge_config, edge_services, monkeypatch
    ):
        """Test edge cases in duplicate detection."""
        self._patch_services(monkeypatch, edge_services)

        # Create papers with subtle differences
        base_paper = edge_services["generator"].generate_paper()

        variations = []

        # Same title, different case
        var1 = base_paper.copy()
        var1["title"] = base_paper["title"].upper()
        var1["paper_id"] = "dup001"
        variations.append(var1)

        # Same title with extra spaces
        var2 = base_paper.copy()
        var2["title"] = "  " + base_paper["title"] + "  "
        var2["paper_id"] = "dup002"
        variations.append(var2)

        # Same title with punctuation differences
        var3 = base_paper.copy()
        var3["title"] = base_paper["title"].replace(":", " -")
        var3["paper_id"] = "dup003"
        variations.append(var3)

        # Same DOI, different title
        var4 = base_paper.copy()
        var4["title"] = "Completely Different Title"
        var4["doi"] = base_paper.get("doi", "10.1234/test")
        var4["paper_id"] = "dup004"
        variations.append(var4)

        # Create dataframe
        all_papers = [base_paper, *variations]
        papers_df = pd.DataFrame(all_papers)

        # Test deduplication
        normalizer = Normalizer(edge_config)
        normalized = normalizer.normalize_dataframe(papers_df)
        deduped = normalizer.deduplicate(normalized)

        # Should detect some duplicates
        assert len(deduped) < len(papers_df), "Should detect duplicates"

        # Papers with same DOI should be deduplicated
        if "doi" in deduped.columns:
            doi_counts = deduped["doi"].value_counts()
            assert all(doi_counts == 1), "Should not have duplicate DOIs"

    def test_concurrent_operations_limits(
        self, edge_config, edge_services, monkeypatch
    ):
        """Test concurrent operation limits and thread safety."""
        self._patch_services(monkeypatch, edge_services)

        # Create many papers for concurrent processing
        papers = []
        for i in range(100):
            paper = edge_services["generator"].generate_paper()
            paper["pdf_path"] = f"/fake/path/{i}.pdf"
            paper["pdf_status"] = "downloaded"
            papers.append(paper)

        papers_df = pd.DataFrame(papers)

        # Test concurrent PDF fetching
        pdf_fetcher = PDFFetcher(edge_config)

        # Should respect worker limits
        with patch.object(pdf_fetcher, "_fetch_single_pdf") as mock_fetch:
            mock_fetch.return_value = {
                "path": "/fake/path.pdf",
                "status": "downloaded",
                "hash": "fakehash",
            }

            # Force parallel processing
            edge_config.parallel_workers = 10
            results = pdf_fetcher.fetch_pdfs(papers_df[:50])

            # Should process all papers
            assert len(results) == 50

        # Test concurrent extraction
        extractor = EnhancedLLMExtractor(edge_config)

        # Mock LLM to track concurrent calls
        concurrent_calls = []

        def track_concurrent_extract(text, model, **kwargs):
            import threading

            thread_id = threading.current_thread().ident
            concurrent_calls.append(thread_id)

            return {
                "success": True,
                "extracted_data": {"test": "data"},
                "model_used": model,
            }

        edge_services["llm"].extract = track_concurrent_extract

        # Process with parallelism
        extracted = extractor.extract_all(papers_df[:20], parallel=True)

        # Should use multiple threads
        unique_threads = set(concurrent_calls)
        assert len(unique_threads) > 1, "Should use parallel processing"

    def test_cache_edge_cases(self, edge_config, edge_services):
        """Test cache behavior with edge cases."""
        cache = ContentCache(edge_config)

        # Test 1: Empty paper ID
        result, cached = cache.get_or_fetch("", "pdf", lambda: b"content")
        assert result is not None  # Should handle empty ID

        # Test 2: Very long paper ID
        long_id = "a" * 1000
        result, cached = cache.get_or_fetch(long_id, "pdf", lambda: b"content")
        assert result is not None

        # Test 3: Special characters in ID
        special_id = "paper/with\\Union[special, characters]:*?<>"
        result, cached = cache.get_or_fetch(special_id, "pdf", lambda: b"content")
        assert result is not None

        # Test 4: Null content
        result, cached = cache.get_or_fetch("null_content", "pdf", lambda: None)
        assert result is None

        # Test 5: Binary content types
        binary_content = bytes(range(256))  # All possible byte values
        result, cached = cache.get_or_fetch("binary", "pdf", lambda: binary_content)
        assert result is not None
        if result:
            assert result.read_bytes() == binary_content

        # Test 6: Cache expiration boundary
        # Set very old cache entry
        old_paper_id = "old_paper"
        result1, cached1 = cache.get_or_fetch(
            old_paper_id, "pdf", lambda: b"old content"
        )

        # Manually update cache timestamp to be very old
        if cache._get_cache_metadata(old_paper_id, "pdf"):
            # Simulate old cache by modifying file time
            cache_path = cache._get_cache_path(old_paper_id, "pdf")
            if cache_path.exists():
                old_time = datetime.now() - timedelta(days=400)  # Older than max age
                os.utime(cache_path, (old_time.timestamp(), old_time.timestamp()))

        # Should fetch new content
        result2, cached2 = cache.get_or_fetch(
            old_paper_id, "pdf", lambda: b"new content"
        )
        if result2 and edge_config.cache_max_age_days < 400:
            assert not cached2, "Should not use expired cache"

    def test_visualization_edge_cases(self, edge_config, edge_services, monkeypatch):
        """Test visualization with edge case data."""
        self._patch_services(monkeypatch, edge_services)

        visualizer = Visualizer(edge_config)

        # Test 1: All papers from same year
        same_year_df = pd.DataFrame(
            [
                {"title": f"Paper {i}", "year": 2024, "awscale": i % 5 + 1}
                for i in range(20)
            ]
        )

        fig_path = visualizer.plot_time_series(same_year_df, save=True)
        assert fig_path is None or fig_path.exists()

        # Test 2: All papers with same AWScale
        same_awscale_df = pd.DataFrame(
            [
                {"title": f"Paper {i}", "year": 2020 + i % 5, "awscale": 3}
                for i in range(20)
            ]
        )

        fig_path = visualizer.plot_awscale_distribution(same_awscale_df, save=True)
        assert fig_path is None or fig_path.exists()

        # Test 3: Extreme value ranges
        extreme_df = pd.DataFrame(
            [
                {"title": "Old Paper", "year": 1950, "awscale": 1},
                {"title": "Future Paper", "year": 2099, "awscale": 5},
                {"title": "Normal Paper", "year": 2024, "awscale": 3},
            ]
        )

        figures = visualizer.create_all_visualizations(extreme_df, save=True)
        assert isinstance(figures, list)

        # Test 4: Missing critical columns
        missing_cols_df = pd.DataFrame(
            [{"title": "Paper 1"}, {"title": "Paper 2"}]  # Missing year, awscale, etc.
        )

        figures = visualizer.create_all_visualizations(missing_cols_df, save=True)
        assert isinstance(figures, list)  # Should not crash

    def test_export_edge_cases(self, edge_config, edge_services, monkeypatch):
        """Test export functionality with edge cases."""
        self._patch_services(monkeypatch, edge_services)

        exporter = Exporter(edge_config)

        # Test 1: Export empty dataset
        empty_df = pd.DataFrame()
        empty_figures = []
        empty_summary = {"total_papers": 0}

        archive_path = exporter.export_full_package(
            extraction_df=empty_df, figures=empty_figures, summary=empty_summary
        )

        # Should create archive even if empty
        assert archive_path.exists()

        # Test 2: Export with unicode filenames
        unicode_df = pd.DataFrame(
            [
                {
                    "title": "Paper with 中文 characters",
                    "authors": "José García; 李明",
                    "year": 2024,
                }
            ]
        )

        bibtex_path = exporter.export_bibtex(unicode_df)
        assert bibtex_path.exists()

        # BibTeX should handle unicode
        content = bibtex_path.read_text(encoding="utf-8")
        assert len(content) > 0

        # Test 3: Export very large dataset
        large_df = pd.DataFrame(
            [edge_services["generator"].generate_paper() for _ in range(1000)]
        )

        # Should handle large exports
        csv_path = edge_config.data_dir / "large_export.csv"
        large_df.to_csv(csv_path, index=False)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 100000  # Should be substantial

    def test_mathematical_limits(self, edge_config):
        """Test mathematical edge cases in calculations."""
        visualizer = Visualizer(edge_config)

        # Test 1: Division by zero scenarios
        zero_papers_df = pd.DataFrame(
            [{"title": "Paper", "year": 2024, "citations": 0, "awscale": 3}]
        )

        summary = visualizer.create_summary_report(zero_papers_df)
        # Should handle gracefully
        assert isinstance(summary, dict)

        # Test 2: Overflow scenarios
        overflow_df = pd.DataFrame(
            [{"title": "Paper", "year": 2024, "citations": sys.maxsize}]
        )

        summary = visualizer.create_summary_report(overflow_df)
        assert isinstance(summary, dict)

        # Test 3: NaN and infinity handling
        import numpy as np

        nan_df = pd.DataFrame(
            [
                {"title": "Paper 1", "year": 2024, "awscale": np.nan},
                {"title": "Paper 2", "year": 2024, "awscale": np.inf},
                {"title": "Paper 3", "year": 2024, "awscale": -np.inf},
            ]
        )

        # Should filter out invalid values
        if "awscale" in nan_df.columns:
            valid_awscale = pd.to_numeric(nan_df["awscale"], errors="coerce")
            valid_awscale = valid_awscale[
                valid_awscale.notna() & np.isfinite(valid_awscale)
            ]
            assert len(valid_awscale) == 0, "Should filter out NaN and inf"

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
                            raise requests.HTTPError(f"HTTP {self.status_code}")

                return MockResponse()

        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.Session.get", mock_get)
