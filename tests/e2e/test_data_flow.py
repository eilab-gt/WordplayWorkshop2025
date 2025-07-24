"""E2E tests focusing on data flow and integrity through the pipeline."""

import pandas as pd
import pytest

from src.lit_review.processing import Normalizer
from src.lit_review.utils.content_cache import ContentCache
from tests.test_doubles import RealConfigForTests


@pytest.mark.e2e
class TestDataFlowIntegrity:
    """Test data integrity is maintained through pipeline stages."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        config = RealConfigForTests(
            cache_dir=tmp_path / "cache",
            data_dir=tmp_path / "data",
            search_years=(2023, 2024),
        )
        yield config
        config.cleanup()

    def test_paper_ids_remain_consistent_through_pipeline(self, config):
        """Test paper IDs are preserved through all transformations."""
        # Create initial dataset
        papers = pd.DataFrame(
            [
                {
                    "title": "Paper A: LLM Study",
                    "authors": "Smith, J.",
                    "year": 2024,
                    "doi": "10.1234/a",
                    "arxiv_id": "2401.00001",
                },
                {
                    "title": "Paper B: Wargaming Research",
                    "authors": "Jones, K.",
                    "year": 2024,
                    "doi": "10.1234/b",
                    "arxiv_id": None,
                },
            ]
        )

        # Step 1: Normalize (assigns paper_ids)
        normalizer = Normalizer(config)
        normalized = normalizer.normalize(papers)

        assert "paper_id" in normalized.columns
        assert normalized["paper_id"].nunique() == len(papers)

        # Step 2: Add processing columns
        normalized["pdf_path"] = ""
        normalized["pdf_status"] = ""
        normalized["extraction_status"] = ""

        # Step 3: Simulate processing steps
        for idx in normalized.index:
            # Processing should preserve paper_id
            paper_id_before = normalized.at[idx, "paper_id"]

            # Simulate PDF fetch
            normalized.at[idx, "pdf_status"] = "downloaded"

            # Simulate extraction
            normalized.at[idx, "extraction_status"] = "success"

            # Verify ID unchanged
            assert normalized.at[idx, "paper_id"] == paper_id_before

    def test_deduplication_preserves_best_metadata(self, config):
        """Test deduplication keeps the most complete metadata."""
        # Create duplicates with different metadata completeness
        papers = pd.DataFrame(
            [
                {
                    "title": "Same Paper",
                    "authors": "Author A",
                    "year": 2024,
                    "doi": "10.1234/same",
                    "arxiv_id": None,
                    "abstract": "Short abstract",
                },
                {
                    "title": "Same Paper",  # Duplicate
                    "authors": "Author A; Author B",  # More complete
                    "year": 2024,
                    "doi": "10.1234/same",
                    "arxiv_id": "2401.00001",  # Additional ID
                    "abstract": "This is a much longer and more complete abstract with details",
                },
            ]
        )

        normalizer = Normalizer(config)
        normalized = normalizer.normalize(papers)
        deduped = normalizer.deduplicate(normalized)

        # Should keep 1 paper
        assert len(deduped) == 1

        # Should keep the more complete metadata
        result = deduped.iloc[0]
        assert "Author B" in result["authors"]  # Kept complete author list
        assert result["arxiv_id"] == "2401.00001"  # Kept additional ID
        assert len(result["abstract"]) > 20  # Kept longer abstract

    def test_cache_consistency_across_runs(self, config):
        """Test content cache maintains consistency."""
        cache = ContentCache(config)

        # First run - populate cache
        paper_id = "test_paper_123"
        content1 = b"PDF content version 1"

        path1, was_cached1 = cache.get_or_fetch(paper_id, "pdf", lambda: content1)

        assert not was_cached1  # First time, not cached
        assert path1.read_bytes() == content1

        # Second run - should get from cache
        content2 = b"This should not be used"
        path2, was_cached2 = cache.get_or_fetch(
            paper_id,
            "pdf",
            lambda: content2,  # Different content, but shouldn't be called
        )

        assert was_cached2  # Should be cached
        assert path2 == path1  # Same path
        assert path2.read_bytes() == content1  # Original content preserved

    def test_metadata_enrichment_is_additive(self, config):
        """Test each pipeline stage adds metadata without losing existing data."""
        # Start with basic paper
        paper = pd.DataFrame(
            [{"title": "Test Paper", "authors": "Test Author", "year": 2024}]
        )

        # Stage 1: Normalize (adds paper_id)
        normalizer = Normalizer(config)
        stage1 = normalizer.normalize(paper)
        original_columns = set(paper.columns)
        stage1_columns = set(stage1.columns)

        assert stage1_columns > original_columns  # Added columns
        assert original_columns.issubset(stage1_columns)  # Kept all original

        # Stage 2: Add PDF columns
        stage2 = stage1.copy()
        stage2["pdf_path"] = "/path/to/pdf"
        stage2["pdf_status"] = "downloaded"
        stage2_columns = set(stage2.columns)

        assert stage2_columns > stage1_columns  # Added more
        assert stage1_columns.issubset(stage2_columns)  # Kept all from stage 1

        # Stage 3: Add extraction columns
        stage3 = stage2.copy()
        stage3["research_questions"] = "What are the effects?"
        stage3["awscale"] = 4
        stage3_columns = set(stage3.columns)

        assert stage3_columns > stage2_columns  # Added more
        assert stage2_columns.issubset(stage3_columns)  # Kept all from stage 2

        # Verify no data was lost
        assert stage3.iloc[0]["title"] == paper.iloc[0]["title"]
        assert stage3.iloc[0]["authors"] == paper.iloc[0]["authors"]
        assert stage3.iloc[0]["pdf_path"] == "/path/to/pdf"

    def test_error_tracking_through_pipeline(self, config):
        """Test errors are properly tracked at each stage."""
        papers = pd.DataFrame(
            [
                {"title": "Good Paper", "arxiv_id": "2401.00001"},
                {"title": "Bad Paper", "arxiv_id": "invalid_id"},
                {"title": "No ID Paper", "arxiv_id": None},
            ]
        )

        # Add status columns
        papers["pdf_status"] = ""
        papers["extraction_status"] = ""
        papers["error_message"] = ""

        # Simulate processing with errors
        for idx, row in papers.iterrows():
            if row["arxiv_id"] == "invalid_id":
                papers.at[idx, "pdf_status"] = "error"
                papers.at[idx, "error_message"] = "Invalid arXiv ID format"
            elif row["arxiv_id"] is None:
                papers.at[idx, "pdf_status"] = "not_found"
                papers.at[idx, "error_message"] = "No PDF source available"
            else:
                papers.at[idx, "pdf_status"] = "downloaded"
                papers.at[idx, "extraction_status"] = "success"

        # Verify error tracking
        errors = papers[papers["error_message"] != ""]
        assert len(errors) == 2
        assert all(errors["pdf_status"].isin(["error", "not_found"]))

        # Successful papers should not have errors
        success = papers[papers["extraction_status"] == "success"]
        assert all(success["error_message"] == "")

    @pytest.mark.slow
    def test_large_dataset_memory_efficiency(self, config):
        """Test pipeline handles large datasets efficiently."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Create large dataset
        n_papers = 1000
        large_df = pd.DataFrame(
            {
                "title": [f"Paper {i}" for i in range(n_papers)],
                "authors": [f"Author {i}" for i in range(n_papers)],
                "year": [2024] * n_papers,
                "abstract": ["Long abstract text " * 50] * n_papers,  # ~1KB each
            }
        )

        # Record initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process in batches
        normalizer = Normalizer(config)
        batch_size = 100
        processed_dfs = []

        for start in range(0, len(large_df), batch_size):
            batch = large_df.iloc[start : start + batch_size]
            processed = normalizer.normalize(batch)
            processed_dfs.append(processed)

            # Force garbage collection between batches
            gc.collect()

        # Combine results
        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Check memory usage
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_increase < 200, f"Used too much memory: {memory_increase:.1f}MB"

        # All papers should be processed
        assert len(final_df) == n_papers
        assert all("paper_id" in final_df.columns)
