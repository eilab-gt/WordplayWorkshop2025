"""Comprehensive tests for the BatchProcessor module to improve coverage."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.lit_review.processing.batch_processor import BatchProcessor


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration object."""
    config = Mock()
    # Set attributes before setting __dict__
    data_dir = tmp_path
    production_batch_size = 100
    parallel_workers = 4
    memory_limit_gb = 8

    # Set the attributes
    config.data_dir = data_dir
    config.production_batch_size = production_batch_size
    config.parallel_workers = parallel_workers
    config.memory_limit_gb = memory_limit_gb

    # Create a proper __dict__ for hashing
    config.__dict__.update(
        {
            "data_dir": str(data_dir),
            "production_batch_size": production_batch_size,
            "parallel_workers": parallel_workers,
            "memory_limit_gb": memory_limit_gb,
        }
    )

    return config


@pytest.fixture
def sample_papers_df():
    """Create sample paper DataFrame for testing."""
    np.random.seed(42)
    n_papers = 500
    return pd.DataFrame(
        {
            "title": [
                f"Paper {i}: Research on Topic {i % 10}" for i in range(n_papers)
            ],
            "authors": [f"Author{i}; CoAuthor{i+1}" for i in range(n_papers)],
            "year": np.random.randint(2020, 2025, n_papers),
            "doi": [f"10.1234/test{i}" if i % 3 != 0 else "" for i in range(n_papers)],
            "arxiv_id": [
                f"2301.{i:05d}" if i % 5 == 0 else "" for i in range(n_papers)
            ],
            "citations": np.random.randint(0, 100, n_papers),
            "abstract": ["This is a test abstract " * 20 for _ in range(n_papers)],
            "source_db": np.random.choice(
                ["arxiv", "crossref", "semantic_scholar"], n_papers
            ),
        }
    )


@pytest.fixture
def small_papers_df():
    """Create small paper DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": ["Paper A", "Paper B", "Paper C", "Paper D", "Paper E"],
            "authors": ["Author1", "Author2", "Author3", "Author4", "Author5"],
            "year": [2023, 2023, 2024, 2024, 2023],
            "doi": ["10.1234/a", "10.1234/b", "", "10.1234/d", "10.1234/e"],
            "citations": [10, 5, 15, 3, 8],
            "abstract": ["Abstract " * 10] * 5,
        }
    )


class TestBatchProcessor:
    """Test cases for the BatchProcessor class."""

    def test_init(self, mock_config):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(mock_config)

        assert processor.config == mock_config
        assert processor.batch_size == 100
        assert processor.max_workers == 4
        assert processor.memory_limit_gb == 8
        assert (
            processor.progress_db
            == Path(str(mock_config.data_dir)) / "batch_progress.db"
        )
        assert processor.metrics["processed"] == 0
        assert processor.metrics["failed"] == 0

    def test_init_progress_db(self, mock_config):
        """Test progress database initialization."""
        processor = BatchProcessor(mock_config)

        # Check database was created
        assert processor.progress_db.exists()

        # Check tables were created
        conn = sqlite3.connect(str(processor.progress_db))
        cursor = conn.cursor()

        # Check batch_jobs table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='batch_jobs'"
        )
        assert cursor.fetchone() is not None

        # Check batch_progress table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='batch_progress'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_process_papers_batch(self, mock_config, sample_papers_df):
        """Test batch processing of papers."""
        mock_config.production_batch_size = 50

        processor = BatchProcessor(mock_config)

        # Mock processor function
        def mock_processor_func(batch_df, **kwargs):
            # Simulate processing - return subset with added column
            result = batch_df.copy()
            result["processed"] = True
            result["score"] = np.random.rand(len(batch_df))
            return result

        # Process papers
        result_df = processor.process_papers_batch(
            sample_papers_df,
            mock_processor_func,
            job_type="test_processing",
            test_param="test_value",
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_papers_df)
        assert "processed" in result_df.columns
        assert "score" in result_df.columns
        assert result_df["processed"].all()
        assert processor.metrics["processed"] == len(sample_papers_df)

    def test_process_papers_batch_with_resume(self, mock_config, sample_papers_df):
        """Test resuming batch processing."""
        mock_config.production_batch_size = 50

        processor = BatchProcessor(mock_config)

        # Create a job with some progress
        job_id = "test_job_001"
        processor._create_job(job_id, "test", len(sample_papers_df))

        # Log some completed batches
        processor._log_batch_progress(job_id, 0, 50, 1.0)
        processor._log_batch_progress(job_id, 1, 50, 1.1)

        # Mock processor function
        def mock_processor_func(batch_df, **kwargs):
            result = batch_df.copy()
            result["processed"] = True
            return result

        # Mock _get_completed_batches to return completed batches
        with patch.object(processor, "_get_completed_batches", return_value=[0, 1]):
            # Resume processing from batch 2
            result_df = processor.process_papers_batch(
                sample_papers_df,
                mock_processor_func,
                job_type="test_processing",
                resume_job_id=job_id,
            )

        # Should process only remaining papers (batch 2 onwards)
        assert len(result_df) <= len(sample_papers_df) - 100  # First 2 batches skipped

    def test_process_single_batch(self, mock_config, small_papers_df):
        """Test processing a single batch."""
        processor = BatchProcessor(mock_config)

        # Mock processor function
        def mock_processor_func(batch_df, **kwargs):
            result = batch_df.copy()
            result["score"] = kwargs.get("multiplier", 1) * 10
            return result

        result = processor._process_single_batch(
            small_papers_df, mock_processor_func, batch_idx=0, multiplier=5
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(small_papers_df)
        assert result["score"].iloc[0] == 50

    def test_process_oversized_batch(self, mock_config, sample_papers_df):
        """Test handling of oversized batches."""
        processor = BatchProcessor(mock_config)

        # Mock memory estimation to trigger oversized batch handling
        with patch.object(
            processor, "_estimate_batch_memory", return_value=10.0
        ):  # Over limit
            # Mock processor function
            def mock_processor_func(batch_df, **kwargs):
                return batch_df.copy()

            result = processor._process_single_batch(
                sample_papers_df, mock_processor_func, batch_idx=0
            )

            # Should still process successfully by subdividing
            assert result is not None

    def test_parallel_process_sources(self, mock_config):
        """Test parallel processing of multiple sources."""
        processor = BatchProcessor(mock_config)

        # Create source configurations
        source_configs = [
            {"name": "source1", "param": "value1"},
            {"name": "source2", "param": "value2"},
            {"name": "source3", "param": "value3"},
        ]

        # Mock processor function
        def mock_source_processor(config):
            # Return different sized DataFrames for each source
            sizes = {"source1": 10, "source2": 20, "source3": 15}
            size = sizes.get(config["name"], 5)
            return pd.DataFrame(
                {"source": [config["name"]] * size, "data": range(size)}
            )

        results = processor.parallel_process_sources(
            source_configs, mock_source_processor, max_workers=2
        )

        assert len(results) == 3
        assert "source1" in results
        assert "source2" in results
        assert "source3" in results
        assert len(results["source1"]) == 10
        assert len(results["source2"]) == 20
        assert len(results["source3"]) == 15

    def test_parallel_process_sources_with_error(self, mock_config):
        """Test parallel processing with source errors."""
        processor = BatchProcessor(mock_config)

        source_configs = [
            {"name": "good_source", "param": "value1"},
            {"name": "bad_source", "param": "value2"},
        ]

        def mock_source_processor(config):
            if config["name"] == "bad_source":
                raise Exception("Source processing error")
            return pd.DataFrame({"data": [1, 2, 3]})

        results = processor.parallel_process_sources(
            source_configs, mock_source_processor
        )

        assert len(results) == 2
        assert len(results["good_source"]) == 3
        assert len(results["bad_source"]) == 0  # Empty DataFrame on error

    def test_chunked_deduplication(self, mock_config):
        """Test chunked deduplication."""
        processor = BatchProcessor(mock_config)

        # Create DataFrame with duplicates
        df = pd.DataFrame(
            {
                "title": ["Paper A", "Paper A", "Paper B", "Paper B", "Paper C"],
                "doi": ["10.1234/a", "10.1234/a", "", "", "10.1234/c"],
                "year": [2023, 2023, 2024, 2024, 2023],
                "citations": [10, 5, 20, 15, 8],
            }
        )

        deduplicated = processor.chunked_deduplication(df, chunk_size=2)

        # Should remove DOI duplicates and title duplicates
        assert len(deduplicated) < len(df)
        # Check no duplicate DOIs (excluding empty)
        doi_counts = deduplicated[deduplicated["doi"] != ""]["doi"].value_counts()
        assert doi_counts.max() == 1

    def test_chunked_title_dedup(self, mock_config):
        """Test chunked title-based deduplication."""
        processor = BatchProcessor(mock_config)

        # Create DataFrame with similar titles
        df = pd.DataFrame(
            {
                "title": [
                    "Deep Learning for NLP",
                    "Deep Learning for NLP.",  # Same with period
                    "deep learning for nlp",  # Lowercase
                    "Machine Learning Applications",
                    "Machine Learning Applications!",  # With punctuation
                ],
                "year": [2023, 2023, 2024, 2024, 2023],
                "citations": [10, 5, 20, 15, 8],
            }
        )

        deduplicated = processor._chunked_title_dedup(df, chunk_size=2)

        # Should remove similar titles
        assert len(deduplicated) == 2  # Only unique normalized titles

    def test_normalize_title(self, mock_config):
        """Test title normalization."""
        processor = BatchProcessor(mock_config)

        test_cases = [
            ("Deep Learning for NLP!", "deep learning for nlp"),
            ("  Spaces   Everywhere  ", "spaces everywhere"),
            ("Title-with-hyphens", "title with hyphens"),
            ("Title: Subtitle (2023)", "title subtitle 2023"),
            ("", ""),
        ]

        for input_title, expected in test_cases:
            normalized = processor._normalize_title(input_title)
            assert normalized == expected

    def test_stream_large_dataset(self, mock_config):
        """Test streaming large datasets."""
        processor = BatchProcessor(mock_config)

        # Create a test CSV file
        test_file = Path(str(mock_config.data_dir)) / "test_data.csv"
        sample_df = pd.DataFrame(
            {
                "title": [f"Paper {i}" for i in range(1000)],
                "year": list(range(2020, 2025)) * 200,
            }
        )
        sample_df.to_csv(test_file, index=False)

        # Stream the file
        chunks = list(processor.stream_large_dataset(test_file, chunk_size=250))

        assert len(chunks) == 4  # 1000 / 250
        assert all(len(chunk) == 250 for chunk in chunks)
        assert chunks[0]["title"].iloc[0] == "Paper 0"

    def test_batch_iterator(self, mock_config, sample_papers_df):
        """Test batch iterator."""
        mock_config.production_batch_size = 100

        processor = BatchProcessor(mock_config)

        batches = list(processor._batch_iterator(sample_papers_df))

        assert len(batches) == 5  # 500 papers / 100 batch size
        assert all(len(batch) == 100 for batch in batches[:-1])
        assert len(batches[-1]) == 100  # Last batch might be smaller

    def test_estimate_batch_memory(self, mock_config, small_papers_df):
        """Test batch memory estimation."""
        processor = BatchProcessor(mock_config)

        memory_gb = processor._estimate_batch_memory(small_papers_df)

        assert isinstance(memory_gb, float)
        assert memory_gb > 0
        assert memory_gb < 1  # Small DataFrame should be < 1GB

    def test_check_memory_usage(self, mock_config):
        """Test memory usage monitoring."""
        processor = BatchProcessor(mock_config)

        # Mock psutil import and usage
        mock_psutil_module = Mock()
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 4 * 1024**3  # 4GB
        mock_psutil_module.Process.return_value = mock_process

        with patch.dict("sys.modules", {"psutil": mock_psutil_module}):
            processor._check_memory_usage()

            assert len(processor.metrics["memory_usage"]) == 1
            assert processor.metrics["memory_usage"][0] == 4.0

    def test_check_memory_usage_no_psutil(self, mock_config):
        """Test memory usage monitoring without psutil."""
        processor = BatchProcessor(mock_config)

        # Simulate ImportError by not having psutil in modules
        with patch.dict("sys.modules", {"psutil": None}):
            processor._check_memory_usage()  # Should not raise error

        assert len(processor.metrics["memory_usage"]) == 0

    def test_log_progress_summary(self, mock_config, caplog):
        """Test progress summary logging."""
        processor = BatchProcessor(mock_config)
        processor.metrics["start_time"] = time.time() - 60  # 1 minute ago
        processor.metrics["processed"] = 200
        processor.metrics["failed"] = 5
        processor.metrics["batch_times"] = [1.0, 1.2, 0.9, 1.1, 1.0]

        with caplog.at_level("INFO"):
            processor._log_progress_summary(current_batch=2, total_batches=10)

        log_text = caplog.text
        assert "Progress: 20.0%" in log_text
        assert "Processed: 200" in log_text
        assert "Failed: 5" in log_text

    def test_generate_job_id(self, mock_config):
        """Test job ID generation."""
        processor = BatchProcessor(mock_config)

        job_id = processor._generate_job_id("test_job")

        assert job_id.startswith("test_job_")
        assert len(job_id) > 15  # Has timestamp

    def test_create_and_complete_job(self, mock_config):
        """Test job creation and completion."""
        processor = BatchProcessor(mock_config)

        job_id = "test_job_001"
        processor._create_job(job_id, "test_type", 1000)

        # Check job was created
        conn = sqlite3.connect(str(processor.progress_db))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,))
        job = cursor.fetchone()
        conn.close()

        assert job is not None
        assert job[5] == "running"  # status

        # Complete the job
        processor._complete_job(job_id, 950)

        # Check job was completed
        conn = sqlite3.connect(str(processor.progress_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, processed_items FROM batch_jobs WHERE job_id = ?", (job_id,)
        )
        status, processed = cursor.fetchone()
        conn.close()

        assert status == "completed"
        assert processed == 950

    def test_fail_job(self, mock_config):
        """Test job failure handling."""
        processor = BatchProcessor(mock_config)

        job_id = "test_job_002"
        processor._create_job(job_id, "test_type", 1000)

        # Fail the job
        processor._fail_job(job_id, "Test error message")

        # Check job was marked as failed
        conn = sqlite3.connect(str(processor.progress_db))
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM batch_jobs WHERE job_id = ?", (job_id,))
        status = cursor.fetchone()[0]
        conn.close()

        assert "failed" in status
        assert "Test error message" in status

    def test_log_batch_error(self, mock_config):
        """Test batch error logging."""
        processor = BatchProcessor(mock_config)

        job_id = "test_job_003"
        processor._log_batch_error(job_id, 5, "Batch processing failed")

        # Check error was logged
        conn = sqlite3.connect(str(processor.progress_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, error_message FROM batch_progress WHERE job_id = ? AND batch_id = ?",
            (job_id, 5),
        )
        status, error_msg = cursor.fetchone()
        conn.close()

        assert status == "failed"
        assert error_msg == "Batch processing failed"

    def test_get_job_status(self, mock_config):
        """Test job status retrieval."""
        processor = BatchProcessor(mock_config)

        # Create a job with some progress
        job_id = "test_job_004"
        processor._create_job(job_id, "test_type", 500)
        processor._log_batch_progress(job_id, 0, 100, 1.5)
        processor._log_batch_progress(job_id, 1, 100, 1.2)
        processor._log_batch_error(job_id, 2, "Test error")

        status = processor.get_job_status(job_id)

        assert "job_info" in status
        assert status["completed_batches"] == 2
        assert status["failed_batches"] == 1
        assert status["avg_batch_time"] > 0
        assert status["progress_percentage"] == 40.0  # 2 completed out of 5 expected

    def test_get_job_status_not_found(self, mock_config):
        """Test job status retrieval for non-existent job."""
        processor = BatchProcessor(mock_config)

        status = processor.get_job_status("non_existent_job")

        assert "error" in status
        assert status["error"] == "Job not found"

    def test_process_batch_with_error(self, mock_config, sample_papers_df):
        """Test batch processing with errors."""
        mock_config.production_batch_size = 100

        processor = BatchProcessor(mock_config)

        # Mock processor function that fails for some batches
        call_count = [0]

        def mock_processor_func(batch_df, **kwargs):
            # Fail for second batch
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated batch error")
            return batch_df.copy()

        result_df = processor.process_papers_batch(
            sample_papers_df, mock_processor_func, job_type="test_with_errors"
        )

        # Should still return results from successful batches
        assert isinstance(result_df, pd.DataFrame)
        # When _process_single_batch returns None, the batch is skipped but papers aren't counted as failed
        # The actual failure counting happens at a different level
        assert len(result_df) < len(sample_papers_df)  # Some batches failed
