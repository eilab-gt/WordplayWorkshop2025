"""Batch processing module for production-scale literature review pipeline."""

import hashlib
import logging
import multiprocessing as mp
import sqlite3
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BatchProcessor:
    """High-performance batch processor for production-scale operations."""

    def __init__(self, config):
        """Initialize batch processor with production configuration."""
        self.config = config
        self.batch_size = getattr(config, "production_batch_size", 1000)
        self.max_workers = getattr(config, "parallel_workers", mp.cpu_count())
        self.memory_limit_gb = getattr(config, "memory_limit_gb", 8)

        # Processing state tracking
        self.progress_db = Path(config.data_dir) / "batch_progress.db"
        self._init_progress_db()

        # Performance monitoring
        self.metrics = {
            "processed": 0,
            "failed": 0,
            "start_time": None,
            "batch_times": [],
            "memory_usage": [],
        }

    def _init_progress_db(self):
        """Initialize progress tracking database."""
        self.progress_db.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT,
                total_items INTEGER,
                processed_items INTEGER,
                failed_items INTEGER,
                status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                config_hash TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_progress (
                job_id TEXT,
                batch_id INTEGER,
                batch_size INTEGER,
                processing_time REAL,
                memory_usage REAL,
                status TEXT,
                error_message TEXT,
                timestamp TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def process_papers_batch(
        self,
        papers_df: pd.DataFrame,
        processor_func: Callable,
        job_type: str = "processing",
        resume_job_id: str | None = None,
        **processor_kwargs,
    ) -> pd.DataFrame:
        """Process papers in optimized batches with progress tracking.

        Args:
            papers_df: DataFrame of papers to process
            processor_func: Function to apply to each batch
            job_type: Type of processing job for tracking
            resume_job_id: Job ID to resume from
            **processor_kwargs: Additional arguments for processor function

        Returns:
            DataFrame of processed results
        """
        # Setup job tracking
        job_id = resume_job_id or self._generate_job_id(job_type)

        if resume_job_id:
            # Resume existing job
            processed_batches = self._get_completed_batches(job_id)
            start_idx = len(processed_batches) * self.batch_size
            papers_df = papers_df.iloc[start_idx:]
            logger.info(f"Resuming job {job_id} from batch {len(processed_batches)}")
        else:
            # Start new job
            self._create_job(job_id, job_type, len(papers_df))

        logger.info(
            f"Processing {len(papers_df):,} papers in batches of {self.batch_size}"
        )

        self.metrics["start_time"] = time.time()
        results = []

        try:
            # Process in batches
            for batch_idx, batch_df in enumerate(self._batch_iterator(papers_df)):
                batch_start = time.time()

                try:
                    # Process batch
                    batch_result = self._process_single_batch(
                        batch_df, processor_func, batch_idx, **processor_kwargs
                    )

                    if batch_result is not None and len(batch_result) > 0:
                        results.append(batch_result)

                    # Update metrics
                    batch_time = time.time() - batch_start
                    self.metrics["batch_times"].append(batch_time)
                    self.metrics["processed"] += len(batch_df)

                    # Log progress
                    self._log_batch_progress(
                        job_id, batch_idx, len(batch_df), batch_time
                    )

                    # Memory management
                    self._check_memory_usage()

                    if batch_idx % 10 == 0:  # Log every 10 batches
                        self._log_progress_summary(
                            batch_idx, len(papers_df) // self.batch_size
                        )

                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    self.metrics["failed"] += len(batch_df)
                    self._log_batch_error(job_id, batch_idx, str(e))
                    continue

            # Combine results
            if results:
                final_df = pd.concat(results, ignore_index=True)
            else:
                final_df = pd.DataFrame()

            # Mark job complete
            self._complete_job(job_id, len(final_df))

            logger.info(f"Batch processing complete: {len(final_df):,} results")
            return final_df

        except Exception as e:
            self._fail_job(job_id, str(e))
            logger.error(f"Batch processing failed: {e}")
            raise

    def _process_single_batch(
        self, batch_df: pd.DataFrame, processor_func: Callable, batch_idx: int, **kwargs
    ) -> pd.DataFrame | None:
        """Process a single batch with error handling."""
        try:
            # Check if batch is too large for memory
            if self._estimate_batch_memory(batch_df) > self.memory_limit_gb:
                logger.warning(f"Batch {batch_idx} too large, splitting further")
                return self._process_oversized_batch(batch_df, processor_func, **kwargs)

            # Process batch
            result = processor_func(batch_df, **kwargs)

            return result

        except Exception as e:
            logger.error(f"Batch {batch_idx} processing error: {e}")
            return None

    def _process_oversized_batch(
        self, batch_df: pd.DataFrame, processor_func: Callable, **kwargs
    ) -> pd.DataFrame | None:
        """Handle batches that are too large for memory by sub-dividing."""
        sub_batch_size = len(batch_df) // 4  # Quarter the size
        sub_results = []

        for i in range(0, len(batch_df), sub_batch_size):
            sub_batch = batch_df.iloc[i : i + sub_batch_size]
            try:
                sub_result = processor_func(sub_batch, **kwargs)
                if sub_result is not None:
                    sub_results.append(sub_result)
            except Exception as e:
                logger.error(f"Sub-batch processing failed: {e}")
                continue

        if sub_results:
            return pd.concat(sub_results, ignore_index=True)
        return None

    def parallel_process_sources(
        self,
        source_configs: list[dict[str, Any]],
        processor_func: Callable,
        max_workers: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Process multiple sources in parallel.

        Args:
            source_configs: List of source configuration dictionaries
            processor_func: Function to process each source
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping source names to results
        """
        max_workers = max_workers or min(len(source_configs), self.max_workers)

        logger.info(
            f"Processing {len(source_configs)} sources with {max_workers} workers"
        )

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all source processing tasks
            future_to_source = {
                executor.submit(processor_func, config): config["name"]
                for config in source_configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    result = future.result()
                    results[source_name] = result
                    logger.info(
                        f"Completed processing {source_name}: {len(result)} items"
                    )
                except Exception as e:
                    logger.error(f"Source {source_name} processing failed: {e}")
                    results[source_name] = pd.DataFrame()

        return results

    def chunked_deduplication(
        self, df: pd.DataFrame, chunk_size: int | None = None
    ) -> pd.DataFrame:
        """Perform memory-efficient deduplication on large datasets."""
        chunk_size = chunk_size or self.batch_size

        logger.info(f"Starting chunked deduplication: {len(df):,} papers")

        # Stage 1: DOI-based deduplication (most efficient)
        df_with_doi = df[df["doi"].notna() & (df["doi"] != "")]
        df_without_doi = df[df["doi"].isna() | (df["doi"] == "")]

        if len(df_with_doi) > 0:
            df_with_doi = df_with_doi.drop_duplicates(subset=["doi"], keep="first")

        # Stage 2: Chunked title-based deduplication for papers without DOI
        if len(df_without_doi) > 0:
            df_without_doi = self._chunked_title_dedup(df_without_doi, chunk_size)

        # Combine results
        result_df = pd.concat([df_with_doi, df_without_doi], ignore_index=True)

        logger.info(f"Deduplication complete: {len(df):,} → {len(result_df):,} papers")
        return result_df

    def _chunked_title_dedup(self, df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
        """Perform title-based deduplication in chunks to manage memory."""
        if len(df) == 0:
            return df

        # Sort by year desc, citations desc to keep best papers
        df = df.sort_values(["year", "citations"], ascending=[False, False])

        seen_titles = set()
        keep_indices = []

        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]

            for idx, row in chunk_df.iterrows():
                title = str(row.get("title", "")).lower().strip()
                if not title:
                    continue

                # Simple title normalization for efficiency
                normalized_title = self._normalize_title(title)

                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    keep_indices.append(idx)

        return df.loc[keep_indices].reset_index(drop=True)

    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication matching."""
        # Remove common punctuation and extra whitespace
        import re

        title = re.sub(r"[^\w\s]", " ", title.lower())
        title = " ".join(title.split())
        return title

    def stream_large_dataset(
        self, file_path: Path, chunk_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """Stream large CSV files in chunks to avoid memory issues."""
        logger.info(f"Streaming dataset from {file_path} in chunks of {chunk_size}")

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming dataset: {e}")
            raise

    def _batch_iterator(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Create iterator over dataframe batches."""
        for i in range(0, len(df), self.batch_size):
            yield df.iloc[i : i + self.batch_size]

    def _estimate_batch_memory(self, batch_df: pd.DataFrame) -> float:
        """Estimate memory usage of batch in GB."""
        return batch_df.memory_usage(deep=True).sum() / (1024**3)

    def _check_memory_usage(self):
        """Monitor and log memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            self.metrics["memory_usage"].append(memory_gb)

            if memory_gb > self.memory_limit_gb * 0.9:
                logger.warning(f"High memory usage: {memory_gb:.1f}GB")

        except ImportError:
            pass  # psutil not available

    def _log_progress_summary(self, current_batch: int, total_batches: int):
        """Log comprehensive progress summary."""
        elapsed = time.time() - self.metrics["start_time"]
        progress_pct = (current_batch / total_batches) * 100

        avg_batch_time = (
            np.mean(self.metrics["batch_times"][-10:])
            if self.metrics["batch_times"]
            else 0
        )
        estimated_remaining = avg_batch_time * (total_batches - current_batch)

        logger.info(
            f"Progress: {progress_pct:.1f}% ({current_batch}/{total_batches} batches) | "
            f"Processed: {self.metrics['processed']:,} | "
            f"Failed: {self.metrics['failed']:,} | "
            f"Avg batch time: {avg_batch_time:.1f}s | "
            f"ETA: {estimated_remaining/60:.1f}min"
        )

    # Job tracking methods
    def _generate_job_id(self, job_type: str) -> str:
        """Generate unique job ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{job_type}_{timestamp}"

    def _create_job(self, job_id: str, job_type: str, total_items: int):
        """Create new job record."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        config_hash = hashlib.md5(str(self.config.__dict__).encode()).hexdigest()[:8]

        cursor.execute(
            """
            INSERT INTO batch_jobs
            (job_id, job_type, total_items, processed_items, failed_items, status, start_time, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                job_id,
                job_type,
                total_items,
                0,
                0,
                "running",
                datetime.now(),
                config_hash,
            ),
        )

        conn.commit()
        conn.close()

    def _log_batch_progress(
        self, job_id: str, batch_idx: int, batch_size: int, processing_time: float
    ):
        """Log progress of individual batch."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        memory_usage = (
            self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else 0
        )

        cursor.execute(
            """
            INSERT INTO batch_progress
            (job_id, batch_id, batch_size, processing_time, memory_usage, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                job_id,
                batch_idx,
                batch_size,
                processing_time,
                memory_usage,
                "completed",
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

    def _log_batch_error(self, job_id: str, batch_idx: int, error_message: str):
        """Log batch processing error."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO batch_progress
            (job_id, batch_id, status, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (job_id, batch_idx, "failed", error_message, datetime.now()),
        )

        conn.commit()
        conn.close()

    def _complete_job(self, job_id: str, final_count: int):
        """Mark job as completed."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE batch_jobs
            SET end_time = ?, status = ?, processed_items = ?
            WHERE job_id = ?
        """,
            (datetime.now(), "completed", final_count, job_id),
        )

        conn.commit()
        conn.close()

    def _fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE batch_jobs
            SET end_time = ?, status = ?
            WHERE job_id = ?
        """,
            (datetime.now(), f"failed: {error_message}", job_id),
        )

        conn.commit()
        conn.close()

    def _get_completed_batches(self, job_id: str) -> list[int]:
        """Get list of completed batch IDs for job."""
        conn = sqlite3.connect(str(self.progress_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT batch_id FROM batch_progress
            WHERE job_id = ? AND status = 'completed'
            ORDER BY batch_id
        """,
            (job_id,),
        )

        batch_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return batch_ids

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get comprehensive job status."""
        conn = sqlite3.connect(str(self.progress_db))

        # Get job info
        job_df = pd.read_sql_query(
            """
            SELECT * FROM batch_jobs WHERE job_id = ?
        """,
            conn,
            params=(job_id,),
        )

        if len(job_df) == 0:
            conn.close()
            return {"error": "Job not found"}

        # Get batch progress
        progress_df = pd.read_sql_query(
            """
            SELECT * FROM batch_progress WHERE job_id = ?
            ORDER BY batch_id
        """,
            conn,
            params=(job_id,),
        )

        conn.close()

        job_info = job_df.iloc[0].to_dict()

        # Calculate statistics
        completed_batches = len(progress_df[progress_df["status"] == "completed"])
        failed_batches = len(progress_df[progress_df["status"] == "failed"])

        if len(progress_df) > 0:
            avg_batch_time = progress_df["processing_time"].mean()
            total_processing_time = progress_df["processing_time"].sum()
        else:
            avg_batch_time = 0
            total_processing_time = 0

        return {
            "job_info": job_info,
            "completed_batches": completed_batches,
            "failed_batches": failed_batches,
            "avg_batch_time": avg_batch_time,
            "total_processing_time": total_processing_time,
            "progress_percentage": (
                completed_batches / max(1, job_info["total_items"] // self.batch_size)
            )
            * 100,
        }
