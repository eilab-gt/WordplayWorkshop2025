"""Production-optimized harvester with aggressive scaling and robustness."""

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Paper
from .search_harvester import SearchHarvester

logger = logging.getLogger(__name__)


class ProductionHarvester(SearchHarvester):
    """Production-scale harvester optimized for maximum throughput and reliability."""

    def __init__(self, config):
        """Initialize production harvester with enhanced capabilities."""
        super().__init__(config)

        # Production settings
        self.progress_db_path = Path(config.data_dir) / "harvest_progress.db"
        self.resume_enabled = True
        self.batch_size = getattr(config, "production_batch_size", 1000)
        self.checkpoint_interval = getattr(config, "checkpoint_interval", 100)

        # Aggressive rate limits for production
        self.production_rate_limits = {
            "arxiv": {
                "requests_per_second": 10,
                "delay_milliseconds": 100,
            },  # 10x increase
            "semantic_scholar": {
                "requests_per_second": 50,
                "delay_milliseconds": 20,
            },  # 5x increase
            "crossref": {
                "requests_per_second": 100,
                "delay_milliseconds": 10,
            },  # 2x increase
            "google_scholar": {
                "requests_per_hour": 500,
                "delay_seconds": 7.2,
            },  # 5x increase
        }

        # Retry configuration
        self.retry_config = {
            "max_retries": 5,
            "backoff_factor": 2.0,
            "max_backoff": 300,  # 5 minutes max
            "retryable_errors": [
                "rate_limit",
                "timeout",
                "connection_error",
                "server_error",
            ],
        }

        # Initialize progress tracking
        self._init_progress_db()

        # Deduplication tracking
        self.seen_papers: set[str] = set()
        self.paper_hashes: dict[str, str] = {}

    def _init_progress_db(self):
        """Initialize SQLite database for tracking harvest progress."""
        self.progress_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS harvest_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                sources TEXT,
                total_papers INTEGER,
                config_hash TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS harvest_progress (
                session_id TEXT,
                source TEXT,
                query_hash TEXT,
                page INTEGER,
                papers_found INTEGER,
                timestamp TIMESTAMP,
                status TEXT,
                error_message TEXT,
                FOREIGN KEY (session_id) REFERENCES harvest_sessions (session_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_cache (
                paper_hash TEXT PRIMARY KEY,
                source TEXT,
                paper_data TEXT,
                timestamp TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def search_production_scale(
        self,
        sources: list[str] | None = None,
        max_results_total: int = 50000,  # Production scale
        resume_session: str | None = None,
        checkpoint_callback: callable | None = None,
    ) -> pd.DataFrame:
        """Execute production-scale search with checkpointing and resume capability.

        Args:
            sources: List of sources to search (None = all sources)
            max_results_total: Total maximum results across all sources
            resume_session: Session ID to resume from
            checkpoint_callback: Function called at each checkpoint

        Returns:
            DataFrame of combined results
        """
        session_id = resume_session or self._generate_session_id()

        logger.info(f"Starting production harvest session: {session_id}")
        logger.info(
            f"Target: {max_results_total:,} papers across {sources or 'all'} sources"
        )

        # Initialize or resume session
        if resume_session:
            logger.info(f"Resuming session {session_id}")
            session_data = self._load_session(session_id)
            if not session_data:
                logger.error(f"Session {session_id} not found, starting new session")
                session_id = self._generate_session_id()
                self._create_session(session_id, sources, max_results_total)
        else:
            self._create_session(session_id, sources, max_results_total)

        # Determine sources and allocate quotas
        if sources is None:
            sources = list(self.harvesters.keys())

        # Dynamic quota allocation based on source performance
        source_quotas = self._allocate_quotas(sources, max_results_total)

        all_papers = []

        # Execute harvesting with checkpointing
        try:
            for source in sources:
                source_quota = source_quotas[source]
                logger.info(f"Harvesting {source}: quota {source_quota:,} papers")

                source_papers = self._harvest_source_production(
                    source, source_quota, session_id, checkpoint_callback
                )

                all_papers.extend(source_papers)

                # Checkpoint after each source
                self._checkpoint_progress(session_id, source, len(source_papers))

                if checkpoint_callback:
                    checkpoint_callback(
                        session_id, source, len(source_papers), len(all_papers)
                    )

                logger.info(f"Completed {source}: {len(source_papers):,} papers")

            # Final processing
            df = self._papers_to_dataframe(all_papers)
            df = self._production_deduplication(df)

            # Mark session complete
            self._complete_session(session_id, len(df))

            logger.info(f"Production harvest complete: {len(df):,} unique papers")
            return df

        except Exception as e:
            logger.error(f"Production harvest failed: {e}")
            self._mark_session_failed(session_id, str(e))
            raise

    def _allocate_quotas(self, sources: list[str], total_quota: int) -> dict[str, int]:
        """Dynamically allocate quotas based on source performance and capacity."""
        # Source performance weights (based on historical yield and reliability)
        performance_weights = {
            "semantic_scholar": 0.35,  # High yield, good API
            "arxiv": 0.25,  # Focused domain, reliable
            "crossref": 0.25,  # Large corpus, good metadata
            "google_scholar": 0.15,  # High yield but rate limited
        }

        quotas = {}
        for source in sources:
            weight = performance_weights.get(source, 0.2)  # Default weight
            quotas[source] = int(total_quota * weight)

        # Ensure minimum quotas
        min_quota = max(100, total_quota // (len(sources) * 10))
        for source in quotas:
            quotas[source] = max(quotas[source], min_quota)

        # Redistribute if over total
        current_total = sum(quotas.values())
        if current_total > total_quota:
            scale_factor = total_quota / current_total
            for source in quotas:
                quotas[source] = int(quotas[source] * scale_factor)

        logger.info(f"Quota allocation: {quotas}")
        return quotas

    def _harvest_source_production(
        self,
        source: str,
        quota: int,
        session_id: str,
        checkpoint_callback: callable | None = None,
    ) -> list[Paper]:
        """Harvest from single source with production optimizations."""
        harvester = self.harvesters[source]

        # Override rate limits for production
        if hasattr(harvester, "rate_limits"):
            production_limits = self.production_rate_limits.get(source, {})
            harvester.rate_limits.update(production_limits)
            if hasattr(harvester, "delay_milliseconds"):
                harvester.delay_milliseconds = production_limits.get(
                    "delay_milliseconds", 100
                )

        papers = []
        page_size = min(500, quota)  # Larger page sizes for efficiency

        # Build optimized queries for production
        queries = self._build_production_queries(source)

        for query_idx, query in enumerate(queries):
            if len(papers) >= quota:
                break

            logger.info(
                f"{source}: Query {query_idx + 1}/{len(queries)}: {query[:100]}..."
            )

            # Check if this query was already processed
            query_hash = self._hash_query(query)
            if self._is_query_completed(session_id, source, query_hash):
                logger.info(f"{source}: Skipping completed query {query_hash}")
                continue

            remaining_quota = quota - len(papers)
            query_papers = self._harvest_query_with_retry(
                harvester, query, min(page_size, remaining_quota), session_id, source
            )

            papers.extend(query_papers)

            # Mark query as completed
            self._mark_query_completed(
                session_id, source, query_hash, len(query_papers)
            )

            logger.info(
                f"{source}: Query yielded {len(query_papers)} papers, total: {len(papers)}"
            )

        return papers

    def _build_production_queries(self, source: str) -> list[str]:
        """Build optimized query set for maximum coverage."""
        base_query = self._build_combined_query()

        # For production, create multiple query variants for better coverage
        queries = [base_query]  # Start with base query

        if source == "arxiv":
            # Add category-specific queries for arXiv
            categories = ["cs.AI", "cs.CL", "cs.LG", "cs.GT", "cs.MA"]
            for category in categories:
                category_query = f"({base_query}) AND cat:{category}"
                queries.append(category_query)

        elif source == "semantic_scholar":
            # Add field-specific queries for Semantic Scholar
            fields = ["Computer Science", "Political Science", "Economics"]
            for field in fields:
                field_query = f"{base_query} AND field:{field}"
                queries.append(field_query)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        logger.info(f"{source}: Generated {len(unique_queries)} optimized queries")
        return unique_queries

    def _harvest_query_with_retry(
        self, harvester, query: str, max_results: int, session_id: str, source: str
    ) -> list[Paper]:
        """Execute query with exponential backoff retry logic."""
        for attempt in range(self.retry_config["max_retries"] + 1):
            try:
                papers = harvester.search(query, max_results)

                # Cache successful results
                for paper in papers:
                    self._cache_paper(paper)

                return papers

            except Exception as e:
                error_type = self._classify_error(e)

                if error_type not in self.retry_config["retryable_errors"]:
                    logger.error(f"{source}: Non-retryable error: {e}")
                    return []

                if attempt < self.retry_config["max_retries"]:
                    delay = min(
                        self.retry_config["backoff_factor"] ** attempt,
                        self.retry_config["max_backoff"],
                    )
                    logger.warning(
                        f"{source}: Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{source}: Max retries exceeded: {e}")
                    return []

        return []

    def _production_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced deduplication for production scale."""
        logger.info(f"Starting production deduplication: {len(df)} papers")

        # Multi-stage deduplication
        original_count = len(df)

        # Stage 1: Exact DOI matching
        df_with_doi = df[df["doi"].notna() & (df["doi"] != "")]
        df_without_doi = df[df["doi"].isna() | (df["doi"] == "")]

        df_with_doi = df_with_doi.drop_duplicates(subset=["doi"], keep="first")

        # Stage 2: Fuzzy title matching for papers without DOI
        df_without_doi = self._fuzzy_title_dedup(df_without_doi)

        # Stage 3: Cross-source arXiv ID matching
        df_combined = pd.concat([df_with_doi, df_without_doi], ignore_index=True)
        df_combined = self._arxiv_id_dedup(df_combined)

        # Stage 4: URL-based deduplication
        df_final = self._url_based_dedup(df_combined)

        logger.info(
            f"Deduplication complete: {original_count} → {len(df_final)} papers ({len(df_final)/original_count:.1%} retained)"
        )

        return df_final

    def _fuzzy_title_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove papers with very similar titles."""
        if len(df) == 0:
            return df

        try:
            from difflib import SequenceMatcher

            # Sort by year desc to keep newer papers
            df = df.sort_values("year", ascending=False)

            keep_indices = []
            seen_titles = []

            for idx, row in df.iterrows():
                title = str(row.get("title", "")).lower().strip()
                if not title:
                    continue

                is_duplicate = False
                for seen_title in seen_titles:
                    similarity = SequenceMatcher(None, title, seen_title).ratio()
                    if similarity > 0.9:  # 90% similarity threshold
                        is_duplicate = True
                        break

                if not is_duplicate:
                    keep_indices.append(idx)
                    seen_titles.append(title)

            return df.loc[keep_indices]

        except ImportError:
            logger.warning("difflib not available, skipping fuzzy title deduplication")
            return df

    def _arxiv_id_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate based on arXiv IDs."""
        df_with_arxiv = df[df["arxiv_id"].notna() & (df["arxiv_id"] != "")]
        df_without_arxiv = df[df["arxiv_id"].isna() | (df["arxiv_id"] == "")]

        if len(df_with_arxiv) > 0:
            df_with_arxiv = df_with_arxiv.drop_duplicates(
                subset=["arxiv_id"], keep="first"
            )

        return pd.concat([df_with_arxiv, df_without_arxiv], ignore_index=True)

    def _url_based_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final deduplication based on URLs."""
        df_with_url = df[df["url"].notna() & (df["url"] != "")]
        df_without_url = df[df["url"].isna() | (df["url"] == "")]

        if len(df_with_url) > 0:
            df_with_url = df_with_url.drop_duplicates(subset=["url"], keep="first")

        return pd.concat([df_with_url, df_without_url], ignore_index=True)

    # Progress tracking and session management methods
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"harvest_{timestamp}"

    def _create_session(self, session_id: str, sources: list[str], max_results: int):
        """Create new harvest session."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        config_hash = hashlib.md5(str(self.config.__dict__).encode()).hexdigest()[:8]

        cursor.execute(
            """
            INSERT INTO harvest_sessions
            (session_id, start_time, status, sources, total_papers, config_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                datetime.now(),
                "running",
                json.dumps(sources),
                max_results,
                config_hash,
            ),
        )

        conn.commit()
        conn.close()

    def _checkpoint_progress(self, session_id: str, source: str, papers_found: int):
        """Save checkpoint progress."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO harvest_progress
            (session_id, source, papers_found, timestamp, status)
            VALUES (?, ?, ?, ?, ?)
        """,
            (session_id, source, papers_found, datetime.now(), "completed"),
        )

        conn.commit()
        conn.close()

    def _complete_session(self, session_id: str, total_papers: int):
        """Mark session as completed."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE harvest_sessions
            SET end_time = ?, status = ?, total_papers = ?
            WHERE session_id = ?
        """,
            (datetime.now(), "completed", total_papers, session_id),
        )

        conn.commit()
        conn.close()

    def _hash_query(self, query: str) -> str:
        """Generate hash for query deduplication."""
        return hashlib.md5(query.encode()).hexdigest()[:12]

    def _is_query_completed(
        self, session_id: str, source: str, query_hash: str
    ) -> bool:
        """Check if query was already completed."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM harvest_progress
            WHERE session_id = ? AND source = ? AND query_hash = ? AND status = 'completed'
        """,
            (session_id, source, query_hash),
        )

        result = cursor.fetchone()[0] > 0
        conn.close()
        return result

    def _mark_query_completed(
        self, session_id: str, source: str, query_hash: str, papers_found: int
    ):
        """Mark query as completed."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO harvest_progress
            (session_id, source, query_hash, papers_found, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (session_id, source, query_hash, papers_found, datetime.now(), "completed"),
        )

        conn.commit()
        conn.close()

    def _cache_paper(self, paper: Paper):
        """Cache paper to avoid re-processing."""
        paper_hash = hashlib.md5(f"{paper.title}{paper.source_db}".encode()).hexdigest()

        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO paper_cache
            (paper_hash, source, paper_data, timestamp)
            VALUES (?, ?, ?, ?)
        """,
            (paper_hash, paper.source_db, json.dumps(paper.to_dict()), datetime.now()),
        )

        conn.commit()
        conn.close()

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for retry logic."""
        error_str = str(error).lower()

        if any(
            term in error_str for term in ["rate limit", "too many requests", "429"]
        ):
            return "rate_limit"
        elif any(term in error_str for term in ["timeout", "timed out"]):
            return "timeout"
        elif any(term in error_str for term in ["connection", "network"]):
            return "connection_error"
        elif any(term in error_str for term in ["500", "502", "503", "504"]):
            return "server_error"
        else:
            return "unknown"

    def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get detailed status of harvest session."""
        conn = sqlite3.connect(str(self.progress_db_path))

        # Get session info
        session_df = pd.read_sql_query(
            """
            SELECT * FROM harvest_sessions WHERE session_id = ?
        """,
            conn,
            params=(session_id,),
        )

        if len(session_df) == 0:
            conn.close()
            return {"error": "Session not found"}

        # Get progress info
        progress_df = pd.read_sql_query(
            """
            SELECT * FROM harvest_progress WHERE session_id = ?
        """,
            conn,
            params=(session_id,),
        )

        conn.close()

        session_info = session_df.iloc[0].to_dict()
        progress_by_source = (
            progress_df.groupby("source")["papers_found"].sum().to_dict()
        )

        return {
            "session_info": session_info,
            "progress_by_source": progress_by_source,
            "total_progress": progress_df["papers_found"].sum(),
            "last_activity": (
                progress_df["timestamp"].max() if len(progress_df) > 0 else None
            ),
        }

    def list_sessions(self) -> pd.DataFrame:
        """List all harvest sessions."""
        conn = sqlite3.connect(str(self.progress_db_path))

        df = pd.read_sql_query(
            """
            SELECT session_id, start_time, end_time, status, total_papers
            FROM harvest_sessions
            ORDER BY start_time DESC
        """,
            conn,
        )

        conn.close()
        return df
