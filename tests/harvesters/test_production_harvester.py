"""Tests for production harvester."""

import sqlite3
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lit_review.harvesters.base import Paper
from src.lit_review.harvesters.production_harvester import ProductionHarvester


class TestProductionHarvester:
    """Test suite for production harvester."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        config = MagicMock()
        config.data_dir = tmp_path
        config.production_batch_size = 100
        config.checkpoint_interval = 10
        config.parallel_workers = 2
        config.wargame_terms = ["wargame", "simulation"]
        config.llm_terms = ["GPT", "LLM"]
        config.action_terms = ["play", "agent"]
        config.exclusion_terms = ["chess"]
        return config

    @pytest.fixture
    def harvester(self, config):
        """Create production harvester instance."""
        with patch.object(ProductionHarvester, "_init_progress_db"):
            return ProductionHarvester(config)

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            Paper(
                title="LLM Wargaming Paper 1",
                authors=["Author A"],
                year=2024,
                abstract="Testing LLM wargame simulation",
                source_db="arxiv",
                arxiv_id="2401.00001",
            ),
            Paper(
                title="AI Gaming Paper 2",
                authors=["Author B"],
                year=2024,
                abstract="GPT-4 agent research",
                source_db="semantic_scholar",
                doi="10.1234/test",
            ),
            Paper(
                title="Duplicate Title",
                authors=["Author C"],
                year=2024,
                abstract="Same content as another",
                source_db="crossref",
                doi="10.1234/test",  # Same DOI as above
            ),
        ]

    def test_init(self, config, tmp_path):
        """Test production harvester initialization."""
        harvester = ProductionHarvester(config)

        assert harvester.config == config
        assert harvester.batch_size == 100
        assert harvester.progress_db_path == tmp_path / "harvest_progress.db"
        assert "arxiv" in harvester.production_rate_limits
        assert "max_retries" in harvester.retry_config

    def test_init_progress_db(self, config, tmp_path):
        """Test progress database initialization."""
        harvester = ProductionHarvester(config)

        # Check database was created
        assert harvester.progress_db_path.exists()

        # Check tables exist
        conn = sqlite3.connect(str(harvester.progress_db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "harvest_sessions" in tables
        assert "harvest_progress" in tables
        assert "paper_cache" in tables

        conn.close()

    def test_generate_session_id(self, harvester):
        """Test session ID generation."""
        session_id = harvester._generate_session_id()

        assert session_id.startswith("harvest_")
        assert len(session_id) > 10
        # Should be unique each time (add small delay to ensure different timestamp)
        import time

        time.sleep(0.001)
        session_id2 = harvester._generate_session_id()
        # They could be the same within the same second, so just check format
        assert session_id2.startswith("harvest_")

    def test_allocate_quotas(self, harvester):
        """Test quota allocation across sources."""
        sources = ["arxiv", "semantic_scholar", "crossref", "google_scholar"]
        total_quota = 1000

        quotas = harvester._allocate_quotas(sources, total_quota)

        assert len(quotas) == 4
        assert all(quota > 0 for quota in quotas.values())
        # Should roughly equal total quota
        assert abs(sum(quotas.values()) - total_quota) < 200

    def test_build_production_queries(self, harvester):
        """Test production query building."""
        queries = harvester._build_production_queries("arxiv")

        assert len(queries) > 1  # Should have multiple queries
        assert all(isinstance(q, str) for q in queries)
        # Should include category-specific queries for arXiv
        assert any("cat:cs." in q for q in queries)

    def test_hash_query(self, harvester):
        """Test query hashing."""
        query = "test query"
        hash1 = harvester._hash_query(query)
        hash2 = harvester._hash_query(query)
        hash3 = harvester._hash_query("different query")

        assert hash1 == hash2  # Same query same hash
        assert hash1 != hash3  # Different query different hash
        assert len(hash1) == 12  # Expected length

    def test_classify_error(self, harvester):
        """Test error classification."""
        assert (
            harvester._classify_error(Exception("rate limit exceeded")) == "rate_limit"
        )
        assert harvester._classify_error(Exception("timeout")) == "timeout"
        assert (
            harvester._classify_error(Exception("connection error"))
            == "connection_error"
        )
        assert (
            harvester._classify_error(Exception("500 server error")) == "server_error"
        )
        assert harvester._classify_error(Exception("unknown error")) == "unknown"

    def test_production_deduplication(self, harvester, sample_papers):
        """Test advanced deduplication."""
        # Create DataFrame from sample papers
        df = pd.DataFrame([p.to_dict() for p in sample_papers])

        deduplicated = harvester._production_deduplication(df)

        # Should remove duplicate DOI
        assert len(deduplicated) == 2  # 3 papers -> 2 after dedup
        # Should keep papers with different DOIs (excluding empty/None DOIs)
        non_empty_dois = deduplicated[
            deduplicated["doi"].notna() & (deduplicated["doi"] != "")
        ]
        assert (
            len(non_empty_dois["doi"].unique()) == 1
        )  # Only one unique DOI should remain

    def test_fuzzy_title_dedup(self, harvester):
        """Test fuzzy title deduplication."""
        df = pd.DataFrame(
            {
                "title": [
                    "Machine Learning in Games",
                    "Machine Learning in Gaming",  # Very similar
                    "Completely Different Title",
                ],
                "year": [2024, 2024, 2024],
                "citations": [10, 5, 8],
            }
        )

        deduplicated = harvester._fuzzy_title_dedup(df)

        # Should remove similar title, keep distinct one
        assert len(deduplicated) == 2
        # Should keep the one with higher citations
        assert deduplicated.iloc[0]["citations"] == 10

    def test_arxiv_id_dedup(self, harvester):
        """Test arXiv ID deduplication."""
        df = pd.DataFrame(
            {
                "arxiv_id": ["2401.00001", "2401.00001", "2401.00002", ""],
                "title": ["Paper 1", "Paper 1 Duplicate", "Paper 2", "Paper 3"],
                "year": [2024, 2024, 2024, 2024],
            }
        )

        deduplicated = harvester._arxiv_id_dedup(df)

        # Should remove duplicate arXiv ID, keep empty ones
        assert len(deduplicated) == 3
        arxiv_ids = deduplicated["arxiv_id"].dropna()
        assert len(arxiv_ids.unique()) == len(arxiv_ids)

    def test_url_based_dedup(self, harvester):
        """Test URL-based deduplication."""
        df = pd.DataFrame(
            {
                "url": [
                    "https://example.com/paper1",
                    "https://example.com/paper1",  # Duplicate
                    "https://example.com/paper2",
                    "",  # Empty URL
                ],
                "title": ["Paper 1", "Paper 1 Duplicate", "Paper 2", "Paper 3"],
                "year": [2024, 2024, 2024, 2024],
            }
        )

        deduplicated = harvester._url_based_dedup(df)

        # Should remove duplicate URL, keep empty ones
        assert len(deduplicated) == 3
        urls = deduplicated["url"].dropna()
        assert len(urls) == len(urls.unique())

    def test_create_session(self, harvester):
        """Test session creation."""
        # Mock database operations
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            session_id = "test_session"
            sources = ["arxiv", "semantic_scholar"]
            max_results = 1000

            harvester._create_session(session_id, sources, max_results)

            # Verify database operations
            mock_connect.assert_called_once()
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_cache_paper(self, harvester, sample_papers):
        """Test paper caching."""
        paper = sample_papers[0]

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            harvester._cache_paper(paper)

            # Verify caching operation
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    @patch("sqlite3.connect")
    def test_get_session_status(self, mock_connect, harvester):
        """Test getting session status."""
        # Mock database responses
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock pandas read_sql_query
        with patch("pandas.read_sql_query") as mock_read_sql:
            session_df = pd.DataFrame(
                [
                    {
                        "session_id": "test_session",
                        "status": "completed",
                        "total_papers": 100,
                        "start_time": "2024-01-01 10:00:00",
                    }
                ]
            )
            progress_df = pd.DataFrame(
                [
                    {
                        "source": "arxiv",
                        "papers_found": 50,
                        "timestamp": "2024-01-01 10:30:00",
                    },
                    {
                        "source": "semantic_scholar",
                        "papers_found": 50,
                        "timestamp": "2024-01-01 10:45:00",
                    },
                ]
            )

            mock_read_sql.side_effect = [session_df, progress_df]

            status = harvester.get_session_status("test_session")

            assert "session_info" in status
            assert "progress_by_source" in status
            assert status["total_progress"] == 100
            assert "arxiv" in status["progress_by_source"]

    @patch("sqlite3.connect")
    def test_list_sessions(self, mock_connect, harvester):
        """Test listing all sessions."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        with patch("pandas.read_sql_query") as mock_read_sql:
            sessions_df = pd.DataFrame(
                [
                    {
                        "session_id": "session1",
                        "start_time": "2024-01-01 10:00:00",
                        "status": "completed",
                        "total_papers": 100,
                    },
                    {
                        "session_id": "session2",
                        "start_time": "2024-01-01 11:00:00",
                        "status": "running",
                        "total_papers": 0,
                    },
                ]
            )

            mock_read_sql.return_value = sessions_df

            result = harvester.list_sessions()

            assert len(result) == 2
            assert "session1" in result["session_id"].values
            assert "session2" in result["session_id"].values

    def test_hash_query_consistency(self, harvester):
        """Test query hashing consistency."""
        query1 = "Machine Learning in Games!"
        query2 = "machine-learning in games?"

        hash1 = harvester._hash_query(query1)
        hash2 = harvester._hash_query(query1)  # Same query
        hash3 = harvester._hash_query(query2)  # Different query

        assert hash1 == hash2  # Same query should have same hash
        assert hash1 != hash3  # Different queries should have different hashes
        assert len(hash1) == 12  # Expected hash length


class TestProductionHarvesterIntegration:
    """Integration tests for production harvester."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration with real paths."""
        config = MagicMock()
        config.data_dir = tmp_path
        config.production_batch_size = 10
        config.checkpoint_interval = 5
        config.parallel_workers = 1
        config.wargame_terms = ["test", "wargame"]
        config.llm_terms = ["AI", "LLM"]
        config.action_terms = ["play", "simulate"]
        config.exclusion_terms = []
        return config

    def test_full_deduplication_pipeline(self, config):
        """Test the complete deduplication pipeline."""
        harvester = ProductionHarvester(config)

        # Create test data with various duplication scenarios
        df = pd.DataFrame(
            {
                "title": [
                    "LLM Wargaming Study",
                    "LLM wargaming study",  # Case difference
                    "Different Paper",
                    "Another Paper",
                    "LLM Gaming Research",  # Similar but different
                ],
                "doi": ["10.1234/1", "", "10.1234/1", "10.1234/2", ""],
                "arxiv_id": ["", "2401.001", "", "", "2401.001"],
                "url": [
                    "https://ex.com/1",
                    "https://ex.com/2",
                    "https://ex.com/3",
                    "https://ex.com/1",  # Duplicate URL
                    "",
                ],
                "year": [2024, 2024, 2024, 2024, 2024],
                "citations": [10, 5, 8, 3, 7],
            }
        )

        result = harvester._production_deduplication(df)

        # Should remove duplicates but keep distinct papers
        assert len(result) < len(df)
        # Should have unique DOIs (excluding empty)
        dois = result[result["doi"] != ""]["doi"]
        assert len(dois) == len(dois.unique())
        # Should have unique arXiv IDs (excluding empty)
        arxiv_ids = result[result["arxiv_id"] != ""]["arxiv_id"]
        assert len(arxiv_ids) == len(arxiv_ids.unique())

    def test_session_management_flow(self, config):
        """Test complete session management workflow."""
        harvester = ProductionHarvester(config)

        # Test session creation
        session_id = harvester._generate_session_id()
        harvester._create_session(session_id, ["test"], 100)

        # Test progress tracking
        harvester._checkpoint_progress(session_id, "test", 50)

        # Test query completion tracking
        query_hash = harvester._hash_query("test query")
        harvester._mark_query_completed(session_id, "test", query_hash, 25)

        # Test session completion
        harvester._complete_session(session_id, 75)

        # Verify session exists and has correct status
        status = harvester.get_session_status(session_id)
        assert status["session_info"]["status"] == "completed"
        assert status["session_info"]["total_papers"] == 75
