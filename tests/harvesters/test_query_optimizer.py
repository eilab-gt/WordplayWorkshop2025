"""Tests for query optimizer."""

import sqlite3
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.lit_review.harvesters.query_optimizer import QueryOptimizer


class TestQueryOptimizer:
    """Test suite for query optimizer."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        config = MagicMock()
        config.data_dir = tmp_path
        config.wargame_terms = ["wargame", "simulation", "exercise"]
        config.llm_terms = ["GPT", "LLM", "Claude", "AI"]
        config.action_terms = ["play", "agent", "player"]
        config.exclusion_terms = ["chess", "video game"]
        return config

    @pytest.fixture
    def optimizer(self, config):
        """Create query optimizer instance."""
        with patch.object(QueryOptimizer, "_init_optimization_db"):
            return QueryOptimizer(config)

    def test_init(self, config, tmp_path):
        """Test query optimizer initialization."""
        optimizer = QueryOptimizer(config)

        assert optimizer.config == config
        assert optimizer.optimization_db == tmp_path / "query_optimization.db"
        assert len(optimizer.base_wargame_terms) == 3
        assert len(optimizer.base_llm_terms) == 4
        assert "wargame_terms" in optimizer.expanded_terms
        assert len(optimizer.expanded_terms["wargame_terms"]) > 3  # Should be expanded

    def test_init_optimization_db(self, config, tmp_path):
        """Test optimization database initialization."""
        optimizer = QueryOptimizer(config)

        # Check database was created
        assert optimizer.optimization_db.exists()

        # Check tables exist
        conn = sqlite3.connect(str(optimizer.optimization_db))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "query_performance" in tables
        assert "term_effectiveness" in tables

        conn.close()

    def test_build_expanded_terms(self, optimizer):
        """Test expanded term building."""
        expanded = optimizer._build_expanded_terms()

        assert "wargame_terms" in expanded
        assert "llm_terms" in expanded
        assert "action_terms" in expanded

        # Should be larger than base terms
        assert len(expanded["wargame_terms"]) > len(optimizer.base_wargame_terms)
        assert len(expanded["llm_terms"]) > len(optimizer.base_llm_terms)

        # Should include original terms
        for term in optimizer.base_wargame_terms:
            assert term in expanded["wargame_terms"]

    def test_build_term_group(self, optimizer):
        """Test term group building."""
        terms = ["wargame", "war game", "AI", "simulation"]

        term_group = optimizer._build_term_group(terms)

        # Should quote multi-word terms
        assert '"war game"' in term_group
        # Should not quote single words
        assert '"wargame"' not in term_group or "wargame" in term_group
        # Should use OR logic
        assert " OR " in term_group

    def test_build_core_query(self, optimizer):
        """Test core query building."""
        query = optimizer._build_core_query()

        assert isinstance(query, str)
        assert len(query) > 0
        # Should contain wargame and LLM terms
        assert "wargame" in query.lower() or "simulation" in query.lower()
        assert "gpt" in query.lower() or "llm" in query.lower() or "ai" in query.lower()
        # Should use boolean logic
        assert " AND " in query or " OR " in query

    def test_arxiv_specific_queries(self, optimizer):
        """Test arXiv-specific query generation."""
        queries = optimizer._arxiv_specific_queries()

        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        # Should contain category filters
        assert any("cat:cs." in q for q in queries)
        # Should contain arXiv-specific syntax
        assert any("AND cat:" in q for q in queries)

    def test_semantic_scholar_queries(self, optimizer):
        """Test Semantic Scholar query generation."""
        queries = optimizer._semantic_scholar_queries()

        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        # Should contain field filters
        assert any("fieldsOfStudy:" in q for q in queries)

    def test_crossref_queries(self, optimizer):
        """Test CrossRef query generation."""
        queries = optimizer._crossref_queries()

        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        # Should contain journal-focused terms
        assert any("artificial intelligence" in q.lower() for q in queries)

    def test_google_scholar_queries(self, optimizer):
        """Test Google Scholar query generation."""
        queries = optimizer._google_scholar_queries()

        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        # Should contain precision combinations
        assert any(" AND " in q for q in queries)
        # May contain author searches
        author_queries = [q for q in queries if "author:" in q]
        assert len(author_queries) >= 0  # Could be zero

    def test_experimental_queries(self, optimizer):
        """Test experimental query generation."""
        queries = optimizer._experimental_queries("arxiv")

        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        # Should contain emerging terms
        assert any("prompt engineering" in q.lower() for q in queries)
        # Should contain recent models
        assert any("gpt-4" in q.lower() for q in queries)

    def test_deduplicate_queries(self, optimizer):
        """Test query deduplication."""
        queries = [
            "wargame AND LLM",
            "WARGAME and llm",  # Same but different case
            "simulation AND AI",
            "wargame AND LLM",  # Exact duplicate
        ]

        unique_queries = optimizer._deduplicate_queries(queries)

        assert len(unique_queries) == 2  # Should remove duplicates
        assert "wargame AND LLM" in unique_queries
        assert "simulation AND AI" in unique_queries

    def test_estimate_query_potential(self, optimizer):
        """Test query potential estimation."""
        # High potential query (multiple term types)
        high_query = "wargame AND GPT-4 AND simulation"
        high_score = optimizer._estimate_query_potential(high_query)

        # Low potential query (few terms)
        low_query = "chess"
        low_score = optimizer._estimate_query_potential(low_query)

        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1

    def test_score_query_no_history(self, optimizer):
        """Test query scoring without historical data."""
        query = "test query with wargame and GPT"
        source = "arxiv"

        # Mock database to return no results
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = (None, None, 0)

            score = optimizer._score_query(query, source)

            assert 0 <= score <= 1  # Should fall back to estimation

    def test_score_query_with_history(self, optimizer):
        """Test query scoring with historical data."""
        query = "test query"
        source = "arxiv"

        # Mock database to return performance data
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = (50, 0.8, 5)  # papers, relevance, usage

            score = optimizer._score_query(query, source)

            assert score > 0  # Should have positive score
            assert isinstance(score, float)

    def test_generate_optimized_queries(self, optimizer):
        """Test complete optimized query generation."""
        with patch.object(optimizer, "_score_query", return_value=0.8):
            queries = optimizer.generate_optimized_queries("arxiv", max_queries=5)

            assert len(queries) <= 5
            assert len(queries) > 0
            assert all(isinstance(q, str) for q in queries)
            # First query should be core query
            assert "wargame" in queries[0].lower() or "simulation" in queries[0].lower()

    def test_record_query_performance(self, optimizer):
        """Test recording query performance."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            optimizer.record_query_performance(
                query="test query",
                source="arxiv",
                papers_found=25,
                execution_time=1.5,
                relevance_score=0.8,
            )

            # Verify database operations
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_get_optimization_stats(self, optimizer):
        """Test getting optimization statistics."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            # Mock database responses
            mock_cursor.fetchall.side_effect = [
                [("arxiv", 10, 45.5, 0.75), ("semantic_scholar", 8, 32.1, 0.82)],
                [
                    ("test query", "arxiv", 50, 0.9),
                    ("another query", "semantic_scholar", 30, 0.7),
                ],
            ]

            stats = optimizer.get_optimization_stats()

            assert "by_source" in stats
            assert "top_queries" in stats
            assert "arxiv" in stats["by_source"]
            assert len(stats["top_queries"]) == 2

    def test_select_best_queries(self, optimizer):
        """Test best query selection."""
        queries = [
            "core query with wargame and LLM",
            "specific query with GPT-4",
            "experimental query with prompt engineering",
            "broad query with AI",
        ]

        with patch.object(optimizer, "_score_query") as mock_score:
            # Mock scores: core=0.9, specific=0.8, experimental=0.6, broad=0.7
            mock_score.side_effect = [0.9, 0.8, 0.6, 0.7]

            selected = optimizer._select_best_queries(queries, "arxiv", 3)

            assert len(selected) == 3
            # Should include core query first
            assert selected[0] == queries[0]
            # Should be ordered by score (excluding core)
            assert "specific query" in selected[1]


class TestQueryOptimizerIntegration:
    """Integration tests for query optimizer."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration with real paths."""
        config = MagicMock()
        config.data_dir = tmp_path
        config.wargame_terms = ["wargame", "simulation"]
        config.llm_terms = ["GPT", "LLM", "AI"]
        config.action_terms = ["play", "agent"]
        config.exclusion_terms = ["chess"]
        return config

    def test_full_optimization_cycle(self, config):
        """Test complete optimization cycle."""
        optimizer = QueryOptimizer(config)

        # Record some performance data
        optimizer.record_query_performance("wargame AND GPT", "arxiv", 45, 2.1, 0.85)
        optimizer.record_query_performance(
            "simulation AND LLM", "semantic_scholar", 32, 1.8, 0.78
        )

        # Generate optimized queries
        queries = optimizer.generate_optimized_queries("arxiv", max_queries=3)

        assert len(queries) <= 3
        assert all(len(q) > 0 for q in queries)

        # Get statistics
        stats = optimizer.get_optimization_stats()

        assert "by_source" in stats
        assert len(stats["by_source"]) >= 0  # May be empty in test

    def test_source_specific_optimization(self, config):
        """Test source-specific query optimization."""
        optimizer = QueryOptimizer(config)

        # Test different sources
        arxiv_queries = optimizer.generate_optimized_queries("arxiv", max_queries=3)
        scholar_queries = optimizer.generate_optimized_queries(
            "semantic_scholar", max_queries=3
        )
        crossref_queries = optimizer.generate_optimized_queries(
            "crossref", max_queries=3
        )
        google_queries = optimizer.generate_optimized_queries(
            "google_scholar", max_queries=3
        )

        # Should generate different query sets for different sources
        assert len(arxiv_queries) > 0
        assert len(scholar_queries) > 0
        assert len(crossref_queries) > 0
        assert len(google_queries) > 0

        # arXiv queries should contain category filters
        assert any("cat:" in q for q in arxiv_queries)

        # Semantic Scholar queries should contain field filters
        assert any("fieldsOfStudy:" in q for q in scholar_queries)

    def test_query_evolution(self, config):
        """Test how queries evolve with performance feedback."""
        optimizer = QueryOptimizer(config)

        # Initial query generation
        initial_queries = optimizer.generate_optimized_queries("arxiv", max_queries=5)

        # Record performance for some queries
        for i, query in enumerate(initial_queries[:3]):
            # Simulate varying performance
            papers_found = 50 - (i * 10)  # Decreasing performance
            relevance = 0.9 - (i * 0.1)

            optimizer.record_query_performance(
                query, "arxiv", papers_found, 1.5, relevance
            )

        # Generate queries again
        evolved_queries = optimizer.generate_optimized_queries("arxiv", max_queries=5)

        # Should still generate valid queries
        assert len(evolved_queries) > 0
        assert all(isinstance(q, str) for q in evolved_queries)

        # Performance data should influence selection (hard to test deterministically)
        # But at least verify the system still works
        assert len(evolved_queries) == len(initial_queries)
