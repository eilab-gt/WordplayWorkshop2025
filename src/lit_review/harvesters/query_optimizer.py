"""Dynamic query optimization for maximum paper discovery and relevance."""

import hashlib
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Intelligent query optimization system for maximizing paper discovery."""

    def __init__(self, config):
        """Initialize query optimizer with configuration."""
        self.config = config
        self.optimization_db = Path(config.data_dir) / "query_optimization.db"
        self._init_optimization_db()

        # Base search terms from config
        self.base_wargame_terms = getattr(config, "wargame_terms", [])
        self.base_llm_terms = getattr(config, "llm_terms", [])
        self.base_action_terms = getattr(config, "action_terms", [])
        self.exclusion_terms = getattr(config, "exclusion_terms", [])

        # Query performance tracking
        self.query_performance = {}

        # Expanded term sets for production
        self.expanded_terms = self._build_expanded_terms()

    def _init_optimization_db(self):
        """Initialize database for tracking query performance."""
        self.optimization_db.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.optimization_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_performance (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                source TEXT,
                papers_found INTEGER,
                execution_time REAL,
                success_rate REAL,
                relevance_score REAL,
                timestamp TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS term_effectiveness (
                term TEXT,
                term_type TEXT,
                source TEXT,
                papers_found INTEGER,
                relevance_score REAL,
                usage_count INTEGER,
                last_used TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def _build_expanded_terms(self) -> dict[str, list[str]]:
        """Build comprehensive expanded term sets for production coverage."""
        expanded = {
            "wargame_terms": self.base_wargame_terms
            + [
                # Strategic terms
                "war game",
                "wargaming",
                "tabletop exercise",
                "TTX",
                "crisis simulation",
                "crisis game",
                "policy simulation",
                "strategic planning exercise",
                "scenario planning",
                "red team",
                "blue team",
                "red teaming",
                # Military/Defense
                "military exercise",
                "defense simulation",
                "combat simulation",
                "operational planning",
                "strategic assessment",
                "threat assessment",
                "vulnerability assessment",
                # Gaming frameworks
                "serious game",
                "serious gaming",
                "simulation game",
                "role-playing exercise",
                "role playing game",
                "RPG",
                "decision game",
                "strategy game",
                # International relations
                "diplomatic simulation",
                "negotiation simulation",
                "Model UN",
                "Model United Nations",
                "MUN",
                "international relations simulation",
                # Specific games/systems
                "Diplomacy game",
                "Risk analysis",
                "scenario analysis",
                "contingency planning",
                "strategic communication exercise",
            ],
            "llm_terms": self.base_llm_terms
            + [
                # Model architectures
                "transformer",
                "attention mechanism",
                "neural language model",
                "autoregressive model",
                "seq2seq",
                "encoder-decoder",
                # Specific models
                "GPT-3",
                "GPT-3.5",
                "GPT-4",
                "ChatGPT",
                "InstructGPT",
                "Claude-2",
                "Claude-3",
                "PaLM-2",
                "Gemini",
                "Bard",
                "LLaMA-2",
                "Llama-2",
                "Alpaca",
                "Vicuna",
                "WizardLM",
                "T5",
                "UL2",
                "PaLM",
                "LaMDA",
                "Chinchilla",
                # Technical terms
                "foundation model",
                "pre-trained model",
                "fine-tuned model",
                "prompt engineering",
                "in-context learning",
                "few-shot learning",
                "zero-shot learning",
                "chain of thought",
                "reasoning",
                "natural language generation",
                "NLG",
                "text generation",
                "conversational AI",
                "dialogue system",
                "chatbot",
                # Capabilities
                "language understanding",
                "text comprehension",
                "natural language inference",
                "NLI",
                "question answering",
                "summarization",
                "text summarization",
            ],
            "action_terms": self.base_action_terms
            + [
                # AI roles
                "AI player",
                "AI agent",
                "artificial player",
                "automated player",
                "AI participant",
                "virtual participant",
                "digital participant",
                "AI facilitator",
                "AI moderator",
                "AI adjudicator",
                # Analysis roles
                "AI analyst",
                "AI advisor",
                "AI assistant",
                "AI consultant",
                "decision support",
                "analytical support",
                "strategic analysis",
                # Generation capabilities
                "scenario generation",
                "content generation",
                "text generation",
                "narrative generation",
                "story generation",
                "plot generation",
                "event generation",
                "outcome generation",
                # Interaction types
                "human-AI interaction",
                "human-machine interaction",
                "collaborative AI",
                "AI collaboration",
                "AI partnership",
                # Evaluation & Assessment
                "performance evaluation",
                "strategy evaluation",
                "decision evaluation",
                "outcome assessment",
                "effectiveness assessment",
                "impact assessment",
            ],
        }

        # Remove duplicates while preserving order
        for category in expanded:
            seen = set()
            unique_terms = []
            for term in expanded[category]:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append(term)
            expanded[category] = unique_terms

        logger.info(
            f"Expanded terms: {len(expanded['wargame_terms'])} wargame, "
            f"{len(expanded['llm_terms'])} LLM, {len(expanded['action_terms'])} action"
        )

        return expanded

    def generate_optimized_queries(
        self, source: str, max_queries: int = 10, include_experimental: bool = True
    ) -> list[str]:
        """Generate optimized queries for a specific source."""
        logger.info(f"Generating optimized queries for {source}")

        queries = []

        # Core comprehensive query
        core_query = self._build_core_query()
        queries.append(core_query)

        # Source-specific optimizations
        if source == "arxiv":
            queries.extend(self._arxiv_specific_queries())
        elif source == "semantic_scholar":
            queries.extend(self._semantic_scholar_queries())
        elif source == "crossref":
            queries.extend(self._crossref_queries())
        elif source == "google_scholar":
            queries.extend(self._google_scholar_queries())

        # Experimental queries for maximum coverage
        if include_experimental:
            queries.extend(self._experimental_queries(source))

        # Remove duplicates and limit
        unique_queries = self._deduplicate_queries(queries)

        # Select best queries based on historical performance
        selected_queries = self._select_best_queries(
            unique_queries, source, max_queries
        )

        logger.info(f"Generated {len(selected_queries)} optimized queries for {source}")
        return selected_queries

    def _build_core_query(self) -> str:
        """Build the most comprehensive core query."""
        # Use expanded terms for maximum coverage
        wargame_part = self._build_term_group(self.expanded_terms["wargame_terms"])
        llm_part = self._build_term_group(self.expanded_terms["llm_terms"])
        action_part = self._build_term_group(self.expanded_terms["action_terms"])

        # Build core query with OR logic for maximum recall
        core_query = f"({wargame_part}) AND ({llm_part})"

        # Add action terms as optional (increase recall)
        if action_part:
            core_query = f"({core_query}) OR (({wargame_part}) AND ({action_part}))"

        return core_query

    def _build_term_group(self, terms: list[str]) -> str:
        """Build optimized term group with proper escaping."""
        if not terms:
            return ""

        # Quote multi-word terms, leave single words unquoted for broader matching
        formatted_terms = []
        for term in terms:
            if " " in term or "-" in term:
                formatted_terms.append(f'"{term}"')
            else:
                formatted_terms.append(term)

        return " OR ".join(formatted_terms)

    def _arxiv_specific_queries(self) -> list[str]:
        """Generate arXiv-specific optimized queries."""
        queries = []

        # Category-focused queries
        categories = ["cs.AI", "cs.CL", "cs.LG", "cs.GT", "cs.MA", "cs.HC"]

        for category in categories:
            # Focused queries per category
            wargame_core = self._build_term_group(
                self.expanded_terms["wargame_terms"][:10]
            )
            llm_core = self._build_term_group(self.expanded_terms["llm_terms"][:15])

            query = f"({wargame_core}) AND ({llm_core}) AND cat:{category}"
            queries.append(query)

            # Broader queries per category (LLM + gaming concepts)
            broader_gaming = ["game", "simulation", "exercise", "scenario", "strategy"]
            gaming_part = " OR ".join([f'"{term}"' for term in broader_gaming])

            query = f"({llm_core}) AND ({gaming_part}) AND cat:{category}"
            queries.append(query)

        return queries

    def _semantic_scholar_queries(self) -> list[str]:
        """Generate Semantic Scholar optimized queries."""
        queries = []

        # Field-specific searches
        fields = [
            "Computer Science",
            "Political Science",
            "Economics",
            "Psychology",
            "Mathematics",
        ]

        for field in fields:
            # Core query with field filter
            core = self._build_core_query()
            queries.append(f"({core}) AND fieldsOfStudy:{field}")

            # Broader queries
            if field == "Computer Science":
                ai_terms = ["artificial intelligence", "machine learning", "NLP", "AI"]
                gaming_terms = ["game theory", "simulation", "multi-agent"]

                for ai_term in ai_terms:
                    for gaming_term in gaming_terms:
                        query = (
                            f'"{ai_term}" AND "{gaming_term}" AND fieldsOfStudy:{field}'
                        )
                        queries.append(query)

        return queries

    def _crossref_queries(self) -> list[str]:
        """Generate CrossRef optimized queries."""
        queries = []

        # Journal-focused searches
        target_journals = [
            "artificial intelligence",
            "machine learning",
            "natural language",
            "international security",
            "defense",
            "simulation",
            "games",
        ]

        core_terms = (
            self.expanded_terms["wargame_terms"][:8]
            + self.expanded_terms["llm_terms"][:12]
        )

        for journal_term in target_journals:
            for term in core_terms:
                query = f'"{term}" AND "{journal_term}"'
                queries.append(query)

        return queries

    def _google_scholar_queries(self) -> list[str]:
        """Generate Google Scholar optimized queries."""
        queries = []

        # High-precision queries for Google Scholar's broad search
        precision_combinations = [
            ("wargame", "GPT", "analysis"),
            ("war game", "language model", "simulation"),
            ("crisis simulation", "AI", "decision"),
            ("tabletop exercise", "artificial intelligence", "scenario"),
            ("policy game", "ChatGPT", "strategy"),
            ("diplomatic simulation", "LLM", "negotiation"),
            ("red team", "large language model", "exercise"),
            ("defense simulation", "GPT-4", "planning"),
        ]

        for combo in precision_combinations:
            query = " AND ".join([f'"{term}"' for term in combo])
            queries.append(query)

        # Author-based searches for known researchers
        known_authors = [
            "Philip Tetlock",
            "Phillip Tetlock",
            "Reid Hoffman",
            "Dario Amodei",
            "Anthropic",
            "OpenAI",
            "DeepMind",
        ]

        core_concepts = ["wargame", "simulation", "AI safety", "language model"]

        for author in known_authors:
            for concept in core_concepts:
                query = f'author:"{author}" "{concept}"'
                queries.append(query)

        return queries

    def _experimental_queries(self, source: str) -> list[str]:
        """Generate experimental queries for edge case discovery."""
        queries = []

        # Emerging terminology
        emerging_terms = [
            "prompt engineering",
            "in-context learning",
            "few-shot learning",
            "chain of thought",
            "constitutional AI",
            "RLHF",
            "alignment",
            "AI safety",
            "interpretability",
        ]

        wargame_variants = [
            "strategic exercise",
            "policy simulation",
            "crisis game",
            "red team exercise",
            "threat modeling",
            "scenario planning",
        ]

        # Cross-product of emerging terms with wargame variants
        for emerging in emerging_terms[:5]:  # Limit to avoid too many queries
            for wargame in wargame_variants[:3]:
                query = f'"{emerging}" AND "{wargame}"'
                queries.append(query)

        # Temporal queries (recent developments)
        recent_models = ["GPT-4", "Claude-3", "Gemini", "GPT-4o"]
        for model in recent_models:
            query = f'"{model}" AND ("wargame" OR "simulation" OR "exercise")'
            queries.append(query)

        return queries

    def _deduplicate_queries(self, queries: list[str]) -> list[str]:
        """Remove duplicate queries while preserving order."""
        seen = set()
        unique_queries = []

        for query in queries:
            # Normalize query for comparison
            normalized = re.sub(r"\s+", " ", query.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query)

        return unique_queries

    def _select_best_queries(
        self, queries: list[str], source: str, max_queries: int
    ) -> list[str]:
        """Select best queries based on historical performance."""
        if len(queries) <= max_queries:
            return queries

        # Score queries based on historical performance
        scored_queries = []

        for query in queries:
            score = self._score_query(query, source)
            scored_queries.append((score, query))

        # Sort by score (descending) and take top queries
        scored_queries.sort(reverse=True)

        # Always include the core query first
        selected = [queries[0]]  # First query is always core

        # Add top scoring queries
        for score, query in scored_queries[1:max_queries]:
            if query != queries[0]:  # Avoid duplicate core query
                selected.append(query)

        return selected[:max_queries]

    def _score_query(self, query: str, source: str) -> float:
        """Score query based on historical performance and complexity."""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Get historical performance
        conn = sqlite3.connect(str(self.optimization_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT AVG(papers_found), AVG(relevance_score), COUNT(*)
            FROM query_performance
            WHERE query_hash = ? AND source = ?
        """,
            (query_hash, source),
        )

        result = cursor.fetchone()
        conn.close()

        if result and result[0] is not None:
            avg_papers = result[0]
            avg_relevance = result[1] or 0.5
            usage_count = result[2]

            # Score based on papers found, relevance, and usage
            score = (
                (avg_papers * 0.6)
                + (avg_relevance * 0.3)
                + (min(usage_count, 10) * 0.1)
            )
        else:
            # No historical data, score based on query complexity and breadth
            score = self._estimate_query_potential(query)

        return score

    def _estimate_query_potential(self, query: str) -> float:
        """Estimate query potential based on structure and terms."""
        score = 0.5  # Base score

        # Count term types
        wargame_matches = sum(
            1
            for term in self.expanded_terms["wargame_terms"]
            if term.lower() in query.lower()
        )
        llm_matches = sum(
            1
            for term in self.expanded_terms["llm_terms"]
            if term.lower() in query.lower()
        )
        action_matches = sum(
            1
            for term in self.expanded_terms["action_terms"]
            if term.lower() in query.lower()
        )

        # Bonus for term diversity
        score += min(wargame_matches * 0.1, 0.3)
        score += min(llm_matches * 0.1, 0.3)
        score += min(action_matches * 0.1, 0.2)

        # Bonus for specific high-value terms
        high_value_terms = ["GPT-4", "Claude", "wargame", "simulation", "exercise"]
        for term in high_value_terms:
            if term.lower() in query.lower():
                score += 0.1

        return min(score, 1.0)

    def record_query_performance(
        self,
        query: str,
        source: str,
        papers_found: int,
        execution_time: float,
        relevance_score: float = 0.5,
    ):
        """Record query performance for future optimization."""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        conn = sqlite3.connect(str(self.optimization_db))
        cursor = conn.cursor()

        success_rate = 1.0 if papers_found > 0 else 0.0

        cursor.execute(
            """
            INSERT OR REPLACE INTO query_performance
            (query_hash, query_text, source, papers_found, execution_time,
             success_rate, relevance_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                query_hash,
                query[:500],
                source,
                papers_found,
                execution_time,
                success_rate,
                relevance_score,
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

    def get_optimization_stats(self) -> dict[str, any]:
        """Get query optimization statistics."""
        conn = sqlite3.connect(str(self.optimization_db))

        stats = {}

        # Overall performance
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT source, COUNT(*), AVG(papers_found), AVG(relevance_score)
            FROM query_performance
            GROUP BY source
        """
        )

        source_stats = {}
        for source, count, avg_papers, avg_relevance in cursor.fetchall():
            source_stats[source] = {
                "total_queries": count,
                "avg_papers_found": avg_papers or 0,
                "avg_relevance": avg_relevance or 0,
            }

        stats["by_source"] = source_stats

        # Top performing queries
        cursor.execute(
            """
            SELECT query_text, source, papers_found, relevance_score
            FROM query_performance
            ORDER BY papers_found DESC
            LIMIT 10
        """
        )

        top_queries = []
        for query, source, papers, relevance in cursor.fetchall():
            top_queries.append(
                {
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "source": source,
                    "papers_found": papers,
                    "relevance_score": relevance,
                }
            )

        stats["top_queries"] = top_queries

        conn.close()
        return stats
