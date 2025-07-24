"""Test doubles that behave like real components (not mocks)."""

import random
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

# Import realistic test data generator
try:
    from tests.test_data_generators import RealisticTestDataGenerator
except ImportError:
    from test_data_generators import RealisticTestDataGenerator


class FakeLLMService:
    """Fake LLM service that behaves like real service without network calls."""

    def __init__(self, port: int = 8000, healthy: bool = True):
        self.port = port
        self.healthy = healthy
        self.available_models = {
            "gemini/gemini-pro": {"available": True, "context_window": 32000},
            "gpt-3.5-turbo": {"available": True, "context_window": 4096},
            "claude-3-haiku-20240307": {"available": False},
        }
        self.call_history: list[dict[str, Any]] = []

    def check_health(self) -> bool:
        """Simulate health check."""
        return self.healthy

    def get_models(self) -> dict[str, Any]:
        """Return available models."""
        return self.available_models

    def extract(self, text: str, model: str, **kwargs) -> dict[str, Any]:
        """Simulate extraction with realistic responses."""
        self.call_history.append(
            {
                "method": "extract",
                "text_length": len(text),
                "model": model,
                "kwargs": kwargs,
            }
        )

        if not self.available_models.get(model, {}).get("available"):
            return {"success": False, "error": f"Model {model} not available"}

        # Generate realistic extraction based on text content
        extracted = {
            "research_questions": self._extract_research_questions(text),
            "key_contributions": self._extract_contributions(text),
            "simulation_approach": self._extract_simulation(text),
            "llm_usage": self._extract_llm_usage(text),
            "human_llm_comparison": "Humans performed better in strategic thinking",
            "evaluation_metrics": [
                "accuracy",
                "decision quality",
                "time to completion",
            ],
            "prompting_strategies": "Few-shot prompting with role instructions",
            "emerging_behaviors": "LLMs showed unexpected coalition formation",
            "datasets_used": ["Custom wargame scenarios"],
            "limitations": "Limited to text-based interactions",
        }

        return {
            "success": True,
            "extracted_data": extracted,
            "model_used": model,
            "tokens_used": len(text) // 4,  # Rough approximation
        }

    def _extract_research_questions(self, text: str) -> str:
        """Extract research questions based on keywords."""
        if "wargam" in text.lower():
            return "How can LLMs enhance strategic wargaming simulations?"
        elif "llm" in text.lower():
            return "What are the capabilities of LLMs in complex reasoning?"
        return "General AI research questions"

    def _extract_contributions(self, text: str) -> str:
        """Extract key contributions."""
        contributions = []
        if "novel" in text.lower() or "new" in text.lower():
            contributions.append("Novel framework for LLM integration")
        if "evaluat" in text.lower():
            contributions.append("Comprehensive evaluation methodology")
        if "dataset" in text.lower():
            contributions.append("New benchmark dataset")
        return "; ".join(contributions) if contributions else "Incremental improvements"

    def _extract_simulation(self, text: str) -> str:
        """Extract simulation approach."""
        if "matrix" in text.lower() or "tabletop" in text.lower():
            return "Matrix game with human facilitators"
        elif "automated" in text.lower() or "digital" in text.lower():
            return "Fully automated digital simulation"
        return "Hybrid human-AI simulation"

    def _extract_llm_usage(self, text: str) -> str:
        """Extract LLM usage patterns."""
        if "agent" in text.lower():
            return "LLMs as autonomous agents"
        elif "assist" in text.lower():
            return "LLMs as decision support"
        return "LLMs for analysis and generation"


class FakeArxivAPI:
    """Fake arXiv API that returns predictable results."""

    def __init__(self, seed: int | None = 42):
        self.generator = RealisticTestDataGenerator(seed=seed)
        self.papers = []
        self.call_count = 0
        # Pre-generate some papers
        self._initialize_papers()

    def _initialize_papers(self):
        """Initialize with realistic papers."""
        # Generate 50 papers across different years
        raw_papers = self.generator.generate_paper_batch(50, year_range=(2020, 2024))

        # Convert to arxiv format
        for paper in raw_papers:
            if "arxiv_id" in paper:
                arxiv_paper = {
                    "id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": [
                        author.split(" (")[0] for author in paper["authors"]
                    ],  # Remove affiliations
                    "abstract": paper["abstract"],
                    "pdf_url": paper.get(
                        "pdf_url", f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
                    ),
                    "published": paper["published"],
                    "categories": paper["categories"],
                }
                self.papers.append(arxiv_paper)

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Return fake search results."""
        self.call_count += 1

        # Filter based on query terms
        query_lower = query.lower()
        query_terms = query_lower.split()

        results = []
        for paper in self.papers:
            # Check if any query term appears in title or abstract
            paper_text = (paper["title"] + " " + paper["abstract"]).lower()

            # Score based on term matches
            score = sum(1 for term in query_terms if term in paper_text)

            # Include papers with at least one matching term
            if score > 0:
                results.append((score, paper))

        # Sort by score (descending) and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [paper for _, paper in results[:max_results]]

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get specific paper by ID."""
        for paper in self.papers:
            if paper["id"] == arxiv_id:
                return paper
        return None

    def add_paper(self, paper_dict: dict[str, Any]):
        """Add a custom paper for testing."""
        self.papers.append(paper_dict)

    def get_tex_source(self, arxiv_id: str) -> str | None:
        """Return fake TeX source."""
        paper = self.get_paper(arxiv_id)
        if paper:
            return f"""\\documentclass{{article}}
\\title{{{paper['title']}}}
\\author{{{' and '.join(paper['authors'])}}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{paper['abstract']}
\\end{{abstract}}
\\section{{Introduction}}
This paper presents research on LLMs in strategic contexts...

\\section{{Related Work}}
Previous work has explored various aspects of AI in gaming...

\\section{{Methodology}}
We employ a novel approach combining LLMs with game theory...

\\section{{Results}}
Our experiments demonstrate significant improvements...

\\section{{Conclusion}}
This work contributes to the growing field of AI-assisted decision making...

\\bibliographystyle{{plain}}
\\bibliography{{references}}
\\end{{document}}"""
        return None


class FakePDFServer:
    """Fake PDF server that serves test PDFs."""

    def __init__(self):
        self.available_pdfs = {}
        self.download_count = {}
        self.rate_limit_enabled = False
        self.rate_limit_threshold = 5
        self.failure_rate = 0.0  # Probability of random failure

    def serve_pdf(self, arxiv_id: str) -> tuple[bytes, int]:
        """Serve a fake PDF for the given arXiv ID."""
        # Track download count for rate limiting
        self.download_count[arxiv_id] = self.download_count.get(arxiv_id, 0) + 1

        # Simulate rate limiting
        if (
            self.rate_limit_enabled
            and self.download_count[arxiv_id] > self.rate_limit_threshold
        ):
            return b"Rate limit exceeded", 429

        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            return b"Server error", 500

        # Generate or return cached PDF content
        if arxiv_id not in self.available_pdfs:
            # Generate fake PDF content
            pdf_content = self._generate_fake_pdf(arxiv_id)
            self.available_pdfs[arxiv_id] = pdf_content

        return self.available_pdfs[arxiv_id], 200

    def _generate_fake_pdf(self, arxiv_id: str) -> bytes:
        """Generate fake PDF content."""
        # Create a simple text that simulates PDF content
        content = f"""
        %PDF-1.4
        This is a fake PDF for paper {arxiv_id}

        Title: Research on LLMs in Wargaming Contexts

        Abstract:
        This paper explores the application of large language models
        in strategic wargaming scenarios. We present novel approaches
        to integrating LLMs with traditional game theory frameworks.

        1. Introduction
        The intersection of artificial intelligence and strategic gaming
        presents unique opportunities for enhancing decision-making...

        2. Methodology
        We employ a multi-agent framework where LLMs act as autonomous
        agents in complex strategic scenarios...

        3. Results
        Our experiments demonstrate that LLM-augmented teams achieve
        superior performance in various metrics...

        4. Conclusion
        This work contributes to the growing field of AI-assisted
        strategic planning and decision support systems.

        References:
        [1] Smith et al., "LLMs in Gaming", 2023
        [2] Jones et al., "Strategic AI", 2024

        %%EOF
        """
        return content.encode("utf-8")

    def add_pdf(self, arxiv_id: str, content: bytes):
        """Add a custom PDF for testing."""
        self.available_pdfs[arxiv_id] = content

    def reset_rate_limits(self):
        """Reset rate limit counters."""
        self.download_count.clear()


class FakeDatabase:
    """In-memory SQLite database for testing."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Papers table
        cursor.execute(
            """
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                doi TEXT UNIQUE,
                arxiv_id TEXT UNIQUE,
                abstract TEXT,
                pdf_path TEXT,
                extraction_status TEXT
            )
        """
        )

        # Metrics table
        cursor.execute(
            """
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY,
                paper_id INTEGER,
                metric_type TEXT,
                value REAL,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """
        )

        self.conn.commit()

    def insert_paper(self, paper: dict[str, Any]) -> int:
        """Insert paper and return ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO papers (title, authors, year, doi, arxiv_id, abstract)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                paper.get("title"),
                paper.get("authors"),
                paper.get("year"),
                paper.get("doi"),
                paper.get("arxiv_id"),
                paper.get("abstract"),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_papers(self) -> pd.DataFrame:
        """Get all papers as DataFrame."""
        return pd.read_sql_query("SELECT * FROM papers", self.conn)

    def close(self):
        """Close database connection."""
        self.conn.close()


class RealConfigForTests:
    """Real configuration object for tests (not a mock)."""

    def __init__(self, **overrides):
        # Set defaults
        self.cache_dir = Path(tempfile.mkdtemp()) / "cache"
        self.output_dir = Path(tempfile.mkdtemp()) / "output"
        self.data_dir = Path(tempfile.mkdtemp()) / "data"
        self.log_dir = Path(tempfile.mkdtemp()) / "logs"

        # Paths
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.parallel_workers = 2
        self.pdf_timeout_seconds = 30
        self.pdf_max_size_mb = 50
        self.cache_max_age_days = 90
        self.use_cache = True
        self.llm_model = "gemini/gemini-pro"
        self.llm_temperature = 0.1
        self.batch_size_pdf = 10
        self.unpaywall_email = "test@example.com"

        # Search settings
        self.wargame_terms = ["wargame", "wargaming", "war game"]
        self.llm_terms = ["LLM", "large language model", "GPT", "Claude"]
        self.search_years = (2020, 2024)

        # Rate limits
        self.rate_limits = {
            "arxiv": {"delay_milliseconds": 0},  # No delay in tests
            "semantic_scholar": {"requests_per_second": 100},
        }

        # Apply overrides
        for key, value in overrides.items():
            setattr(self, key, value)

    def cleanup(self):
        """Clean up temporary directories."""
        import shutil

        for dir_path in [self.cache_dir, self.output_dir, self.data_dir, self.log_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path.parent)  # Remove temp dir
