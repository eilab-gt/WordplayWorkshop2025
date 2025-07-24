"""Common test fixtures and utilities for the literature review pipeline."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.lit_review.harvesters.base import Paper


class MockConfig:
    """Mock configuration object with all common settings."""

    def __init__(self):
        # Search configuration
        self.wargame_terms = ["wargame", "wargaming", "war game"]
        self.llm_terms = ["LLM", "language model", "GPT", "BERT"]
        self.action_terms = ["simulation", "training", "decision"]
        self.exclusion_terms = ["review", "survey"]
        self.search_years = (2018, 2025)

        # Search configuration object (for nested access)
        self.search = MagicMock()
        self.search.years.start = 2018
        self.search.years.end = 2025
        self.search.llm_min_params = 100_000_000

        # Rate limits
        self.rate_limits = {
            "arxiv": {
                "requests_per_second": 100,
                "delay_milliseconds": 10,
                "burst_limit": 200,
            },
            "crossref": {
                "requests_per_second": 100,
                "delay_milliseconds": 10,
                "burst_limit": 200,
            },
            "semantic_scholar": {
                "requests_per_second": 100,
                "delay_milliseconds": 10,
                "burst_limit": 200,
            },
            "google_scholar": {
                "requests_per_second": 1,
                "delay_milliseconds": 1000,
                "burst_limit": 10,
            },
        }

        # Email configuration
        self.unpaywall_email = "test@example.com"

        # Parallel processing
        self.parallel_workers = 3

        # Paths
        self.pdf_cache_dir = Path("/tmp/test_pdf_cache")
        self.content_cache_dir = Path("/tmp/test_content_cache")

        # Content cache settings
        self.content_cache = MagicMock()
        self.content_cache.enabled = True
        self.content_cache.directory = self.content_cache_dir
        self.content_cache.max_size_mb = 100
        self.content_cache.ttl_days = 7


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    return MockConfig()


@pytest.fixture
def sample_paper():
    """Create a sample Paper object."""
    return Paper(
        title="LLM-powered Wargaming Simulation",
        authors=["John Doe", "Jane Smith"],
        year=2024,
        abstract="This paper presents a novel approach to wargaming using large language models.",
        source_db="test",
        url="https://example.com/paper1",
        doi="10.1234/test1",
        arxiv_id="2401.12345",
        venue="Journal of AI Warfare",
        citations=10,
        pdf_url="https://example.com/paper1.pdf",
        keywords=["wargaming", "LLM", "simulation"],
    )


@pytest.fixture
def sample_papers():
    """Create a list of sample Paper objects."""
    return [
        Paper(
            title="LLM-powered Wargaming Simulation",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            abstract="This paper presents a novel approach to wargaming using large language models.",
            source_db="test",
            url="https://example.com/paper1",
            doi="10.1234/test1",
            arxiv_id="2401.12345",
            venue="Journal of AI Warfare",
            citations=10,
            pdf_url="https://example.com/paper1.pdf",
            keywords=["wargaming", "LLM", "simulation"],
        ),
        Paper(
            title="GPT-4 in Military Decision Making",
            authors=["Alice Johnson"],
            year=2023,
            abstract="Exploring the use of GPT-4 for military strategic planning.",
            source_db="test",
            url="https://example.com/paper2",
            doi="10.5678/test2",
            venue="Conference on AI Security",
            citations=5,
            keywords=["GPT-4", "military", "strategy"],
        ),
        Paper(
            title="Old Paper Outside Date Range",
            authors=["Bob Wilson"],
            year=2015,
            abstract="This paper is too old to be included.",
            source_db="test",
            url="https://example.com/paper3",
            doi="10.9999/old1",
            venue="Old Journal",
            citations=100,
        ),
    ]


@pytest.fixture
def sample_dataframe(sample_papers):
    """Create a sample DataFrame from papers."""
    data = [paper.to_dict() for paper in sample_papers]
    return pd.DataFrame(data)


def create_mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] = None,
    raise_for_status: bool = True,
) -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code

    if json_data:
        response.json.return_value = json_data

    if raise_for_status and status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        response.raise_for_status = MagicMock()

    return response


def create_arxiv_response(papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a mock arXiv API response."""
    entries = []
    for paper in papers:
        entry = {
            "id": paper.get("arxiv_id", "2401.12345"),
            "title": paper.get("title", "Test Paper"),
            "summary": paper.get("abstract", "Test abstract"),
            "authors": [
                {"name": author} for author in paper.get("authors", ["Test Author"])
            ],
            "published": paper.get("published", "2024-01-15T00:00:00Z"),
            "updated": paper.get("updated", "2024-01-15T00:00:00Z"),
            "links": [
                {
                    "href": paper.get(
                        "pdf_url", "https://arxiv.org/pdf/2401.12345.pdf"
                    ),
                    "type": "application/pdf",
                },
                {
                    "href": paper.get("url", "https://arxiv.org/abs/2401.12345"),
                    "type": "text/html",
                },
            ],
            "arxiv:primary_category": {"term": paper.get("category", "cs.AI")},
            "arxiv:comment": paper.get("comment", ""),
        }
        entries.append(entry)

    return {
        "entries": entries,
        "opensearch:totalResults": {"$": str(len(entries))},
        "opensearch:startIndex": {"$": "0"},
        "opensearch:itemsPerPage": {"$": str(len(entries))},
    }


def create_crossref_response(papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a mock CrossRef API response."""
    items = []
    for paper in papers:
        item = {
            "DOI": paper.get("doi", "10.1234/test"),
            "title": [paper.get("title", "Test Paper")],
            "author": [
                {"given": name.split()[0], "family": " ".join(name.split()[1:])}
                for name in paper.get("authors", ["Test Author"])
            ],
            "published-print": {
                "date-parts": [
                    [
                        paper.get("year", 2024),
                        paper.get("month", 1),
                        paper.get("day", 1),
                    ]
                ]
            },
            "abstract": paper.get("abstract", "Test abstract"),
            "URL": paper.get("url", "https://doi.org/10.1234/test"),
            "type": paper.get("type", "journal-article"),
            "container-title": [paper.get("venue", "Test Journal")],
            "is-referenced-by-count": paper.get("citations", 0),
        }

        if paper.get("pdf_url"):
            item["link"] = [
                {"URL": paper["pdf_url"], "content-type": "application/pdf"}
            ]

        items.append(item)

    return {"status": "ok", "message": {"items": items, "total-results": len(items)}}


def create_semantic_scholar_response(papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a mock Semantic Scholar API response."""
    data = []
    for paper in papers:
        item = {
            "paperId": paper.get("paper_id", "abc123"),
            "title": paper.get("title", "Test Paper"),
            "abstract": paper.get("abstract", "Test abstract"),
            "authors": [
                {"name": author} for author in paper.get("authors", ["Test Author"])
            ],
            "year": paper.get("year", 2024),
            "venue": paper.get("venue", "Test Conference"),
            "citationCount": paper.get("citations", 0),
            "url": paper.get("url", "https://www.semanticscholar.org/paper/abc123"),
            "externalIds": {
                "DOI": paper.get("doi"),
                "ArXiv": paper.get("arxiv_id"),
            },
            "isOpenAccess": paper.get("is_open_access", True),
            "openAccessPdf": (
                {"url": paper.get("pdf_url")} if paper.get("pdf_url") else None
            ),
        }
        data.append(item)

    return {"total": len(data), "offset": 0, "data": data}


def assert_paper_equals(
    actual: Paper, expected: Paper, ignore_fields: list[str] = None
):
    """Assert that two Paper objects are equal, optionally ignoring certain fields."""
    ignore_fields = ignore_fields or []

    fields = [
        "title",
        "authors",
        "year",
        "abstract",
        "source_db",
        "url",
        "doi",
        "arxiv_id",
        "venue",
        "citations",
        "pdf_url",
        "keywords",
    ]

    for field in fields:
        if field not in ignore_fields:
            actual_val = getattr(actual, field)
            expected_val = getattr(expected, field)
            assert (
                actual_val == expected_val
            ), f"Field {field} mismatch: {actual_val} != {expected_val}"


def create_test_pdf(path: Path, content: str = "Test PDF content"):
    """Create a test PDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # For testing, we'll just create a text file that pretends to be a PDF
    # In real tests, you might want to use a library like reportlab to create actual PDFs
    path.write_text(f"%PDF-1.4\n{content}\n%%EOF", encoding="utf-8")


def create_test_cache_structure(base_dir: Path) -> dict[str, Path]:
    """Create a test cache directory structure."""
    dirs = {
        "pdf": base_dir / "pdf_cache",
        "tex": base_dir / "content_cache" / "tex",
        "html": base_dir / "content_cache" / "html",
        "extracted": base_dir / "content_cache" / "extracted",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


class MockLLMService:
    """Mock LLM service for testing extraction."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.health_status = True
        self.available_models = {
            "gemini/gemini-pro": {"available": True},
            "gpt-3.5-turbo": {"available": True},
            "claude-3-haiku-20240307": {"available": True},
        }
        self.extraction_response = {
            "success": True,
            "extracted_data": {
                "research_questions": "How can LLMs enhance wargaming?",
                "key_contributions": "Novel LLM integration framework",
                "simulation_approach": "Matrix-based tabletop approach",
                "llm_usage": "GPT-4 for decision support",
                "human_llm_comparison": "Humans make strategic decisions, LLMs provide analysis",
                "evaluation_metrics": "Decision quality, time to decision",
                "prompting_strategies": "Chain of thought, few-shot examples",
                "emerging_behaviors": "Strategic reasoning patterns",
                "datasets_used": "Custom wargaming scenarios",
                "limitations": "Limited to text-based scenarios",
            },
        }

    def mock_health_check(self, *args, **kwargs):
        """Mock health check response."""
        response = MagicMock()
        response.status_code = 200 if self.health_status else 503
        return response

    def mock_get_models(self, *args, **kwargs):
        """Mock get models response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = self.available_models
        return response

    def mock_extract(self, *args, **kwargs):
        """Mock extraction response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = self.extraction_response
        return response


@pytest.fixture
def mock_llm_service():
    """Provide a mock LLM service."""
    return MockLLMService()


# Common test data patterns
TEST_ARXIV_IDS = ["2401.12345", "2312.67890", "2403.11111"]
TEST_DOIS = ["10.1234/test1", "10.5678/test2", "10.9999/test3"]
TEST_YEARS = [2024, 2023, 2022]
TEST_VENUES = [
    "Journal of AI Warfare",
    "Conference on AI Security",
    "Workshop on LLM Applications",
]
