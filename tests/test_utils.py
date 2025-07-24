"""Helper functions and utilities for testing."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.lit_review.harvesters.base import Paper


@contextmanager
def mock_time_sleep():
    """Context manager to mock time.sleep for rate limiting tests."""
    with patch("time.sleep") as mock_sleep:
        yield mock_sleep


@contextmanager
def mock_requests_get(responses: list[MagicMock] = None, side_effect=None):
    """Context manager to mock requests.get with predefined responses."""
    with patch("requests.get") as mock_get:
        if side_effect:
            mock_get.side_effect = side_effect
        elif responses:
            if len(responses) == 1:
                mock_get.return_value = responses[0]
            else:
                mock_get.side_effect = responses
        yield mock_get


@contextmanager
def mock_requests_post(response: MagicMock = None, side_effect=None):
    """Context manager to mock requests.post."""
    with patch("requests.post") as mock_post:
        if side_effect:
            mock_post.side_effect = side_effect
        elif response:
            mock_post.return_value = response
        yield mock_post


def assert_rate_limiting_applied(
    mock_sleep: MagicMock, expected_calls: int, min_delay_ms: int = 0
):
    """Assert that rate limiting was properly applied."""
    assert (
        mock_sleep.call_count >= expected_calls - 1
    ), f"Expected at least {expected_calls - 1} sleep calls, got {mock_sleep.call_count}"

    if min_delay_ms > 0:
        for call in mock_sleep.call_args_list:
            delay = call[0][0] if call[0] else 0
            expected_delay = min_delay_ms / 1000.0
            assert (
                delay >= expected_delay * 0.9
            ), f"Expected delay of at least {expected_delay}s, got {delay}s"


def assert_papers_filtered_by_year(papers: list[Paper], start_year: int, end_year: int):
    """Assert that all papers are within the specified year range."""
    for paper in papers:
        assert (
            paper.year >= start_year and paper.year <= end_year
        ), f"Paper '{paper.title}' year {paper.year} outside range {start_year}-{end_year}"


def assert_dataframe_has_columns(df: pd.DataFrame, required_columns: list[str]):
    """Assert that a DataFrame has all required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    assert not missing_columns, f"DataFrame missing columns: {missing_columns}"


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


def create_temp_file(tmp_path: Path, filename: str, content: str) -> Path:
    """Create a temporary file with given content."""
    file_path = tmp_path / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_mock_pdf_file(
    path: Path, title: str = "Test Paper", abstract: str = "Test abstract"
) -> Path:
    """Create a mock PDF file with embedded text."""
    # Simple text file masquerading as PDF for testing
    content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R >>
endobj
xref
0 4
trailer
<< /Root 1 0 R >>

Title: {title}
Abstract: {abstract}

This is the main content of the paper.
%%EOF"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def create_mock_tex_content(
    title: str = "Test Paper",
    abstract: str = "Test abstract",
    sections: dict[str, str] = None,
) -> str:
    """Create mock TeX content."""
    sections = sections or {
        "Introduction": "This is the introduction.",
        "Related Work": "Previous work in this area.",
        "Method": "Our approach uses LLMs for wargaming.",
        "Results": "We achieved significant improvements.",
        "Conclusion": "This work demonstrates the potential of LLMs.",
    }

    content = f"""\\documentclass{{article}}
\\title{{{title}}}
\\author{{Test Author}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

"""

    for section_title, section_content in sections.items():
        content += f"\\section{{{section_title}}}\n{section_content}\n\n"

    content += "\\end{document}"
    return content


def create_mock_html_content(
    title: str = "Test Paper",
    abstract: str = "Test abstract",
    sections: dict[str, str] = None,
) -> str:
    """Create mock HTML content."""
    sections = sections or {
        "Introduction": "This is the introduction.",
        "Related Work": "Previous work in this area.",
        "Method": "Our approach uses LLMs for wargaming.",
        "Results": "We achieved significant improvements.",
        "Conclusion": "This work demonstrates the potential of LLMs.",
    }

    content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<main>
<h1>{title}</h1>
<div class="abstract">
<h2>Abstract</h2>
<p>{abstract}</p>
</div>
"""

    for section_title, section_content in sections.items():
        content += f"""
<section>
<h2>{section_title}</h2>
<p>{section_content}</p>
</section>"""

    content += """
</main>
</body>
</html>"""
    return content


def compare_papers(
    paper1: Paper, paper2: Paper, ignore_fields: list[str] = None
) -> list[str]:
    """Compare two papers and return list of differing fields."""
    ignore_fields = ignore_fields or []
    differences = []

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
            val1 = getattr(paper1, field)
            val2 = getattr(paper2, field)
            if val1 != val2:
                differences.append(f"{field}: {val1} != {val2}")

    return differences


def create_extraction_dataframe(
    papers: list[Paper], extracted_data: dict[str, dict[str, Any]] = None
) -> pd.DataFrame:
    """Create a DataFrame with extraction columns populated."""
    df = pd.DataFrame([p.to_dict() for p in papers])

    # Add PDF path column
    df["pdf_path"] = df.apply(
        lambda row: f"/tmp/pdfs/{row['doi']}.pdf" if row["doi"] else "", axis=1
    )
    df["pdf_status"] = "downloaded"

    # Add extraction columns
    extraction_cols = [
        "content_type",
        "research_questions",
        "key_contributions",
        "simulation_approach",
        "llm_usage",
        "human_llm_comparison",
        "evaluation_metrics",
        "prompting_strategies",
        "emerging_behaviors",
        "datasets_used",
        "limitations",
        "awscale",
        "extraction_status",
        "extraction_model",
        "extraction_confidence",
    ]

    for col in extraction_cols:
        df[col] = ""

    # Populate with provided extracted data
    if extracted_data:
        for idx, data in extracted_data.items():
            for col, value in data.items():
                if col in df.columns:
                    df.at[idx, col] = value

    return df


class BatchProcessHelper:
    """Helper for testing batch processing operations."""

    @staticmethod
    def create_mock_batch_response(
        success: bool = True, processed: int = 10, failed: int = 0
    ) -> dict[str, Any]:
        """Create a mock batch processing response."""
        return {
            "success": success,
            "processed": processed,
            "failed": failed,
            "errors": [] if success else ["Processing error"],
            "duration": 1.5,
        }

    @staticmethod
    def assert_batch_completed(
        result: dict[str, Any], expected_processed: int, expected_failed: int = 0
    ):
        """Assert batch processing completed successfully."""
        assert result["success"], f"Batch processing failed: {result.get('errors', [])}"
        assert (
            result["processed"] == expected_processed
        ), f"Expected {expected_processed} processed, got {result['processed']}"
        assert (
            result["failed"] == expected_failed
        ), f"Expected {expected_failed} failed, got {result['failed']}"


def mock_harvester_search_results(harvester_type: str, count: int = 5) -> list[Paper]:
    """Generate mock search results for a specific harvester type."""
    papers = []
    for i in range(count):
        paper = Paper(
            title=f"{harvester_type} Paper {i+1}: LLM Wargaming Study",
            authors=[f"Author{i+1} Name"],
            year=2024 - (i % 3),
            abstract=f"Abstract for {harvester_type} paper {i+1} about LLM wargaming.",
            source_db=harvester_type.lower(),
            url=f"https://{harvester_type.lower()}.com/paper{i+1}",
            doi=f"10.{1000+i}/{harvester_type.lower()}{i+1}",
            venue=f"{harvester_type} Journal {i % 3 + 1}",
            citations=i * 5,
        )
        papers.append(paper)
    return papers


def validate_extraction_results(
    df: pd.DataFrame, expected_success_count: int, required_fields: list[str] = None
):
    """Validate extraction results in a DataFrame."""
    required_fields = required_fields or [
        "research_questions",
        "key_contributions",
        "simulation_approach",
        "llm_usage",
        "evaluation_metrics",
    ]

    success_mask = df["extraction_status"] == "success"
    success_count = success_mask.sum()

    assert (
        success_count == expected_success_count
    ), f"Expected {expected_success_count} successful extractions, got {success_count}"

    # Check that successful extractions have required fields
    for idx in df[success_mask].index:
        for field in required_fields:
            value = df.at[idx, field]
            assert (
                value and value != ""
            ), f"Paper at index {idx} missing required field '{field}'"


# Decorator for tests that require external services
def requires_service(service_name: str):
    """Decorator to skip tests if a required service is not available."""

    def decorator(test_func):
        return pytest.mark.skipif(
            not check_service_available(service_name),
            reason=f"{service_name} service not available",
        )(test_func)

    return decorator


def check_service_available(service_name: str) -> bool:
    """Check if an external service is available."""
    # This is a placeholder - implement actual service checks as needed
    service_checks = {
        "llm_service": lambda: check_url_accessible("http://localhost:8000/health"),
        "internet": lambda: check_url_accessible("https://google.com"),
    }

    check_func = service_checks.get(service_name)
    if check_func:
        try:
            return check_func()
        except:
            return False
    return True


def check_url_accessible(url: str, timeout: int = 2) -> bool:
    """Check if a URL is accessible."""
    try:
        import requests

        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


# Test data generators
def generate_test_papers(
    count: int, source_db: str = "test", year_range: tuple = (2020, 2024)
) -> list[Paper]:
    """Generate test papers with varied data."""
    papers = []
    venues = ["Journal A", "Conference B", "Workshop C"]

    for i in range(count):
        year = year_range[0] + (i % (year_range[1] - year_range[0] + 1))
        paper = Paper(
            title=f"Test Paper {i+1}: {source_db} Study on LLM Wargaming",
            authors=[f"Author {j+1}" for j in range((i % 3) + 1)],
            year=year,
            abstract=f"Abstract for test paper {i+1} from {source_db}. " * 5,
            source_db=source_db,
            url=f"https://example.com/{source_db}/paper{i+1}",
            doi=f"10.{1000+i}/{source_db}.{i+1}" if i % 2 == 0 else None,
            arxiv_id=f"{year}.{10000+i}" if i % 3 == 0 else None,
            venue=venues[i % len(venues)],
            citations=i * 10,
            pdf_url=(
                f"https://example.com/{source_db}/paper{i+1}.pdf"
                if i % 2 == 0
                else None
            ),
            keywords=[source_db, "test", f"keyword{i}"],
        )
        papers.append(paper)

    return papers
