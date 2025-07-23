"""Pytest configuration and shared fixtures for literature review pipeline tests."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration for testing."""
    config = {
        "search": {
            "queries": {
                "preset1": '"LLM" AND ("wargaming" OR "wargame") AND ("AI" OR "artificial intelligence")',
                "preset2": '"Large Language Model" AND ("strategic game" OR "conflict simulation")',
            },
            "sources": {
                "google_scholar": {"enabled": True, "max_results": 100},
                "arxiv": {"enabled": True, "max_results": 50},
                "semantic_scholar": {"enabled": True, "max_results": 50},
                "crossref": {"enabled": True, "max_results": 50},
            },
        },
        "api_keys": {
            "openai": "test-key",
            "semantic_scholar": "test-key",
            "zenodo": "test-key",
        },
        "extraction": {
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 4000,
            "prompt": "Extract information about LLM wargaming from this paper",
        },
        "failure_vocabularies": {
            "escalation": ["escalation", "nuclear", "escalate", "brinkmanship"],
            "bias": ["bias", "biased", "unfair", "skew"],
            "hallucination": [
                "hallucination",
                "hallucinate",
                "confabulate",
                "fabricate",
            ],
            "prompt_sensitivity": [
                "prompt sensitivity",
                "prompt engineering",
                "fragile",
            ],
            "data_leakage": ["data leakage", "memorization", "training data"],
            "deception": ["deception", "deceive", "mislead", "manipulation"],
        },
        "paths": {
            "data_dir": str(Path(temp_dir) / "data"),
            "pdf_cache": str(Path(temp_dir) / "pdf_cache"),
            "output_dir": str(Path(temp_dir) / "outputs"),
            "log_db": str(Path(temp_dir) / "logs.db"),
        },
        "viz": {
            "charts": {
                "timeline": {"enabled": True, "figsize": [10, 6]},
                "venue_dist": {"enabled": True, "figsize": [8, 6]},
                "failure_modes": {"enabled": True, "figsize": [10, 8]},
                "llm_families": {"enabled": True, "figsize": [8, 6]},
                "game_types": {"enabled": True, "figsize": [8, 6]},
                "awscale": {"enabled": True, "figsize": [10, 6]},
            }
        },
        "export": {
            "zenodo": {
                "enabled": False,
                "title": "Test Dataset",
                "creators": [{"name": "Test Author"}],
                "description": "Test description",
                "access_right": "open",
                "license": "cc-by",
                "upload_type": "dataset",
            }
        },
    }

    config_path = Path(temp_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create necessary directories
    Path(config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["pdf_cache"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    return config_path


@pytest.fixture
def sample_papers_df():
    """Create a sample papers DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Using LLMs in Strategic Wargaming",
                "GPT-4 as a Crisis Simulation Agent",
                "Evaluating Language Models in Military Games",
            ],
            "authors": [
                "Smith, John; Doe, Jane",
                "Johnson, Alice",
                "Brown, Bob; Wilson, Carol",
            ],
            "year": [2024, 2023, 2024],
            "abstract": [
                "This paper explores the application of large language models in strategic wargaming scenarios...",
                "We investigate how GPT-4 can serve as an autonomous agent in crisis simulation exercises...",
                "A comprehensive evaluation of various language models in military gaming contexts...",
            ],
            "source_db": ["google_scholar", "arxiv", "semantic_scholar"],
            "url": [
                "https://example.com/paper1",
                "https://arxiv.org/abs/2301.12345",
                "https://example.com/paper3",
            ],
            "doi": ["10.1234/example.2024.001", "", "10.5678/example.2024.003"],
            "arxiv_id": ["", "2301.12345", ""],
            "venue": [
                "International Conference on AI and Games",
                "",
                "Journal of Military AI",
            ],
            "citations": [15, 8, 22],
            "pdf_url": ["", "https://arxiv.org/pdf/2301.12345.pdf", ""],
            "keywords": [
                "wargaming; LLM; strategy",
                "crisis simulation; GPT-4; agent",
                "military; AI; evaluation",
            ],
        }
    )


@pytest.fixture
def sample_screening_df():
    """Create a sample screening DataFrame for testing."""
    df = pd.DataFrame(
        {
            "screening_id": ["SCREEN_0001", "SCREEN_0002", "SCREEN_0003"],
            "title": [
                "Using LLMs in Strategic Wargaming",
                "GPT-4 as a Crisis Simulation Agent",
                "Evaluating Language Models in Military Games",
            ],
            "authors": [
                "Smith, John; Doe, Jane",
                "Johnson, Alice",
                "Brown, Bob; Wilson, Carol",
            ],
            "year": [2024, 2023, 2024],
            "venue": [
                "International Conference on AI and Games",
                "",
                "Journal of Military AI",
            ],
            "abstract": [
                "This paper explores the application of large language models in strategic wargaming scenarios...",
                "We investigate how GPT-4 can serve as an autonomous agent in crisis simulation exercises...",
                "A comprehensive evaluation of various language models in military gaming contexts...",
            ],
            "doi": ["10.1234/example.2024.001", "", "10.5678/example.2024.003"],
            "url": [
                "https://example.com/paper1",
                "https://arxiv.org/abs/2301.12345",
                "https://example.com/paper3",
            ],
            "source_db": ["google_scholar", "arxiv", "semantic_scholar"],
            "include_ta": ["yes", "yes", ""],
            "reason_ta": ["", "", ""],
            "notes_ta": ["Looks highly relevant", "Relevant to LLM agents", ""],
            "include_ft": ["", "", ""],
            "reason_ft": ["", "", ""],
            "notes_ft": ["", "", ""],
            "relevance_score": [5, 4, None],
            "quality_score": [None, None, None],
            "pdf_path": [
                "pdf_cache/Smith_2024_Using_LLMs.pdf",
                "pdf_cache/Johnson_2023_GPT4.pdf",
                "",
            ],
            "pdf_status": ["downloaded_direct", "downloaded_arxiv", "not_found"],
        }
    )
    return df


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content='{"venue_type": "conference", "game_type": "matrix", "open_ended": "yes", "quantitative": "yes", "llm_family": "GPT-4", "llm_role": "player"}'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_scholarly():
    """Mock scholarly library for testing."""
    with patch("scholarly.scholarly") as mock:
        # Mock search results
        mock_results = [
            {
                "bib": {
                    "title": "Using LLMs in Strategic Wargaming",
                    "author": "Smith, John and Doe, Jane",
                    "pub_year": "2024",
                    "abstract": "This paper explores the application of large language models...",
                    "venue": "International Conference on AI and Games",
                    "pub_url": "https://example.com/paper1",
                },
                "num_citations": 15,
            }
        ]
        mock.search_pubs.return_value = (r for r in mock_results)
        yield mock


@pytest.fixture
def mock_arxiv():
    """Mock arxiv library for testing."""
    with patch("arxiv.Search") as mock_search:
        mock_result = Mock()
        mock_result.title = "GPT-4 as a Crisis Simulation Agent"
        mock_result.authors = [Mock(name="Johnson, Alice")]
        mock_result.published = Mock(year=2023)
        mock_result.summary = (
            "We investigate how GPT-4 can serve as an autonomous agent..."
        )
        mock_result.entry_id = "https://arxiv.org/abs/2301.12345"
        mock_result.doi = None
        mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345.pdf"

        mock_search.return_value.results.return_value = [mock_result]
        yield mock_search
