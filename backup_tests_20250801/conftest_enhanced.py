"""Enhanced pytest configuration and fixtures."""

import logging
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from _pytest.config import Config
from _pytest.nodes import Item

from tests.test_doubles import (
    FakeArxivAPI,
    FakeDatabase,
    FakeLLMService,
    RealConfigForTests,
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom settings."""
    # Add custom markers dynamically
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests by path
        if "test_harvesters" in str(item.fspath):
            item.add_marker(pytest.mark.harvester)
        elif "test_extraction" in str(item.fspath):
            item.add_marker(pytest.mark.extractor)
        elif "test_processing" in str(item.fspath):
            item.add_marker(pytest.mark.processor)
        elif "test_visualization" in str(item.fspath):
            item.add_marker(pytest.mark.visualization)
        elif "test_utils" in str(item.fspath):
            item.add_marker(pytest.mark.utils)

        # Auto-mark by test name patterns
        if "test_downloads_" in item.name or "test_fetches_" in item.name:
            item.add_marker(pytest.mark.network)
        if "parallel" in item.name or "concurrent" in item.name:
            item.add_marker(pytest.mark.heavy_compute)
        if "llm" in item.name.lower() or "extract" in item.name:
            item.add_marker(pytest.mark.llm_service)

        # Mark slow tests
        if any(keyword in item.name for keyword in ["integration", "e2e", "full"]):
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.fast)


# Global fixtures
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path]:
    """Provide temporary directory that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def real_config(temp_dir: Path) -> Generator[RealConfigForTests]:
    """Provide real configuration object for tests."""
    config = RealConfigForTests(
        cache_dir=temp_dir / "cache",
        output_dir=temp_dir / "output",
        data_dir=temp_dir / "data",
        log_dir=temp_dir / "logs",
    )
    yield config
    config.cleanup()


@pytest.fixture
def fake_llm_service() -> FakeLLMService:
    """Provide fake LLM service for tests."""
    return FakeLLMService(healthy=True)


@pytest.fixture
def fake_arxiv_api() -> FakeArxivAPI:
    """Provide fake arXiv API for tests."""
    return FakeArxivAPI()


@pytest.fixture
def fake_database() -> Generator[FakeDatabase]:
    """Provide in-memory database for tests."""
    db = FakeDatabase()
    yield db
    db.close()


# Parametrized fixtures for common test scenarios
@pytest.fixture(params=[True, False])
def parallel_mode(request) -> bool:
    """Parametrize tests for both parallel and sequential modes."""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def batch_size(request) -> int:
    """Parametrize tests with different batch sizes."""
    return request.param


# Skip markers for conditional tests
def pytest_runtest_setup(item: Item) -> None:
    """Skip tests based on markers and conditions."""
    # Skip WIP tests in CI
    if item.get_closest_marker("wip") and item.config.getoption("--ci"):
        pytest.skip("WIP test skipped in CI")

    # Skip network tests if offline
    if item.get_closest_marker("network") and item.config.getoption("--offline"):
        pytest.skip("Network test skipped in offline mode")

    # Skip LLM tests if service not available
    if item.get_closest_marker("llm_service"):
        # Could check if service is actually running
        pass


# Custom command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--ci", action="store_true", default=False, help="Running in CI environment"
    )
    parser.addoption(
        "--offline", action="store_true", default=False, help="Running in offline mode"
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--smoke", action="store_true", default=False, help="Run smoke tests only"
    )


# Test result hooks
def pytest_runtest_makereport(item: Item, call):
    """Add custom test result handling."""
    if call.when == "call" and call.excinfo is not None:
        # Log additional info for failures
        if hasattr(item, "funcargs"):
            logging.error(f"Test {item.name} failed with args: {item.funcargs}")


# Fixtures for common test data
@pytest.fixture
def sample_paper_data():
    """Provide sample paper data for tests."""
    return {
        "title": "LLM Wargaming: A Novel Framework",
        "authors": ["Smith, A.", "Jones, B."],
        "year": 2024,
        "abstract": "We present a novel framework for integrating LLMs into wargaming simulations...",
        "doi": "10.1234/test.2024.001",
        "arxiv_id": "2401.00001",
        "keywords": ["LLM", "wargaming", "simulation"],
        "venue": "International Conference on AI",
        "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
    }


@pytest.fixture
def sample_bibtex():
    """Provide sample BibTeX entry for tests."""
    return """@article{smith2024llm,
  title={LLM Wargaming: A Novel Framework},
  author={Smith, A. and Jones, B.},
  journal={International Conference on AI},
  year={2024},
  doi={10.1234/test.2024.001}
}"""


# Performance profiling fixture
@pytest.fixture
def profile_performance(request):
    """Profile test performance when requested."""
    if request.config.getoption("--profile"):
        import cProfile
        import pstats
        from io import StringIO

        profiler = cProfile.Profile()
        profiler.enable()

        yield

        profiler.disable()
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(10)  # Top 10 functions
        print(f"\nProfile for {request.node.name}:")
        print(stream.getvalue())
