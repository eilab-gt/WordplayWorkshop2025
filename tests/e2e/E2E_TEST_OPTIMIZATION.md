# E2E Test Optimization Guide

Last Updated: July 27, 2025

## Current State Analysis

The E2E test suite exhibits significant performance issues due to over-engineering:

### Key Issues Identified
- **3,875 lines** of test code across 7 files for 53 test methods
- **Excessive data generation**: 2,000-10,000 papers generated per test
- **Threading overhead**: Complex concurrent execution patterns
- **Memory pressure**: Tests consume 500MB+ with explicit garbage collection
- **Long execution time**: 5-10 minutes for E2E suite
- **Test flakiness**: 10-20% failure rate due to timing and concurrency

### Performance Bottlenecks
1. **Data Generation Overhead**
   - `test_performance_load.py` generates 2,000 papers in setup
   - Each paper includes full metadata, abstracts, and simulated PDFs
   - Memory tests create 10,000 paper dataframes

2. **Redundant Testing**
   - Search performance: 20 baseline searches + 100 concurrent searches
   - Pipeline throughput: 500 papers processed repeatedly
   - Scalability tests: Up to 5,000 papers tested

3. **Complex Test Doubles**
   - FakeArxivAPI pre-generates 50 papers with full metadata
   - FakeLLMService simulates extraction logic
   - FakePDFServer generates fake PDF content

## Optimization Strategy

### Phase 1: Quick Wins (50% improvement)
Focus on immediate changes that require minimal code modification.

### Phase 2: Structural Changes (80% improvement)
Refactor test organization and simplify test doubles.

### Phase 3: Infrastructure Optimization (90% improvement)
Implement parallel execution and CI/CD improvements.

## Implementation Guide

### 1. Reduce Test Data Volume (Immediate)

**File: `test_performance_load.py`**
```python
# Line 70: Reduce from 2000 to 100 papers
for i in range(100):  # was range(2000)

# Line 307: Reduce from 10000 to 1000 papers
large_df = pd.DataFrame(
    [load_services["generator"].generate_paper() for _ in range(1000)]  # was 10000
)

# Line 517: Reduce test sizes
sizes = [10, 50, 100, 200, 500]  # was [100, 500, 1000, 2000, 5000]
```

**File: `test_realistic_scenarios.py`**
```python
# Line 36: Reduce sample size
sample_size=5,  # was 20

# Line 74: Limit papers with PDFs
for paper in arxiv.papers[:5]:  # was [:20]
```

### 2. Add Test Markers for Selective Running

**Add to all performance tests:**
```python
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.skipif(
    not os.environ.get("RUN_PERFORMANCE_TESTS"),
    reason="Set RUN_PERFORMANCE_TESTS=1 to run performance tests"
)
class TestPerformanceAndLoad:
```

**Configure pytest.ini:**
```ini
[tool:pytest]
addopts = -n auto --dist loadscope
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    performance: performance benchmarks
    integration: integration tests
    serial: tests that must run serially
```

### 3. Create Minimal Test Fixtures

**New file: `tests/fixtures/minimal_test_data.py`**
```python
MINIMAL_PAPERS = [
    {
        "id": "2024.00001",
        "title": "LLMs in Strategic Gaming",
        "authors": ["Smith, J.", "Doe, A."],
        "abstract": "We study LLMs in wargaming contexts...",
        "published": "2024-01-15",
        "categories": ["cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2024.00001.pdf"
    },
    # Add 4 more minimal papers
]

@pytest.fixture(scope="session")
def cached_test_papers():
    cache_file = Path("tests/fixtures/test_papers.pkl")
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    papers = pd.DataFrame(MINIMAL_PAPERS)
    cache_file.parent.mkdir(exist_ok=True)
    pd.to_pickle(papers, cache_file)
    return papers
```

### 4. Simplify Test Doubles

**Optimize `test_doubles.py`:**
```python
class FakeArxivAPI:
    def __init__(self, seed=42, max_papers=10):  # Reduced default
        self.papers = self._load_minimal_papers(max_papers)

    def _load_minimal_papers(self, count):
        # Load from static fixtures instead of generating
        from tests.fixtures.minimal_test_data import MINIMAL_PAPERS
        return MINIMAL_PAPERS[:count]

class FakeLLMService:
    def extract(self, text: str, model: str, **kwargs):
        # Simplified - no complex logic
        return {
            "success": True,
            "extracted_data": {
                "research_questions": "Test question",
                "key_contributions": "Test contribution"
            },
            "tokens_used": 100
        }
```

### 5. Remove Threading from E2E Tests

**Replace concurrent tests with sequential versions:**
```python
def test_multiple_searches_work(self, config, services, monkeypatch):
    """Test multiple searches sequentially."""
    harvester = SearchHarvester(config)

    queries = ["LLM", "wargaming", "AI agents"]
    results = []

    for query in queries:
        df = harvester.search_all(query, max_results=5)
        results.append(len(df))

    assert all(r > 0 for r in results)
    # No threading, no complex metrics
```

### 6. Move Performance Tests

**Create `tests/benchmarks/` directory:**
```
benchmarks/
├── __init__.py
├── test_search_performance.py
├── test_pipeline_throughput.py
└── test_memory_usage.py
```

**Example benchmark test:**
```python
@pytest.mark.benchmark
def test_search_performance_baseline(benchmark, services):
    """Dedicated performance benchmark."""
    harvester = SearchHarvester(test_config)
    result = benchmark(harvester.search_all, "LLM", max_results=100)
    assert len(result) > 0
```

### 7. Optimize CI/CD Pipeline

**.github/workflows/test.yml:**
```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit -n auto

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/e2e -m "not slow" -n auto

  performance-tests:
    runs-on: ubuntu-latest
    schedule:
      - cron: '0 2 * * *'  # Nightly only
    steps:
      - run: pytest tests/benchmarks --benchmark-only
```

## Performance Targets

### Before Optimization
- E2E test runtime: 5-10 minutes
- Memory usage: 500MB+
- Flaky tests: 10-20%
- CI pipeline: 15+ minutes

### After Optimization
- E2E test runtime: 30-60 seconds
- Memory usage: <100MB
- Flaky tests: <2%
- CI pipeline: 3-5 minutes

### Key Metrics to Track
- Test execution time per file
- Peak memory usage
- Test failure rate
- CI pipeline duration

## Quick Wins

### 1. Environment Variable for Quick Testing
```bash
# Run minimal E2E tests
QUICK_TEST=1 pytest tests/e2e/

# Skip performance tests by default
pytest tests/e2e/ -m "not slow"
```

### 2. Parallel Execution
```bash
# Use all CPU cores
pytest tests/e2e/ -n auto

# Run specific test categories
pytest tests/e2e/ -m "integration and not performance"
```

### 3. Fixture Caching
```python
# Add to conftest.py for session-wide caching
@pytest.fixture(scope="session")
def shared_test_data():
    return load_minimal_test_fixtures()
```

### 4. Mock Heavy Operations
```python
@pytest.fixture(autouse=True)
def fast_llm_extraction(monkeypatch):
    """Auto-mock LLM calls in E2E tests."""
    if not os.environ.get("USE_REAL_LLM"):
        monkeypatch.setattr(
            "src.lit_review.extraction.LLMExtractor.extract",
            lambda *args, **kwargs: {"extracted": True, "time": 0.001}
        )
```

### 5. Selective Test Running
```bash
# Daily development - fast tests only
pytest tests/e2e/ -m "not (slow or performance)"

# Pre-commit - integration tests
pytest tests/e2e/ -m "integration"

# Nightly - full suite
RUN_PERFORMANCE_TESTS=1 pytest tests/
```

## Implementation Priority

1. **Immediate (1 hour)**
   - Add skip markers to performance tests
   - Reduce data volumes in test files
   - Set up environment variables

2. **Short term (1 day)**
   - Create minimal test fixtures
   - Simplify test doubles
   - Remove threading from E2E tests

3. **Medium term (1 week)**
   - Separate performance tests
   - Implement parallel execution
   - Optimize CI/CD pipeline

## Recent Optimization Work

The following optimizations have already been implemented:
- Added pytest markers for test categorization
- Created environment-based test skipping
- Reduced default test data sizes in some files
- Started separating unit and integration concerns

## Next Steps

1. Apply the data volume reductions outlined above
2. Create the minimal test fixtures directory
3. Move performance tests to dedicated benchmark suite
4. Configure pytest for parallel execution
5. Update CI/CD workflows for optimized test running

By following this guide, the E2E test suite will transform from a slow, flaky bottleneck into a fast, reliable safety net that enables rapid development iteration.
