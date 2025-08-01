# E2E Test Optimization Action Plan

## Phase 1: Immediate Optimizations (Quick Wins)

### 1. Reduce Test Data Volume
**File**: `test_performance_load.py`
- Change line 70: `range(2000)` → `range(100)`
- Change line 307: `range(10000)` → `range(1000)`
- Change line 459: `range(1000)` → `range(100)`

### 2. Add Test Markers for Selective Running
```python
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.skip(reason="Run only in performance CI job")
class TestPerformanceAndLoad:
```

### 3. Cache Test Data Generation
```python
# Add to conftest.py
@pytest.fixture(scope="session")
def cached_test_papers():
    cache_file = Path("tests/fixtures/test_papers.pkl")
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    # Generate once and save
    papers = generate_test_papers(100)
    cache_file.parent.mkdir(exist_ok=True)
    pd.to_pickle(papers, cache_file)
    return papers
```

## Phase 2: Structural Improvements

### 1. Split Performance Tests
Create separate test files:
- `tests/benchmarks/test_search_performance.py`
- `tests/benchmarks/test_pipeline_throughput.py`
- `tests/benchmarks/test_memory_usage.py`

### 2. Simplify E2E Tests
Focus each E2E test on one scenario:
```python
class TestCoreIntegration:
    def test_search_to_export_flow(self):
        # 5 papers, no performance metrics
        papers = search_papers("LLM", max_results=5)
        processed = process_papers(papers)
        assert len(processed) == 5

    def test_error_handling_flow(self):
        # Test with known failure cases
        papers = get_error_test_papers()
        result = process_with_errors(papers)
        assert result.errors_handled_gracefully()
```

### 3. Mock Heavy Operations
```python
@pytest.fixture
def mock_llm_extraction(monkeypatch):
    def fast_extract(text, model):
        return {"extracted": True, "time": 0.001}

    monkeypatch.setattr(
        "src.lit_review.extraction.EnhancedLLMExtractor.extract",
        fast_extract
    )
```

## Phase 3: Test Data Optimization

### 1. Static Test Fixtures
Create `tests/fixtures/`:
```
fixtures/
  ├── minimal_papers.json     # 5 papers for smoke tests
  ├── standard_papers.json    # 20 papers for integration
  ├── edge_cases.json        # 10 papers with edge cases
  └── error_papers.json      # 5 papers that trigger errors
```

### 2. Fixture Loading Utility
```python
class TestFixtures:
    @staticmethod
    def load_papers(fixture_type="standard"):
        path = Path(f"tests/fixtures/{fixture_type}_papers.json")
        return pd.read_json(path)

    @staticmethod
    def get_paper_by_scenario(scenario):
        # Return specific test case
        scenarios = {
            "missing_pdf": {"id": "test.001", "pdf_url": None},
            "timeout": {"id": "test.002", "pdf_url": "http://slow.example.com"},
            "large_pdf": {"id": "test.003", "pdf_size_mb": 50},
        }
        return scenarios.get(scenario)
```

## Phase 4: Parallel Test Execution

### 1. Configure pytest-xdist
```ini
# pytest.ini
[tool:pytest]
addopts =
    -n auto  # Use all CPU cores
    --dist loadscope  # Distribute by test class
    --max-worker-restart 0  # Prevent worker restarts

# Mark tests that can't run in parallel
markers =
    serial: marks tests that must run serially
```

### 2. Make Tests Parallel-Safe
```python
@pytest.fixture
def isolated_config(tmp_path_factory):
    # Each test gets unique temp directory
    base = tmp_path_factory.mktemp("test_run")
    return Config(
        cache_dir=base / "cache",
        output_dir=base / "output"
    )
```

## Phase 5: CI/CD Optimization

### 1. Test Categorization
```yaml
# .github/workflows/test.yml
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
      - cron: '0 2 * * *'  # Nightly
    steps:
      - run: pytest tests/benchmarks --benchmark-only
```

### 2. Test Result Caching
```yaml
- name: Cache test data
  uses: actions/cache@v3
  with:
    path: tests/fixtures/generated/
    key: test-data-${{ hashFiles('tests/test_data_generators.py') }}
```

## Expected Results

### Before Optimization
- E2E test runtime: 5-10 minutes
- Memory usage: 500MB+
- Flaky tests: 10-20%
- CI time: 15+ minutes

### After Optimization
- E2E test runtime: 30-60 seconds
- Memory usage: <100MB
- Flaky tests: <2%
- CI time: 3-5 minutes

## Implementation Priority

1. **Week 1**: Phase 1 (Quick wins) - 50% improvement
2. **Week 2**: Phase 2 & 3 (Structure & Data) - 80% improvement
3. **Week 3**: Phase 4 & 5 (Parallel & CI) - 90% improvement

## Monitoring

Track these metrics:
- Test execution time per file
- Memory usage peak
- Test failure rate
- CI pipeline duration

Use pytest plugins:
- `pytest-benchmark` for performance tracking
- `pytest-timeout` to catch hanging tests
- `pytest-memray` for memory profiling
