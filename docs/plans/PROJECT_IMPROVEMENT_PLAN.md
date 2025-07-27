# 📋 Project Improvement Plan

## ✅ Completed
- **Test Coverage**: Achieved 78% (exceeded 70% target)
- **Critical Tests Added**: llm_service, enhanced_llm_extractor, arxiv_harvester
- **Comprehensive Analysis**: Identified all areas needing improvement

## 🔧 Priority 1: Fix Failing Tests (This Week)

### 1. LLM Service Test Fixes
```python
# In test_llm_service.py, update the mocking strategy:
@patch('litellm.completion')
def test_extract_success(self, mock_completion, client):
    # Ensure the mock properly simulates litellm behavior
    # Add proper response structure with usage attribute
```

### 2. Enhanced Extractor Test Fixes
```python
# Use caplog fixture instead of capsys for logger testing:
def test_log_statistics(self, extractor, caplog):
    with caplog.at_level(logging.INFO):
        extractor._log_statistics()
    assert "Enhanced LLM extraction statistics:" in caplog.text
```

### 3. ArXiv Harvester Test Fixes
```python
# Fix author mock objects:
author_mock = Mock()
author_mock.name = "Author Name"  # Set as attribute, not Mock parameter
```

## 📝 Priority 2: Documentation (Next 2 Weeks)

### 1. Create User Guide
```markdown
# docs/USER_GUIDE.md
- Getting Started
- Configuration Options
- Running the Pipeline
- Using Enhanced Features
- Troubleshooting
```

### 2. API Documentation
```markdown
# docs/API_REFERENCE.md
- LLM Service Endpoints
- Enhanced Extractor Methods
- Harvester Interfaces
```

### 3. Configuration Guide
```markdown
# docs/CONFIGURATION.md
- All config options explained
- Example configurations
- Performance tuning tips
```

## 🧹 Priority 3: Code Cleanup (Next Month)

### 1. Extract Common Patterns

Create `src/lit_review/common/rate_limiter.py`:
```python
class RateLimiter:
    """Unified rate limiting for all harvesters."""
    def __init__(self, delay_ms: int):
        self.delay_ms = delay_ms

    def wait(self):
        time.sleep(self.delay_ms / 1000.0)
```

### 2. Reduce Code Duplication

Create base error handling:
```python
# src/lit_review/common/error_handler.py
def handle_api_error(func):
    """Decorator for consistent API error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            logger.error(f"API error in {func.__name__}: {e}")
            return None
    return wrapper
```

### 3. Break Up Large Functions

Split `create_summary_report()` into:
- `_calculate_statistics()`
- `_generate_failure_analysis()`
- `_create_recommendations()`

## 🚀 Priority 4: Performance Improvements

### 1. Implement Async Harvesters
```python
# Create async versions for parallel API calls
import asyncio
import aiohttp

class AsyncArxivHarvester:
    async def search_async(self, query: str) -> list[Paper]:
        # Implement async search
```

### 2. Add Response Caching
```python
# src/lit_review/common/cache.py
from functools import lru_cache
import hashlib

class ResponseCache:
    @lru_cache(maxsize=1000)
    def get_cached_response(self, url: str, params: dict):
        # Cache API responses
```

### 3. Optimize DataFrame Operations
- Use `pd.concat()` instead of appending in loops
- Implement chunked processing for large datasets
- Add progress bars for long operations

## 📊 Priority 5: Monitoring & Observability

### 1. Add Structured Logging
```python
import structlog

logger = structlog.get_logger()
logger.info("api_call",
    service="arxiv",
    endpoint="search",
    query=query,
    results=len(papers)
)
```

### 2. Add Metrics Collection
```python
from prometheus_client import Counter, Histogram

api_calls = Counter('harvester_api_calls_total', 'Total API calls')
api_duration = Histogram('harvester_api_duration_seconds', 'API call duration')
```

### 3. Create Health Dashboard
- API success rates
- Processing times
- Error frequencies
- Resource usage

## 🧪 Additional Testing Needs

### 1. Missing Unit Tests
- `test_semantic_scholar.py`
- `test_crossref.py`
- `test_google_scholar.py`
- `test_base.py`

### 2. Integration Tests
```python
# tests/integration/test_enhanced_pipeline.py
def test_full_pipeline_with_50_papers():
    """Test complete pipeline with enhanced features."""
    # Harvest → Filter → Extract → Visualize
```

### 3. Performance Tests
```python
# tests/performance/test_benchmarks.py
def test_extraction_speed():
    """Benchmark extraction performance."""
    # Measure time for 10, 50, 100 papers
```

## 🎯 Quick Wins (Do Today)

1. **Fix the 11 failing tests** - 2-3 hours
2. **Add docstrings to public methods** - 1 hour
3. **Create basic README for new features** - 30 minutes
4. **Set up pre-commit hooks** - 30 minutes
```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

## 📈 Success Metrics

- ✅ Test coverage maintained above 75%
- ✅ All tests passing in CI/CD
- ✅ Documentation coverage > 90%
- ✅ API response time < 500ms
- ✅ Zero critical bugs in production

## 🏁 Final Checklist

- [ ] Fix all failing tests
- [ ] Document all new features
- [ ] Add remaining unit tests
- [ ] Create integration test suite
- [ ] Set up CI/CD with coverage enforcement
- [ ] Deploy LLM service to production
- [ ] Create user onboarding guide
- [ ] Set up monitoring and alerts
- [ ] Plan v2.0 features

---

**Timeline**:
- Week 1: Fix tests, basic documentation
- Week 2-3: Code cleanup, remaining tests
- Week 4+: Performance optimization, monitoring

**Effort Estimate**: ~40-60 hours total
