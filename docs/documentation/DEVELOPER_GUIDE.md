# Developer Guide

This guide explains how to extend and customize the literature review pipeline.

## Architecture Overview

The pipeline follows a modular architecture with clear separation of concerns:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Harvesters │────▶│ Processing  │────▶│ Extraction  │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Analysis   │────▶│ Visualization│
                    └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Export    │
                                        └─────────────┘
```

## Adding a New Paper Source

To add a new academic database or paper source:

### 1. Create a Harvester Class

Create a new file `src/lit_review/harvesters/newsource_harvester.py`:

```python
from typing import Dict, List, Optional
import pandas as pd
from .base import BaseHarvester

class NewSourceHarvester(BaseHarvester):
    """Harvester for NewSource database."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api.newsource.com/v1"

    def search(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """Search NewSource for papers matching query."""
        papers = []

        # Implement API calls here
        response = self._make_request(query, max_results)

        # Parse results
        for item in response['results']:
            paper = {
                'title': item['title'],
                'authors': '; '.join(item['authors']),
                'year': item['year'],
                'abstract': item['abstract'],
                'doi': item.get('doi', ''),
                'url': item['url'],
                'source_db': 'newsource'
            }
            papers.append(paper)

        return pd.DataFrame(papers)

    def _make_request(self, query: str, limit: int) -> Dict:
        """Make API request to NewSource."""
        # Implement request logic
        pass
```

### 2. Update SearchHarvester

Add your harvester to `src/lit_review/harvesters/search_harvester.py`:

```python
from .newsource_harvester import NewSourceHarvester

class SearchHarvester:
    def __init__(self, config_path: str):
        # ... existing code ...

        # Add new source
        if self.config.get('search.sources.newsource.enabled', False):
            api_key = self.config.get('api_keys.newsource')
            self.newsource = NewSourceHarvester(api_key)

    def search_newsource(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """Search NewSource database."""
        if not hasattr(self, 'newsource'):
            logger.warning("NewSource not configured")
            return pd.DataFrame()

        try:
            return self.newsource.search(query, max_results)
        except Exception as e:
            logger.error(f"NewSource search failed: {e}")
            return pd.DataFrame()
```

### 3. Update Configuration

Add to `config/config.yaml.example`:

```yaml
search:
  sources:
    newsource:
      enabled: true
      max_results: 100
      rate_limit: 1

api_keys:
  newsource: ${NEWSOURCE_API_KEY}
```

## Adding New Extraction Fields

To extract additional information from papers:

### 1. Update Extraction Schema

Modify `src/lit_review/extraction/llm_extractor.py`:

```python
def _create_extraction_prompt(self) -> str:
    return """
    Extract the following information:
    - venue_type: conference, journal, workshop, or tech-report
    - game_type: seminar, matrix, digital, or hybrid
    - NEW_FIELD: description of what to extract
    ... rest of prompt ...
    """
```

### 2. Update Data Structures

Add to `data/templates/extraction_schema.json`:

```json
{
  "properties": {
    "new_field": {
      "type": "string",
      "description": "Description of new field",
      "enum": ["option1", "option2", "option3"]
    }
  }
}
```

### 3. Update Visualizations

Add visualization for new field in `src/lit_review/visualization/visualizer.py`:

```python
def create_new_field_chart(self, df: pd.DataFrame, output_path: Path) -> Path:
    """Create chart for new field distribution."""
    if 'new_field' not in df.columns:
        logger.warning("new_field column not found")
        return None

    # Count occurrences
    counts = df['new_field'].value_counts()

    # Create chart
    plt.figure(figsize=(8, 6))
    counts.plot(kind='bar')
    plt.title('New Field Distribution')
    plt.xlabel('New Field Value')
    plt.ylabel('Count')
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    return output_path
```

## Adding New Failure Modes

### 1. Update Configuration

Add to `config/config.yaml.example`:

```yaml
failure_vocabularies:
  new_failure_mode:
    - keyword1
    - keyword2
    - "multi word phrase"
```

### 2. Update Tagger Patterns

If needed, add custom regex patterns in `src/lit_review/analysis/tagger.py`:

```python
def __init__(self, config_path: str):
    # ... existing code ...

    # Add custom patterns
    self.custom_patterns = {
        'new_pattern': re.compile(
            r'\b(specific|pattern|to|match)\b',
            re.IGNORECASE
        )
    }
```

## Creating Custom Visualizations

### 1. Create Visualization Function

```python
def create_custom_viz(self, df: pd.DataFrame, output_path: Path) -> Path:
    """Create custom visualization."""

    # Data processing
    data = self._process_data_for_viz(df)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot
    ax1.plot(data['x'], data['y'])
    ax1.set_title('Custom Plot 1')

    # Second subplot
    ax2.scatter(data['x'], data['z'])
    ax2.set_title('Custom Plot 2')

    # Styling
    plt.suptitle('Custom Visualization')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
```

### 2. Add to Chart Generation

```python
def generate_all_charts(self, df: pd.DataFrame) -> Dict[str, Path]:
    """Generate all configured charts."""
    charts = {}

    # ... existing charts ...

    if self.viz_config['charts'].get('custom_viz', {}).get('enabled', False):
        output_path = self.output_dir / 'custom_viz.png'
        charts['custom_viz'] = self.create_custom_viz(df, output_path)

    return charts
```

## Testing Your Extensions

### 1. Write Unit Tests

Create `tests/test_newsource.py`:

```python
import pytest
from unittest.mock import Mock, patch
from lit_review.harvesters import NewSourceHarvester

class TestNewSourceHarvester:
    def test_search(self):
        harvester = NewSourceHarvester(api_key='test-key')

        with patch.object(harvester, '_make_request') as mock_request:
            mock_request.return_value = {
                'results': [{
                    'title': 'Test Paper',
                    'authors': ['Author One'],
                    'year': 2024,
                    'abstract': 'Test abstract',
                    'url': 'https://example.com'
                }]
            }

            df = harvester.search('test query', max_results=10)

            assert len(df) == 1
            assert df.iloc[0]['title'] == 'Test Paper'
            assert df.iloc[0]['source_db'] == 'newsource'
```

### 2. Run Tests

```bash
# Run your new tests
pytest tests/test_newsource.py -v

# Run all tests to ensure nothing broke
./scripts/run_tests.sh
```

## Performance Optimization

### 1. Implement Caching

```python
from functools import lru_cache
import hashlib

class CachedHarvester(BaseHarvester):
    def __init__(self):
        super().__init__()
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)

    def search(self, query: str, max_results: int = 100) -> pd.DataFrame:
        # Check cache first
        cache_key = self._get_cache_key(query, max_results)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            return pd.read_pickle(cache_file)

        # Perform search
        results = self._perform_search(query, max_results)

        # Cache results
        results.to_pickle(cache_file)
        return results

    def _get_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key from parameters."""
        content = f"{query}:{max_results}"
        return hashlib.md5(content.encode()).hexdigest()
```

### 2. Implement Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class AsyncHarvester:
    async def search_multiple_async(self, queries: List[str]) -> pd.DataFrame:
        """Search multiple queries asynchronously."""
        tasks = []

        async with aiohttp.ClientSession() as session:
            for query in queries:
                task = self._search_async(session, query)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        # Combine results
        return pd.concat(results, ignore_index=True)
```

## Error Handling Best Practices

### 1. Custom Exceptions

```python
class HarvesterError(Exception):
    """Base exception for harvester errors."""
    pass

class APILimitError(HarvesterError):
    """Raised when API limit is reached."""
    pass

class AuthenticationError(HarvesterError):
    """Raised when authentication fails."""
    pass
```

### 2. Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustHarvester:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _make_request(self, url: str) -> Dict:
        """Make HTTP request with retry logic."""
        response = requests.get(url)

        if response.status_code == 429:
            raise APILimitError("Rate limit exceeded")

        response.raise_for_status()
        return response.json()
```

## Contributing Guidelines

### 1. Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Keep functions focused and under 50 lines

### 2. Documentation

- Update API documentation for new features
- Add examples to the Quick Start Guide
- Document configuration options

### 3. Testing

- Write tests for all new functionality
- Maintain >70% code coverage
- Test edge cases and error conditions

### 4. Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-source`
3. Make changes and test thoroughly
4. Update documentation
5. Submit PR with clear description

## Debugging Tips

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in code
logger.setLevel(logging.DEBUG)
```

### 2. Use Interactive Debugging

```python
import pdb

def problematic_function():
    # ... some code ...
    pdb.set_trace()  # Debugger stops here
    # ... rest of code ...
```

### 3. Inspect Database

```python
from lit_review.utils import LoggingDatabase

# Check recent errors
db = LoggingDatabase('logs/logging.db')
errors = db.query_logs(level='ERROR', limit=10)

for error in errors:
    print(f"{error['timestamp']}: {error['message']}")
    if error['extra']:
        print(f"  Extra: {error['extra']}")
```

## Advanced Configuration

### 1. Plugin System

Create a plugin system for custom components:

```python
# src/lit_review/plugins/base.py
from abc import ABC, abstractmethod

class Plugin(ABC):
    """Base class for pipeline plugins."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe."""
        pass

# src/lit_review/plugins/loader.py
def load_plugins(plugin_dir: Path) -> List[Plugin]:
    """Load all plugins from directory."""
    plugins = []

    for file in plugin_dir.glob("*.py"):
        if file.name.startswith("_"):
            continue

        module = import_module(f"plugins.{file.stem}")

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin):
                plugins.append(obj())

    return plugins
```

### 2. Custom Workflows

Define custom workflows in YAML:

```yaml
# workflows/custom_workflow.yaml
name: "Custom Literature Review"
steps:
  - name: "Harvest"
    module: "harvesters.SearchHarvester"
    method: "search_all"
    params:
      sources: ["arxiv", "semantic_scholar"]
      max_results: 200

  - name: "Filter"
    module: "plugins.CustomFilter"
    method: "filter_by_year"
    params:
      min_year: 2020

  - name: "Extract"
    module: "extraction.LLMExtractor"
    method: "extract_batch"
    params:
      model: "gpt-4"
```

## Deployment Considerations

### 1. Docker Support

Create `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install -e .

# Set up entry point
ENTRYPOINT ["python", "run.py"]
```

### 2. Environment-Specific Configs

```python
# src/lit_review/utils/config.py
def load_config_for_env(env: str = "development") -> Config:
    """Load environment-specific configuration."""
    base_config = load_config("config/config.yaml")
    env_config_path = f"config.{env}.yaml"

    if Path(env_config_path).exists():
        env_config = load_config(env_config_path)
        # Merge configurations
        return merge_configs(base_config, env_config)

    return base_config
```

Happy coding! 🚀
