# Proxy Setup for Google Scholar

## Overview

Google Scholar aggressively rate-limits and blocks automated requests. The `scholarly` library uses proxy servers to avoid these blocks. This document explains how proxy support is implemented in our pipeline.

## Configuration

### Enable/Disable Proxy

Proxy usage is controlled by the `use_proxy` configuration option:

```yaml
# config.yml
development:
  use_proxy: true  # Enable proxy for production
```

For testing, set `use_proxy: false` to avoid network calls during tests.

### Test Configuration

The `RealConfigForTests` class sets `use_proxy = False` by default to prevent test hanging.

## Implementation Details

### GoogleScholarHarvester

The harvester implements lazy proxy initialization:

1. **Conditional Setup**: Only sets up proxy if `config.use_proxy = True`
2. **Lazy Import**: Imports `scholarly` only when needed (not at module level)
3. **FreeProxies**: Uses free proxy rotation with 1-second timeout
4. **Graceful Fallback**: Continues without proxy if setup fails

### Code Structure

```python
class GoogleScholarHarvester:
    def __init__(self, config):
        self.use_proxy = getattr(config, 'use_proxy', True)
        if self.use_proxy:
            self._setup_proxy()

    def _setup_proxy(self):
        from scholarly import ProxyGenerator, scholarly
        pg = ProxyGenerator()
        success = pg.FreeProxies(timeout=1)
        if success:
            scholarly.use_proxy(pg)
```

## Testing

### Known Issue

The `scholarly` library's ProxyGenerator can hang during initialization when it tries to fetch proxy lists from the internet. This causes tests to timeout.

### Solution

1. Set `use_proxy = False` in test configurations
2. Mock `scholarly` in tests that need Google Scholar functionality
3. Use lazy imports to avoid initialization during test collection

### Example Test Mock

```python
def _patch_external_services(self, monkeypatch, fake_services):
    from unittest.mock import MagicMock
    mock_scholarly = MagicMock()
    mock_scholarly.search_pubs = MagicMock(return_value=iter([]))
    monkeypatch.setattr("scholarly.scholarly", mock_scholarly)
```

## Production Usage

For production use:

1. Keep `use_proxy = True`
2. Consider using a paid proxy service (ScraperAPI) for reliability
3. Monitor rate limit errors in logs
4. Implement retry logic for failed requests

## Rate Limits

Without proxy, expect to be blocked after:
- 10-20 queries
- IP ban lasting hours or days

With FreeProxies:
- Variable reliability
- May need to retry failed requests
- Consider fallback to other sources

## Troubleshooting

### Test Hanging

If tests hang:
1. Verify `use_proxy = False` in test config
2. Check for module-level imports of `scholarly`
3. Ensure mocks are applied before imports

### Proxy Failures

If proxy setup fails:
1. Check internet connectivity
2. Try upgrading `fake-useragent`: `pip install --upgrade fake-useragent`
3. Consider using SingleProxy with known working proxy

### Rate Limiting

If getting rate limited despite proxy:
1. Increase delay between requests
2. Use different proxy service
3. Reduce request volume
4. Implement exponential backoff
