# Content Caching System

The literature review pipeline now includes a comprehensive content caching system that caches PDF, HTML, and LaTeX/TeX content to avoid redundant downloads and improve performance.

## Overview

The `ContentCache` class provides unified caching for all content types with the following features:

- **Automatic caching** of downloaded PDFs, HTML pages, and TeX sources
- **Cache validation** using MD5 hashes and age checks
- **SQLite metadata tracking** with access statistics
- **Thread-safe concurrent access** with per-item locking
- **CLI commands** for cache management

## How It Works

### Automatic Caching

When the pipeline fetches content (PDFs, HTML, or TeX), it automatically:

1. Checks if the content is already cached
2. Validates the cache (age and file integrity)
3. Returns cached content if valid, or fetches new content if not
4. Stores newly fetched content in the cache

### Cache Organization

Content is organized by type in the cache directory:
```
content_cache/
├── pdfs/         # PDF files
├── html/         # HTML content (text)
├── tex/          # LaTeX/TeX sources
└── content_cache.db  # Metadata database
```

### Cache Keys

Papers are identified using:
- DOI (preferred)
- arXiv ID (for arXiv papers)
- Generated hash (fallback based on title/author/year)

## Configuration

Add these settings to your `config/config.yaml`:

```yaml
# Cache settings
cache_dir: "./content_cache"      # Where to store cached files
cache_max_age_days: 90            # How long to keep cached files
use_cache: true                   # Enable/disable caching
```

## CLI Commands

### View Cache Statistics

```bash
python run.py cache-stats
```

Shows:
- Total cache entries and size
- Cache hit/miss rates
- Time and bandwidth saved
- Breakdown by content type

### Clean Cache

Remove old entries (older than 90 days by default):
```bash
python run.py cache-clean --days 90
```

Remove specific content type:
```bash
python run.py cache-clean --type pdf
python run.py cache-clean --type html
python run.py cache-clean --type tex
```

Remove all cached content:
```bash
python run.py cache-clean --type all
```

Skip confirmation prompts:
```bash
python run.py cache-clean --type all --force
```

## Performance Benefits

The caching system provides significant benefits:

1. **Reduced API calls**: Avoids re-downloading from arXiv, publishers
2. **Faster pipeline runs**: Cached content loads instantly
3. **Bandwidth savings**: Typical PDFs are 1-5MB each
4. **Reliability**: Works offline for cached papers

## Cache Validation

The cache validates content using:

1. **Age check**: Content older than `cache_max_age_days` is re-fetched
2. **Hash validation**: MD5 hash ensures file integrity
3. **File existence**: Handles manually deleted files gracefully

## Thread Safety

The cache supports concurrent access:
- Multiple processes can read from cache simultaneously
- Write operations are synchronized per paper
- Database operations use SQLite's built-in thread safety

## Monitoring

Cache statistics are logged during pipeline runs:
```
Cache hit for pdf content: arxiv:2301.00234
Cached tex content for arxiv:2301.00234 (245.3 KB in 2.1s)
```

View accumulated statistics:
```bash
python run.py cache-stats
```

## Troubleshooting

### Cache not working

1. Check if caching is enabled in config: `use_cache: true`
2. Verify cache directory permissions
3. Check available disk space

### Corrupted cache

If you suspect cache corruption:
```bash
# Clear specific type
python run.py cache-clean --type pdf --force

# Clear everything
python run.py cache-clean --type all --force
```

### Performance issues

For very large caches (>10GB):
1. Clean old entries regularly: `cache-clean --days 30`
2. Consider shorter `cache_max_age_days`
3. Monitor disk usage

## Implementation Details

The caching is integrated at multiple levels:

1. **PDFFetcher**: Uses ContentCache for all PDF downloads
2. **EnhancedLLMExtractor**: Caches HTML and TeX content from arXiv
3. **ContentCache**: Provides unified caching interface

All content types benefit from the same caching infrastructure, validation, and management tools.
