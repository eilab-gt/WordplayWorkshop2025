# DEPRECATED - Do Not Use These Scripts

**⚠️ WARNING: These scripts are deprecated and should NOT be used.**

## Why These Scripts Are Deprecated

The individual scripts in this directory led to inconsistent pipeline execution, including:
- Running with limited sources (e.g., only arxiv + semantic_scholar instead of all 4 sources)
- Inconsistent configuration handling
- No proper config tracking with outputs
- Fragmented pipeline execution

## Use the Main Module Instead

The literature review pipeline should be run using the main module:

```bash
# Run complete pipeline with ALL configured sources
python -m src.lit_review run

# Run individual stages
python -m src.lit_review harvest
python -m src.lit_review fetch
python -m src.lit_review extract
python -m src.lit_review visualize
python -m src.lit_review export

# Run with options
python -m src.lit_review run --max-per-source 500 --output-dir results/
python -m src.lit_review harvest --sources arxiv semantic_scholar google_scholar crossref
```

## Benefits of Using the Main Module

1. **Consistent source usage**: Uses ALL configured sources by default
2. **Config tracking**: Automatically saves config snapshot with outputs
3. **Single entry point**: No confusion about which script to run
4. **Proper CLI**: Clean command-line interface with help
5. **Reproducibility**: Every run is tracked with config and timestamps

## Migration Guide

| Old Script | New Command |
|------------|-------------|
| `scripts/production_harvest.py` | `python -m src.lit_review harvest` |
| `scripts/test_v3_full_pipeline.py` | `python -m src.lit_review run` |
| Various test scripts | Use the main module with appropriate options |

## Scripts to Keep

Only utility and development scripts should remain:
- `check_deps.py` - Dependency checking
- `setup_dev.sh` - Development environment setup
- `run_tests.sh` - Test runner
- Other pure utility scripts

Pipeline execution scripts should be removed or clearly marked as deprecated.