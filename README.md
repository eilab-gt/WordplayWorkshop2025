# Literature Review Pipeline for LLM-Powered Wargames

A comprehensive pipeline for conducting systematic literature reviews on LLM-powered wargaming research. This tool automates paper discovery, screening, information extraction, and analysis.

## Overview

This pipeline implements the systematic review protocol defined in `review_protocol_v0_3.md` for studying LLM-powered wargames. It provides:

- **Multi-source paper harvesting** from Google Scholar, arXiv, Semantic Scholar, and Crossref
- **Intelligent deduplication** using DOI matching and fuzzy title comparison
- **PDF fetching** with fallback strategies including Sci-Hub
- **LLM-powered extraction** of key information using OpenAI GPT-4
- **Failure mode detection** using regex patterns
- **Visualization generation** for publication trends and analysis
- **Export packaging** with Zenodo integration

## Project Structure

```
WordplayWorkshop2025/
├── src/lit_review/
│   ├── harvesters/       # Paper discovery modules
│   ├── processing/       # Data cleaning and PDF handling
│   ├── extraction/       # LLM extraction and tagging
│   ├── analysis/         # Failure detection and metrics
│   ├── visualization/    # Chart generation
│   └── utils/           # Configuration and utilities
├── data/
│   ├── raw/             # Harvested papers
│   ├── processed/       # Screening progress
│   ├── extracted/       # Extraction results
│   └── templates/       # Data structure templates
├── outputs/             # Visualizations and exports
├── pdf_cache/           # Downloaded PDFs
├── logs/                # SQLite logs
├── tests/               # Comprehensive test suite
├── notebooks/           # Jupyter notebooks
├── scripts/             # Utility scripts
├── config/              # Configuration files
└── run.py              # CLI interface
```

## Installation

### Prerequisites

- Python 3.13+
- [UV](https://github.com/astral-sh/uv) package manager
- OpenAI API key (for LLM extraction)
- Optional: Semantic Scholar API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd WordplayWorkshop2025
```

2. Create virtual environment with UV:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Copy and configure settings:
```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your API keys and preferences
```

## Quick Start

### 1. Harvest Papers

Search for papers across multiple academic databases:

```bash
python run.py harvest --query preset1
```

Options:
- `--query`: Use preset queries or provide custom search string
- `--sources`: Specify sources (default: all enabled in config)
- `--max-results`: Maximum results per source (default: 100)
- `--parallel/--no-parallel`: Enable parallel searching

### 2. Prepare Screening Sheet

Generate Excel file for manual screening:

```bash
python run.py prepare-screen --input data/raw/papers_raw.csv
```

The Excel file includes:
- Paper metadata and abstracts
- Screening decision columns
- Data validation for include/exclude decisions
- Statistics and instructions sheets

### 3. Extract Information

Use LLM to extract structured information:

```bash
python run.py extract --input data/processed/screening_progress.csv
```

Extracts:
- Venue type (conference, journal, workshop, tech-report)
- Game type (seminar, matrix, digital, hybrid)
- Open-ended vs quantitative classification
- LLM family and role
- Evaluation metrics
- Failure modes

### 4. Generate Visualizations

Create charts and analysis:

```bash
python run.py visualise --input data/extracted/extraction.csv
```

Generates:
- Publication timeline
- Venue distribution
- Failure modes frequency
- LLM families usage
- Game types distribution
- Creative-Analytical Scale distribution (1-7)

### 5. Export Dataset

Package results for sharing:

```bash
python run.py export \
    --papers data/raw/papers_raw.csv \
    --extraction data/extracted/extraction.csv
```

Creates a ZIP package with:
- All data files (CSV format)
- Visualizations
- README and metadata
- Optional: Upload to Zenodo for DOI

## Configuration

Edit `config/config.yaml` to customize:

### Search Settings
```yaml
search:
  queries:
    preset1: '"LLM" AND ("wargaming" OR "wargame")'
    preset2: '"Large Language Model" AND "strategic game"'
  sources:
    google_scholar:
      enabled: true
      max_results: 100
```

### API Keys
```yaml
api_keys:
  openai: ${OPENAI_API_KEY}  # Can use environment variables
  semantic_scholar: your-key-here
```

### Failure Vocabularies
```yaml
failure_vocabularies:
  escalation: [escalation, nuclear, brinkmanship]
  bias: [bias, biased, unfair, skew]
  hallucination: [hallucination, confabulate, fabricate]
```

## Advanced Usage

### Using Different LLM Models

Configure in `config/config.yaml`:
```yaml
extraction:
  model: gpt-4  # or gpt-3.5-turbo
  temperature: 0.3
  max_tokens: 4000
```

### Custom Search Queries

```bash
python run.py harvest --query '"transformer model" AND "military simulation"'
```

### Selective Source Harvesting

```bash
python run.py harvest --sources arxiv,crossref --max-results 50
```

### Monitoring Pipeline Status

```bash
python run.py status
```

Shows:
- Log summary by level
- Recent activity
- Error tracking

### Running Tests

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test categories
./scripts/run_tests.sh unit
./scripts/run_tests.sh fast
```

## Development

### Running Tests with Make

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run tests verbosely
make test-verbose
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run all quality checks
make all
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically before each commit:

```bash
make pre-commit
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the virtual environment
2. **API rate limits**: Configure rate limits in `config/config.yaml`
3. **PDF download failures**: Check internet connection and try Sci-Hub mirrors
4. **LLM extraction errors**: Verify OpenAI API key and quota

### Debug Mode

Enable detailed logging:
```bash
python run.py --debug harvest
```

### Database Logs

Query the SQLite log database:
```python
from lit_review.utils import LoggingDatabase
db = LoggingDatabase('logs/logging.db')
errors = db.query_logs(level='ERROR')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `./scripts/run_tests.sh`
4. Submit pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{llm_wargame_review,
  title = {Literature Review Pipeline for LLM-Powered Wargames},
  author = {Your Name},
  year = {2024},
  url = {repository-url}
}
```

## License

[Specify your license here]

## Acknowledgments

This pipeline was developed to support systematic reviews of LLM-powered wargaming research. Special thanks to the developers of the scholarly, arxiv, and other libraries that make this tool possible.
