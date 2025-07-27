# 🧭 Project Navigation Index

## 📚 Documentation Hub

### 🎯 Getting Started
- [README](../README.md) - Project overview & setup
- [QUICK_START_GUIDE](QUICK_START_GUIDE.md) - Fast track guide
- [DEVELOPER_GUIDE](DEVELOPER_GUIDE.md) - Development workflow

### 🏗️ Architecture & Design
- [PROJECT_INDEX](../PROJECT_INDEX.md) - Comprehensive project structure
- [MODULE_API_REFERENCE](../MODULE_API_REFERENCE.md) - API documentation
- [API_DOCUMENTATION](API_DOCUMENTATION.md) - Detailed API specs
- [CLI_ARCHITECTURE_REDESIGN](reports/CLI_ARCHITECTURE_REDESIGN.md) - CLI design

### 🔬 Research Protocol
- [review_protocol_v0_3](review_protocol_v0_3.md) - Systematic review methodology
- [literature_review_coding_plan](literature_review_coding_plan.md) - Coding strategy
- [SEED_DATA_GUIDE](SEED_DATA_GUIDE.md) - Seed data management

### 🚀 Features & Improvements
- [ENHANCED_PIPELINE_PLAN](ENHANCED_PIPELINE_PLAN.md) - Enhancement roadmap
- [PRODUCTION_IMPROVEMENTS](PRODUCTION_IMPROVEMENTS.md) - Production features
- [SEARCH_IMPROVEMENTS](SEARCH_IMPROVEMENTS.md) - Search enhancements
- [CACHING](CACHING.md) - Caching system design

### 🧪 Testing & Quality
- [TESTING_REPORT](TESTING_REPORT.md) - Test coverage analysis
- [TESTING_ANALYSIS](TESTING_ANALYSIS.md) - Testing strategy
- [TESTING_IMPROVEMENTS](TESTING_IMPROVEMENTS.md) - Test enhancements
- [E2E_TEST_IMPROVEMENTS](E2E_TEST_IMPROVEMENTS.md) - E2E testing

### 📊 Reports & Analysis
- [PROJECT_ANALYSIS_REPORT](reports/PROJECT_ANALYSIS_REPORT.md) - Project analysis
- [PROJECT_IMPROVEMENT_PLAN](reports/PROJECT_IMPROVEMENT_PLAN.md) - Improvement plan
- [ARXIV_API_LIMITATION_ANALYSIS](reports/ARXIV_API_LIMITATION_ANALYSIS.md) - API limitations
- [ENHANCED_PIPELINE_COMPLETE](reports/ENHANCED_PIPELINE_COMPLETE.md) - Completion report

## 🗂️ Quick Links

### Core Modules
- **Harvesters**: `src/lit_review/harvesters/`
  - [arxiv_harvester.py](../src/lit_review/harvesters/arxiv_harvester.py)
  - [google_scholar.py](../src/lit_review/harvesters/google_scholar.py)
  - [production_harvester.py](../src/lit_review/harvesters/production_harvester.py)

- **Processing**: `src/lit_review/processing/`
  - [normalizer.py](../src/lit_review/processing/normalizer.py)
  - [pdf_fetcher.py](../src/lit_review/processing/pdf_fetcher.py)
  - [batch_processor.py](../src/lit_review/processing/batch_processor.py)

- **Extraction**: `src/lit_review/extraction/`
  - [llm_extractor.py](../src/lit_review/extraction/llm_extractor.py)
  - [enhanced_llm_extractor.py](../src/lit_review/extraction/enhanced_llm_extractor.py)

### Configuration
- [config.yaml](../config/config.yaml) - Main configuration
- [pyproject.toml](../pyproject.toml) - Project metadata
- [.gitattributes](../.gitattributes) - Git LFS config

### Scripts & Tools
- [run.py](../run.py) - Main CLI interface
- [run_tests.sh](../scripts/run_tests.sh) - Test runner
- [generate_stats.py](../scripts/generate_stats.py) - Statistics
- [setup_dev.sh](../scripts/setup_dev.sh) - Dev setup

## 🔍 Search Patterns

### Find by Feature
```bash
# Harvesting logic
grep -r "class.*Harvester" src/

# LLM extraction
grep -r "extract.*llm" src/

# PDF handling
grep -r "fetch.*pdf" src/

# Visualization
grep -r "plot_" src/
```

### Find by Technology
```bash
# OpenAI integration
grep -r "openai" src/

# Async operations
grep -r "async def" src/

# Database queries
grep -r "SELECT.*FROM" src/
```

## 📈 Metrics & Status

### Test Coverage
- Unit Tests: `tests/harvesters/`, `tests/processing/`
- Integration: `tests/test_enhanced_pipeline.py`
- E2E Tests: `tests/e2e/`

### Code Quality
- Linting: `make lint`
- Type Check: `make type-check`
- Format: `make format`

### Performance
- Benchmarks: `scripts/generate_stats.py`
- Profiling: See `PRODUCTION_IMPROVEMENTS.md`

## 🎯 Common Tasks

### Development
1. Setup: `./scripts/setup_dev.sh`
2. Test: `./scripts/run_tests.sh`
3. Lint: `make lint`
4. Deploy: See `DEVELOPER_GUIDE.md`

### Research Pipeline
1. Harvest: `python run.py harvest`
2. Screen: `python run.py prepare-screen`
3. Extract: `python run.py extract`
4. Visualize: `python run.py visualise`
5. Export: `python run.py export`

### Maintenance
1. Clean: `./scripts/clean.sh`
2. Stats: `python scripts/generate_stats.py`
3. Logs: `python scripts/inspect_db.py`

## 🔗 External Resources

### APIs Used
- [OpenAI API](https://platform.openai.com/docs)
- [arXiv API](https://arxiv.org/help/api)
- [CrossRef API](https://www.crossref.org/services/metadata-delivery/rest-api/)
- [Semantic Scholar API](https://api.semanticscholar.org/)

### Dependencies
- [scholarly](https://scholarly.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- [click](https://click.palletsprojects.com/)
- [rich](https://rich.readthedocs.io/)
