# 📚 WordplayWorkshop2025 Project Index

## 🎯 Overview
**Purpose**: Systematic literature review pipeline for LLM-powered wargaming research
**Domain**: Academic research automation
**Stack**: Python 3.13+, OpenAI GPT-4, Multi-source harvesting
**Status**: Active development, comprehensive test coverage

## 🏗️ Architecture

### Core Components
```
src/lit_review/
├── harvesters/     # 📥 Paper discovery (Scholar/arXiv/CrossRef/S2)
├── processing/     # ⚙️ Data cleaning, PDF handling, normalization
├── extraction/     # 🤖 LLM-powered information extraction
├── analysis/       # 📊 Failure detection, metrics generation
├── visualization/  # 📈 Charts, trends, analysis outputs
└── utils/         # 🔧 Config, cache, export, logging
```

### Data Flow
```
harvest → normalize → deduplicate → fetch_pdfs → extract → analyze → visualize → export
   ↓          ↓            ↓            ↓           ↓         ↓          ↓         ↓
raw.csv → cleaned → unique_papers → pdf_cache → extracted → metrics → charts → zenodo
```

## 📦 Key Modules

### Harvesters (`src/lit_review/harvesters/`)
- **SearchHarvester**: Orchestrates multi-source searches
- **ArxivHarvester**: arXiv API integration
- **GoogleScholar**: Scholarly library wrapper
- **CrossRef**: CrossRef API client
- **SemanticScholar**: S2 API integration
- **ProductionHarvester**: Enhanced production-ready harvesting
- **QueryOptimizer**: Query expansion & optimization

### Processing (`src/lit_review/processing/`)
- **Normalizer**: Data cleaning & standardization
- **PDFFetcher**: Multi-source PDF retrieval (incl. Sci-Hub)
- **BatchProcessor**: Parallel processing orchestration
- **ScreenUI**: Excel generation for manual screening

### Extraction (`src/lit_review/extraction/`)
- **LLMExtractor**: OpenAI GPT extraction
- **EnhancedLLMExtractor**: Production-grade extraction
- **Tagger**: Failure mode detection via regex

### Utils (`src/lit_review/utils/`)
- **Config**: YAML configuration management
- **ContentCache**: Intelligent caching system
- **LoggingDB**: SQLite-based logging
- **Exporter**: Zenodo integration & packaging

## 🔌 CLI Commands

### Primary Commands
```bash
harvest      # Search academic databases
prepare-screen # Generate screening Excel
extract      # LLM information extraction
visualise    # Generate analysis charts
export       # Package for sharing
status       # Pipeline monitoring
```

### Advanced Options
- `--parallel/--sequential`: Control execution mode
- `--sources`: Select specific databases
- `--max-results`: Limit per-source results
- `--filter-keywords`: Include filtering
- `--exclude-keywords`: Exclude filtering

## 📊 Configuration

### Search Presets
```yaml
preset1: '"LLM" AND ("wargaming" OR "wargame")'
preset2: '"Large Language Model" AND "strategic game"'
preset3: '"GPT" AND ("military simulation" OR "defense game")'
```

### Failure Vocabularies
- **escalation**: [escalation, nuclear, brinkmanship]
- **bias**: [bias, biased, unfair, skew]
- **hallucination**: [hallucination, confabulate, fabricate]
- **transparency**: [opaque, "black box", unexplainable]

## 🧪 Testing Infrastructure

### Test Categories
- **Unit Tests**: Component isolation (pytest markers)
- **Integration Tests**: Module interactions
- **E2E Tests**: Full pipeline validation
- **Performance Tests**: Load & scale testing

### Coverage Targets
- Unit: ≥80%
- Integration: ≥70%
- Overall: ≥75%

## 📁 Data Structures

### Paper Schema
```json
{
  "title": str,
  "authors": List[str],
  "year": int,
  "abstract": str,
  "doi": Optional[str],
  "url": str,
  "source": str,
  "pdf_url": Optional[str]
}
```

### Extraction Schema
```json
{
  "venue_type": ["conference", "journal", "workshop", "tech-report"],
  "game_type": ["seminar", "matrix", "digital", "hybrid"],
  "llm_family": str,
  "llm_role": str,
  "evaluation_approach": str,
  "failure_modes": List[str]
}
```

## 🚀 Development Workflow

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp config/config.yaml.example config/config.yaml
```

### Quality Checks
```bash
make format      # Black + isort
make lint        # Ruff linting
make type-check  # MyPy validation
make test        # Full test suite
```

### Scripts
- `run_tests.sh`: Categorized test execution
- `setup_dev.sh`: Development environment setup
- `generate_stats.py`: Pipeline statistics
- `inspect_db.py`: Database exploration
- `check_deps.py`: Dependency validation

## 📈 Performance Metrics

### Processing Capacity
- Papers/minute: ~100-200 (harvesting)
- PDFs/minute: ~10-20 (fetching)
- Extractions/minute: ~2-5 (GPT-4)

### Resource Usage
- Memory: ~500MB-2GB typical
- Storage: ~10MB/100 papers
- API Calls: Rate-limited per config

## 🔒 Security & Privacy

### API Key Management
- Environment variables preferred
- Config file fallback
- No keys in version control

### Data Handling
- Local PDF caching
- No PII in logs
- Configurable data retention

## 🎯 Project Goals

### Primary Objectives
1. Automate systematic literature review
2. Ensure reproducible research methods
3. Enable collaborative review processes
4. Generate publication-ready datasets

### Quality Standards
- ✅ Comprehensive test coverage
- ✅ Type safety (MyPy strict)
- ✅ Code formatting (Black/Ruff)
- ✅ Documentation coverage
- ✅ Error handling & recovery

## 📝 Recent Changes
- Enhanced production harvesting capabilities
- Comprehensive test suite implementation
- Git LFS integration for PDF caching
- Performance optimizations
- Documentation improvements

## 🔗 Integration Points

### External Services
- OpenAI API (GPT-3.5/4)
- Google Scholar
- arXiv API
- CrossRef API
- Semantic Scholar API
- Sci-Hub (fallback)
- Zenodo (export)

### File Formats
- Input: CSV, JSON
- Processing: SQLite, YAML
- Output: CSV, Excel, PNG, ZIP
- Documentation: Markdown

## 🚦 Status Indicators
- 🟢 Core pipeline: Stable
- 🟢 Testing: Comprehensive
- 🟡 Documentation: Expanding
- 🟢 Performance: Optimized
- 🟢 Error handling: Robust
