# API Documentation

This document describes how to use the literature review pipeline modules programmatically.

## Table of Contents

1. [SearchHarvester](#searchharvester)
2. [Normalizer](#normalizer)
3. [PDFFetcher](#pdffetcher)
4. [ScreenUI](#screenui)
5. [LLMExtractor](#llmextractor)
6. [Tagger](#tagger)
7. [Visualizer](#visualizer)
8. [Exporter](#exporter)
9. [Config](#config)
10. [LoggingDatabase](#loggingdatabase)

## SearchHarvester

The SearchHarvester module combines searches across multiple academic databases.

### Basic Usage

```python
from lit_review.harvesters import SearchHarvester

# Initialize with config file
harvester = SearchHarvester('config/config.yaml')

# Search all enabled sources
papers_df = harvester.search_all(
    query='LLM wargaming',
    max_results_per_source=100,
    parallel=True
)

# Search specific sources only
papers_df = harvester.search_all(
    sources=['arxiv', 'crossref'],
    max_results_per_source=50
)
```

### Individual Source Methods

```python
# Search Google Scholar
gs_df = harvester.search_google_scholar("LLM wargaming", max_results=100)

# Search arXiv
arxiv_df = harvester.search_arxiv("LLM wargaming", max_results=50)

# Search Semantic Scholar
ss_df = harvester.search_semantic_scholar("LLM wargaming", max_results=50)

# Search Crossref
cr_df = harvester.search_crossref("LLM wargaming", max_results=50)
```

### Output Format

Returns a pandas DataFrame with columns:
- `title`: Paper title
- `authors`: Semicolon-separated author list
- `year`: Publication year
- `abstract`: Paper abstract
- `source_db`: Source database
- `url`: Paper URL
- `doi`: DOI if available
- `arxiv_id`: arXiv ID if applicable
- `venue`: Publication venue
- `citations`: Citation count
- `pdf_url`: Direct PDF URL if available
- `keywords`: Paper keywords

## Normalizer

The Normalizer handles deduplication and data cleaning.

### Basic Usage

```python
from lit_review.processing import Normalizer

normalizer = Normalizer('config/config.yaml')

# Normalize and deduplicate papers
normalized_df = normalizer.normalize(papers_df)
```

### Features

- **DOI-based deduplication**: Removes exact duplicates
- **Fuzzy title matching**: Catches near-duplicates (configurable threshold)
- **Author normalization**: Standardizes author name formats
- **Screening ID generation**: Creates unique IDs (SCREEN_0001, etc.)
- **Missing data handling**: Cleans incomplete records

### Configuration

```python
# Adjust fuzzy matching threshold
normalizer.fuzzy_threshold = 90  # More strict matching
```

## PDFFetcher

Downloads PDFs from various sources with fallback strategies.

### Basic Usage

```python
from lit_review.processing import PDFFetcher

fetcher = PDFFetcher('config/config.yaml')

# Batch download PDFs
updated_df = fetcher.fetch_batch(
    screening_df,
    max_workers=4  # Parallel downloads
)
```

### Download Sources

1. **Direct PDF URLs**: From paper metadata
2. **arXiv**: Using arXiv IDs
3. **DOI resolution**: Via Unpaywall and Sci-Hub
4. **Publisher websites**: Following redirects

### Output

Updates DataFrame with:
- `pdf_path`: Local path to downloaded PDF
- `pdf_status`: Download status (downloaded_direct, downloaded_arxiv, not_found, etc.)

## ScreenUI

Generates Excel screening sheets for manual review.

### Basic Usage

```python
from lit_review.processing import ScreenUI

screen_ui = ScreenUI('config/config.yaml')

# Generate screening sheet
excel_path = screen_ui.generate_sheet(
    normalized_df,
    output_path='screening.xlsx'
)

# Load screening progress
progress_df = screen_ui.load_progress('screening_completed.xlsx')
```

### Excel Features

- **Screening sheet**: Paper metadata with decision columns
- **Data validation**: Dropdown lists for include/exclude
- **Conditional formatting**: Visual highlighting
- **Instructions sheet**: Screening guidelines
- **Statistics sheet**: Progress tracking

### Screening Columns

- `include_ta`: Title/abstract screening decision
- `reason_ta`: Exclusion reason code
- `notes_ta`: Screening notes
- `include_ft`: Full-text screening decision
- `reason_ft`: Full-text exclusion reason
- `notes_ft`: Full-text notes

## LLMExtractor

Extracts structured information using OpenAI's GPT models.

### Basic Usage

```python
from lit_review.extraction import LLMExtractor

extractor = LLMExtractor('config/config.yaml')

# Extract from batch
extraction_df = extractor.extract_batch(
    screening_df,
    max_workers=3  # Parallel API calls
)

# Extract from single paper
result = extractor.extract_single(paper_series)
```

### Extraction Schema

```python
{
    "venue_type": "conference|journal|workshop|tech-report",
    "game_type": "seminar|matrix|digital|hybrid",
    "open_ended": "yes|no",
    "quantitative": "yes|no",
    "llm_family": "GPT-4|Claude|Llama-70B|...",
    "llm_role": "player|generator|analyst",
    "eval_metrics": "free text",
    "failure_modes": ["escalation", "bias", ...],
    "code_release": "github.com/... or none",
    "grey_lit_flag": true|false
}
```

### AWScale Calculation

Automatically calculates Creative ↔ Analytical scale (1-7) based on:
- Seminar/Red Team games: 1-2 (Ultra/Strongly Creative)
- Matrix/Tabletop games: 3 (Moderately Creative)
- Mixed approaches: 4 (Balanced)
- Course of Action games: 5 (Moderately Analytical)
- Digital/Computer games: 6-7 (Strongly/Ultra Analytical)

## Tagger

Performs regex-based failure mode and metadata detection.

### Basic Usage

```python
from lit_review.analysis import Tagger

tagger = Tagger('config/config.yaml')

# Tag failure modes and detect patterns
tagged_df = tagger.tag_failures(df)
```

### Detection Features

- **Failure modes**: Based on configured vocabularies
- **LLM families**: GPT-4, Claude, Llama, etc.
- **Game types**: Seminar, matrix, digital, hybrid
- **Evaluation metrics**: Win rate, human evaluation, etc.
- **Code availability**: GitHub URLs

### Output Columns

- `failure_modes_regex`: Pipe-separated failure modes
- `llm_detected`: Detected LLM family
- `game_type_detected`: Detected game type
- `metrics_detected`: Detected evaluation metrics
- `code_detected`: Detected code URLs

## Visualizer

Creates publication analysis charts.

### Basic Usage

```python
from lit_review.analysis import Visualizer

viz = Visualizer('config/config.yaml')

# Generate all charts
chart_paths = viz.generate_all_charts(extraction_df)

# Generate specific charts
timeline_path = viz.create_timeline(df, 'timeline.png')
venue_path = viz.create_venue_distribution(df, 'venues.png')
```

### Available Charts

1. **Timeline**: Publications by year
2. **Venue Distribution**: Conference vs journal vs workshop
3. **Failure Modes**: Frequency of each failure type
4. **LLM Families**: Distribution of models used
5. **Game Types**: Types of wargames
6. **AWScale**: Creative ↔ Analytical distribution (1-7 scale)

### Customization

```python
# Customize chart appearance
viz.viz_config['charts']['timeline']['figsize'] = [12, 8]
viz.viz_config['charts']['timeline']['style'] = 'ggplot'
```

## Exporter

Creates dataset packages for sharing and archival.

### Basic Usage

```python
from lit_review.utils import Exporter

exporter = Exporter('config/config.yaml')

# Create export package
package_path = exporter.create_package(
    papers_df=papers_df,
    extraction_df=extraction_df,
    viz_paths=chart_paths,
    output_path='dataset.zip',
    custom_metadata={'version': '1.0'}
)

# Upload to Zenodo (if configured)
doi = exporter.upload_to_zenodo(package_path)
```

### Package Contents

```
dataset.zip/
├── data/
│   ├── papers_raw.csv
│   ├── screening_progress.csv
│   └── extraction_results.csv
├── visualizations/
│   ├── timeline.png
│   ├── venue_distribution.png
│   └── ...
├── README.md
└── metadata.json
```

## Config

Configuration management with environment variable support.

### Basic Usage

```python
from lit_review.utils import Config, load_config

# Load configuration
config = load_config('config/config.yaml')

# Access configuration values
api_key = config.openai_key
search_years = config.search_years
failure_vocab = config.failure_vocab
```

### Environment Variables

Use `${VAR_NAME}` syntax in config/config.yaml:

```yaml
api_keys:
  openai: ${OPENAI_API_KEY}
```

## LoggingDatabase

SQLite-based logging for debugging and monitoring.

### Basic Usage

```python
from lit_review.utils import LoggingDatabase, setup_db_logging

# Setup logging
db_handler = setup_db_logging('logs.db', level=logging.INFO)

# Query logs
db = LoggingDatabase('logs.db')

# Get all errors
errors = db.query_logs(level='ERROR')

# Get logs from specific time range
recent_logs = db.query_logs(
    start_time='2024-01-01T00:00:00',
    end_time='2024-01-02T00:00:00'
)

# Get summary statistics
summary = db.get_summary()
print(f"Total logs: {summary['total_logs']}")
print(f"Errors: {summary['by_level']['ERROR']}")
```

### Log Schema

```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    level TEXT,
    logger TEXT,
    message TEXT,
    extra TEXT  -- JSON field for additional data
);
```

## Complete Pipeline Example

```python
from lit_review.harvesters import SearchHarvester
from lit_review.processing import Normalizer, PDFFetcher, ScreenUI
from lit_review.extraction import LLMExtractor
from lit_review.analysis import Tagger, Visualizer
from lit_review.utils import Exporter, load_config

# Load configuration
config = load_config('config/config.yaml')

# 1. Harvest papers
harvester = SearchHarvester(config)
papers_df = harvester.search_all(max_results_per_source=100)

# 2. Normalize and deduplicate
normalizer = Normalizer(config)
normalized_df = normalizer.normalize(papers_df)

# 3. Generate screening sheet
screen_ui = ScreenUI(config)
screen_ui.generate_sheet(normalized_df, 'screening.xlsx')

# ... Manual screening ...

# 4. Load screened papers and fetch PDFs
screened_df = screen_ui.load_progress('screening_completed.xlsx')
fetcher = PDFFetcher(config)
pdf_df = fetcher.fetch_batch(screened_df)

# 5. Extract information
extractor = LLMExtractor(config)
extraction_df = extractor.extract_batch(pdf_df)

# 6. Tag failure modes
tagger = Tagger(config)
tagged_df = tagger.tag_failures(extraction_df)

# 7. Generate visualizations
viz = Visualizer(config)
chart_paths = viz.generate_all_charts(tagged_df)

# 8. Export dataset
exporter = Exporter(config)
package_path = exporter.create_package(
    papers_df=papers_df,
    extraction_df=tagged_df,
    viz_paths=chart_paths
)

print(f"Dataset package created: {package_path}")
```
