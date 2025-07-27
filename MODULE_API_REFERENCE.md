# 🔌 Module API Reference

## 🎯 Core APIs

### SearchHarvester
```python
class SearchHarvester:
    """Multi-source paper discovery orchestrator"""

    def search_all(sources: List[str], max_results: int, parallel: bool) → DataFrame
    def normalize_results() → DataFrame
    def deduplicate(threshold: float = 0.85) → DataFrame
```

### ProductionHarvester
```python
class ProductionHarvester(SearchHarvester):
    """Enhanced production-ready harvesting"""

    def harvest_with_cache(query: str, cache_ttl: int) → DataFrame
    def batch_harvest(queries: List[str]) → DataFrame
    def validate_results() → Dict[str, Any]
```

### LLMExtractor
```python
class LLMExtractor:
    """OpenAI GPT-based information extraction"""

    def extract(text: str, schema: Dict) → Dict
    def batch_extract(texts: List[str]) → List[Dict]
    def validate_extraction(result: Dict) → bool
```

### EnhancedLLMExtractor
```python
class EnhancedLLMExtractor(LLMExtractor):
    """Production-grade extraction with retry logic"""

    def extract_with_retry(text: str, max_retries: int = 3) → Dict
    def extract_with_fallback(text: str, fallback_model: str) → Dict
    async def extract_async(texts: List[str]) → List[Dict]
```

### PDFFetcher
```python
class PDFFetcher:
    """Multi-source PDF retrieval"""

    def fetch(url: str, use_scihub: bool = True) → Optional[bytes]
    def fetch_batch(urls: List[str], parallel: bool = True) → Dict[str, bytes]
    def get_pdf_text(pdf_bytes: bytes) → str
```

### ContentCache
```python
class ContentCache:
    """Intelligent caching system"""

    def get(key: str) → Optional[Any]
    def set(key: str, value: Any, ttl: int = 3600) → None
    def invalidate_pattern(pattern: str) → int
    def get_stats() → Dict[str, int]
```

### Normalizer
```python
class Normalizer:
    """Data cleaning & standardization"""

    def normalize_authors(authors: Union[str, List]) → List[str]
    def normalize_title(title: str) → str
    def normalize_date(date: Any) → Optional[int]
    def clean_abstract(abstract: str) → str
```

### BatchProcessor
```python
class BatchProcessor:
    """Parallel processing orchestration"""

    async def process_batch(items: List, processor: Callable, max_workers: int) → List
    def chunk_data(data: List, chunk_size: int) → List[List]
    def merge_results(results: List[DataFrame]) → DataFrame
```

### Visualizer
```python
class Visualizer:
    """Chart & analysis generation"""

    def plot_timeline(df: DataFrame) → Figure
    def plot_venues(df: DataFrame) → Figure
    def plot_failure_modes(df: DataFrame) → Figure
    def generate_report(df: DataFrame, output_dir: Path) → None
```

### LoggingDatabase
```python
class LoggingDatabase:
    """SQLite-based structured logging"""

    def log(level: str, message: str, metadata: Dict) → None
    def query_logs(level: str = None, limit: int = 100) → List[Dict]
    def get_summary() → Dict[str, int]
    def export_logs(format: str = "csv") → Path
```

## 📊 Data Models

### Paper
```python
@dataclass
class Paper:
    title: str
    authors: List[str]
    year: int
    abstract: str
    doi: Optional[str]
    url: str
    source: str
    pdf_url: Optional[str]
    metadata: Dict[str, Any]
```

### ExtractionResult
```python
@dataclass
class ExtractionResult:
    paper_id: str
    venue_type: str
    game_type: str
    llm_family: str
    llm_role: str
    evaluation_approach: str
    failure_modes: List[str]
    confidence: float
    raw_response: str
```

### HarvestResult
```python
@dataclass
class HarvestResult:
    query: str
    source: str
    papers: List[Paper]
    timestamp: datetime
    errors: List[str]
    stats: Dict[str, int]
```

## 🔧 Configuration Objects

### HarvesterConfig
```python
class HarvesterConfig:
    sources: List[str]
    max_results_per_source: int
    rate_limits: Dict[str, float]
    parallel: bool
    timeout: int
```

### ExtractionConfig
```python
class ExtractionConfig:
    model: str  # gpt-4, gpt-3.5-turbo
    temperature: float
    max_tokens: int
    retry_attempts: int
    fallback_model: Optional[str]
```

### CacheConfig
```python
class CacheConfig:
    backend: str  # memory, redis, disk
    ttl: int
    max_size: int
    eviction_policy: str
```

## 🎨 Utility Functions

### Query Building
```python
def expand_query(query: str, synonyms: Dict[str, List[str]]) → List[str]
def build_boolean_query(terms: List[str], operators: List[str]) → str
def validate_query_syntax(query: str) → bool
```

### Data Validation
```python
def validate_paper(paper: Dict) → bool
def validate_doi(doi: str) → bool
def validate_extraction(result: Dict, schema: Dict) → bool
```

### Text Processing
```python
def extract_keywords(text: str, n: int = 10) → List[str]
def summarize_abstract(abstract: str, max_length: int = 200) → str
def detect_language(text: str) → str
```

## 🔌 Extension Points

### Custom Harvesters
```python
class CustomHarvester(BaseHarvester):
    """Extend for new data sources"""

    def search(self, query: str, max_results: int) → List[Paper]:
        # Implement source-specific logic
        pass
```

### Custom Extractors
```python
class CustomExtractor(BaseExtractor):
    """Extend for different extraction approaches"""

    def extract(self, text: str) → Dict:
        # Implement extraction logic
        pass
```

### Export Formats
```python
class CustomExporter(BaseExporter):
    """Extend for new export formats"""

    def export(self, data: DataFrame, output_path: Path) → None:
        # Implement export logic
        pass
```

## 🚀 Async APIs

### Async Harvesting
```python
async def harvest_async(queries: List[str]) → List[HarvestResult]
async def fetch_pdfs_async(urls: List[str]) → Dict[str, bytes]
async def extract_batch_async(texts: List[str]) → List[ExtractionResult]
```

### Stream Processing
```python
async def stream_papers(source: AsyncIterator[Paper]) → None
async def process_stream(stream: AsyncIterator, processor: Callable) → None
```

## 📡 Event Hooks

### Pipeline Events
```python
on_harvest_start(query: str, sources: List[str])
on_harvest_complete(results: List[Paper])
on_extraction_start(paper: Paper)
on_extraction_complete(result: ExtractionResult)
on_error(error: Exception, context: Dict)
```

### Progress Callbacks
```python
def register_progress_callback(callback: Callable[[float, str], None])
def update_progress(percent: float, message: str)
```
