# Enhanced Literature Review Pipeline Plan

## 🎯 Objectives

1. **Scale to 50 papers** with robust filtering and processing
2. **Add TeX/HTML extraction** for arXiv papers (faster than PDF)
3. **Model-agnostic LLM integration** using LiteLLM
4. **FastAPI service** for flexible LLM access
5. **Support multiple models** including Google Gemini
6. **Comprehensive testing** at each enhancement

## 📋 Task Breakdown

### Phase 1: Enhanced Data Extraction
- [ ] Add arXiv TeX/HTML source fetching
- [ ] Create TeX parser for paper content
- [ ] Implement HTML extraction as fallback
- [ ] Add abstract keyword filtering

### Phase 2: Model-Agnostic LLM Service
- [ ] Install and configure LiteLLM
- [ ] Create FastAPI service wrapper
- [ ] Add support for multiple models:
  - OpenAI GPT-3.5/4
  - Google Gemini Pro
  - Anthropic Claude
  - Local models (optional)
- [ ] Implement retry logic and fallbacks

### Phase 3: Pipeline Enhancement
- [ ] Update configuration for 50+ papers
- [ ] Add progress tracking
- [ ] Implement batch processing
- [ ] Enhanced error handling
- [ ] Memory optimization

### Phase 4: Testing & Validation
- [ ] Unit tests for new components
- [ ] Integration tests for LLM service
- [ ] Performance testing with 50 papers
- [ ] Validate all extraction methods

## 🏗️ Architecture Design

### 1. Enhanced Harvester Architecture
```
SearchHarvester
├── ArXivHarvester (enhanced)
│   ├── search() - existing
│   ├── fetch_tex_source() - NEW
│   └── fetch_html_source() - NEW
├── SemanticScholarHarvester
└── CrossrefHarvester
```

### 2. LLM Service Architecture
```
FastAPI Service
├── /extract endpoint
├── LiteLLM Integration
│   ├── OpenAI
│   ├── Gemini
│   └── Claude
└── Response caching
```

### 3. Processing Flow
```
1. Search → 50 papers
2. Filter by abstract keywords
3. Fetch content (priority order):
   - TeX source (fastest)
   - HTML source
   - PDF (fallback)
4. Extract via LLM service
5. Tag and analyze
6. Generate visualizations
```

## 💡 Key Enhancements

### 1. Abstract Keyword Filtering
```python
keyword_filters = {
    "include": ["wargame", "LLM", "GPT", "simulation"],
    "exclude": ["medical", "biology"],
    "min_matches": 2
}
```

### 2. TeX/HTML Extraction
- arXiv provides `.tex` source files
- Much faster than PDF parsing
- Better structure preservation
- Direct LaTeX parsing

### 3. LiteLLM Configuration
```python
models = {
    "primary": "gemini/gemini-pro",
    "fallback": "gpt-3.5-turbo",
    "local": "ollama/llama2"
}
```

### 4. Batch Processing
- Process papers in batches of 5-10
- Parallel LLM calls
- Progress tracking
- Checkpoint saving

## 📊 Expected Improvements

| Metric | Current | Enhanced |
|--------|---------|----------|
| Papers processed | 5 | 50+ |
| Processing time | 2 min | 15-20 min |
| PDF parsing | 100% | 20% (TeX/HTML preferred) |
| Model flexibility | OpenAI only | 5+ models |
| Error recovery | Basic | Advanced |

## 🚀 Implementation Steps

### Step 1: Install Dependencies
```bash
pip install litellm fastapi uvicorn beautifulsoup4 pylatexenc
```

### Step 2: Create LLM Service
- FastAPI application
- LiteLLM integration
- Model configuration
- Endpoint implementation

### Step 3: Enhance Harvesters
- Add TeX/HTML fetching
- Implement keyword filtering
- Update paper extraction

### Step 4: Update Pipeline
- Batch processing logic
- Progress tracking
- Enhanced error handling

### Step 5: Comprehensive Testing
- Unit tests for each component
- Integration tests
- Performance benchmarks
- End-to-end validation

## 🔍 Testing Strategy

### Unit Tests
- TeX parsing functions
- HTML extraction
- Keyword filtering
- LLM service endpoints

### Integration Tests
- Full pipeline with 10 papers
- Model switching
- Error scenarios
- Performance limits

### System Tests
- 50 paper processing
- Multiple source types
- Various models
- Complete workflow

## 📈 Success Metrics

1. **Throughput**: Process 50 papers successfully
2. **Performance**: <30 seconds per paper average
3. **Quality**: >90% extraction success rate
4. **Flexibility**: Support 3+ LLM providers
5. **Reliability**: <5% failure rate with retry

## 🎯 Deliverables

1. Enhanced harvester with TeX/HTML support
2. FastAPI LLM service with LiteLLM
3. Updated pipeline for 50+ papers
4. Comprehensive test suite
5. Performance benchmarks
6. Documentation updates
