# Literature Review Pipeline Testing Strategy

## 🎯 Objective
Systematically test the harvesting, deduplication, and organization components of the literature review pipeline through incremental, small-scale tests to identify and fix issues.

## 📊 Current State
- **Test Suite**: 100% passing (136/136 tests)
- **Code Coverage**: 78.21%
- **Known Limitations**:
  - Google Scholar proxy disabled (rate limiting risk)
  - No real API credentials configured
  - Potential network/authentication issues

## 🧪 Testing Philosophy
1. **Start Small**: Begin with minimal queries (1-5 papers)
2. **Isolate Components**: Test each harvester individually
3. **Incremental Complexity**: Gradually increase scale and complexity
4. **Fail Fast**: Identify issues early with detailed logging
5. **Document Everything**: Track all issues and solutions

## 📋 Testing Phases

### Phase 1: Environment Setup & Configuration
**Goal**: Ensure basic configuration and connectivity

1. **Configuration Validation**
   - Create minimal test config
   - Verify API keys (if available)
   - Test logging system
   - Check file system permissions

2. **Network Connectivity**
   - Test basic HTTP requests
   - Verify API endpoints reachable
   - Check proxy/firewall issues

### Phase 2: Individual Harvester Testing
**Goal**: Test each harvester in isolation with minimal queries

1. **arXiv Harvester** (Start here - most reliable)
   - Query: "LLM wargaming" (expect <10 results)
   - No authentication required
   - Test PDF URL extraction
   - Monitor rate limiting

2. **Crossref Harvester**
   - Query: "artificial intelligence game" (limit to 5)
   - Test DOI extraction
   - Verify metadata completeness

3. **Semantic Scholar** (if API key available)
   - Query: "large language model simulation"
   - Test with/without API key
   - Monitor rate limits

4. **Google Scholar** (Most problematic)
   - Query: "AI wargame" (expect CAPTCHA/blocks)
   - Test fallback behavior
   - Document proxy requirements

### Phase 3: Combined Harvesting
**Goal**: Test multiple harvesters together

1. **Sequential Execution**
   - Run 2 harvesters with same query
   - Monitor memory usage
   - Check data consistency

2. **Parallel Execution**
   - Test thread safety
   - Monitor resource usage
   - Verify result aggregation

### Phase 4: Deduplication Testing
**Goal**: Verify normalizer effectiveness

1. **Artificial Duplicates**
   - Create test dataset with known duplicates
   - Test each dedup method:
     - DOI exact matching
     - Title fuzzy matching
     - arXiv ID matching
     - Content hash

2. **Real-World Duplicates**
   - Use harvested data
   - Verify cross-source deduplication
   - Check false positive rate

### Phase 5: Data Organization
**Goal**: Test data flow and persistence

1. **CSV Export**
   - Verify column consistency
   - Test special character handling
   - Check data types

2. **Database Integration**
   - Test logging database
   - Verify transaction handling
   - Check error recovery

## 🔍 Expected Issues & Mitigations

### 1. API Rate Limiting
- **Detection**: 429 errors, timeouts
- **Mitigation**: Implement exponential backoff, use delays
- **Testing**: Intentionally trigger limits to test handling

### 2. Authentication Failures
- **Detection**: 401/403 errors
- **Mitigation**: Graceful degradation, clear error messages
- **Testing**: Test with invalid/missing credentials

### 3. Network Issues
- **Detection**: Connection errors, timeouts
- **Mitigation**: Retry logic, timeout configuration
- **Testing**: Simulate network failures

### 4. Data Quality Issues
- **Detection**: Missing fields, malformed data
- **Mitigation**: Validation, default values
- **Testing**: Edge cases, empty results

### 5. Memory/Performance
- **Detection**: High memory usage, slow queries
- **Mitigation**: Batch processing, streaming
- **Testing**: Larger datasets gradually

## 🛠️ Testing Tools & Commands

### Basic Test Commands
```bash
# Test single harvester
python -m lit_review harvest --config test_config.yaml --sources arxiv --max-results 5

# Test with specific query
python -m lit_review harvest --query "test query" --sources crossref --max-results 3

# Test deduplication only
python -m lit_review normalize --input test_data.csv --output deduped.csv

# Enable debug logging
python -m lit_review harvest --debug --verbose
```

### Monitoring Commands
```bash
# Watch memory usage
watch -n 1 'ps aux | grep python'

# Monitor network connections
netstat -an | grep ESTABLISHED

# Check log output
tail -f logs/lit_review.log
```

## 📈 Success Metrics

1. **Functionality**
   - Each harvester returns results
   - Deduplication removes >90% of duplicates
   - No data loss during processing

2. **Performance**
   - <5s for 10 paper queries
   - <100MB memory for 100 papers
   - <1s deduplication for 100 papers

3. **Reliability**
   - Graceful handling of all errors
   - Clear error messages
   - Automatic recovery where possible

## 🔄 Iterative Testing Process

1. **Execute Test**
   - Run minimal command
   - Capture all output
   - Monitor resources

2. **Analyze Results**
   - Check for errors
   - Verify data quality
   - Measure performance

3. **Document Issues**
   - Error messages
   - Stack traces
   - Environmental factors

4. **Implement Fix**
   - Code changes
   - Configuration updates
   - Documentation

5. **Verify Fix**
   - Rerun test
   - Confirm resolution
   - Update test suite

## 📝 Issue Tracking Template

```markdown
### Issue: [Brief Description]
**Component**: [Harvester/Normalizer/Config]
**Severity**: [Critical/High/Medium/Low]
**Test Command**: `python -m lit_review ...`

**Error Output**:
```
[paste error here]
```

**Expected Behavior**:
[What should happen]

**Actual Behavior**:
[What actually happened]

**Root Cause**:
[Analysis of why it failed]

**Fix Applied**:
[Code changes or configuration updates]

**Verification**:
[How we confirmed the fix works]
```

## 🚀 Next Steps

1. Create test configuration file
2. Set up test data directory
3. Begin Phase 1 testing
4. Document all findings
5. Iterate through phases systematically
