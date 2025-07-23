# Complete Pipeline Testing Plan

## 🎯 Objective
Test the entire literature review pipeline end-to-end with our harvested papers, documenting where papers are filtered out and identifying all failure points.

## 📊 Current State Analysis

### Available Test Data
- **Combined Test**: 5 papers (3 ArXiv + 2 Crossref)
  - All have complete metadata
  - ArXiv papers have PDF URLs
  - Mixed relevance levels

### Pipeline Stages to Test
1. **Harvesting** ✅ Complete - 5 papers available
2. **PDF Fetching** 🔄 To test - Download PDFs from URLs
3. **Screening Sheet** 🔄 To test - Generate Excel for manual review
4. **Extraction** 🔄 To test - LLM-based information extraction
5. **Tagging** 🔄 To test - Failure mode detection
6. **Visualization** 🔄 To test - Generate charts and graphs
7. **Export** 🔄 To test - Package results for distribution

## 🔍 Expected Filtering Points

### 1. PDF Fetching Stage
**Potential Issues:**
- ArXiv PDFs should download successfully
- Crossref papers may lack PDF URLs
- Network timeouts or rate limiting
- File size limits (50MB default)

**Expected Output:** 3-5 PDFs downloaded

### 2. Screening Sheet Stage
**Potential Issues:**
- Excel generation errors
- Missing required columns
- Unicode encoding issues
- Memory constraints with large datasets

**Expected Output:** Excel file with all 5 papers

### 3. Extraction Stage
**Potential Issues:**
- No OpenAI API key → Skip or mock
- Token limits exceeded
- PDF parsing failures
- JSON parsing errors

**Expected Output:** Extracted data for papers with PDFs

### 4. Tagging Stage
**Potential Issues:**
- Pattern matching failures
- Missing text fields
- Regex performance with large texts

**Expected Output:** Tagged papers with failure modes identified

### 5. Visualization Stage
**Potential Issues:**
- Insufficient data for meaningful charts
- Missing required fields
- Matplotlib backend issues

**Expected Output:** Basic charts saved as PNG files

### 6. Export Stage
**Potential Issues:**
- File permissions
- Compression errors
- Missing components

**Expected Output:** ZIP archive with all outputs

## 🧪 Testing Strategy

### Phase 1: Prepare Test Environment
```bash
# Create output directories
mkdir -p test_output/pipeline/{pdfs,screening,extraction,viz,export}

# Use our existing harvested data
cp test_output/results/combined_test.csv test_output/pipeline/input.csv
```

### Phase 2: Sequential Stage Testing

#### Stage 1: PDF Fetching
```bash
python run.py prepare-screen \
  --input test_output/pipeline/input.csv \
  --output test_output/pipeline/screening/papers.xlsx
```
**Monitor:** PDF download success rate, errors, timeouts

#### Stage 2: Manual Screening Simulation
- Mark 3 papers as "include_ft: yes"
- Mark 2 papers as "include_ft: no"
- Add screening notes

#### Stage 3: Extraction
```bash
python run.py extract \
  --input test_output/pipeline/screening/papers.xlsx \
  --output test_output/pipeline/extraction/extracted.csv \
  --skip-llm  # If no API key
```
**Monitor:** Extraction errors, parsing issues

#### Stage 4: Visualization
```bash
python run.py visualise \
  --input test_output/pipeline/extraction/extracted.csv \
  --output-dir test_output/pipeline/viz
```
**Monitor:** Chart generation, missing data handling

#### Stage 5: Export
```bash
python run.py export \
  --input test_output/pipeline/extraction/extracted.csv \
  --output test_pipeline_results
```
**Monitor:** Archive creation, file inclusion

## 📈 Metrics to Track

### Per-Stage Metrics
1. **Input Count**: Papers entering stage
2. **Output Count**: Papers exiting stage
3. **Filter Reason**: Why papers were excluded
4. **Error Count**: Technical failures
5. **Processing Time**: Performance metrics

### Overall Pipeline Metrics
- **Total Throughput**: Papers in vs papers out
- **Stage Efficiency**: Success rate per stage
- **Data Completeness**: Fields populated
- **Quality Score**: Based on extraction success

## 🚨 Issue Documentation Template

```markdown
### Issue: [Stage] - [Brief Description]
**Input**: X papers
**Output**: Y papers
**Filtered**: Z papers (reason)

**Error Details**:
```
[Error message or stack trace]
```

**Root Cause**: [Analysis]
**Fix Applied**: [If any]
**Workaround**: [If applicable]
```

## 🔄 Iterative Testing Process

1. **Run Stage**: Execute with logging
2. **Analyze Output**: Check data quality
3. **Document Issues**: Record all problems
4. **Apply Fixes**: Modify code/config
5. **Re-test**: Verify fixes work
6. **Progress**: Move to next stage

## 📊 Expected Outcomes

### Best Case Scenario
- 5/5 papers processed completely
- 3-4 PDFs downloaded
- All visualizations generated
- Clean export package

### Realistic Scenario
- 3/5 papers fully processed
- 2 PDFs downloaded
- Basic visualizations only
- Some manual intervention needed

### Worst Case Scenario
- Pipeline breaks at extraction
- No LLM processing possible
- Manual workarounds required
- Limited functionality demo

## 🎯 Success Criteria

1. **Pipeline Completion**: All stages execute without fatal errors
2. **Data Integrity**: No data loss between stages
3. **Error Handling**: Graceful failures with clear messages
4. **Documentation**: Complete record of issues and solutions
5. **Reproducibility**: Others can run the same test

## 📝 Next Steps After Testing

1. **Fix Critical Issues**: Address show-stoppers first
2. **Optimize Performance**: Improve slow stages
3. **Enhance Error Handling**: Better recovery mechanisms
4. **Scale Testing**: Increase to 50-100 papers
5. **Production Readiness**: Prepare for real usage
