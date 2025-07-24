# Quick Start Guide

This guide walks you through conducting a literature review on LLM-powered wargames from start to finish.

## Prerequisites

Before starting, ensure you have:
- Python 3.13+ installed
- UV package manager installed
- OpenAI API key
- Internet connection for paper harvesting

## Step 1: Installation

```bash
# Clone the repository
git clone <repository-url>
cd WordplayWorkshop2025

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .

# Copy configuration template
cp config/config.yaml.example config/config.yaml
```

## Step 2: Configure API Keys

Edit `config/config.yaml` and add your API keys:

```yaml
api_keys:
  openai: 'sk-your-openai-key-here'
  # Optional but recommended:
  semantic_scholar: 'your-s2-key-here'
```

Or use environment variables:
```bash
export OPENAI_API_KEY='sk-your-openai-key-here'
```

## Step 3: Search for Papers

Run your first paper search:

```bash
# Use preset search query
python run.py harvest --query preset1

# Or use custom query
python run.py harvest --query '"LLM agents" AND "military simulation"'

# Search specific sources only
python run.py harvest --sources arxiv,crossref --max-results 50
```

**Output**: `data/raw/papers_raw.csv` with all discovered papers

## Step 4: Prepare for Screening

Generate an Excel file for manual screening:

```bash
python run.py prepare-screen
```

**Output**: `outputs/screening_sheet_YYYYMMDD_HHMMSS.xlsx`

### Screening Instructions

Open the Excel file and for each paper:

1. Read title and abstract
2. In `include_ta` column, enter:
   - `yes` - if relevant to LLM wargaming
   - `no` - if not relevant
   - Leave blank if unsure

3. If excluding, add reason code in `reason_ta`:
   - `E1` - Not about wargaming
   - `E2` - No LLM/AI component
   - `E3` - Not research (news, blog, etc.)
   - `E4` - Duplicate
   - `E5` - Other (explain in notes)

4. Add any notes in `notes_ta` column

5. Save the Excel file when done

## Step 5: Extract Information

After screening, extract structured data using LLM:

```bash
# Use your completed screening sheet
python run.py extract --input outputs/screening_sheet_completed.xlsx

# Or use default path
python run.py extract
```

**Output**: `data/extracted/extraction_YYYYMMDD_HHMMSS.csv`

### What Gets Extracted

- Venue type (conference, journal, workshop)
- Game type (seminar, matrix, digital, hybrid)
- Open-ended vs quantitative classification
- LLM model family and role
- Evaluation metrics used
- Failure modes identified
- Code availability

## Step 6: Visualize Results

Generate analysis charts:

```bash
python run.py visualise

# Or specify input file
python run.py visualise --input data/extracted/extraction_results.csv
```

**Output**: Charts saved to `outputs/` directory:
- `timeline.png` - Publications by year
- `venue_distribution.png` - Conference vs journal
- `failure_modes.png` - Common failure patterns
- `llm_families.png` - Models used
- `game_types.png` - Types of wargames
- `awscale_distribution.png` - Analytical vs Creative spectrum

## Step 7: Export Dataset

Package everything for sharing:

```bash
python run.py export
```

**Output**: `outputs/dataset_YYYYMMDD_HHMMSS.zip`

## Complete Workflow Example

Here's a complete workflow from start to finish:

```bash
# 1. Setup
uv venv && source .venv/bin/activate
uv pip install -e .
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your API keys

# 2. Harvest papers (searches all configured sources)
python run.py harvest --query preset1 --max-results 100

# 3. Prepare screening sheet
python run.py prepare-screen

# 4. Manual screening
# Open outputs/screening_sheet_*.xlsx in Excel
# Review papers and mark include/exclude decisions
# Save as screening_completed.xlsx

# 5. Extract information from included papers
python run.py extract --input screening_completed.xlsx

# 6. Generate visualizations
python run.py visualise

# 7. Export final dataset
python run.py export

# 8. Check pipeline status
python run.py status
```

## Tips for Better Results

### Search Strategies

1. **Start broad**: Use general terms first
   ```bash
   python run.py harvest --query '"large language model" AND game'
   ```

2. **Then refine**: Add specific terms
   ```bash
   python run.py harvest --query '"LLM" AND "wargaming" AND "military"'
   ```

3. **Check grey literature**: Include arXiv for preprints
   ```bash
   python run.py harvest --sources arxiv --query 'LLM simulation'
   ```

### Screening Best Practices

1. **Be consistent**: Develop clear inclusion criteria
2. **When in doubt, include**: Better to review in full-text stage
3. **Document decisions**: Use notes column for edge cases
4. **Take breaks**: Screening is mentally taxing

### Extraction Optimization

1. **Check PDF downloads**:
   ```bash
   ls -la pdf_cache/
   ```

2. **Monitor API usage**: Extraction uses OpenAI credits

3. **Handle failures**: Re-run extraction for failed papers
   ```bash
   python run.py extract --retry-failed
   ```

## Troubleshooting

### No papers found
- Check internet connection
- Verify search query syntax
- Try different sources
- Check API rate limits

### Extraction failures
- Verify OpenAI API key
- Check API quota/credits
- Ensure PDFs downloaded successfully
- Try with smaller batches

### Excel file issues
- Use Excel 2016+ or LibreOffice Calc
- Check file permissions
- Close file before running commands

### Memory/performance issues
- Reduce `--max-results` parameter
- Disable parallel processing: `--no-parallel`
- Process in smaller batches

## Getting Help

1. Check logs:
   ```bash
   python run.py status
   ```

2. Enable debug mode:
   ```bash
   python run.py --debug harvest
   ```

3. Review error details in `logs/logging.db`

4. See full documentation in `docs/API_DOCUMENTATION.md`

## Next Steps

After completing your first review:

1. **Refine search queries** based on discovered papers
2. **Adjust failure vocabularies** in `config/config.yaml`
3. **Customize visualizations** for your analysis
4. **Share your dataset** via Zenodo for DOI
5. **Contribute improvements** via pull requests

Happy reviewing! 🎉
