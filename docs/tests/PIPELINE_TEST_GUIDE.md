# 🧪 Pipeline Testing Guide

## 🎯 Overview

This guide helps you test the WordplayWorkshop2025 pipeline with a small, controlled dataset to verify functionality without processing thousands of papers.

## 🚀 Quick Test Strategy

### 1. **Small-Scale Test Configuration**

Create a test configuration file:

```bash
cp config/config.yaml config/test_config.yaml
```

Edit `config/test_config.yaml`:
```yaml
# Limit results for testing
development:
  sample_size: 10  # Process only 10 papers
  debug: true
  use_cache: true

# Reduce rate limits for faster testing
rate_limits:
  google_scholar:
    requests_per_hour: 100
    delay_seconds: 2

# Use test query
search:
  queries:
    test_query: '"LLM" AND "wargaming" AND "2024"'  # Recent papers only
```

### 2. **Test Data Options**

#### Option A: Use Seed Papers (Recommended)
```bash
# Create a test CSV from seed papers
python -c "
import json
import pandas as pd
with open('data/seed_papers.json') as f:
    data = json.load(f)
papers = pd.DataFrame(data['seed_papers'])
papers.to_csv('data/test_papers.csv', index=False)
print(f'Created test dataset with {len(papers)} papers')
"
```

#### Option B: Limited Harvest
```bash
# Harvest just 5-10 papers per source
python run.py harvest \
    --config config/test_config.yaml \
    --query test_query \
    --max-results 5 \
    --sources arxiv  # Start with just arXiv
```

### 3. **Step-by-Step Pipeline Test**

#### Step 1: Test Harvesting
```bash
# Test with single source first
python run.py harvest \
    --config config/test_config.yaml \
    --query '"Snow Globe" OR "WarAgent"' \
    --sources arxiv \
    --max-results 3

# Check results
python -c "
import pandas as pd
df = pd.read_csv('data/raw/papers_raw.csv')
print(f'Harvested {len(df)} papers')
print(df[['title', 'source']].head())
"
```

#### Step 2: Test Screening Preparation
```bash
# Generate screening sheet
python run.py prepare-screen \
    --input data/raw/papers_raw.csv \
    --output data/test_screening.xlsx

# Verify Excel file created
ls -la data/test_screening.xlsx
```

#### Step 3: Test PDF Fetching
```bash
# Test PDF fetching with just 2 papers
python -c "
from src.lit_review.processing import PDFFetcher
import pandas as pd

df = pd.read_csv('data/raw/papers_raw.csv').head(2)
fetcher = PDFFetcher()

for _, paper in df.iterrows():
    print(f'Fetching: {paper.title[:50]}...')
    pdf_data = fetcher.fetch(paper.get('pdf_url', paper['url']))
    if pdf_data:
        print(f'  ✓ Success: {len(pdf_data)} bytes')
    else:
        print('  ✗ Failed')
"
```

#### Step 4: Test LLM Extraction
```bash
# Test extraction on 1-2 papers
python -c "
from src.lit_review.extraction import LLMExtractor
from src.lit_review.utils import load_config
import pandas as pd

config = load_config('config/test_config.yaml')
extractor = LLMExtractor(config)

# Test with first paper
df = pd.read_csv('data/raw/papers_raw.csv').head(1)
paper = df.iloc[0]

print(f'Testing extraction on: {paper.title[:50]}...')
result = extractor.extract(paper['abstract'])
print('Extraction result:', result)
"
```

#### Step 5: Full Pipeline Test
```bash
# Run complete pipeline on test data
python run.py extract \
    --config config/test_config.yaml \
    --input data/raw/papers_raw.csv \
    --output data/test_extraction.csv

# Generate visualizations
python run.py visualise \
    --input data/test_extraction.csv \
    --output-dir outputs/test_viz/
```

### 4. **Validation Checklist**

Run this validation script:

```python
# save as: scripts/validate_pipeline.py
import pandas as pd
from pathlib import Path
import json

def validate_pipeline():
    checks = []

    # 1. Check harvested data
    if Path('data/raw/papers_raw.csv').exists():
        df = pd.read_csv('data/raw/papers_raw.csv')
        checks.append(f"✓ Harvested {len(df)} papers")
        checks.append(f"  Sources: {df['source'].value_counts().to_dict()}")
    else:
        checks.append("✗ No harvested data found")

    # 2. Check PDFs
    pdf_count = len(list(Path('pdf_cache').glob('*.pdf')))
    checks.append(f"✓ Downloaded {pdf_count} PDFs")

    # 3. Check extractions
    if Path('data/test_extraction.csv').exists():
        df = pd.read_csv('data/test_extraction.csv')
        checks.append(f"✓ Extracted {len(df)} papers")
        checks.append(f"  Game types: {df['game_type'].value_counts().to_dict()}")
    else:
        checks.append("✗ No extraction data found")

    # 4. Check visualizations
    viz_count = len(list(Path('outputs/test_viz').glob('*.png')))
    checks.append(f"✓ Generated {viz_count} visualizations")

    # 5. Check logs
    if Path('logs/pipeline.log').exists():
        with open('logs/pipeline.log') as f:
            error_count = sum(1 for line in f if 'ERROR' in line)
        checks.append(f"✓ Log errors: {error_count}")

    print("\n📊 Pipeline Validation Report")
    print("=" * 40)
    for check in checks:
        print(check)

if __name__ == "__main__":
    validate_pipeline()
```

### 5. **Common Issues & Solutions**

#### API Key Issues
```bash
# Test OpenAI connection
python -c "
import openai
from src.lit_review.utils import load_config
config = load_config('config/test_config.yaml')
client = openai.OpenAI(api_key=config['api_keys']['openai'])
print('OpenAI connection: OK')
"
```

#### Rate Limiting
```bash
# Add delays in test config
# config/test_config.yaml:
rate_limits:
  google_scholar:
    delay_seconds: 10  # Increase delay
```

#### Memory Issues
```bash
# Process in smaller batches
processing:
  batch_sizes:
    harvesting: 100
    pdf_download: 5
    llm_extraction: 2
```

### 6. **Test Data Cleanup**

After testing:
```bash
# Create cleanup script
cat > scripts/cleanup_test.sh << 'EOF'
#!/bin/bash
# Backup test results
mkdir -p test_results_backup
cp -r data/test_* test_results_backup/
cp -r outputs/test_* test_results_backup/

# Clean test data
rm -f data/test_*
rm -rf outputs/test_*
rm -f config/test_config.yaml

echo "Test data cleaned. Backup saved in test_results_backup/"
EOF

chmod +x scripts/cleanup_test.sh
```

### 7. **Progressive Testing**

Start small and expand:
1. **Minimal**: 3-5 papers, arXiv only
2. **Small**: 10-20 papers, 2 sources
3. **Medium**: 50 papers, all sources
4. **Full**: 100+ papers, production config

### 8. **Performance Monitoring**

```python
# Monitor resource usage during test
import psutil
import time

def monitor_pipeline():
    process = psutil.Process()
    print(f"CPU: {process.cpu_percent()}%")
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Open files: {len(process.open_files())}")
```

## 🎯 Expected Results

For a test with 5-10 papers:
- **Harvest time**: 30-60 seconds
- **PDF fetching**: 1-2 minutes
- **LLM extraction**: 2-5 minutes
- **Total time**: ~10 minutes
- **Cost**: ~$0.10-0.50 (OpenAI API)

## 🚦 Success Indicators

✅ All papers have titles and abstracts
✅ At least 50% PDFs downloaded successfully
✅ LLM extractions contain expected fields
✅ Visualizations generated without errors
✅ No critical errors in logs
✅ Export package created successfully

## 🔍 Next Steps

Once small-scale testing succeeds:
1. Gradually increase paper count
2. Add more sources
3. Test error recovery
4. Validate data quality
5. Run full production harvest

Remember: Start small, validate thoroughly, then scale up!
