#!/bin/bash
# Script to run the enhanced pipeline with 50 papers

echo "=== Enhanced Literature Review Pipeline - 50 Papers ==="
echo "This script will:"
echo "1. Harvest 50 papers with keyword filtering"
echo "2. Fetch PDFs (and TeX/HTML when available)"
echo "3. Prepare screening sheet"
echo "4. Extract information using LLM service"
echo "5. Create visualizations"
echo "6. Export results"
echo ""

# Check if LLM service is running
echo "Checking LLM service..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ LLM service is not running!"
    echo "Please start it in another terminal:"
    echo "  python -m src.lit_review.llm_service"
    echo ""
    echo "Press Enter when ready..."
    read
fi

# Create output directory
OUTPUT_DIR="test_output/50_papers_run"
mkdir -p $OUTPUT_DIR

# Step 1: Harvest papers with keyword filtering
echo ""
echo "Step 1: Harvesting 50 papers..."
uv run python run.py harvest \
    --max-results 50 \
    --sources arxiv semantic_scholar \
    --filter-keywords "wargame,simulation,game,agent,LLM,GPT,AI,military,strategy" \
    --exclude-keywords "medical,biology,chemistry" \
    --min-keyword-matches 2 \
    --output $OUTPUT_DIR/harvested_papers.csv

# Check results
if [ ! -f "$OUTPUT_DIR/harvested_papers.csv" ]; then
    echo "❌ Harvesting failed!"
    exit 1
fi

PAPER_COUNT=$(tail -n +2 "$OUTPUT_DIR/harvested_papers.csv" | wc -l)
echo "✓ Harvested $PAPER_COUNT papers"

# Step 2: Fetch PDFs
echo ""
echo "Step 2: Fetching PDFs and source files..."
uv run python run.py fetch-pdfs \
    --input $OUTPUT_DIR/harvested_papers.csv \
    --output $OUTPUT_DIR/papers_with_pdfs.csv \
    --cache-dir $OUTPUT_DIR/pdf_cache

# Step 3: Prepare screening sheet
echo ""
echo "Step 3: Preparing screening sheet..."
uv run python run.py prepare-screen \
    --input $OUTPUT_DIR/papers_with_pdfs.csv \
    --output $OUTPUT_DIR/screening_sheet.xlsx

echo "✓ Screening sheet created at: $OUTPUT_DIR/screening_sheet.xlsx"
echo ""
echo "For this demo, we'll auto-include all papers..."
cp $OUTPUT_DIR/papers_with_pdfs.csv $OUTPUT_DIR/screening_results.csv

# Step 4: Extract with enhanced LLM
echo ""
echo "Step 4: Extracting information with enhanced LLM..."
uv run python run.py extract \
    --input $OUTPUT_DIR/screening_results.csv \
    --output $OUTPUT_DIR/extraction_results.csv \
    --use-enhanced \
    --prefer-tex \
    --parallel

# Check if extraction worked
if [ ! -f "$OUTPUT_DIR/extraction_results.csv" ]; then
    echo "❌ Extraction failed!"
    echo "Make sure the LLM service is running and API keys are configured"
    exit 1
fi

# Step 5: Create visualizations
echo ""
echo "Step 5: Creating visualizations..."
uv run python run.py visualise \
    --input $OUTPUT_DIR/extraction_results.csv \
    --output-dir $OUTPUT_DIR/figures

# Step 6: Export results
echo ""
echo "Step 6: Exporting final package..."
uv run python run.py export \
    --input $OUTPUT_DIR/extraction_results.csv \
    --output $OUTPUT_DIR/final_results.zip \
    --include-pdfs

echo ""
echo "=== Pipeline Complete! ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "- Harvested papers: $OUTPUT_DIR/harvested_papers.csv"
echo "- Extraction results: $OUTPUT_DIR/extraction_results.csv"
echo "- Visualizations: $OUTPUT_DIR/figures/"
echo "- Final package: $OUTPUT_DIR/final_results.zip"

# Show summary statistics
echo ""
echo "Summary statistics:"
EXTRACTED_COUNT=$(grep -c "success" "$OUTPUT_DIR/extraction_results.csv" 2>/dev/null || echo "0")
echo "- Papers harvested: $PAPER_COUNT"
echo "- Papers extracted: $EXTRACTED_COUNT"
