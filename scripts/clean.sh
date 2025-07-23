#!/bin/bash
# Clean up temporary files and cache

set -e

echo "🧹 Cleaning Literature Review Pipeline"
echo "====================================="
echo ""

# Function to get directory size
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Show current usage
echo "📊 Current disk usage:"
echo "   PDF cache: $(get_size pdf_cache)"
echo "   Outputs: $(get_size outputs)"
echo "   Logs: $(get_size logs)"
echo "   Coverage: $(get_size htmlcov)"
echo "   Pytest cache: $(get_size .pytest_cache)"
echo ""

# Confirmation
read -p "🤔 Clean all temporary files? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Cleaning temporary files..."

# Clean Python cache
echo "   Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Clean test artifacts
echo "   Removing test artifacts..."
rm -rf .pytest_cache 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true
rm -f .coverage 2>/dev/null || true
rm -f coverage.xml 2>/dev/null || true

# Clean build artifacts
echo "   Removing build artifacts..."
rm -rf build 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf src/*.egg-info 2>/dev/null || true

# Ask about data directories
echo ""
read -p "🤔 Clean data directories? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Cleaning data directories..."
    rm -rf data/raw/* 2>/dev/null || true
    rm -rf data/processed/* 2>/dev/null || true
    rm -rf data/extracted/* 2>/dev/null || true
    # Keep template files
    echo "   ✅ Data directories cleaned (templates preserved)"
fi

# Ask about PDF cache
echo ""
read -p "🤔 Clean PDF cache? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Cleaning PDF cache..."
    rm -rf pdf_cache/* 2>/dev/null || true
    echo "   ✅ PDF cache cleaned"
fi

# Ask about outputs
echo ""
read -p "🤔 Clean output files? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Cleaning outputs..."
    rm -rf outputs/* 2>/dev/null || true
    echo "   ✅ Outputs cleaned"
fi

# Ask about logs
echo ""
read -p "🤔 Clean logs? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Cleaning logs..."
    rm -f logs/*.db 2>/dev/null || true
    rm -f logs/*.log 2>/dev/null || true
    echo "   ✅ Logs cleaned"
fi

# Show new usage
echo ""
echo "📊 New disk usage:"
echo "   PDF cache: $(get_size pdf_cache)"
echo "   Outputs: $(get_size outputs)"
echo "   Logs: $(get_size logs)"
echo "   Coverage: $(get_size htmlcov)"
echo "   Pytest cache: $(get_size .pytest_cache)"

echo ""
echo "✨ Cleanup complete!"