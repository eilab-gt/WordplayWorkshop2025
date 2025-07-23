#!/bin/bash
# Test runner script for literature review pipeline

set -e

echo "🧪 Running Literature Review Pipeline Tests"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  No virtual environment detected."
    echo "   Attempting to activate .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found. Please run 'uv venv' first."
        exit 1
    fi
fi

# Install test dependencies if needed
echo "📦 Checking test dependencies..."
uv pip install pytest pytest-cov pytest-mock --quiet

# Run tests with coverage
echo ""
echo "🚀 Running tests with coverage..."
echo ""

# Run all tests
pytest

# Run specific test categories if requested
if [ "$1" = "unit" ]; then
    echo ""
    echo "🔬 Running unit tests only..."
    pytest -m unit
elif [ "$1" = "integration" ]; then
    echo ""
    echo "🔗 Running integration tests only..."
    pytest -m integration
elif [ "$1" = "fast" ]; then
    echo ""
    echo "⚡ Running fast tests only..."
    pytest -m "not slow"
fi

# Generate coverage report
echo ""
echo "📊 Coverage Report Summary:"
echo ""
pytest --cov=src/lit_review --cov-report=term-missing --no-header --tb=no -q

echo ""
echo "✨ Testing complete! HTML coverage report available at htmlcov/index.html"