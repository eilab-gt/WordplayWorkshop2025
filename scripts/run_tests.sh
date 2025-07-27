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
if ! python -c "import pytest" 2>/dev/null; then
    echo "   Installing test dependencies..."
    uv pip install pytest pytest-cov pytest-mock --quiet
else
    echo "   ✅ Test dependencies already installed"
fi

# Run tests with coverage
echo ""
echo "🚀 Running tests with coverage..."
echo ""

# Set pytest arguments based on test type
PYTEST_ARGS=""
TEST_TYPE="all"

case "$1" in
    unit)
        PYTEST_ARGS="-m unit"
        TEST_TYPE="unit"
        echo "🔬 Running unit tests only..."
        ;;
    integration)
        PYTEST_ARGS="-m integration"
        TEST_TYPE="integration"
        echo "🔗 Running integration tests only..."
        ;;
    fast)
        PYTEST_ARGS="-m 'not slow'"
        TEST_TYPE="fast"
        echo "⚡ Running fast tests only..."
        ;;
    *)
        echo "🚀 Running all tests with coverage..."
        ;;
esac

# Run tests once with coverage
echo ""
pytest $PYTEST_ARGS --cov=src/lit_review --cov-report=term-missing --cov-report=html

echo ""
echo "✨ Testing complete! HTML coverage report available at htmlcov/index.html"
