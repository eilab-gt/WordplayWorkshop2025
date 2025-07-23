#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up Literature Review Pipeline Development Environment"
echo "============================================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.13"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version"

# Check for UV
echo ""
echo "📌 Checking for UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install from: https://github.com/astral-sh/uv"
    exit 1
fi
echo "✅ UV is installed"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
uv pip install -e .
uv pip install pytest pytest-cov pytest-mock pre-commit

# Set up pre-commit hooks
echo ""
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo ""
echo "📁 Creating directory structure..."
directories=(
    "data/raw"
    "data/processed" 
    "data/extracted"
    "data/templates"
    "outputs"
    "pdf_cache"
    "logs"
    "notebooks"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "   ✅ $dir"
done

# Copy configuration template
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo "✅ Created config.yaml from template"
        echo "⚠️  Remember to add your API keys to config.yaml"
    else
        echo "⚠️  config.yaml.example not found"
    fi
else
    echo "✅ config.yaml already exists"
fi

# Create .env file if it doesn't exist
echo ""
echo "🔐 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Literature Review Pipeline Environment Variables
# Add your API keys here

# Required for LLM extraction
OPENAI_API_KEY=

# Optional but recommended
SEMANTIC_SCHOLAR_API_KEY=

# Optional for dataset publishing
ZENODO_API_KEY=

# Optional for PDF fetching
UNPAYWALL_EMAIL=
EOF
    echo "✅ Created .env file"
    echo "⚠️  Remember to add your API keys to .env"
else
    echo "✅ .env file already exists"
fi

# Run initial tests
echo ""
echo "🧪 Running quick tests..."
python -c "import lit_review; print('✅ Package imports successfully')" || {
    echo "❌ Package import failed"
    exit 1
}

# Display next steps
echo ""
echo "✨ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your API keys to config.yaml or .env"
echo "2. Run tests: ./scripts/run_tests.sh"
echo "3. Try the pipeline: python run.py --help"
echo "4. Read the Quick Start Guide: docs/QUICK_START_GUIDE.md"
echo ""
echo "Happy coding! 🎉"