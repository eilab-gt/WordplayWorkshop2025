.PHONY: help install test lint format type-check clean docs pre-commit all

help:
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linting (ruff)"
	@echo "  make format       - Format code (black)"
	@echo "  make type-check   - Run type checking (mypy)"
	@echo "  make clean        - Clean up temporary files"
	@echo "  make docs         - Build documentation"
	@echo "  make pre-commit   - Run pre-commit hooks"
	@echo "  make all          - Run format, lint, type-check, and test"

install:
	uv sync --dev

test:
	uv run pytest

test-verbose:
	uv run pytest -v

test-coverage:
	uv run pytest --cov-report=html --cov-report=term

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/

format:
	uv run black src/ tests/

type-check:
	uv run mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

docs:
	@echo "Documentation building not configured yet"

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install

all: format lint type-check test