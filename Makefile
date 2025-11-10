.PHONY: help install install-dev test lint format type-check clean run all

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linters (flake8)"
	@echo "  make format       - Format code (black, isort)"
	@echo "  make type-check   - Run type checker (mypy)"
	@echo "  make clean        - Clean temporary files"
	@echo "  make run          - Run the agent"
	@echo "  make all          - Format, lint, type-check, and test"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

run:
	python -m src.hyperliquid_analytics.main

all: format lint type-check test
	@echo "✅ All checks passed!"
