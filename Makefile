# FashionMatch - Common Development Commands
# Usage: make <command>

.PHONY: help install install-dev setup clean lint format type-check test test-cov run scrape

# Default target
help:
	@echo "FashionMatch - Available Commands"
	@echo "=================================="
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make setup        - Full project setup (venv + deps + pre-commit)"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make lint         - Run linters (flake8)"
	@echo "  make format       - Format code (black + isort)"
	@echo "  make type-check   - Run type checking (mypy)"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make run          - Run Streamlit app"
	@echo "  make scrape       - Run scraper script"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	playwright install chromium

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements-dev.txt
	. venv/bin/activate && playwright install chromium
	. venv/bin/activate && pre-commit install
	cp -n config/config.example.yaml config/config.yaml || true
	@echo "Setup complete! Activate venv with: source venv/bin/activate"

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

# Code Quality
lint:
	flake8 src tests --max-line-length=100 --extend-ignore=E203,E501,W503

format:
	black src tests
	isort src tests

type-check:
	mypy src --ignore-missing-imports

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Running
run:
	streamlit run src/ui/app.py

scrape:
	python -m scripts.scrape

# Pre-commit
pre-commit:
	pre-commit run --all-files
