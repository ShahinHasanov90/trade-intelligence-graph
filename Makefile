.PHONY: help install test lint format clean serve build-graph docker-up docker-down

PYTHON := python3
PIP := pip3
SRC_DIR := src/graph_intel
TEST_DIR := tests

help: ## Show this help message
	@echo "Trade Intelligence Graph - Development Commands"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and package in development mode
	$(PIP) install -e ".[dev]"
	pre-commit install

test: ## Run test suite with coverage
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

lint: ## Run linters (ruff, mypy)
	ruff check $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports

format: ## Auto-format code (black, isort via ruff)
	black $(SRC_DIR) $(TEST_DIR)
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

serve: ## Start GraphQL API server
	$(PYTHON) -m uvicorn graph_intel.api.app:app --host 0.0.0.0 --port 8000 --reload

build-graph: ## Build graph from sample declaration data
	$(PYTHON) -m graph_intel.graph.builder --config config/graph_config.yaml

docker-up: ## Start Docker services (Neo4j + API)
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --strict --ignore-missing-imports

security-check: ## Run security checks
	pip-audit
	bandit -r $(SRC_DIR)
