.PHONY: help install dev-up dev-down test lint format typecheck verify

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync --extra dev

dev-up: ## Start infrastructure services
	docker compose up -d
	@echo "ChromaDB:  http://localhost:8001"
	@echo "MLflow:    http://localhost:5001"
	@echo "Postgres:  localhost:5432"

dev-down: ## Stop infrastructure services
	docker compose down

test: ## Run all tests
	uv run pytest tests/ -v --cov=src/vulnremedy --cov-report=term-missing

lint: ## Run linter
	uv run ruff check src/ tests/

format: ## Format code
	uv run ruff format src/ tests/

typecheck: ## Run type checker
	uv run mypy src/

verify: lint typecheck test ## Run all checks

ollama-status: ## Check Ollama models
	ollama list