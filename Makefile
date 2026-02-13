.PHONY: setup \
		clean-cache-temp-files \
		lint code-check check-dead-code \
		test \
		pipeline pre-commit all

.DEFAULT_GOAL := all

SOURCE_PATH ?= src
TEST_PATH ?= tests
PATH_PROJECT_ROOT ?= .

setup:
	@echo "Installing dependencies..."
	@uv sync --all-extras
	@uv run pre-commit install
	@echo "✅ Dependencies installed."

clean-cache-temp-files:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@echo "✅ Clean complete."

lint:
	@echo "Running lint checks..."
	@uv run ruff format $(PATH_PROJECT_ROOT)
	@uv run ruff check --fix $(PATH_PROJECT_ROOT)
	@uv run isort $(PATH_PROJECT_ROOT)
	@echo "✅ Linting complete."

code-check:
	@echo "Running static code checks..."
	@uv run mypy $(SOURCE_PATH)
	@uv run complexipy -f $(SOURCE_PATH)
	@uv run bandit -r $(SOURCE_PATH)
	@echo "✅ Code and security checks complete."

check-dead-code:
	@echo "Checking dead code..."
	@uv run deadcode $(SOURCE_PATH)
	@echo "✅ Dead code check complete."
	
test:
	@echo "Running tests..."
	@uv run pytest $(TEST_PATH)
	@echo "✅ Tests complete."

pipeline: clean-cache-temp-files lint code-check
	@echo "✅ Pipeline complete."

pre-commit: clean-cache-temp-files lint code-check
	@echo "✅ Pipeline pre-commit complete."

all: setup pipeline
	@echo "✅ All tasks complete."
