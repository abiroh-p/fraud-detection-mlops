# ============================================================
# Fraud Detection MLOps — Developer Commands
# ============================================================

.PHONY: help install install-dev lint format test test-unit \
        test-integration clean docker-up docker-down

# Show available commands
help:
	@echo ""
	@echo "Usage: make <command>"
	@echo ""
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install all dependencies including dev tools"
	@echo "  lint           Run ruff linter"
	@echo "  format         Run black formatter"
	@echo "  test           Run all tests with coverage"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo "  clean          Remove all cache and build artifacts"
	@echo "  docker-up      Start all services with Docker Compose"
	@echo "  docker-down    Stop all services"
	@echo ""

# Install only production deps
install:
	pip install -e .

# Install everything including dev tools
install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Lint with ruff
lint:
	ruff check src/ tests/

# Format with black
format:
	black src/ tests/

# Lint + format together (run before every commit)
check: lint format

# Run all tests
test:
	pytest tests/

# Run only unit tests (fast — no external services needed)
test-unit:
	pytest tests/unit/ -v

# Run only integration tests (requires Docker services)
test-int:
	pytest tests/integration/ -v

# Remove all generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "Cleaned."

# Start all Docker services
docker-up:
	docker compose up -d

# Stop all Docker services
docker-down:
	docker compose down

# Stop and remove all volumes (WARNING: deletes database data)
docker-reset:
	docker compose down -v
