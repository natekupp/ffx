# Install dependencies
install:
	uv sync

install-dev:
	uv sync --dev

lint:
	uv run ruff check ffx

format:
	uv run ruff format ffx

typecheck:
	uv run ty check ffx

validate: lint format typecheck

# Testing
test:
	uv run pytest ffx_tests/

test-cov:
	uv run pytest ffx_tests/ --cov=ffx

# Build and publish
build:
	uv build

pypi:
	uv build
	uv publish

# Clean
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
