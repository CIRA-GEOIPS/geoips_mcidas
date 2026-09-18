#!/usr/bin/env bash
# Everything CI runs, in one command, failing on the first problem.
set -euo pipefail
uv run --no-project --with ruff ruff check src tests
uv run --no-project --with ruff ruff format --check src tests
uv run --extra lint --extra test --no-sync mypy ./src
uv run --extra test --no-sync pytest -q
