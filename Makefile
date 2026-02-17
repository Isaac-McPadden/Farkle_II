PYTHON ?= python3
COV_THRESHOLD ?= 90

.PHONY: install lint format typecheck test ci

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src tests

format:
	black --check .

typecheck:
	mypy src/farkle

test:
	pytest -q --cov=src/farkle --cov-report=term-missing --cov-report=xml --cov-fail-under=$(COV_THRESHOLD)

ci: lint format typecheck test
