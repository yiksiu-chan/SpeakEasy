.PHONY: install install-dev lint format test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[all,dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	mypy src/speakeasy/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
