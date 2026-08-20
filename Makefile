.PHONY: install test lint format typecheck run-baseline run-multi benchmark check clean

install:
	pip install -e ".[dev,llm]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

run-baseline:
	python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art"

run-multi:
	python -m multi_agent_research_lab.cli multi-agent --query "When is a multi-agent architecture better than a single agent?" --trace-out reports/trace_demo.json

benchmark:
	python -m multi_agent_research_lab.cli benchmark --out reports/benchmark_report.md

check: lint test
	ruff format --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
