.PHONY: help install lint format test run compose-up compose-down

VENV ?= .venv

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sed 's/:.*//' | sort

install:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --extra dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

run:
	uv run python -m nodepragagent.cli

compose-up:
	docker compose --env-file .env up

compose-down:
	docker compose --env-file .env down
