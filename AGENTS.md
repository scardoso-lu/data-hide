# Repository Guidelines

## Project Structure & Module Organization

This repository contains a stateless Python pipeline for anonymizing Microsoft Fabric OneLake Delta tables. The main application lives in `app/`, with `main.py` as a thin shim entrypoint. Tests live in `tests/` and are split by behavior: orchestration, anonymization, audit database, alerts, key vault, and helpers. Runtime packaging is defined by `Dockerfile`, local orchestration by `docker-compose.yml`, and dependencies by `pyproject.toml` plus the committed `uv.lock` lock file.

## Build, Test, and Development Commands

The project uses [uv](https://docs.astral.sh/uv/) as its package manager. uv reads `.python-version` and `pyproject.toml`, manages the `.venv` directory automatically, and resolves against the committed `uv.lock`.

- `uv sync`: create the venv and install runtime + dev dependencies from `uv.lock`.
- `uv sync --no-dev`: production install — runtime deps only.
- `uv run python -m spacy download en_core_web_lg && uv run python -m spacy download fr_core_news_lg && uv run python -m spacy download de_core_news_lg`: install the required NLP models (English, French, German/Luxembourgish) for full anonymization tests and local runs.
- `uv run pytest`: run the test suite configured by `pytest.ini`.
- `uv run pytest -m "not requires_spacy and not slow"`: skip spaCy-dependent tests.
- `uv lock --upgrade`: refresh `uv.lock` to the latest versions within the ranges declared in `pyproject.toml`.
- `docker compose up --build`: build and run the pipeline with local PostgreSQL.
- `docker build -t fabric-pii-pipeline:latest .`: build the standalone container image.

Commit `uv.lock` whenever `pyproject.toml` changes — the Dockerfile uses `uv sync --frozen` and will fail the build if the lock is stale.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where they clarify function contracts, and focused functions with explicit names. Keep constants in uppercase, environment variable names uppercase, and tests named `test_<behavior>.py`. Prefer structured data handling with pandas and existing helper functions over ad hoc parsing. Keep runtime behavior stateless; do not add container-local file writes.

## Testing Guidelines

The project uses `pytest` with `tests` as the configured test root. Mark tests that require the full spaCy model with `requires_spacy` or `slow` as appropriate. Mock external I/O in unit and orchestration tests: Delta Lake, Azure credentials, PostgreSQL, Purview, Key Vault, and alert webhooks. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

Git history uses concise imperative commits, including Conventional Commit style such as `feat: identifier hashing and JSON/nested document support`. Prefer `feat:`, `fix:`, `test:`, or `docs:` when the scope is clear. Pull requests should describe behavior changes, list verification commands, mention security or data-handling impact, and link related issues. Include screenshots only for documentation or UI-adjacent changes.

## Security & Configuration Tips

Never commit `.env` or live credentials. Use `.env.example` for configuration shape. Keep Azure, Purview, PostgreSQL, and webhook secrets in environment variables only. Preserve the non-root, no-runtime-files container posture described in the README.
