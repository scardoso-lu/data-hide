# Repository Guidelines

## Release Status

The project has not been released. Do not add backward-compatibility shims, overloaded method signatures, fallback code paths, or migration helpers unless explicitly instructed.

## Project Structure & Module Organization

This repository contains a stateless Python pipeline for anonymizing Microsoft Fabric OneLake Delta tables. The main application lives in `main.py`, which handles Azure authentication, Delta reads/writes, Presidio anonymization, Purview checks, audit logging, and alerts. Tests live in `tests/` and are split by behavior: orchestration, anonymization, audit database, alerts, and helpers. Runtime packaging is defined by `Dockerfile`, local orchestration by `docker-compose.yml`, and dependencies by `requirements.txt` plus `requirements-dev.txt`.

## Build, Test, and Development Commands

Prefix shell commands with `rtk`; use `rtk proxy powershell -Command "..."` for PowerShell built-ins.

- `rtk proxy powershell -Command "python -m venv .venv"`: create a local virtual environment.
- `rtk proxy powershell -Command "pip install -r requirements-dev.txt"`: install runtime and test dependencies.
- `rtk proxy powershell -Command "python -m spacy download en_core_web_lg && python -m spacy download fr_core_news_lg && python -m spacy download de_core_news_lg"`: install the required NLP models (English, French, German/Luxembourgish) for full anonymization tests and local runs.
- `rtk proxy powershell -Command "pytest"`: run the test suite configured by `pytest.ini`.
- `rtk docker compose up --build`: build and run the pipeline with local PostgreSQL.
- `rtk docker build -t fabric-pii-pipeline:latest .`: build the standalone container image.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where they clarify function contracts, and focused functions with explicit names. Keep constants in uppercase, environment variable names uppercase, and tests named `test_<behavior>.py`. Prefer structured data handling with pandas and existing helper functions over ad hoc parsing. Keep runtime behavior stateless; do not add container-local file writes.

## Testing Guidelines

The project uses `pytest` with `tests` as the configured test root. Mark tests that require the full spaCy model with `requires_spacy` or `slow` as appropriate. Mock external I/O in unit and orchestration tests: Delta Lake, Azure credentials, PostgreSQL, Purview, and alert webhooks. Run `rtk proxy powershell -Command "pytest"` before submitting changes.

## Commit & Pull Request Guidelines

Git history uses concise imperative commits, including Conventional Commit style such as `feat: identifier hashing and JSON/nested document support`. Prefer `feat:`, `fix:`, `test:`, or `docs:` when the scope is clear. Pull requests should describe behavior changes, list verification commands, mention security or data-handling impact, and link related issues. Include screenshots only for documentation or UI-adjacent changes.

## Security & Configuration Tips

Never commit `.env` or live credentials. Use `.env.example` for configuration shape. Keep Azure, Purview, PostgreSQL, and webhook secrets in environment variables only. Preserve the non-root, no-runtime-files container posture described in the README.
