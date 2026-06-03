# uv handles venv creation, dependency resolution, and Python version selection.
# Targets below are thin convenience wrappers around `uv` commands; running the
# `uv` command directly works just as well.

SPACY_MODELS := en_core_web_lg fr_core_news_lg de_core_news_lg


.PHONY: install
install:
	uv sync

.PHONY: install-models
install-models:
	@for model in $(SPACY_MODELS); do \
	    echo "Downloading $$model ..."; \
	    uv run python -m spacy download $$model; \
	done

.PHONY: setup
setup: install install-models

.PHONY: lock
lock:
	uv lock

.PHONY: upgrade
upgrade:
	uv lock --upgrade

.PHONY: test
test:
	uv run pytest -m ""

.PHONY: test-fast
test-fast:
	uv run pytest -m "not requires_spacy and not slow"

.PHONY: clean
clean:
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup          uv sync + download spaCy models"
	@echo "  install        uv sync (runtime + dev deps from pyproject.toml)"
	@echo "  install-models Download the three required spaCy models"
	@echo "  lock           Regenerate uv.lock from pyproject.toml"
	@echo "  upgrade        Refresh uv.lock to latest versions within ranges"
	@echo "  test           Run the full test suite including spaCy-dependent tests"
	@echo "  test-fast      Run only the spaCy-independent tests"
	@echo "  clean          Remove .venv and caches"
