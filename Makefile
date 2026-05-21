PYTHON  ?= python
VENV    := .venv
PIP     := $(VENV)/Scripts/pip
PYTEST  := $(VENV)/Scripts/pytest
SPACY   := $(VENV)/Scripts/python -m spacy

SPACY_MODELS := en_core_web_lg fr_core_news_lg de_core_news_lg

# ── setup ─────────────────────────────────────────────────────────────────────

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: venv
	$(VENV)/Scripts/python -m pip install --upgrade pip
	$(PIP) install -r requirements-dev.txt

.PHONY: install-models
install-models:
	@for model in $(SPACY_MODELS); do \
	    echo "Downloading $$model ..."; \
	    $(SPACY) download $$model; \
	done

.PHONY: setup
setup: install install-models

# ── test ──────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTEST) -m ""

# ── utility ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup          Create venv, install all deps, download spaCy models"
	@echo "  install        Create venv + install Python packages only"
	@echo "  install-models Download the three required spaCy models"
	@echo "  test           Run the full test suite including spaCy-dependent tests"
	@echo "  clean          Remove venv, caches"
