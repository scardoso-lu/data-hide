"""
Shared fixtures for the entire test suite.

The Presidio AnalyzerEngine loads spaCy models for English, French, German,
and Luxembourgish (via the German model). Using scope="session" ensures all
models are loaded exactly once no matter how many test functions request the
fixture.

Tests that require the spaCy models are marked with @pytest.mark.requires_spacy.
Run `pytest -m "not requires_spacy"` to skip all model-dependent tests.
"""

import pytest
from unittest import mock

from app.domain.anonymization import SPACY_MODELS as _SPACY_MODELS

# Resolved from SPACY_MODELS so the gate always matches the configured
# defaults (en/fr/de `_md`, overridable via SPACY_MODEL_<LANG> env vars).
_REQUIRED_SPACY_MODELS = sorted(set(_SPACY_MODELS.values()))


def _spacy_available() -> bool:
    try:
        import spacy
        for model in _REQUIRED_SPACY_MODELS:
            spacy.load(model)
        return True
    except Exception:
        return False


SPACY_AVAILABLE = _spacy_available()


@pytest.fixture(scope="session")
def analyzer():
    """Return an AnalyzerEngine built once per test session.

    Skipped automatically when any required spaCy model is not installed.
    """
    if not SPACY_AVAILABLE:
        pytest.skip(
            f"Required spaCy models ({', '.join(_REQUIRED_SPACY_MODELS)}) not all installed"
            " — skipping spaCy-dependent test"
        )
    from main import build_engines
    return build_engines()


# Backward-compatible alias kept for any tests still referencing presidio_engines.
presidio_engines = analyzer


@pytest.fixture()
def mocker(request):
    """Small pytest-mock compatible subset used by this repo's tests."""

    class Mocker:
        MagicMock = mock.MagicMock

        def patch(self, target, *args, **kwargs):
            patcher = mock.patch(target, *args, **kwargs)
            mocked = patcher.start()
            request.addfinalizer(patcher.stop)
            return mocked

        def patch_object(self, target, attribute, *args, **kwargs):
            patcher = mock.patch.object(target, attribute, *args, **kwargs)
            mocked = patcher.start()
            request.addfinalizer(patcher.stop)
            return mocked

    return Mocker()
