"""
Shared fixtures for the entire test suite.

The Presidio AnalyzerEngine loads a ~800 MB spaCy model.
Using scope="session" ensures the model is loaded exactly once no matter
how many test functions request the fixture.

Tests that require the spaCy model are marked with @pytest.mark.requires_spacy.
Run `pytest -m "not requires_spacy"` to skip all model-dependent tests.
"""

import pytest
from unittest import mock


def _spacy_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_lg")
        return True
    except Exception:
        return False


SPACY_AVAILABLE = _spacy_available()


@pytest.fixture(scope="session")
def analyzer():
    """Return an AnalyzerEngine built once per test session.

    Skipped automatically when en_core_web_lg is not installed.
    """
    if not SPACY_AVAILABLE:
        pytest.skip("en_core_web_lg not installed — skipping spaCy-dependent test")
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
