"""
Shared fixtures for the entire test suite.

The Presidio engines (AnalyzerEngine + AnonymizerEngine) load a ~800 MB spaCy
model.  Using scope="session" ensures the model is loaded exactly once no
matter how many test functions request the fixture.
"""

import pytest

from main import build_engines


@pytest.fixture(scope="session")
def presidio_engines():
    """Return a (AnalyzerEngine, AnonymizerEngine) tuple, built once per session."""
    return build_engines()
