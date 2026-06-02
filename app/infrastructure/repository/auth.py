"""Azure credential and token acquisition.

All mutable singletons (``DefaultAzureCredential``, ``_credential``,
``_token_cache``) live in the *package* namespace (``app.repository``)
so that tests can patch ``app.infrastructure.repository.DefaultAzureCredential`` and
the change is immediately visible to every function here.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

ONELAKE_TOKEN_SCOPE = "https://storage.azure.com/.default"
SQL_TOKEN_SCOPE = "https://database.windows.net/.default"
FABRIC_TOKEN_SCOPE = "https://api.fabric.microsoft.com/.default"


def _credential_instance() -> object:
    """Return (and cache) the DefaultAzureCredential singleton.

    Reads ``DefaultAzureCredential`` and ``_credential`` from the package
    namespace so that ``mocker.patch("app.infrastructure.repository.DefaultAzureCredential",
    FakeCredential)`` is always visible here.
    """
    import app.infrastructure.repository as _r
    if _r.DefaultAzureCredential is None:
        from azure.identity import DefaultAzureCredential as _DC
        _r.DefaultAzureCredential = _DC
    if _r._credential is None:
        _r._credential = _r.DefaultAzureCredential()
    return _r._credential


def acquire_token(scope: str) -> str:
    token = _credential_instance().get_token(scope)
    return token.token


def acquire_cached_token(scope: str) -> str:
    import app.infrastructure.repository as _r
    now = time.time()
    cached = _r._token_cache.get(scope)
    if cached and cached[1] - now > 300:
        return cached[0]
    token = _credential_instance().get_token(scope)
    expires_on = float(getattr(token, "expires_on", now + 3600))
    _r._token_cache[scope] = (token.token, expires_on)
    return token.token
