"""Azure Key Vault RSA-bound pseudonymization for identifier columns.

The pipeline pseudonymizes direct identifiers (employee_id, customer_id, ...)
to stable 24-hex-character tokens.  The secret material that controls the
mapping is an RSA key held in Azure Key Vault and never exported.

Mechanism (HSM-bound key derivation):

  1. At construction time, sign a fixed constant with the configured RSA key
     using RS256 (RSA + SHA-256 + PKCS#1 v1.5 padding).  PKCS#1 v1.5 signing
     is *deterministic*: signing the same digest with the same key always
     yields the same signature.  Key Vault performs the private-key operation
     inside the HSM; the key itself never leaves Key Vault.
  2. SHA-256 the signature to obtain a 32-byte secret.  This secret lives
     only in process memory for the duration of the pipeline run.
  3. Per-row pseudonymization uses HMAC-SHA256(secret, value) locally and
     returns the first 24 hex characters — same length and shape as the
     previous SHA-256 token, so downstream consumers see no schema change.

Why derive once instead of signing every value:
    Key Vault has no batch-sign API; one HTTPS round-trip per unique
    identifier value would dominate runtime for any non-trivial table.
    Deriving a per-run secret keeps the cryptographic root in the HSM
    while making per-row work a local HMAC.

Key rotation: the pipeline always uses the latest enabled version of the
configured RSA key.  Rotating the key in Key Vault therefore rotates every
pseudonym produced after the rotation — old and new pseudonyms will not
join.  Plan rotations accordingly.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Fixed domain separator used by LocalHashPseudonymizer when no HASH_SALT is set.
_LOCAL_HASH_DEFAULT_SALT = b"fabric-pii-pipeline:local-hash:v1"

# Fixed constant used as the input to the RSA signing operation.  Changing
# this string invalidates every pseudonym ever produced — treat as a versioned
# domain separator.
_DERIVATION_CONSTANT = b"fabric-pii-pipeline:identifier-pseudonym-key:v1"

_PSEUDONYM_HEX_LENGTH = 24


class KeyVaultPseudonymizer:
    """Produce deterministic identifier pseudonyms from an RSA key in Key Vault.

    A single instance derives its secret once during ``__init__`` and is
    reused for every identifier value in the pipeline run.
    """

    def __init__(
        self,
        vault_url: str,
        key_name: str,
        *,
        credential: Any = None,
        crypto_client: Any = None,
        key_version: str | None = None,
    ) -> None:
        if not vault_url or not key_name:
            raise ValueError("vault_url and key_name are required")

        self.vault_url = vault_url
        self.key_name = key_name
        if crypto_client is None:
            self._crypto_client, self.key_version = self._build_crypto_client(
                vault_url, key_name, credential,
            )
        else:
            self._crypto_client = crypto_client
            self.key_version = key_version
        self._derived_secret = self._derive_secret(self._crypto_client)
        logger.info(
            "Initialised Key Vault pseudonymizer (vault=%s, key=%s, version=%s)",
            vault_url, key_name, self.key_version or "unknown",
        )

    @staticmethod
    def _build_crypto_client(
        vault_url: str, key_name: str, credential: Any,
    ) -> tuple[Any, str | None]:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.keys import KeyClient
        from azure.keyvault.keys.crypto import CryptographyClient

        credential = credential or DefaultAzureCredential()
        key = KeyClient(vault_url=vault_url, credential=credential).get_key(key_name)
        if key.key_type not in ("RSA", "RSA-HSM"):
            raise ValueError(
                f"Key '{key_name}' in {vault_url} is not an RSA key (got key_type={key.key_type!r})."
            )
        # KeyVaultKey.properties.version is the version string resolved by
        # Key Vault when no explicit version was requested — record it for
        # audit reporting so each pipeline run is reproducibly traceable to
        # the exact key material that was active at the time.
        version = getattr(getattr(key, "properties", None), "version", None)
        return CryptographyClient(key, credential=credential), version

    @staticmethod
    def _derive_secret(crypto_client: Any) -> bytes:
        from azure.keyvault.keys.crypto import SignatureAlgorithm

        digest = hashlib.sha256(_DERIVATION_CONSTANT).digest()
        signature = crypto_client.sign(SignatureAlgorithm.rs256, digest).signature
        return hashlib.sha256(signature).digest()

    def pseudonymize(self, value: object) -> object:
        try:
            if pd.isna(value):
                return value
        except (TypeError, ValueError):
            pass
        raw = value if isinstance(value, str) else str(value)
        token = hmac.new(self._derived_secret, raw.encode("utf-8"), hashlib.sha256).hexdigest()
        return token[:_PSEUDONYM_HEX_LENGTH]

    __call__ = pseudonymize


class LocalHashPseudonymizer:
    """Deterministic pseudonymizer that runs entirely in process — no Key Vault.

    Produces the same 24-hex-character tokens as ``KeyVaultPseudonymizer``.
    The secret is derived from ``HASH_SALT``; if the salt is not supplied a
    fixed default is used (tokens are still deterministic, but anyone who
    knows the default can reverse-look-up short identifiers — only use the
    default in development / testing environments).
    """

    key_version: str | None = None  # no Key Vault key material

    def __init__(self, salt: str | None = None) -> None:
        if salt:
            self._secret = hashlib.sha256(salt.encode("utf-8")).digest()
            logger.info("LocalHashPseudonymizer initialised with HASH_SALT")
        else:
            self._secret = hashlib.sha256(_LOCAL_HASH_DEFAULT_SALT).digest()
            logger.warning(
                "LocalHashPseudonymizer: HASH_SALT is not set — using fixed default. "
                "Only suitable for development/testing; set HASH_SALT in production."
            )

    def pseudonymize(self, value: object) -> object:
        try:
            if pd.isna(value):
                return value
        except (TypeError, ValueError):
            pass
        raw = value if isinstance(value, str) else str(value)
        return hmac.new(self._secret, raw.encode("utf-8"), hashlib.sha256).hexdigest()[
            :_PSEUDONYM_HEX_LENGTH
        ]

    __call__ = pseudonymize


def build_pseudonymizer_from_env(
    vault_url: str | None,
    key_name: str | None,
    *,
    enable_key_vault: bool = True,
    hash_salt: str | None = None,
) -> KeyVaultPseudonymizer | LocalHashPseudonymizer | None:
    """Build a pseudonymizer from environment configuration.

    When *enable_key_vault* is ``False`` a ``LocalHashPseudonymizer`` is
    returned regardless of *vault_url* / *key_name* — Key Vault is never
    contacted.  Set ``ENABLE_KEY_VAULT=0`` and optionally ``HASH_SALT`` in
    ``.env`` to use this mode.
    """
    if not enable_key_vault:
        return LocalHashPseudonymizer(hash_salt)
    if not vault_url and not key_name:
        return None
    if not (vault_url and key_name):
        raise RuntimeError(
            "KEY_VAULT_URL and KEY_VAULT_RSA_KEY_NAME must both be set to enable "
            "identifier pseudonymization, or both left unset to disable it."
        )
    return KeyVaultPseudonymizer(vault_url, key_name)
