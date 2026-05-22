"""Unit tests for the Azure Key Vault RSA pseudonymizer.

The real ``CryptographyClient`` is never instantiated — every test injects a
fake client whose ``sign`` method returns a controllable signature.  This
verifies the key-derivation contract without requiring Key Vault credentials.
"""

from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pandas as pd
import pytest

from app.keyvault import (
    KeyVaultPseudonymizer,
    _DERIVATION_CONSTANT,
    build_pseudonymizer_from_env,
)


class _FakeCryptoClient:
    """Records sign() calls and returns a deterministic fake signature."""

    def __init__(self, signature: bytes = b"deterministic-signature") -> None:
        self.signature_bytes = signature
        self.calls: list[tuple] = []

    def sign(self, algorithm, digest):
        self.calls.append((algorithm, digest))
        return SimpleNamespace(signature=self.signature_bytes)


def _expected_token(signature_bytes: bytes, value: str) -> str:
    derived = hashlib.sha256(signature_bytes).digest()
    return hmac.new(derived, value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


class TestKeyVaultPseudonymizerInit:

    def test_signs_fixed_constant_on_init(self):
        client = _FakeCryptoClient()
        KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=client)

        assert len(client.calls) == 1
        _algorithm, digest = client.calls[0]
        assert digest == hashlib.sha256(_DERIVATION_CONSTANT).digest()

    def test_uses_rs256_algorithm(self):
        from azure.keyvault.keys.crypto import SignatureAlgorithm

        client = _FakeCryptoClient()
        KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=client)

        algorithm, _digest = client.calls[0]
        assert algorithm == SignatureAlgorithm.rs256

    def test_signs_only_once_regardless_of_calls(self):
        client = _FakeCryptoClient()
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=client)
        for v in ["a", "b", "c", "d", "e"]:
            p.pseudonymize(v)
        assert len(client.calls) == 1

    def test_missing_vault_url_raises(self):
        with pytest.raises(ValueError):
            KeyVaultPseudonymizer("", "k", crypto_client=_FakeCryptoClient())

    def test_missing_key_name_raises(self):
        with pytest.raises(ValueError):
            KeyVaultPseudonymizer("https://v.vault.azure.net/", "", crypto_client=_FakeCryptoClient())


class TestKeyVaultPseudonymize:

    def test_string_value_returns_24_hex(self):
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient())
        token = p.pseudonymize("EMP001")
        assert isinstance(token, str)
        assert len(token) == 24
        int(token, 16)  # must be valid hex

    def test_token_matches_expected_hmac(self):
        sig = b"my-signature"
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient(sig))
        assert p.pseudonymize("EMP001") == _expected_token(sig, "EMP001")

    def test_same_value_same_token(self):
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient())
        assert p.pseudonymize("EMP001") == p.pseudonymize("EMP001")

    def test_different_values_different_tokens(self):
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient())
        assert p.pseudonymize("EMP001") != p.pseudonymize("EMP002")

    def test_different_signatures_yield_different_tokens(self):
        p1 = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient(b"sig-a"))
        p2 = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient(b"sig-b"))
        assert p1.pseudonymize("EMP001") != p2.pseudonymize("EMP001")

    def test_null_value_passes_through(self):
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient())
        assert p.pseudonymize(None) is None
        assert pd.isna(p.pseudonymize(pd.NA))

    def test_integer_value_pseudonymized_as_string_repr(self):
        sig = b"sig"
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient(sig))
        assert p.pseudonymize(12345) == _expected_token(sig, "12345")

    def test_callable_protocol(self):
        """The instance itself is a Callable[[value], token]."""
        p = KeyVaultPseudonymizer("https://v.vault.azure.net/", "k", crypto_client=_FakeCryptoClient())
        assert p("EMP001") == p.pseudonymize("EMP001")


class TestKeyVaultKeyVersionCapture:
    """Verify the resolved key version is captured for audit reporting."""

    def test_injected_version_is_exposed(self):
        p = KeyVaultPseudonymizer(
            "https://v.vault.azure.net/", "k",
            crypto_client=_FakeCryptoClient(),
            key_version="abc123version",
        )
        assert p.key_version == "abc123version"

    def test_version_defaults_to_none_when_not_provided(self):
        p = KeyVaultPseudonymizer(
            "https://v.vault.azure.net/", "k",
            crypto_client=_FakeCryptoClient(),
        )
        assert p.key_version is None

    def test_build_crypto_client_reads_version_from_key_properties(self, mocker):
        """_build_crypto_client returns (client, key.properties.version)."""
        fake_key = SimpleNamespace(
            key_type="RSA",
            properties=SimpleNamespace(version="2025-05-22-resolved-version"),
        )
        key_client_instance = mocker.MagicMock()
        key_client_instance.get_key.return_value = fake_key

        mocker.patch("azure.keyvault.keys.KeyClient", return_value=key_client_instance)
        mocker.patch("azure.identity.DefaultAzureCredential", return_value=object())
        mocker.patch("azure.keyvault.keys.crypto.CryptographyClient", return_value=_FakeCryptoClient())

        _client, version = KeyVaultPseudonymizer._build_crypto_client(
            "https://v.vault.azure.net/", "k", credential=None,
        )
        assert version == "2025-05-22-resolved-version"

    def test_build_crypto_client_rejects_non_rsa(self, mocker):
        fake_key = SimpleNamespace(
            key_type="EC",
            properties=SimpleNamespace(version="v1"),
        )
        key_client_instance = mocker.MagicMock()
        key_client_instance.get_key.return_value = fake_key
        mocker.patch("azure.keyvault.keys.KeyClient", return_value=key_client_instance)
        mocker.patch("azure.identity.DefaultAzureCredential", return_value=object())

        with pytest.raises(ValueError, match="not an RSA key"):
            KeyVaultPseudonymizer._build_crypto_client(
                "https://v.vault.azure.net/", "k", credential=None,
            )


class TestBuildPseudonymizerFromEnv:

    def test_returns_none_when_unset(self):
        assert build_pseudonymizer_from_env(None, None) is None
        assert build_pseudonymizer_from_env("", "") is None

    def test_raises_when_partially_set(self):
        with pytest.raises(RuntimeError, match="both be set"):
            build_pseudonymizer_from_env("https://v.vault.azure.net/", None)
        with pytest.raises(RuntimeError, match="both be set"):
            build_pseudonymizer_from_env(None, "key")
