"""Phase 1 — fail-closed data-store credential guard (F8/D2/D4).

Covers the pure ``is_weak_secret`` classifier and the ``enforce_credential``
gate: it raises on a weak/blank secret, passes a strong one, and downgrades to
a warning under the ``CONTINUUM_ALLOW_INSECURE`` escape hatch.
"""

from __future__ import annotations

import pytest

from continuum.exceptions import InsecureConfigurationError
from continuum.security import secrets_guard
from continuum.security.secrets_guard import (
    ALLOW_INSECURE_ENV,
    enforce_credential,
    is_weak_secret,
)


class TestIsWeakSecret:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "sdk123456789",
            "myredissecret",
            "MyRedisSecret",  # case-insensitive
            "CHANGEME_generate_with_openssl_rand_hex_32",  # substring
            "changeme",
        ],
    )
    def test_weak_values(self, value):
        assert is_weak_secret(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "f3a1c0de9b8877665544332211aabbccddeeff00112233445566778899aabbcc",
            "a-genuinely-strong-passphrase-2026",
            "S3cure!Redis#Pw",
        ],
    )
    def test_strong_values(self, value):
        assert is_weak_secret(value) is False


class TestEnforceCredential:
    def test_strong_credential_passes(self):
        # Should not raise.
        enforce_credential(
            service="Session Redis",
            credential="f3a1c0de9b8877665544332211aabbccddeeff0011223344",
            env_var="SESSION_REDIS_PASSWORD",
        )

    def test_weak_credential_raises(self, monkeypatch):
        monkeypatch.delenv(ALLOW_INSECURE_ENV, raising=False)
        with pytest.raises(InsecureConfigurationError, match="SESSION_REDIS_PASSWORD"):
            enforce_credential(
                service="Session Redis",
                credential="myredissecret",
                env_var="SESSION_REDIS_PASSWORD",
            )

    def test_blank_credential_raises(self, monkeypatch):
        monkeypatch.delenv(ALLOW_INSECURE_ENV, raising=False)
        with pytest.raises(InsecureConfigurationError):
            enforce_credential(
                service="Qdrant", credential=None, env_var="QDRANT_API_KEY"
            )

    def test_escape_hatch_downgrades_to_warning(self, monkeypatch):
        monkeypatch.setenv(ALLOW_INSECURE_ENV, "1")
        warnings: list[str] = []
        monkeypatch.setattr(
            secrets_guard.logger,
            "warning",
            lambda msg, *a, **k: warnings.append(msg % a if a else msg),
        )
        # Must NOT raise when the hatch is set.
        enforce_credential(
            service="Session Redis",
            credential="myredissecret",
            env_var="SESSION_REDIS_PASSWORD",
        )
        assert warnings, "expected a warning when escape hatch is set"

    def test_escape_hatch_false_value_still_raises(self, monkeypatch):
        monkeypatch.setenv(ALLOW_INSECURE_ENV, "0")
        with pytest.raises(InsecureConfigurationError):
            enforce_credential(
                service="Session Redis",
                credential="myredissecret",
                env_var="SESSION_REDIS_PASSWORD",
            )
