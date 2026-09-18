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
            enforce_credential(service="Qdrant", credential=None, env_var="QDRANT_API_KEY")

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


class TestMinimumLength:
    """A length floor for secrets whose attacker can brute-force offline.

    ``is_weak_secret`` catches placeholders it has been told about. That is the
    right shape for a Redis password — an attacker has to come through the
    network to test a guess. It is not enough for a key like SESSION_ID_SECRET,
    where any user of the system holds a matched (plaintext -> digest) pair from
    their own session and can grind guesses locally with no rate limit and no
    logs. There, anything short or memorable falls in milliseconds.

    So ``min_length`` is opt-in per call site rather than global: the callers
    that predate it keep their exact behaviour, and no deployment is broken by a
    rule its credential was never held to.
    """

    STRONG = "d27d3f15bdd2236ac32c8333ddc38b0546f49a7db0276293b93ae4174d597641"

    @pytest.mark.parametrize(
        "value",
        [
            "abc",
            "your-secret-key",  # a placeholder nobody thought to replace
            "# required when SESSION_HASH_IDS=true",  # a mis-parsed inline comment
            "short-but-random-x9",
        ],
    )
    def test_values_under_the_floor_are_weak(self, value):
        assert is_weak_secret(value, min_length=32) is True

    def test_a_full_length_secret_passes(self):
        assert is_weak_secret(self.STRONG, min_length=32) is False

    def test_exactly_at_the_floor_passes(self):
        assert is_weak_secret("a" * 32, min_length=32) is False

    def test_one_short_of_the_floor_fails(self):
        assert is_weak_secret("a" * 31, min_length=32) is True

    def test_length_is_measured_after_stripping(self):
        """Trailing whitespace is not entropy."""
        assert is_weak_secret("a" * 20 + "            ", min_length=32) is True

    def test_existing_callers_are_unaffected(self):
        """No min_length argument means exactly the previous behaviour, so the
        Redis and vector-store guards keep passing credentials they accepted
        before this rule existed."""
        assert is_weak_secret("S3cure!Redis#Pw") is False
        assert is_weak_secret("S3cure!Redis#Pw", min_length=0) is False

    def test_a_known_placeholder_is_still_weak_at_any_length(self):
        """Length does not redeem a value that is already on the list."""
        assert is_weak_secret("CHANGEME_generate_with_openssl_rand_hex_32", min_length=32) is True

    def test_enforce_credential_applies_the_floor(self, monkeypatch):
        monkeypatch.delenv(ALLOW_INSECURE_ENV, raising=False)
        with pytest.raises(InsecureConfigurationError):
            enforce_credential(
                service="Session id hashing",
                credential="abc",
                env_var="SESSION_ID_SECRET",
                min_length=32,
            )

    def test_enforce_credential_passes_a_strong_secret(self):
        enforce_credential(
            service="Session id hashing",
            credential=self.STRONG,
            env_var="SESSION_ID_SECRET",
            min_length=32,
        )

    def test_escape_hatch_still_applies(self, monkeypatch):
        """Local development must not be blocked by the floor."""
        monkeypatch.setenv(ALLOW_INSECURE_ENV, "1")
        enforce_credential(
            service="Session id hashing",
            credential="abc",
            env_var="SESSION_ID_SECRET",
            min_length=32,
        )
