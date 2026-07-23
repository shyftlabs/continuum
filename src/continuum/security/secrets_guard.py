"""Fail-closed guard for data-store credentials (security findings F8 / D2 / D4).

The bundled dev stack ships weak placeholder secrets (``miniosecret``, a
``CHANGEME...`` Redis password, etc.). Those are safe on a loopback-only dev
box, but a real deployment that reuses them exposes the data store to anyone
who can reach it. This guard makes that failure *loud*: when a connector is
about to build a client with a missing or known-weak credential, it refuses to
start instead of silently connecting unprotected.

Scope: the two connectors that guard *your* data with a secret *you* set —
Redis (sessions) and the vector store (Qdrant/Milvus). Outbound API clients
(Langfuse, Temporal) authenticate you to an external service rather than lock a
data store, so they are intentionally out of scope.

Escape hatch: set ``CONTINUUM_ALLOW_INSECURE=1`` to downgrade the refusal to a
warning (local development / throwaway CI only).
"""

from __future__ import annotations

import os

from continuum.exceptions import InsecureConfigurationError
from continuum.logging import get_logger

logger = get_logger(__name__)

#: Environment variable that relaxes the guard from "refuse" to "warn".
ALLOW_INSECURE_ENV = "CONTINUUM_ALLOW_INSECURE"

#: Truthy values recognised for :data:`ALLOW_INSECURE_ENV`.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Known-weak credential values shipped as placeholders/defaults across the
#: repo's ``.env.template`` and ``docker-compose.yml``. Compared case-insensitively.
WEAK_SECRETS = frozenset(
    {
        "miniosecret",
        "sdk123456789",
        "myredissecret",
        "mysecret",
        "mysalt",
        "clickhouse",
        "changeme",
        # Langfuse's shipped default ENCRYPTION_KEY.
        "7b443ebc4c3a0944f7c8f5cb72077e71444a3beda18d845433c55ec506164c16",
    }
)


def is_weak_secret(value: str | None) -> bool:
    """Return True if *value* is missing, blank, or a known-weak placeholder.

    A credential is considered weak when it is ``None``/empty/whitespace, an
    exact (case-insensitive) match for a shipped placeholder in
    :data:`WEAK_SECRETS`, or contains the substring ``changeme`` (which catches
    placeholder variants like ``CHANGEME_generate_with_openssl_rand_hex_32``).
    """
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in WEAK_SECRETS:
        return True
    return "changeme" in lowered


def _allow_insecure() -> bool:
    """True if the operator opted out of fail-closed via the escape hatch."""
    return os.environ.get(ALLOW_INSECURE_ENV, "").strip().lower() in _TRUTHY


def enforce_credential(*, service: str, credential: str | None, env_var: str) -> None:
    """Refuse to proceed when *credential* is missing or a known-weak default.

    Args:
        service: Human-readable store name for the message (e.g. "Session Redis").
        credential: The secret value being used to authenticate to the store.
        env_var: The environment variable the operator should set to fix it
            (e.g. ``SESSION_REDIS_PASSWORD``), named in the error/warning.

    Raises:
        InsecureConfigurationError: when the credential is weak and the
            ``CONTINUUM_ALLOW_INSECURE`` escape hatch is not set.
    """
    if not is_weak_secret(credential):
        return

    hint = (
        f"{service} is configured with a missing or weak credential. Set "
        f"{env_var} to a strong, unique secret (e.g. `openssl rand -hex 32`)"
    )

    if _allow_insecure():
        logger.warning(
            "INSECURE credential allowed via %s=1 — %s.", ALLOW_INSECURE_ENV, hint
        )
        return

    raise InsecureConfigurationError(
        f"Refusing to start: {hint}. To override for local/testing only, set "
        f"{ALLOW_INSECURE_ENV}=1.",
        config_key=env_var,
    )
