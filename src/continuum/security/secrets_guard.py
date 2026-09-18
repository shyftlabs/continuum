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

#: Known-weak values shipped as defaults for the credentials this guard checks
#: — the Session Redis password and the Qdrant/Milvus token. Compared
#: case-insensitively. Blank/``changeme`` placeholders are handled separately in
#: :func:`is_weak_secret`. (Secrets consumed only by the bundled Langfuse stack —
#: MinIO/ClickHouse/Langfuse keys — are intentionally NOT listed here: the guard
#: never sees them, so listing them would be dead weight.)
WEAK_SECRETS = frozenset(
    {
        "sdk123456789",  # old committed SESSION_REDIS_PASSWORD (finding D2)
        "myredissecret",  # docker-compose.yml Redis default (REDIS_AUTH)
    }
)


#: Length floor for a credential an attacker can attack offline. 32 characters is
#: the output of the ``openssl rand -hex 16`` this project recommends doubling —
#: short enough not to reject a genuinely random secret, long enough that nothing
#: typed from memory survives it.
MIN_OFFLINE_SECRET_LENGTH = 32


def is_weak_secret(value: str | None, *, min_length: int = 0) -> bool:
    """Return True if *value* is missing, blank, or a known-weak placeholder.

    A credential is considered weak when it is ``None``/empty/whitespace, an
    exact (case-insensitive) match for a shipped placeholder in
    :data:`WEAK_SECRETS`, or contains the substring ``changeme`` (which catches
    placeholder variants like ``CHANGEME_generate_with_openssl_rand_hex_32``).

    ``min_length`` additionally rejects anything shorter, measured after
    stripping. It is opt-in per call site rather than global, because it suits
    only some credentials. A Redis password is guessed *through the network*,
    where the server rate-limits the attacker; a key like ``SESSION_ID_SECRET``
    is guessed *offline*, because any user of the system holds a matched
    (plaintext, digest) pair from their own session and can grind candidates
    locally with no rate limit and no logs. Only the second kind needs a floor —
    and making it opt-in means no existing deployment is failed over a rule its
    credential was never held to.
    """
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        # A mis-parsed inline comment, not a secret. python-dotenv only strips
        # '#' when a value precedes it, so `KEY= # note` yields the note as the
        # value — and it is long enough to clear any length floor. Caught here
        # because such a value would otherwise be accepted as a real credential,
        # identical in every deployment that copied the file, and readable by
        # anyone with the repository. No generated secret starts with '#'.
        return True
    if min_length and len(stripped) < min_length:
        return True
    lowered = stripped.lower()
    if lowered in WEAK_SECRETS:
        return True
    return "changeme" in lowered


def _allow_insecure() -> bool:
    """True if the operator opted out of fail-closed via the escape hatch."""
    return os.environ.get(ALLOW_INSECURE_ENV, "").strip().lower() in _TRUTHY


def enforce_credential(
    *,
    service: str,
    credential: str | None,
    env_var: str,
    min_length: int = 0,
) -> None:
    """Refuse to proceed when *credential* is missing or a known-weak default.

    Args:
        service: Human-readable store name for the message (e.g. "Session Redis").
        credential: The secret value being used to authenticate to the store.
        env_var: The environment variable the operator should set to fix it
            (e.g. ``SESSION_REDIS_PASSWORD``), named in the error/warning.
        min_length: Reject anything shorter than this. Defaults to 0 (no floor)
            so existing call sites keep their exact behaviour; pass
            :data:`MIN_OFFLINE_SECRET_LENGTH` for a secret an attacker can
            brute-force offline. See :func:`is_weak_secret`.

    Raises:
        InsecureConfigurationError: when the credential is weak and the
            ``CONTINUUM_ALLOW_INSECURE`` escape hatch is not set.
    """
    if not is_weak_secret(credential, min_length=min_length):
        return

    hint = (
        f"{service} is configured with a missing or weak credential. Set "
        f"{env_var} to a strong, unique secret (e.g. `openssl rand -hex 32`)"
    )

    if _allow_insecure():
        logger.warning("INSECURE credential allowed via %s=1 — %s.", ALLOW_INSECURE_ENV, hint)
        return

    raise InsecureConfigurationError(
        f"Refusing to start: {hint}. To override for local/testing only, set "
        f"{ALLOW_INSECURE_ENV}=1.",
        config_key=env_var,
    )
