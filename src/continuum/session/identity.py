"""
Session identity — how a session's storage key is derived.

A session id plays two roles at once, and they pull in opposite directions:

    name        the storage key; must be deterministic, so a returning user
                resolves to the same session without a lookup table
    capability  whoever holds it reaches the data; must be unguessable

The original scheme optimised for the first and gave up the second: the key was
built in plaintext from the identifiers it scopes, so ``u:{user_id}`` could be
*constructed* by anyone who knew a user id. No leak was required.

This module keeps determinism and restores unguessability by putting the same
derived key through an HMAC under a deployment secret.

Why keyed, and not a plain hash: SHA-256 is public, so hashing a guessable
input yields an equally guessable output — an attacker with the user id
reproduces the digest exactly. The secret is the part they do not have, so the
secret *is* the fix. That is also why ``SessionConfig`` refuses to start with
hashing enabled and no secret configured, instead of quietly falling back to
plaintext: a control that reports "enabled" while doing nothing is worse than
one that is honestly off.

The secret must be identical in every process and stable across restarts — it
is a derivation parameter, not a per-process random. A secret generated at boot
would give each worker its own key space (a user's history would appear and
disappear depending on which worker answered) and would orphan every stored
session on redeploy. Hence: supplied by configuration, or hashing stays off.

Rotating the secret changes every id and is therefore the same kind of event as
the plaintext→hashed migration, with the same dual-read handling.
"""

from __future__ import annotations

import hashlib
import hmac

from continuum.session.types import generate_session_id
from continuum.utils.sanitization import validate_conversation_id, validate_user_id

# Marks a key produced by the keyed scheme, so hashed and legacy plaintext keys
# are distinguishable at a glance in Redis, logs and dashboards.
HASHED_PREFIX = "s_"

# Half of a SHA-256 digest. 128 bits is far beyond brute-force reach for an
# online guessing attack while keeping keys short enough to read in a log line.
_DIGEST_CHARS = 32


def derive_plaintext_key(
    user_id: str | None,
    conversation_id: str | None,
) -> str | None:
    """Return the deterministic plaintext key for these identifiers.

    Returns None when the identifiers do not determine a key (no user id), in
    which case the caller falls back to a random session id.

    The ``c:``/``u:`` namespace prefixes prevent a bare ``user_id="foo:bar"``
    colliding with ``conversation_id="foo"`` + ``user_id="bar"``.
    """
    user_id = validate_user_id(user_id)
    conversation_id = validate_conversation_id(conversation_id)
    if conversation_id and user_id:
        return f"c:{conversation_id}:u:{user_id}"
    if user_id:
        return f"u:{user_id}"
    return None


def hash_key(plaintext_key: str, secret: str) -> str:
    """HMAC a derived plaintext key under the deployment secret."""
    digest = hmac.new(
        secret.encode("utf-8"),
        plaintext_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{HASHED_PREFIX}{digest[:_DIGEST_CHARS]}"


def compute_session_id(
    session_id: str | None,
    user_id: str | None,
    conversation_id: str | None,
    *,
    secret: str | None = None,
) -> str:
    """Compute the session id for these inputs.

    Priority:
        - explicit ``session_id`` → used as-is (internal handoff calls supply
          framework-generated ids, which are already unguessable)
        - identifiers that determine a key → derived, then HMAC'd when a secret
          is configured
        - nothing to derive from → a random id

    Raises:
        InvalidIdentifierError: if the identifiers contain characters unsafe for
            a storage key fragment. An explicit ``session_id`` is trusted as-is.
    """
    if session_id:
        return session_id

    plaintext = derive_plaintext_key(user_id, conversation_id)
    if plaintext is None:
        # Anonymous: a random id is already unguessable, so hashing adds nothing.
        return generate_session_id()

    if secret:
        return hash_key(plaintext, secret)
    return plaintext


def legacy_session_id(
    session_id: str | None,
    user_id: str | None,
    conversation_id: str | None,
) -> str | None:
    """The pre-migration plaintext id for these inputs, if one exists.

    Used for the dual-read that lets a deployment switch hashing on without
    orphaning live conversations. Returns None when there is nothing to look up:
    an explicit id was never derived, and an anonymous id was random, so neither
    has a plaintext predecessor.
    """
    if session_id:
        return None
    return derive_plaintext_key(user_id, conversation_id)
