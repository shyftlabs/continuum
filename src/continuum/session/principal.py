"""
The principal — who is asking, as opposed to what they are asking for.

A session ownership check compares two values: the owner stored with the
session, and the identity of the caller. The second one is the hard part,
because it must arrive from somewhere the caller cannot forge.

It is deliberately NOT a function argument. Session data is reached through
more than one door — ``AgentRunner.run()`` takes a ``user_id``, but
``LLMClient.achat()`` reads and writes history with only a ``session_id`` and
has no identity parameter at all. Threading a new argument through every layer
in between would be invasive and would be missed somewhere. A ContextVar flows
through async calls on its own, so the application sets it once at its trust
boundary and every layer below inherits it::

    user = verify_jwt(request.headers["Authorization"])   # the app's own check
    with bind_principal(user.id):
        await runner.run(...)          # or llm_client.achat(...) — either door

The framework never derives a principal by itself, and deliberately provides no
way to set one from request data. It cannot tell a verified identity from a
string somebody typed; only the application knows which of its values came from
an authenticated credential.

Note on the neighbouring variable: ``continuum.core.context._user_id`` looks
like it would do the job and must not be used for it. That one is populated by
logging and tracing from a caller-supplied argument, so a request carrying
``user_id="victim"`` would set it — letting an attacker fill in the answer to
the security question. Keeping the two separate, with different names and a
narrow setter here, is what stops a telemetry field from silently becoming an
access-control input.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

# Distinct name from the telemetry user_id on purpose — see the module docstring.
_principal: ContextVar[str | None] = ContextVar("continuum_session_principal", default=None)


def get_principal() -> str | None:
    """The identity bound for the current execution context, if any.

    None means the caller asserted no identity — not that they are anonymous.
    How that is treated is an ownership-policy decision, not a fact about the
    caller. See :mod:`continuum.session.ownership`.
    """
    return _principal.get()


def set_principal(principal: str | None) -> Token[str | None]:
    """Bind a principal, returning a token for :func:`reset_principal`.

    Prefer :func:`bind_principal`; this is for framework code that cannot use a
    ``with`` block (middleware that sets up and tears down in separate hooks).
    """
    return _principal.set(principal)


def reset_principal(token: Token[str | None]) -> None:
    """Restore the principal that was bound before ``token`` was issued."""
    _principal.reset(token)


@contextmanager
def bind_principal(principal: str | None) -> Iterator[None]:
    """Bind the verified caller identity for the duration of the block.

    Restores the previous value on exit, including when the body raises, so a
    failed request cannot leave its identity behind for the next one. Safe under
    concurrency: contextvars are per-task, so two requests interleaved on one
    event loop never observe each other's principal.
    """
    token = _principal.set(principal)
    try:
        yield
    finally:
        _principal.reset(token)
