"""
Ambient policy context for data-label enforcement.

The model-routing gate (and future label gates) needs the active run's
``policy_store``, subject, and live ``data_labels``. Rather than thread these
through every ``LLMClient.chat()`` caller — fragile, and silently bypassed by
any call site that forgets — they are published once per agent execution into a
``contextvar`` and read by ``chat()`` as a fallback.

This mirrors the existing ambient-context pattern in
``continuum.observability.trace_context`` (trace/span/session ids) and the
``get_current_session_id()`` fallback already used inside ``chat()``.

Concurrency: ``contextvars`` are per-async-task. Tasks spawned with
``asyncio.create_task``/``gather`` copy the context at creation, so each
parallel branch inherits the value active when it was spawned and cannot clobber
a sibling's. Always set via :func:`use_active_policy` so the previous value is
restored on exit (no leakage across runs).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ActivePolicy",
    "get_active_policy",
    "reset_active_policy",
    "resolve_active_policy",
    "set_active_policy",
    "use_active_policy",
]


@dataclass(frozen=True)
class ActivePolicy:
    """The policy context for the currently-executing agent.

    ``context`` is the live ``RunContext`` (held as ``Any`` to avoid coupling the
    security layer to the agent layer); labels are read from it on demand so
    taint added mid-run is reflected.
    """

    policy_store: Any
    subject: str
    context: Any  # RunContext — read .data_labels live

    @property
    def data_labels(self) -> set[str]:
        return set(getattr(self.context, "data_labels", set()) or set())


_active_policy: ContextVar[ActivePolicy | None] = ContextVar("active_policy", default=None)


def get_active_policy() -> ActivePolicy | None:
    """Return the policy context for the current async task, or None."""
    return _active_policy.get()


def resolve_active_policy(
    policy_store: Any,
    subject: str | None,
    data_labels: set[str] | None,
) -> tuple[Any, str | None, set[str] | None]:
    """Resolve the effective (policy_store, subject, data_labels) for a gate.

    Explicit arguments win; when no ``policy_store`` is passed, fall back to the
    ambient policy published for the current run (and its live labels). Used by
    every label gate (LLM routing, memory write) so a single ambient publish
    covers call sites that don't thread policy params themselves.
    """
    if policy_store is None:
        ambient = _active_policy.get()
        if ambient is not None:
            return (
                ambient.policy_store,
                ambient.subject,
                data_labels if data_labels is not None else ambient.data_labels,
            )
    return policy_store, subject, data_labels


def set_active_policy(policy_store: Any, subject: str, context: Any) -> Token[ActivePolicy | None]:
    """Publish the active policy context; returns a token to pass to
    :func:`reset_active_policy`. For use with try/finally where a context
    manager would force re-indenting a large body (e.g. AgentRunner.run)."""
    return _active_policy.set(
        ActivePolicy(policy_store=policy_store, subject=subject, context=context)
    )


def reset_active_policy(token: Token[ActivePolicy | None]) -> None:
    """Restore the previous active policy context."""
    _active_policy.reset(token)


@contextmanager
def use_active_policy(policy_store: Any, subject: str, context: Any) -> Iterator[None]:
    """Publish the active policy context for the duration of the block.

    Restores the previous value on exit (supports nesting, e.g. handoffs that
    re-enter with a different agent's policy store).
    """
    token = set_active_policy(policy_store, subject, context)
    try:
        yield
    finally:
        reset_active_policy(token)
