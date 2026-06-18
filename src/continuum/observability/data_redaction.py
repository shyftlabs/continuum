"""
Label-aware telemetry redaction.

Applied to trace input/output before it reaches the observability backend
(Langfuse). Two protections:

1. **Label-driven (redact mode):** if the run carries data-sensitivity labels
   and a policy denies them for the ``telemetry`` resource, the payload is
   replaced with a redacted placeholder — the trace skeleton (timings, tokens,
   status) is preserved, the content is not leaked.
2. **Always-on secret masking:** ``redact_dict`` masks API keys / tokens /
   passwords regardless of labels, closing the hole where these egressed to
   telemetry unmasked.

Policy resolution reuses the shared ambient resolver, so a single run-level
publish covers emission points that don't thread policy params themselves.
"""

from __future__ import annotations

from typing import Any

from continuum.security.policy_context import resolve_active_policy
from continuum.utils.secrets import redact_dict

__all__ = ["redact_for_telemetry"]


def redact_for_telemetry(
    data: Any,
    *,
    policy_store: Any | None = None,
    subject: str | None = None,
    labels: set[str] | None = None,
) -> Any:
    """Return a telemetry-safe version of ``data``.

    If a policy (explicit or ambient) denies the run's labels for ``telemetry``,
    returns a redacted placeholder. Otherwise returns ``data`` with secrets
    masked (dicts only; non-dicts pass through unchanged when allowed).
    """
    eff_store, eff_subject, eff_labels = resolve_active_policy(policy_store, subject, labels)
    if eff_store is not None and eff_subject is not None:
        subjects = [eff_subject, *sorted(eff_labels)] if eff_labels else eff_subject
        decision = eff_store.check(subjects, "telemetry")
        if not decision.allowed:
            placeholder = "restricted by data-label policy"
            if decision.policy_name:
                placeholder += f" '{decision.policy_name}'"
            return {"_redacted": placeholder}

    return redact_dict(data) if isinstance(data, dict) else data
