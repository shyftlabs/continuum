"""Safe metadata shared by workflow exception wrappers."""

from __future__ import annotations

from hashlib import sha256


def workflow_error_context(
    run_id: str | None,
    cause: BaseException,
) -> dict[str, str]:
    """Return correlatable diagnostics without retaining raw failure text."""

    context = {"root_cause_type": type(cause).__name__}
    if run_id:
        correlation = sha256(run_id.encode("utf-8")).hexdigest()[:16]
        context["run_correlation"] = f"sha256:{correlation}"
    return context
