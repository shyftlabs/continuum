"""System One fast path for the reflection and supervised quality gates.

With ``verdict_mode="system_one_classifier"`` a System One classifier is asked
whether a draft fully answers its request before the LLM judge (the critic, or
the supervisor) is. The classifier may only *approve*: P(pass) at or above the
threshold passes the draft with no judge call. Anything else -- a lower P, a
backend error, the kill switch -- returns None and the LLM judge decides exactly
as it would without the classifier, so a fail and its feedback never come from
the classifier. The one new risk is a confident false pass, which is what the
conservative default threshold (0.9) is set against.
"""

from __future__ import annotations

from typing import Literal

from continuum.logging import get_logger
from continuum.system_one import BinaryQuestion, SystemOneError, classify, system_one_disabled

logger = get_logger(__name__)

VerdictMode = Literal["llm", "system_one_classifier"]
VERDICT_MODES: frozenset[str] = frozenset({"llm", "system_one_classifier"})

# The true-criterion is a statement, so a local NLI backend has something a
# premise can entail. Measured on openrouter:typesafe/jev-1.13 (22 hand-written
# cases x 2 runs): good drafts 0.87-0.99, subtly bad ones 0.01-0.13.
QUALITY_QUESTION = BinaryQuestion(
    instructions="Judging by `draft`, does it fully and correctly answer `request`?",
    true_criteria=(
        "The draft fully and correctly answers the request: every part asked for is "
        "present, the facts are right, and any requested format or length is followed."
    ),
    false_criteria=(
        "The draft misses part of the request, contains a factual error, ignores the "
        "requested format or length, is cut off, refuses, or answers a different question."
    ),
)


def validate_gate_settings(verdict_mode: str, threshold: float) -> None:
    """Reject a verdict_mode or pass threshold the gate cannot act on."""
    if verdict_mode not in VERDICT_MODES:
        raise ValueError(
            f"verdict_mode must be one of {sorted(VERDICT_MODES)}, got {verdict_mode!r}"
        )
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"system_one_pass_threshold must be in (0, 1], got {threshold}")


async def system_one_approves(
    *,
    seam: str,
    request: str,
    draft: str,
    backend: str | None,
    threshold: float,
    review_criteria: str | None = None,
) -> float | None:
    """P(pass) when the classifier confidently approves ``draft``, else None.

    None means "let the LLM judge decide": P(pass) was below ``threshold``, the
    classifier failed, or SYSTEM_ONE_DISABLED is set. ``review_criteria`` is a
    task-specific checklist, sent alongside the request when there is one.
    """
    if system_one_disabled():
        logger.info("%s quality gate decided_by=legacy (SYSTEM_ONE_DISABLED)", seam)
        return None

    state = {"request": request, "draft": draft}
    if review_criteria:
        state["review_criteria"] = review_criteria
    try:
        resp = await classify(state, {"passes": QUALITY_QUESTION}, spec=backend)
    except SystemOneError as e:
        logger.warning(
            "%s: System One quality check failed (%s); using the LLM judge",
            seam,
            type(e).__name__,
        )
        return None

    p_pass = resp.binary("passes").probability
    approved = p_pass >= threshold
    logger.info(
        "%s quality gate decided_by=system_one backend=%s p_pass=%.3f threshold=%s -> %s",
        seam,
        resp.provenance.backend,
        p_pass,
        threshold,
        "pass" if approved else "ask the LLM judge",
    )
    return p_pass if approved else None
