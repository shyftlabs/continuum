"""The shared layer every seam calls: :func:`classify`.

A seam asks typed questions and gets validated distributions back. Between the
two, this module does the work no backend should have to get right on its own:

1. **Kill switch.** ``SYSTEM_ONE_DISABLED`` refuses here too, so a direct call
   cannot bypass the switch ops flipped during an incident.
2. **Egress check.** The run's data-label policy is asked about
   ``system_one:<egress>:<backend>:<model>`` before any state is sent, the same
   way ``LLMClient`` asks about ``llm:<model>``. A ``phi``-tainted run can be
   denied ``system_one:remote:*`` and still use a local backend.
3. **Filling in kinds.** A backend without score support gets a score as a
   choice over its levels; one without choice support gets a choice as one
   binary question per label. Done once here, flagged ``filled_in``, because a
   derived distribution may be less well calibrated than a native one.
4. **State.** A text-only backend receives JSON state as JSON text.
5. **Validation and confidence.** Every distribution is checked (right outcome
   keys, finite, non-negative, not all zero) and normalised. Confidence is
   ``1 - normalised entropy``, computed here, so it means the same whichever
   backend answered; the backend's own confidence is kept as ``raw_confidence``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from continuum.config import settings
from continuum.logging import get_logger
from continuum.protocols import ISystemOneClassifier
from continuum.system_one.exceptions import (
    SystemOneAccessDeniedError,
    SystemOneCapabilityError,
    SystemOneDisabledError,
    SystemOneResponseError,
)
from continuum.system_one.types import (
    Answer,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    Provenance,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
    SystemOneResponse,
)

logger = get_logger(__name__)

# Separates a question ID from the label or level a filled-in sub-question
# stands for: "team::billing", "frustration::2".
_SUB = "::"


def system_one_disabled() -> bool:
    """Is the global kill switch on? Read live, so flipping it needs no restart."""
    return bool(getattr(settings, "system_one_disabled", False))


def egress_resource(classifier: ISystemOneClassifier) -> str:
    """The policy resource a call to ``classifier`` is checked against."""
    return f"system_one:{classifier.capabilities.egress}:{classifier.name}:{classifier.model}"


async def classify(
    state: State,
    questions: dict[str, Question],
    *,
    classifier: ISystemOneClassifier | None = None,
    spec: str | None = None,
) -> SystemOneResponse:
    """Ask ``questions`` about ``state`` and return validated answers.

    ``classifier`` is used as given; otherwise ``spec`` (e.g. ``"jev:jev-latest"``)
    is resolved, and with neither, the container/settings default is. Raises a
    :class:`~continuum.system_one.SystemOneError` subclass whenever no trusted
    answer exists -- seams decide what that means for them.
    """
    if not questions:
        raise ValueError("classify() needs at least one question.")
    if system_one_disabled():
        raise SystemOneDisabledError()

    if classifier is None:
        from continuum.system_one.registry import resolve_classifier

        classifier = resolve_classifier(spec)

    _enforce_egress(classifier)

    caps = classifier.capabilities
    wire: dict[str, Question] = {}
    for qid, question in questions.items():
        wire.update(_plan(qid, question, caps.question_types))

    sent_state: Any = state
    if not caps.structured_state and not isinstance(state, str):
        sent_state = json.dumps(state, ensure_ascii=False, default=str)

    started = time.perf_counter()
    distributions: dict[str, dict[Any, float]] = {}
    raw_confidence: dict[str, float] = {}
    usage: dict[str, Any] = {}
    model = ""
    for batch in _batches(wire, caps.max_questions):
        raw = await classifier.classify(sent_state, batch)
        distributions.update(raw.distributions)
        raw_confidence.update(raw.raw_confidence or {})
        _merge_usage(usage, raw.usage or {})
        model = raw.model or model
    latency_ms = (time.perf_counter() - started) * 1000

    answers = {
        qid: _answer(qid, question, caps.question_types, distributions, raw_confidence)
        for qid, question in questions.items()
    }
    return SystemOneResponse(
        answers=answers,
        provenance=Provenance(
            backend=classifier.name,
            model=model or classifier.model,
            latency_ms=latency_ms,
            usage=usage,
        ),
    )


# ---------------------------------------------------------------------------
# Egress
# ---------------------------------------------------------------------------


def _enforce_egress(classifier: ISystemOneClassifier) -> None:
    from continuum.security.policy_context import resolve_active_policy

    store, subject, labels = resolve_active_policy(None, None, None)
    if store is None or subject is None:
        return
    resource = egress_resource(classifier)
    subjects = [subject, *sorted(labels)] if labels else subject
    decision = store.check(subjects, resource)
    if not decision.allowed:
        raise SystemOneAccessDeniedError(
            resource,
            policy_name=decision.policy_name,
            denial_message=decision.denial_message,
        )


# ---------------------------------------------------------------------------
# Planning: which questions actually go to the backend
# ---------------------------------------------------------------------------


def _plan(qid: str, question: Question, kinds: frozenset[str]) -> dict[str, Question]:
    if question.kind in kinds:
        return {qid: question}
    if isinstance(question, ScoreQuestion):
        as_choice = _score_as_choice(question)
        if "choice" in kinds:
            return {qid: as_choice}
        if "binary" in kinds:
            return _choice_as_binaries(qid, as_choice)
    if isinstance(question, ChoiceQuestion) and "binary" in kinds:
        return _choice_as_binaries(qid, question)
    raise SystemOneCapabilityError(
        f"Backend supports {sorted(kinds)}; a {question.kind} question ('{qid}') "
        "cannot be answered or derived from them."
    )


def _score_as_choice(question: ScoreQuestion) -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions=question.instructions,
        labels={str(i): level for i, level in enumerate(question.levels)},
    )


def _choice_as_binaries(qid: str, question: ChoiceQuestion) -> dict[str, Question]:
    return {
        f"{qid}{_SUB}{label}": BinaryQuestion(
            instructions={
                "question": question.instructions,
                "candidate_answer": label,
                "candidate_description": description,
                "judge": "Is the candidate answer the correct answer to the question?",
            },
        )
        for label, description in question.labels.items()
    }


def _batches(wire: dict[str, Question], max_questions: int | None) -> list[dict[str, Question]]:
    if not max_questions or len(wire) <= max_questions:
        return [wire]
    items = list(wire.items())
    return [dict(items[i : i + max_questions]) for i in range(0, len(items), max_questions)]


def _merge_usage(total: dict[str, Any], part: dict[str, Any]) -> None:
    """Add counts across batches; keep flags as flags, true if any batch set one.

    ``bool`` is checked first because it is an ``int`` in Python: summing it
    turned a backend's ``truncated: True`` into ``truncated: 1``.
    """
    for key, value in part.items():
        if isinstance(value, bool):
            total[key] = bool(total.get(key, False)) or value
        elif isinstance(value, int | float) and isinstance(total.get(key, 0), int | float):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


# ---------------------------------------------------------------------------
# Answers: validate, normalise, compute confidence
# ---------------------------------------------------------------------------


def _answer(
    qid: str,
    question: Question,
    kinds: frozenset[str],
    distributions: dict[str, dict[Any, float]],
    raw_confidence: dict[str, float],
) -> Answer:
    filled = question.kind not in kinds
    raw_conf = None if filled else raw_confidence.get(qid)

    if isinstance(question, BinaryQuestion):
        dist = _normalise(qid, _get(qid, distributions), {"true", "false"})
        return BinaryAnswer(
            probability=dist["true"],
            confidence=_confidence(dist),
            raw_confidence=raw_conf,
        )

    if isinstance(question, ChoiceQuestion):
        dist = _choice_distribution(qid, question, kinds, distributions)
        return ChoiceAnswer(
            label=max(dist, key=lambda label: dist[label]),
            probabilities=dist,
            confidence=_confidence(dist),
            raw_confidence=raw_conf,
            filled_in=filled,
        )

    as_choice = _score_as_choice(question)
    if "score" in kinds:
        raw = _get(qid, distributions)
        by_level = _normalise(qid, _int_keys(qid, raw), set(range(len(question.levels))))
    else:
        by_label = _choice_distribution(qid, as_choice, kinds, distributions)
        by_level = {int(label): p for label, p in by_label.items()}
    return ScoreAnswer(
        expected=sum(level * p for level, p in by_level.items()),
        probabilities=by_level,
        confidence=_confidence(by_level),
        raw_confidence=raw_conf,
        filled_in=filled,
    )


def _choice_distribution(
    qid: str,
    question: ChoiceQuestion,
    kinds: frozenset[str],
    distributions: dict[str, dict[Any, float]],
) -> dict[str, float]:
    labels = set(question.labels)
    if "choice" in kinds:
        return _normalise(qid, _get(qid, distributions), labels)
    # Derived from one binary per label: each label's P(true), renormalised.
    yes = {}
    for label in question.labels:
        sub = f"{qid}{_SUB}{label}"
        yes[label] = _normalise(sub, _get(sub, distributions), {"true", "false"})["true"]
    return _normalise(qid, yes, labels)


def _get(qid: str, distributions: dict[str, dict[Any, float]]) -> dict[Any, float]:
    if qid not in distributions:
        raise SystemOneResponseError(f"The backend returned no answer for '{qid}'.")
    return distributions[qid]


def _int_keys(qid: str, dist: dict[Any, float]) -> dict[int, float]:
    try:
        return {int(k): v for k, v in dist.items()}
    except (TypeError, ValueError) as e:
        raise SystemOneResponseError(f"Score levels for '{qid}' are not integers.") from e


def _normalise(qid: str, dist: dict[Any, float], expected: set[Any]) -> dict[Any, float]:
    if set(dist) != expected:
        raise SystemOneResponseError(
            f"The answer for '{qid}' has outcomes {sorted(map(str, dist))}, "
            f"expected {sorted(map(str, expected))}."
        )
    for value in dist.values():
        if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise SystemOneResponseError(f"The answer for '{qid}' has an invalid probability.")
        if value > 1 and len(expected) == 2:
            # A binary P above 1 is not an unnormalised score, it is wrong.
            raise SystemOneResponseError(f"The answer for '{qid}' has a probability above 1.")
    total = sum(dist.values())
    if total <= 0:
        raise SystemOneResponseError(f"The answer for '{qid}' is all zeros.")
    return {k: v / total for k, v in dist.items()}


def _confidence(dist: dict[Any, float]) -> float:
    """1 - normalised Shannon entropy: 0 for an even split, 1 for certainty."""
    n = len(dist)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))
