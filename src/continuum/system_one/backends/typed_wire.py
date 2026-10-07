"""The typed-decision wire format shared by Jev and Laya.

TypeSafe's Jev and Convai's open-weight Laya speak the same format: three
question types (``noul`` / ``choice`` / ``score``) and answers carrying
``noul``, ``probabilities`` and ``confidence``. One translation serves both, so
a fix to it reaches every backend that speaks the format.

    BinaryQuestion  <->  noul    {"true": p, "false": 1-p}   <-  {"noul": p}
    ChoiceQuestion  <->  choice  labels -> probabilities      <-  {"probabilities", "confidence"}
    ScoreQuestion   <->  score   levels -> probabilities      <-  {"probabilities": {"0": p}}
"""

from __future__ import annotations

import math
from typing import Any

from continuum.system_one.exceptions import SystemOneCapabilityError, SystemOneResponseError
from continuum.system_one.types import (
    BinaryQuestion,
    ChoiceQuestion,
    Question,
    ScoreQuestion,
    SystemOneRawResult,
)

WIRE_TYPE = {"binary": "noul", "choice": "choice", "score": "score"}


def to_wire(qid: str, question: Question, *, vendor: str) -> dict[str, Any]:
    """Translate one Continuum question into the typed-decision wire format.

    Criteria are sent only when given: a binary question without them simply
    has no ``criteria`` key (LangChain's AutoModeMiddleware meant to send
    defaults and sent none).
    """
    if question.kind not in WIRE_TYPE:
        raise SystemOneCapabilityError(
            f"{vendor} cannot answer a {question.kind} question ('{qid}')."
        )
    wire: dict[str, Any] = {
        "type": WIRE_TYPE[question.kind],
        "instructions": question.instructions,
    }
    if isinstance(question, BinaryQuestion):
        if question.true_criteria is not None or question.false_criteria is not None:
            wire["criteria"] = {"true": question.true_criteria, "false": question.false_criteria}
    elif isinstance(question, ChoiceQuestion):
        wire["criteria"] = dict(question.labels)
    elif isinstance(question, ScoreQuestion):
        wire["criteria"] = list(question.levels)
    return wire


def parse_answers(
    body: Any, questions: dict[str, Question], model: str, *, vendor: str
) -> SystemOneRawResult:
    """Turn a typed-decision response body into raw distributions.

    Every asked question must come back with an answer of the right type;
    anything else is a :class:`SystemOneResponseError`, never a guess.
    """
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise SystemOneResponseError(f"{vendor}'s response has no answers object.")
    answers = body["answers"]
    distributions: dict[str, dict[Any, float]] = {}
    raw_confidence: dict[str, float] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        expected = WIRE_TYPE[question.kind]
        if not isinstance(answer, dict) or answer.get("type") != expected:
            raise SystemOneResponseError(f"{vendor} returned no {expected} answer for '{qid}'.")
        try:
            if isinstance(question, BinaryQuestion):
                p = float(answer["noul"])
                distributions[qid] = {"true": p, "false": 1.0 - p}
            elif isinstance(question, ChoiceQuestion):
                distributions[qid] = {str(k): float(v) for k, v in answer["probabilities"].items()}
            else:
                distributions[qid] = {int(k): float(v) for k, v in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise SystemOneResponseError(f"{vendor}'s answer for '{qid}' is malformed.") from e
        conf = answer.get("confidence")
        if isinstance(conf, int | float) and math.isfinite(conf):
            raw_confidence[qid] = float(conf)
    raw_usage = body.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return SystemOneRawResult(
        distributions=distributions,
        raw_confidence=raw_confidence,
        model=str(body.get("model") or model),
        usage={k: v for k, v in usage.items() if v is not None},
    )
