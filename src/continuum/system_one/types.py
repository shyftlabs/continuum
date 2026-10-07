"""Question, answer and capability types for the System One layer.

"System One" in Kahneman's sense: fast, cheap, typed judgement, as opposed to the
slow reasoning of a general-purpose LLM. A System One classifier takes a state
and a set of named questions and returns a probability distribution for each.
It never generates text.

The vocabulary is Continuum's own, not any vendor's. Jev calls a yes/no question
a ``Noul``; here it is a :class:`BinaryQuestion`, and the Jev adapter translates.
Keeping the names neutral is what lets a second backend plug in without every
seam learning a second vocabulary.

Two layers of result exist on purpose:

* :class:`SystemOneRawResult` is what a backend returns: one raw distribution per
  question, in the backend's own scale, plus whatever confidence the backend
  reports.
* :class:`SystemOneResponse` is what a seam receives from
  :func:`continuum.system_one.classify`: validated, normalised distributions,
  the backend's own confidence where it reports one, and provenance.

A backend therefore never has to get normalisation right, and a seam never sees a
distribution that was not checked. Continuum computes no confidence of its own:
backends are calibrated differently, so a shared formula would not make equal
numbers mean equal certainty. A seam that gates on confidence uses the backend's,
with a threshold set for that backend; ``probabilities`` stay for any other measure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

QuestionKind = Literal["binary", "choice", "score"]

# What a question may say, and what state may be: text or JSON-shaped data.
QuestionContent = str | dict[str, Any] | list[Any]
State = str | dict[str, Any] | list[Any]


def _require_instructions(instructions: QuestionContent) -> None:
    if isinstance(instructions, str) and not instructions.strip():
        raise ValueError("A System One question needs non-empty instructions.")
    if not isinstance(instructions, str) and not instructions:
        raise ValueError("A System One question needs non-empty instructions.")


@dataclass(frozen=True)
class BinaryQuestion:
    """A yes/no judgement. The answer is the probability that it is true.

    ``true_criteria`` / ``false_criteria`` optionally say what should count as
    each outcome when the instructions alone leave room for interpretation.
    """

    instructions: QuestionContent
    true_criteria: Any = None
    false_criteria: Any = None

    kind: ClassVar[QuestionKind] = "binary"

    def __post_init__(self) -> None:
        _require_instructions(self.instructions)


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one label from a fixed set. ``labels`` maps each label to an optional
    description of when it applies. Include a ``none`` label when the set may
    not cover every input."""

    instructions: QuestionContent
    labels: Mapping[str, Any]

    kind: ClassVar[QuestionKind] = "choice"

    def __post_init__(self) -> None:
        _require_instructions(self.instructions)
        # Copied so a caller editing its dict after asking cannot change a
        # question that is already in flight or recorded.
        labels = dict(self.labels)
        if len(labels) < 2:
            raise ValueError("A ChoiceQuestion needs at least two labels.")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class ScoreQuestion:
    """Place the state on an ordered scale. ``levels`` are described in order,
    and numbered from zero; the answer's ``expected`` value may fall between
    levels."""

    instructions: QuestionContent
    levels: Sequence[Any]

    kind: ClassVar[QuestionKind] = "score"

    def __post_init__(self) -> None:
        _require_instructions(self.instructions)
        levels = list(self.levels)
        if len(levels) < 2:
            raise ValueError("A ScoreQuestion needs at least two levels.")
        object.__setattr__(self, "levels", levels)


Question = BinaryQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class SystemOneCapabilities:
    """What a backend can do, declared so the shared layer can plan around it.

    ``egress`` is exposed to the policy check: ``remote`` means the state leaves
    the process for a third party, ``local`` means it does not. ``calibrated``
    records whether the backend *claims* calibrated probabilities -- a claim, not
    something the SDK has verified.
    """

    question_types: frozenset[QuestionKind]
    max_questions: int | None = None
    structured_state: bool = True
    egress: Literal["remote", "local"] = "remote"
    calibrated: bool = False


@dataclass
class SystemOneRawResult:
    """A backend's answer, before the shared layer checks it.

    ``distributions`` maps each question ID to its outcomes: ``{"true", "false"}``
    for a binary question, the labels for a choice, the integer levels for a
    score. ``raw_confidence`` is the backend's own confidence where it reports
    one; it becomes the answer's ``confidence``.
    """

    distributions: dict[str, dict[Any, float]]
    raw_confidence: dict[str, float] = field(default_factory=dict)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BinaryAnswer:
    probability: float
    confidence: float | None = None  # the backend's own; None if it reports none
    filled_in: bool = False

    kind: ClassVar[QuestionKind] = "binary"

    @property
    def probabilities(self) -> dict[str, float]:
        return {"true": self.probability, "false": 1.0 - self.probability}


@dataclass(frozen=True)
class ChoiceAnswer:
    label: str
    probabilities: dict[str, float]
    confidence: float | None = None  # the backend's own; None if it reports none
    filled_in: bool = False

    kind: ClassVar[QuestionKind] = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    expected: float
    probabilities: dict[int, float]
    confidence: float | None = None  # the backend's own; None if it reports none
    filled_in: bool = False

    kind: ClassVar[QuestionKind] = "score"


Answer = BinaryAnswer | ChoiceAnswer | ScoreAnswer
_A = TypeVar("_A", BinaryAnswer, ChoiceAnswer, ScoreAnswer)


@dataclass(frozen=True)
class Provenance:
    """Who answered, so a trace can say which backend made a decision."""

    backend: str
    model: str
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemOneResponse:
    answers: dict[str, Answer]
    provenance: Provenance

    # Typed accessors, so a seam reads ``resp.choice("route").label`` without
    # narrowing the Answer union itself.

    def binary(self, qid: str) -> BinaryAnswer:
        return self._typed(qid, BinaryAnswer)

    def choice(self, qid: str) -> ChoiceAnswer:
        return self._typed(qid, ChoiceAnswer)

    def score(self, qid: str) -> ScoreAnswer:
        return self._typed(qid, ScoreAnswer)

    def _typed(self, qid: str, cls: type[_A]) -> _A:
        answer = self.answers[qid]
        if not isinstance(answer, cls):
            raise TypeError(f"Answer '{qid}' is a {answer.kind} answer, not {cls.kind}.")
        return answer
