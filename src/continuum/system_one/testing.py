"""A contract suite every System One adapter must pass.

Shipped in the package so third-party adapters can run the same checks as the
built-in ones::

    from continuum.system_one.testing import SystemOneContract

    class TestMyBackend(SystemOneContract):
        def make_classifier(self):
            return MyBackend(transport=fake_transport)

        def make_failing_classifier(self):      # optional
            return MyBackend(transport=always_500)

The suite checks the adapter directly, not through ``classify``: the shared layer
normalises and validates, and running the suite through it would hide exactly
the adapter bugs it exists to find.

``latency_budget_ms`` is what makes "System One" a checked property rather than
a label. Set it for your backend; remote backends need slack for network time.
"""

from __future__ import annotations

import math
import time
from typing import Any

from continuum.system_one.exceptions import SystemOneCapabilityError, SystemOneError
from continuum.system_one.types import (
    BinaryQuestion,
    ChoiceQuestion,
    Question,
    ScoreQuestion,
)

REFERENCE_STATE = (
    "The deploy failed twice and customers are seeing 500 errors. Can someone look now?"
)

_REFERENCE_QUESTIONS: dict[str, Question] = {
    "urgent": BinaryQuestion(instructions="Does this message need attention right now?"),
    "team": ChoiceQuestion(
        instructions="Which team should handle this message?",
        labels={
            "billing": "Payments, invoices, subscriptions.",
            "technical": "Outages, bugs, errors.",
            "none": "No listed team fits.",
        },
    ),
    "severity": ScoreQuestion(
        instructions="How severe is the reported problem?",
        levels=["Cosmetic.", "Degraded but usable.", "Customers are blocked."],
    ),
}

_TOLERANCE = 1e-3


def _outcomes(question: Question) -> set[Any]:
    if isinstance(question, BinaryQuestion):
        return {"true", "false"}
    if isinstance(question, ChoiceQuestion):
        return set(question.labels)
    return set(range(len(question.levels)))


class SystemOneContract:
    """Mix into a pytest test class and implement :meth:`make_classifier`."""

    latency_budget_ms: float = 500.0
    latency_runs: int = 5
    reference_state: Any = REFERENCE_STATE

    def make_classifier(self) -> Any:
        raise NotImplementedError

    def make_failing_classifier(self) -> Any | None:
        """A classifier whose backend fails (HTTP error, model error). Optional."""
        return None

    # -- helpers ----------------------------------------------------------

    def _supported(self, classifier: Any) -> dict[str, Question]:
        kinds = classifier.capabilities.question_types
        return {qid: q for qid, q in _REFERENCE_QUESTIONS.items() if q.kind in kinds}

    async def _ask(self, classifier: Any) -> tuple[dict[str, Question], Any]:
        questions = self._supported(classifier)
        state = self.reference_state
        if not classifier.capabilities.structured_state and not isinstance(state, str):
            state = str(state)
        return questions, await classifier.classify(state, questions)

    # -- the contract -----------------------------------------------------

    def test_declares_capabilities(self) -> None:
        caps = self.make_classifier().capabilities
        assert caps.question_types, "a backend must support at least one question kind"
        assert caps.question_types <= {"binary", "choice", "score"}
        assert caps.egress in ("remote", "local")

    def test_names_itself(self) -> None:
        classifier = self.make_classifier()
        assert isinstance(classifier.name, str) and classifier.name
        assert isinstance(classifier.model, str) and classifier.model

    async def test_answer_keys_match_question_ids(self) -> None:
        questions, raw = await self._ask(self.make_classifier())
        assert set(raw.distributions) == set(questions), (
            f"answered {sorted(raw.distributions)}, asked {sorted(questions)}"
        )

    async def test_probabilities_are_in_range_and_sum_to_one(self) -> None:
        questions, raw = await self._ask(self.make_classifier())
        for qid, question in questions.items():
            dist = raw.distributions[qid]
            keys = {int(k) for k in dist} if isinstance(question, ScoreQuestion) else set(dist)
            assert keys == _outcomes(question), f"'{qid}' has outcomes {sorted(map(str, dist))}"
            for p in dist.values():
                assert isinstance(p, int | float) and math.isfinite(p), f"'{qid}': {p!r}"
                assert 0.0 <= p <= 1.0, f"'{qid}' has probability {p}"
            assert abs(sum(dist.values()) - 1.0) <= _TOLERANCE, (
                f"'{qid}' sums to {sum(dist.values())}"
            )

    async def test_undeclared_kinds_are_refused(self) -> None:
        classifier = self.make_classifier()
        kinds = classifier.capabilities.question_types
        for qid, question in _REFERENCE_QUESTIONS.items():
            if question.kind in kinds:
                continue
            try:
                await classifier.classify(self.reference_state, {qid: question})
            except SystemOneCapabilityError:
                continue
            raise AssertionError(
                f"backend declares {sorted(kinds)} but answered a {question.kind} question"
            )

    async def test_failures_map_to_continuum_exceptions(self) -> None:
        classifier = self.make_failing_classifier()
        if classifier is None:
            return
        questions = self._supported(classifier)
        try:
            await classifier.classify(self.reference_state, questions)
        except SystemOneError:
            return
        except Exception as e:  # noqa: BLE001 - the point is to catch the wrong type
            raise AssertionError(
                f"backend raised {type(e).__name__}, not a SystemOneError subclass"
            ) from e
        raise AssertionError("the failing backend did not raise")

    async def test_meets_its_latency_budget(self) -> None:
        classifier = self.make_classifier()
        timings = []
        for _ in range(self.latency_runs):
            started = time.perf_counter()
            await self._ask(classifier)
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
        assert p95 <= self.latency_budget_ms, (
            f"p95 {p95:.1f} ms exceeds the {self.latency_budget_ms} ms budget"
        )
