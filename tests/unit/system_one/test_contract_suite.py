"""The contract suite runs against the fake backend, and catches a broken one.

The suite is shipped (``continuum.system_one.testing``) so third-party adapters
can run it too. Running it here against a correct fake proves the suite passes
when it should; the broken-backend tests prove it fails when it should, which a
suite that only ever passed would never show.
"""

from __future__ import annotations

import pytest

from continuum.system_one.testing import SystemOneContract
from tests.unit.system_one.conftest import FakeClassifier


class _Erroring(FakeClassifier):
    async def classify(self, state, questions):
        from continuum.system_one import SystemOneBackendError

        raise SystemOneBackendError("backend unavailable", backend=self.name)


class TestFakeBackendMeetsTheContract(SystemOneContract):
    def make_classifier(self):
        return FakeClassifier()

    def make_failing_classifier(self):
        return _Erroring()


class TestTheSuiteCatchesABrokenBackend:
    async def test_a_backend_that_drops_answers_fails(self):
        class Dropping(FakeClassifier):
            async def classify(self, state, questions):
                raw = await super().classify(state, questions)
                raw.distributions.clear()
                return raw

        suite = TestFakeBackendMeetsTheContract()
        suite.make_classifier = Dropping  # type: ignore[method-assign]
        with pytest.raises(AssertionError):
            await suite.test_answer_keys_match_question_ids()

    async def test_a_backend_that_leaks_raw_exceptions_fails(self):
        class Leaking(FakeClassifier):
            async def classify(self, state, questions):
                raise RuntimeError("socket closed")

        suite = TestFakeBackendMeetsTheContract()
        suite.make_failing_classifier = Leaking  # type: ignore[method-assign]
        with pytest.raises(AssertionError):
            await suite.test_failures_map_to_continuum_exceptions()

    async def test_a_backend_over_its_latency_budget_fails(self):
        import asyncio

        class Slow(FakeClassifier):
            async def classify(self, state, questions):
                await asyncio.sleep(0.05)
                return await super().classify(state, questions)

        suite = TestFakeBackendMeetsTheContract()
        suite.make_classifier = Slow  # type: ignore[method-assign]
        suite.latency_budget_ms = 10
        with pytest.raises(AssertionError):
            await suite.test_meets_its_latency_budget()

    async def test_a_backend_that_answers_an_undeclared_kind_fails(self):
        """Declaring binary-only but answering choice anyway means capabilities
        lie, and the shared layer's gap-filling decisions are made on a lie."""

        class Overclaiming(FakeClassifier):
            def __init__(self):
                super().__init__(kinds=frozenset({"binary"}))

            async def classify(self, state, questions):
                from continuum.system_one import SystemOneRawResult

                return SystemOneRawResult(
                    distributions={qid: {"a": 0.5, "b": 0.5} for qid in questions},
                    raw_confidence={},
                    model=self.model,
                    usage={},
                )

        suite = TestFakeBackendMeetsTheContract()
        suite.make_classifier = Overclaiming  # type: ignore[method-assign]
        with pytest.raises(AssertionError):
            await suite.test_undeclared_kinds_are_refused()
