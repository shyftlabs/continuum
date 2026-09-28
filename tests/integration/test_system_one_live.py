"""Live System One backends: the real local NLI model, and Jev when a key is set.

Opt-in, because each needs something unit tests must not: the local run
downloads a ~180 MB model on first use, and the Jev run spends API credit.

    SYSTEM_ONE_LIVE_LOCAL=1 pytest tests/integration/test_system_one_live.py -m integration
    TYPESAFE_API_KEY=...    pytest tests/integration/test_system_one_live.py -m integration

Each backend runs the shipped contract suite for real (including its latency
budget, which is what "System One" is supposed to mean), plus a few judgements
whose right answer is not in doubt -- a backend that inverts entailment or reads
the wrong output column passes every shape check and fails these.
"""

from __future__ import annotations

import os

import pytest

from continuum.system_one.testing import SystemOneContract

pytestmark = pytest.mark.integration

LOCAL_MODEL = os.getenv("SYSTEM_ONE_LIVE_LOCAL_MODEL", "cross-encoder/nli-deberta-v3-small")

requires_local = pytest.mark.skipif(
    os.getenv("SYSTEM_ONE_LIVE_LOCAL") != "1",
    reason="set SYSTEM_ONE_LIVE_LOCAL=1 to download and run the local NLI model",
)
requires_jev = pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"), reason="set TYPESAFE_API_KEY to call Jev"
)

_local_backend = None


def _local():
    """One loaded model for the whole module: loading per test would measure
    model loading, not classification."""
    global _local_backend
    if _local_backend is None:
        from continuum.system_one.backends.local_nli import LocalNLIClassifier

        _local_backend = LocalNLIClassifier(model=LOCAL_MODEL)
    return _local_backend


async def _judgements(classifier) -> None:
    from continuum.system_one import BinaryQuestion, ChoiceQuestion, ScoreQuestion, classify

    urgent = BinaryQuestion(
        instructions="Does this message need attention right now?",
        true_criteria="Customers are affected right now.",
        false_criteria="Nothing is broken.",
    )
    team = ChoiceQuestion(
        instructions="Which team should handle this message?",
        labels={
            "billing": "This message is about a payment or an invoice.",
            "technical": "This message is about a software error or an outage.",
        },
    )
    severity = ScoreQuestion(
        instructions="How severe is the problem?",
        levels=[
            "There is no problem.",
            "Something is slightly inconvenient.",
            "Customers cannot use the product at all.",
        ],
    )
    outage = await classify(
        "Production is down and every customer is getting 500 errors.",
        {"urgent": urgent, "team": team, "severity": severity},
        classifier=classifier,
    )
    thanks = await classify(
        "Thanks for sorting out my invoice last week, all good now.",
        {"urgent": urgent, "team": team, "severity": severity},
        classifier=classifier,
    )

    assert outage.answers["urgent"].probability > thanks.answers["urgent"].probability
    assert outage.answers["team"].label == "technical"
    assert thanks.answers["team"].label == "billing"
    assert outage.answers["severity"].expected > thanks.answers["severity"].expected


@requires_local
class TestLocalNLILive(SystemOneContract):
    latency_budget_ms = 1000.0  # CPU, small DeBERTa, one batch of reference pairs

    def make_classifier(self):
        return _local()

    async def test_meets_its_latency_budget(self):
        # Warm the model first: the budget is for answering, not for loading.
        await self._ask(_local())
        await super().test_meets_its_latency_budget()

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(_local())


@requires_jev
class TestJevLive(SystemOneContract):
    latency_budget_ms = 3000.0  # remote: network time included

    def make_classifier(self):
        from continuum.system_one.backends.jev import JevClassifier

        return JevClassifier(model=os.getenv("SYSTEM_ONE_LIVE_JEV_MODEL", "jev-latest"))

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(self.make_classifier())
