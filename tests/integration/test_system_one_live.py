"""Live System One backends: local NLI, Laya, and Jev when a key is set.

Opt-in, because each needs something unit tests must not: the local runs
download a model on first use (NLI ~0.5 GB, Laya ~1 GB) and the Jev run spends
API credit.

    SYSTEM_ONE_LIVE_LOCAL=1     pytest tests/integration/test_system_one_live.py -m integration
    SYSTEM_ONE_LIVE_LAYA=1      pytest ... (needs the [laya] extra)
    SYSTEM_ONE_LIVE_LAYA_MLX=1  pytest ... (needs [laya-mlx]: Apple Silicon only)
    TYPESAFE_API_KEY=...        pytest tests/integration/test_system_one_live.py -m integration
    OPENROUTER_API_KEY=...      pytest ... (Jev through OpenRouter's Decisions API)

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

requires_openrouter = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="set OPENROUTER_API_KEY to call Jev via OpenRouter"
)
requires_laya = pytest.mark.skipif(
    os.getenv("SYSTEM_ONE_LIVE_LAYA") != "1",
    reason="set SYSTEM_ONE_LIVE_LAYA=1 (and install [laya]) to run the Laya model",
)
requires_laya_mlx = pytest.mark.skipif(
    os.getenv("SYSTEM_ONE_LIVE_LAYA_MLX") != "1",
    reason="set SYSTEM_ONE_LIVE_LAYA_MLX=1 (Apple Silicon, [laya-mlx]) to run Laya on MLX",
)

_local_backend = None
_laya_backends: dict[str, object] = {}


def _laya(kind: str):
    """One loaded model per kind for the whole module (see _local)."""
    if kind not in _laya_backends:
        from continuum.system_one.backends.laya import LayaClassifier, LayaMLXClassifier

        cls = LayaMLXClassifier if kind == "mlx" else LayaClassifier
        _laya_backends[kind] = cls()
    return _laya_backends[kind]


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


@requires_laya
class TestLayaLive(SystemOneContract):
    latency_budget_ms = 1000.0  # local torch; MPS/CUDA far faster than CPU

    def make_classifier(self):
        return _laya("upstream")

    async def test_meets_its_latency_budget(self):
        await self._ask(_laya("upstream"))  # warm: the budget is for answering
        await super().test_meets_its_latency_budget()

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(_laya("upstream"))


@requires_laya_mlx
class TestLayaMLXLive(SystemOneContract):
    latency_budget_ms = 500.0

    def make_classifier(self):
        return _laya("mlx")

    async def test_meets_its_latency_budget(self):
        await self._ask(_laya("mlx"))
        await super().test_meets_its_latency_budget()

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(_laya("mlx"))


@requires_jev
class TestJevLive(SystemOneContract):
    latency_budget_ms = 3000.0  # remote: network time included

    def make_classifier(self):
        from continuum.system_one.backends.jev import JevClassifier

        return JevClassifier(model=os.getenv("SYSTEM_ONE_LIVE_JEV_MODEL", "jev-latest"))

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(self.make_classifier())


@requires_openrouter
class TestJevOpenRouterLive(SystemOneContract):
    latency_budget_ms = 3000.0  # remote, through OpenRouter's edge

    def make_classifier(self):
        from continuum.system_one.backends.jev import JevOpenRouterClassifier

        return JevOpenRouterClassifier(
            model=os.getenv("SYSTEM_ONE_LIVE_OPENROUTER_MODEL", "typesafe/jev-1.13")
        )

    async def test_judgements_whose_answer_is_not_in_doubt(self):
        await _judgements(self.make_classifier())
