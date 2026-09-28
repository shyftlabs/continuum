"""Shared fixtures for the System One layer.

``FakeClassifier`` is a backend whose answers are scripted per question kind,
so a test can state exactly what the backend "said" and check what the shared
layer made of it -- without a network, a model download, or a vendor key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


class FakeClassifier:
    """A scripted System One backend.

    ``answer`` receives each question the layer sends and returns its raw
    distribution. The default answers every kind with a fixed, legal
    distribution, so tests that only care about plumbing need not script one.
    """

    name = "fake"

    def __init__(
        self,
        *,
        kinds: frozenset[str] = frozenset({"binary", "choice", "score"}),
        structured_state: bool = True,
        egress: str = "local",
        max_questions: int | None = None,
        model: str = "fake-1",
        answer: Callable[[str, Any], dict[Any, float]] | None = None,
        raw_confidence: float | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        from continuum.system_one import SystemOneCapabilities

        self.model = model
        self.capabilities = SystemOneCapabilities(
            question_types=kinds,
            max_questions=max_questions,
            structured_state=structured_state,
            egress=egress,  # type: ignore[arg-type]
            calibrated=False,
        )
        self._answer = answer or _default_answer
        self._raw_confidence = raw_confidence
        self._usage = usage or {}
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def classify(self, state: Any, questions: dict[str, Any]) -> Any:
        from continuum.system_one import SystemOneCapabilityError, SystemOneRawResult

        self.calls.append((state, dict(questions)))
        for qid, q in questions.items():
            if q.kind not in self.capabilities.question_types:
                raise SystemOneCapabilityError(f"fake backend cannot answer {q.kind} ({qid})")
        return SystemOneRawResult(
            distributions={qid: self._answer(qid, q) for qid, q in questions.items()},
            raw_confidence=(
                {qid: self._raw_confidence for qid in questions}
                if self._raw_confidence is not None
                else {}
            ),
            model=self.model,
            usage=self._usage,
        )


def _default_answer(qid: str, q: Any) -> dict[Any, float]:
    if q.kind == "binary":
        return {"true": 0.8, "false": 0.2}
    if q.kind == "choice":
        labels = list(q.labels)
        rest = 0.4 / (len(labels) - 1)
        return {label: (0.6 if i == 0 else rest) for i, label in enumerate(labels)}
    levels = len(q.levels)
    return {i: 1.0 / levels for i in range(levels)}


@pytest.fixture
def fake_classifier_cls() -> type[FakeClassifier]:
    return FakeClassifier


@pytest.fixture(autouse=True)
def _clean_system_one_state(monkeypatch):
    """Every test starts with no backend configured, the kill switch off, and no
    cached classifier instances -- so a test cannot pass on state another left."""
    from continuum.config import settings
    from continuum.core.container import reset_container

    monkeypatch.setattr(settings, "system_one_backend", None)
    monkeypatch.setattr(settings, "system_one_disabled", False)
    reset_container()
    from continuum.system_one.registry import clear_classifier_cache

    clear_classifier_cache()
    yield
    clear_classifier_cache()
    reset_container()
