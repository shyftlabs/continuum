"""Local NLI backend: ``local:<hf-model-id>``, e.g. ``local:cross-encoder/nli-deberta-v3-small``.

A natural-language-inference cross-encoder scores (premise, hypothesis) pairs as
contradiction / entailment / neutral. Here the state is the premise, and each
question becomes hypotheses:

* **binary**: its ``true_criteria`` when that is a statement, else its
  instructions. NLI judges statements, not questions -- "The task is complete."
  can be entailed by a premise, "Is the task complete?" cannot -- so seams give
  their binary questions a declarative true-criterion.
  ``P(true) = entailment / (entailment + contradiction)``; neutral is dropped.
* **choice**: one hypothesis per label (its description, else the label itself);
  the entailment probabilities are normalised across labels.
* **score**: not native. The shared layer asks it as a choice over the levels.

Runs on this machine, so it declares egress ``local``: the backend a run tainted
with ``phi`` can still use when a remote one is denied. It is text-only; the
shared layer sends JSON state as JSON text.

Needs ``sentence-transformers`` (the ``[embeddings]`` extra). The model is loaded
on first use, once, in a worker thread; constructing the backend loads nothing.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from continuum.logging import get_logger
from continuum.system_one.exceptions import (
    SystemOneBackendError,
    SystemOneCapabilityError,
    SystemOneNotConfiguredError,
)
from continuum.system_one.types import (
    BinaryQuestion,
    ChoiceQuestion,
    Question,
    SystemOneCapabilities,
    SystemOneRawResult,
)

logger = get_logger(__name__)

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


class LocalNLIClassifier:
    """An NLI cross-encoder running in-process as a System One backend."""

    name = "local"
    capabilities = SystemOneCapabilities(
        question_types=frozenset({"binary", "choice"}),
        max_questions=None,
        structured_state=False,
        egress="local",
        calibrated=False,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cross_encoder: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self._device = device
        self._ce = cross_encoder
        self._label_index: dict[str, int] | None = None
        self._load_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"LocalNLIClassifier(model={self.model!r})"

    def _load(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise SystemOneNotConfiguredError(
                "The local System One backend needs sentence-transformers: "
                "pip install 'shyftlabs-continuum[embeddings]'.",
                config_key="system_one_backend",
            ) from e
        logger.info("Loading local System One NLI model %s", self.model)
        return CrossEncoder(self.model, device=self._device)

    def _encoder(self) -> Any:
        with self._load_lock:
            if self._ce is None:
                self._ce = self._load()
            if self._label_index is None:
                self._label_index = _label_index(self._ce, self.model)
            return self._ce

    async def classify(self, state: Any, questions: dict[str, Question]) -> SystemOneRawResult:
        premise = _text(state)
        pairs: list[tuple[str, str]] = []
        spans: dict[str, tuple[int, list[str]]] = {}
        for qid, question in questions.items():
            if isinstance(question, BinaryQuestion):
                hypothesis = (
                    question.true_criteria
                    if isinstance(question.true_criteria, str) and question.true_criteria.strip()
                    else _text(question.instructions)
                )
                spans[qid] = (len(pairs), ["true"])
                pairs.append((premise, hypothesis))
            elif isinstance(question, ChoiceQuestion):
                labels = list(question.labels)
                spans[qid] = (len(pairs), labels)
                for label in labels:
                    description = question.labels[label]
                    hypothesis = (
                        description
                        if isinstance(description, str) and description.strip()
                        else label
                    )
                    pairs.append((premise, hypothesis))
            else:
                raise SystemOneCapabilityError(
                    f"The local NLI backend cannot answer a {question.kind} question ('{qid}')."
                )

        probs = await asyncio.to_thread(self._predict, pairs)
        assert self._label_index is not None
        ent, con = self._label_index["entailment"], self._label_index["contradiction"]

        distributions: dict[str, dict[Any, float]] = {}
        for qid, (start, outcomes) in spans.items():
            question = questions[qid]
            if isinstance(question, BinaryQuestion):
                e, c = float(probs[start][ent]), float(probs[start][con])
                total = e + c
                p = e / total if total > 0 else 0.5
                distributions[qid] = {"true": p, "false": 1.0 - p}
            else:
                scores = {label: float(probs[start + i][ent]) for i, label in enumerate(outcomes)}
                total = sum(scores.values())
                distributions[qid] = {
                    label: (s / total if total > 0 else 1.0 / len(scores))
                    for label, s in scores.items()
                }
        return SystemOneRawResult(distributions=distributions, model=self.model)

    def _predict(self, pairs: list[tuple[str, str]]) -> Any:
        encoder = self._encoder()
        try:
            return encoder.predict(pairs, apply_softmax=True, show_progress_bar=False)
        except Exception as e:
            raise SystemOneBackendError(
                f"The local NLI model failed ({type(e).__name__}).", backend=self.name
            ) from e


def _label_index(encoder: Any, model: str) -> dict[str, int]:
    """Where 'entailment' and 'contradiction' sit in this model's output.

    Read from the model's own config: models disagree on the order, and a
    hard-coded index would read contradiction as agreement on some of them.
    """
    id2label = None
    for owner in (encoder, getattr(encoder, "model", None)):
        config = getattr(owner, "config", None)
        id2label = getattr(config, "id2label", None)
        if id2label:
            break
    index = {str(name).lower(): int(i) for i, name in (id2label or {}).items()}
    if "entailment" not in index or "contradiction" not in index:
        raise SystemOneNotConfiguredError(
            f"Model {model!r} does not label its outputs 'entailment'/'contradiction'; "
            "the local System One backend needs an NLI cross-encoder.",
            config_key="system_one_backend",
        )
    return index
