"""Laya backends: Convai's open-weight typed-decision model, run in-process.

``laya:<checkpoint>`` runs upstream ``laya`` (PyPI; Apache-2.0 code and weights;
CPU, CUDA, MPS or XPU) -- the ``[laya]`` extra. ``laya-mlx:<checkpoint>`` runs
the independent MLX port on Apple Silicon only -- the ``[laya-mlx]`` extra. The
two share an API and answer format; on the same checkpoint they were measured
to agree within 0.0018.

Laya speaks the typed-decision wire format Jev does (``typed_wire``), so this
module holds only what is Laya's own:

* **Local and in-process.** Egress ``local``: a run labelled ``phi`` that is
  denied a remote backend can still use it. The model loads lazily, once, and
  every prediction runs in a worker thread, off the event loop.
* **Structured state.** Laya accepts text, JSON objects and conversation lists,
  so state is passed through as-is.
* **Model name.** Laya reports ``"laya-rl-agent"`` for every checkpoint; the
  checkpoint id is what a trace needs, so that is what is reported.
* **Silent truncation.** Laya cuts state that does not fit its context (512
  tokens for the English checkpoint) without saying so -- from the right for
  text and JSON, from the left for conversation lists. ``usage.input_tokens`` is
  summed over one row per question, each capped at ``max_len``; when every row
  is at the cap the state did not fit, and the result is flagged
  ``usage["truncated"] = True`` and logged (never the state itself). A row cut
  while rows for shorter questions fit is not detected: the check under-reports
  and never invents truncation.

Checkpoint choice, measured on Continuum's own seam questions (router, loop,
approval): the English base ``convaiinnovations/laya`` is the default. The
fine-tuned ``laya-typed-decisions`` scores higher on Convai's benchmark but
separated risky from safe tool calls less well here (0.44 / 0.50 against
0.32 / 0.54).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from continuum.logging import get_logger
from continuum.system_one.backends.typed_wire import parse_answers, to_wire
from continuum.system_one.exceptions import (
    SystemOneBackendError,
    SystemOneNotConfiguredError,
)
from continuum.system_one.types import Question, SystemOneCapabilities, SystemOneRawResult

logger = get_logger(__name__)

DEFAULT_MODEL = "convaiinnovations/laya"
DEFAULT_MLX_MODEL = "aac6fef/laya-mlx"


class LayaClassifier:
    """Upstream ``laya`` as a System One backend."""

    name = "laya"
    capabilities = SystemOneCapabilities(
        question_types=frozenset({"binary", "choice", "score"}),
        max_questions=None,  # Laya batches question rows itself
        structured_state=True,
        egress="local",
        calibrated=True,  # Convai's claim (RLCD + temperature calibration), not verified here
    )
    _module = "laya"
    _extra = "laya"
    _install_note = ""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        agent: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self._device = device
        self._agent = agent
        self._load_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _load(self) -> Any:
        try:
            module = __import__(self._module)
        except ImportError as e:
            raise SystemOneNotConfiguredError(
                f"The {self.name} System One backend needs the '{self._module}' package: "
                f"pip install 'shyftlabs-continuum[{self._extra}]'.{self._install_note}",
                config_key="system_one_backend",
            ) from e
        logger.info("Loading %s checkpoint %s", self.name, self.model)
        return module.load(self.model, device=self._device)

    def _loaded(self) -> Any:
        with self._load_lock:
            if self._agent is None:
                self._agent = self._load()
            return self._agent

    async def classify(self, state: Any, questions: dict[str, Question]) -> SystemOneRawResult:
        wire = {qid: to_wire(qid, q, vendor="Laya") for qid, q in questions.items()}
        body = await asyncio.to_thread(self._predict, state, wire)
        raw = parse_answers(body, questions, self.model, vendor="Laya")
        # Laya calls every checkpoint "laya-rl-agent".
        raw.model = self.model
        self._flag_truncation(raw, len(wire))
        return raw

    def _predict(self, state: Any, wire: dict[str, Any]) -> Any:
        agent = self._loaded()
        try:
            return agent.predict(state, wire)
        except SystemOneNotConfiguredError:
            raise
        except Exception as e:
            # ValueError included: Laya raises it for questions it cannot fit
            # (e.g. options over head_max_len). Not re-worded with the message:
            # it can quote the question, and the question can quote the state.
            raise SystemOneBackendError(
                f"The {self.name} model failed ({type(e).__name__}).", backend=self.name
            ) from e

    def _flag_truncation(self, raw: SystemOneRawResult, rows: int) -> None:
        max_len = getattr(self._agent, "cfg", {}).get("max_len")
        tokens = raw.usage.get("input_tokens")
        if not isinstance(max_len, int) or not isinstance(tokens, int) or rows == 0:
            return
        if tokens >= max_len * rows:
            raw.usage["truncated"] = True
            logger.warning(
                "%s truncated the state to its %s-token context for every question; the "
                "decision was made on part of the input (checkpoint %s).",
                self.name,
                max_len,
                self.model,
            )


class LayaMLXClassifier(LayaClassifier):
    """The MLX port of Laya, for Apple Silicon. Same API and answers."""

    name = "laya-mlx"
    _module = "laya_mlx"
    _extra = "laya-mlx"
    _install_note = " It installs only on Apple Silicon Macs (MLX); elsewhere use laya:."

    def __init__(
        self,
        model: str = DEFAULT_MLX_MODEL,
        *,
        agent: Any | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model, agent=agent, device=device)
