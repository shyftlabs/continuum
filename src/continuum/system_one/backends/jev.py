"""Jev (TypeSafe System One) backend: ``jev:<model>``.

Calls ``POST {TYPESAFE_BASE_URL}/v1/systemone`` directly over httpx -- the same
native-SDK stance that removed LiteLLM; LangChain's ``langchain-typesafe`` is a
reference for the wire format, not a dependency.

Translation, both ways:

    BinaryQuestion  <->  noul    {"true": p, "false": 1-p}   <-  {"noul": p}
    ChoiceQuestion  <->  choice  labels -> probabilities      <-  {"probabilities", "confidence"}
    ScoreQuestion   <->  score   levels -> probabilities      <-  {"probabilities": {"0": p}}

Failures: one retry on 408/429/5xx and connection errors, honouring
``Retry-After``/``retry-after-ms`` but never sleeping past the call's timeout; a
timeout is not retried (the budget is spent). Every failure is a
``SystemOneError``, and none carries the response body: a 4xx body can echo the
request, which is the very state being classified.
"""

from __future__ import annotations

import asyncio
import math
import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from continuum.config import settings
from continuum.logging import get_logger
from continuum.system_one.exceptions import (
    SystemOneBackendError,
    SystemOneCapabilityError,
    SystemOneNotConfiguredError,
    SystemOneResponseError,
    SystemOneTimeoutError,
)
from continuum.system_one.types import (
    BinaryQuestion,
    ChoiceQuestion,
    Question,
    ScoreQuestion,
    SystemOneCapabilities,
    SystemOneRawResult,
)

logger = get_logger(__name__)

_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504, 529})
_WIRE_TYPE = {"binary": "noul", "choice": "choice", "score": "score"}


def _user_agent() -> str:
    from continuum import __version__

    return f"continuum/{__version__}"


class JevClassifier:
    """TypeSafe's Jev as a System One backend. Remote: state leaves the host."""

    name = "jev"
    capabilities = SystemOneCapabilities(
        question_types=frozenset({"binary", "choice", "score"}),
        max_questions=None,
        structured_state=True,
        egress="remote",
        calibrated=True,  # TypeSafe's claim (RL-calibrated), not verified by the SDK
    )
    # Delay before the single retry when the server gives no Retry-After.
    retry_backoff_s = 0.25

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or settings.typesafe_api_key
        if not key or not key.strip():
            raise SystemOneNotConfiguredError(
                "The Jev backend needs a TypeSafe API key: set TYPESAFE_API_KEY or pass api_key.",
                config_key="typesafe_api_key",
            )
        self.model = model
        self._api_key = key.strip()
        self._endpoint = f"{(base_url or settings.typesafe_base_url).rstrip('/')}/v1/systemone"
        self._timeout = float(timeout or settings.system_one_timeout_seconds)
        # Injected clients are used as-is (tests, custom transports); the
        # default one is long-lived so connections are pooled.
        self._client = client or httpx.AsyncClient(timeout=self._timeout)

    def __repr__(self) -> str:
        return f"JevClassifier(model={self.model!r}, endpoint={self._endpoint!r})"

    async def classify(self, state: Any, questions: dict[str, Question]) -> SystemOneRawResult:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {qid: _wire_question(qid, q) for qid, q in questions.items()},
        }
        body = await self._post(payload)
        return _parse(body, questions, self.model)

    async def _post(self, payload: dict[str, Any]) -> Any:
        deadline = time.monotonic() + self._timeout
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": _user_agent(),
        }
        for attempt in (1, 2):
            remaining = deadline - time.monotonic()
            try:
                response = await self._client.post(
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    timeout=max(0.1, remaining),
                )
            except httpx.TimeoutException as e:
                raise SystemOneTimeoutError(
                    f"Jev did not answer within {self._timeout}s.", backend=self.name
                ) from e
            except httpx.HTTPError as e:
                if attempt == 1 and self._can_wait(self.retry_backoff_s, deadline):
                    await asyncio.sleep(self.retry_backoff_s)
                    continue
                raise SystemOneBackendError(
                    f"Could not reach Jev ({type(e).__name__}).", backend=self.name
                ) from e

            if response.is_success:
                try:
                    return response.json()
                except ValueError as e:
                    raise SystemOneResponseError("Jev returned a body that is not JSON.") from e

            retry_after = _retry_after(response.headers)
            if response.status_code in _RETRYABLE and attempt == 1:
                delay = self.retry_backoff_s if retry_after is None else retry_after
                if self._can_wait(delay, deadline):
                    await asyncio.sleep(delay)
                    continue
            raise SystemOneBackendError(
                f"Jev returned HTTP {response.status_code}"
                + _request_id(response)
                + ".",  # the body is deliberately never included
                backend=self.name,
                status=response.status_code,
                retry_after=retry_after,
            )
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _can_wait(delay: float, deadline: float) -> bool:
        return delay + 0.1 < deadline - time.monotonic()


def _wire_question(qid: str, question: Question) -> dict[str, Any]:
    if question.kind not in _WIRE_TYPE:
        raise SystemOneCapabilityError(f"Jev cannot answer a {question.kind} question ('{qid}').")
    wire: dict[str, Any] = {
        "type": _WIRE_TYPE[question.kind],
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


def _parse(body: Any, questions: dict[str, Question], model: str) -> SystemOneRawResult:
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise SystemOneResponseError("Jev's response has no answers object.")
    answers = body["answers"]
    distributions: dict[str, dict[Any, float]] = {}
    raw_confidence: dict[str, float] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        expected = _WIRE_TYPE[question.kind]
        if not isinstance(answer, dict) or answer.get("type") != expected:
            raise SystemOneResponseError(f"Jev returned no {expected} answer for '{qid}'.")
        try:
            if isinstance(question, BinaryQuestion):
                p = float(answer["noul"])
                distributions[qid] = {"true": p, "false": 1.0 - p}
            elif isinstance(question, ChoiceQuestion):
                distributions[qid] = {str(k): float(v) for k, v in answer["probabilities"].items()}
            else:
                distributions[qid] = {int(k): float(v) for k, v in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise SystemOneResponseError(f"Jev's answer for '{qid}' is malformed.") from e
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


def _request_id(response: httpx.Response) -> str:
    rid = response.headers.get("x-typesafe-request-id")
    return f" (request_id={rid})" if rid else ""


def _retry_after(headers: httpx.Headers) -> float | None:
    """Seconds the server asked to wait, from retry-after-ms or retry-after."""
    raw_ms = headers.get("retry-after-ms")
    if raw_ms is not None:
        try:
            ms = float(raw_ms)
            if math.isfinite(ms) and ms >= 0:
                return ms / 1000
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None
