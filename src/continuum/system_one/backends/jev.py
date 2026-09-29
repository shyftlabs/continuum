"""Jev (TypeSafe System One) backend: ``jev:<model>``.

Calls ``POST {TYPESAFE_BASE_URL}/v1/systemone`` directly over httpx -- the same
native-SDK stance that removed LiteLLM; LangChain's ``langchain-typesafe`` is a
reference for the wire format, not a dependency.

The question/answer translation is the typed-decision wire format in
``typed_wire``, shared with the Laya backend, which speaks the same format.

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
from continuum.system_one.backends.typed_wire import parse_answers, to_wire
from continuum.system_one.exceptions import (
    SystemOneBackendError,
    SystemOneNotConfiguredError,
    SystemOneResponseError,
    SystemOneTimeoutError,
)
from continuum.system_one.types import (
    Question,
    SystemOneCapabilities,
    SystemOneRawResult,
)

logger = get_logger(__name__)

# 524: an upstream timeout reported by OpenRouter's edge.
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504, 524, 529})


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

    # Where and how this backend is reached; the OpenRouter subclass overrides these.
    _default_model = "jev-latest"
    _key_setting = "typesafe_api_key"
    _key_env = "TYPESAFE_API_KEY"
    _base_url_setting = "typesafe_base_url"
    _path = "/v1/systemone"
    _vendor = "Jev"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Each backend reads only its own key: a TypeSafe key sent to OpenRouter,
        # or the reverse, would hand one vendor's credential to the other.
        key = api_key or getattr(settings, self._key_setting, None)
        if not key or not key.strip():
            raise SystemOneNotConfiguredError(
                f"The {self.name} backend needs an API key: set {self._key_env} or pass api_key.",
                config_key=self._key_setting,
            )
        self.model = model or self._default_model
        self._api_key = key.strip()
        base = base_url or getattr(settings, self._base_url_setting)
        self._endpoint = f"{base.rstrip('/')}{self._path}"
        self._timeout = float(timeout or settings.system_one_timeout_seconds)
        # Injected clients are used as-is (tests, custom transports); the
        # default one is long-lived so connections are pooled.
        self._client = client or httpx.AsyncClient(timeout=self._timeout)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, endpoint={self._endpoint!r})"

    async def classify(self, state: Any, questions: dict[str, Question]) -> SystemOneRawResult:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {qid: self._wire(qid, q) for qid, q in questions.items()},
        }
        body = await self._post(payload)
        return parse_answers(body, questions, self.model, vendor=self._vendor)

    def _wire(self, qid: str, question: Question) -> dict[str, Any]:
        return to_wire(qid, question, vendor=self._vendor)

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
                f"{self._vendor} returned HTTP {response.status_code}"
                + _request_id(response)
                + _status_hint(response.status_code)
                + ".",  # the body is deliberately never included
                backend=self.name,
                status=response.status_code,
                retry_after=retry_after,
            )
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _can_wait(delay: float, deadline: float) -> bool:
        return delay + 0.1 < deadline - time.monotonic()


class JevOpenRouterClassifier(JevClassifier):
    """Jev through OpenRouter's Decisions API: ``openrouter:typesafe/jev-1.13``.

    OpenRouter serves Jev on an alpha route, ``POST /api/alpha/decisions``, in the
    same typed-decision format as TypeSafe's ``/v1/systemone``. So everything but
    the endpoint and key is shared with :class:`JevClassifier`, and an
    OpenRouter key (``OPENROUTER_API_KEY``) replaces a TypeSafe one.

    Two differences are handled here. The documented schema requires a yes/no
    question's criteria to carry both ``true`` and ``false``; a question without
    them gets neutral ones ("Yes." / "No.") that add no meaning. And OpenRouter
    answers with a dated snapshot (``typesafe/jev-1.13-20260917``), which is
    reported as the model, along with ``usage.cost``.

    The route is alpha: if OpenRouter moves it, ``OPENROUTER_BASE_URL`` and this
    class's ``_path`` are what change.
    """

    name = "openrouter"
    _default_model = "typesafe/jev-1.13"
    _key_setting = "openrouter_api_key"
    _key_env = "OPENROUTER_API_KEY"
    _base_url_setting = "openrouter_base_url"
    _path = "/alpha/decisions"
    _vendor = "OpenRouter"

    def _wire(self, qid: str, question: Question) -> dict[str, Any]:
        wire = super()._wire(qid, question)
        if wire["type"] == "noul":
            given = wire.get("criteria") or {}
            wire["criteria"] = {
                "true": "Yes." if given.get("true") is None else given["true"],
                "false": "No." if given.get("false") is None else given["false"],
            }
        return wire


def _status_hint(status: int) -> str:
    """A fixed hint for statuses whose fix is on the caller's side (never the body)."""
    return {
        401: " (the API key was not accepted)",
        402: " (insufficient credits on the account)",
    }.get(status, "")


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
