"""The Jev (TypeSafe System One) adapter, against a mock transport.

What is checked is the translation both ways -- Continuum's neutral questions to
Jev's wire format, Jev's answers back to raw distributions -- and that every
failure surfaces as a SystemOneError without the response body or the state,
either of which can carry the data being classified.

No live call runs here; that needs TYPESAFE_API_KEY (see tests/integration).
"""

from __future__ import annotations

import json

import httpx
import pytest

from continuum.system_one.testing import SystemOneContract

ENDPOINT = "https://api.typesafe.test/v1/systemone"


def _answers_for(payload: dict) -> dict:
    """A plausible Jev response for whatever was asked."""
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.82}
        elif q["type"] == "choice":
            labels = list(q["criteria"])
            rest = 0.3 / (len(labels) - 1)
            probs = {label: (0.7 if i == 0 else rest) for i, label in enumerate(labels)}
            answers[qid] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": probs,
                "confidence": 0.61,
            }
        else:
            n = len(q["criteria"])
            answers[qid] = {
                "type": "score",
                "score": (n - 1) / 2,
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {str(i): 1.0 / n for i in range(n)},
                "confidence": 0.0,
            }
    return {
        "model": payload["model"],
        "answers": answers,
        "usage": {"input_tokens": 40, "output_tokens": 3},
    }


class Recorder:
    """A mock transport handler that records requests and plays scripted replies."""

    def __init__(self, replies=None):
        self.requests: list[httpx.Request] = []
        self._replies = list(replies or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._replies:
            reply = self._replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return httpx.Response(200, json=_answers_for(json.loads(request.content)))

    @property
    def payloads(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def _jev(handler, **kwargs):
    from continuum.system_one.backends.jev import JevClassifier

    kwargs.setdefault("api_key", "ts-test-key")
    kwargs.setdefault("base_url", "https://api.typesafe.test")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevClassifier(client=client, **kwargs)


def _q():
    from continuum.system_one import BinaryQuestion, ChoiceQuestion, ScoreQuestion

    return {
        "urgent": BinaryQuestion(
            instructions="Does this need attention now?",
            true_criteria="Customers are affected.",
            false_criteria="Nobody is blocked.",
        ),
        "team": ChoiceQuestion(
            instructions="Which team?", labels={"billing": "Payments.", "technical": "Bugs."}
        ),
        "severity": ScoreQuestion(instructions="How severe?", levels=["low", "high"]),
    }


class TestTheRequest:
    async def test_it_posts_to_the_systemone_endpoint_with_a_bearer_key(self):
        rec = Recorder()
        await _jev(rec).classify("state", _q())

        req = rec.requests[0]
        assert str(req.url) == ENDPOINT
        assert req.method == "POST"
        assert req.headers["authorization"] == "Bearer ts-test-key"
        assert req.headers["user-agent"].startswith("continuum/")

    async def test_questions_are_translated_to_jevs_vocabulary(self):
        rec = Recorder()
        await _jev(rec, model="jev-2026-09").classify({"msg": "hi"}, _q())

        payload = rec.payloads[0]
        assert payload["model"] == "jev-2026-09"
        assert payload["state"] == {"msg": "hi"}
        assert payload["questions"]["urgent"] == {
            "type": "noul",
            "instructions": "Does this need attention now?",
            "criteria": {"true": "Customers are affected.", "false": "Nobody is blocked."},
        }
        assert payload["questions"]["team"] == {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments.", "technical": "Bugs."},
        }
        assert payload["questions"]["severity"] == {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["low", "high"],
        }

    async def test_criteria_are_omitted_when_none_were_given(self):
        """LangChain's AutoModeMiddleware sent none even when it meant to. Here
        a binary question without criteria simply has no criteria key."""
        from continuum.system_one import BinaryQuestion

        rec = Recorder()
        await _jev(rec).classify("s", {"q": BinaryQuestion(instructions="Is it spam?")})
        assert "criteria" not in rec.payloads[0]["questions"]["q"]


class TestTheResponse:
    async def test_answers_become_raw_distributions(self):
        raw = await _jev(Recorder()).classify("s", _q())

        assert raw.distributions["urgent"] == {
            "true": pytest.approx(0.82),
            "false": pytest.approx(0.18),
        }
        assert raw.distributions["team"] == {
            "billing": pytest.approx(0.7),
            "technical": pytest.approx(0.3),
        }
        assert raw.distributions["severity"] == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}

    async def test_jevs_own_confidence_is_passed_on_as_raw(self):
        raw = await _jev(Recorder()).classify("s", _q())
        assert raw.raw_confidence["team"] == pytest.approx(0.61)

    async def test_model_and_usage_are_reported(self):
        raw = await _jev(Recorder(), model="jev-latest").classify("s", _q())
        assert raw.model == "jev-latest"
        assert raw.usage == {"input_tokens": 40, "output_tokens": 3}

    async def test_a_missing_answer_is_a_response_error(self):
        from continuum.system_one import SystemOneResponseError

        reply = httpx.Response(200, json={"model": "jev", "answers": {}, "usage": {}})
        with pytest.raises(SystemOneResponseError):
            await _jev(Recorder([reply])).classify("s", _q())

    async def test_an_answer_of_the_wrong_type_is_a_response_error(self):
        from continuum.system_one import BinaryQuestion, SystemOneResponseError

        reply = httpx.Response(
            200,
            json={"model": "jev", "answers": {"q": {"type": "score", "probabilities": {}}}},
        )
        with pytest.raises(SystemOneResponseError):
            await _jev(Recorder([reply])).classify("s", {"q": BinaryQuestion(instructions="x?")})

    async def test_a_body_that_is_not_json_is_a_response_error(self):
        from continuum.system_one import BinaryQuestion, SystemOneResponseError

        reply = httpx.Response(200, content=b"<html>gateway</html>")
        with pytest.raises(SystemOneResponseError):
            await _jev(Recorder([reply])).classify("s", {"q": BinaryQuestion(instructions="x?")})


class TestFailures:
    def _binary(self):
        from continuum.system_one import BinaryQuestion

        return {"q": BinaryQuestion(instructions="Is it urgent?")}

    async def test_an_auth_failure_is_a_backend_error_and_is_not_retried(self):
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.Response(401, json={"error": "bad key"})])
        with pytest.raises(SystemOneBackendError) as exc:
            await _jev(rec).classify("s", self._binary())
        assert exc.value.status == 401
        assert len(rec.requests) == 1

    async def test_a_rate_limit_is_retried_once_after_retry_after(self):
        rec = Recorder([httpx.Response(429, headers={"retry-after-ms": "1"})])
        raw = await _jev(rec).classify("s", self._binary())
        assert len(rec.requests) == 2
        assert "q" in raw.distributions

    async def test_a_second_server_error_gives_up(self):
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.Response(503), httpx.Response(503)])
        with pytest.raises(SystemOneBackendError) as exc:
            await _jev(rec).classify("s", self._binary())
        assert exc.value.status == 503
        assert len(rec.requests) == 2

    async def test_a_retry_after_beyond_the_timeout_is_not_waited_for(self):
        """Sleeping 60s inside a 10s budget would turn a rate limit into a hang."""
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.Response(429, headers={"retry-after": "60"})])
        with pytest.raises(SystemOneBackendError) as exc:
            await _jev(rec, timeout=2.0).classify("s", self._binary())
        assert exc.value.retry_after == pytest.approx(60.0)
        assert len(rec.requests) == 1

    async def test_a_timeout_is_a_timeout_error(self):
        from continuum.system_one import SystemOneTimeoutError

        rec = Recorder([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
        with pytest.raises(SystemOneTimeoutError):
            await _jev(rec).classify("s", self._binary())

    async def test_a_connection_failure_is_a_backend_error(self):
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        with pytest.raises(SystemOneBackendError):
            await _jev(rec).classify("s", self._binary())

    async def test_an_error_never_carries_the_body_or_the_state(self):
        """A 4xx body can echo the request -- the very state being classified."""
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.Response(400, json={"echo": "patient SSN 123-45-6789"})])
        with pytest.raises(SystemOneBackendError) as exc:
            await _jev(rec).classify("patient SSN 123-45-6789", self._binary())
        rendered = f"{exc.value} {exc.value!r} {exc.value.context}"
        assert "123-45-6789" not in rendered


class TestConfiguration:
    def test_a_missing_api_key_fails_at_construction(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import SystemOneNotConfiguredError
        from continuum.system_one.backends.jev import JevClassifier

        monkeypatch.setattr(settings, "typesafe_api_key", None)
        with pytest.raises(SystemOneNotConfiguredError):
            JevClassifier()

    def test_the_key_is_read_from_settings(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one.backends.jev import JevClassifier

        monkeypatch.setattr(settings, "typesafe_api_key", "from-settings")
        assert JevClassifier()._api_key == "from-settings"

    def test_the_key_never_appears_in_repr(self):
        backend = _jev(Recorder(), api_key="ts-super-secret")
        assert "ts-super-secret" not in repr(backend)

    def test_it_declares_itself_remote(self):
        """State leaves the host: the egress policy check depends on this."""
        backend = _jev(Recorder())
        assert backend.capabilities.egress == "remote"
        assert backend.name == "jev"

    def test_the_registry_builds_it_from_a_spec(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import create_classifier

        monkeypatch.setattr(settings, "typesafe_api_key", "k")
        backend = create_classifier("jev:jev-latest")
        assert (backend.name, backend.model) == ("jev", "jev-latest")


class TestJevMeetsTheContract(SystemOneContract):
    latency_budget_ms = 200.0  # mock transport: this checks the adapter's own overhead

    def make_classifier(self):
        return _jev(Recorder())

    def make_failing_classifier(self):
        return _jev(Recorder([httpx.Response(500), httpx.Response(500)]))
