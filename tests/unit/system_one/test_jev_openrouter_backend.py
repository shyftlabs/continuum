"""Jev through OpenRouter's Decisions API: ``openrouter:typesafe/jev-1.13``.

OpenRouter serves Jev on an alpha route, ``POST /api/alpha/decisions``, which takes
the same typed-decision format as TypeSafe's own ``/v1/systemone`` -- so the same
translation is reused, and an OpenRouter key replaces a TypeSafe one. Responses
match the documented example: answers in Jev's format, ``model`` as a dated
snapshot ("typesafe/jev-1.13-20260917"), ``provider``, and ``usage.cost``.

What differs from TypeSafe direct, and is checked here: the URL and key, the
default model id, 402 (insufficient credits) and 524 (upstream timeout), and that
a yes/no question always carries criteria -- the documented schema requires them.
"""

from __future__ import annotations

import json

import httpx
import pytest

from continuum.system_one.testing import SystemOneContract

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def _documented_reply(payload: dict) -> dict:
    """A response shaped like OpenRouter's documented example."""
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"noul": 0.96, "type": "noul"}
        elif q["type"] == "choice":
            labels = list(q["criteria"])
            rest = 0.25 / (len(labels) - 1)
            answers[qid] = {
                "choice": labels[0],
                "confidence": 0.75,
                "probabilities": {
                    label: (0.75 if i == 0 else rest) for i, label in enumerate(labels)
                },
                "type": "choice",
            }
        else:
            n = len(q["criteria"])
            answers[qid] = {
                "confidence": 0.9,
                "probabilities": {
                    str(i): (0.99 if i == n - 1 else 0.01 / (n - 1)) for i in range(n)
                },
                "score": n - 1.01,
                "type": "score",
            }
    return {
        "answers": answers,
        "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "usage": {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
    }


class Recorder:
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
        return httpx.Response(200, json=_documented_reply(json.loads(request.content)))

    @property
    def payloads(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def _or(handler, **kwargs):
    from continuum.system_one.backends.jev import JevOpenRouterClassifier

    kwargs.setdefault("api_key", "sk-or-v1-test")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevOpenRouterClassifier(client=client, **kwargs)


def _q():
    from continuum.system_one import BinaryQuestion, ChoiceQuestion, ScoreQuestion

    return {
        "is_bug": BinaryQuestion(
            instructions="Is the customer reporting a software defect?",
            true_criteria="The customer describes broken or unexpected product behavior.",
            false_criteria="The customer is asking a question or requesting a feature.",
        ),
        "team": ChoiceQuestion(
            instructions="Which team?",
            labels={"payments": "Checkout, billing.", "frontend": "Rendering, layout."},
        ),
        "severity": ScoreQuestion(instructions="How severe?", levels=["low", "mid", "high"]),
    }


class TestTheRequest:
    async def test_it_posts_to_the_decisions_route_with_the_openrouter_key(self):
        rec = Recorder()
        await _or(rec).classify("state", _q())

        req = rec.requests[0]
        assert str(req.url) == ENDPOINT
        assert req.headers["authorization"] == "Bearer sk-or-v1-test"

    async def test_the_default_model_is_openrouters_jev_id(self):
        rec = Recorder()
        await _or(rec).classify("s", _q())
        assert rec.payloads[0]["model"] == "typesafe/jev-1.13"

    async def test_questions_use_the_same_typed_format_as_typesafe(self):
        rec = Recorder()
        await _or(rec).classify({"ticket": "blank screen"}, _q())

        payload = rec.payloads[0]
        assert payload["state"] == {"ticket": "blank screen"}
        assert payload["questions"]["is_bug"]["criteria"] == {
            "true": "The customer describes broken or unexpected product behavior.",
            "false": "The customer is asking a question or requesting a feature.",
        }
        assert payload["questions"]["severity"]["criteria"] == ["low", "mid", "high"]

    async def test_a_yes_no_question_without_criteria_gets_neutral_ones(self):
        """The documented schema requires noul criteria with true and false. A
        question with none gets neutral ones that add no meaning -- the same
        judgement as asking without criteria on TypeSafe direct."""
        from continuum.system_one import BinaryQuestion

        rec = Recorder()
        await _or(rec).classify("s", {"q": BinaryQuestion(instructions="Is it spam?")})
        assert rec.payloads[0]["questions"]["q"]["criteria"] == {"true": "Yes.", "false": "No."}

    async def test_one_given_criterion_is_kept_and_the_other_filled(self):
        from continuum.system_one import BinaryQuestion

        rec = Recorder()
        await _or(rec).classify(
            "s", {"q": BinaryQuestion(instructions="Is it spam?", true_criteria="Unsolicited ad.")}
        )
        assert rec.payloads[0]["questions"]["q"]["criteria"] == {
            "true": "Unsolicited ad.",
            "false": "No.",
        }


class TestTheResponse:
    async def test_the_documented_response_parses(self):
        raw = await _or(Recorder()).classify("s", _q())
        assert raw.distributions["is_bug"] == {
            "true": pytest.approx(0.96),
            "false": pytest.approx(0.04),
        }
        assert raw.distributions["team"]["payments"] == pytest.approx(0.75)
        assert raw.distributions["severity"][2] == pytest.approx(0.99)

    async def test_the_dated_snapshot_is_reported_as_the_model(self):
        """More precise than the requested id: it says which snapshot decided."""
        raw = await _or(Recorder()).classify("s", _q())
        assert raw.model == "typesafe/jev-1.13-20260917"

    async def test_cost_is_kept_in_usage(self):
        raw = await _or(Recorder()).classify("s", _q())
        assert raw.usage["cost"] == pytest.approx(0.000019992)
        assert raw.usage["input_tokens"] == 476


class TestFailures:
    def _one(self):
        from continuum.system_one import BinaryQuestion

        return {"q": BinaryQuestion(instructions="x?", true_criteria="a", false_criteria="b")}

    async def test_insufficient_credits_is_a_clear_backend_error_and_not_retried(self):
        from continuum.system_one import SystemOneBackendError

        body = {
            "error": {
                "code": 402,
                "message": "Insufficient credits. Add more using https://openrouter.ai/credits",
            }
        }
        rec = Recorder([httpx.Response(402, json=body)])
        with pytest.raises(SystemOneBackendError) as exc:
            await _or(rec).classify("s", self._one())
        assert exc.value.status == 402
        assert "credits" in str(exc.value).lower()
        assert len(rec.requests) == 1

    async def test_an_upstream_timeout_524_is_retried_once(self):
        rec = Recorder([httpx.Response(524, json={"error": {"code": 524}})])
        raw = await _or(rec).classify("s", self._one())
        assert len(rec.requests) == 2
        assert "q" in raw.distributions

    async def test_an_unknown_key_is_an_auth_error(self):
        from continuum.system_one import SystemOneBackendError

        rec = Recorder(
            [httpx.Response(401, json={"error": {"code": 401, "message": "User not found."}})]
        )
        with pytest.raises(SystemOneBackendError) as exc:
            await _or(rec).classify("s", self._one())
        assert exc.value.status == 401

    async def test_an_error_never_carries_the_body_or_the_state(self):
        from continuum.system_one import SystemOneBackendError

        rec = Recorder([httpx.Response(400, json={"echo": "SSN 123-45-6789"})])
        with pytest.raises(SystemOneBackendError) as exc:
            await _or(rec).classify("SSN 123-45-6789", self._one())
        assert "123-45-6789" not in f"{exc.value} {exc.value!r} {exc.value.context}"


class TestConfiguration:
    def test_a_missing_key_names_openrouter_api_key(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import SystemOneNotConfiguredError
        from continuum.system_one.backends.jev import JevOpenRouterClassifier

        monkeypatch.setattr(settings, "openrouter_api_key", None)
        with pytest.raises(SystemOneNotConfiguredError) as exc:
            JevOpenRouterClassifier()
        assert "OPENROUTER_API_KEY" in str(exc.value)

    def test_the_key_is_read_from_settings(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one.backends.jev import JevOpenRouterClassifier

        monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-v1-from-settings")
        assert JevOpenRouterClassifier()._api_key == "sk-or-v1-from-settings"

    def test_the_typesafe_key_is_not_used(self, monkeypatch):
        """Two vendors, two keys: a TypeSafe key sent to OpenRouter would leak it."""
        from continuum.config import settings
        from continuum.system_one import SystemOneNotConfiguredError
        from continuum.system_one.backends.jev import JevOpenRouterClassifier

        monkeypatch.setattr(settings, "openrouter_api_key", None)
        monkeypatch.setattr(settings, "typesafe_api_key", "ts-secret")
        with pytest.raises(SystemOneNotConfiguredError):
            JevOpenRouterClassifier()

    def test_the_key_never_appears_in_repr(self):
        assert "sk-or-v1-test" not in repr(_or(Recorder()))

    def test_it_declares_itself_remote(self):
        backend = _or(Recorder())
        assert backend.capabilities.egress == "remote"
        assert backend.name == "openrouter"

    def test_the_registry_builds_it_from_a_spec(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import create_classifier

        monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-v1-x")
        backend = create_classifier("openrouter:typesafe/jev-1.13")
        assert (backend.name, backend.model) == ("openrouter", "typesafe/jev-1.13")

    def test_the_egress_resource_names_openrouter(self):
        """Policy can allow Jev direct and deny it through OpenRouter, or the reverse."""
        from continuum.system_one import egress_resource

        assert egress_resource(_or(Recorder())) == "system_one:remote:openrouter:typesafe/jev-1.13"


class TestJevOpenRouterMeetsTheContract(SystemOneContract):
    latency_budget_ms = 200.0  # mock transport: the adapter's own overhead

    def make_classifier(self):
        return _or(Recorder())

    def make_failing_classifier(self):
        return _or(Recorder([httpx.Response(500), httpx.Response(500)]))
