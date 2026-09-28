"""RouterAgent with routing_strategy="system_one_classifier".

Today's LLM route asks for an agent name in free text and matches it by
substring in both directions, so an empty reply matches the first route. A
System One classifier answers a Choice over the route names plus "none", so
there is nothing to parse. The seam stays off unless a router opts in; when it
does, a missing backend is a construction error, the kill switch sends it back
to the LLM route, and a classifier that fails means "no route" -- the fallback
agent -- never a guess.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.system_one.conftest import FakeClassifier


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from continuum.config import settings
    from continuum.core.container import reset_container
    from continuum.system_one.registry import clear_classifier_cache

    monkeypatch.setattr(settings, "system_one_backend", None)
    monkeypatch.setattr(settings, "system_one_disabled", False)
    reset_container()
    clear_classifier_cache()
    yield
    clear_classifier_cache()
    reset_container()


def _routes():
    from continuum.agent.types import Route

    return [
        Route(agent_name="billing-agent", description="Billing, payments, invoices, refunds"),
        Route(agent_name="technical-agent", description="Technical issues, bugs, outages"),
    ]


def _router(**config):
    from continuum.agent.config import RouterConfig
    from continuum.agent.workflow.router import RouterAgent

    return RouterAgent(
        name="triage",
        routes=_routes(),
        fallback_agent_name="general-agent",
        router_config=RouterConfig(routing_strategy="system_one_classifier", **config),
    )


def _backend(dist, **kwargs):
    return FakeClassifier(answer=lambda qid, q: dict(dist), **kwargs)


def _use(backend):
    from continuum.core.container import get_container

    get_container().set_system_one_classifier(backend)
    return backend


def _llm(reply="technical-agent"):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content=reply))
    return llm


class TestOffByDefault:
    def test_the_default_strategy_is_still_llm(self):
        from continuum.agent.config import RouterConfig

        assert RouterConfig().routing_strategy == "llm"


class TestConstruction:
    def test_opting_in_with_no_backend_is_an_error(self):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError) as exc:
            _router()
        assert "triage" in str(exc.value)

    def test_a_container_backend_satisfies_it(self):
        _use(_backend({}))
        _router()

    def test_the_routers_own_backend_spec_satisfies_it(self):
        _router(system_one_backend="jev:jev-latest")

    def test_the_spec_is_serialised(self):
        router = _router(system_one_backend="jev:jev-latest")
        assert router.router_config.to_dict()["system_one_backend"] == "jev:jev-latest"


class TestRouting:
    async def test_the_most_probable_route_is_chosen(self):
        _use(_backend({"billing-agent": 0.1, "technical-agent": 0.8, "none": 0.1}))
        llm = _llm()
        assert await _router().route("Production is down", llm_client=llm) == "technical-agent"
        llm.chat.assert_not_called()

    async def test_the_question_offers_every_route_plus_none(self):
        backend = _use(_backend({"billing-agent": 0.8, "technical-agent": 0.1, "none": 0.1}))
        await _router().route("My card was charged twice")

        state, questions = backend.calls[0]
        (question,) = questions.values()
        assert question.kind == "choice"
        assert set(question.labels) == {"billing-agent", "technical-agent", "none"}
        assert "Billing, payments" in question.labels["billing-agent"]
        assert state == "My card was charged twice"

    async def test_none_means_no_route(self):
        backend = _use(_backend({"billing-agent": 0.1, "technical-agent": 0.1, "none": 0.8}))
        assert await _router().route("What's the weather?") is None
        assert backend.calls, "None must come from the classifier's answer, not from no branch"

    async def test_the_routers_own_backend_is_used_over_the_default(self):
        from continuum.system_one import register_backend

        default = _use(_backend({"billing-agent": 0.9, "technical-agent": 0.05, "none": 0.05}))
        own = _backend({"billing-agent": 0.05, "technical-agent": 0.9, "none": 0.05})
        register_backend("routerfake", lambda model: own)

        result = await _router(system_one_backend="routerfake:m").route("x")
        assert result == "technical-agent"
        assert default.calls == []

    async def test_a_custom_router_still_runs_first(self):
        from continuum.agent.config import RouterConfig
        from continuum.agent.workflow.router import RouterAgent

        backend = _use(_backend({"billing-agent": 0.9, "technical-agent": 0.05, "none": 0.05}))
        router = RouterAgent(
            name="triage",
            routes=_routes(),
            router_config=RouterConfig(routing_strategy="system_one_classifier"),
            custom_router=lambda text, routes: "technical-agent",
        )
        assert await router.route("x") == "technical-agent"
        assert backend.calls == []


class TestWhenTheClassifierCannotAnswer:
    async def test_a_failure_means_no_route_not_a_guess(self):
        """The router's defined 'no route' path: fallback_agent_name, or
        NoRouteFoundError. Silently switching to the LLM would be a second,
        unrequested decision-maker."""

        class Down(FakeClassifier):
            asked = False

            async def classify(self, state, questions):
                from continuum.system_one import SystemOneBackendError

                Down.asked = True
                raise SystemOneBackendError("down", backend="fake")

        _use(Down())
        llm = _llm()
        assert await _router().route("x", llm_client=llm) is None
        assert Down.asked, "the classifier must have been asked"
        llm.chat.assert_not_called()

    async def test_a_policy_denial_also_means_no_route(self):
        from continuum.security.policy import AccessPolicy, PolicyStore
        from continuum.security.policy_context import use_active_policy

        backend = _use(_backend({"billing-agent": 0.9, "technical-agent": 0.05, "none": 0.05}))
        backend.capabilities = type(backend.capabilities)(
            question_types=backend.capabilities.question_types, egress="remote"
        )
        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="phi-local", subjects=["phi"], resources=["system_one:remote:*"], effect="deny"
            )
        )

        class Ctx:
            data_labels = {"phi"}

        router = _router()
        with use_active_policy(store, "triage", Ctx()):
            assert await router.route("x") is None
        assert backend.calls == []
        # The same router, outside the tainted run, does reach the classifier:
        # the None above was the denial, not a seam that never asks.
        assert await router.route("x") == "billing-agent"


class TestTheKillSwitch:
    async def test_disabled_falls_back_to_the_llm_route(self, monkeypatch):
        """'Previous behaviour' for a router is the LLM route it had before
        opting in."""
        from continuum.config import settings

        backend = _use(_backend({"billing-agent": 0.9, "technical-agent": 0.05, "none": 0.05}))
        monkeypatch.setattr(settings, "system_one_disabled", True)
        llm = _llm("technical-agent")

        assert await _router().route("x", llm_client=llm) == "technical-agent"
        llm.chat.assert_called_once()
        assert backend.calls == []


class TestProvenance:
    def _span(self):
        span = MagicMock()
        span.__aenter__ = AsyncMock(return_value=span)
        span.__aexit__ = AsyncMock(return_value=False)
        return span

    async def test_the_trace_says_the_classifier_decided(self):
        _use(_backend({"billing-agent": 0.1, "technical-agent": 0.8, "none": 0.1}))
        span = self._span()
        with patch("continuum.agent.workflow.router.SpanScope", return_value=span):
            await _router().route("x")

        output = span.set_output.call_args.args[0]
        assert output["decided_by"] == "system_one"
        assert output["selected_route"] == "technical-agent"
        assert output["backend"] == "fake"
        assert output["probabilities"]["technical-agent"] == pytest.approx(0.8)

    async def test_the_trace_says_when_the_kill_switch_sent_it_to_the_llm(self, monkeypatch):
        from continuum.config import settings

        _use(_backend({}))
        monkeypatch.setattr(settings, "system_one_disabled", True)
        span = self._span()
        with patch("continuum.agent.workflow.router.SpanScope", return_value=span):
            await _router().route("x", llm_client=_llm())

        assert span.set_output.call_args.args[0]["decided_by"] == "legacy"


class TestTheFactory:
    def test_create_router_agent_accepts_the_strategy(self):
        from continuum.agent.workflow.router import create_router_agent

        _use(_backend({}))
        router = create_router_agent(
            "triage", [("billing", "Billing")], strategy="system_one_classifier"
        )
        assert router.router_config.routing_strategy == "system_one_classifier"
