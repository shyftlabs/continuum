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

    async def test_labels_are_statements_an_nli_backend_can_test(self):
        """Found in the live run against cross-encoder/nli-deberta-v3-{small,base}.
        An NLI backend sees only premise + hypothesis. "The request is about:
        <list>" and "fits none of the listed agents" (a list it never sees) sent
        a pancake recipe to billing on both models. Plain sentences -- "This
        request is about X." / "... something else." -- routed all three probes
        correctly on the base model."""
        backend = _use(_backend({"billing-agent": 0.8, "technical-agent": 0.1, "none": 0.1}))
        await _router().route("x")

        (question,) = backend.calls[0][1].values()
        assert question.labels["billing-agent"] == (
            "This request is about Billing, payments, invoices, refunds."
        )
        assert question.labels["none"] == "This request is about something else."

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


class TestTheDecisionIsLogged:
    """The loop and the quality gates log a ``decided_by=system_one`` line for
    every decision; the router logged only the kill switch and failures, so an
    application could not show which route the classifier picked, or how sure
    it was, without reaching into the trace span. The line carries the route
    *name* (configuration) and the probability, never the request text."""

    def _logs(self):
        import logging

        records = []

        class H(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = H()
        logging.getLogger("continuum.agent.workflow.router").addHandler(handler)
        return records, handler

    def _detach(self, handler):
        import logging

        logging.getLogger("continuum.agent.workflow.router").removeHandler(handler)

    async def test_a_route_decision_is_logged_with_backend_and_confidence(self):
        import logging

        _use(_backend({"billing-agent": 0.1, "technical-agent": 0.8, "none": 0.1}))
        records, handler = self._logs()
        try:
            await _router().route("Production is down")
        finally:
            self._detach(handler)
        lines = [r.getMessage() for r in records if r.levelno == logging.INFO]
        line = next(m for m in lines if "decided_by=system_one" in m)
        assert "triage" in line
        assert "backend=fake" in line
        assert "route=technical-agent" in line
        assert "p=0.800" in line
        assert "Production is down" not in line

    async def test_none_is_logged_as_no_route(self):
        _use(_backend({"billing-agent": 0.1, "technical-agent": 0.1, "none": 0.8}))
        records, handler = self._logs()
        try:
            await _router().route("What's the weather?")
        finally:
            self._detach(handler)
        line = next(r.getMessage() for r in records if "decided_by=system_one" in r.getMessage())
        assert "route=none" in line


class TestAMinimumConfidence:
    """TypeSafe's confidence-gated / intent routing: "the answer tells you what;
    confidence tells you whether to act" -- below a floor, don't act on the top
    route. Opt-in: RouterConfig.system_one_min_confidence (None = act on the top
    route, as before). It reads the backend's own confidence (raw_confidence),
    because published thresholds belong to the backend's definition: live, Jev
    gave 0.45 where the SDK's entropy-based figure was 0.42, and for a 0.7/0.1/
    0.1/0.1 split the two are 0.60 and 0.32. Below the floor the request gets no
    route -- fallback_agent_name, the same path as "none" or a failure."""

    def _backend(self, dist, raw):
        return _use(_backend(dist, raw_confidence=raw))

    async def test_by_default_low_confidence_still_routes(self):
        self._backend({"billing-agent": 0.4, "technical-agent": 0.35, "none": 0.25}, raw=0.1)
        assert await _router().route("x") == "billing-agent"

    async def test_below_the_floor_there_is_no_route(self):
        self._backend({"billing-agent": 0.59, "technical-agent": 0.3, "none": 0.11}, raw=0.45)
        llm = _llm()
        assert await _router(system_one_min_confidence=0.5).route("x", llm_client=llm) is None
        llm.chat.assert_not_called()

    async def test_at_or_above_the_floor_it_routes(self):
        self._backend({"billing-agent": 0.65, "technical-agent": 0.25, "none": 0.1}, raw=0.5)
        assert await _router(system_one_min_confidence=0.5).route("x") == "billing-agent"

    async def test_it_reads_the_backends_confidence_not_the_sdks(self):
        """0.7/0.15/0.15 here: the SDK's entropy figure is ~0.25, the backend's 0.6."""
        self._backend({"billing-agent": 0.7, "technical-agent": 0.15, "none": 0.15}, raw=0.6)
        assert await _router(system_one_min_confidence=0.5).route("x") == "billing-agent"

    async def test_a_backend_that_reports_no_confidence_gets_no_route(self):
        """Opting in to a floor the backend cannot meet is not a silent pass."""
        import logging

        self._backend({"billing-agent": 0.9, "technical-agent": 0.05, "none": 0.05}, raw=None)
        records, handler = TestTheDecisionIsLogged()._logs()
        try:
            assert await _router(system_one_min_confidence=0.5).route("x") is None
        finally:
            TestTheDecisionIsLogged()._detach(handler)
        warnings = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
        assert any("reports no confidence" in m for m in warnings)

    async def test_none_is_still_no_route(self):
        self._backend({"billing-agent": 0.05, "technical-agent": 0.05, "none": 0.9}, raw=0.95)
        assert await _router(system_one_min_confidence=0.5).route("x") is None

    async def test_the_log_line_and_trace_say_why(self):
        self._backend({"billing-agent": 0.59, "technical-agent": 0.3, "none": 0.11}, raw=0.45)
        records, handler = TestTheDecisionIsLogged()._logs()
        span = TestProvenance()._span()
        try:
            with patch("continuum.agent.workflow.router.SpanScope", return_value=span):
                await _router(system_one_min_confidence=0.5).route("x")
        finally:
            TestTheDecisionIsLogged()._detach(handler)
        line = next(r.getMessage() for r in records if "decided_by=system_one" in r.getMessage())
        assert "route=billing-agent" in line
        assert "confidence=0.450" in line
        assert "below min_confidence=0.5" in line
        out = span.set_output.call_args.args[0]
        assert out["selected_route"] is None
        assert out["low_confidence"] is True
        assert out["raw_confidence"] == pytest.approx(0.45)

    async def test_execute_hands_a_low_confidence_request_to_the_fallback(self):
        from continuum.agent.types import AgentResponse

        self._backend({"billing-agent": 0.59, "technical-agent": 0.3, "none": 0.11}, raw=0.45)
        runner = MagicMock()
        runner.llm_client = _llm()
        runner.get_agent = MagicMock(side_effect=lambda name: MagicMock(name=name))
        runner.run = AsyncMock(return_value=AgentResponse(content="ok"))
        await _router(system_one_min_confidence=0.5).execute("x", runner)
        assert runner.get_agent.call_args.args[0] == "general-agent"

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_the_floor_must_be_a_probability(self, bad):
        from continuum.agent.config import RouterConfig

        with pytest.raises(ValueError):
            RouterConfig(system_one_min_confidence=bad)

    def test_the_floor_is_serialised(self):
        from continuum.agent.config import RouterConfig

        assert RouterConfig(system_one_min_confidence=0.5).to_dict()[
            "system_one_min_confidence"
        ] == pytest.approx(0.5)
        assert RouterConfig().system_one_min_confidence is None
