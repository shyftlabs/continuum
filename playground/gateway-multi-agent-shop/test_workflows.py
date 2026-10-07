"""How this demo's workflows are configured.

Lives beside the demo rather than under tests/, like
gateway-local-shop/test_agent_logging.py: it asserts something about *this
playground's* code, and a bare `pytest` does not collect it (testpaths =
["tests"]). Run it by path:

    pytest playground/gateway-multi-agent-shop/test_workflows.py

The naming split in this directory: `*_test.py` files are runnable demo scripts
that want live servers (headroom_multiagent_test.py); a `test_*.py` prefix means
pytest.

Run one playground's tests at a time. This directory and gateway-local-shop both
have bare modules named config, cli, server and web, and neither can be a
package (the hyphens). pytest's default prepend import mode puts each test file's
directory at sys.path[0] at *collection* time, so a single command naming both
playgrounds' test files resolves `config` to whichever was collected last, and
the other playground's tests import the wrong one. This file's own imports are
scoped (see _this_playground) so its tests pass either way; the other
playground's may not. It cannot be fixed from inside a test file, because it
happens before any test runs.
"""

from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# Modules this playground shares a name with in gateway-local-shop, imported bare
# -- workflows.py does `from config import ...`. In one pytest process that
# collects both playgrounds' tests, a bare import resolves to whichever directory
# is first on sys.path and whichever copy is already in sys.modules, so
# workflows.py gets local-shop's config.py, or local-shop's agent.py gets this one.
_SHARED_NAMES = ("config", "workflows", "web", "cli")


@contextlib.contextmanager
def _this_playground():
    """Resolve bare imports to this directory for the duration, then put the
    process back exactly as it was.

    Scoped deliberately. A first version purged the foreign copies and left this
    directory at the front of sys.path, which only moved the collision: every
    local-shop test that ran afterwards imported this playground's config.py and
    failed -- 29 of them.
    """
    saved_path = list(sys.path)
    saved_modules = {n: sys.modules.pop(n) for n in _SHARED_NAMES if n in sys.modules}
    sys.path.insert(0, HERE)
    try:
        yield
    finally:
        for name in _SHARED_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _debate_agent():
    """DebateShop's DebateAgent, built without connecting to anything."""
    with _this_playground():
        from config import default_config
        from workflows import DebateShop

        shop = DebateShop.__new__(DebateShop)
        shop.config = default_config
        shop._build_workflow()
    return shop._agent


class TestDebateShop:
    """A live run showed the judge receiving the first 2000 characters of each
    side -- 49% of a 4056-character argument and 58% of a 3436-character one --
    because this demo used DebateConfig's defaults: summarise_arguments=False,
    truncate_chars=2000. The cut is a plain character slice, so it drops each
    argument's end, which is where a persuasive case lands its conclusion.

    The demo exists to show a debate being judged, so it uses the mode the SDK
    provides for exactly this: each side condenses its own argument into bullet
    points before the judge sees it, keeping what matters rather than whatever
    fits in the first 2000 characters. Two extra LLM calls, run in parallel,
    and only for a side that is actually over the limit.
    """

    def test_each_side_summarises_its_own_argument_for_the_judge(self):
        assert _debate_agent().debate_config.summarise_arguments is True

    def test_the_limit_still_decides_which_sides_need_summarising(self):
        """A side within the limit reaches the judge verbatim, with no extra
        call; only a side over it is summarised. The limit is not disabled."""
        assert _debate_agent().debate_config.truncate_chars == 2000


# =============================================================================
# System One modes: router / loop / reflection / supervised, opted in
# =============================================================================
# The four original modes are unchanged whatever .env says. Four more modes run
# the same workflows with System One turned on, using the backend .env names in
# SYSTEM_ONE_BACKEND (Jev, Laya, a local NLI model...). Without one, the System
# One modes are offered but refuse to run, with a message saying what to set --
# never a silent run of the plain workflow under a System One label.

import asyncio  # noqa: E402

import pytest  # noqa: E402

JEV = "openrouter:typesafe/jev-1.13"
SYSTEM_ONE_MODES = {
    "router-system-one": "router",
    "loop-system-one": "loop",
    "reflection-system-one": "reflection",
    "supervised-system-one": "supervised",
}


@pytest.fixture
def backend(monkeypatch):
    """Like SYSTEM_ONE_BACKEND set in .env. Returns a setter; None means unset."""
    from continuum.config import settings
    from continuum.core.container import reset_container
    from continuum.system_one.registry import clear_classifier_cache

    def set_backend(spec):
        monkeypatch.setattr(settings, "system_one_backend", spec)
        reset_container()
        clear_classifier_cache()

    # The backend's key, so "configured" does not depend on the real .env.
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "system_one_disabled", False)
    set_backend(None)
    yield set_backend
    clear_classifier_cache()
    reset_container()


@pytest.fixture
def no_key(monkeypatch):
    """The backend is named in .env but its API key is not (commented out)."""
    from continuum.config import settings
    from continuum.system_one.registry import clear_classifier_cache

    monkeypatch.setattr(settings, "openrouter_api_key", None)
    clear_classifier_cache()


def _shop(mode):
    """A mode's workflow, built without connecting to anything."""
    with _this_playground():
        from config import default_config
        from workflows import MODES

        cls = MODES[mode]
        shop = cls.__new__(cls)
        shop.config = default_config
        shop._tools, shop._tool_executor = [], None
        shop._build_workflow()
    return shop


class TestTheOriginalModesAreUnchanged:
    """Whatever .env says, the four original modes never opt in."""

    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    def test_router_still_routes_with_the_llm(self):
        assert _shop("router")._agent.router_config.routing_strategy == "llm"

    def test_loop_still_stops_on_the_found_marker(self):
        from continuum.agent.types import TerminationType

        term = _shop("loop")._agent.termination
        assert term.type == TerminationType.OUTPUT_MATCH
        assert term.pattern == "FOUND:"

    def test_reflection_and_supervised_still_use_the_llm_judge(self):
        assert _shop("reflection")._agent.reflection_config.verdict_mode == "llm"
        assert _shop("supervised")._agent.supervised_config.verdict_mode == "llm"


class TestTheSystemOneModes:
    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    def test_all_four_are_offered_next_to_the_originals(self):
        with _this_playground():
            from workflows import MODES
        for mode, original in SYSTEM_ONE_MODES.items():
            assert mode in MODES and original in MODES
            assert issubclass(MODES[mode], MODES[original]), "same workflow, opted in"

    def test_router_system_one_asks_system_one_over_the_same_routes(self):
        plain, s1 = _shop("router")._agent, _shop("router-system-one")._agent
        assert s1.router_config.routing_strategy == "system_one_classifier"
        assert [r.agent_name for r in s1.routes] == [r.agent_name for r in plain.routes]
        # The fallback differs on purpose: see test_an_unsure_router_asks_a_clarifying_question.

    def test_router_system_one_acts_only_on_a_confident_route(self):
        """TypeSafe's intent-routing floor: below confidence 0.5, don't act on the
        top route -- the request goes to the fallback (support-agent)."""
        assert _shop("router-system-one")._agent.router_config.system_one_min_confidence == 0.5
        assert _shop("router")._agent.router_config.system_one_min_confidence is None

    def test_an_unsure_router_asks_a_clarifying_question(self):
        """Live, the floor sent 'my puppy keeps chewing shoes, find a toy for that and
        add it to my cart' (Jev confidence 0.40) to support-agent, which has no tools
        and replied "I can't assist with adding items to your cart". router-system-one
        now falls back to a clarify-agent that asks instead of answering."""
        s1 = _shop("router-system-one")
        assert s1._agent.fallback_agent_name == "clarify-agent"
        clarify = s1._specialist_agents["clarify-agent"]
        assert not clarify.tools, "it asks; it does not act"
        assert "clarifying question" in clarify.instructions
        # Routing is per message, without history: a bare "yes" would be unsure
        # again, so the agent asks for one concrete request the router can place.
        assert "one concrete request" in clarify.instructions

    def test_support_no_longer_claims_unclear_requests(self):
        """Its route said "... greetings, unclear intent", so a vague request could
        be routed to support-agent with full confidence instead of reaching the
        clarifier. In router-system-one, unclear means ask."""
        desc = {r.agent_name: r.description for r in _shop("router-system-one")._agent.routes}
        assert "unclear intent" not in desc["support-agent"]
        for kept in ("pet care advice", "nutrition questions", "general help", "greetings"):
            assert kept in desc["support-agent"]

    def test_the_plain_router_keeps_its_support_route(self):
        desc = {r.agent_name: r.description for r in _shop("router")._agent.routes}
        assert "unclear intent" in desc["support-agent"]

    def test_the_plain_router_still_falls_back_to_support(self):
        plain = _shop("router")
        assert plain._agent.fallback_agent_name == "support-agent"
        assert "clarify-agent" not in plain._specialist_agents

    def test_support_stays_a_real_route(self):
        """A pet-care question still reaches support-agent, not the clarifier."""
        routes = [r.agent_name for r in _shop("router-system-one")._agent.routes]
        assert "support-agent" in routes and "clarify-agent" not in routes

    def test_no_route_hands_the_message_to_the_clarifier(self):
        from unittest.mock import AsyncMock, MagicMock

        from continuum.agent.types import AgentResponse

        shop = _shop("router-system-one")
        shop._initialized, shop._container = True, None
        shop._agent.route = AsyncMock(return_value=None)
        shop._runner = MagicMock()
        shop._runner.run = AsyncMock(return_value=AgentResponse(content="Which would you like?"))
        with patch_container():
            reply = asyncio.run(shop.chat("find a toy and add it", "u", "c"))
        assert shop._runner.run.await_args.kwargs["agent"].name == "clarify-agent"
        assert reply.startswith("[→ clarify-agent]")

    def test_both_router_modes_offer_an_ambiguous_query(self):
        with _this_playground():
            import cli
        # Live, Jev's confidence on this one was 0.34-0.43 in 4 calls: below the
        # 0.5 floor, so router-system-one hands it to the fallback.
        assert any("chewing shoes" in q for q in cli.EXAMPLES["router"])

    def test_loop_system_one_lets_system_one_decide_completion(self):
        from continuum.agent.types import TerminationType

        term = _shop("loop-system-one")._agent.termination
        assert term.type == TerminationType.SYSTEM_ONE_CLASSIFIER
        assert term.max_iterations == 5

    def test_reflection_system_one_keeps_the_shops_checklist(self):
        plain = _shop("reflection")._agent.reflection_config
        s1 = _shop("reflection-system-one")._agent.reflection_config
        assert s1.verdict_mode == "system_one_classifier"
        assert s1.critique_prompt == plain.critique_prompt, "the checklist reaches System One"

    def test_supervised_system_one_keeps_the_quality_bar(self):
        s1 = _shop("supervised-system-one")._agent.supervised_config
        assert s1.verdict_mode == "system_one_classifier"
        assert s1.quality_threshold == pytest.approx(0.7)

    @pytest.mark.parametrize("mode", list(SYSTEM_ONE_MODES))
    def test_they_use_the_backend_named_in_env(self, mode):
        """No backend spec of their own, so SYSTEM_ONE_BACKEND decides."""
        agent = _shop(mode)._agent
        cfg = {
            "router-system-one": lambda a: a.router_config,
            "loop-system-one": lambda a: a.termination,
            "reflection-system-one": lambda a: a.reflection_config,
            "supervised-system-one": lambda a: a.supervised_config,
        }[mode](agent)
        assert cfg.system_one_backend is None

    def test_every_mode_has_a_description(self):
        with _this_playground():
            from config import default_config
        for mode in SYSTEM_ONE_MODES:
            assert "System One" in default_config.mode_descriptions[mode]


class TestWithoutABackendInEnv:
    def test_the_system_one_modes_say_what_to_set(self, backend):
        with _this_playground():
            from workflows import system_one_unavailable
        for mode in SYSTEM_ONE_MODES:
            msg = system_one_unavailable(mode)
            assert msg and "SYSTEM_ONE_BACKEND" in msg
        for original in SYSTEM_ONE_MODES.values():
            assert system_one_unavailable(original) is None

    def test_a_system_one_mode_cannot_be_built_without_one(self, backend):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError):
            _shop("router-system-one")

    def test_the_original_modes_still_build(self, backend):
        for original in SYSTEM_ONE_MODES.values():
            _shop(original)

    def test_status_reports_it(self, backend):
        with _this_playground():
            from workflows import system_one_status
        assert system_one_status()["configured"] is False
        backend(JEV)
        with _this_playground():
            from workflows import system_one_status
        status = system_one_status()
        assert status["configured"] is True and status["backend"] == JEV


class TestWithABackendButNoKey:
    """SYSTEM_ONE_BACKEND set but its key commented out: the backend cannot be
    built, so every System One decision would fail and fall back while the
    dropdown offered the modes as working."""

    def test_status_is_configured_but_not_ready(self, backend, no_key):
        backend(JEV)
        with _this_playground():
            from workflows import system_one_status
        status = system_one_status()
        assert status["configured"] is True
        assert status["ready"] is False
        assert "OPENROUTER_API_KEY" in status["problem"]

    def test_with_the_key_it_is_ready(self, backend):
        backend(JEV)
        with _this_playground():
            from workflows import system_one_status
        status = system_one_status()
        assert status["ready"] is True and status["problem"] is None

    def test_the_system_one_modes_name_the_missing_key(self, backend, no_key):
        backend(JEV)
        with _this_playground():
            from workflows import system_one_unavailable
        for mode in SYSTEM_ONE_MODES:
            msg = system_one_unavailable(mode)
            assert msg and "OPENROUTER_API_KEY" in msg
        for original in SYSTEM_ONE_MODES.values():
            assert system_one_unavailable(original) is None

    def test_the_page_shows_them_disabled_naming_the_key(self, backend, no_key):
        backend(JEV)
        with _this_playground():
            import web
        page = asyncio.run(web.index())
        for mode in SYSTEM_ONE_MODES:
            assert f'value="{mode}" disabled' in page
        assert "OPENROUTER_API_KEY" in page


class TestTheReplyShowsWhatSystemOneDid:
    """The workflows log each System One decision; a System One mode appends the
    ones made for this message to its reply, so the UI shows them."""

    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    def _chat(self, mode, *, logged=(), before=()):
        import logging
        from unittest.mock import patch

        shop = _shop(mode)
        gate = logging.getLogger("continuum.agent.workflow._quality_gate")
        for line in before:
            gate.info(line)

        async def fake_chat(self, message, user_id, conversation_id):
            # A plain string is a quality-gate line; (module, line) logs on
            # continuum.agent.workflow.<module>.
            for item in logged:
                module, line = item if isinstance(item, tuple) else ("_quality_gate", item)
                logging.getLogger(f"continuum.agent.workflow.{module}").info(line)
            return "the reply"

        # Patch the classes this shop was built from: each _this_playground()
        # imports workflows afresh, so a new import would be a different copy.
        mro = {c.__name__: c for c in type(shop).__mro__}
        targets = [mro[n] for n in ("_BaseWorkflow", "RouterShop") if n in mro]
        with contextlib.ExitStack() as stack:
            for cls in targets:
                stack.enter_context(patch.object(cls, "chat", fake_chat))
            return asyncio.run(shop.chat("hi", "u", "c"))

    def test_the_decisions_are_appended_with_the_backend(self):
        reply = self._chat(
            "supervised-system-one",
            logged=["quality gate decided_by=system_one backend=openrouter p_pass=0.960 -> pass"],
        )
        assert reply.startswith("the reply")
        assert f"[System One · {JEV}]" in reply
        assert "p_pass=0.960 -> pass" in reply

    def test_lines_from_outside_this_message_are_not_shown(self):
        reply = self._chat(
            "supervised-system-one",
            before=["decided_by=system_one p_pass=0.111 (another message)"],
            logged=["decided_by=system_one p_pass=0.950 -> pass"],
        )
        assert "0.950" in reply and "0.111" not in reply

    def test_two_messages_at_once_each_get_only_their_own_lines(self):
        """The web UI serves requests concurrently on one event loop: a reply
        must not pick up a decision made for another user's message."""
        import logging
        from unittest.mock import patch

        gate = logging.getLogger("continuum.agent.workflow._quality_gate")
        shop = _shop("reflection-system-one")

        async def fake_chat(self, message, user_id, conversation_id):
            await asyncio.sleep(0)
            gate.info(f"decided_by=system_one for {message}")
            await asyncio.sleep(0)
            return f"reply to {message}"

        async def both():
            return await asyncio.gather(shop.chat("A", "u1", "c1"), shop.chat("B", "u2", "c2"))

        base = {c.__name__: c for c in type(shop).__mro__}["_BaseWorkflow"]
        with patch.object(base, "chat", fake_chat):
            reply_a, reply_b = asyncio.run(both())
        assert "for A" in reply_a and "for B" not in reply_a
        assert "for B" in reply_b and "for A" not in reply_b

    def test_no_decision_is_said_plainly(self):
        reply = self._chat("loop-system-one")
        assert "no System One decision" in reply

    def test_the_router_mode_shows_them_too(self):
        reply = self._chat(
            "router-system-one",
            logged=[
                "Router 'router-shop' decided_by=system_one backend=openrouter route=cart-agent p=0.910"
            ],
        )
        assert "route=cart-agent" in reply

    def test_the_original_modes_add_nothing(self):
        reply = self._chat("reflection", logged=["decided_by=system_one p_pass=0.9"])
        assert reply == "the reply"


class TestTheWebUI:
    def _web(self):
        with _this_playground():
            import web
        return web

    def test_the_page_offers_the_system_one_modes(self, backend):
        backend(JEV)
        page = asyncio.run(self._web().index())
        for mode in SYSTEM_ONE_MODES:
            assert f'value="{mode}"' in page
            assert f'value="{mode}" disabled' not in page
        assert JEV in page

    def test_without_a_backend_they_are_shown_disabled(self, backend):
        page = asyncio.run(self._web().index())
        for mode in SYSTEM_ONE_MODES:
            assert f'value="{mode}" disabled' in page
        assert "SYSTEM_ONE_BACKEND" in page

    def test_chat_refuses_a_system_one_mode_without_building_it(self, backend):
        from unittest.mock import patch

        web = self._web()
        req = web.ChatRequest(
            message="hi", user_id="u", conversation_id="c", mode="router-system-one"
        )
        with patch.object(web, "create_workflow", side_effect=AssertionError("must not build")):
            out = asyncio.run(web.chat(req))
        assert "SYSTEM_ONE_BACKEND" in out["response"]

    def test_an_init_failure_with_no_message_is_reported_not_a_500(self, backend):
        """Live: an MCP ConnectTimeout reached get_workflow with an empty message,
        so `if error:` was false and chat() called .chat() on None -- HTTP 500,
        and again on every later request (the error is cached)."""
        from unittest.mock import patch

        web = self._web()
        web._init_errors.clear()
        web._workflows.clear()

        class Boom(Exception):
            def __str__(self):
                return ""

        failing = type("F", (), {"initialize": staticmethod(_raise(Boom()))})
        req = web.ChatRequest(message="hi", user_id="u", conversation_id="c", mode="router")
        try:
            with patch.object(web, "create_workflow", return_value=failing()):
                first = asyncio.run(web.chat(req))
            again = asyncio.run(web.chat(req))
        finally:
            web._init_errors.clear()
        for out in (first, again):
            assert "Failed to initialize 'router'" in out["response"]
            assert "Boom" in out["response"], "an empty message still names the error"

    def test_status_includes_system_one(self, backend):
        backend(JEV)
        status = asyncio.run(self._web().status())
        assert status["system_one"]["configured"] is True
        assert status["system_one"]["backend"] == JEV


class TestTheCLI:
    def test_every_system_one_mode_has_example_queries(self):
        with _this_playground():
            import cli
        for mode, original in SYSTEM_ONE_MODES.items():
            assert cli.EXAMPLES[mode] == cli.EXAMPLES[original]


# The reflection mode pairs each draft's System One score with the critic's
# verdict on it. The lines are the SDK's own log formats (reflection.py,
# _quality_gate.py); the critic's reason is content and is not shown.
_ATTEMPT = "ReflectionAgent 'reflection-shop': attempt {n} / 3"
_GATE = (
    "ReflectionAgent 'reflection-shop' quality gate decided_by=system_one "
    "backend=openrouter p_pass={p} threshold=0.9 -> {to}"
)
_VERDICT = (
    "===== CRITIQUE VERDICT [reflection-shop] =====\n"
    "outcome={outcome} reason=<withheld>\n========================="
)


class TestTheCriticVerdictIsShownPerDraft:
    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    _chat = TestTheReplyShowsWhatSystemOneDid._chat

    def _reflect(self, *events):
        return self._chat("reflection-system-one", logged=list(events))

    def test_each_draft_shows_system_ones_score_and_the_critics_verdict(self):
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            _GATE.format(p="0.230", to="ask the LLM judge"),
            ("reflection", _VERDICT.format(outcome="NEEDS IMPROVEMENT")),
            ("reflection", _ATTEMPT.format(n=2)),
            _GATE.format(p="0.720", to="ask the LLM judge"),
            ("reflection", _VERDICT.format(outcome="PASS")),
        )
        assert "draft 1: System One p_pass 0.230 (needs ≥ 0.9) → critic: NEEDS IMPROVEMENT" in reply
        assert "draft 2: System One p_pass 0.720 (needs ≥ 0.9) → critic: PASS" in reply
        assert "<withheld>" not in reply, "the critic's reason is content"

    def test_a_confident_pass_says_no_critic_was_asked(self):
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            _GATE.format(p="0.960", to="pass"),
            (
                "reflection",
                "ReflectionAgent 'reflection-shop': passed by the System One classifier on attempt 1 (p_pass=0.960)",
            ),
        )
        assert "draft 1: System One p_pass 0.960 (≥ 0.9) → passed, no critic call" in reply

    def test_the_last_draft_is_marked_unchecked(self):
        """max_reflections=2: the third draft is returned without a check."""
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            _GATE.format(p="0.230", to="ask the LLM judge"),
            ("reflection", _VERDICT.format(outcome="NEEDS IMPROVEMENT")),
            ("reflection", _ATTEMPT.format(n=2)),
            _GATE.format(p="0.510", to="ask the LLM judge"),
            ("reflection", _VERDICT.format(outcome="NEEDS IMPROVEMENT")),
            ("reflection", _ATTEMPT.format(n=3)),
        )
        assert "draft 3: last attempt, returned without a check" in reply

    def test_an_unavailable_critic_is_said_so(self):
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            _GATE.format(p="0.400", to="ask the LLM judge"),
            (
                "reflection",
                "ReflectionAgent 'reflection-shop': critique unavailable on attempt 1; returning the current draft unverified (not treated as PASS)",
            ),
        )
        assert (
            "draft 1: System One p_pass 0.400 (needs ≥ 0.9) → critic: unavailable, returned unverified"
            in reply
        )

    def test_the_kill_switch_shows_the_critic_alone(self):
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            "ReflectionAgent 'reflection-shop' quality gate decided_by=legacy (SYSTEM_ONE_DISABLED)",
            ("reflection", _VERDICT.format(outcome="PASS")),
        )
        assert "draft 1: System One off (SYSTEM_ONE_DISABLED) → critic: PASS" in reply

    def test_a_failed_system_one_check_still_shows_the_critic(self):
        reply = self._reflect(
            ("reflection", _ATTEMPT.format(n=1)),
            "ReflectionAgent 'reflection-shop': System One quality check failed (SystemOneTimeoutError); using the LLM judge",
            ("reflection", _VERDICT.format(outcome="PASS")),
        )
        assert "draft 1: System One failed (SystemOneTimeoutError) → critic: PASS" in reply


def _raise(exc):
    async def _fail(*a, **k):
        raise exc

    return _fail


@contextlib.contextmanager
def patch_container():
    """RouterShop.chat asks the container for an LLM client; give it a stub."""
    from unittest.mock import MagicMock, patch

    fake = MagicMock()
    with patch("continuum.core.container.get_container", return_value=fake):
        yield fake


# =============================================================================
# The loop: a budget search, and saying when it ran out of rounds
# =============================================================================
# Live, loop-system-one on "find me something under $10" ran all 5 rounds:
# search_products has no price filter, the literal query "under $10" matched
# nothing, the agent asked the user which product type -- and inside a loop no
# one answers, so it asked again each round. Jev was right every time (a
# question does not complete the task), but the note never said the loop had
# stopped at its cap rather than finished.


class TestTheBudgetSearch:
    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    @pytest.mark.parametrize("mode", ["loop", "loop-system-one"])
    def test_the_searcher_filters_by_price_itself(self, mode):
        """Both loop modes, so a side-by-side comparison stays fair."""
        text = _shop(mode)._agent.agent.instructions
        assert "cannot filter by price" in text
        assert "compare their prices yourself" in text

    @pytest.mark.parametrize("mode", ["loop", "loop-system-one"])
    def test_the_searcher_does_not_ask_the_user(self, mode):
        assert "no one can answer" in _shop(mode)._agent.agent.instructions


class TestTheLoopNoteSaysWhenItRanOutOfRounds:
    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    _chat = TestTheReplyShowsWhatSystemOneDid._chat
    _CONTINUE = (
        "Loop termination decided_by=system_one backend=openrouter "
        "p_complete=0.020 threshold=0.5 -> continue"
    )

    def test_hitting_the_cap_is_said(self):
        reply = self._chat(
            "loop-system-one",
            logged=[("loop", self._CONTINUE)] * 5 + [("loop", "Loop reached max iterations (5)")],
        )
        assert "stopped at max_iterations=5 without completing" in reply

    def test_a_normal_stop_says_nothing_extra(self):
        reply = self._chat(
            "loop-system-one",
            logged=[("loop", self._CONTINUE.replace("0.020", "0.910").replace("continue", "stop"))],
        )
        assert "max_iterations" not in reply
        assert "-> stop" in reply


# The supervised mode pairs each attempt's System One score with the supervisor's
# score on it, like reflection's critic (SDK formats: supervised.py, _quality_gate.py).
_S_ATTEMPT = "SupervisedSequential step 1/1 'writer-agent' — attempt {n}"
_S_GATE = (
    "SupervisedSequentialAgent 'supervised-shop' step 1 quality gate decided_by=system_one "
    "backend=openrouter p_pass={p} threshold=0.9 -> {to}"
)
_S_SCORE = "SupervisedSequential step 1 'writer-agent' score={s} (threshold=0.7)"


class TestTheSupervisorScoreIsShownPerAttempt:
    @pytest.fixture(autouse=True)
    def _configured(self, backend):
        backend(JEV)

    _chat = TestTheReplyShowsWhatSystemOneDid._chat

    def _supervise(self, *events):
        return self._chat("supervised-system-one", logged=list(events))

    def test_a_first_attempt_the_supervisor_accepts(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            _S_GATE.format(p="0.830", to="ask the LLM judge"),
            ("supervised", _S_SCORE.format(s="0.85")),
            ("supervised", "SupervisedSequential step 1 passed (score=0.85)"),
        )
        assert (
            "step 1, attempt 1: System One p_pass 0.830 (needs ≥ 0.9) "
            "→ supervisor: score 0.85 (passes ≥ 0.7)" in reply
        )

    def test_a_retry_and_then_a_pass(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            _S_GATE.format(p="0.580", to="ask the LLM judge"),
            ("supervised", _S_SCORE.format(s="0.55")),
            (
                "supervised",
                "SupervisedSequential step 1 below threshold (score=0.55) — retrying with feedback",
            ),
            ("supervised", _S_ATTEMPT.format(n=2)),
            _S_GATE.format(p="0.700", to="ask the LLM judge"),
            ("supervised", _S_SCORE.format(s="0.80")),
        )
        assert (
            "attempt 1: System One p_pass 0.580 (needs ≥ 0.9) → supervisor: score 0.55 (needs ≥ 0.7), retried"
            in reply
        )
        assert (
            "attempt 2: System One p_pass 0.700 (needs ≥ 0.9) → supervisor: score 0.80 (passes ≥ 0.7)"
            in reply
        )

    def test_a_confident_pass_says_no_supervisor_was_asked(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            _S_GATE.format(p="0.940", to="pass"),
        )
        assert (
            "step 1, attempt 1: System One p_pass 0.940 (≥ 0.9) → passed, no supervisor call"
            in reply
        )

    def test_retries_running_out_is_said(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=3)),
            _S_GATE.format(p="0.500", to="ask the LLM judge"),
            ("supervised", _S_SCORE.format(s="0.62")),
            ("supervised", "SupervisedSequential step 1 exhausted retries (best score=0.65)"),
        )
        assert (
            "attempt 3: System One p_pass 0.500 (needs ≥ 0.9) → supervisor: score 0.62 (needs ≥ 0.7), retries exhausted"
            in reply
        )

    def test_an_unscored_output_is_said(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            _S_GATE.format(p="0.600", to="ask the LLM judge"),
            (
                "supervised",
                "SupervisedSequential step 1 'writer-agent': supervisor could not score the output (timeout); keeping it unscored",
            ),
        )
        assert (
            "attempt 1: System One p_pass 0.600 (needs ≥ 0.9) → supervisor: could not score, kept unscored"
            in reply
        )

    def test_the_kill_switch_shows_the_supervisor_alone(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            "SupervisedSequentialAgent 'supervised-shop' step 1 quality gate decided_by=legacy (SYSTEM_ONE_DISABLED)",
            ("supervised", _S_SCORE.format(s="0.85")),
        )
        assert (
            "attempt 1: System One off (SYSTEM_ONE_DISABLED) → supervisor: score 0.85 (passes ≥ 0.7)"
            in reply
        )

    def test_a_failed_system_one_check_still_shows_the_supervisor(self):
        reply = self._supervise(
            ("supervised", _S_ATTEMPT.format(n=1)),
            "SupervisedSequentialAgent 'supervised-shop' step 1: System One quality check failed (SystemOneTimeoutError); using the LLM judge",
            ("supervised", _S_SCORE.format(s="0.85")),
        )
        assert (
            "attempt 1: System One failed (SystemOneTimeoutError) → supervisor: score 0.85 (passes ≥ 0.7)"
            in reply
        )
