"""
Phase 2 — data-label MODEL-ROUTING gate.

A run tainted with a label (e.g. "pii") can be denied a model/provider via
policy: `deny(subjects=["pii"], resources=["llm:gemini/*"])`. Enforcement lives
in LLMClient.chat() — the single chokepoint all LLM calls pass through — and
mirrors the tool gate: data labels are folded into the policy `subjects`, the
model is the resource (`llm:<model>`), and a deny raises before the provider is
ever selected or called.

These tests use a mock PolicyStore + patched get_provider, so they assert the
wiring (subjects/resource, deny-before-send) without real network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from continuum.agent.utils.context_utils import create_run_context
from continuum.llm.client import LLMClient
from continuum.llm.config import LLMConfig
from continuum.security.policy import PolicyDecision

MODEL = "gemini/gemini-2.5-flash"


def _client() -> LLMClient:
    return LLMClient(config=LLMConfig(model=MODEL), enable_langfuse=False)


def _fake_provider() -> MagicMock:
    provider = MagicMock()
    resp = MagicMock()
    resp.content = "ok"
    resp.tool_calls = None
    resp.function_call = None
    provider.acomplete = AsyncMock(return_value=resp)
    return provider


async def _chat(client, **policy_kwargs):
    return await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        auto_session=False,
        **policy_kwargs,
    )


class TestModelRoutingPolicy:
    def test_exception_exists(self):
        from continuum.agent.exceptions import ModelAccessDeniedError

        err = ModelAccessDeniedError(model=MODEL, policy_name="p")
        assert MODEL in str(err)

    async def test_denied_label_raises_before_provider(self):
        from continuum.agent.exceptions import ModelAccessDeniedError

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(
            allowed=False, policy_name="no_pii_gemini", reason="deny"
        )
        provider = _fake_provider()
        with patch("continuum.llm.client.get_provider", return_value=provider) as gp:
            with pytest.raises(ModelAccessDeniedError):
                await _chat(_client(), policy_store=ps, policy_subject="agent", data_labels={"pii"})
            # Blocked before the provider is even selected or called.
            gp.assert_not_called()
            provider.acomplete.assert_not_called()
        ps.check.assert_called_once_with(["agent", "pii"], f"llm:{MODEL}")

    async def test_allowed_label_proceeds_to_provider(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        provider = _fake_provider()
        with patch("continuum.llm.client.get_provider", return_value=provider):
            await _chat(_client(), policy_store=ps, policy_subject="agent", data_labels={"pii"})
            provider.acomplete.assert_awaited_once()
        ps.check.assert_called_once_with(["agent", "pii"], f"llm:{MODEL}")

    async def test_no_policy_store_means_no_gate(self):
        provider = _fake_provider()
        with patch("continuum.llm.client.get_provider", return_value=provider):
            await _chat(_client())  # no policy params at all
            provider.acomplete.assert_awaited_once()

    async def test_subject_only_when_no_labels(self):
        # With a policy_subject but no labels, the bare subject is checked.
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        provider = _fake_provider()
        with patch("continuum.llm.client.get_provider", return_value=provider):
            await _chat(_client(), policy_store=ps, policy_subject="agent")
        ps.check.assert_called_once_with("agent", f"llm:{MODEL}")


# ---------------------------------------------------------------------------
# Ambient policy context — chat() picks up the run's policy without the caller
# threading it, so every call site is gated automatically (not just the 2 wired).
# ---------------------------------------------------------------------------


def _fake_stream_provider() -> MagicMock:
    provider = MagicMock()

    async def astream(*a, **k):
        yield MagicMock(content="ok")

    provider.astream = astream
    return provider


class TestChatStreamGate:
    """Streaming runs go through chat_stream(), which must enforce the same
    model-routing gate as chat() — otherwise streaming bypasses it entirely."""

    async def test_stream_denied_raises_before_streaming(self):
        from continuum.agent.exceptions import ModelAccessDeniedError
        from continuum.security.policy_context import use_active_policy

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=False, policy_name="p", reason="deny")
        provider = _fake_stream_provider()
        ctx = create_run_context(data_labels={"pii"})
        with patch("continuum.llm.client.get_provider", return_value=provider) as gp:
            with use_active_policy(ps, "agent", ctx):
                with pytest.raises(ModelAccessDeniedError):
                    async for _ in _client().chat_stream(
                        messages=[{"role": "user", "content": "hi"}]
                    ):
                        pass
            gp.assert_not_called()  # blocked before provider selection
        assert call(["agent", "pii"], f"llm:{MODEL}") in ps.check.call_args_list

    async def test_stream_allowed_proceeds(self):
        from continuum.security.policy_context import use_active_policy

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        provider = _fake_stream_provider()
        ctx = create_run_context(data_labels={"pii"})
        chunks = []
        with patch("continuum.llm.client.get_provider", return_value=provider):
            with use_active_policy(ps, "agent", ctx):
                async for chunk in _client().chat_stream(
                    messages=[{"role": "user", "content": "hi"}]
                ):
                    chunks.append(chunk)
        assert len(chunks) == 1


class TestAmbientPolicyContext:
    def test_use_active_policy_sets_and_resets(self):
        from continuum.security.policy_context import get_active_policy, use_active_policy

        assert get_active_policy() is None
        ctx = create_run_context(data_labels={"pii"})
        with use_active_policy(MagicMock(), "agent", ctx):
            ap = get_active_policy()
            assert ap is not None
            assert ap.subject == "agent"
            assert ap.data_labels == {"pii"}
        assert get_active_policy() is None  # restored on exit

    def test_active_policy_reads_labels_live(self):
        from continuum.security.policy_context import get_active_policy, use_active_policy

        ctx = create_run_context()
        with use_active_policy(MagicMock(), "agent", ctx):
            ctx.taint("phi")  # tainted AFTER the context was published
            assert get_active_policy().data_labels == {"phi"}

    async def test_chat_uses_ambient_policy_when_not_passed(self):
        from continuum.agent.exceptions import ModelAccessDeniedError
        from continuum.security.policy_context import use_active_policy

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(
            allowed=False, policy_name="no_pii_gemini", reason="deny"
        )
        provider = _fake_provider()
        ctx = create_run_context(data_labels={"pii"})
        with patch("continuum.llm.client.get_provider", return_value=provider):
            with use_active_policy(ps, "agent", ctx):
                # No policy args passed to chat() — it must read the ambient context.
                with pytest.raises(ModelAccessDeniedError):
                    await _chat(_client())
            provider.acomplete.assert_not_called()
        # The model-routing gate consulted the ambient policy with the llm: resource.
        # (@observe telemetry redaction may add "telemetry" checks too, so assert
        # the specific gate call rather than a total count.)
        assert call(["agent", "pii"], f"llm:{MODEL}") in ps.check.call_args_list

    async def test_explicit_params_take_precedence_over_ambient(self):
        from continuum.security.policy_context import use_active_policy

        ambient_ps = MagicMock()
        ambient_ps.check.return_value = PolicyDecision(allowed=False, reason="deny")
        explicit_ps = MagicMock()
        explicit_ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        provider = _fake_provider()
        ctx = create_run_context(data_labels={"pii"})
        with patch("continuum.llm.client.get_provider", return_value=provider):
            with use_active_policy(ambient_ps, "ambient", ctx):
                # Explicit allow store passed → used instead of ambient deny store.
                await _chat(_client(), policy_store=explicit_ps, policy_subject="explicit")
            provider.acomplete.assert_awaited_once()
        # The LLM gate used the explicit store, not the ambient one. (The ambient
        # store may still be consulted for @observe telemetry redaction, so assert
        # specifically about the llm: resource rather than total call count.)
        assert any(c.args[1].startswith("llm:") for c in explicit_ps.check.call_args_list)
        assert not any(c.args[1].startswith("llm:") for c in ambient_ps.check.call_args_list)

    async def test_no_ambient_no_explicit_means_no_gate(self):
        provider = _fake_provider()
        with patch("continuum.llm.client.get_provider", return_value=provider):
            await _chat(_client())  # no ambient set, no explicit params
            provider.acomplete.assert_awaited_once()


class TestExecuteLoopPublishesAmbientPolicy:
    async def test_execute_loop_sets_ambient_for_nested_calls(self, monkeypatch):
        from continuum.agent.execution.executor import Executor
        from continuum.security.policy_context import get_active_policy

        captured: dict = {}

        async def fake_impl(self, agent, messages, context, run_state):
            captured["ap"] = get_active_policy()
            return MagicMock()

        monkeypatch.setattr(Executor, "_execute_loop_impl", fake_impl, raising=False)

        ex = Executor.__new__(Executor)
        agent = MagicMock()
        agent.policy_store = MagicMock()
        agent.name = "agent-x"
        ctx = create_run_context(data_labels={"pii"})

        await ex.execute_loop(agent, [], ctx, MagicMock())

        ap = captured["ap"]
        assert ap is not None
        assert ap.subject == "agent-x"
        assert ap.data_labels == {"pii"}
        # And it's reset after execute_loop returns.
        assert get_active_policy() is None


class TestRunnerPublishesAmbientPolicyRunWide:
    """The outermost run() wrap must publish the ambient for the WHOLE run, not
    just execute_loop — so smart-layer/workflow chat() calls are gated too.

    We capture the ambient inside finalizer.finalize, which runs AFTER
    execute_loop returns (its nested ambient already reset). If the ambient is
    still set there, only the run-level wrap can be holding it.
    """

    async def test_run_publishes_ambient_outside_execute_loop(self, monkeypatch):
        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.types import AgentResponse, ResponseStatus
        from continuum.security.policy import PolicyStore
        from continuum.security.policy_context import get_active_policy

        runner = AgentRunner()

        captured: dict = {}

        async def fake_execute_loop(agent, messages, context, run_state):
            return AgentResponse(content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS)

        async def fake_finalize(*args, **kwargs):
            # finalize runs after execute_loop returns — execute_loop's nested
            # ambient is already reset here, so a set value proves the run-level wrap.
            captured["ap"] = get_active_policy()

        monkeypatch.setattr(runner._executor, "execute_loop", fake_execute_loop)
        monkeypatch.setattr(runner._finalizer, "finalize", fake_finalize)

        agent = BaseAgent(name="run-agent", instructions="t", config=AgentConfig())
        agent.policy_store = PolicyStore()
        ctx = create_run_context(data_labels={"pii"})

        await runner.run(agent, "hello", context=ctx)

        ap = captured["ap"]
        assert ap is not None, "run() did not publish the ambient policy run-wide"
        assert ap.subject == "run-agent"
        assert ap.data_labels == {"pii"}
        # Cleaned up once run() exits (finally).
        assert get_active_policy() is None
