"""
MEDIUM fix — gate LLM calls made during message preparation.

`message_builder.prepare_messages` (called inside `_prepare_run`) can trigger
proactive context compression, which makes its own `llm_client.chat()` call for
summarization. That happens BEFORE the run-level ambient policy was published,
so the model-routing gate didn't cover it — a tainted run could send the
conversation to a denied model for summarization.

Fix: publish the ambient policy around the prepare_messages call too. This test
asserts the ambient is active while prepare_messages runs.
"""

from __future__ import annotations

from continuum.agent.utils.context_utils import create_run_context


class TestPrepareMessagesGated:
    async def test_prepare_messages_runs_under_ambient_policy(self, monkeypatch):
        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.types import AgentResponse, ResponseStatus
        from continuum.security.policy import PolicyStore
        from continuum.security.policy_context import get_active_policy

        runner = AgentRunner()
        captured: dict = {}

        async def spy_prepare(*args, **kwargs):
            captured["ap"] = get_active_policy()
            return ([], 0)  # (messages, user_message_index)

        async def fake_loop(agent, messages, context, run_state):
            return AgentResponse(
                content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS
            )

        async def fake_finalize(*a, **k):
            return None

        monkeypatch.setattr(runner._message_builder, "prepare_messages", spy_prepare)
        monkeypatch.setattr(runner._executor, "execute_loop", fake_loop)
        monkeypatch.setattr(runner._finalizer, "finalize", fake_finalize)

        agent = BaseAgent(name="prep-agent", instructions="t", config=AgentConfig())
        agent.policy_store = PolicyStore()
        ctx = create_run_context(data_labels={"pii"})

        await runner.run(agent, "hello", context=ctx)

        ap = captured["ap"]
        assert ap is not None, "prepare_messages ran without an ambient policy (compression ungated)"
        assert ap.subject == "prep-agent"
        assert ap.data_labels == {"pii"}
