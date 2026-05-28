"""
Tests for MessageBuilder.prepare_messages() refactored behaviors:
- Returns (messages, user_message_index) tuple
- Injects pipeline_context from context.metadata as a system message
- Skips Redis session history when context.is_handoff=True
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agent.utils.context_utils import create_run_context


def _make_agent(system_prompt=None, session_history_turns=None, react_mode=False):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="builder-agent",
        instructions=system_prompt or "You are helpful.",
        config=AgentConfig(
            session_history_turns=session_history_turns,
            react_mode=react_mode,
            input_sanitization=False,
            injection_detection=False,
        ),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _make_builder(history=None):
    from orchestrator.agent.execution.message_builder import MessageBuilder

    mem_svc = MagicMock()
    mem_svc.retrieve_memories = AsyncMock(return_value=[])

    sess_svc = MagicMock()
    sess_svc.get_conversation_history = AsyncMock(return_value=history or [])

    return MessageBuilder(memory_service=mem_svc, session_service=sess_svc), mem_svc, sess_svc


class TestReturnsTuple:
    async def test_returns_tuple_of_messages_and_index(self):
        builder, _, _ = _make_builder()
        agent = _make_agent()
        ctx = create_run_context()

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await builder.prepare_messages(agent, "hello", ctx)

        assert isinstance(result, tuple)
        assert len(result) == 2
        messages, index = result
        assert isinstance(messages, list)
        assert isinstance(index, int)

    async def test_index_points_to_user_message(self):
        builder, _, _ = _make_builder()
        agent = _make_agent(system_prompt="system instructions")
        ctx = create_run_context()

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, index = await builder.prepare_messages(agent, "user question", ctx)

        assert messages[index]["role"] == "user"
        assert messages[index]["content"] == "user question"

    async def test_index_accounts_for_system_messages(self):
        builder, _, _ = _make_builder()
        # Agent has a system prompt → messages[0] = system, messages[1] = user
        agent = _make_agent(system_prompt="Be helpful.")
        ctx = create_run_context()

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, index = await builder.prepare_messages(agent, "hi", ctx)

        assert index >= 1
        assert messages[index]["role"] == "user"


class TestPipelineContextInjection:
    async def test_pipeline_context_injected_as_system_message(self):
        builder, _, _ = _make_builder()
        agent = _make_agent()
        ctx = create_run_context()
        ctx.metadata["pipeline_context"] = "Step 1 output: the sky is blue."

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, _ = await builder.prepare_messages(agent, "next question", ctx)

        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert any("Step 1 output: the sky is blue." in c for c in system_contents)

    async def test_no_pipeline_context_when_metadata_empty(self):
        builder, _, _ = _make_builder()
        agent = _make_agent()
        ctx = create_run_context()  # metadata = {}

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, _ = await builder.prepare_messages(agent, "q", ctx)

        # Should not inject any pipeline context message
        pipeline_msgs = [
            m
            for m in messages
            if m["role"] == "system" and "Prior pipeline" in (m.get("content") or "")
        ]
        assert len(pipeline_msgs) == 0

    async def test_pipeline_context_appears_before_user_message(self):
        builder, _, _ = _make_builder()
        agent = _make_agent()
        ctx = create_run_context()
        ctx.metadata["pipeline_context"] = "step context"

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, index = await builder.prepare_messages(agent, "q", ctx)

        pipeline_idx = next(
            i
            for i, m in enumerate(messages)
            if m["role"] == "system" and "step context" in (m.get("content") or "")
        )
        assert pipeline_idx < index


class TestHandoffSkipsHistory:
    async def test_history_not_loaded_on_handoff(self):
        builder, _, sess_svc = _make_builder()
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")
        ctx.is_handoff = True

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await builder.prepare_messages(agent, "handoff input", ctx)

        sess_svc.get_conversation_history.assert_not_called()

    async def test_history_loaded_on_normal_turn(self):
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        builder, _, sess_svc = _make_builder(history=history)
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")
        ctx.is_handoff = False

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            messages, _ = await builder.prepare_messages(agent, "new question", ctx)

        sess_svc.get_conversation_history.assert_called_once()
        # History should appear in messages
        contents = [m.get("content") for m in messages]
        assert "prev question" in contents
        assert "prev answer" in contents

    async def test_history_skipped_without_session_id(self):
        builder, _, sess_svc = _make_builder()
        agent = _make_agent()
        ctx = create_run_context()  # no session_id

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await builder.prepare_messages(agent, "q", ctx)

        sess_svc.get_conversation_history.assert_not_called()


class TestHistoryLimitDefault:
    async def test_default_limit_is_20_turns(self):
        builder, _, sess_svc = _make_builder()
        agent = _make_agent(session_history_turns=None)
        ctx = create_run_context(session_id="sess-1")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await builder.prepare_messages(agent, "q", ctx)

        sess_svc.get_conversation_history.assert_called_once_with("sess-1", limit=20)

    async def test_agent_specific_limit_overrides_default(self):
        builder, _, sess_svc = _make_builder()
        agent = _make_agent(session_history_turns=5)
        ctx = create_run_context(session_id="sess-1")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await builder.prepare_messages(agent, "q", ctx)

        sess_svc.get_conversation_history.assert_called_once_with("sess-1", limit=5)
