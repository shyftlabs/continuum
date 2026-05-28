"""
Tests for SessionService.save_messages() with the new user_message_index parameter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_service(add_message_mock=None):
    from orchestrator.agent.services.session_service import SessionService

    sc = MagicMock()
    sc.is_enabled = True
    sc.add_message = add_message_mock or AsyncMock()
    return SessionService(session_client=sc), sc


def _make_agent(name="agent-a", store_memories=False):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(store_memories=store_memories),
    )


def _msgs(*pairs):
    """Build a flat message list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in pairs]


class TestUserMessageIndexFiltering:
    async def test_saves_only_messages_after_index(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(
            ("system", "sys"),
            ("system", "sys2"),
            ("user", "hello"),
            ("assistant", "hi"),
        )
        await svc.save_messages(agent, messages, user_message_index=2, session_id="s1")

        # Only the user+assistant after index=2 should be saved (system skipped)
        saved_roles = [c.kwargs["message"].role for c in sc.add_message.call_args_list]
        assert saved_roles == ["user", "assistant"]

    async def test_index_zero_saves_all_non_system(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("user", "q"), ("assistant", "a"))
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        assert sc.add_message.call_count == 2

    async def test_index_beyond_list_saves_nothing(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("user", "q"), ("assistant", "a"))
        await svc.save_messages(agent, messages, user_message_index=10, session_id="s1")
        sc.add_message.assert_not_called()


class TestMessageTypeFiltering:
    async def test_skips_system_messages(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("system", "instructions"), ("user", "q"), ("assistant", "a"))
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        saved_roles = [c.kwargs["message"].role for c in sc.add_message.call_args_list]
        assert "system" not in saved_roles

    async def test_skips_tool_messages(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "run tool"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        saved_roles = [c.kwargs["message"].role for c in sc.add_message.call_args_list]
        assert "tool" not in saved_roles

    async def test_skips_intermediate_assistant_with_tool_calls(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "final answer"},
        ]
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        saved_contents = [c.kwargs["message"].content for c in sc.add_message.call_args_list]
        assert "final answer" in saved_contents
        # Intermediate assistant with tool_calls must not be saved
        assert None not in saved_contents

    async def test_saves_final_assistant_without_tool_calls(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("user", "q"), ("assistant", "clean answer"))
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        saved = [c.kwargs["message"].content for c in sc.add_message.call_args_list]
        assert "clean answer" in saved


class TestToolExecutionSummary:
    async def test_summary_attached_to_assistant_message_metadata(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("user", "q"), ("assistant", "done"))
        summary = {"tool_count": 2, "latency_ms": 300}
        await svc.save_messages(
            agent,
            messages,
            user_message_index=0,
            session_id="s1",
            tool_execution_summary=summary,
        )
        calls = {c.kwargs["message"].role: c for c in sc.add_message.call_args_list}
        assistant_metadata = calls["assistant"].kwargs.get("metadata", {})
        assert assistant_metadata.get("tool_execution_summary") == summary

    async def test_no_summary_when_not_provided(self):
        svc, sc = _make_service()
        agent = _make_agent()
        messages = _msgs(("user", "q"), ("assistant", "done"))
        await svc.save_messages(agent, messages, user_message_index=0, session_id="s1")
        for call in sc.add_message.call_args_list:
            meta = call.kwargs.get("metadata", {})
            assert "tool_execution_summary" not in meta


class TestSessionDisabled:
    async def test_noop_when_session_disabled(self):
        from orchestrator.agent.services.session_service import SessionService

        sc = MagicMock()
        sc.is_enabled = False
        sc.add_message = AsyncMock()
        svc = SessionService(session_client=sc)
        agent = _make_agent()

        await svc.save_messages(agent, _msgs(("user", "q")), user_message_index=0, session_id="s1")
        sc.add_message.assert_not_called()

    async def test_noop_when_no_session_client(self):
        from orchestrator.agent.services.session_service import SessionService

        svc = SessionService(session_client=None)
        agent = _make_agent()
        # Should not raise
        await svc.save_messages(agent, _msgs(("user", "q")), user_message_index=0, session_id="s1")


class TestHistoryDefaultLimit:
    async def test_get_conversation_history_default_limit_is_20(self):
        from orchestrator.agent.services.session_service import SessionService

        sc = MagicMock()
        sc.is_enabled = True
        sc.get_conversation_history = AsyncMock(return_value=[])
        svc = SessionService(session_client=sc)

        await svc.get_conversation_history("sess-1")
        sc.get_conversation_history.assert_called_once_with("sess-1", limit=20)

    async def test_get_conversation_history_custom_limit(self):
        from orchestrator.agent.services.session_service import SessionService

        sc = MagicMock()
        sc.is_enabled = True
        sc.get_conversation_history = AsyncMock(return_value=[])
        svc = SessionService(session_client=sc)

        await svc.get_conversation_history("sess-1", limit=5)
        sc.get_conversation_history.assert_called_once_with("sess-1", limit=5)
