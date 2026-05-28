"""
Tests for AgentRunner.save_turn() — the method used by workflow agents to write
exactly one (user, assistant) pair to Redis after a multi-step pipeline finishes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def _make_runner(session_client=None):
    """Build an AgentRunner with all heavy dependencies mocked out."""
    from orchestrator.agent.runner import AgentRunner

    llm = MagicMock()
    llm.is_enabled = True

    if session_client is None:
        sc = MagicMock()
        sc.is_enabled = True
        sc.add_message = AsyncMock()
    else:
        sc = session_client

    with patch("orchestrator.agent.runner.get_container") as mock_container:
        container = MagicMock()
        container.llm_client = llm
        container.memory_client = MagicMock()
        container.session_client = sc
        container.tool_executor = MagicMock()
        mock_container.return_value = container

        runner = AgentRunner(
            llm_client=llm,
            session_client=sc,
            memory_client=MagicMock(),
            tool_executor=MagicMock(),
        )
    runner._session_client = sc
    return runner, sc


def _make_agent(store_memories=False, extraction_prompt=None):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    mem_cfg = AgentMemoryConfig(
        store_memories=store_memories,
        extraction_prompt=extraction_prompt,
    )
    return BaseAgent(
        name="test-agent",
        instructions="You are helpful.",
        config=AgentConfig(),
        memory_config=mem_cfg,
    )


class TestSaveTurnBasic:
    async def test_writes_user_then_assistant(self):
        runner, sc = _make_runner()
        await runner.save_turn("sess-1", "hello", "world")

        assert sc.add_message.call_count == 2
        calls = sc.add_message.call_args_list
        first_msg = calls[0].kwargs["message"]
        second_msg = calls[1].kwargs["message"]
        assert first_msg.role == "user"
        assert first_msg.content == "hello"
        assert second_msg.role == "assistant"
        assert second_msg.content == "world"

    async def test_passes_session_id_to_both_calls(self):
        runner, sc = _make_runner()
        await runner.save_turn("sess-xyz", "q", "a")

        for call in sc.add_message.call_args_list:
            assert call.kwargs["session_id"] == "sess-xyz"

    async def test_noop_when_session_disabled(self):
        sc = MagicMock()
        sc.is_enabled = False
        sc.add_message = AsyncMock()
        runner, _ = _make_runner(session_client=sc)
        runner._session_client = sc

        await runner.save_turn("sess-1", "q", "a")
        sc.add_message.assert_not_called()

    async def test_noop_when_session_client_is_none(self):
        runner, sc = _make_runner()
        runner._session_client = None

        await runner.save_turn("sess-1", "q", "a")
        sc.add_message.assert_not_called()


class TestSaveTurnWithAgent:
    async def test_no_memory_storage_when_agent_is_none(self):
        runner, sc = _make_runner()
        await runner.save_turn("sess-1", "q", "a", agent=None)

        for call in sc.add_message.call_args_list:
            assert call.kwargs.get("store_in_memory") is False

    async def test_no_memory_storage_when_store_memories_false(self):
        runner, sc = _make_runner()
        agent = _make_agent(store_memories=False)
        await runner.save_turn("sess-1", "q", "a", agent=agent)

        for call in sc.add_message.call_args_list:
            assert call.kwargs.get("store_in_memory") is False

    async def test_memory_storage_when_store_memories_true(self):
        runner, sc = _make_runner()
        agent = _make_agent(store_memories=True)
        await runner.save_turn("sess-1", "q", "a", agent=agent)

        for call in sc.add_message.call_args_list:
            assert call.kwargs.get("store_in_memory") is True

    async def test_agent_id_passed_when_agent_provided(self):
        runner, sc = _make_runner()
        agent = _make_agent()
        await runner.save_turn("sess-1", "q", "a", agent=agent)

        for call in sc.add_message.call_args_list:
            assert call.kwargs.get("agent_id") == "test-agent"

    async def test_extraction_prompt_passed_through(self):
        runner, sc = _make_runner()
        agent = _make_agent(store_memories=True, extraction_prompt="Extract facts only.")
        await runner.save_turn("sess-1", "q", "a", agent=agent)

        for call in sc.add_message.call_args_list:
            assert call.kwargs.get("extraction_prompt") == "Extract facts only."
