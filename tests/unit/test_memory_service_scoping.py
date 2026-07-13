"""
Tests for MemoryService.retrieve_memories() isolation mode scoping.
Verifies each mode (user, agent, conversation, shared) passes the correct
identifier to memory_client.search().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.utils.context_utils import create_run_context


def _make_memory_result(results=None):
    result = MagicMock()
    result.results = results or []
    result.total_results = len(result.results)
    return result


def _make_memory_client(isolation="user", search_result=None):
    mc = MagicMock()
    mc.is_enabled = True
    mc.config = MagicMock()
    mc.config.memory_isolation = isolation
    mc.search = AsyncMock(return_value=search_result or _make_memory_result())
    return mc


def _make_agent(search_memories=True, search_limit=5):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="scoping-agent",
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(
            search_memories=search_memories,
            search_limit=search_limit,
        ),
    )


def _make_service(memory_client, session_client=None):
    from continuum.agent.services.memory_service import MemoryService

    return MemoryService(memory_client=memory_client, session_client=session_client)


class TestUserIsolation:
    async def test_passes_user_id_to_search(self):
        mc = _make_memory_client(isolation="user")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context(user_id="user-abc")

        await svc.retrieve_memories(agent, "query", ctx)

        mc.search.assert_called_once()
        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs["user_id"] == "user-abc"
        assert call_kwargs["agent_id"] is None
        assert call_kwargs.get("conversation_id") is None

    async def test_passes_none_agent_id(self):
        mc = _make_memory_client(isolation="user")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context(user_id="user-1")

        await svc.retrieve_memories(agent, "query", ctx)
        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs["agent_id"] is None


class TestAgentIsolation:
    async def test_falls_back_to_agent_name_without_session(self):
        mc = _make_memory_client(isolation="agent")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context()  # no session_id

        await svc.retrieve_memories(agent, "query", ctx)

        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs["agent_id"] == "scoping-agent"
        assert call_kwargs["user_id"] is None

    async def test_uses_session_metadata_agent_id_when_available(self):
        mc = _make_memory_client(isolation="agent")

        session_meta = MagicMock()
        session_meta.agent_id = "agent-from-metadata"
        sc = MagicMock()
        sc.is_enabled = True
        sc.get_session_metadata = AsyncMock(return_value=session_meta)

        svc = _make_service(mc, session_client=sc)
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")

        await svc.retrieve_memories(agent, "query", ctx)

        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs["agent_id"] == "agent-from-metadata"

    async def test_falls_back_to_agent_name_when_metadata_has_no_agent_id(self):
        mc = _make_memory_client(isolation="agent")

        session_meta = MagicMock()
        session_meta.agent_id = None
        sc = MagicMock()
        sc.is_enabled = True
        sc.get_session_metadata = AsyncMock(return_value=session_meta)

        svc = _make_service(mc, session_client=sc)
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")

        await svc.retrieve_memories(agent, "query", ctx)

        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs["agent_id"] == "scoping-agent"


class TestConversationIsolation:
    async def test_passes_conversation_id_to_search(self):
        mc = _make_memory_client(isolation="conversation")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context(conversation_id="conv-xyz")

        await svc.retrieve_memories(agent, "query", ctx)

        call_kwargs = mc.search.call_args.kwargs
        assert call_kwargs.get("conversation_id") == "conv-xyz"
        assert call_kwargs["user_id"] is None
        assert call_kwargs["agent_id"] is None

    async def test_warns_when_conversation_id_is_none(self):
        mc = _make_memory_client(isolation="conversation")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context()  # no conversation_id

        with patch("continuum.agent.services.memory_service.logger") as mock_logger:
            await svc.retrieve_memories(agent, "query", ctx)

        all_warnings = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert "conversation_id" in all_warnings

    async def test_search_still_called_when_conversation_id_missing(self):
        mc = _make_memory_client(isolation="conversation")
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context()

        await svc.retrieve_memories(agent, "query", ctx)
        mc.search.assert_called_once()


class TestSkipWhenDisabled:
    async def test_does_not_resolve_lazy_client_when_search_memories_false(self):
        from continuum.agent.services.memory_service import MemoryService

        resolver = MagicMock(return_value=_make_memory_client(isolation="user"))
        svc = MemoryService(memory_client_resolver=resolver)
        agent = _make_agent(search_memories=False)
        ctx = create_run_context(user_id="user-1")

        result = await svc.retrieve_memories(agent, "query", ctx)

        assert result == []
        resolver.assert_not_called()

    async def test_resolves_lazy_client_once_when_memory_is_requested(self):
        from continuum.agent.services.memory_service import MemoryService

        memory_client = _make_memory_client(isolation="user")
        resolver = MagicMock(return_value=memory_client)
        svc = MemoryService(memory_client_resolver=resolver)
        agent = _make_agent()
        ctx = create_run_context(user_id="user-1")

        await svc.retrieve_memories(agent, "first", ctx)
        await svc.retrieve_memories(agent, "second", ctx)

        resolver.assert_called_once_with()
        assert memory_client.search.call_count == 2

    async def test_returns_empty_when_search_memories_false(self):
        mc = _make_memory_client(isolation="user")
        svc = _make_service(mc)
        agent = _make_agent(search_memories=False)
        ctx = create_run_context(user_id="user-1")

        result = await svc.retrieve_memories(agent, "query", ctx)

        assert result == []
        mc.search.assert_not_called()

    async def test_returns_empty_when_no_memory_client(self):
        from continuum.agent.services.memory_service import MemoryService

        svc = MemoryService(memory_client=None)
        agent = _make_agent()
        ctx = create_run_context(user_id="user-1")

        result = await svc.retrieve_memories(agent, "query", ctx)
        assert result == []


class TestMemoryLoggingPrivacy:
    async def test_logs_exclude_query_memory_content_and_scope_identifiers(self):
        secret_query = "Find jane@example.test account 4242 with private escalation notes"
        secret_memory = "Jane Example owes 9100 and asked for a confidential refund"
        secret_user_id = "customer-jane-example"

        memory = MagicMock()
        memory.memory = secret_memory
        memory.user_id = secret_user_id
        memory.metadata = {"_user_id": secret_user_id}
        memory.score = 0.91
        memory.to_dict.return_value = {"memory": secret_memory, "user_id": secret_user_id}

        mc = _make_memory_client(
            isolation="user",
            search_result=_make_memory_result([memory]),
        )
        svc = _make_service(mc)
        agent = _make_agent()
        ctx = create_run_context(user_id=secret_user_id)

        with patch("continuum.agent.services.memory_service.logger") as mock_logger:
            result = await svc.retrieve_memories(agent, secret_query, ctx)

        log_text = "\n".join(
            str(call)
            for method in (
                mock_logger.debug,
                mock_logger.info,
                mock_logger.warning,
                mock_logger.error,
            )
            for call in method.call_args_list
        )
        assert secret_query not in log_text
        assert secret_query[:100] not in log_text
        assert secret_memory not in log_text
        assert secret_memory[:100] not in log_text
        assert secret_user_id not in log_text
        assert result == [{"memory": secret_memory, "user_id": secret_user_id}]

    async def test_empty_search_log_excludes_query_and_scope_identifier(self):
        secret_query = "Confidential prompt for jane@example.test account 4242"
        secret_user_id = "customer-jane-example"
        svc = _make_service(_make_memory_client(isolation="user"))
        agent = _make_agent()
        ctx = create_run_context(user_id=secret_user_id)

        with patch("continuum.agent.services.memory_service.logger") as mock_logger:
            result = await svc.retrieve_memories(agent, secret_query, ctx)

        log_text = "\n".join(
            str(call)
            for method in (
                mock_logger.debug,
                mock_logger.info,
                mock_logger.warning,
                mock_logger.error,
            )
            for call in method.call_args_list
        )
        assert secret_query not in log_text
        assert secret_user_id not in log_text
        assert result == []
