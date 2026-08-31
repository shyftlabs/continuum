from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentMemoryConfig
from continuum.agent.exceptions import MemoryAccessDeniedError
from continuum.agent.services.memory_service import MemoryService
from continuum.agent.utils.context_utils import create_run_context
from continuum.memory.client import MemoryClient
from continuum.memory.config import MemoryConfig
from continuum.memory.types import MemorySearchResult
from continuum.security.policy import AccessPolicy, PolicyStore
from continuum.security.policy_context import use_active_policy


def _memory_client() -> MemoryClient:
    provider = MagicMock()
    provider.is_initialized = True
    provider.search = AsyncMock(return_value=MemorySearchResult(results=[], query="query", limit=5))
    return MemoryClient(
        config=MemoryConfig(enabled=True, memory_isolation="user"),
        provider=provider,
    )


def _deny_pii_reads() -> PolicyStore:
    store = PolicyStore()
    store.add_policy(
        AccessPolicy(
            name="deny_pii_memory_reads",
            subjects=["pii"],
            resources=["memory:*"],
            effect="deny",
        )
    )
    return store


class TestMemoryReadPolicy:
    async def test_explicit_labels_are_checked_before_search(self):
        client = _memory_client()

        with pytest.raises(MemoryAccessDeniedError):
            await client.search(
                "query",
                user_id="user-1",
                policy_store=_deny_pii_reads(),
                subject="assistant",
                data_labels={"pii"},
            )

        client.provider.search.assert_not_awaited()

    async def test_automatic_retrieval_uses_ambient_policy(self):
        client = _memory_client()
        service = MemoryService(memory_client=client)
        agent = BaseAgent(
            name="assistant",
            instructions="test",
            memory_config=AgentMemoryConfig(search_memories=True),
            policy_store=_deny_pii_reads(),
        )
        context = create_run_context(user_id="user-1", data_labels={"pii"})

        with use_active_policy(agent.policy_store, agent.name, context):
            memories = await service.retrieve_memories(agent, "query", context)

        assert memories == []
        client.provider.search.assert_not_awaited()

    async def test_automatic_retrieval_proceeds_when_allowed(self):
        client = _memory_client()
        service = MemoryService(memory_client=client)
        agent = BaseAgent(
            name="assistant",
            instructions="test",
            memory_config=AgentMemoryConfig(search_memories=True),
            policy_store=PolicyStore(),
        )
        context = create_run_context(user_id="user-1", data_labels={"pii"})

        with use_active_policy(agent.policy_store, agent.name, context):
            await service.retrieve_memories(agent, "query", context)

        client.provider.search.assert_awaited_once()
