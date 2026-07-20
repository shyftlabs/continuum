"""
Unit tests for ReflectionAgent critique temperature resolution.

The critique LLM call must not hardcode a temperature. It inherits the inner
agent's configured temperature by default, accepts an explicit override via
ReflectionConfig.reflection_temperature, and omits the parameter entirely when
the inherited/overridden value is None (for providers that reject it).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.workflow.reflection import ReflectionAgent, ReflectionConfig


def _mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=SimpleNamespace(content="PASS", usage=None))
    return client


def _captured_temperature(client: AsyncMock):
    return client.chat.await_args.kwargs["config"].temperature


@pytest.mark.asyncio
class TestReflectionCritiqueTemperature:
    async def test_inherits_inner_agent_temperature(self):
        inner = BaseAgent(name="worker", temperature=0.42)
        ref = ReflectionAgent(name="reflector", agent=inner)

        client = _mock_llm()
        await ref._critique(response_content="output", llm_client=client)

        assert _captured_temperature(client) == 0.42

    async def test_explicit_override_wins(self):
        inner = BaseAgent(name="worker", temperature=0.42)
        ref = ReflectionAgent(
            name="reflector",
            agent=inner,
            reflection_config=ReflectionConfig(reflection_temperature=0.9),
        )

        client = _mock_llm()
        await ref._critique(response_content="output", llm_client=client)

        assert _captured_temperature(client) == 0.9

    async def test_none_inner_temperature_is_omitted(self):
        # Inner agent configured to omit temperature (unsupported provider) →
        # the critique inherits None, which LLMConfig drops from the request.
        inner = BaseAgent(name="worker", temperature=None)
        ref = ReflectionAgent(name="reflector", agent=inner)

        client = _mock_llm()
        await ref._critique(response_content="output", llm_client=client)

        cfg = client.chat.await_args.kwargs["config"]
        assert cfg.temperature is None
        assert "temperature" not in cfg.to_kwargs()
