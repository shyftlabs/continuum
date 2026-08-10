"""
Unit tests: workflow agents read their configurable temperature (no hardcoded
literal) and pass it through to the auxiliary LLM call — including None, which
LLMConfig must omit.

These are mocked (no API) so they run in CI. The live end-to-end equivalent
lives in playground/local/ (gitignored).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import ParallelConfig
from continuum.agent.types import AgentResponse, MergeStrategy, ResponseStatus
from continuum.agent.workflow import ParallelAgent


def _mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=SimpleNamespace(content="merged", usage=None))
    return client


def _results() -> dict[str, AgentResponse]:
    return {
        "a": AgentResponse(content="sun is hot", agent_name="a", status=ResponseStatus.SUCCESS),
        "b": AgentResponse(content="moon is cold", agent_name="b", status=ResponseStatus.SUCCESS),
    }


def _parallel(summary_temperature: float | None) -> ParallelAgent:
    return ParallelAgent(
        name="merge-test",
        agents=[BaseAgent(name="a"), BaseAgent(name="b")],
        parallel_config=ParallelConfig(
            merge_strategy=MergeStrategy.LLM_SUMMARIZE,
            summary_temperature=summary_temperature,
        ),
    )


def _captured_config(client: AsyncMock):
    return client.chat.await_args.kwargs["config"]


@pytest.mark.asyncio
class TestParallelMergeTemperature:
    async def test_merge_uses_configured_temperature(self):
        client = _mock_llm()
        agent = _parallel(summary_temperature=0.55)

        out = await agent._merge_results(_results(), "tell me about the sky", client)

        assert out == "merged"
        assert _captured_config(client).temperature == 0.55

    async def test_merge_temperature_none_is_omitted(self):
        client = _mock_llm()
        agent = _parallel(summary_temperature=None)

        await agent._merge_results(_results(), "tell me about the sky", client)

        cfg = _captured_config(client)
        assert cfg.temperature is None
        # The omission itself is the provider's job (each _build_kwargs skips
        # a None temperature); asserted there, not through a config-level dump.
