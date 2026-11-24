"""Model calls through the Smart Gateway carry the caller's labels.

The gateway attributes cost and applies budgets from body.metadata. Continuum
sent only session_id and trace_id, so a product could not ask the gateway for
cost by tenant, fund or feature; and building that metadata replaced the
agent's own extra_body outright.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from continuum.agent.execution import executor
from continuum.agent.execution.executor import _enrich_config_for_gateway
from continuum.agent.types import RunContext
from continuum.llm.config import LLMConfig


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(executor.settings, "smart_gateway_url", "https://gateway.test/v1")


def test_caller_labels_and_tags_are_sent(gateway):
    ctx = RunContext(
        run_id="run-1",
        session_id="s-1",
        trace_id="t-1",
        tags=["ask-arnie"],
        metadata={"gateway_metadata": {"org_id": "org-7", "fund_id": "F-12", "skip": None}},
    )
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta == {
        "org_id": "org-7",
        "fund_id": "F-12",
        "tags": "ask-arnie",
        "session_id": "s-1",
        "trace_id": "t-1",
    }


def test_ids_cannot_be_overridden_by_labels(gateway):
    ctx = RunContext(run_id="run-1", metadata={"gateway_metadata": {"session_id": "spoof"}})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta["session_id"] == "run-1"


def test_agent_extra_body_is_kept(gateway):
    cfg = LLMConfig(model="auto", extra_body={"top_k": 5, "metadata": {"team": "research"}})
    out = _enrich_config_for_gateway(cfg, RunContext(run_id="r", session_id="s"))
    assert out.extra_body["top_k"] == 5
    assert out.extra_body["metadata"]["team"] == "research"
    assert out.extra_body["metadata"]["session_id"] == "s"
    assert cfg.extra_body == {"top_k": 5, "metadata": {"team": "research"}}  # not mutated


def test_internal_run_metadata_is_not_sent(gateway):
    ctx = RunContext(run_id="r", metadata={"_filtered_tools": ["x"], "promoted_tools": {"y"}})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert set(meta) == {"session_id", "trace_id"}


def test_unchanged_without_a_gateway(monkeypatch):
    monkeypatch.setattr(executor.settings, "smart_gateway_url", None)
    cfg = LLMConfig(model="gpt-4o-mini")
    assert _enrich_config_for_gateway(cfg, RunContext(run_id="r")) is cfg


def test_label_that_is_not_a_mapping_is_ignored(gateway):
    for bad in (["org-7"], "org-7", True, 7):
        ctx = RunContext(run_id="r", metadata={"gateway_metadata": bad})
        with patch.object(executor, "logger") as log:
            meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
        assert set(meta) == {"session_id", "trace_id"}
        assert log.warning.called


def test_labels_are_capped_and_nested_values_are_json(gateway):
    labels = {f"k{i:02d}": "v" * 600 for i in range(20)}
    labels["org"] = {"id": 7}
    labels["x" * 100] = "long key"
    ctx = RunContext(run_id="r", session_id="s", tags=["a"], metadata={"gateway_metadata": labels})
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert len(meta) == 16
    assert meta["session_id"] == "s" and meta["tags"] == "a" and "trace_id" in meta
    assert all(len(k) <= 64 and len(v or "") <= 512 for k, v in meta.items())
    assert log.warning.called  # labels beyond the 16-key limit were dropped

    nested = RunContext(run_id="r", metadata={"gateway_metadata": {"org": {"id": 7}}})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), nested).extra_body["metadata"]
    assert meta["org"] == '{"id":7}'


async def test_handoff_target_keeps_labels_tags_and_its_own_tools(gateway):
    """The target's calls carry the labels and tags, but not the source's tools.

    Only the gateway entry is copied: ``metadata`` also holds per-agent state
    such as ``_filtered_tools``, which would hand the source's tools to the target.
    """
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig
    from continuum.agent.types import Handoff
    from tests.unit.test_handoff_lifecycle_integration import (
        _build_runner,
        _final_response,
        _handoff_response,
    )

    def tool(name: str) -> dict:
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    agent_b = BaseAgent(
        name="agent-b",
        instructions="specialist",
        model="model-b",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
        tools=[tool("tool_b")],
    )
    agent_a = BaseAgent(
        name="agent-a",
        instructions="triage",
        model="model-a",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
        tools=[tool("tool_a")],
        handoffs=[Handoff(target_agent="agent-b", description="specialist work")],
    )
    responses = {
        "model-a": [_handoff_response("agent-b"), _final_response("A", "model-a")],
        "model-b": [_final_response("B", "model-b")],
    }
    calls: list[tuple[str, dict, list[str]]] = []

    async def fake_chat(*, messages, tools=None, config=None, **kwargs):
        names = [t["function"]["name"] for t in tools or []]
        calls.append((config.model, config.extra_body["metadata"], names))
        return responses[config.model].pop(0)

    fake_llm = MagicMock()
    fake_llm.chat = fake_chat
    runner = _build_runner(agent_a, agent_b, fake_llm)

    with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
        await runner.run(
            agent_a,
            "please help",
            tags=["ask"],
            metadata={
                "gateway_metadata": {"fund_id": "F-12"},
                # What message_builder's tool attention leaves for the source agent.
                "_filtered_tools": agent_a.get_tools_for_llm(),
            },
        )

    b_calls = [c for c in calls if c[0] == "model-b"]
    assert b_calls
    for _, meta, names in b_calls:
        assert meta["fund_id"] == "F-12"
        assert meta["tags"] == "ask"
        assert "tool_b" in names and "tool_a" not in names
