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


@pytest.fixture(autouse=True)
def _fresh_warnings():
    executor._GATEWAY_WARNED.clear()
    yield
    executor._GATEWAY_WARNED.clear()


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
    for i, bad in enumerate((["org-7"], "org-7", True, 7)):
        ctx = RunContext(run_id=f"r{i}", metadata={"gateway_metadata": bad})
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


def test_labels_json_cannot_encode_are_sent_as_text(gateway):
    """Mixed or non-string keys, tuple keys and self-reference must not fail the run."""
    looped: dict = {"name": "F-12"}
    looped["self"] = looped
    labels = {
        "fund": {2024: "q1", "name": "F-12"},
        "pair": {("a", "b"): 1},
        "loop": looped,
    }
    ctx = RunContext(run_id="r", metadata={"gateway_metadata": labels})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta["fund"] == str({2024: "q1", "name": "F-12"})
    assert meta["pair"] == str({("a", "b"): 1})
    assert meta["loop"].startswith("{'name': 'F-12'")


def test_agent_metadata_counts_towards_the_cap_and_is_sent_as_strings(gateway):
    own = {f"own{i:02d}": i for i in range(20)}
    own["nested"] = {"a": 1}
    cfg = LLMConfig(model="auto", extra_body={"metadata": own})
    ctx = RunContext(
        run_id="r",
        session_id="s",
        metadata={"gateway_metadata": {"own00": "caller", "session_id": "spoof"}},
    )
    meta = _enrich_config_for_gateway(cfg, ctx).extra_body["metadata"]
    assert len(meta) == 16
    assert all(isinstance(v, str) for k, v in meta.items() if k != "trace_id")
    assert meta["own00"] == "caller"  # a caller label wins over the agent's
    assert meta["own01"] == "1"
    assert meta["session_id"] == "s"  # the ids win over both


def test_keys_cut_to_the_same_64_characters_are_logged(gateway):
    prefix = "k" * 64
    ctx = RunContext(
        run_id="r", metadata={"gateway_metadata": {prefix + "-a": "1", prefix + "-b": "2"}}
    )
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta[prefix] == "2"
    assert any("same 64 characters" in c.args[0] for c in log.warning.call_args_list)


def test_label_warnings_are_logged_once_per_run(gateway):
    labels = {f"k{i:02d}": "v" for i in range(20)}
    with patch.object(executor, "logger") as log:
        for _ in range(3):
            ctx = RunContext(run_id="r", metadata={"gateway_metadata": labels})
            _enrich_config_for_gateway(LLMConfig(model="auto"), ctx)
        assert log.warning.call_count == 1
        ctx = RunContext(run_id="r2", metadata={"gateway_metadata": labels})
        _enrich_config_for_gateway(LLMConfig(model="auto"), ctx)
        assert log.warning.call_count == 2


@pytest.mark.parametrize("missing", ["none", "absent"])
async def test_handoff_works_when_the_context_has_no_tags_or_metadata(missing):
    """A caller-built context may carry ``tags=None`` or no ``tags``/``metadata`` at all."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from continuum.agent.base import BaseAgent
    from continuum.agent.execution.handoff_executor import HandoffExecutor
    from continuum.agent.handoff.manager import HandoffManager
    from continuum.agent.types import Handoff, RunState

    inner = MagicMock()
    inner.execute_loop = AsyncMock(
        return_value=SimpleNamespace(
            content="done",
            structured_output=None,
            messages=[],
            turn_count=1,
            usage=SimpleNamespace(total_tokens=0),
            agents_used=[],
            status="success",
        )
    )
    handoffs = HandoffExecutor(
        handoff_manager=HandoffManager(llm_client=None), agent_registry={}, executor=inner
    )
    source = BaseAgent(
        name="a", instructions="a", handoffs=[Handoff(target_agent="b", description="")]
    )
    handoffs.register_agent(source)
    handoffs.register_agent(BaseAgent(name="b", instructions="b"))
    ctx = SimpleNamespace(
        run_id="r",
        session_id=None,
        trace_id=None,
        user_id=None,
        conversation_id=None,
        agent_stack=[],
        recorder=None,
        max_turns=25,
        data_labels=set(),
        disable_memory_writes=False,
    )
    if missing == "none":
        ctx.tags = None
        ctx.metadata = None
    call = SimpleNamespace(
        function=SimpleNamespace(name="handoff_to_b", arguments='{"reason": "x"}'), id="t1"
    )
    result = await handoffs.execute_handoff(
        agent=source,
        target_name="b",
        tool_call=call,
        messages=[],
        context=ctx,
        run_state=RunState(run_id="r"),
    )
    assert result.success is True
    target_context = inner.execute_loop.call_args.kwargs.get("context") or next(
        a for a in inner.execute_loop.call_args.args if hasattr(a, "is_handoff")
    )
    assert target_context.tags == [] and target_context.metadata == {}


def test_caller_labels_take_the_slots_before_the_agents_metadata(gateway):
    """14 static agent keys plus two per-run labels: the per-run labels are sent."""
    own = {f"own{i:02d}": "x" for i in range(14)}
    cfg = LLMConfig(model="auto", extra_body={"metadata": own})
    ctx = RunContext(
        run_id="r",
        session_id="s",
        metadata={"gateway_metadata": {"org_id": "org-7", "fund_id": "F-12"}},
    )
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(cfg, ctx).extra_body["metadata"]
    assert len(meta) == 16
    assert meta["org_id"] == "org-7" and meta["fund_id"] == "F-12"
    assert sum(k.startswith("own") for k in meta) == 12  # 16 - ids (2) - caller labels (2)
    assert any(
        "Dropped 2 gateway label" in (c.args[0] % c.args[1:]) for c in log.warning.call_args_list
    )

    # An agent key cut to the same 64 characters as a caller key does not replace it.
    prefix = "k" * 64
    cfg = LLMConfig(model="auto", extra_body={"metadata": {prefix + "-agent": "agent"}})
    ctx = RunContext(run_id="r", metadata={"gateway_metadata": {prefix + "-caller": "caller"}})
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(cfg, ctx).extra_body["metadata"]
    assert meta[prefix] == "caller"
    assert any("same 64 characters" in c.args[0] for c in log.warning.call_args_list)


def test_tags_are_cut_at_a_tag_boundary_and_the_drop_is_logged(gateway):
    tags = [f"tag-{i:04d}" for i in range(100)]  # 8 characters each
    ctx = RunContext(run_id="r", tags=tags)
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    sent = meta["tags"].split(",")
    assert len(meta["tags"]) <= 512
    assert sent == tags[: len(sent)]  # only whole tags, in order
    assert len(sent) == 57  # 57 tags of 8 characters and 56 commas: exactly 512
    messages = [c.args[0] % c.args[1:] for c in log.warning.call_args_list]
    assert any(f"Dropped {100 - len(sent)} gateway tag" in m for m in messages)


def test_tag_containing_a_comma_is_skipped_and_logged(gateway):
    ctx = RunContext(run_id="r", tags=["ask", "a,b", "fund"])
    with patch.object(executor, "logger") as log:
        meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta["tags"] == "ask,fund"
    assert any("containing a comma" in c.args[0] for c in log.warning.call_args_list)
