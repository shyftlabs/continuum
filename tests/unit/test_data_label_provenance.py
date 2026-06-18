"""
Tests for data-label PROVENANCE tainting (Phase 0 + 1).

Background
----------
`RunContext.data_labels` is a taint set (e.g. {"pii", "phi"}) that rides along
with a run. Historically nothing *set* it automatically — labels were manual-only
and, in practice, always empty — so the one consumer that reads them (the tool
gate) never fired.

This adds the PRODUCER half, with no PII detector in the SDK. Instead the
integrator declares *provenance* — which sources carry which labels — and the
runtime taints the run when data crosses those boundaries. Three declaration
sites:

  1. Tool      — AgentConfig.tool_data_labels[tool_name] -> labels;
                 a tool's result taints the run.
  2. Memory    — AgentMemoryConfig.scope_data_labels[scope] -> labels;
                 reading from that scope taints the run ("read = taint").
  3. Run-level — create_run_context(data_labels=...) seeds the run at start.

Plus the Phase-0 plumbing the producer needs: RunContext.taint() and
RunContext.from_dict() (so labels survive serialize/resume).

No detector is shipped: tests declare provenance explicitly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.agent.utils.context_utils import create_run_context

# ---------------------------------------------------------------------------
# Phase 0 — core plumbing (pure; no runtime harness)
# ---------------------------------------------------------------------------


class TestRunContextTaint:
    def test_taint_adds_a_label(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint("pii")
        assert "pii" in ctx.data_labels

    def test_taint_multiple_and_is_set_semantics(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint("pii", "phi")
        ctx.taint("pii")  # duplicate is a no-op (set)
        assert ctx.data_labels == {"pii", "phi"}

    def test_taint_no_args_is_noop(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint()
        assert ctx.data_labels == set()


class TestRunContextFromDict:
    def test_round_trips_data_labels(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1", data_labels={"pii", "phi"})
        restored = RunContext.from_dict(ctx.to_dict())
        assert restored.data_labels == {"pii", "phi"}

    def test_restores_core_fields(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(
            run_id="r1",
            session_id="s1",
            user_id="u1",
            conversation_id="c1",
            priority=7,
            data_labels={"pii"},
        )
        restored = RunContext.from_dict(ctx.to_dict())
        assert restored.run_id == "r1"
        assert restored.session_id == "s1"
        assert restored.user_id == "u1"
        assert restored.conversation_id == "c1"
        assert restored.priority == 7

    def test_missing_data_labels_defaults_empty(self):
        from continuum.agent.types import RunContext

        restored = RunContext.from_dict({"run_id": "r1"})
        assert restored.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 3: run-level provenance (seed at start)
# ---------------------------------------------------------------------------


class TestRunLevelProvenance:
    def test_create_run_context_seeds_data_labels(self):
        ctx = create_run_context(data_labels={"pii"})
        assert ctx.data_labels == {"pii"}

    def test_create_run_context_defaults_empty(self):
        ctx = create_run_context()
        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 1: tool provenance (a declared tool's result taints the run)
# ---------------------------------------------------------------------------


def _tool_call(name: str):
    from continuum.llm.types import FunctionCall, ToolCall

    return ToolCall(id="tc-1", type="function", function=FunctionCall(name=name, arguments="{}"))


def _agent_with_tool_labels(tool_labels: dict[str, set[str]]):
    """A BaseAgent whose tool_executor returns a canned tool result."""
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig

    agent = BaseAgent(
        name="prov-agent",
        instructions="test",
        config=AgentConfig(tool_data_labels=tool_labels),
    )
    # Fake executor: registry hit + a successful tool-result message.
    executor = MagicMock()
    executor.tool_registry = {name: (MagicMock(name="server"), object()) for name in tool_labels}
    executor.tool_registry.setdefault("plain_tool", (MagicMock(name="server"), object()))
    executor.execute_tool_calls = AsyncMock(
        return_value=[{"role": "tool", "tool_call_id": "tc-1", "content": "done"}]
    )
    agent.tool_executor = executor
    agent.on_tool_call = None
    return agent


def _tool_service():
    from continuum.agent.services.tool_service import ToolService

    return ToolService(tool_executor=None)


class TestToolProvenance:
    def test_config_field_defaults_empty(self):
        from continuum.agent.config import AgentConfig

        assert AgentConfig().tool_data_labels == {}

    async def test_declared_tool_result_taints_run(self):
        agent = _agent_with_tool_labels({"fetch_record": {"phi"}})
        svc = _tool_service()
        ctx = create_run_context()

        await svc.execute_tool_call(agent, _tool_call("fetch_record"), ctx)

        assert "phi" in ctx.data_labels

    async def test_undeclared_tool_does_not_taint(self):
        agent = _agent_with_tool_labels({"fetch_record": {"phi"}})
        svc = _tool_service()
        ctx = create_run_context()

        await svc.execute_tool_call(agent, _tool_call("plain_tool"), ctx)

        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 2: memory-scope provenance (reading a labeled scope taints)
# ---------------------------------------------------------------------------


def _memory_result(n=1):
    res = MagicMock()
    items = []
    for i in range(n):
        m = MagicMock()
        m.to_dict.return_value = {"memory": f"fact-{i}"}
        m.metadata = {}
        m.user_id = "u1"
        m.score = 0.9
        m.memory = f"fact-{i}"
        items.append(m)
    res.results = items
    res.total_results = len(items)
    return res


def _memory_client(isolation="user", result=None):
    mc = MagicMock()
    mc.is_enabled = True
    mc.config = MagicMock()
    mc.config.memory_isolation = isolation
    mc.search = AsyncMock(return_value=result if result is not None else _memory_result())
    return mc


def _agent_with_scope_labels(scope_labels: dict[str, set[str]]):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="mem-prov-agent",
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(scope_data_labels=scope_labels),
    )


def _memory_service(mc):
    from continuum.agent.services.memory_service import MemoryService

    return MemoryService(memory_client=mc, session_client=None)


class TestMemoryScopeProvenance:
    def test_config_field_defaults_empty(self):
        from continuum.agent.config import AgentMemoryConfig

        assert AgentMemoryConfig().scope_data_labels == {}

    async def test_read_from_labeled_scope_taints_run(self):
        svc = _memory_service(_memory_client(isolation="user"))
        agent = _agent_with_scope_labels({"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert "pii" in ctx.data_labels

    async def test_no_results_does_not_taint(self):
        # No data actually flowed out of the scope → no taint.
        svc = _memory_service(_memory_client(isolation="user", result=_memory_result(0)))
        agent = _agent_with_scope_labels({"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()

    async def test_undeclared_scope_does_not_taint(self):
        svc = _memory_service(_memory_client(isolation="user"))
        agent = _agent_with_scope_labels({"agent": {"pii"}})  # labels a different scope
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()
