"""
Integration test for lifecycle-hook ordering across a real handoff.

Drives a full `runner.run()` with two REAL BaseAgents and the REAL executor /
handoff-executor wiring (only the LLM and peripheral services are mocked), to
lock down the end-to-end hook order:

    A.on_start  ->  B.on_start  ->  B.on_end  ->  A.on_end

This complements the focused unit tests in test_handoff_context.py
(TestHandoffRecipientLifecycleHooks), which assert the hooks fire at the
execute_handoff seam. Here we prove the *ordering* through the public entry
point with return_to_parent=True (the default), where A's LLM runs twice.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.types import Handoff, RunState

# ---------------------------------------------------------------------------
# Fake LLM plumbing
# ---------------------------------------------------------------------------


def _usage() -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10)


class _FakeFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _FakeFn(name, arguments)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _handoff_response(target: str) -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[_FakeToolCall("tc-1", f"handoff_to_{target}", '{"reason": "specialist"}')],
        usage=_usage(),
        model="model-a",
    )


def _final_response(text: str, model: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=None, usage=_usage(), model=model)


# ---------------------------------------------------------------------------
# Runner assembly: real executor + handoff path, mocked periphery
# ---------------------------------------------------------------------------


def _build_runner(agent_a: BaseAgent, agent_b: BaseAgent, fake_llm) -> object:
    from continuum.agent.execution.executor import Executor
    from continuum.agent.execution.handoff_executor import HandoffExecutor
    from continuum.agent.execution.tool_handler import ToolHandler
    from continuum.agent.handoff.manager import HandoffManager
    from continuum.agent.runner import AgentRunner, RunnerConfig
    from continuum.agent.services.tool_service import ToolService
    from continuum.agent.utils.circuit_breaker import CircuitBreaker

    runner = AgentRunner.__new__(AgentRunner)
    runner._llm_client = fake_llm
    runner._memory_client = None
    runner._session_client = None
    runner._tool_executor = None
    runner._tracing_manager = None
    runner._state_manager = None
    runner._config = RunnerConfig()
    runner._agent_registry = {agent_a.name: agent_a, agent_b.name: agent_b}
    runner._circuit_breaker = CircuitBreaker(threshold=5, cooldown=60)
    runner._artifact_lock = asyncio.Lock()

    # Mocked HandoffManager (deterministic; summarization/tracing are not under test)
    hm = MagicMock(spec=HandoffManager)
    hm._max_depth = 10
    hm.detect_cycle = MagicMock(return_value=False)
    hm.prepare_handoff = AsyncMock(
        return_value=MagicMock(
            handoff_id="h1", to_dict=MagicMock(return_value={"to_agent": agent_b.name})
        )
    )
    hm.build_handoff_messages = MagicMock(return_value=[{"role": "user", "content": "handle this"}])
    hm.trace_handoff = AsyncMock()
    runner._handoff_manager = hm

    # REAL execution components
    runner._tool_service = ToolService(tool_executor=None, config=runner._config)
    runner._tool_handler = ToolHandler(tool_service=runner._tool_service)
    runner._handoff_executor = HandoffExecutor(
        handoff_manager=hm, agent_registry=runner._agent_registry
    )
    runner._executor = Executor(
        llm_client=fake_llm,
        tool_handler=runner._tool_handler,
        handoff_executor=runner._handoff_executor,
    )
    runner._handoff_executor.set_executor(runner._executor)
    for a in runner._agent_registry.values():
        runner._handoff_executor.register_agent(a)

    # Mocked peripheral services
    def _make_run_state(agent, context):
        rs = RunState(run_id=context.run_id)
        rs.push_agent(agent.name)
        return rs

    runner._context_service = MagicMock()
    runner._context_service.create_run_state = AsyncMock(side_effect=_make_run_state)
    runner._context_service.save_run_state = AsyncMock()

    runner._lifecycle = MagicMock()
    runner._lifecycle.start_trace = AsyncMock()
    runner._lifecycle.report_metrics = AsyncMock()
    runner._lifecycle.report_error = AsyncMock()
    runner._lifecycle.end_trace = AsyncMock()

    runner._message_builder = MagicMock()
    runner._message_builder.prepare_messages = AsyncMock(
        return_value=([{"role": "user", "content": "hi"}], 0)
    )

    runner._session_service = MagicMock()
    runner._finalizer = MagicMock()
    runner._finalizer.finalize = AsyncMock()
    runner._finalizer.handle_error = AsyncMock()

    return runner


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def test_handoff_hook_ordering_end_to_end():
    events: list[str] = []

    def hook(label: str):
        return lambda *a, **k: events.append(label)

    agent_b = BaseAgent(
        name="agent-b",
        instructions="specialist",
        model="model-b",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
        on_start=hook("b_start"),
        on_end=hook("b_end"),
    )
    agent_a = BaseAgent(
        name="agent-a",
        instructions="triage",
        model="model-a",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
        handoffs=[Handoff(target_agent="agent-b", description="for specialist work")],
        on_start=hook("a_start"),
        on_end=hook("a_end"),
    )

    # model-a: first call hands off, second call (after return-to-parent) finalizes.
    # model-b: finalizes immediately.
    responses = {
        "model-a": [_handoff_response("agent-b"), _final_response("A final answer", "model-a")],
        "model-b": [_final_response("B answer", "model-b")],
    }

    async def fake_chat(*, messages, tools=None, config=None, **kwargs):
        return responses[config.model].pop(0)

    fake_llm = MagicMock()
    fake_llm.chat = fake_chat

    runner = _build_runner(agent_a, agent_b, fake_llm)

    with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
        result = await runner.run(agent_a, "please help")

    # The default return_to_parent=True means A produces the final user-facing answer.
    assert result.content == "A final answer"
    # The headline assertion: every agent's lifecycle boundary fired, in order.
    assert events == ["a_start", "b_start", "b_end", "a_end"]
