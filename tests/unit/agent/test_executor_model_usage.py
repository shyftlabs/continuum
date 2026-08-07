"""
Tests for per-resolved-model usage attribution (TokenUsage.model_usage).

The executor must key each LLM call's tokens on ``response.model`` — the id the
provider/gateway actually served — because under the Smart Gateway the agent's
configured model is the literal placeholder "auto". Downstream metering
(audiex MeteringRunner) prices per model_usage entry and falls back to the
configured model only when model_usage is empty, which yields zero-cost ledger
rows for "auto". These tests drive the REAL Executor with fake LLM responses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.execution.executor import Executor
from continuum.agent.types import (
    AgentResponse,
    Handoff,
    HandoffResult,
    ResponseStatus,
    RunContext,
    RunState,
    TokenUsage,
)


def _usage(prompt, completion):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFn(name, arguments)

    def to_dict(self):
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _final_response(model, prompt, completion, content="done"):
    return SimpleNamespace(
        content=content,
        tool_calls=None,
        usage=_usage(prompt, completion),
        model=model,
    )


def _think_response(model, prompt, completion):
    # A "think" tool call is handled inline by the executor (no ToolHandler
    # needed) and keeps the loop going — the cleanest way to force a 2nd turn.
    return SimpleNamespace(
        content="",
        tool_calls=[_FakeToolCall("tc-think", "think", '{"thought": "hmm"}')],
        usage=_usage(prompt, completion),
        model=model,
    )


def _agent(**kwargs):
    kwargs.setdefault("name", "meter-agent")
    kwargs.setdefault("instructions", "answer")
    kwargs.setdefault("config", AgentConfig())
    kwargs.setdefault("model", "configured-model")
    return BaseAgent(**kwargs)


def _ctx(max_turns=6):
    return RunContext(run_id="run-meter", max_turns=max_turns)


def _run_state():
    rs = RunState(run_id="run-meter")
    rs.push_agent("meter-agent")
    return rs


class TestPerModelAttribution:
    async def test_two_turn_run_attributes_per_resolved_model(self):
        # Turn 1 resolves to one model, turn 2 to another (gateway tier routing
        # can do exactly this) → two model_usage entries with the right sums.
        llm = SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    _think_response("openai/gpt-4o-mini", 10, 5),
                    _final_response("anthropic/claude-sonnet-4-5", 20, 7),
                ]
            )
        )
        ex = Executor(llm_client=llm)

        response = await ex.execute_loop(
            _agent(), [{"role": "user", "content": "q"}], _ctx(), _run_state()
        )

        assert response.usage.model_usage == {
            "openai/gpt-4o-mini": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "anthropic/claude-sonnet-4-5": {
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "total_tokens": 27,
            },
        }
        # Top-level totals still equal the sum of the per-model breakdown.
        assert response.usage.prompt_tokens == 30
        assert response.usage.completion_tokens == 12
        assert response.usage.total_tokens == 42

    async def test_same_resolved_model_sums_into_one_entry(self):
        llm = SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    _think_response("openai/gpt-4o", 10, 5),
                    _final_response("openai/gpt-4o", 20, 7),
                ]
            )
        )
        ex = Executor(llm_client=llm)

        response = await ex.execute_loop(
            _agent(), [{"role": "user", "content": "q"}], _ctx(), _run_state()
        )

        assert response.usage.model_usage == {
            "openai/gpt-4o": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
            }
        }
        assert response.usage.total_tokens == 42

    async def test_missing_response_model_falls_back_to_configured_model(self):
        # Provider omitted `model` → attribute to the agent's configured model
        # rather than dropping the tokens from the per-model map.
        llm = SimpleNamespace(chat=AsyncMock(return_value=_final_response("", 8, 4)))
        ex = Executor(llm_client=llm)

        response = await ex.execute_loop(
            _agent(model="configured-model"),
            [{"role": "user", "content": "q"}],
            _ctx(),
            _run_state(),
        )

        assert response.usage.model_usage == {
            "configured-model": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            }
        }

    async def test_reasoning_pass_usage_is_attributed(self):
        # reasoning_mode adds a silent think-first call before the turn loop;
        # its tokens must land in model_usage too (keyed on ITS resolved model).
        llm = SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    _final_response("openai/gpt-4o-mini", 3, 2, content="thinking..."),
                    _final_response("anthropic/claude-sonnet-4-5", 10, 5),
                ]
            )
        )
        ex = Executor(llm_client=llm)

        response = await ex.execute_loop(
            _agent(config=AgentConfig(reasoning_mode=True)),
            [{"role": "user", "content": "q"}],
            _ctx(),
            _run_state(),
        )

        assert response.usage.model_usage == {
            "openai/gpt-4o-mini": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            "anthropic/claude-sonnet-4-5": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        assert response.usage.total_tokens == 20

    async def test_react_loop_attributes_resolved_model(self):
        llm = SimpleNamespace(
            chat=AsyncMock(
                return_value=_final_response(
                    "openai/gpt-4o-mini", 4, 2, content="Action: Final Answer\nFinal Answer: ok"
                )
            )
        )
        ex = Executor(llm_client=llm)

        response = await ex._execute_react_loop(
            _agent(), [{"role": "user", "content": "q"}], _ctx(), _run_state()
        )

        assert response.content == "ok"
        assert response.usage.model_usage == {
            "openai/gpt-4o-mini": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            }
        }


class _StubHandoff:
    """Mimics HandoffExecutor: pushes target, records the hop, returns a child
    response whose usage carries its OWN per-model breakdown (as a real child
    executor loop now produces)."""

    def __init__(self, child_usage: TokenUsage):
        self._executor = None
        self._child_usage = child_usage

    async def execute_handoff(self, agent, target_name, tool_call, messages, context, run_state):
        run_state.push_agent(target_name)
        run_state.handoff_chain.append({"to_agent": target_name})
        return HandoffResult(
            handoff_id="h",
            from_agent=agent.name,
            to_agent=target_name,
            success=True,
            response=AgentResponse(
                content="specialist answer",
                agent_name=target_name,
                status=ResponseStatus.SUCCESS,
                usage=self._child_usage,
            ),
        )


class TestHandoffMerge:
    async def test_child_model_usage_merges_into_parent_totals(self):
        # return_to_parent handoff: the child's per-model map must merge into
        # the parent run's usage via TokenUsage.add() — no direct response.model
        # exists at the handoff accumulation site (aggregated multi-turn usage).
        child_usage = TokenUsage(
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            model_usage={
                "google/gemini-2.5-flash": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                }
            },
        )
        handoff_call = SimpleNamespace(
            content="",
            tool_calls=[_FakeToolCall("tc-h", "handoff_to_billing", "{}")],
            usage=_usage(10, 5),
            model="openai/gpt-4o",
        )
        llm = SimpleNamespace(
            chat=AsyncMock(side_effect=[handoff_call, _final_response("openai/gpt-4o", 20, 10)])
        )
        agent = _agent(
            name="triage",
            handoffs=[Handoff(target_agent="billing", description="billing q")],
        )
        ex = Executor(llm_client=llm, handoff_executor=_StubHandoff(child_usage))
        rs = RunState(run_id="run-meter")
        rs.push_agent("triage")

        response = await ex.execute_loop(agent, [{"role": "user", "content": "refund"}], _ctx(), rs)

        assert response.usage.model_usage == {
            "openai/gpt-4o": {
                "prompt_tokens": 30,
                "completion_tokens": 15,
                "total_tokens": 45,
            },
            "google/gemini-2.5-flash": {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
            },
        }
        assert response.usage.prompt_tokens == 35
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 55
