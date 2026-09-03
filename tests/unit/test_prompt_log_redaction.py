"""Regression tests for structural-only agent request logging."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from continuum.agent.base import BaseAgent
from continuum.agent.config import (
    AgentConfig,
    AgentMemoryConfig,
    ParallelConfig,
    PlanningConfig,
    RunnerConfig,
)
from continuum.agent.exceptions import AgentToolError, ParallelWorkflowError, PlannerWorkflowError
from continuum.agent.execution.executor import Executor
from continuum.agent.execution.handoff_executor import HandoffExecutor
from continuum.agent.execution.message_builder import MessageBuilder
from continuum.agent.handoff.manager import HandoffManager
from continuum.agent.services.tool_service import ToolService
from continuum.agent.types import (
    AgentResponse,
    Handoff,
    HandoffResult,
    MergeStrategy,
    ResponseStatus,
    RunState,
    TokenUsage,
)
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.parallel import ParallelAgent
from continuum.agent.workflow.planner import PlannerAgent
from continuum.agent.workflow.scatter import ScatterAgent, ScatterConfig
from continuum.llm.types import FunctionCall, ToolCall

pytestmark = pytest.mark.asyncio


def _logged_text(mock_logger: MagicMock) -> str:
    """Render stdlib-style logger calls captured by a mock."""
    rendered: list[str] = []
    for level in ("debug", "info", "warning", "error", "critical"):
        for call in getattr(mock_logger, level).call_args_list:
            template, *args = call.args
            rendered.append(template % tuple(args) if args else str(template))
            if call.kwargs:
                rendered.append(repr(call.kwargs))
    return "\n".join(rendered)


def _tool_definition(schema_secret: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup_record",
            "description": "Look up a record",
            "parameters": {
                "type": "object",
                "properties": {schema_secret: {"type": "string"}},
            },
        },
    }


async def test_message_builder_ignores_full_prompt_logging_and_logs_only_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "system": "SYSTEM_PROMPT_SECRET",
        "user": "USER_CONTENT_SECRET",
        "history": "HISTORY_CONTENT_SECRET",
        "memory": "RETRIEVED_KNOWLEDGE_SECRET",
        "rag": "RAG_KNOWLEDGE_SECRET",
        "schema": "TOOL_PARAMETER_SCHEMA_SECRET",
        "session": "123e4567-e89b-12d3-a456-426614174000",
    }
    memory_service = SimpleNamespace(
        retrieve_memories=AsyncMock(return_value=[{"memory": secrets["memory"]}])
    )
    session_service = SimpleNamespace(
        get_conversation_history=AsyncMock(
            return_value=[{"role": "assistant", "content": secrets["history"]}]
        )
    )
    builder = MessageBuilder(memory_service=memory_service, session_service=session_service)
    agent = BaseAgent(
        name="builder-audit",
        instructions=secrets["system"],
        tools=[_tool_definition(secrets["schema"])],
        config=AgentConfig(
            input_sanitization=False,
            injection_detection=False,
            rag_context=secrets["rag"],
        ),
        memory_config=AgentMemoryConfig(search_memories=True),
    )
    context = create_run_context(session_id=secrets["session"])
    mock_logger = MagicMock()
    monkeypatch.setenv("LOG_FULL_PROMPT", "true")

    with patch("continuum.agent.execution.message_builder.logger", mock_logger):
        messages, _ = await builder.prepare_messages(agent, secrets["user"], context)

    message_contents = "\n".join(str(message.get("content", "")) for message in messages)
    assert all(
        secrets[key] in message_contents for key in ("system", "user", "history", "memory", "rag")
    )
    assert any(
        secrets["schema"] in tool.get("function", {}).get("parameters", {}).get("properties", {})
        for tool in context.metadata["_filtered_tools"]
    )
    logs = _logged_text(mock_logger)
    assert "agent=builder-audit" in logs
    assert "message_count=" in logs
    assert "role_sequence=" in logs
    assert "lookup_record" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_handoff_logs_messages_tools_and_failures_without_sensitive_values() -> None:
    secrets = {
        "prompt": "HANDOFF_PROMPT_SECRET",
        "user": "HANDOFF_USER_SECRET",
        "result": "HANDOFF_TOOL_RESULT_SECRET",
        "arguments": "HANDOFF_ARGUMENT_SECRET",
        "schema": "HANDOFF_SCHEMA_SECRET",
        "error": "HANDOFF_EXCEPTION_SECRET",
        "session": "de305d54-75b4-431b-adb2-eb6b9e546014",
    }
    source = BaseAgent(name="source-agent")
    target = BaseAgent(
        name="target-agent",
        instructions=secrets["prompt"],
        tools=[_tool_definition(secrets["schema"])],
    )
    manager = MagicMock(spec=HandoffManager)
    manager._max_depth = 10
    manager.detect_cycle.return_value = False
    handoff_data = MagicMock()
    handoff_data.handoff_id = "handoff-id-secret"
    handoff_data.to_dict.return_value = {"to_agent": target.name}
    manager.prepare_handoff = AsyncMock(return_value=handoff_data)
    manager.build_handoff_messages.return_value = [
        {"role": "system", "content": secrets["prompt"]},
        {"role": "user", "content": secrets["user"]},
        {"role": "tool", "content": secrets["result"]},
    ]
    manager.trace_handoff = AsyncMock()
    inner_executor = SimpleNamespace(
        execute_loop=AsyncMock(side_effect=RuntimeError(secrets["error"]))
    )
    executor = HandoffExecutor(
        handoff_manager=manager,
        agent_registry={target.name: target},
        executor=inner_executor,
    )
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            arguments=('{"reason": "HANDOFF_ARGUMENT_SECRET", "context": "HANDOFF_USER_SECRET"}')
        )
    )
    run_state = RunState(run_id="run-id-secret")
    run_state.push_agent(source.name)
    context = create_run_context(session_id=secrets["session"])
    mock_logger = MagicMock()

    with patch("continuum.agent.execution.handoff_executor.logger", mock_logger):
        result = await executor.execute_handoff(
            source,
            target.name,
            tool_call,
            [],
            context,
            run_state,
        )

    assert result.success is False
    assert manager.prepare_handoff.await_args.kwargs["reason"] == secrets["arguments"]
    assert manager.prepare_handoff.await_args.kwargs["context"] == secrets["user"]
    handed_off_messages = inner_executor.execute_loop.await_args.kwargs["messages"]
    assert all(
        secret in "\n".join(message["content"] for message in handed_off_messages)
        for secret in (secrets["prompt"], secrets["user"], secrets["result"])
    )
    logs = _logged_text(mock_logger)
    assert "agent=target-agent" in logs
    assert "role_sequence=['system', 'user', 'tool']" in logs
    assert "tool_names=['lookup_record']" in logs
    assert "error_type=RuntimeError" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_return_to_parent_logs_only_message_and_tool_structure() -> None:
    secrets = {
        "user": "RETURN_PARENT_USER_SECRET",
        "result": "RETURN_PARENT_RESULT_SECRET",
        "final": "RETURN_PARENT_FINAL_SECRET",
        "arguments": "RETURN_PARENT_ARGUMENT_SECRET",
        "call_id": "123e4567-e89b-12d3-a456-426614174066",
    }
    parent = BaseAgent(
        name="parent-agent",
        handoffs=[
            Handoff(
                target_agent="child-agent",
                description="delegate",
                return_to_parent=True,
            )
        ],
    )
    tool_call = ToolCall(
        id=secrets["call_id"],
        function=FunctionCall(
            name="handoff_to_child-agent",
            arguments=f'{{"reason": "{secrets["arguments"]}"}}',
        ),
    )
    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    content="",
                    tool_calls=[tool_call],
                    usage=TokenUsage(),
                    model=None,
                ),
                SimpleNamespace(
                    content=secrets["final"],
                    tool_calls=None,
                    usage=TokenUsage(),
                    model=None,
                ),
            ]
        )
    )

    class StubHandoffExecutor:
        async def execute_handoff(
            self, *, agent, target_name, tool_call, messages, context, run_state
        ):
            run_state.push_agent(target_name)
            return HandoffResult(
                handoff_id="handoff-id",
                from_agent=agent.name,
                to_agent=target_name,
                success=True,
                response=AgentResponse(
                    content=secrets["result"],
                    agent_name=target_name,
                    status=ResponseStatus.SUCCESS,
                ),
            )

    executor = Executor(llm_client=llm, handoff_executor=StubHandoffExecutor())
    context = create_run_context()
    run_state = RunState(run_id=context.run_id)
    run_state.push_agent(parent.name)
    mock_logger = MagicMock()

    with patch("continuum.agent.execution.executor.logger", mock_logger):
        response = await executor.execute_loop(
            parent,
            [{"role": "user", "content": secrets["user"]}],
            context,
            run_state,
        )

    return_messages = llm.chat.await_args_list[1].kwargs["messages"]
    assert secrets["user"] in repr(return_messages)
    assert secrets["result"] in repr(return_messages)
    assert secrets["arguments"] in repr(return_messages)
    assert response.content == secrets["final"]
    logs = _logged_text(mock_logger)
    assert "agent=parent-agent" in logs
    assert "target_agent=child-agent" in logs
    assert "message_count=3" in logs
    assert "role_sequence=['user', 'assistant', 'tool']" in logs
    assert "tool_names=['handoff_to_child-agent']" in logs
    assert "result_type=str" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_executor_react_and_structured_output_logs_exclude_results_and_errors() -> None:
    secrets = {
        "arguments": "REACT_ARGUMENT_SECRET",
        "result": "REACT_RESULT_SECRET",
        "final": "REACT_FINAL_SECRET",
        "error": "EXECUTOR_FORMAT_EXCEPTION_SECRET",
    }
    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    content=(
                        "Thought: use the tool\n"
                        "Action: lookup_record\n"
                        f'Action Input: {{"query": "{secrets["arguments"]}"}}'
                    ),
                    usage=None,
                ),
                SimpleNamespace(
                    content=(f"Action: Final Answer\nFinal Answer: {secrets['final']}"),
                    usage=None,
                ),
            ]
        )
    )
    executor = Executor(llm_client=llm)
    execute_react_tool = AsyncMock(return_value=secrets["result"])
    executor._execute_react_tool = execute_react_tool
    agent = BaseAgent(name="react-agent")
    context = create_run_context()
    run_state = RunState(run_id=context.run_id)
    mock_logger = MagicMock()

    with patch("continuum.agent.execution.executor.logger", mock_logger):
        response = await executor._execute_react_loop(
            agent,
            [{"role": "user", "content": "react request"}],
            context,
            run_state,
        )

        failing_handler = SimpleNamespace(
            execute_tools_batch=AsyncMock(side_effect=RuntimeError(secrets["error"]))
        )
        tool_error = await Executor(
            llm_client=llm, tool_handler=failing_handler
        )._execute_react_tool(
            agent,
            "lookup_record",
            {"query": secrets["arguments"]},
            context,
        )

        class OutputSchema(BaseModel):
            answer: str

        structured_executor = Executor(llm_client=llm)
        structured_format_call = AsyncMock(side_effect=RuntimeError(secrets["error"]))
        structured_executor._structured_format_call = structured_format_call
        structured, structured_error = await structured_executor._resolve_structured_output(
            BaseAgent(name="structured-agent", output_schema=OutputSchema),
            [{"role": "user", "content": secrets["arguments"]}],
            "not valid json",
            context,
        )

    assert execute_react_tool.await_args.kwargs["tool_args"]["query"] == secrets["arguments"]
    assert response.content == secrets["final"]
    assert secrets["result"] in repr(response.messages)
    assert secrets["error"] in tool_error
    assert structured is None
    assert secrets["error"] in (structured_error or "")
    logs = _logged_text(mock_logger)
    assert "agent=react-agent" in logs
    assert "tool=lookup_record" in logs
    assert "result_type=str" in logs
    assert "agent=structured-agent" in logs
    assert "error_type=RuntimeError" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_parallel_merge_and_branch_logs_exclude_prompt_results_and_errors() -> None:
    secrets = {
        "input": "PARALLEL_USER_SECRET",
        "output": "PARALLEL_RESULT_SECRET",
        "summary": "PARALLEL_SUMMARY_PROMPT_SECRET",
        "error": "PARALLEL_EXCEPTION_SECRET",
    }
    workflow = ParallelAgent(
        name="parallel-audit",
        agents=[BaseAgent(name="branch-a"), BaseAgent(name="branch-b")],
        parallel_config=ParallelConfig(
            merge_strategy=MergeStrategy.LLM_SUMMARIZE,
            summary_prompt=secrets["summary"],
        ),
    )
    results = {
        "branch-a": AgentResponse(
            content=secrets["output"],
            agent_name="branch-a",
            status=ResponseStatus.SUCCESS,
        )
    }
    failing_llm = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError(secrets["error"])))
    failing_runner = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError(secrets["error"])))
    mock_logger = MagicMock()

    with patch("continuum.agent.workflow.parallel.logger", mock_logger):
        await workflow._merge_results(results, secrets["input"], failing_llm)
        with pytest.raises(RuntimeError):
            await workflow._run_agent_safe(
                workflow.agents[0],
                secrets["input"],
                failing_runner,
                create_run_context(),
            )

    merge_prompt = failing_llm.chat.await_args.kwargs["messages"][0]["content"]
    assert all(secrets[key] in merge_prompt for key in ("output", "summary"))
    logs = _logged_text(mock_logger)
    assert "agent=parallel-audit" in logs
    assert "role_sequence=['user']" in logs
    assert "error_type=RuntimeError" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_scatter_split_merge_and_branch_logs_exclude_sensitive_values() -> None:
    secrets = {
        "input": "SCATTER_USER_SECRET",
        "slice_a": "SCATTER_SLICE_A_SECRET",
        "slice_b": "SCATTER_SLICE_B_SECRET",
        "output": "SCATTER_RESULT_SECRET",
        "summary": "SCATTER_SUMMARY_PROMPT_SECRET",
        "error": "SCATTER_EXCEPTION_SECRET",
    }
    workflow = ScatterAgent(
        name="scatter-audit",
        agents=[BaseAgent(name="branch-a"), BaseAgent(name="branch-b")],
        scatter_config=ScatterConfig(
            merge_strategy=MergeStrategy.LLM_SUMMARIZE,
            summary_prompt=secrets["summary"],
        ),
    )
    split_llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=SimpleNamespace(
                content=f'["{secrets["slice_a"]}", "{secrets["slice_b"]}"]'
            )
        )
    )
    results = {
        "branch-a": AgentResponse(
            content=secrets["output"],
            agent_name="branch-a",
            status=ResponseStatus.SUCCESS,
        )
    }
    failing_llm = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError(secrets["error"])))
    failing_runner = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError(secrets["error"])))
    mock_logger = MagicMock()

    with patch("continuum.agent.workflow.scatter.logger", mock_logger):
        slices = await workflow._llm_split(secrets["input"], split_llm)
        await workflow._merge_results(results, secrets["input"], failing_llm)
        with pytest.raises(RuntimeError):
            await workflow._run_agent_safe(
                workflow.agents[0],
                secrets["input"],
                failing_runner,
                create_run_context(),
            )

    assert slices == [secrets["slice_a"], secrets["slice_b"]]
    split_prompt = split_llm.chat.await_args.kwargs["messages"][0]["content"]
    merge_prompt = failing_llm.chat.await_args.kwargs["messages"][0]["content"]
    assert secrets["input"] in split_prompt
    assert all(secrets[key] in merge_prompt for key in ("output", "summary"))
    logs = _logged_text(mock_logger)
    assert "agent=scatter-audit" in logs
    assert "agents=['branch-a', 'branch-b']" in logs
    assert "role_sequence=['user']" in logs
    assert "error_type=RuntimeError" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_tool_service_logs_structure_without_arguments_results_ids_or_errors() -> None:
    secrets = {
        "arguments": "TOOL_ARGUMENT_SECRET",
        "result": "TOOL_RESULT_SECRET",
        "malformed": "MALFORMED_ARGUMENT_SECRET",
        "error": "TOOL_EXCEPTION_SECRET",
        "call_id": "123e4567-e89b-12d3-a456-426614174099",
    }
    executor = MagicMock()
    executor.tool_registry = {}
    successful_execute = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_call_id": secrets["call_id"],
                "content": secrets["result"],
            }
        ]
    )
    executor.execute_tool_calls = successful_execute
    agent = BaseAgent(name="tool-audit", tool_executor=executor)
    service = ToolService()
    context = create_run_context()
    tool_call = ToolCall(
        id=secrets["call_id"],
        function=FunctionCall(
            name="lookup_record",
            arguments=f'{{"query": "{secrets["arguments"]}"}}',
        ),
    )
    malformed_call = ToolCall(
        id=secrets["call_id"],
        function=FunctionCall(
            name="lookup_record",
            arguments=f'{{"query": "{secrets["malformed"]}"',
        ),
    )
    mock_logger = MagicMock()

    with patch("continuum.agent.services.tool_service.logger", mock_logger):
        result, metadata = await service.execute_tool_call(agent, tool_call, context)
        malformed_result, _ = await service.execute_tool_call(agent, malformed_call, context)
        executor.execute_tool_calls = AsyncMock(side_effect=RuntimeError(secrets["error"]))
        failed_result, failed_metadata = await service.execute_tool_call(agent, tool_call, context)

        global_executor = MagicMock()
        global_executor.tool_registry = {}
        global_executor.execute_tool_calls = AsyncMock(
            return_value=[
                {
                    "role": "tool",
                    "tool_call_id": secrets["call_id"],
                    "content": secrets["result"],
                }
            ]
        )
        global_result, _ = await ToolService(tool_executor=global_executor).execute_tool_call(
            BaseAgent(name="global-tool-audit"),
            tool_call,
            context,
        )

        global_executor.execute_tool_calls = AsyncMock(side_effect=RuntimeError(secrets["error"]))
        with pytest.raises(AgentToolError) as global_error:
            await ToolService(tool_executor=global_executor).execute_tool_call(
                BaseAgent(name="global-tool-audit"),
                tool_call,
                context,
            )

        parallel_service = ToolService(config=RunnerConfig(parallel_tool_calls=True))
        parallel_service.execute_tool_call = AsyncMock(side_effect=RuntimeError(secrets["error"]))
        batch_results = await parallel_service.execute_tools_batch(
            agent,
            [tool_call, tool_call.model_copy(update={"id": "second-call"})],
            context,
        )

    assert result["content"] == secrets["result"]
    assert malformed_result["content"] == secrets["result"]
    assert metadata["success"] is True
    assert failed_result["role"] == "tool"
    assert failed_metadata["success"] is False
    assert global_result["content"] == secrets["result"]
    assert secrets["error"] in str(global_error.value.original_error)
    assert all(secrets["error"] in item["content"] for item in batch_results)
    first_executor_call = successful_execute.await_args_list[0].kwargs["tool_calls"][0]
    assert secrets["arguments"] in first_executor_call.function.arguments
    logs = _logged_text(mock_logger)
    assert "agent=tool-audit" in logs
    assert "tool=lookup_record" in logs
    assert "status=success" in logs
    assert "result_type=str" in logs
    assert "error_type=RuntimeError" in logs
    assert "error_type=JSONDecodeError" in logs
    for secret in secrets.values():
        assert secret not in logs


async def test_planner_logs_structure_without_plan_instructions_or_errors() -> None:
    secrets = {
        "goal": "PLANNER_GOAL_SECRET",
        "instruction": "PLANNER_INSTRUCTION_SECRET",
        "step_id": "PLANNER_STEP_ID_SECRET",
        "output": "PLANNER_OUTPUT_SECRET",
        "error": "PLANNER_EXCEPTION_SECRET",
        "parse": "PLANNER_PARSE_SECRET",
        "run_id": "123e4567-e89b-12d3-a456-426614174077",
    }
    planner = PlannerAgent(
        name="planner-audit",
        agent=BaseAgent(name="worker-agent"),
        planning_config=PlanningConfig(replan_on_failure=False),
    )
    plan_content = (
        '{"steps": [{"step_id": "'
        + secrets["step_id"]
        + '", "instruction": "'
        + secrets["instruction"]
        + '"}]}'
    )
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(content=plan_content, usage=None))
    )
    successful_run = AsyncMock(
        return_value=AgentResponse(
            content=secrets["output"],
            agent_name="worker-agent",
            status=ResponseStatus.SUCCESS,
        )
    )
    runner = SimpleNamespace(run=successful_run)
    workflow_span = MagicMock()
    step_span = MagicMock()
    step_span.__aenter__ = AsyncMock(return_value=step_span)
    step_span.__aexit__ = AsyncMock(return_value=False)
    mock_logger = MagicMock()
    reported: list[tuple[dict, str]] = []

    def capture_report(error) -> None:
        reported.append((dict(error.context), str(error)))

    with (
        patch("continuum.agent.workflow.planner.logger", mock_logger),
        patch("continuum.agent.workflow.planner.SpanScope", return_value=step_span),
        patch("continuum.exceptions._error_reporter", capture_report),
    ):
        steps, _ = await planner._generate_plan(secrets["goal"], llm)
        result = await planner._drive(
            steps,
            secrets["goal"],
            runner,
            create_run_context(),
            workflow_span=workflow_span,
            start_stage=1,
            goal=secrets["goal"],
            llm_client=llm,
            initial_usage=TokenUsage(),
        )

        failing_llm = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError(secrets["error"])))
        failed_steps, _ = await planner._generate_plan(secrets["goal"], failing_llm)
        await planner._maybe_replan(
            secrets["goal"],
            [],
            steps,
            secrets["output"],
            failing_llm,
        )
        await planner._replan_on_failure(
            secrets["goal"],
            [],
            steps[0],
            [],
            secrets["error"],
            failing_llm,
        )
        assert planner._parse_steps(secrets["parse"]) == []

        runner.run = AsyncMock(side_effect=RuntimeError(secrets["error"]))
        with pytest.raises(PlannerWorkflowError) as planner_error:
            await planner._drive(
                steps,
                secrets["goal"],
                runner,
                create_run_context(run_id=secrets["run_id"]),
                workflow_span=workflow_span,
                start_stage=1,
                goal=secrets["goal"],
                llm_client=llm,
                initial_usage=TokenUsage(),
            )

    assert steps[0]["instruction"] == secrets["instruction"]
    assert secrets["goal"] in llm.chat.await_args_list[0].kwargs["messages"][0]["content"]
    assert secrets["instruction"] in successful_run.await_args_list[0].kwargs["input"]
    assert result.content == secrets["output"]
    assert failed_steps == []
    logs = _logged_text(mock_logger)
    assert "agent=planner-audit" in logs
    assert "step_index=1" in logs
    assert "delegate=worker-agent" in logs
    assert "response_type=str" in logs
    assert "error_type=RuntimeError" in logs
    assert "error_type=JSONDecodeError" in logs
    for secret in secrets.values():
        assert secret not in logs
        assert secret not in str(planner_error.value)
    assert planner_error.value.failed_agent == "worker-agent"
    assert planner_error.value.run_id == secrets["run_id"]
    assert isinstance(planner_error.value.__cause__, RuntimeError)
    assert secrets["error"] in str(planner_error.value.__cause__)
    workflow_error_text = workflow_span.set_error.call_args.args[0]
    assert workflow_error_text == "Planner step failed (step_index=1, error_type=RuntimeError)"
    assert step_span.set_error.call_args.args[0] == workflow_error_text
    correlation = sha256(secrets["run_id"].encode("utf-8")).hexdigest()[:16]
    assert reported == [
        (
            {
                "root_cause_type": "RuntimeError",
                "run_correlation": f"sha256:{correlation}",
            },
            (
                "[ORCHESTRATOR_ERROR] Planner step failed "
                "(step_index=1, error_type=RuntimeError) | "
                f"Context: root_cause_type=RuntimeError, run_correlation=sha256:{correlation}"
            ),
        )
    ]


async def test_parallel_and_scatter_wrappers_redact_chained_exception_messages() -> None:
    error_secret = "WORKFLOW_WRAPPER_EXCEPTION_SECRET"
    run_id_secret = "123e4567-e89b-12d3-a456-426614174088"
    runner = MagicMock()
    runner.ensure_recorder.return_value = False
    runner.run = AsyncMock()
    context = create_run_context(run_id=run_id_secret)
    parallel = ParallelAgent(name="parallel-wrapper", agents=[BaseAgent(name="branch")])
    scatter = ScatterAgent(name="scatter-wrapper", agents=[BaseAgent(name="branch")])
    mock_logger = MagicMock()
    reported: list[tuple[dict, str]] = []

    def capture_report(error) -> None:
        reported.append((dict(error.context), str(error)))

    def close_branch_coroutine(coroutine):
        coroutine.close()
        return MagicMock()

    with (
        patch("continuum.agent.workflow.parallel.logger", mock_logger),
        patch(
            "continuum.agent.workflow.parallel.asyncio.create_task",
            side_effect=close_branch_coroutine,
        ),
        patch(
            "continuum.agent.workflow.parallel.asyncio.wait",
            AsyncMock(side_effect=RuntimeError(error_secret)),
        ),
        patch("continuum.exceptions._error_reporter", capture_report),
        pytest.raises(ParallelWorkflowError) as parallel_error,
    ):
        await parallel.execute(error_secret, runner, context)

    with (
        patch(
            "continuum.agent.workflow.scatter.asyncio.create_task",
            side_effect=close_branch_coroutine,
        ),
        patch(
            "continuum.agent.workflow.scatter.asyncio.wait",
            AsyncMock(side_effect=RuntimeError(error_secret)),
        ),
        patch("continuum.exceptions._error_reporter", capture_report),
        pytest.raises(ParallelWorkflowError) as scatter_error,
    ):
        await scatter._scatter([error_secret], runner, create_run_context(run_id=run_id_secret))

    for wrapped in (parallel_error.value, scatter_error.value):
        assert error_secret not in str(wrapped)
        assert run_id_secret not in str(wrapped)
        assert wrapped.original_error is None
        assert wrapped.run_id == run_id_secret
        assert isinstance(wrapped.__cause__, RuntimeError)
        assert error_secret in str(wrapped.__cause__)
    assert error_secret not in _logged_text(mock_logger)
    assert run_id_secret not in _logged_text(mock_logger)
    correlation = sha256(run_id_secret.encode("utf-8")).hexdigest()[:16]
    expected_context = {
        "root_cause_type": "RuntimeError",
        "run_correlation": f"sha256:{correlation}",
    }
    assert [context for context, _text in reported] == [expected_context, expected_context]
    assert all(
        error_secret not in text and run_id_secret not in text for _context, text in reported
    )
