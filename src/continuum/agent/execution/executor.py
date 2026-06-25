"""
Executor - Core execution logic for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.agent.exceptions import (
    HandoffLoopError,
    MaxTurnsExceededError,
    StructuredOutputError,
)
from continuum.agent.execution.trace_capture import (
    capture_snapshot,
    record_llm_turn,
    record_tool_steps,
)
from continuum.agent.interfaces.executor_interface import IExecutor
from continuum.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    RunState,
    TokenUsage,
    ToolExecutionSummary,
)
from continuum.config import settings
from continuum.llm.config import LLMConfig
from continuum.llm.structured_output import (
    coerce_and_validate,
    is_pydantic_schema,
    schema_prompt,
    to_openai_response_format,
)
from continuum.logging import get_logger
from continuum.observability.metrics import get_metrics_collector
from continuum.observability.trace_context import SpanScope, truncate_data
from continuum.tools.tool_attention.router import _tool_name

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.execution.handoff_executor import HandoffExecutor
    from continuum.agent.execution.tool_handler import ToolHandler
    from continuum.llm import LLMClient

logger = get_logger(__name__)

# Max consecutive handoffs to the SAME target before we declare a handoff loop.
# Catches routing agents stuck re-routing under return_to_parent=True before they
# silently exhaust max_turns.
_MAX_CONSECUTIVE_HANDOFFS = 3

# Extra LLM calls allowed to coax a valid structured output after the first
# attempt fails validation. 1 = one retry (see ADR / fix plan, decision #3).
_MAX_STRUCTURED_OUTPUT_RETRIES = 1


def _enrich_config_for_gateway(config: LLMConfig, context: RunContext) -> LLMConfig:
    """Inject gateway metadata into body.metadata when Smart Gateway is configured."""
    if not settings.smart_gateway_url:
        return config
    return config.with_overrides(
        extra_body={
            "metadata": {
                "session_id": context.session_id or context.run_id,
                "trace_id": context.trace_id,
            }
        }
    )


class Executor(IExecutor):
    """
    Core executor for agent runs.

    Handles the main conversation loop with tool calls and handoffs.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_handler: ToolHandler | None = None,
        handoff_executor: HandoffExecutor | None = None,
    ):
        """
        Initialize executor.

        Args:
            llm_client: LLM client for model calls
            tool_handler: Tool handler for tool execution
            handoff_executor: Handoff executor for agent handoffs
        """
        self._llm_client = llm_client
        self._tool_handler = tool_handler
        self._handoff_executor = handoff_executor

        # Set executor reference in handoff executor for recursive execution
        if self._handoff_executor and hasattr(self._handoff_executor, "_executor"):
            self._handoff_executor._executor = self

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client."""
        if not self._llm_client:
            raise RuntimeError("LLMClient not provided to Executor")
        return self._llm_client

    async def execute_loop(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AgentResponse:
        """
        Execute the main conversation loop.

        Args:
            agent: Agent to execute
            messages: Initial messages
            context: Run context
            run_state: Run state

        Returns:
            AgentResponse with the result
        """
        turn = 0
        total_usage = TokenUsage()
        metrics = get_metrics_collector()

        # Collect tool execution summaries for session storage.
        # Capped to max_turns to prevent unbounded growth in long-running agents.
        all_tool_summaries: list[ToolExecutionSummary] = []
        _MAX_TOOL_SUMMARIES = context.max_turns

        # Two-pass reasoning: silent think-first call before the main loop
        if agent.config and agent.config.reasoning_mode:
            reasoning_text, reasoning_usage = await self._run_reasoning_pass(
                messages=messages,
                agent=agent,
                context=context,
            )
            messages.append(
                {"role": "system", "content": f"<reasoning>\n{reasoning_text}\n</reasoning>"}
            )
            total_usage = total_usage.add(reasoning_usage)
            logger.info(
                f"🧠 Reasoning pass completed for agent {agent.name} "
                f"({reasoning_usage.total_tokens} tokens)"
            )

        while turn < context.max_turns:
            turn += 1
            run_state.turn_count = turn

            # Create span for this turn
            async with SpanScope(
                f"turn.{turn}",
                input=truncate_data(
                    {
                        "turn": turn,
                        "message_count": len(messages),
                        "last_message_role": messages[-1].get("role") if messages else None,
                    }
                ),
                metadata={
                    "agent_name": agent.name,
                    "turn_number": turn,
                    "max_turns": context.max_turns,
                },
            ) as turn_span:
                # Filtered tools set by message_builder via apply_tool_attention.
                tools = (
                    context.metadata.get("_filtered_tools") if context.metadata else None
                ) or agent.get_tools_for_llm()
                tool_names = [t.get("function", {}).get("name", "") for t in tools] if tools else []
                turn_span.add_metadata("available_tools", tool_names[:20])

                # Phase 1: insert tool catalogue after system messages, before history.
                # Ephemeral — not added to `messages` (session history).
                _phase1 = context.metadata.get("tool_summary_message") if context.metadata else None
                if _phase1:
                    _insert_at = 0
                    for _i, _msg in enumerate(messages):
                        if _msg.get("role") == "system":
                            _insert_at = _i + 1
                        else:
                            break
                    llm_messages = messages[:_insert_at] + [_phase1] + messages[_insert_at:]
                else:
                    llm_messages = messages

                # Make LLM call
                try:
                    # Create LLMConfig for this agent (includes JSON mode if enabled)
                    llm_config = _enrich_config_for_gateway(
                        LLMConfig.from_agent_config(agent), context
                    )

                    # Log JSON mode status
                    if agent.enable_json_mode:
                        json_mode_info = "enabled"
                        if agent.json_schema:
                            if isinstance(agent.json_schema, type):
                                json_mode_info += f" with schema: {agent.json_schema.__name__}"
                            else:
                                json_mode_info += " with JSON schema dict"
                        else:
                            json_mode_info += " (simple json_object mode)"
                        logger.info(
                            f"📋 JSON mode {json_mode_info} for agent {agent.name}",
                            extra={
                                "agent_name": agent.name,
                                "json_mode": True,
                                "json_schema": (
                                    agent.json_schema.__name__
                                    if isinstance(agent.json_schema, type)
                                    else "dict"
                                    if isinstance(agent.json_schema, dict)
                                    else None
                                ),
                            },
                        )

                    # Structured output: constrain THIS call to the schema only when
                    # the agent has no tools (every call is then a final answer).
                    # Tool-using agents are NOT constrained here — that would fight
                    # tool calling (R1); they get a separate formatting call after
                    # the loop. Constraint = schema-in-prompt (universal) + a
                    # json_schema response_format dict (honored natively by OpenAI,
                    # ignored safely elsewhere).
                    if is_pydantic_schema(agent.output_schema) and not agent.get_tools_for_llm():
                        llm_config = llm_config.model_copy(
                            update={
                                "response_format": to_openai_response_format(agent.output_schema)
                            }
                        )
                        llm_messages = llm_messages + [
                            {"role": "system", "content": schema_prompt(agent.output_schema)}
                        ]

                    # NOTE: We pass auto_session=False because Executor manages the
                    # conversation loop including tool calls.
                    response = await self.llm_client.chat(
                        messages=llm_messages,
                        tools=tools if tools else None,
                        config=llm_config,
                        session_id=context.session_id,
                        trace_metadata={"session_id": context.session_id}
                        if context.session_id
                        else None,
                        auto_session=False,  # Executor manages the message loop
                        priority=context.priority,
                        stage_priority=agent.config.stage_priority if agent.config else 5,
                    )

                    # NEED_TOOL fallback: if LLM signals a missing tool, expand and retry once.
                    if response.content and "NEED_TOOL:" in response.content:
                        needed = (
                            response.content.split("NEED_TOOL:")[1].strip().split()[0].rstrip(".,;")
                        )
                        all_tools = agent.get_tools_for_llm()
                        extra = [t for t in all_tools if _tool_name(t) == needed]
                        if extra:
                            logger.info("tool-attention fallback: adding %s and retrying", needed)
                            expanded_tools = tools + [t for t in extra if t not in tools]
                            if context.metadata is not None:
                                promoted = context.metadata.get("promoted_tools", set())
                                context.metadata["promoted_tools"] = promoted | {needed}
                                # Persist expanded tools so subsequent turns include this tool.
                                context.metadata["_filtered_tools"] = expanded_tools
                            response = await self.llm_client.chat(
                                messages=llm_messages,
                                tools=expanded_tools,
                                config=llm_config,
                                session_id=context.session_id,
                                trace_metadata={"session_id": context.session_id}
                                if context.session_id
                                else None,
                                auto_session=False,
                                priority=context.priority,
                                stage_priority=agent.config.stage_priority if agent.config else 5,
                            )

                except Exception as e:
                    turn_span.set_error(str(e))
                    raise

                # Track usage
                if response.usage:
                    total_usage = total_usage.add(
                        TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens or 0,
                            completion_tokens=response.usage.completion_tokens or 0,
                            total_tokens=response.usage.total_tokens or 0,
                        )
                    )
                    turn_span.add_metadata(
                        "tokens",
                        {
                            "prompt": response.usage.prompt_tokens,
                            "completion": response.usage.completion_tokens,
                            "total": response.usage.total_tokens,
                        },
                    )

                    # Track token usage in metrics
                    metrics.track_tokens(
                        f"turn_{turn}_llm",
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        model=agent.model,
                    )

                from continuum.agent.utils.message_utils import message_to_dict

                # Decision trace: checkpoint the exact messages SENT this turn —
                # the true resume point for fork() — BEFORE the assistant output
                # is appended below. (llm_messages aliases `messages` on the
                # non-reasoning path, so this must be captured pre-append or the
                # snapshot would also contain this turn's own answer and fork()
                # would replay an already-finished conversation.)
                recorder = context.recorder
                # This agent's handoff stack (root → … → this agent), read from
                # its OWN run_state per turn and passed into each record() call —
                # never a shared recorder field — so concurrent agents can't
                # clobber each other's stack. Recomputed each turn, so it's also
                # correct after a return-to-parent handoff pops the child.
                _agent_stack = run_state.get_agent_stack_snapshot()
                # Checkpoint the messages sent this turn BEFORE the assistant
                # output is appended below (shared with the streaming path).
                _snapshot = capture_snapshot(recorder, llm_messages)

                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)

                run_state.messages = [message_to_dict(m) for m in messages]

                # Record this turn's LLM decision (nests reasoning/tool steps below it).
                llm_step_id = record_llm_turn(
                    recorder,
                    agent.name,
                    turn,
                    content=response.content,
                    has_tool_calls=bool(response.tool_calls),
                    usage=response.usage,
                    snapshot=_snapshot,
                    agent_stack=_agent_stack,
                )

                # Log LLM response details
                if response.model and settings.smart_gateway_url:
                    logger.info("🎯 Gateway selected model: %s", response.model)
                if not response.tool_calls:
                    logger.debug(
                        f"💬 LLM response (no tool calls) on turn {turn}: "
                        f"content_preview={(response.content or '')[:150]}, "
                        f"messages_in_context={len(messages)}"
                    )
                else:
                    tool_names = [
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "")
                        for tc in response.tool_calls
                    ]
                    logger.info(
                        f"🔧 LLM response (with {len(response.tool_calls)} tool calls) on turn {turn}: {', '.join(tool_names)}"
                    )

                # Handle tool calls
                if response.tool_calls:
                    called_tool_names = [
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "")
                        for tc in response.tool_calls
                    ]
                    turn_span.add_metadata("tool_calls", called_tool_names)
                    logger.info(
                        f"🤖 LLM requesting {len(response.tool_calls)} tool(s): {', '.join(called_tool_names)}"
                    )

                    # Separate handoffs from regular tools
                    handoff_calls = []
                    regular_tool_calls = []

                    for tc in response.tool_calls:
                        tool_name = (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            handoff_calls.append((tc, target))
                        else:
                            regular_tool_calls.append(tc)

                    # Handle think tool calls inline (ReAct reasoning step)
                    # think() is a no-op that logs the LLM's reasoning and returns immediately
                    think_calls = [
                        tc
                        for tc in regular_tool_calls
                        if (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        == "think"
                    ]
                    regular_tool_calls = [
                        tc
                        for tc in regular_tool_calls
                        if (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        != "think"
                    ]
                    for tc in think_calls:
                        import json as _json

                        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                        args_str = (
                            tc.function.arguments
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("arguments", "{}")
                        )
                        try:
                            thought = _json.loads(args_str).get("thought", "")
                        except Exception:
                            thought = str(args_str)
                        logger.info(f"💭 Agent thought: {thought}")
                        if recorder is not None:
                            recorder.record_reasoning(
                                agent.name,
                                turn,
                                thought,
                                parent_id=llm_step_id,
                                agent_stack=_agent_stack,
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "Thought recorded.",
                            }
                        )

                    # Execute regular tools
                    if regular_tool_calls and self._tool_handler:
                        # Create summary for this turn's tool executions
                        turn_tool_summary = ToolExecutionSummary()

                        tool_results = await self._tool_handler.execute_tools_batch(
                            agent=agent,
                            tool_calls=regular_tool_calls,
                            context=context,
                            tool_summary=turn_tool_summary,
                        )
                        messages.extend(tool_results)

                        if recorder is not None:
                            record_tool_steps(
                                recorder,
                                agent.name,
                                turn,
                                regular_tool_calls,
                                tool_results,
                                parent_id=llm_step_id,
                                agent_stack=_agent_stack,
                            )

                        # Store the summary if tools were executed (capped to prevent leak)
                        if not turn_tool_summary.is_empty():
                            if len(all_tool_summaries) >= _MAX_TOOL_SUMMARIES:
                                # Merge oldest into the second-oldest to keep the list bounded
                                all_tool_summaries[0] = (
                                    self._merge_tool_summaries(
                                        [all_tool_summaries[0], all_tool_summaries.pop(1)]
                                    )
                                    or all_tool_summaries[0]
                                )
                            all_tool_summaries.append(turn_tool_summary)

                    # Execute handoffs sequentially (they may return early)
                    if handoff_calls and self._handoff_executor:
                        for tc, target in handoff_calls:
                            _hc = agent.get_handoff(target)
                            # Guard against handoff loops: a routing agent under
                            # return_to_parent=True can keep re-handing-off to the same
                            # target every turn (the recipient is popped from the stack,
                            # so cycle/depth checks don't catch it), silently burning
                            # turns until MaxTurnsExceededError. Only return_to_parent
                            # handoffs can loop this way — a return_to_parent=False
                            # handoff returns its result directly and cannot re-route —
                            # so the guard is scoped to them to avoid false positives on
                            # legitimate repeated same-target handoffs.
                            if _hc is None or _hc.return_to_parent:
                                _streak = 0
                                for _h in reversed(run_state.handoff_chain):
                                    if _h.get("to_agent") == target:
                                        _streak += 1
                                    else:
                                        break
                                if _streak >= _MAX_CONSECUTIVE_HANDOFFS:
                                    raise HandoffLoopError(
                                        from_agent=agent.name,
                                        to_agent=target,
                                        count=_streak,
                                        run_id=context.run_id,
                                        trace_id=context.trace_id,
                                    )
                            if recorder is not None:
                                recorder.record_handoff(
                                    agent.name,
                                    target,
                                    turn,
                                    parent_id=llm_step_id,
                                    agent_stack=_agent_stack,
                                    return_to_parent=bool(_hc and _hc.return_to_parent),
                                )
                            handoff_result = await self._handoff_executor.execute_handoff(
                                agent=agent,
                                target_name=target,
                                tool_call=tc,
                                messages=messages,
                                context=context,
                                run_state=run_state,
                            )

                            if handoff_result.success and handoff_result.response:
                                handoff_config = agent.get_handoff(target)
                                return_to_parent = (
                                    handoff_config and handoff_config.return_to_parent
                                )

                                if return_to_parent:
                                    # Add executor's result as a tool result so the
                                    # orchestrator's LLM runs again and generates the
                                    # final user-facing response.
                                    tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                                    executor_content = handoff_result.response.content or ""
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tc_id,
                                            "content": executor_content,
                                        }
                                    )
                                    logger.info(
                                        f"🔁 RETURN TO PARENT [{agent.name}] ← [{target}]\n"
                                        f"[tool] {executor_content[:500]}\n" + "=" * 30
                                    )
                                    total_usage = total_usage.add(handoff_result.response.usage)
                                    # Pop the target agent so the parent can hand off
                                    # to the same target again on the next turn.
                                    run_state.pop_agent()
                                    run_state.current_agent = agent.name
                                    # (No recorder-stack reset needed: the parent's
                                    # next turn recomputes _agent_stack from run_state,
                                    # which is back to the parent's stack after the pop.)
                                    turn_span.set_output(
                                        {
                                            "handoff_to": target,
                                            "success": True,
                                            "return_to_parent": True,
                                        }
                                    )
                                    logger.info(
                                        f"===== RETURN TURN PROMPT [{agent.name}] =====\n"
                                        + "\n".join(
                                            f"[{m.get('role', '?')}] {str(m.get('content', '') or '')[:300]}"
                                            for m in messages
                                        )
                                        + "\n"
                                        + "=" * 30
                                    )
                                    continue
                                else:
                                    # No return_to_parent — return executor's response directly.
                                    if handoff_result.response.content:
                                        messages.append(
                                            {
                                                "role": "assistant",
                                                "content": handoff_result.response.content,
                                            }
                                        )
                                    turn_span.set_output(
                                        {
                                            "handoff_to": target,
                                            "success": True,
                                            "return_to_parent": False,
                                        }
                                    )
                                    return AgentResponse(
                                        content=handoff_result.response.content,
                                        agent_name=handoff_result.response.agent_name,
                                        status=ResponseStatus.SUCCESS,
                                        usage=total_usage.add(handoff_result.response.usage),
                                        turn_count=turn,
                                        handoff_result=handoff_result,
                                        messages=messages,
                                    )
                            else:
                                # Handoff failed. execute_handoff pushes the target
                                # only AFTER its early depth/cycle/registry checks
                                # pass, so the target is on the stack only in
                                # post-push failures — where it sits on top. Pop it
                                # whenever it's on top, regardless of return_to_parent,
                                # so a failed hop never leaves the target stuck (which
                                # would trip false cycle detection on a retry).
                                if run_state.agent_stack and run_state.agent_stack[-1] == target:
                                    run_state.pop_agent()
                                    run_state.current_agent = agent.name
                                # Add error as tool result so the LLM can react
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": (
                                            tc.id if hasattr(tc, "id") else tc.get("id", "")
                                        ),
                                        "content": f"Handoff failed: {handoff_result.error or 'Unknown error'}",
                                    }
                                )

                    # Update state
                    from continuum.agent.utils.message_utils import message_to_dict

                    run_state.messages = [message_to_dict(m) for m in messages]

                    # Update span with tool execution summary
                    turn_span.set_output(
                        {
                            "tool_calls_executed": len(regular_tool_calls),
                            "handoffs_attempted": len(handoff_calls),
                            "continuing_to_next_turn": True,
                        }
                    )

                    # Continue to next turn
                    continue

                # No tool calls, we're done
                turn_span.set_output(
                    {
                        "response_preview": (response.content or "")[:200],
                        "final_turn": True,
                    }
                )

                # Merge all tool summaries into one for the response
                merged_tool_summary = self._merge_tool_summaries(all_tool_summaries)

                # Structured output: produce a validated instance, or a clear error.
                # The model was already steered toward the schema (prompt + response_format
                # for no-tool agents). For tool agents the loop ran unconstrained, so
                # _resolve_structured_output makes a separate formatting call here.
                structured_output = None
                structured_output_error = None
                if is_pydantic_schema(agent.output_schema):
                    (
                        structured_output,
                        structured_output_error,
                    ) = await self._resolve_structured_output(
                        agent, messages, response.content, context
                    )
                    if structured_output is not None:
                        logger.info(
                            f"✅ structured_output ready for agent {agent.name} "
                            f"({agent.output_schema.__name__})",
                            extra={"agent_name": agent.name},
                        )
                    elif agent.output_schema_strict:
                        raise StructuredOutputError(
                            schema_name=agent.output_schema.__name__,
                            reason=structured_output_error or "no valid structured output",
                            agent_name=agent.name,
                            run_id=context.run_id,
                            trace_id=context.trace_id,
                        )
                    else:
                        # Soft failure: visible (warning + error field), not silent.
                        logger.warning(
                            f"⚠️ structured_output unavailable for agent {agent.name}: "
                            f"{structured_output_error}",
                            extra={
                                "agent_name": agent.name,
                                "output_schema": agent.output_schema.__name__,
                                "error": structured_output_error,
                            },
                        )
                elif agent.enable_json_mode and response.content:
                    # Legacy path: JSON mode without output_schema — just verify it's JSON.
                    import json as _json

                    try:
                        _json.loads(response.content.strip())
                    except _json.JSONDecodeError as e:
                        logger.warning(
                            f"⚠️ JSON mode enabled but response is not valid JSON for "
                            f"agent {agent.name}: {e}"
                        )

                # No tool calls, we're done
                agent_response = AgentResponse(
                    content=response.content or "",
                    structured_output=structured_output,
                    structured_output_error=structured_output_error,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=turn,
                    messages=messages,
                )
                # Store tool summary in metadata for session storage
                if (
                    merged_tool_summary
                    and not merged_tool_summary.is_empty()
                    and context.metadata is not None
                ):
                    context.metadata["tool_execution_summary"] = merged_tool_summary.to_dict()

                return agent_response

        # Max turns exceeded
        raise MaxTurnsExceededError(
            max_turns=context.max_turns,
            current_turn=turn,
            agent_name=agent.name,
            run_id=context.run_id,
        )

    async def _run_reasoning_pass(
        self,
        messages: list[dict[str, Any]],
        agent: BaseAgent,
        context: RunContext,
    ) -> tuple[str, TokenUsage]:
        """
        Run a silent think-first LLM call and return (reasoning_text, token_usage).

        The reasoning output is NOT shown to the user; it is injected as a
        <reasoning> system message to guide the main turn loop.
        """
        reasoning_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Before responding to the user, think step-by-step about how to best "
                    "answer the request. Lay out your reasoning clearly."
                ),
            }
        ]

        reasoning_config = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), context)
        reasoning_config.max_tokens = 512

        response = await self.llm_client.chat(
            messages=reasoning_messages,
            config=reasoning_config,
            session_id=context.session_id,
            auto_session=False,
            priority=context.priority,
            stage_priority=agent.config.stage_priority if agent.config else 5,
        )

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return response.content or "", usage

    async def _execute_react_loop(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AgentResponse:
        """
        Execute the ReAct (Reason+Act) loop.

        Instead of function calling, the LLM writes actions as text:
            Thought: ...
            Action: <tool_name or 'Reason' or 'Final Answer'>
            Action Input: {...}

        This method parses that text, executes the named tool via ToolHandler,
        then injects the real result as:
            Observation: <result>

        The loop repeats until the LLM writes 'Action: Final Answer'.
        """
        from continuum.agent.utils.message_utils import message_to_dict

        turn = 0
        total_usage = TokenUsage()

        while turn < context.max_turns:
            turn += 1
            run_state.turn_count = turn

            llm_config = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), context)

            # ReAct does NOT use function calling — LLM writes actions as plain text
            response = await self.llm_client.chat(
                messages=messages,
                tools=None,
                config=llm_config,
                session_id=context.session_id,
                auto_session=False,
                priority=context.priority,
                stage_priority=agent.config.stage_priority if agent.config else 5,
            )

            if response.usage:
                total_usage = total_usage.add(
                    TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        total_tokens=response.usage.total_tokens or 0,
                    )
                )

            content = response.content or ""
            action, action_input, final_answer = self._parse_react_action(content)

            logger.info(f"🔄 ReAct turn {turn}: action={action!r}")

            # Final Answer — stop the loop
            if final_answer is not None:
                messages.append({"role": "assistant", "content": content})
                run_state.messages = [message_to_dict(m) for m in messages]
                return AgentResponse(
                    content=final_answer,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=turn,
                    messages=messages,
                )

            # No action parsed — treat full response as final answer
            if action is None:
                messages.append({"role": "assistant", "content": content})
                return AgentResponse(
                    content=content,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=turn,
                    messages=messages,
                )

            # Action: Reason — LLM uses its own knowledge, no tool call needed
            if action.lower() == "reason":
                messages.append({"role": "assistant", "content": content})
                run_state.messages = [message_to_dict(m) for m in messages]
                continue

            # Action: tool name — execute via ToolHandler
            # Strip any fabricated Observation the LLM may have written
            truncated = self._truncate_before_observation(content)
            messages.append({"role": "assistant", "content": truncated})

            observation = await self._execute_react_tool(
                agent=agent,
                tool_name=action,
                tool_args=action_input or {},
                context=context,
            )

            logger.info(f"✅ ReAct observation for '{action}': {observation[:200]}")

            # Inject the real observation so the LLM sees it on the next turn
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            run_state.messages = [message_to_dict(m) for m in messages]

        raise MaxTurnsExceededError(
            max_turns=context.max_turns,
            current_turn=turn,
            agent_name=agent.name,
            run_id=context.run_id,
        )

    def _parse_react_action(
        self,
        content: str,
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """
        Parse a ReAct-format LLM response.

        Returns:
            (action, action_input, final_answer)
            - action:       tool name, 'Reason', or 'Final Answer'
            - action_input: parsed dict of arguments (for tool calls)
            - final_answer: the answer text when action == 'Final Answer'
        """
        import json
        import re

        # Find all Action lines — use the last one as the current step
        action_matches = list(re.finditer(r"^Action:\s*(.+?)$", content, re.MULTILINE))
        if not action_matches:
            return None, None, None

        action = action_matches[-1].group(1).strip()

        if action == "Final Answer":
            final_match = re.search(r"Final Answer:\s*(.*?)$", content, re.DOTALL)
            answer = final_match.group(1).strip() if final_match else content
            return "Final Answer", None, answer

        # Parse Action Input JSON
        input_match = re.search(r"Action Input:\s*(\{.*?\})", content, re.DOTALL)
        action_input = None
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}

        return action, action_input, None

    def _truncate_before_observation(self, content: str) -> str:
        """Strip any fabricated 'Observation:' line the LLM wrote so we can inject the real one."""
        import re

        match = re.search(r"\nObservation:", content)
        if match:
            return content[: match.start()].rstrip()
        return content

    async def _execute_react_tool(
        self,
        agent: BaseAgent,
        tool_name: str,
        tool_args: dict[str, Any],
        context: RunContext,
    ) -> str:
        """Execute a single tool in ReAct mode and return its result as a string."""
        import json

        from continuum.llm.types import FunctionCall
        from continuum.llm.types import ToolCall as LLMToolCall

        tc = LLMToolCall(
            id=f"react_{tool_name}_{context.run_id[:8]}",
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(tool_args),
            ),
        )

        if self._tool_handler:
            try:
                results = await self._tool_handler.execute_tools_batch(
                    agent=agent,
                    tool_calls=[tc],
                    context=context,
                )
                if results:
                    return str(results[0].get("content", "No result"))
            except Exception as e:
                logger.warning(f"ReAct tool '{tool_name}' failed: {e}")
                return f"Error executing '{tool_name}': {e}"

        return f"Tool '{tool_name}' is not available"

    async def _resolve_structured_output(
        self,
        agent: BaseAgent,
        base_messages: list[dict[str, Any]],
        content: str | None,
        context: RunContext,
    ) -> tuple[Any, str | None]:
        """Produce a validated ``output_schema`` instance, or (None, reason).

        First tries the content already produced. For no-tool agents that content
        came from a schema-constrained call, so it counts as the primary attempt and
        only ``_MAX_STRUCTURED_OUTPUT_RETRIES`` dedicated formatting calls follow
        (total = 1 constrained + 1 retry). For tool agents the loop ran
        unconstrained, so the prose content is not a structured attempt and the
        full ``1 + _MAX_STRUCTURED_OUTPUT_RETRIES`` formatting-call budget applies.
        """
        schema = agent.output_schema
        assert schema is not None  # caller guards on agent.output_schema

        obj, err = coerce_and_validate(content, schema)
        if obj is not None:
            return obj, None

        # The constrained inline call (no-tool agents) already spent the primary
        # attempt; tool agents' prose content did not, so they keep the full budget.
        primary_already_spent = not agent.get_tools_for_llm()
        format_calls = (1 + _MAX_STRUCTURED_OUTPUT_RETRIES) - (1 if primary_already_spent else 0)

        prior = content
        last_err = err
        for _ in range(format_calls):
            try:
                fmt_content = await self._structured_format_call(
                    agent, base_messages, prior, last_err, context
                )
            except Exception as e:  # provider rejected the request, etc.
                logger.warning(
                    f"structured-output formatting call failed for agent {agent.name}: {e}"
                )
                last_err = f"formatting call failed: {e}"
                break
            obj, last_err = coerce_and_validate(fmt_content, schema)
            if obj is not None:
                return obj, None
            prior = fmt_content
        return None, last_err

    async def _structured_format_call(
        self,
        agent: BaseAgent,
        base_messages: list[dict[str, Any]],
        prior_content: str | None,
        error: str | None,
        context: RunContext,
    ) -> str | None:
        """One dedicated, tool-free, schema-constrained call to format the answer."""
        schema = agent.output_schema
        assert schema is not None

        instruction = schema_prompt(schema)
        if prior_content:
            instruction += f"\n\nConvert the following into that JSON object:\n{prior_content}"
        if error:
            instruction += (
                f"\n\nThe previous attempt was invalid ({error}). Return corrected JSON only."
            )
        fmt_messages = list(base_messages) + [{"role": "user", "content": instruction}]

        cfg = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), context)
        cfg = cfg.model_copy(update={"response_format": to_openai_response_format(schema)})

        resp = await self.llm_client.chat(
            messages=fmt_messages,
            tools=None,  # no tools — this is a pure formatting pass
            config=cfg,
            session_id=context.session_id,
            auto_session=False,
            priority=context.priority,
            stage_priority=agent.config.stage_priority if agent.config else 5,
        )
        return resp.content

    def _merge_tool_summaries(
        self,
        summaries: list[ToolExecutionSummary],
    ) -> ToolExecutionSummary | None:
        """Merge multiple turn tool summaries into one."""
        if not summaries:
            return None

        merged = ToolExecutionSummary()

        for summary in summaries:
            merged.tools_used.extend(summary.tools_used)
            merged.tool_count += summary.tool_count
            merged.total_latency_ms += summary.total_latency_ms
            merged.tool_latencies.update(summary.tool_latencies)
            merged.success_count += summary.success_count
            merged.error_count += summary.error_count
            merged.errors.extend(summary.errors)
            merged.input_tokens += summary.input_tokens
            merged.output_tokens += summary.output_tokens

            # Merge servers (unique)
            for server in summary.servers_used:
                if server not in merged.servers_used:
                    merged.servers_used.append(server)

            # Merge auth info
            merged.auth_info.update(summary.auth_info)

        return merged
