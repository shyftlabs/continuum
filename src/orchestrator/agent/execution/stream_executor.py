"""
Stream Executor - Streaming execution logic for agents.

Extracted from AgentRunner to provide clean separation of concerns.
Simplified version for streaming that shares base logic with Executor.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from orchestrator.agent.execution.executor import _enrich_config_for_gateway
from orchestrator.agent.interfaces.executor_interface import IStreamExecutor
from orchestrator.agent.types import AgentEvent, EventType, RunContext, RunState
from orchestrator.llm.config import LLMConfig
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.llm import LLMClient

logger = get_logger(__name__)


class StreamExecutor(IStreamExecutor):
    """
    Stream executor for agent runs with streaming output.

    Simplified version that streams events as execution progresses.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ):
        """
        Initialize stream executor.

        Args:
            llm_client: LLM client for model calls
        """
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client."""
        if not self._llm_client:
            raise RuntimeError("LLMClient not provided to StreamExecutor")
        return self._llm_client

    async def execute_stream(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AsyncIterator[AgentEvent]:
        """
        .. deprecated::
            Use ``AgentRunner.run_stream()`` instead. This class does not execute
            tools — it emits placeholder events only.

        Execute with streaming output.

        Args:
            agent: Agent to execute
            messages: Initial messages
            context: Run context
            run_state: Run state

        Yields:
            AgentEvent for each step
        """
        warnings.warn(
            "StreamExecutor.execute_stream() does not execute tools and will be removed. "
            "Use AgentRunner.run_stream() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        turn = 0

        # Emit run start
        yield AgentEvent(
            type=EventType.RUN_START,
            agent_name=agent.name,
            run_id=context.run_id,
            data={"input": str(messages[-1].get("content", "")) if messages else ""},
            trace_id=context.trace_id,
        )

        try:
            while turn < context.max_turns:
                turn += 1
                run_state.turn_count = turn

                # Get tools
                tools = agent.get_tools_for_llm()

                # Stream LLM response
                content_parts = []
                tool_calls = []

                llm_config = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), context)

                async for chunk in self.llm_client.chat_stream(
                    messages=messages,
                    tools=tools if tools else None,
                    config=llm_config,
                    trace_metadata={"session_id": context.session_id}
                    if context.session_id
                    else None,
                ):
                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield AgentEvent(
                            type=EventType.CONTENT_DELTA,
                            agent_name=agent.name,
                            run_id=context.run_id,
                            data={"content": chunk.content},
                            trace_id=context.trace_id,
                        )

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                content = "".join(content_parts)

                # Emit content complete
                if content:
                    yield AgentEvent(
                        type=EventType.CONTENT_COMPLETE,
                        agent_name=agent.name,
                        run_id=context.run_id,
                        data={"content": content},
                        trace_id=context.trace_id,
                    )

                # Handle tool calls (simplified - full support requires non-streaming mode)
                if tool_calls:
                    # Add assistant message
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": [
                                tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls
                            ],
                        }
                    )

                    for tc in tool_calls:
                        tool_name = (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")

                        # Check for handoff
                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            yield AgentEvent(
                                type=EventType.HANDOFF_START,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={"target": target},
                                trace_id=context.trace_id,
                            )
                            # Note: Full handoff support requires non-streaming mode
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": f"Handoff to {target} initiated. Note: Full handoff support requires non-streaming mode.",
                                }
                            )
                            yield AgentEvent(
                                type=EventType.HANDOFF_END,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={
                                    "target": target,
                                    "note": "Streaming mode has limited handoff support",
                                },
                                trace_id=context.trace_id,
                            )
                            continue

                        # Execute tool (simplified - emit events)
                        yield AgentEvent(
                            type=EventType.TOOL_CALL_START,
                            agent_name=agent.name,
                            run_id=context.run_id,
                            data={"tool_name": tool_name},
                            trace_id=context.trace_id,
                        )

                        # Note: Actual tool execution would go here
                        # For streaming, we emit events but don't execute fully
                        yield AgentEvent(
                            type=EventType.TOOL_CALL_END,
                            agent_name=agent.name,
                            run_id=context.run_id,
                            data={"tool_name": tool_name, "result": "Streaming mode"},
                            trace_id=context.trace_id,
                        )

                    # Continue loop for next turn
                    continue

                # No tool calls, we're done
                break

            # Emit agent end
            yield AgentEvent(
                type=EventType.AGENT_END,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"turn_count": turn},
                trace_id=context.trace_id,
            )

            # Emit run end
            yield AgentEvent(
                type=EventType.RUN_END,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"content": content, "turn_count": turn},
                trace_id=context.trace_id,
            )

        except Exception as e:
            # Emit error event
            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"error": str(e), "error_type": type(e).__name__},
                trace_id=context.trace_id,
            )
            raise
