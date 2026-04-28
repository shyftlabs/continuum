"""Run lifecycle management -- trace creation, ending, error reporting, and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe
from orchestrator.observability.error_reporter import report_error
from orchestrator.observability.trace_context import (
    clear_trace_context,
    get_current_trace_client,
    get_current_trace_id,
    set_trace_context,
    truncate_data,
)

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import AgentResponse, RunContext, RunState
    from orchestrator.observability.metrics import MetricsCollector

logger = get_logger(__name__)


class RunLifecycle:
    """Manages the lifecycle of an agent run (trace start/end, metrics)."""

    async def start_trace(
        self,
        agent: BaseAgent,
        context: RunContext,
        run_state: RunState,
        input_preview: str = "",
    ) -> None:
        """Create trace and set trace context for the run."""
        try:
            existing_trace_id = get_current_trace_id()
            if existing_trace_id:
                context.trace_id = existing_trace_id
                trace_client = get_current_trace_client()
                if trace_client:
                    context._langfuse_trace = trace_client
                logger.debug(f"Using existing trace context: {existing_trace_id}")
            elif not context.trace_id:
                from orchestrator.observability import TracingManager

                tracing_manager = TracingManager()
                trace = tracing_manager.create_trace(
                    name=f"agent-run-{agent.name}",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    input=truncate_data({"query": input_preview[:500]}),
                    metadata={
                        "run_id": context.run_id,
                        "agent_name": agent.name,
                        "model": agent.model,
                        "max_turns": context.max_turns,
                    },
                    tags=context.tags + agent.tags,
                )
                if trace:
                    context.trace_id = trace.id
                    context._langfuse_trace = trace.langfuse_trace

            set_trace_context(
                trace_id=context.trace_id,
                trace_client=getattr(context, "_langfuse_trace", None),
                user_id=context.user_id,
                session_id=context.session_id,
                agent_name=agent.name,
                run_id=context.run_id,
            )

            logger.debug(f"Trace context set: trace_id={context.trace_id}")

        except Exception as e:
            logger.warning(f"Failed to set trace context: {e}")

    @observe(name="trace_run_end", capture_output=False)
    async def end_trace(
        self,
        agent: BaseAgent,
        context: RunContext,
        response: AgentResponse,
    ) -> None:
        """Update trace with final output and clear trace context."""
        try:
            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                try:
                    trace.update(
                        output=truncate_data(
                            {
                                "response": response.content[:1000] if response.content else None,
                                "status": response.status.value,
                            }
                        ),
                        metadata={
                            "run_id": context.run_id,
                            "agent_name": agent.name,
                            "status": response.status.value,
                            "turn_count": response.turn_count,
                            "latency_ms": response.latency_ms,
                            "agents_used": response.agents_used,
                            "handoff_chain": response.handoff_chain,
                            "usage": response.usage.to_dict() if response.usage else {},
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to update trace: {e}")

            clear_trace_context()
            logger.debug(f"Trace context cleared for trace_id={context.trace_id}")

        except Exception as e:
            logger.warning(f"Failed to trace run end: {e}")
            try:
                clear_trace_context()
            except Exception:
                pass

    @observe(name="trace_run_error", capture_output=False)
    async def report_error(
        self,
        agent: BaseAgent,
        context: RunContext,
        error: Exception,
        run_state: RunState | None = None,
    ) -> None:
        """Report error to trace and clear trace context."""
        try:
            error_metadata: dict[str, Any] = {
                "run_id": context.run_id,
                "agent_name": agent.name,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "session_id": context.session_id,
                "user_id": context.user_id,
            }

            if run_state:
                error_metadata.update(
                    {
                        "current_turn": run_state.turn_count,
                        "agent_stack": run_state.agent_stack,
                        "handoff_chain": [h.get("to_agent") for h in run_state.handoff_chain],
                        "status": run_state.status.value if run_state.status else "unknown",
                    }
                )

            if agent.tools:
                error_metadata["available_tools"] = [
                    t.function.name
                    if hasattr(t, "function")
                    else t.get("function", {}).get("name", "")
                    for t in agent.tools[:10]
                ]

            report_error(
                error,
                context="agent_run",
                trace_id=context.trace_id,
                user_id=context.user_id,
                session_id=context.session_id,
                metadata=error_metadata,
            )

            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                try:
                    trace.update(
                        output={"error": str(error)[:500]},
                        level="ERROR",
                        status_message=str(error)[:200],
                    )
                except Exception:
                    pass

            clear_trace_context()

        except Exception as e:
            logger.warning(f"Failed to trace run error: {e}")
            try:
                clear_trace_context()
            except Exception:
                pass

    @observe(name="report_metrics", capture_output=False)
    async def report_metrics(
        self,
        context: RunContext,
        metrics: MetricsCollector,
    ) -> None:
        """Report collected metrics to the Langfuse trace."""
        try:
            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                metrics.report_to_trace(trace)
                logger.debug(f"Metrics reported to trace: {context.trace_id}")
        except Exception as e:
            logger.warning(f"Failed to report metrics to trace: {e}")
