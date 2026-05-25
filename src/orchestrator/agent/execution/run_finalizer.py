"""Run finalization -- post-execution tasks (session save, artifacts, metrics)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from orchestrator.agent.types import ResponseStatus, RunStatus
from orchestrator.logging import get_logger
from orchestrator.observability.metrics import get_metrics_collector

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.execution.run_lifecycle import RunLifecycle
    from orchestrator.agent.services.context_service import ContextService
    from orchestrator.agent.services.session_service import SessionService
    from orchestrator.agent.types import AgentResponse, RunContext, RunState

logger = get_logger(__name__)


class RunFinalizer:
    """Handles post-execution tasks (session save, artifacts, metrics)."""

    def __init__(
        self,
        session_service: SessionService,
        context_service: ContextService,
        lifecycle: RunLifecycle,
        tool_executor: Any = None,
        session_client: Any = None,
    ):
        self._session_service = session_service
        self._context_service = context_service
        self._lifecycle = lifecycle
        self._tool_executor = tool_executor
        self._session_client = session_client

    async def finalize(
        self,
        agent: BaseAgent,
        context: RunContext,
        run_state: RunState,
        response: AgentResponse,
        user_message_index: int,
        tool_context_state: Any,
        start_time: float,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Complete post-execution tasks."""
        metrics = get_metrics_collector()

        response.run_id = context.run_id
        response.latency_ms = int((time.time() - start_time) * 1000)
        response.trace_id = context.trace_id
        # Workflow agents bypass runner.run(), so agent_stack is always the right source.
        response.agents_used = list(set(run_state.agent_stack))
        response.handoff_chain = [h.get("to_agent", "") for h in run_state.handoff_chain]

        self.attach_run_artifacts(agent, response)

        # Run product output scanners (e.g. LLM Guard Sensitive/PII for TaxPilot).
        # Scanners redact PII in response.content before it is saved to session or returned.
        # Fail-open: if a scanner crashes the response is still returned unmodified.
        if agent.config and agent.config.output_scanners and response.content:
            prompt = ""
            if messages:
                # Use the last user message as the prompt context for output scanners
                for m in reversed(messages):
                    if m.get("role") == "user":
                        prompt = str(m.get("content", ""))
                        break
            for scanner in agent.config.output_scanners:
                try:
                    sanitized, _, _ = scanner(prompt, response.content)
                    response.content = sanitized
                except Exception as e:
                    logger.warning(
                        "Output scanner %s failed (fail-open): %s",
                        getattr(scanner, "__name__", repr(scanner)), e,
                    )

        run_state.status = RunStatus.COMPLETED
        await self._context_service.save_run_state(run_state)

        _, updated_context_state = self.track_mcp_session(agent, context)

        if response.status != ResponseStatus.MAX_TURNS_REACHED:
            await self.save_session_data(
                agent, context, user_message_index,
                tool_context_state, updated_context_state, messages,
            )

        e2e_latency_ms = (time.time() - start_time) * 1000
        metrics.record_latency(
            "agent_run_e2e",
            e2e_latency_ms,
            metadata={"agent_name": agent.name, "run_id": context.run_id},
        )

        if response.usage:
            metrics.track_tokens(
                "agent_run",
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                model=agent.model,
            )

        await self._lifecycle.report_metrics(context, metrics)
        await self._lifecycle.end_trace(agent, context, response)

    async def handle_error(
        self,
        agent: BaseAgent,
        context: RunContext,
        run_state: RunState,
        error: Exception,
        start_time: float,
    ) -> None:
        """Shared error handling for run() and run_stream()."""
        metrics = get_metrics_collector()
        metrics.track_error(
            "agent_run", error,
            metadata={"agent_name": agent.name, "run_id": context.run_id},
        )

        run_state.status = RunStatus.FAILED
        run_state.metadata["error"] = str(error)
        await self._context_service.save_run_state(run_state)

        if agent.on_error:
            agent.on_error(agent, error, {"context": context})

        await self._lifecycle.report_metrics(context, metrics)
        await self._lifecycle.report_error(agent, context, error, run_state)

    def attach_run_artifacts(self, agent: BaseAgent, response: AgentResponse) -> None:
        """Attach MCP artifacts to response (merge with existing e.g. model_tier routing)."""
        merged: dict[str, Any] = dict(response.run_artifacts or {})

        run_artifacts_dict = None
        if agent.tool_executor and hasattr(agent.tool_executor, "run_artifacts"):
            run_artifacts = agent.tool_executor.run_artifacts
            if not run_artifacts.is_empty():
                run_artifacts_dict = run_artifacts.to_dict()
        elif self._tool_executor and hasattr(self._tool_executor, "run_artifacts"):
            run_artifacts = self._tool_executor.run_artifacts
            if not run_artifacts.is_empty():
                run_artifacts_dict = run_artifacts.to_dict()

        if run_artifacts_dict:
            merged.update(run_artifacts_dict)
            logger.debug(
                f"Attached {len(run_artifacts_dict.get('tool_artifacts', []))} artifacts to response"
            )

        response.run_artifacts = merged if merged else None

    def track_mcp_session(
        self,
        agent: BaseAgent,
        context: RunContext,
    ) -> tuple[str | None, Any]:
        """Track MCP session from tool executors. Returns (mcp_session_id, updated_context_state)."""
        updated_context_state = None
        mcp_session_id = None

        if agent.tool_executor and hasattr(agent.tool_executor, "context_state"):
            updated_context_state = agent.tool_executor.context_state
            logger.debug("Using agent.tool_executor.context_state")
        elif self._tool_executor and hasattr(self._tool_executor, "context_state"):
            updated_context_state = self._tool_executor.context_state
            logger.debug("Using self._tool_executor.context_state")

        if updated_context_state:
            all_namespaces = updated_context_state.get_all_namespaces()
            logger.debug(
                f"Checking {len(all_namespaces)} namespaces for MCP session_id: {all_namespaces}"
            )
            for namespace in all_namespaces:
                captured_session_id = updated_context_state.get(namespace, "session_id")
                if captured_session_id:
                    mcp_session_id = captured_session_id
                    original = context.session_id
                    if mcp_session_id != original:
                        logger.debug(
                            f"Found MCP session_id (namespace={namespace}): {mcp_session_id[:8]}... "
                            f"(our session: {original[:8] if original else 'None'}...)"
                        )
                    break
        else:
            logger.debug("No context_state found on tool executors")

        return mcp_session_id, updated_context_state

    async def save_session_data(
        self,
        agent: BaseAgent,
        context: RunContext,
        user_message_index: int,
        tool_context_state: Any,
        updated_context_state: Any,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save messages and tool context to session."""
        original_session_id = context.session_id
        if not (agent.config.log_to_session and not context.suppress_session_log and original_session_id and self._session_client):
            return

        try:
            await self._session_service.save_messages(
                agent=agent,
                messages=messages or [],
                user_message_index=user_message_index,
                session_id=original_session_id,
                trace_id=context.trace_id,
                tool_execution_summary=context.metadata.get("tool_execution_summary"),
                run_id=context.run_id,
            )
        except Exception as e:
            logger.warning(f"Failed to save messages to session: {e}")

        final_context_state = updated_context_state or tool_context_state
        if final_context_state and not final_context_state.is_empty():
            try:
                await self._session_service.save_tool_context_state(
                    session_id=original_session_id,
                    context_state=final_context_state,
                    trace_id=context.trace_id,
                )
                logger.debug(f"Saved tool context to session {original_session_id[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to save tool context: {e}")
