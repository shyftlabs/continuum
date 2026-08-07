"""
Tool Service - Handles tool execution for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from continuum.agent.exceptions import AgentToolError
from continuum.agent.interfaces.service_interface import IToolService
from continuum.agent.types import ToolExecutionSummary
from continuum.logging import get_logger
from continuum.observability.metrics import get_metrics_collector
from continuum.observability.trace_context import SpanScope, truncate_data

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import RunnerConfig
    from continuum.agent.types import RunContext
    from continuum.llm.types import ToolCallInput

logger = get_logger(__name__)

# The separator build_namespaced_tool_name() puts between server and tool.
_NAMESPACE_SEPARATOR = "__"


def resolve_tool_data_labels(tool_data_labels: dict[str, set[str]], called_name: str) -> set[str]:
    """Labels declared for ``called_name``, by namespaced or raw key.

    ``tool_data_labels`` is keyed by tool name, but a server given a ``name=``
    delivers every tool as ``<server>__<tool>``. Matching only the exact string
    meant a raw declaration missed, no taint was applied, and every protection
    keyed on that taint silently stopped applying. Observed live: a tool
    returning an SSN left the run reporting ``clean``, so the rule denying
    exfiltration tools to a PHI run never fired and a summary was emailed
    outside with no gate tripped.

    Both spellings are accepted because neither alone is right:

    * the **raw** name is what a user knows -- it is in their server's source --
      and is the only usable form when no ``name=`` was given, since the derived
      one is ``streamable_http_http_localhost_8911_mcp__lookup_patient`` and
      changes whenever the URL does;
    * the **namespaced** name is needed when two servers expose the same tool
      name, and so wins when both are declared.

    Telling users to write namespaced keys instead would fix today and break on
    the next rename -- the same silent failure, restored. ToolContextConfig
    already compares raw names for exactly this reason.

    Only a whole trailing segment counts: ``lookup_patient`` must not match
    ``clinic__bulk_lookup_patient``. Over-tainting is safe, but labelling the
    wrong tool makes every declaration untrustworthy.
    """
    if not tool_data_labels:
        return set()
    exact = tool_data_labels.get(called_name)
    if exact:
        return set(exact)
    if _NAMESPACE_SEPARATOR in called_name:
        raw = called_name.rsplit(_NAMESPACE_SEPARATOR, 1)[-1]
        return set(tool_data_labels.get(raw) or ())
    return set()


def _matching_registry_keys(registry: dict[str, Any] | None, declared: str) -> list[str]:
    """Registry entries a declared label name applies to."""
    if not registry:
        return []
    return sorted(
        key
        for key in registry
        if key == declared or key.rsplit(_NAMESPACE_SEPARATOR, 1)[-1] == declared
    )


class ToolService(IToolService):
    """
    Service for tool execution.

    Handles executing individual tools and batches of tools,
    with support for parallel execution.
    """

    def __init__(
        self,
        tool_executor: Any | None = None,
        config: RunnerConfig | None = None,
    ):
        """
        Initialize tool service.

        Args:
            tool_executor: Tool executor instance
            config: Runner configuration
        """
        self._tool_executor = tool_executor
        self._config = config

    @property
    def tool_executor(self) -> Any | None:
        """Get tool executor."""
        return self._tool_executor

    def _warn_on_unresolvable_data_labels(self, agent: BaseAgent) -> None:
        """Report ``tool_data_labels`` entries that name no tool, or too many.

        A declaration matching nothing protects nothing and says nothing --
        the same silent-no-op that
        ``ToolExecutor._warn_on_unmatched_context_tool_names`` exists to catch
        for ``capture_from`` / ``inject_into``, guarding something more
        important. It is what a rename leaves behind.

        A raw name matching several servers still taints all of them, which
        fails closed and is therefore safe. It is reported anyway: labelling an
        unrelated server's tool PHI blocks benign work, and the natural
        response to that is to loosen the policy rather than to disambiguate.

        Once per agent -- this runs on every batch, and a warning repeated each
        turn is one people filter out.
        """
        if getattr(agent, "_warned_unresolvable_data_labels", False):
            return
        registry = getattr(getattr(agent, "tool_executor", None), "tool_registry", None)
        if not registry:
            return  # nothing to compare against yet
        try:
            agent._warned_unresolvable_data_labels = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass  # a frozen/slotted agent just gets the warning again

        unmatched: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for declared in agent.config.tool_data_labels:
            matches = _matching_registry_keys(registry, declared)
            if not matches:
                unmatched.append(declared)
            elif len(matches) > 1:
                ambiguous[declared] = matches

        if unmatched:
            logger.warning(
                f"Agent '{agent.name}' declares data labels for {sorted(unmatched)}, which "
                f"match no tool on any connected server, so those labels are never applied "
                f"and anything gated on them silently stops applying. Known tools: "
                f"{sorted(registry)}"
            )
        for declared, matches in sorted(ambiguous.items()):
            logger.warning(
                f"Agent '{agent.name}' declares data labels for '{declared}', which matches "
                f"{matches} on more than one server -- all of them are labelled. Use the "
                f"namespaced name to label only the one you mean."
            )

    async def execute_tool_call(
        self,
        agent: BaseAgent,
        tool_call: ToolCallInput,
        context: RunContext,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute a single tool call with full tracing and metrics.

        Returns:
            Tuple of (tool_result_message, execution_metadata)
            - tool_result_message: The ChatMessage dict for the tool result
            - execution_metadata: Dict with tool_name, latency_ms, server_name, success, error
        """
        tool_name = (
            tool_call.function.name
            if hasattr(tool_call, "function")
            else tool_call.get("function", {}).get("name", "")
        )
        tool_args_str = (
            tool_call.function.arguments
            if hasattr(tool_call, "function")
            else tool_call.get("function", {}).get("arguments", "{}")
        )
        tool_call_id = tool_call.id if hasattr(tool_call, "id") else tool_call.get("id", "")

        try:
            tool_args = (
                json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            )
        except json.JSONDecodeError as e:
            logger.warning(
                f"Malformed JSON in tool arguments for '{tool_name}': {e}. "
                f"Raw args: {str(tool_args_str)[:200]}. Proceeding with empty args.",
            )
            tool_args = {}

        # Hallucination gate: if tool-attention is active and this tool was not promoted,
        # return a structured error so the LLM can recover by choosing an available tool.
        promoted: set[str] | None = (
            context.metadata.get("promoted_tools") if context.metadata else None
        )
        if promoted is not None and tool_name not in promoted:
            logger.warning(
                "tool-attention gate: '%s' not in promoted set %s",
                tool_name,
                sorted(promoted),
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {
                        "error": "tool_not_available",
                        "message": f"Tool '{tool_name}' is not available for this turn.",
                        "available": sorted(promoted),
                    }
                ),
            }, {
                "tool_name": tool_name,
                "latency_ms": 0.0,
                "server_name": None,
                "success": False,
                "error": "tool_not_in_promoted_set",
            }

        start_time = time.time()
        metrics = get_metrics_collector()

        # Initialize execution metadata
        exec_metadata: dict[str, Any] = {
            "tool_name": tool_name,
            "latency_ms": 0.0,
            "server_name": None,
            "success": False,
            "error": None,
        }

        # Create span for tool execution with full details
        async with SpanScope(
            f"tool.{tool_name}",
            input=truncate_data(
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": tool_args,
                }
            ),
            metadata={
                "agent_name": agent.name,
                "tool_type": (
                    "mcp" if hasattr(agent, "mcp_servers") and agent.mcp_servers else "custom"
                ),
            },
        ) as span:
            # Log tool call for debugging
            logger.info(
                f"🔧 TOOL CALL: {tool_name}",
                extra={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_call_id": tool_call_id,
                },
            )
            logger.debug(f"  Arguments: {json.dumps(tool_args, indent=2)}")

            # Run tool hook
            if agent.on_tool_call:
                agent.on_tool_call(agent, tool_name, tool_args)

            # Build a ToolCall-like object for the executor
            from continuum.llm.types import FunctionCall
            from continuum.llm.types import ToolCall as LLMToolCall

            tc_obj = LLMToolCall(
                id=tool_call_id,
                type="function",
                function=FunctionCall(
                    name=tool_name,
                    arguments=(
                        tool_args_str if isinstance(tool_args_str, str) else json.dumps(tool_args)
                    ),
                ),
            )

            # Try agent's tool executor first
            agent_policy_store = getattr(agent, "policy_store", None)
            _agent_executor_failed = False
            if agent.tool_executor:
                try:
                    # Get server name from tool registry if available
                    if (
                        hasattr(agent.tool_executor, "tool_registry")
                        and tool_name in agent.tool_executor.tool_registry
                    ):
                        server, _ = agent.tool_executor.tool_registry[tool_name]
                        exec_metadata["server_name"] = server.name

                    results = await agent.tool_executor.execute_tool_calls(
                        tool_calls=[tc_obj],
                        trace_id=context.trace_id,
                        policy_store=agent_policy_store,
                        subject=agent.name if agent_policy_store else None,
                        data_labels=context.data_labels if agent_policy_store else None,
                    )
                    if results:
                        result = self._message_to_dict(results[0])
                        # Tool provenance: a declared tool's result taints the run.
                        if agent.config:
                            labels = resolve_tool_data_labels(
                                agent.config.tool_data_labels, tool_name
                            )
                            if labels:
                                context.taint(*labels)
                        latency_ms = (time.time() - start_time) * 1000

                        # Log tool result
                        result_preview = str(result.get("content", ""))[:200]
                        logger.info(f"✅ TOOL RESULT: {tool_name} -> {result_preview}...")

                        # Update span with result
                        span.set_output(truncate_data(result))
                        span.add_metadata("latency_ms", round(latency_ms, 2))
                        span.add_metadata("success", True)

                        # Record tool latency in metrics
                        metrics.record_latency(
                            f"tool_{tool_name}",
                            latency_ms,
                            metadata={"agent_name": agent.name, "success": True},
                        )

                        # Update execution metadata
                        exec_metadata["latency_ms"] = latency_ms
                        exec_metadata["success"] = True

                        return result, exec_metadata
                except Exception as e:
                    logger.warning(f"❌ TOOL ERROR: {tool_name} failed: {e}")
                    span.set_error(str(e))
                    metrics.track_error(f"tool_{tool_name}", e, metadata={"agent_name": agent.name})
                    exec_metadata["error"] = str(e)[:100]
                    _agent_executor_failed = True

            # Try global tool executor
            if self._tool_executor:
                _reason = (
                    "agent executor failed" if _agent_executor_failed else "agent has no executor"
                )
                logger.warning(
                    f"⚠️ TOOL FALLBACK: {tool_name} retrying on global executor ({_reason})"
                )
                try:
                    # Get server name from tool registry if available
                    if (
                        hasattr(self._tool_executor, "tool_registry")
                        and tool_name in self._tool_executor.tool_registry
                    ):
                        server, _ = self._tool_executor.tool_registry[tool_name]
                        exec_metadata["server_name"] = server.name

                    results = await self._tool_executor.execute_tool_calls(
                        tool_calls=[tc_obj],
                        trace_id=context.trace_id,
                        policy_store=agent_policy_store,
                        subject=agent.name if agent_policy_store else None,
                        data_labels=context.data_labels if agent_policy_store else None,
                    )
                    if results:
                        result = self._message_to_dict(results[0])
                        # Tool provenance: a declared tool's result taints the run.
                        if agent.config:
                            labels = resolve_tool_data_labels(
                                agent.config.tool_data_labels, tool_name
                            )
                            if labels:
                                context.taint(*labels)
                        latency_ms = (time.time() - start_time) * 1000

                        # Log tool result
                        result_preview = str(result.get("content", ""))[:200]
                        logger.info(f"✅ TOOL RESULT: {tool_name} -> {result_preview}...")

                        # Update span with result
                        span.set_output(truncate_data(result))
                        span.add_metadata("latency_ms", round(latency_ms, 2))
                        span.add_metadata("success", True)

                        # Record tool latency in metrics
                        metrics.record_latency(
                            f"tool_{tool_name}",
                            latency_ms,
                            metadata={"agent_name": agent.name, "success": True},
                        )

                        # Update execution metadata
                        exec_metadata["latency_ms"] = latency_ms
                        exec_metadata["success"] = True

                        return result, exec_metadata
                except Exception as e:
                    span.set_error(str(e))
                    span.add_metadata("success", False)
                    logger.error(f"❌ TOOL ERROR: {tool_name} failed: {e}")
                    metrics.track_error(f"tool_{tool_name}", e, metadata={"agent_name": agent.name})
                    exec_metadata["latency_ms"] = (time.time() - start_time) * 1000
                    exec_metadata["error"] = str(e)[:100]
                    raise AgentToolError(
                        f"Tool execution failed: {e}",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_name=agent.name,
                        run_id=context.run_id,
                        original_error=e,
                    ) from e

            # No executor available
            latency_ms = (time.time() - start_time) * 1000
            span.add_metadata("success", False)
            span.set_error(f"Tool '{tool_name}' not available")
            logger.warning(f"⚠️ NO EXECUTOR: Tool '{tool_name}' not available")

            exec_metadata["latency_ms"] = latency_ms
            exec_metadata["error"] = f"Tool '{tool_name}' not available"

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: Tool '{tool_name}' not available",
            }, exec_metadata

    async def execute_tools_batch(
        self,
        agent: BaseAgent,
        tool_calls: list[ToolCallInput],
        context: RunContext,
        tool_summary: ToolExecutionSummary | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute multiple tool calls, optionally in parallel.

        Respects runner config for parallel execution and max concurrency.
        Handles errors gracefully - failed tools return error messages instead of crashing.

        Args:
            agent: The agent making the tool calls
            tool_calls: List of tool calls to execute
            context: Run context
            tool_summary: Optional ToolExecutionSummary to collect metadata into

        Returns:
            List of tool result messages in the same order as input
        """
        if not tool_calls:
            return []

        # Provenance pre-taint: fold the DECLARED data labels of every tool in
        # this batch into the run BEFORE gating/executing any of them. Tool
        # provenance is declared up front (AgentConfig.tool_data_labels), so the
        # run is sensitive the moment such a tool is invoked — not only after it
        # returns. Without this, a batch like [lookup_patient (declares "phi"),
        # send_referral_email] would gate the exfiltration tool against the
        # pre-batch (clean) labels — under parallel execution the sibling's taint
        # lands too late — and a denied tool would slip through. Pre-tainting is
        # order- and concurrency-independent and fails safe (a declared tool that
        # errors/returns nothing still taints, which is the conservative choice).
        if agent.config and agent.config.tool_data_labels:
            self._warn_on_unresolvable_data_labels(agent)
            for tc in tool_calls:
                name = (
                    tc.function.name
                    if hasattr(tc, "function")
                    else tc.get("function", {}).get("name", "")
                )
                labels = resolve_tool_data_labels(agent.config.tool_data_labels, name)
                if labels:
                    context.taint(*labels)

        # Single tool - execute directly
        if len(tool_calls) == 1:
            try:
                result, exec_meta = await self.execute_tool_call(agent, tool_calls[0], context)
                # Add to summary if provided
                if tool_summary:
                    tool_summary.add_tool_execution(
                        tool_name=exec_meta["tool_name"],
                        latency_ms=exec_meta["latency_ms"],
                        server_name=exec_meta.get("server_name"),
                        success=exec_meta["success"],
                        error=exec_meta.get("error"),
                    )
                return [result]
            except Exception as e:
                tc = tool_calls[0]
                tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                tool_name = (
                    tc.function.name
                    if hasattr(tc, "function")
                    else tc.get("function", {}).get("name", "unknown")
                )
                if tool_summary:
                    tool_summary.add_tool_execution(
                        tool_name=tool_name,
                        latency_ms=0,
                        success=False,
                        error=str(e)[:100],
                    )
                return [
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Tool execution error: {e}",
                    }
                ]

        # Check if parallel execution is enabled
        if not self._config or not self._config.parallel_tool_calls:
            # Sequential execution
            results = []
            for tc in tool_calls:
                try:
                    result, exec_meta = await self.execute_tool_call(agent, tc, context)
                    results.append(result)
                    if tool_summary:
                        tool_summary.add_tool_execution(
                            tool_name=exec_meta["tool_name"],
                            latency_ms=exec_meta["latency_ms"],
                            server_name=exec_meta.get("server_name"),
                            success=exec_meta["success"],
                            error=exec_meta.get("error"),
                        )
                except Exception as e:
                    tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                    tool_name = (
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "unknown")
                    )
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": f"Tool execution error: {e}",
                        }
                    )
                    if tool_summary:
                        tool_summary.add_tool_execution(
                            tool_name=tool_name,
                            latency_ms=0,
                            success=False,
                            error=str(e)[:100],
                        )
            return results

        # Parallel execution with semaphore for max concurrency
        max_parallel = self._config.max_parallel_tools if self._config else 5
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(
            tc: ToolCallInput,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            """Execute a single tool with semaphore control."""
            async with semaphore:
                try:
                    return await self.execute_tool_call(agent, tc, context)
                except Exception as e:
                    tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                    tool_name = (
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "unknown")
                    )
                    logger.warning(f"Tool '{tool_name}' failed in parallel batch: {e}")
                    return (
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": f"Tool execution error: {e}",
                        },
                        {
                            "tool_name": tool_name,
                            "latency_ms": 0,
                            "server_name": None,
                            "success": False,
                            "error": str(e)[:100],
                        },
                    )

        # Execute all tools in parallel (limited by semaphore)
        logger.debug(
            f"Executing {len(tool_calls)} tools in parallel (max {max_parallel} concurrent)"
        )

        results_with_meta = await asyncio.gather(
            *[execute_with_semaphore(tc) for tc in tool_calls],
            return_exceptions=False,  # Exceptions handled inside execute_with_semaphore
        )

        # Extract results and add metadata to summary
        results = []
        for result, exec_meta in results_with_meta:
            results.append(result)
            if tool_summary:
                tool_summary.add_tool_execution(
                    tool_name=exec_meta["tool_name"],
                    latency_ms=exec_meta["latency_ms"],
                    server_name=exec_meta.get("server_name"),
                    success=exec_meta["success"],
                    error=exec_meta.get("error"),
                )

        return results

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert a message to dictionary format."""
        if isinstance(message, dict):
            return message
        if hasattr(message, "to_dict"):
            return message.to_dict()
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"role": "user", "content": str(message)}
