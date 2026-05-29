"""
Handoff Executor - Handles agent-to-agent handoffs.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from orchestrator.agent.exceptions import HandoffError
from orchestrator.agent.handoff.manager import HandoffManager
from orchestrator.agent.interfaces.handler_interface import IHandoffExecutor
from orchestrator.agent.types import HandoffResult, generate_handoff_id
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import RunContext, RunState

logger = get_logger(__name__)


class HandoffExecutor(IHandoffExecutor):
    """
    Executor for agent handoffs.

    Handles preparing and executing handoffs to other agents.
    """

    def __init__(
        self,
        handoff_manager: HandoffManager | None = None,
        agent_registry: dict[str, BaseAgent] | None = None,
        executor: Any = None,  # Executor for recursive execution
    ):
        """
        Initialize handoff executor.

        Args:
            handoff_manager: Handoff manager instance
            agent_registry: Registry of available agents
            executor: Executor instance for recursive execution of target agent
        """
        self._handoff_manager = handoff_manager
        self._agent_registry = agent_registry or {}
        self._executor = executor

    def set_executor(self, executor: Any) -> None:
        """Set the executor instance for recursive execution of target agents."""
        self._executor = executor

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for handoffs."""
        self._agent_registry[agent.name] = agent

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a registered agent by name."""
        return self._agent_registry.get(name)

    @observe(name="execute_handoff", capture_output=True)
    async def execute_handoff(
        self,
        agent: BaseAgent,
        target_name: str,
        tool_call: Any,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> HandoffResult:
        """
        Execute a handoff to another agent.

        Args:
            agent: Source agent
            target_name: Target agent name
            tool_call: Tool call that triggered the handoff
            messages: Current conversation messages
            context: Run context
            run_state: Run state

        Returns:
            HandoffResult with execution outcome
        """
        if not self._handoff_manager:
            error_msg = (
                f"HandoffManager not initialized. Cannot execute handoff "
                f"from '{agent.name}' to '{target_name}'."
            )
            logger.error(error_msg)
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=error_msg,
            )

        # Fix #6: Validate executor is set before proceeding
        if not self._executor:
            error_msg = (
                f"Executor not set on HandoffExecutor. Call set_executor() before "
                f"executing handoffs. Handoff from '{agent.name}' to '{target_name}' aborted."
            )
            logger.error(error_msg)
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=error_msg,
            )

        # Fix #12: Check handoff depth BEFORE cycle check
        current_depth = len(run_state.agent_stack)
        max_depth = self._handoff_manager._max_depth
        if current_depth >= max_depth:
            error_msg = (
                f"Handoff depth limit reached ({current_depth}/{max_depth}). "
                f"Cannot hand off from '{agent.name}' to '{target_name}'."
            )
            logger.warning(error_msg)
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=error_msg,
            )

        # Get target agent - try to get from registry first
        target_agent = self.get_agent(target_name)

        # If not found, try to get from agent's handoff definition
        if target_agent is None:
            handoff_def = agent.get_handoff(target_name)
            if handoff_def:
                # Fix #13: Log clearly that agent is defined but not registered
                logger.error(
                    f"Handoff target '{target_name}' is defined in agent '{agent.name}' handoffs "
                    f"but not registered in the agent registry. Register the agent via "
                    f"runner.register_agent() or pass it in agent_registry."
                )
                return HandoffResult(
                    handoff_id=generate_handoff_id(),
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=f"Target agent '{target_name}' defined but not registered. "
                    f"Use runner.register_agent() to register it.",
                )
            else:
                logger.error(
                    f"Handoff target '{target_name}' not found: no handoff definition "
                    f"on agent '{agent.name}' and not in registry."
                )
                return HandoffResult(
                    handoff_id=generate_handoff_id(),
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=f"Target agent '{target_name}' not found and handoff not defined",
                )

        # Check for cycles in the handoff chain
        if self._handoff_manager.detect_cycle(run_state.agent_stack, target_name):
            cycle_path = " → ".join(run_state.agent_stack + [target_name])
            logger.warning(
                f"Handoff cycle detected: {agent.name} → {target_name}. "
                f"Agent '{target_name}' already in chain: {cycle_path}"
            )
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=f"Handoff cycle detected: {target_name} already in handoff chain ({cycle_path})",
            )

        # Parse handoff arguments
        args_str = (
            tool_call.function.arguments
            if hasattr(tool_call, "function")
            else tool_call.get("function", {}).get("arguments", "{}")
        )
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        reason = args.get("reason", "Handoff requested")
        additional_context = args.get("context")

        # Run handoff hook
        if agent.on_handoff:
            agent.on_handoff(agent, target_name, {"reason": reason, "context": additional_context})

        try:
            # Prepare handoff
            handoff_data = await self._handoff_manager.prepare_handoff(
                from_agent=agent,
                to_agent=target_agent,
                reason=reason,
                messages=messages,
                context=additional_context,
                run_context=context,
            )

            # Update run state (thread-safe)
            run_state.push_agent(target_name)
            run_state.handoff_chain.append(handoff_data.to_dict())
            run_state.current_agent = target_name

            # Trace handoff start
            await self._handoff_manager.trace_handoff("start", handoff_data, context)

            # Build messages for target agent
            target_messages = self._handoff_manager.build_handoff_messages(
                handoff_data, target_agent, session_id=context.session_id
            )

            # Create new context for target
            from orchestrator.agent.types import RunContext

            target_context = RunContext(
                run_id=context.run_id,
                session_id=context.session_id,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                trace_id=context.trace_id,
                agent_stack=run_state.agent_stack.copy(),
                max_turns=max(1, context.max_turns - run_state.turn_count),
                is_handoff=True,
                data_labels=context.data_labels.copy(),
            )

            # Log target agent details (mirrors message_builder output for top-level runs)
            mem_cfg = getattr(target_agent, "memory_config", None)
            if mem_cfg:
                logger.info(
                    f"🔍 HANDOFF TARGET MEMORY CONFIG [{target_agent.name}]: "
                    f"search_memories={mem_cfg.search_memories}, store_memories={mem_cfg.store_memories}, "
                    f"search_scope={getattr(mem_cfg, 'search_scope', 'N/A')}, "
                    f"store_scope={getattr(mem_cfg, 'store_scope', 'N/A')}"
                )
            logger.info(
                f"===== HANDOFF FINAL PROMPT [{target_agent.name}] =====\n"
                + "\n".join(
                    f"[{m.get('role','?')}] {str(m.get('content',''))[:300]}"
                    for m in target_messages
                )
                + "\n" + "=" * 30
            )
            _tools = target_agent.get_tools_for_llm()
            if _tools:
                _tools_formatted = "\n".join(
                    f"  - {t.get('function', {}).get('name', '?')}: {str(t.get('function', {}).get('parameters', ''))[:200]}"
                    for t in _tools
                )
                logger.info(
                    f"===== TOOLS [{target_agent.name}] =====\n{_tools_formatted}\n========================"
                )

            # Execute target agent (executor guaranteed to be set by early validation)
            response = None
            try:
                response = await self._executor.execute_loop(
                    agent=target_agent,
                    messages=target_messages,
                    context=target_context,
                    run_state=run_state,
                )
            except Exception as e:
                logger.error(f"Failed to execute target agent '{target_name}': {e}", exc_info=True)
                result = HandoffResult(
                    handoff_id=handoff_data.handoff_id,
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=str(e),
                )
                await self._handoff_manager.trace_handoff("end", handoff_data, context, result)
                return result

            # Trace handoff end
            result = HandoffResult(
                handoff_id=handoff_data.handoff_id,
                from_agent=agent.name,
                to_agent=target_name,
                success=True,
                response=response,
                returned_to_parent=False,
            )
            await self._handoff_manager.trace_handoff("end", handoff_data, context, result)

            return result

        except Exception as e:
            logger.error(
                f"Handoff from '{agent.name}' to '{target_name}' failed: {e}",
                exc_info=True,
            )
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=str(e),
            )
