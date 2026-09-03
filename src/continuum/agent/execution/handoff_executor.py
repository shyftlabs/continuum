"""
Handoff Executor - Handles agent-to-agent handoffs.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from continuum.agent.handoff.manager import HandoffManager
from continuum.agent.interfaces.handler_interface import IHandoffExecutor
from continuum.agent.types import HandoffResult, generate_handoff_id
from continuum.logging import get_logger
from continuum.observability.decorators import observe

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.types import RunContext, RunState

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

    @staticmethod
    def _scan_handoff_payload(
        target_agent: BaseAgent,
        handoff_data: Any,
        target_messages: list[dict[str, Any]],
    ) -> str | None:
        """Apply the target agent's input guards to an inbound handoff payload.

        Returns the blocking reason, or ``None`` if the payload is allowed.

        Scans the model-authored framing (``reason`` / ``context``) and the
        transferred history body together — that is the whole of what the target
        did not write itself. Mirrors MessageBuilder.prepare_messages, using the
        same config flags, so no new settings are introduced: `injection_detection`
        still defaults False and `input_scanners` still defaults empty.

        Returning a reason rather than raising InputBlockedError is deliberate: the
        caller reports failures as HandoffResult objects, and an exception escaping
        here would crash the run instead of failing the transfer cleanly.
        """
        config = getattr(target_agent, "config", None)
        if config is None:
            return None

        # str() everything: `reason` and `context` come from json.loads() of
        # MODEL-generated arguments, so a model emitting {"reason": 123} or a dict
        # would otherwise crash the join. Declared str on HandoffData, but nothing
        # validates the parse.
        parts = [str(handoff_data.reason or ""), str(handoff_data.context or "")]
        parts.extend(
            str(m.get("content") or "")
            for m in target_messages
            if isinstance(m, dict) and m.get("role") != "system"
        )
        payload = "\n".join(p for p in parts if p)
        if not payload:
            return None

        if getattr(config, "injection_detection", False):
            from continuum.utils import detect_injection_patterns

            detected = detect_injection_patterns(payload)
            if detected:
                # Detection-only, matching prepare_messages: the scanners below are
                # what can actually refuse. Logged so the signal is not lost.
                logger.warning(
                    "Potential prompt injection detected in handoff payload to agent '%s': %s",
                    target_agent.name,
                    detected,
                )

        for scanner in getattr(config, "input_scanners", None) or []:
            try:
                _, is_safe, reason = scanner(payload)
            except Exception as e:
                # Fail-open on scanner errors, matching prepare_messages — a broken
                # scanner must not take down every handoff.
                logger.warning("Handoff input scanner failed (fail-open): %s", e)
                continue
            if not is_safe:
                return reason or "blocked"

        return None

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

        # Data-label handoff gate (security finding F4).
        #
        # The run's active labels are passed as additional subjects, exactly like
        # the tool gate in tools/executor.py, so a tainted run can be denied a
        # transfer: "phi may not be handed to external-summarizer".
        #
        # Uses the SOURCE agent's store, while it is still the one in scope. That
        # matters because a handoff is a policy-domain switch -- ToolService reads
        # `policy_store` off whichever agent is currently running, so after the
        # transfer the target's store (possibly None) governs instead. Without a
        # check here, denying a tool on the source is escapable by handing off to
        # an agent that does not deny it.
        #
        # No policy_store configured -> no check, same posture as the tool gate.
        source_policy_store = getattr(agent, "policy_store", None)
        if source_policy_store is not None:
            subjects = [agent.name, *sorted(context.data_labels)]
            decision = source_policy_store.check(subjects, f"handoff:{target_name}")
            if not decision.allowed:
                denial = (
                    decision.denial_message
                    or f"Handoff to '{target_name}' denied by policy "
                    f"'{decision.policy_name or 'unknown'}'"
                )
                logger.warning(
                    "Handoff denied by policy — from=%s to=%s labels=%s policy=%s",
                    agent.name,
                    target_name,
                    sorted(context.data_labels),
                    decision.policy_name,
                )
                return HandoffResult(
                    handoff_id=generate_handoff_id(),
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=denial,
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

            # Run the TARGET agent's own input guards over what it is about to
            # receive (security finding F4). A handoff reaches execute_loop without
            # passing through MessageBuilder.prepare_messages -- the only place
            # sanitization / injection detection / input_scanners run -- so an agent
            # configured with those controls had them bypassed on every handoff,
            # i.e. on its least trustworthy input.
            blocked = self._scan_handoff_payload(target_agent, handoff_data, target_messages)
            if blocked is not None:
                # Unwind the state pushed above so the aborted transfer leaves no
                # trace on the stack; the source agent continues its own loop.
                run_state.pop_agent()
                run_state.current_agent = agent.name
                logger.warning(
                    "Handoff payload blocked by target's input scanner — from=%s to=%s reason=%s",
                    agent.name,
                    target_name,
                    blocked,
                )
                return HandoffResult(
                    handoff_id=handoff_data.handoff_id,
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=f"Handoff payload blocked by scanner: {blocked}",
                )

            # Create new context for target
            from continuum.agent.types import RunContext

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
                # Share the decision-trace recorder so the target agent's own
                # LLM/tool steps are captured into the one DecisionTrace for this
                # run (the recorder is intentionally shared across handoffs).
                recorder=context.recorder,
                # Propagate the fork guard so handed-off sub-agents of a forked
                # (what-if) run also never write to long-term memory.
                disable_memory_writes=context.disable_memory_writes,
            )

            # Log request structure only. Handoff messages can contain user input,
            # retrieved knowledge, and tool results, so their contents must never
            # be copied into production logs.
            _tools = target_agent.get_tools_for_llm()
            known_roles = {"system", "developer", "user", "assistant", "tool", "function"}
            role_sequence = [
                role if (role := message.get("role")) in known_roles else "unknown"
                for message in target_messages
            ]
            tool_names = [
                tool.get("function", {}).get("name", "unknown")
                if isinstance(tool, dict)
                else getattr(getattr(tool, "function", None), "name", "unknown")
                for tool in _tools
            ]
            logger.info(
                "Prepared handoff LLM request: agent=%s message_count=%d "
                "role_sequence=%s tool_names=%s",
                target_agent.name,
                len(target_messages),
                role_sequence,
                tool_names,
            )

            # Execute target agent (executor guaranteed to be set by early validation).
            # The recipient agent runs one level below the top-level runner, so its
            # lifecycle hooks (on_start / on_end / on_error) are fired here — the
            # runner only fires them for the agent passed to run()/run_stream().
            # This keeps the audit trail complete across handoffs. Ordering and call
            # signatures mirror the top-level runner; on_tool_call already fires
            # inside the tool service for whichever agent the loop is driving.
            response = None
            try:
                if target_agent.on_start:
                    target_agent.on_start(
                        target_agent, {"context": target_context, "input": reason}
                    )
                response = await self._executor.execute_loop(
                    agent=target_agent,
                    messages=target_messages,
                    context=target_context,
                    run_state=run_state,
                )
            except Exception as e:
                logger.error(
                    "Failed to execute target agent: agent=%s error_type=%s",
                    target_name,
                    type(e).__name__,
                )
                if target_agent.on_error:
                    target_agent.on_error(target_agent, e, {"context": target_context})
                result = HandoffResult(
                    handoff_id=handoff_data.handoff_id,
                    from_agent=agent.name,
                    to_agent=target_name,
                    success=False,
                    error=str(e),
                )
                await self._handoff_manager.trace_handoff("end", handoff_data, context, result)
                return result

            # on_end fires only after a genuinely successful execution and OUTSIDE
            # the try above — a hook that raises must not mask the success as a
            # failure or double-fire on_error. (A raising on_end then propagates to
            # the outer handler, mirroring the top-level runner's hook handling.)
            if target_agent.on_end:
                target_agent.on_end(target_agent, {"context": target_context, "response": response})

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
                "Handoff failed: from_agent=%s to_agent=%s error_type=%s",
                agent.name,
                target_name,
                type(e).__name__,
            )
            return HandoffResult(
                handoff_id=generate_handoff_id(),
                from_agent=agent.name,
                to_agent=target_name,
                success=False,
                error=str(e),
            )
