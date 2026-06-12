"""
Agent Runner - Executes agents with full observability.

The main entry point for running agents, handling tool calls,
handoffs, and conversation loops.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from continuum.agent.base import BaseAgent
from continuum.agent.config import RunnerConfig
from continuum.agent.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    MaxTurnsExceededError,
)
from continuum.agent.execution.executor import Executor, _enrich_config_for_gateway
from continuum.agent.execution.handoff_executor import HandoffExecutor
from continuum.agent.execution.message_builder import MessageBuilder
from continuum.agent.execution.run_finalizer import RunFinalizer
from continuum.agent.execution.run_lifecycle import RunLifecycle
from continuum.agent.execution.stream_executor import StreamExecutor
from continuum.agent.execution.tool_handler import ToolHandler
from continuum.agent.execution.trace_capture import (
    capture_snapshot,
    latest_step_payload,
    record_llm_turn,
    record_tool_steps,
)
from continuum.agent.handoff.manager import HandoffManager
from continuum.agent.persistence.state import RunStateManager
from continuum.agent.services.context_service import ContextService
from continuum.agent.services.memory_service import MemoryService
from continuum.agent.services.session_service import SessionService
from continuum.agent.services.tool_service import ToolService
from continuum.agent.smart_layer.runner_facade import (
    extract_last_user_text,
    run_model_tier_turn,
    stream_model_tier_turn,
)
from continuum.agent.smart_layer.types import parse_product_tier, tier_dispatch_priority
from continuum.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    generate_run_id,
)
from continuum.agent.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.utils.message_utils import message_to_dict
from continuum.agent.utils.validation_utils import validate_input
from continuum.agent.workflow.router import RouterAgent
from continuum.config import settings
from continuum.config import settings as app_settings
from continuum.core.container import Container, get_container
from continuum.llm.config import LLMConfig
from continuum.logging import get_logger
from continuum.tools.tool_attention.router import _tool_name
from continuum.utils.sanitization import (
    InvalidIdentifierError,
    validate_conversation_id,
    validate_user_id,
)

if TYPE_CHECKING:
    from continuum.llm import LLMClient
    from continuum.llm.types import ChatMessage
    from continuum.memory import MemoryClient
    from continuum.observability import TracingManager
    from continuum.session import SessionClient
    from continuum.tools import ToolExecutor

logger = get_logger(__name__)


class AgentRunner:
    """
    Executes agents with full observability.

    Example:
        ```python
        runner = AgentRunner()
        response = await runner.run(agent, "Hello!", user_id="user-123")
        ```
    """

    def __init__(
        self,
        container: Container | None = None,
        llm_client: LLMClient | None = None,
        memory_client: MemoryClient | None = None,
        session_client: SessionClient | None = None,
        tool_executor: ToolExecutor | None = None,
        tracing_manager: TracingManager | None = None,
        state_manager: RunStateManager | None = None,
        config: RunnerConfig | None = None,
        agent_registry: dict[str, BaseAgent] | None = None,
    ):
        self._container = container or get_container()

        self._llm_client = llm_client or self._container.llm_client
        self._memory_client = memory_client or self._container.memory_client
        self._session_client = session_client or self._container.session_client
        self._tool_executor = tool_executor or self._container.tool_executor
        self._tracing_manager = tracing_manager
        self._state_manager = state_manager
        self._config = config or RunnerConfig()
        self._agent_registry = agent_registry or {}
        self._circuit_breaker = CircuitBreaker(
            threshold=self._config.circuit_breaker_threshold,
            cooldown=self._config.circuit_breaker_cooldown,
        )

        self._handoff_manager = HandoffManager(
            llm_client=self._llm_client,
            tracing_manager=self._tracing_manager,
        )

        # Services
        self._context_service = ContextService(
            state_manager=self._state_manager,
            config=self._config,
        )
        self._memory_service = MemoryService(
            memory_client=self._memory_client,
            session_client=self._session_client,
        )
        self._session_service = SessionService(
            session_client=self._session_client,
        )
        self._tool_service = ToolService(
            tool_executor=self._tool_executor,
            config=self._config,
        )

        # Lifecycle and finalization (extracted from runner)
        self._lifecycle = RunLifecycle()
        self._finalizer = RunFinalizer(
            session_service=self._session_service,
            context_service=self._context_service,
            lifecycle=self._lifecycle,
            tool_executor=self._tool_executor,
            session_client=self._session_client,
        )

        # Execution components
        self._handoff_executor = HandoffExecutor(
            handoff_manager=self._handoff_manager,
            agent_registry=self._agent_registry,
        )
        self._tool_handler = ToolHandler(tool_service=self._tool_service)
        self._executor = Executor(
            llm_client=self._llm_client,
            tool_handler=self._tool_handler,
            handoff_executor=self._handoff_executor,
        )
        self._handoff_executor.set_executor(self._executor)

        for agent_obj in self._agent_registry.values():
            self._handoff_executor.register_agent(agent_obj)

        # Lock for clearing run artifacts safely across concurrent runs
        self._artifact_lock = asyncio.Lock()

        self._stream_executor = StreamExecutor(llm_client=self._llm_client)
        self._message_builder = MessageBuilder(
            memory_service=self._memory_service,
            session_service=self._session_service,
        )

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for handoffs."""
        self._agent_registry[agent.name] = agent
        if self._handoff_executor:
            self._handoff_executor.register_agent(agent)

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a registered agent by name."""
        return self._agent_registry.get(name)

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        agent: BaseAgent | None = None,
    ) -> None:
        """
        Save exactly one conversation turn (user query + final response) to session.

        Use this in sequential multi-agent pipelines where intermediate agents
        run without a session_id (or with log_to_session=False), then call this
        once after the pipeline completes to write only what the user sees to Redis.

        Example:
            response_a = await runner.run(agent_a, user_query)
            response_b = await runner.run(agent_b, response_a.content)
            await runner.save_turn(session_id, user_query, response_b.content, agent=agent_b)

        Args:
            session_id: Session to write to.
            user_message: Original user query shown in the chat window.
            assistant_message: Final response shown in the chat window.
            agent: Agent whose memory config governs fact extraction.
                   If None, memory storage is skipped.
        """
        if not self._session_client or not self._session_client.is_enabled:
            return

        from continuum.llm.types import ChatMessage

        memory_config = getattr(agent, "memory_config", None)
        agent_id = agent.name if agent else None
        should_store = bool(memory_config and memory_config.store_memories)
        extraction_prompt = getattr(memory_config, "extraction_prompt", None)
        pre_store_filter = getattr(memory_config, "pre_store_filter", None)
        on_stored = getattr(memory_config, "on_stored", None)

        for role, content in (("user", user_message), ("assistant", assistant_message)):
            await self._session_client.add_message(
                session_id=session_id,
                message=ChatMessage(role=role, content=content),
                agent_id=agent_id,
                store_in_memory=should_store,
                extraction_prompt=extraction_prompt,
                pre_store_filter=pre_store_filter,
                on_stored=on_stored,
            )

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def memory_client(self) -> MemoryClient | None:
        return self._memory_client

    @property
    def session_client(self) -> SessionClient | None:
        return self._session_client

    @property
    def state_manager(self) -> RunStateManager:
        return self._context_service.state_manager

    # =========================================================================
    # Run preparation
    # =========================================================================

    async def _prepare_run(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[ChatMessage],
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        context: RunContext | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> PrepareRunResult:
        """Prepare for agent run -- shared setup for run() and run_stream()."""
        if agent.mcp_servers and not agent.tool_executor:
            raise AgentConfigurationError(
                f"Agent '{agent.name}' has mcp_servers set but no tool_executor. "
                "mcp_servers alone does nothing — you must also build a ToolExecutor "
                "and pass it via tool_executor=. Example:\n"
                "  executor = ToolExecutor(tool_registry={server: None})\n"
                "  await executor.initialize()\n"
                "  agent = BaseAgent(..., tool_executor=executor, tools=executor.get_tool_definitions())",
                agent_name=agent.name,
                config_key="mcp_servers",
            )

        if agent.name not in self._agent_registry:
            self.register_agent(agent)

        # Validate the scope identifiers at the SDK boundary. This is the one
        # place every runner.run()/run_stream() call passes through, so it
        # guards memory scoping regardless of which app calls in. Validation
        # runs for a caller-supplied context too — otherwise a hand-built
        # RunContext could smuggle raw, unvalidated ids past this check.
        try:
            if context is None:
                context = create_run_context(
                    session_id=session_id,
                    conversation_id=validate_conversation_id(conversation_id),
                    user_id=validate_user_id(user_id),
                    trace_id=trace_id,
                    max_turns=max_turns or agent.config.max_turns,
                    metadata=metadata or {},
                    tags=tags or [],
                )
            else:
                context.user_id = validate_user_id(context.user_id)
                context.conversation_id = validate_conversation_id(context.conversation_id)
        except InvalidIdentifierError as e:
            return PrepareRunResult(
                success=False,
                error_response=AgentResponse(
                    content=str(e),
                    agent_name=agent.name,
                    status=ResponseStatus.ERROR,
                    error=str(e),
                ),
            )

        if agent.input_schema is not None:
            validation_result = await validate_input(agent, input, context)
            if validation_result is not None:
                return PrepareRunResult(
                    success=False,
                    error_response=AgentResponse(
                        content="Input validation failed",
                        agent_name=agent.name,
                        status=ResponseStatus.ERROR,
                        error="Input validation failed",
                    ),
                )

        # Attach a decision-trace recorder (once per run). Reuse an existing one
        # so handoffs/workflow branches that share this context contribute to a
        # single trace spanning the whole run.
        if context.recorder is None:
            from continuum.agent.trace.config import checkpoint_enabled, is_trace_enabled

            if is_trace_enabled():
                from continuum.agent.trace.recorder import TraceRecorder

                query = input if isinstance(input, str) else str(input)
                context.recorder = TraceRecorder(
                    context.run_id, agent.name, query, checkpoint=checkpoint_enabled()
                )

        run_state = await self._context_service.create_run_state(agent, context)
        input_preview = input if isinstance(input, str) else str(input)[:500]
        await self._lifecycle.start_trace(agent, context, run_state, input_preview)

        # Caller is responsible for creating the session before calling runner.run().
        tool_context_state = None
        if context.session_id and self.session_client:
            tool_context_state = await self._session_service.load_tool_context_state(
                session_id=context.session_id,
                trace_id=context.trace_id,
            )

            if not tool_context_state.is_empty():
                all_namespaces = tool_context_state.get_all_namespaces()
                for namespace in all_namespaces:
                    mcp_session_id = tool_context_state.get(namespace, "session_id")
                    if mcp_session_id:
                        logger.info(
                            f"Loaded MCP session_id from tool context: {mcp_session_id[:8]}... "
                            f"(namespace={namespace})"
                        )
                        break

            if agent.tool_executor and hasattr(agent.tool_executor, "context_state"):
                agent.tool_executor.context_state = tool_context_state
            if self._tool_executor and hasattr(self._tool_executor, "context_state"):
                self._tool_executor.context_state = tool_context_state

        # Synchronize artifact clearing to prevent races between concurrent runs
        async with self._artifact_lock:
            if agent.tool_executor and hasattr(agent.tool_executor, "clear_run_artifacts"):
                agent.tool_executor.clear_run_artifacts(run_id=context.run_id)
            if self._tool_executor and hasattr(self._tool_executor, "clear_run_artifacts"):
                self._tool_executor.clear_run_artifacts(run_id=context.run_id)

        if agent.on_start:
            agent.on_start(agent, {"context": context, "input": input})

        messages, user_message_index = await self._message_builder.prepare_messages(
            agent,
            input,
            context,
            tool_context_state=tool_context_state,
        )
        run_state.messages = [message_to_dict(m) for m in messages]

        return PrepareRunResult(
            success=True,
            context=context,
            run_state=run_state,
            user_message_index=user_message_index,
            tool_context_state=tool_context_state,
        )

    # =========================================================================
    # Run (non-streaming)
    # =========================================================================

    async def run(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[ChatMessage],
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        context: RunContext | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AgentResponse:
        """Run an agent to completion."""
        start_time = time.time()

        result = await self._prepare_run(
            agent,
            input,
            session_id,
            conversation_id,
            user_id,
            context,
            max_turns,
            trace_id,
            metadata,
            tags,
        )
        if not result.success:
            return result.error_response

        ctx = result.context
        run_state = result.run_state

        try:
            self._circuit_breaker.check()
        except CircuitBreakerOpen as e:
            logger.error(f"Circuit breaker open for agent '{agent.name}': {e}")
            return AgentResponse(
                content=f"Service temporarily unavailable: {e}",
                agent_name=agent.name,
                status=ResponseStatus.ERROR,
                error=str(e),
            )

        try:
            messages = list(run_state.messages) if run_state.messages else []

            if (
                isinstance(agent, RouterAgent)
                and app_settings.smart_layer_enabled
                and agent.router_config.routing_strategy == "model_tier"
            ):
                user_text = extract_last_user_text(input, messages)
                try:
                    mt_result = await run_model_tier_turn(
                        agent,
                        self._llm_client,
                        user_text=user_text,
                        ctx=ctx,
                    )
                except Exception as e_mt:
                    self._circuit_breaker.record_failure()
                    await self._finalizer.handle_error(agent, ctx, run_state, e_mt, start_time)
                    if isinstance(e_mt, AgentError):
                        raise
                    raise AgentExecutionError(
                        str(e_mt),
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        trace_id=ctx.trace_id,
                        original_error=e_mt,
                    ) from e_mt

                te = parse_product_tier(mt_result.routing.get("tier"))
                if te:
                    ctx.priority = tier_dispatch_priority(te)

                response = AgentResponse(
                    content=mt_result.content,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    trace_id=ctx.trace_id,
                    run_artifacts={"routing": mt_result.routing},
                )
                out_messages = list(messages)
                if mt_result.content:
                    out_messages.append({"role": "assistant", "content": mt_result.content})

                if agent.on_end:
                    agent.on_end(agent, {"context": ctx, "response": response})

                await self._finalizer.finalize(
                    agent,
                    ctx,
                    run_state,
                    response,
                    result.user_message_index,
                    result.tool_context_state,
                    start_time,
                    out_messages,
                )

                self._circuit_breaker.record_success()
                return response

            # Workflow agents must use agent.execute() directly, not runner.run().
            response = await self._executor.execute_loop(
                agent=agent,
                messages=messages,
                context=ctx,
                run_state=run_state,
            )

            if agent.on_end:
                agent.on_end(agent, {"context": ctx, "response": response})

            await self._finalizer.finalize(
                agent,
                ctx,
                run_state,
                response,
                result.user_message_index,
                result.tool_context_state,
                start_time,
                response.messages,
            )

            self._circuit_breaker.record_success()
            return response

        except Exception as e:
            if isinstance(e, MaxTurnsExceededError):
                partial = AgentResponse(
                    content="",
                    agent_name=agent.name,
                    status=ResponseStatus.MAX_TURNS_REACHED,
                    error=str(e),
                    messages=run_state.messages,
                    run_artifacts={
                        "stopped_reason": "max_turns",
                        "turns_used": e.current_turn,
                    },
                    trace_id=ctx.trace_id,
                )
                if agent.on_end:
                    agent.on_end(agent, {"context": ctx, "response": partial})
                await self._finalizer.finalize(
                    agent,
                    ctx,
                    run_state,
                    partial,
                    result.user_message_index,
                    result.tool_context_state,
                    start_time,
                    run_state.messages,
                )
                e.partial_response = partial
                raise

            self._circuit_breaker.record_failure()
            await self._finalizer.handle_error(agent, ctx, run_state, e, start_time)

            if isinstance(e, AgentError):
                raise
            raise AgentExecutionError(
                str(e),
                agent_name=agent.name,
                run_id=ctx.run_id,
                trace_id=ctx.trace_id,
                original_error=e,
            ) from e

    # =========================================================================
    # Fork (time-travel: replay a run from a step with an edit)
    # =========================================================================

    async def fork(
        self,
        run_id: str,
        from_step: str,
        *,
        override: dict[str, Any] | None = None,
        agent: BaseAgent | None = None,
        label: str | None = None,
    ) -> AgentResponse:
        """Re-run a past run from ``from_step`` with an optional edit applied.

        Steps before ``from_step`` are replayed from the persisted trace (the
        restored message checkpoint) — not re-executed — and the agent loop runs
        forward from there. The parent run is never mutated; the new run records
        its lineage. Requires DECISION_TRACE_CHECKPOINT to have been on for the
        parent run.

        Works for single-agent, handoff, and workflow runs: if the root agent (or
        an explicit ``agent=``) implements the ``Forkable`` protocol — all nine
        workflow orchestrators do — the fork delegates to its ``resume_from``;
        otherwise it falls back to the built-in single-agent / handoff snapshot
        replay below. The one unsupported case is a step inside a
        ``return_to_parent`` handoff (raises a clear error — see below).

        Args:
            run_id: The parent run to fork from.
            from_step: The step id to resume at (its message checkpoint is the
                resume point).
            override: Optional edit applied to the restored messages — see
                ``trace.fork.apply_override`` (e.g. ``{"set_tool_result": ...}``).
            agent: The agent to run. Defaults to the parent's root agent looked up
                in the registry.
            label: Optional human label stored on the new run's edit metadata.

        Returns:
            The forked run's ``AgentResponse`` (with its own ``decision_trace``).
        """
        from continuum.agent.trace.config import (
            checkpoint_enabled,
            get_trace_store,
        )
        from continuum.agent.trace.fork import apply_override
        from continuum.agent.trace.recorder import TraceRecorder

        start_time = time.time()

        parent = await get_trace_store().get(run_id)
        if parent is None:
            raise ValueError(f"fork: parent run '{run_id}' not found in trace store")

        step = next((s for s in parent.steps if s.step_id == from_step), None)
        if step is None:
            raise ValueError(f"fork: step '{from_step}' not found in run '{run_id}'")

        # Workflow orchestrators have their own control flow and resume via the
        # Forkable protocol. If a Forkable orchestrator is supplied (agent=) or is
        # the run's root agent, delegate to it; otherwise fall through to the
        # built-in single-agent / handoff resume below.
        from continuum.agent.interfaces.forkable import Forkable

        orchestrator = agent or self._agent_registry.get(parent.root_agent)
        if isinstance(orchestrator, Forkable):
            resume_ctx = create_run_context(max_turns=orchestrator.config.max_turns)
            resume_ctx.disable_memory_writes = True  # a fork is a hypothetical replay
            return await orchestrator.resume_from(
                parent_trace=parent,
                from_step=from_step,
                override=override,
                runner=self,
                context=resume_ctx,
            )

        # Guard: forking a step inside a return-to-parent handoff is unsupported.
        # Resuming the child wouldn't reconstruct the parent's continuation (the
        # parent synthesizes the final answer after the child returns), so we'd
        # silently hand back the child's partial answer. Fail clearly instead.
        from continuum.agent.trace.types import StepKind

        _path = set(step.agent_stack or [])
        if step.agent_name:
            _path.add(step.agent_name)
        _path.discard(parent.root_agent)  # the root isn't reached via a handoff
        if _path:
            for s in parent.steps:
                if (
                    s.kind == StepKind.HANDOFF
                    and isinstance(s.decision, dict)
                    and s.decision.get("return_to_parent")
                    and s.decision.get("handoff_to") in _path
                ):
                    raise ValueError(
                        f"fork: step '{from_step}' is inside a return-to-parent handoff "
                        f"(to '{s.decision.get('handoff_to')}'), which is not forkable yet — "
                        "resuming the child can't reconstruct the parent's final answer. "
                        "Use return_to_parent=False for handoffs you intend to fork."
                    )

        if step.messages_snapshot is None:
            raise ValueError(
                f"fork: step '{from_step}' has no message checkpoint. Enable "
                "DECISION_TRACE_CHECKPOINT on the original run to make it forkable."
            )

        # Resume the agent that produced the forked step (so handoff/multi-agent
        # runs resume the right agent, not always the root), unless the caller
        # overrides with agent=.
        step_agent_name = step.agent_name or parent.root_agent
        target = agent or self._agent_registry.get(step_agent_name)
        if target is None:
            raise ValueError(
                f"fork: agent '{step_agent_name}' not in registry; pass agent= explicitly "
                "or register it via runner.register_agent()."
            )
        if target.name not in self._agent_registry:
            self.register_agent(target)

        # The handoff stack active at the forked step. Restoring it keeps handoff
        # depth/cycle checks correct when the resumed agent (or one it hands off
        # to) hands off again as the run replays forward. Terminal handoffs
        # re-fire naturally through the executor; return-to-parent handoffs (where
        # a parent synthesizes the final answer after a child returns) are not yet
        # reconstructed — the resumed agent's own answer is returned (Phase 3).
        resume_stack = list(step.agent_stack) or [target.name]
        if len(resume_stack) > 1:
            logger.info(
                "fork: resuming multi-agent run '%s' at agent '%s' (stack: %s).",
                run_id,
                target.name,
                " → ".join(resume_stack),
            )

        messages = apply_override(step.messages_snapshot, override)

        ctx = create_run_context(max_turns=target.config.max_turns)
        ctx.disable_memory_writes = True  # a fork is a hypothetical replay
        ctx.recorder = TraceRecorder(
            ctx.run_id, target.name, parent.user_query, checkpoint=checkpoint_enabled()
        )
        ctx.recorder.trace.parent_run_id = run_id
        ctx.recorder.trace.forked_from_step = from_step
        ctx.recorder.trace.edit = {"override": override, "label": label}

        run_state = await self._context_service.create_run_state(target, ctx)
        run_state.agent_stack = resume_stack
        run_state.current_agent = target.name
        run_state.messages = list(messages)

        response = await self._executor.execute_loop(
            agent=target, messages=messages, context=ctx, run_state=run_state
        )
        if target.on_end:
            target.on_end(target, {"context": ctx, "response": response})

        await self._finalizer.finalize(
            target, ctx, run_state, response, 0, None, start_time, run_state.messages
        )
        return response

    # =========================================================================
    # Workflow trace lifecycle (used by workflow orchestrators)
    # =========================================================================
    def ensure_recorder(self, context: RunContext, root_agent: str, query: str = "") -> bool:
        """Ensure a decision-trace recorder exists on ``context``, rooted at
        ``root_agent``.

        Workflow orchestrators run their sub-agents via ``runner.run`` with a
        shared context and ``suppress_session_log=True``, so the normal per-run
        finalization never persists their trace. They call this at the start of
        ``execute`` to own the one trace spanning all sub-agents, then call
        :meth:`persist_decision_trace` at the end.

        Returns True iff this call created the recorder (so the caller owns
        finalization). No-op returning False when tracing is disabled or a
        recorder already exists (e.g. an outer workflow/handoff created it).
        """
        if context.recorder is not None:
            return False
        from continuum.agent.trace.config import checkpoint_enabled, is_trace_enabled

        if not is_trace_enabled():
            return False
        from continuum.agent.trace.recorder import TraceRecorder

        context.recorder = TraceRecorder(
            context.run_id, root_agent, query, checkpoint=checkpoint_enabled()
        )
        return True

    async def persist_decision_trace(self, context: RunContext, response: AgentResponse) -> None:
        """Build, persist, and attach the decision trace for a workflow run.

        Bypasses the suppress-session-log guard (workflow sub-runs are
        suppressed). Only the orchestrator that created the recorder
        (see :meth:`ensure_recorder`) should call this.
        """
        await self._finalizer.persist_decision_trace(context, response)

    # =========================================================================
    # Run (streaming)
    # =========================================================================

    async def run_stream(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]],
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run an agent with streaming output."""
        start_time = time.time()

        result = await self._prepare_run(
            agent,
            input,
            session_id,
            conversation_id,
            user_id,
            None,
            max_turns,
            trace_id,
            metadata,
            None,
        )
        if not result.success:
            _run_id = generate_run_id()
            _err = (
                result.error_response.error
                if result.error_response and result.error_response.error
                else "Input validation failed"
            )
            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=_run_id,
                data={"error": _err, "error_type": "ValidationError"},
                trace_id=trace_id,
            )
            yield AgentEvent(
                type=EventType.RUN_END,
                agent_name=agent.name,
                run_id=_run_id,
                data={"content": "", "turn_count": 0},
                trace_id=trace_id,
            )
            return

        ctx = result.context
        run_state = result.run_state

        yield AgentEvent(
            type=EventType.RUN_START,
            agent_name=agent.name,
            run_id=ctx.run_id,
            data={"input": input if isinstance(input, str) else "[messages]"},
            trace_id=ctx.trace_id,
        )

        turn = 0
        try:
            messages = list(run_state.messages) if run_state.messages else []

            yield AgentEvent(
                type=EventType.AGENT_START,
                agent_name=agent.name,
                run_id=ctx.run_id,
                trace_id=ctx.trace_id,
            )

            if (
                isinstance(agent, RouterAgent)
                and app_settings.smart_layer_enabled
                and agent.router_config.routing_strategy == "model_tier"
            ):
                user_text = extract_last_user_text(input, messages)
                content_parts: list[str] = []
                last_routing: dict = {}
                try:
                    async for ev in stream_model_tier_turn(
                        agent,
                        self._llm_client,
                        user_text=user_text,
                        ctx=ctx,
                    ):
                        if ev.kind == "routing" and ev.routing:
                            last_routing = ev.routing
                            yield AgentEvent(
                                type=EventType.ROUTING,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data=ev.routing,
                                trace_id=ctx.trace_id,
                            )
                        elif ev.kind == "content_delta" and ev.text:
                            content_parts.append(ev.text)
                            yield AgentEvent(
                                type=EventType.CONTENT_DELTA,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={"content": ev.text},
                                trace_id=ctx.trace_id,
                            )
                except Exception as e_mt:
                    await self._finalizer.handle_error(agent, ctx, run_state, e_mt, start_time)
                    if isinstance(e_mt, AgentError):
                        raise
                    raise AgentExecutionError(
                        str(e_mt),
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        trace_id=ctx.trace_id,
                        original_error=e_mt,
                    ) from e_mt

                content = "".join(content_parts)
                te = parse_product_tier(last_routing.get("tier"))
                if te:
                    ctx.priority = tier_dispatch_priority(te)

                if content:
                    yield AgentEvent(
                        type=EventType.CONTENT_COMPLETE,
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        data={"content": content},
                        trace_id=ctx.trace_id,
                    )

                response = AgentResponse(
                    content=content,
                    run_id=ctx.run_id,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    trace_id=ctx.trace_id,
                    run_artifacts={"routing": last_routing} if last_routing else None,
                )

                # Decision-trace capture for the smart-layer (model_tier) streaming
                # router path: record the routing decision + the answer with a
                # checkpoint, so this run is inspectable and forkable like the
                # normal streaming loop. Captured BEFORE the assistant message is
                # appended below (the snapshot is the fork resume point).
                if ctx.recorder is not None:
                    from continuum.agent.trace.types import StepKind

                    _agent_stack = run_state.get_agent_stack_snapshot()
                    _snapshot = capture_snapshot(ctx.recorder, messages)
                    ctx.recorder.record(
                        StepKind.ROUTING,
                        agent.name,
                        agent_stack=_agent_stack,
                        input=user_text,
                        decision=last_routing or {},
                        rationale=(
                            f"model_tier → {last_routing.get('tier')}" if last_routing else ""
                        ),
                        messages_snapshot=_snapshot,
                    )
                    if (_ds := latest_step_payload(ctx.recorder)) is not None:
                        yield AgentEvent(
                            type=EventType.DECISION_STEP,
                            agent_name=agent.name,
                            run_id=ctx.run_id,
                            data=_ds,
                            trace_id=ctx.trace_id,
                        )
                    record_llm_turn(
                        ctx.recorder,
                        agent.name,
                        1,
                        content=content,
                        has_tool_calls=False,
                        usage=None,
                        snapshot=_snapshot,
                        agent_stack=_agent_stack,
                    )
                    if (_ds := latest_step_payload(ctx.recorder)) is not None:
                        yield AgentEvent(
                            type=EventType.DECISION_STEP,
                            agent_name=agent.name,
                            run_id=ctx.run_id,
                            data=_ds,
                            trace_id=ctx.trace_id,
                        )

                if content:
                    messages.append({"role": "assistant", "content": content})

                if agent.on_end:
                    agent.on_end(agent, {"context": ctx, "response": response})

                await self._finalizer.finalize(
                    agent,
                    ctx,
                    run_state,
                    response,
                    result.user_message_index,
                    result.tool_context_state,
                    start_time,
                    messages,
                )

                yield AgentEvent(
                    type=EventType.AGENT_END,
                    agent_name=agent.name,
                    run_id=ctx.run_id,
                    data={"turn_count": 1},
                    trace_id=ctx.trace_id,
                )
                yield AgentEvent(
                    type=EventType.RUN_END,
                    agent_name=agent.name,
                    run_id=ctx.run_id,
                    data={"content": content, "turn_count": 1},
                    trace_id=ctx.trace_id,
                )
                return

            content = ""
            turn = 0
            while turn < ctx.max_turns:
                turn += 1
                # Filtered tools set by message_builder via apply_tool_attention.
                tools = (
                    ctx.metadata.get("_filtered_tools") if ctx.metadata else None
                ) or agent.get_tools_for_llm()
                # Phase 1: insert tool catalogue after system messages, before history.
                # Ephemeral — not persisted to session history.
                _phase1 = ctx.metadata.get("tool_summary_message") if ctx.metadata else None
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

                content_parts: list[str] = []
                tool_calls: list = []
                last_seen_model: str | None = None

                async for chunk in self.llm_client.chat_stream(
                    messages=llm_messages,
                    tools=tools if tools else None,
                    config=_enrich_config_for_gateway(LLMConfig.from_agent_config(agent), ctx),
                    trace_metadata={"session_id": session_id} if session_id else None,
                ):
                    if chunk.model:
                        last_seen_model = chunk.model
                    if chunk.content:
                        content_parts.append(chunk.content)
                        if "NEED_TOOL:" not in chunk.content:
                            yield AgentEvent(
                                type=EventType.CONTENT_DELTA,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={"content": chunk.content},
                                trace_id=ctx.trace_id,
                            )
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                if last_seen_model and settings.smart_gateway_url:
                    logger.info("🎯 Gateway selected model: %s", last_seen_model)

                content = "".join(content_parts)

                # Warn if JSON mode was requested but the streamed response is not JSON.
                _cfg = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), ctx)
                if content and (_cfg.json_mode or _cfg.response_format):
                    stripped = content.strip()
                    if not (
                        (stripped.startswith("{") and stripped.endswith("}"))
                        or (stripped.startswith("[") and stripped.endswith("]"))
                    ):
                        logger.warning(
                            "Streamed response is not JSON despite json_mode being set",
                            extra={"model": _cfg.model, "preview": stripped[:100]},
                        )

                # NEED_TOOL fallback: if LLM signals a missing tool, expand and retry.
                if content and "NEED_TOOL:" in content and not tool_calls:
                    needed = content.split("NEED_TOOL:")[1].strip().split()[0].rstrip(".,;")
                    all_tools = agent.get_tools_for_llm()
                    extra = [t for t in all_tools if _tool_name(t) == needed]
                    if extra:
                        logger.info("tool-attention fallback: adding %s and retrying", needed)
                        expanded_tools = tools + [t for t in extra if t not in tools]
                        if ctx.metadata is not None:
                            promoted = ctx.metadata.get("promoted_tools", set())
                            ctx.metadata["promoted_tools"] = promoted | {needed}
                            ctx.metadata["_filtered_tools"] = expanded_tools
                        content_parts = []
                        tool_calls = []
                        last_seen_model = None
                        async for chunk in self.llm_client.chat_stream(
                            messages=llm_messages,
                            tools=expanded_tools,
                            config=_enrich_config_for_gateway(
                                LLMConfig.from_agent_config(agent), ctx
                            ),
                            trace_metadata={"session_id": session_id} if session_id else None,
                        ):
                            if chunk.model:
                                last_seen_model = chunk.model
                            if chunk.content:
                                content_parts.append(chunk.content)
                                yield AgentEvent(
                                    type=EventType.CONTENT_DELTA,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={"content": chunk.content},
                                    trace_id=ctx.trace_id,
                                )
                            if chunk.tool_calls:
                                tool_calls = chunk.tool_calls
                        if last_seen_model and settings.smart_gateway_url:
                            logger.info("🎯 Gateway selected model: %s", last_seen_model)
                        content = "".join(content_parts)
                else:
                    pass  # CONTENT_DELTA already yielded live in the streaming loop above

                if content:
                    yield AgentEvent(
                        type=EventType.CONTENT_COMPLETE,
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        data={"content": content},
                        trace_id=ctx.trace_id,
                    )

                # Decision-trace capture (parity with the non-streaming executor):
                # checkpoint the messages sent this turn (the fork resume point) and
                # record this turn's LLM decision, BEFORE appending the assistant
                # message below. Records every turn — including the final no-tool
                # turn — so a streamed run is inspectable and forkable.
                _agent_stack = run_state.get_agent_stack_snapshot()
                _snapshot = capture_snapshot(ctx.recorder, llm_messages)
                llm_step_id = record_llm_turn(
                    ctx.recorder,
                    agent.name,
                    turn,
                    content=content,
                    has_tool_calls=bool(tool_calls),
                    usage=None,
                    snapshot=_snapshot,
                    agent_stack=_agent_stack,
                )
                if (_ds := latest_step_payload(ctx.recorder)) is not None:
                    yield AgentEvent(
                        type=EventType.DECISION_STEP,
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        data=_ds,
                        trace_id=ctx.trace_id,
                    )

                if tool_calls:
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

                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            yield AgentEvent(
                                type=EventType.HANDOFF_START,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={"target": target},
                                trace_id=ctx.trace_id,
                            )

                            if ctx.recorder is not None:
                                _hc = agent.get_handoff(target)
                                ctx.recorder.record_handoff(
                                    agent.name,
                                    target,
                                    turn,
                                    parent_id=llm_step_id,
                                    agent_stack=_agent_stack,
                                    return_to_parent=bool(_hc and _hc.return_to_parent),
                                )
                                if (_ds := latest_step_payload(ctx.recorder)) is not None:
                                    yield AgentEvent(
                                        type=EventType.DECISION_STEP,
                                        agent_name=agent.name,
                                        run_id=ctx.run_id,
                                        data=_ds,
                                        trace_id=ctx.trace_id,
                                    )

                            if not self._handoff_executor:
                                yield AgentEvent(
                                    type=EventType.HANDOFF_END,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={
                                        "target": target,
                                        "success": False,
                                        "error": "HandoffExecutor not available in streaming mode",
                                    },
                                    trace_id=ctx.trace_id,
                                )
                                yield AgentEvent(
                                    type=EventType.RUN_END,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={"content": "", "turn_count": turn},
                                    trace_id=ctx.trace_id,
                                )
                                return

                            handoff_result = await self._handoff_executor.execute_handoff(
                                agent=agent,
                                target_name=target,
                                tool_call=tc,
                                messages=messages,
                                context=ctx,
                                run_state=run_state,
                            )

                            if not handoff_result.success:
                                yield AgentEvent(
                                    type=EventType.HANDOFF_END,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={
                                        "target": target,
                                        "success": False,
                                        "error": handoff_result.error,
                                    },
                                    trace_id=ctx.trace_id,
                                )
                                yield AgentEvent(
                                    type=EventType.RUN_END,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={"content": "", "turn_count": turn},
                                    trace_id=ctx.trace_id,
                                )
                                return

                            handoff_content = (
                                handoff_result.response.content if handoff_result.response else ""
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": handoff_content or "",
                                }
                            )

                            yield AgentEvent(
                                type=EventType.HANDOFF_END,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={"target": target, "success": True},
                                trace_id=ctx.trace_id,
                            )
                            if handoff_content:
                                yield AgentEvent(
                                    type=EventType.HANDOFF_RETURN,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={"target": target, "content": handoff_content},
                                    trace_id=ctx.trace_id,
                                )
                                yield AgentEvent(
                                    type=EventType.CONTENT_COMPLETE,
                                    agent_name=agent.name,
                                    run_id=ctx.run_id,
                                    data={"content": handoff_content},
                                    trace_id=ctx.trace_id,
                                )
                                content = handoff_content
                            break

                        yield AgentEvent(
                            type=EventType.TOOL_CALL_START,
                            agent_name=agent.name,
                            run_id=ctx.run_id,
                            data={"tool_name": tool_name},
                            trace_id=ctx.trace_id,
                        )

                        try:
                            tool_result, _ = await self._tool_service.execute_tool_call(
                                agent, tc, ctx
                            )
                            messages.append(tool_result)
                            if ctx.recorder is not None:
                                record_tool_steps(
                                    ctx.recorder,
                                    agent.name,
                                    turn,
                                    [tc],
                                    [tool_result],
                                    parent_id=llm_step_id,
                                    agent_stack=_agent_stack,
                                )
                                if (_ds := latest_step_payload(ctx.recorder)) is not None:
                                    yield AgentEvent(
                                        type=EventType.DECISION_STEP,
                                        agent_name=agent.name,
                                        run_id=ctx.run_id,
                                        data=_ds,
                                        trace_id=ctx.trace_id,
                                    )
                            yield AgentEvent(
                                type=EventType.TOOL_CALL_END,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={
                                    "tool_name": tool_name,
                                    "result": tool_result.get("content", "")[:500],
                                },
                                trace_id=ctx.trace_id,
                            )
                        except Exception as e:
                            yield AgentEvent(
                                type=EventType.TOOL_CALL_ERROR,
                                agent_name=agent.name,
                                run_id=ctx.run_id,
                                data={"tool_name": tool_name, "error": str(e)},
                                trace_id=ctx.trace_id,
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": f"Error executing tool: {e}",
                                }
                            )

                    continue
                break

            response = AgentResponse(
                content=content,
                run_id=ctx.run_id,
                agent_name=agent.name,
                status=ResponseStatus.SUCCESS,
                trace_id=ctx.trace_id,
            )

            # Append final assistant response to messages so it gets saved to Redis session
            if content:
                messages.append({"role": "assistant", "content": content})

            if agent.on_end:
                agent.on_end(agent, {"context": ctx, "response": response})

            await self._finalizer.finalize(
                agent,
                ctx,
                run_state,
                response,
                result.user_message_index,
                result.tool_context_state,
                start_time,
                messages,
            )

            yield AgentEvent(
                type=EventType.AGENT_END,
                agent_name=agent.name,
                run_id=ctx.run_id,
                data={"turn_count": turn},
                trace_id=ctx.trace_id,
            )
            yield AgentEvent(
                type=EventType.RUN_END,
                agent_name=agent.name,
                run_id=ctx.run_id,
                data={"content": content, "turn_count": turn},
                trace_id=ctx.trace_id,
            )

        except Exception as e:
            await self._finalizer.handle_error(agent, ctx, run_state, e, start_time)

            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=ctx.run_id,
                data={"error": str(e), "error_type": type(e).__name__},
                trace_id=ctx.trace_id,
            )
            yield AgentEvent(
                type=EventType.RUN_END,
                agent_name=agent.name,
                run_id=ctx.run_id,
                data={"content": "", "turn_count": turn},
                trace_id=ctx.trace_id,
            )

            if isinstance(e, AgentError):
                raise
            raise AgentExecutionError(
                str(e),
                agent_name=agent.name,
                run_id=ctx.run_id,
                trace_id=ctx.trace_id,
                original_error=e,
            ) from e
