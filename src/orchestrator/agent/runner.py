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

from orchestrator.agent.base import BaseAgent
from orchestrator.tools.tool_attention.router import _tool_name
from orchestrator.agent.config import RunnerConfig
from orchestrator.agent.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    MaxTurnsExceededError,
)
from orchestrator.agent.execution.executor import Executor
from orchestrator.agent.execution.handoff_executor import HandoffExecutor
from orchestrator.agent.execution.message_builder import MessageBuilder
from orchestrator.agent.execution.run_finalizer import RunFinalizer
from orchestrator.agent.execution.run_lifecycle import RunLifecycle
from orchestrator.agent.execution.stream_executor import StreamExecutor
from orchestrator.agent.execution.tool_handler import ToolHandler
from orchestrator.agent.handoff.manager import HandoffManager
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.agent.persistence.state import RunStateManager
from orchestrator.agent.services.context_service import ContextService
from orchestrator.agent.services.memory_service import MemoryService
from orchestrator.agent.services.session_service import SessionService
from orchestrator.agent.services.tool_service import ToolService
from orchestrator.agent.smart_layer.runner_facade import (
    extract_last_user_text,
    run_model_tier_turn,
    stream_model_tier_turn,
)
from orchestrator.agent.smart_layer.types import parse_product_tier, tier_dispatch_priority
from orchestrator.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
    generate_run_id,
)
from orchestrator.agent.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from orchestrator.agent.utils.context_utils import create_run_context
from orchestrator.agent.utils.message_utils import message_to_dict
from orchestrator.agent.utils.validation_utils import validate_input
from orchestrator.config import settings
from orchestrator.core.container import Container, get_container
from orchestrator.llm.config import LLMConfig
from orchestrator.agent.execution.executor import _enrich_config_for_gateway
from orchestrator.logging import get_logger
from orchestrator.config import settings as app_settings

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient
    from orchestrator.llm.types import ChatMessage
    from orchestrator.memory import MemoryClient
    from orchestrator.observability import TracingManager
    from orchestrator.session import SessionClient
    from orchestrator.tools import ToolExecutor

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

        from orchestrator.llm.types import ChatMessage

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

        if context is None:
            context = create_run_context(
                session_id=session_id,
                conversation_id=conversation_id,
                user_id=user_id,
                trace_id=trace_id,
                max_turns=max_turns or agent.config.max_turns,
                metadata=metadata or {},
                tags=tags or [],
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
            agent, input, context, tool_context_state=tool_context_state,
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
            agent, input, session_id, conversation_id, user_id, context, max_turns, trace_id, metadata, tags
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
                agent=agent, messages=messages, context=ctx, run_state=run_state,
            )

            if agent.on_end:
                agent.on_end(agent, {"context": ctx, "response": response})

            await self._finalizer.finalize(
                agent, ctx, run_state, response,
                result.user_message_index, result.tool_context_state,
                start_time, response.messages,
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
                    agent, ctx, run_state, partial,
                    result.user_message_index, result.tool_context_state,
                    start_time, run_state.messages,
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
            agent, input, session_id, conversation_id, user_id, None, max_turns, trace_id, metadata, None
        )
        if not result.success:
            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=generate_run_id(),
                data={"error": "Input validation failed", "error_type": "ValidationError"},
                trace_id=trace_id,
            )
            return

        ctx = result.context
        run_state = result.run_state

        yield AgentEvent(
            type=EventType.RUN_START, agent_name=agent.name,
            run_id=ctx.run_id, data={"input": input if isinstance(input, str) else "[messages]"},
            trace_id=ctx.trace_id,
        )

        try:
            messages = list(run_state.messages) if run_state.messages else []

            yield AgentEvent(
                type=EventType.AGENT_START, agent_name=agent.name,
                run_id=ctx.run_id, trace_id=ctx.trace_id,
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
                    yield AgentEvent(
                        type=EventType.RUN_ERROR,
                        agent_name=agent.name,
                        run_id=ctx.run_id,
                        data={"error": str(e_mt), "error_type": type(e_mt).__name__},
                        trace_id=ctx.trace_id,
                    )
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
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                if last_seen_model and settings.smart_gateway_url:
                    logger.info("🎯 Gateway selected model: %s", last_seen_model)

                content = "".join(content_parts)

                # Warn if JSON mode was requested but the streamed response is not JSON.
                _cfg = _enrich_config_for_gateway(LLMConfig.from_agent_config(agent), ctx)
                if content and (_cfg.json_mode or _cfg.response_format):
                    stripped = content.strip()
                    if not ((stripped.startswith("{") and stripped.endswith("}")) or
                            (stripped.startswith("[") and stripped.endswith("]"))):
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
                            config=_enrich_config_for_gateway(LLMConfig.from_agent_config(agent), ctx),
                            trace_metadata={"session_id": session_id} if session_id else None,
                        ):
                            if chunk.model:
                                last_seen_model = chunk.model
                            if chunk.content:
                                content_parts.append(chunk.content)
                                yield AgentEvent(
                                    type=EventType.CONTENT_DELTA, agent_name=agent.name,
                                    run_id=ctx.run_id, data={"content": chunk.content},
                                    trace_id=ctx.trace_id,
                                )
                            if chunk.tool_calls:
                                tool_calls = chunk.tool_calls
                        if last_seen_model and settings.smart_gateway_url:
                            logger.info("🎯 Gateway selected model: %s", last_seen_model)
                        content = "".join(content_parts)
                else:
                    for part in content_parts:
                        yield AgentEvent(
                            type=EventType.CONTENT_DELTA, agent_name=agent.name,
                            run_id=ctx.run_id, data={"content": part},
                            trace_id=ctx.trace_id,
                        )

                if content:
                    yield AgentEvent(
                        type=EventType.CONTENT_COMPLETE, agent_name=agent.name,
                        run_id=ctx.run_id, data={"content": content}, trace_id=ctx.trace_id,
                    )

                if tool_calls:
                    messages.append({
                        "role": "assistant", "content": content or None,
                        "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls],
                    })

                    for tc in tool_calls:
                        tool_name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
                        tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")

                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            yield AgentEvent(type=EventType.HANDOFF_START, agent_name=agent.name, run_id=ctx.run_id, data={"target": target}, trace_id=ctx.trace_id)

                            if not self._handoff_executor:
                                yield AgentEvent(type=EventType.HANDOFF_END, agent_name=agent.name, run_id=ctx.run_id, data={"target": target, "success": False, "error": "HandoffExecutor not available in streaming mode"}, trace_id=ctx.trace_id)
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
                                yield AgentEvent(type=EventType.HANDOFF_END, agent_name=agent.name, run_id=ctx.run_id, data={"target": target, "success": False, "error": handoff_result.error}, trace_id=ctx.trace_id)
                                return

                            handoff_content = handoff_result.response.content if handoff_result.response else ""
                            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": handoff_content or ""})

                            yield AgentEvent(type=EventType.HANDOFF_END, agent_name=agent.name, run_id=ctx.run_id, data={"target": target, "success": True}, trace_id=ctx.trace_id)
                            if handoff_content:
                                yield AgentEvent(type=EventType.HANDOFF_RETURN, agent_name=agent.name, run_id=ctx.run_id, data={"target": target, "content": handoff_content}, trace_id=ctx.trace_id)
                                yield AgentEvent(type=EventType.CONTENT_COMPLETE, agent_name=agent.name, run_id=ctx.run_id, data={"content": handoff_content}, trace_id=ctx.trace_id)
                                content = handoff_content
                            break

                        yield AgentEvent(type=EventType.TOOL_CALL_START, agent_name=agent.name, run_id=ctx.run_id, data={"tool_name": tool_name}, trace_id=ctx.trace_id)

                        try:
                            tool_result, _ = await self._tool_service.execute_tool_call(agent, tc, ctx)
                            messages.append(tool_result)
                            yield AgentEvent(type=EventType.TOOL_CALL_END, agent_name=agent.name, run_id=ctx.run_id, data={"tool_name": tool_name, "result": tool_result.get("content", "")[:500]}, trace_id=ctx.trace_id)
                        except Exception as e:
                            yield AgentEvent(type=EventType.TOOL_CALL_ERROR, agent_name=agent.name, run_id=ctx.run_id, data={"tool_name": tool_name, "error": str(e)}, trace_id=ctx.trace_id)

                    continue
                break

            response = AgentResponse(
                content=content, run_id=ctx.run_id, agent_name=agent.name,
                status=ResponseStatus.SUCCESS, trace_id=ctx.trace_id,
            )

            # Append final assistant response to messages so it gets saved to Redis session
            if content:
                messages.append({"role": "assistant", "content": content})

            if agent.on_end:
                agent.on_end(agent, {"context": ctx, "response": response})

            await self._finalizer.finalize(
                agent, ctx, run_state, response,
                result.user_message_index, result.tool_context_state,
                start_time, messages,
            )

            yield AgentEvent(type=EventType.AGENT_END, agent_name=agent.name, run_id=ctx.run_id, data={"turn_count": turn}, trace_id=ctx.trace_id)
            yield AgentEvent(type=EventType.RUN_END, agent_name=agent.name, run_id=ctx.run_id, data={"content": content, "turn_count": turn}, trace_id=ctx.trace_id)

        except Exception as e:
            await self._finalizer.handle_error(agent, ctx, run_state, e, start_time)

            yield AgentEvent(type=EventType.RUN_ERROR, agent_name=agent.name, run_id=ctx.run_id, data={"error": str(e), "error_type": type(e).__name__}, trace_id=ctx.trace_id)

            if isinstance(e, AgentError):
                raise
            raise AgentExecutionError(
                str(e), agent_name=agent.name, run_id=ctx.run_id,
                trace_id=ctx.trace_id, original_error=e,
            ) from e
