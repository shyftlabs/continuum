"""
Local Shop Agent.

Single agent using MCPServerStreamableHttp (HTTP transport) — same pattern as commerce-chat
but against a local MCP server instead of the remote one.
"""

import json
import os
import sys
import uuid
from collections.abc import AsyncGenerator
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import ShopConfig, default_config

from continuum import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    MCPServerStreamableHttp,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from continuum.agent.types import EventType
from continuum.core.container import Container, get_container
from continuum.llm.timing_probe import timing_turn
from continuum.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager
from continuum.exceptions import InsecureConfigurationError
from continuum.tools.tool_attention.config import ToolAttentionConfig
from continuum.tools.types import ToolContextConfig, ToolContextVariable
from continuum.tools.util import NAMESPACE_SEPARATOR
from continuum.utils.sanitization import (
    InvalidIdentifierError,
    validate_conversation_id,
    validate_user_id,
)

logger = get_logger(__name__)

# Server tools that are supposed to carry a total (server.py). add_to_cart is
# deliberately absent: it returns only a message and cart_size, so gating it
# here would fire the "NO totals" warning on every successful add.
_CART_TOOLS = {"view_cart", "checkout"}
_TOTAL_KEYS = {"total", "subtotal", "total_cents", "subtotal_cents", "taxes", "tax_cents"}

# Per-turn latency tagging. Both helpers are cheap and unconditional — the probe
# itself decides whether anything is written, so the agent path does not have to
# branch on whether measurement is switched on.
_TURN_QUESTION_MAX = 120


def _new_turn_id() -> str:
    """Short, unique id for one user question."""
    return uuid.uuid4().hex[:12]


def _turn_meta(message: str, config: ShopConfig, entrypoint: str) -> dict[str, Any]:
    """What the rollup needs in order to be readable.

    The question is truncated: it is repeated on every record belonging to the
    turn, and a long paste would dominate the log without telling the reader more
    than its first line does.
    """
    q = " ".join(message.split())
    return {
        "question": q[:_TURN_QUESTION_MAX] + ("…" if len(q) > _TURN_QUESTION_MAX else ""),
        "agent_model": config.agent_model,
        "entrypoint": entrypoint,
    }


def _raw_tool_name(tool_name: str) -> str:
    """Strip the MCP namespace prefix from an LLM-facing tool name.

    _on_tool_result receives tool_call.function.name, which with
    namespace_tools enabled is "shop__view_cart". Only the configured prefix is
    removed, so a tool whose own name contains "__" survives intact and the
    unnamespaced name still matches.
    """
    prefix = f"{default_config.mcp_server_name}{NAMESPACE_SEPARATOR}"
    return tool_name[len(prefix) :] if tool_name.startswith(prefix) else tool_name


class CartDebugToolExecutor(ToolExecutor):
    """ToolExecutor subclass that adds cart-specific debug logging after each tool call."""

    def _on_tool_result(self, tool_name: str, result: str, artifact: Any) -> None:
        tool_name = _raw_tool_name(tool_name)
        sc = artifact.structured_content if artifact else None

        # Log totals from structuredContent for any tool that returns them
        if sc:
            total_fields = {k: v for k, v in sc.items() if k in _TOTAL_KEYS}
            if total_fields:
                logger.info(f"💰 {tool_name} structuredContent totals: {total_fields}")

            # Log cart item count and sample price fields
            items = sc.get("items") or sc.get("cart_items")
            if isinstance(items, list) and items:
                sample = {
                    k: v
                    for k, v in items[0].items()
                    if "price" in k.lower() or "total" in k.lower()
                }
                logger.info(
                    f"🛒 {tool_name} cart: items_count={len(items)}, "
                    f"sample_price_fields={sample}, "
                    f"structuredContent_keys={list(sc.keys())}"
                )

        # Extra detail for known cart tools
        if tool_name in _CART_TOOLS:
            if sc:
                llm_totals = {
                    k: v
                    for k, v in sc.items()
                    if "total" in k.lower() or "subtotal" in k.lower() or "tax" in k.lower()
                }
                if llm_totals:
                    logger.info(
                        f"📤 {tool_name} sending to LLM (structuredContent): totals={llm_totals}"
                    )
                else:
                    logger.warning(
                        f"⚠️ {tool_name} structuredContent has NO totals for LLM! "
                        f"Keys: {list(sc.keys())}"
                    )


class LocalShopAgent:
    def __init__(self, config: ShopConfig | None = None):
        self.config = config or default_config
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[Any] = []
        self._resource_context: str = ""
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False, verify_connections=True, enable_signal_handlers=False
        )
        await self._lifecycle.initialize()

        self._container = get_container()

        await self._connect_mcp()
        self._create_agent()

        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(default_max_turns=self.config.max_turns),
        )
        self._runner.register_agent(self._agent)
        self._initialized = True
        logger.info(f"✓ LocalShopAgent ready! (llm model used: {self.config.agent_model})")

    async def _connect_mcp(self) -> None:
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")

        context_config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    inject_into=["add_to_cart", "view_cart", "checkout"],
                )
            ],
            auto_capture_common=False,
        )

        self._mcp_server = MCPServerStreamableHttp(
            params={"url": self.config.mcp_url},
            name=self.config.mcp_server_name,
            client_session_timeout_seconds=self.config.mcp_timeout,
            context_config=context_config,
        )
        await self._mcp_server.connect()

        self._tool_executor = CartDebugToolExecutor({self._mcp_server: None})
        await self._tool_executor.initialize()

        self._tools = self._tool_executor.get_tool_definitions()
        # Strip injected parameters from schemas so the LLM never sees them as
        # required fields and doesn't ask the user for values the executor provides.
        _injected = {"session_id"}
        for tool_def in self._tools:
            params = tool_def.function.parameters or {}
            props = params.get("properties", {})
            for p in _injected:
                props.pop(p, None)
            params["required"] = [r for r in params.get("required", []) if r not in _injected]

        names = [t.function.name for t in self._tools]
        logger.info(f"✓ Discovered {len(self._tools)} tools: {', '.join(names)}")

        await self._fetch_resources()

    async def _fetch_resources(self) -> None:
        try:
            catalogue = await self._mcp_server.read_resource("shop://catalogue")
            categories = await self._mcp_server.read_resource("shop://categories")
            self._resource_context = f"Product catalogue:\n{catalogue}\n\nCategories:\n{categories}"
            logger.info("✓ Loaded shop resources (catalogue + categories)")
        except Exception as e:
            logger.warning(f"Could not load shop resources: {e}")

    def _create_agent(self) -> None:
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = (
            self.config.enable_memory and memory_client is not None and memory_client.is_enabled
        )

        instructions = self.config.system_instructions

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=instructions,
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            # getattr: this playground config is edited constantly between
            # experiments, and a commented-out field should degrade to "no extra
            # params" rather than crash the agent at startup.
            extra_body=getattr(self.config, "extra_body", None),
            gateway_mode=self.config.gateway_mode,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
                extraction_prompt=(
                    "You are a memory extraction system for a pet shop assistant. "
                    "Only extract long-term facts about the user's pets, animal preferences, "
                    "favourite products, or dietary needs. "
                    "Do NOT store transient actions like adding to cart, checkout requests, "
                    "or one-off searches. Do NOT store unrelated personal facts (e.g. favourite colour)."
                ),
            ),
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
                # Tool-attention: local-shop has ~5 tools so set min_tools=3 to trigger.
                # k=3 means top-3 semantically relevant tools promoted each turn.
                tool_attention=ToolAttentionConfig(k=3, min_tools=3),
            ),
        )

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()

        try:
            user_id = validate_user_id(user_id)
            conversation_id = validate_conversation_id(conversation_id)
        except InvalidIdentifierError as e:
            logger.warning(f"Rejected invalid identifier: {e}")
            return f"Error: {e}"

        namespace = self._mcp_server.name if self._mcp_server else "local-shop"
        cart_session_id = f"{user_id}:{conversation_id}"

        # Seed in-memory context (works when session service is disabled).
        if self._tool_executor:
            self._tool_executor.context_state.set(namespace, "session_id", cart_session_id)

        session_id = None
        if self._container:
            session_client = self._container.session_client
            if session_client and session_client.is_enabled:
                try:
                    session_id = await session_client.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    logger.info(f"✓ Active Session ID: {session_id}")
                    # The runner loads tool context from Redis and overwrites the
                    # in-memory context_state, so we must also persist cart_session_id
                    # to Redis before run() is called.
                    existing = await self._runner._session_service.load_tool_context_state(
                        session_id
                    )
                    existing.set(namespace, "session_id", cart_session_id)
                    await self._runner._session_service.save_tool_context_state(
                        session_id, existing
                    )
                except InsecureConfigurationError as e:
                    # Fail closed: a weak/blank data-store secret is an operator
                    # config error — refuse the request instead of running stateless.
                    logger.error(f"Insecure session config for user {user_id}: {e}")
                    return f"Error: {e}"
                except Exception as e:
                    logger.warning(f"Session init failed for user {user_id}: {e}")

        # One user question fans out into several gateway calls — the agent loop
        # re-enters the model after each tool result. The probe records one line
        # per HTTP call, so without a shared id there is no way to add them back
        # up into "what did this question cost". Inert unless CONTINUUM_TIMING_LOG
        # is set: the context manager only sets a contextvar.
        try:
            with timing_turn(_new_turn_id(), meta=_turn_meta(message, self.config, "chat")):
                response = await self._runner.run(
                    agent=self._agent,
                    input=message,
                    session_id=session_id,
                    user_id=user_id,
                )
            return response.content or ""
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"

    async def chat_stream(
        self, message: str, user_id: str, conversation_id: str
    ) -> AsyncGenerator[str]:
        if not self._initialized:
            await self.initialize()

        try:
            user_id = validate_user_id(user_id)
            conversation_id = validate_conversation_id(conversation_id)
        except InvalidIdentifierError as e:
            logger.warning(f"Rejected invalid identifier: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        namespace = self._mcp_server.name if self._mcp_server else "local-shop"
        cart_session_id = f"{user_id}:{conversation_id}"

        if self._tool_executor:
            self._tool_executor.context_state.set(namespace, "session_id", cart_session_id)

        session_id = None
        if self._container:
            session_client = self._container.session_client
            if session_client and session_client.is_enabled:
                try:
                    session_id = await session_client.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    existing = await self._runner._session_service.load_tool_context_state(
                        session_id
                    )
                    existing.set(namespace, "session_id", cart_session_id)
                    await self._runner._session_service.save_tool_context_state(
                        session_id, existing
                    )
                except InsecureConfigurationError as e:
                    # Fail closed: surface the config error to the client and stop.
                    logger.error(f"Insecure session config for user {user_id}: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    return
                except Exception as e:
                    logger.warning(f"Session init failed for user {user_id}: {e}")

        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'user_id': user_id})}\n\n"

        # See chat(). Note the probe's streaming caveat: httpx fires its response
        # hook when headers arrive, so client_ms on this path is time-to-headers
        # and the gateway usually cannot stamp x-aura-timing onto a body that is
        # already flowing. Measure with the UI's Stream toggle OFF.
        try:
            with timing_turn(
                _new_turn_id(), meta=_turn_meta(message, self.config, "chat_stream")
            ):
                async for event in self._runner.run_stream(
                    agent=self._agent,
                    input=message,
                    session_id=session_id,
                    user_id=user_id,
                ):
                    if event.type == EventType.CONTENT_DELTA:
                        content = event.data.get("content", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

                    elif event.type == EventType.TOOL_CALL_START:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': event.data.get('tool_name', ''), 'status': 'start'})}\n\n"

                    elif event.type == EventType.TOOL_CALL_END:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': event.data.get('tool_name', ''), 'status': 'end'})}\n\n"

                    elif event.type == EventType.RUN_ERROR:
                        yield f"data: {json.dumps({'type': 'error', 'error': event.data.get('error', 'Unknown error')})}\n\n"
                        break

                    elif event.type == EventType.RUN_END:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    async def close(self) -> None:
        if self._mcp_server:
            try:
                await self._mcp_server.cleanup()
            except Exception:
                pass
        if self._lifecycle:
            await self._lifecycle.shutdown()

    @property
    def tools(self) -> list[Any]:
        return self._tools


async def create_shop_agent() -> LocalShopAgent:
    agent = LocalShopAgent()
    await agent.initialize()
    return agent
