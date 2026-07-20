"""
Incident Desk agent — the Headroom glassbox.

An ordinary MCP agent (BaseAgent + AgentRunner + FastMCP tools) whose chat()
returns, alongside the answer, everything Headroom did during the run:

  * tokens before/after and % saved on the LAST LLM call of the run — the one
    that carried the big tool payload (read from
    ``get_headroom_compressor().last_stats``; the compressor is the shipped
    process-global, this rig only observes it)
  * the transforms Headroom applied (router decisions, per-block)
  * CCR hashes newly issued during the run (delta of ``issued_hashes``)
  * every ``continuum_headroom_retrieve`` call the model made (hash + chars returned),
    read back from the run's message list — retrieve is INTERCEPTED in the
    tool loop, so it never appears in ToolService/tool events
  * a sidecar /stats delta (LLM calls compressed, tokens removed) for the run

The rig only CONSUMES shipped SDK API — nothing in src/ changes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from config import (
    SIDECAR_BASE,
    IncidentConfig,
    default_config,
)

from continuum import (
    AgentConfig,
    AgentMemoryConfig,
    AgentRunner,
    BaseAgent,
    MCPServerStreamableHttp,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from continuum.agent.types import EventType
from continuum.core.container import Container, get_container
from continuum.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager
from continuum.llm.headroom.compressor import (
    RETRIEVE_TOOL_NAME,
    get_headroom_compressor,
    reset_headroom_compressor,
)

logger = get_logger(__name__)


class IncidentAgent:
    def __init__(self, config: IncidentConfig | None = None):
        self.config = config or default_config
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[Any] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=False,
            enable_signal_handlers=False,
        )
        await self._lifecycle.initialize()
        self._container = get_container()

        self._mcp_server = MCPServerStreamableHttp(
            params={"url": self.config.mcp_url},
            client_session_timeout_seconds=self.config.mcp_timeout,
        )
        await self._mcp_server.connect()
        self._tool_executor = ToolExecutor({self._mcp_server: None})
        await self._tool_executor.initialize()
        self._tools = self._tool_executor.get_tool_definitions()
        names = [t.function.name for t in self._tools]
        logger.info(f"✓ Discovered {len(self._tools)} tools: {', '.join(names)}")

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=self.config.system_instructions,
            model=self.config.model,
            temperature=self.config.temperature,
            tools=self._tools,
            tool_executor=self._tool_executor,
            # Memory OFF: with it on, mem0 injects a "User profile" system
            # message from whatever earlier demos stored for this user_id —
            # observed live: the clinic demo's PHI prompts leaked into this
            # rig's runs and skewed answers. This rig tests Headroom, only.
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(max_turns=self.config.max_turns),
        )
        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(persist_state=False, default_max_turns=self.config.max_turns),
        )
        self._runner.register_agent(self._agent)
        self._initialized = True
        logger.info(f"✓ IncidentAgent ready — model={self.config.model}")

    # --- glassbox plumbing ---------------------------------------------- #

    @staticmethod
    def _mfield(m: Any, key: str) -> Any:
        if isinstance(m, dict):
            return m.get(key)
        return getattr(m, key, None)

    def _extract_calls(self, resp: Any) -> tuple[list[str], list[dict[str, Any]]]:
        """Walk the run's messages: regular tools called, and every
        continuum_headroom_retrieve call (hash + chars of the returned original)."""
        tools_called: list[str] = []
        retrieves: list[dict[str, Any]] = []
        retrieve_ids: dict[str, str] = {}  # tool_call_id -> hash
        for m in getattr(resp, "messages", None) or []:
            for tc in self._mfield(m, "tool_calls") or []:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                args_raw = getattr(fn, "arguments", None) if fn else None
                tc_id = getattr(tc, "id", None)
                if name is None and isinstance(tc, dict):
                    name = tc.get("function", {}).get("name")
                    args_raw = tc.get("function", {}).get("arguments")
                    tc_id = tc.get("id")
                if not name:
                    continue
                if name == RETRIEVE_TOOL_NAME:
                    try:
                        h = json.loads(args_raw or "{}").get("hash", "?")
                    except Exception:
                        h = "?"
                    retrieve_ids[tc_id or ""] = h
                else:
                    tools_called.append(name)
            if self._mfield(m, "role") == "tool":
                tcid = self._mfield(m, "tool_call_id")
                if tcid in retrieve_ids:
                    content = str(self._mfield(m, "content") or "")
                    retrieves.append({"hash": retrieve_ids[tcid], "chars": len(content)})
        return tools_called, retrieves

    @staticmethod
    def _headroom_run_begin() -> set[str]:
        """Start-of-run snapshot for the glassbox: capture issued hashes and
        reset per-run counters. When Headroom is disabled, touch nothing — no
        compressor is constructed — so a disabled run involves Headroom zero
        times (the SDK is already inert; this keeps the demo honest too)."""
        from continuum.config import settings

        if not settings.headroom_enabled:
            return set()
        comp = get_headroom_compressor()
        hashes = set(comp.issued_hashes)
        comp.reset_run_counters()
        return hashes

    @staticmethod
    def _headroom_glassbox(hashes_before: set[str]) -> dict[str, Any]:
        from continuum.config import settings

        # Disabled → report that and stop; never build the compressor.
        if not settings.headroom_enabled:
            return {"enabled": False}
        comp = get_headroom_compressor()
        stats = comp.last_stats
        # Run total = sum of every apply() this run (the compressor accumulates
        # it internally). The sidecar's /stats endpoint aggregates/lags and read
        # 0 here, so we use the in-process counters instead.
        totals = comp.run_totals
        box: dict[str, Any] = {
            "new_hashes": sorted(comp.issued_hashes - hashes_before),
            "sidecar_delta": {
                "llm_calls_compressed": totals["calls"],
                "tokens_removed": totals["tokens_removed"],
            },
        }
        if stats is not None:
            box["last_call"] = {
                "tokens_before": stats.tokens_before,
                "tokens_after": stats.tokens_after,
                "pct_saved": round((1 - stats.compression_ratio) * 100, 1),
                "transforms": list(stats.transforms_applied),
            }
        else:
            box["last_call"] = None  # sidecar down / fail-open, or nothing compressed yet

        # Message snapshots: truncate each message's content to keep JSON sane.
        def _snap(msgs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
            if not msgs:
                return []
            out = []
            for m in msgs:
                entry: dict[str, Any] = {"role": m.get("role", "?")}
                content = m.get("content")
                if content is not None:
                    s = str(content)
                    entry["content"] = s
                    entry["content_len"] = len(s)
                tcs = m.get("tool_calls")
                if tcs:
                    entry["tool_calls"] = len(tcs)
                tc_id = m.get("tool_call_id")
                if tc_id:
                    entry["tool_call_id"] = tc_id
                out.append(entry)
            return out

        box["messages_before"] = _snap(comp.last_messages_before)
        box["messages_after"] = _snap(comp.last_messages_after)
        return box

    # --- chat (executor path) --------------------------------------------- #

    async def chat(self, message: str, user_id: str = "u1") -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        hashes_before = self._headroom_run_begin()

        resp = await self._runner.run(agent=self._agent, input=message, user_id=user_id)

        tools_called, retrieves = self._extract_calls(resp)
        return {
            "response": resp.content or "",
            "tools_called": tools_called,
            "retrieve_calls": retrieves,
            "headroom": self._headroom_glassbox(hashes_before),
        }

    # --- chat_stream (runner interception path) ---------------------------- #

    async def chat_stream(self, message: str, user_id: str = "u1") -> AsyncIterator[dict[str, Any]]:
        """Streaming twin of chat(). Yields live token/tool events and a final
        'done' event with the same glassbox payload. Note: continuum_headroom_retrieve
        is intercepted BEFORE the runner emits TOOL_CALL_START, so retrieve
        calls (correctly) never show up as tool events — they surface in the
        glassbox via the issued-hash delta."""
        if not self._initialized:
            await self.initialize()
        hashes_before = self._headroom_run_begin()

        content = ""
        tools_called: list[str] = []
        async with aclosing(
            self._runner.run_stream(agent=self._agent, input=message, user_id=user_id)
        ) as stream:
            async for ev in stream:
                if ev.type == EventType.CONTENT_DELTA:
                    delta = ev.data.get("content", "")
                    content += delta
                    yield {"type": "token", "text": delta}
                elif ev.type == EventType.CONTENT_COMPLETE:
                    content = ev.data.get("content", "") or content
                    yield {"type": "message", "text": content}
                elif ev.type == EventType.TOOL_CALL_START:
                    name = ev.data.get("tool_name", "")
                    if name:
                        tools_called.append(name)
                        yield {"type": "tool", "name": name}

        yield {
            "type": "done",
            "response": content,
            "tools_called": tools_called,
            "retrieve_calls": [],  # not observable as stream events (see docstring)
            "headroom": self._headroom_glassbox(hashes_before),
        }

    # --- scenario helpers --------------------------------------------------- #

    async def chat_with_rag_context(
        self, message: str, rag_context: str, user_id: str = "u1"
    ) -> dict[str, Any]:
        """Scenario 10: feed the payload through Continuum's NATIVE RAG slot —
        ``agent.config.rag_context`` — instead of a tool.

        message_builder.py:214-227 wraps rag_context in a
        '--- PROVIDED CONTEXT ---' block and appends it as a **system** message
        (position 7). Headroom skips system/developer messages by default
        (content_router.py:2797 → 'router:protected:system_message'; Continuum
        never sends compress_system_messages=True), so the same bytes that
        SearchCompressor crushes ~98% as a *tool result* (scenario 3) should
        pass through UNCOMPRESSED here. That contrast is the whole point:
        pre-injected RAG context is protected; tool-use RAG is compressed.

        rag_context is set on the shared agent for this one run and restored in
        a finally, so it never leaks into other scenarios.
        """
        if not self._initialized:
            await self.initialize()
        hashes_before = self._headroom_run_begin()

        prev = self._agent.config.rag_context
        self._agent.config.rag_context = rag_context
        try:
            resp = await self._runner.run(agent=self._agent, input=message, user_id=user_id)
        finally:
            self._agent.config.rag_context = prev

        tools_called, retrieves = self._extract_calls(resp)
        return {
            "response": resp.content or "",
            "tools_called": tools_called,
            "retrieve_calls": retrieves,
            "headroom": self._headroom_glassbox(hashes_before),
        }

    def point_sidecar(self, api_base: str | None) -> str:
        """Scenario 5 (fail-open): repoint the SDK at `api_base` (None →
        restore the real sidecar). Rebuilds the process-global compressor, so
        previously issued CCR hashes are forgotten — retrieval of older
        markers will fail-open with the re-run guidance, by design."""
        from continuum.config import settings

        settings.headroom_api_base = api_base or SIDECAR_BASE
        reset_headroom_compressor()
        return settings.headroom_api_base

    async def forge_retrieve(self, hash_value: str = "f" * 24) -> str:
        """Scenario 6 (anti-forgery): hand the compressor a hash it never
        issued. Must be rejected without the sidecar ever being contacted."""
        return await get_headroom_compressor().resolve_retrieve(hash_value)

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
