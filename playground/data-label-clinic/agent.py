"""
Clinic intake agent — wires the data-label enforcement end to end.

What makes this agent a data-label test rig (vs an ordinary MCP agent):

  * ``policy_store=build_policy_store()``  → the four PHI deny rules.
  * ``config.tool_data_labels``            → lookup_patient declared PHI (tool provenance).
  * memory write-gate: ``deny phi → memory:*`` — a PHI-tainted run may never
    persist long-term memory in any scope (read=taint is intentionally unused;
    see config.py).

The chat() flow makes the *model-routing* gate observable: every turn starts on
the cloud model; the moment a PHI tool taints the run, the next cloud turn is
denied (ModelAccessDeniedError) and we transparently re-run on the on-prem model
— the realistic "policy enforced → fall back to the compliant model" pattern.
It returns a glassbox dict (taint, model used, gate events) for the web UI.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import PHI, ClinicConfig, build_policy_store, default_config

from continuum import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    MCPServerStreamableHttp,
    RunContext,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from continuum.agent.exceptions import MemoryAccessDeniedError, ModelAccessDeniedError
from continuum.agent.types import EventType, generate_run_id
from continuum.core.container import Container, get_container
from continuum.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager

logger = get_logger(__name__)


class ClinicAgent:
    def __init__(self, config: ClinicConfig | None = None):
        self.config = config or default_config
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[Any] = []
        self._policy_store = build_policy_store()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=True,
            enable_signal_handlers=False,
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
        logger.info(
            f"✓ ClinicAgent ready — cloud={self.config.cloud_model} onprem={self.config.onprem_model}"
        )

    async def _connect_mcp(self) -> None:
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")
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

    def _create_agent(self) -> None:
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = (
            self.config.enable_memory and memory_client is not None and memory_client.is_enabled
        )

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=self.config.system_instructions,
            model=self.config.cloud_model,  # start on cloud; gate forces on-prem for PHI
            temperature=self.config.temperature,
            tools=self._tools,
            tool_executor=self._tool_executor,
            policy_store=self._policy_store,  # ← the gates read this
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                # read=taint: reading this scope taints the run with PHI.
                scope_data_labels=self.config.scope_data_labels,
            ),
            config=AgentConfig(
                max_turns=self.config.max_turns,
                # tool provenance: lookup_patient result taints the run with PHI.
                tool_data_labels=self.config.tool_data_labels,
                # persist the conversation to short-term memory (session/Redis);
                # the session gate placeholders a tainted turn's answer.
                log_to_session=self.config.enable_session,
                # SDK output-scanner hook: masks SSNs in the visible answer.
                # Composes with the gates (independent, runs at a different point).
                output_scanners=self.config.output_scanners,
            ),
        )

    async def _ensure_session(self, user_id: str, conversation_id: str) -> str | None:
        """Resolve a deterministic session id from (user_id, conversation_id) —
        same (user, conversation) → same session → short-term continuity. Returns
        None when the session client isn't enabled (no Redis)."""
        if not self._container:
            return None
        sc = self._container.session_client
        if not (sc and sc.is_enabled):
            return None
        try:
            return await sc.get_or_create_session(user_id=user_id, conversation_id=conversation_id)
        except Exception as e:
            logger.warning(f"session init failed: {e}")
            return None

    async def _run_once(
        self, message: str, model: str, user_id: str, conversation_id: str, session_id: str | None
    ) -> tuple[Any, RunContext]:
        """Run the agent once on a specific model with a fresh RunContext we own
        (so we can read ctx.data_labels back for the UI). ``session_id`` enables
        short-term (Redis) persistence; only a completed run persists, so the
        cloud attempt that errors mid-run writes nothing."""
        self._agent.model = model
        ctx = RunContext(
            run_id=generate_run_id(),
            user_id=user_id,
            conversation_id=conversation_id,
            # MUST set session_id on the context itself: when a context is passed,
            # runner._prepare_run ignores the run(session_id=...) argument, so the
            # load (message_builder) and save (finalizer) both key off
            # context.session_id. Without this, short-term memory is inert.
            session_id=session_id,
        )
        resp = await self._runner.run(
            agent=self._agent,
            input=message,
            context=ctx,
            session_id=session_id,
            user_id=user_id,
        )
        return resp, ctx

    def _apply_scanner(self, on: bool) -> None:
        """Attach or detach the output scanner on the live agent.

        With the scanner ON the SDK buffers each turn and emits one sanitized
        message (safe, but no token-by-token typing — a half-formed token could
        leak an un-redacted SSN). With it OFF the SDK streams raw CONTENT_DELTA
        tokens (visible typing, but unredacted). The two are mutually exclusive;
        this toggle lets the demo show both. We mutate the live agent's config
        because run_stream reads ``output_scanners`` at the start of each run."""
        self._agent.config.output_scanners = self.config.output_scanners if on else []

    async def chat(
        self, message: str, user_id: str, conversation_id: str, scanner_on: bool = True
    ) -> dict[str, Any]:
        """Run the turn, surfacing every gate decision for the glassbox UI."""
        if not self._initialized:
            await self.initialize()

        gate_events: list[str] = []
        model_used = self.config.cloud_model
        session_id = await self._ensure_session(user_id, conversation_id)
        self._apply_scanner(scanner_on)

        try:
            resp, ctx = await self._run_once(
                message, self.config.cloud_model, user_id, conversation_id, session_id
            )
        except ModelAccessDeniedError as e:
            # The PHI taint tripped the cloud-model deny mid-run. Re-run on the
            # PHI-approved on-prem model (the compliant fallback).
            gate_events.append(
                f"🛡️ MODEL ROUTING — cloud '{self.config.cloud_model}' DENIED for PHI "
                f"(policy '{e.context.get('policy_name')}'). Re-routing to on-prem."
            )
            model_used = self.config.onprem_model
            resp, ctx = await self._run_once(
                message, self.config.onprem_model, user_id, conversation_id, session_id
            )
        finally:
            self._agent.model = self.config.cloud_model  # reset for next request
            self._apply_scanner(True)  # restore default for the next request

        taint = sorted(ctx.data_labels)

        # Tool gate: a denied tool comes back as a "POLICY DENIED" tool message.
        # The final AgentResponse has tool_results=None on a normal completion
        # (the executor only sets it on tool-calling turns), so we scan the full
        # conversation (resp.messages) — which carries every tool-role message,
        # including the denied one from an earlier turn.
        tools_called: list[str] = []
        for m in self._iter_messages(resp):
            role = self._mfield(m, "role")
            content = str(self._mfield(m, "content") or "")
            if role == "tool":
                if "POLICY DENIED" in content:
                    gate_events.append(f"🛡️ TOOL — blocked: {content.strip()[:200]}")
            # assistant turns carry the tool_calls that were issued
            for tc in self._mfield(m, "tool_calls") or []:
                name = getattr(getattr(tc, "function", None), "name", None) or (
                    tc.get("function", {}).get("name") if isinstance(tc, dict) else None
                )
                if name:
                    tools_called.append(name)

        if taint and model_used == self.config.onprem_model:
            pass  # the model-routing event was already logged
        elif taint:
            gate_events.append(f"ℹ️ run tainted {taint} but completed on {model_used}")

        # Long-term memory gate (UI signal only). The ACTUAL write happens once,
        # in the SDK's session auto-store (store_memories=True) — which is also
        # gated, so a PHI turn is blocked there. We do NOT write here (that caused
        # a duplicate). This is a read-only policy check that mirrors the gate's
        # decision so the UI can show it per turn.
        if taint and self.memory_client() is not None:
            decision = self._policy_store.check(
                [self.config.agent_name, *sorted(taint)], f"memory:{user_id}"
            )
            if not decision.allowed:
                gate_events.append(
                    "🛡️ MEMORY — write blocked: sensitive run not persisted to long-term memory"
                )

        # Output scanner (SDK hook): the scanner runs inside the SDK over the FINAL
        # answer. Its marker appearing in resp.content is the observable signal that
        # it fired this turn — independent of the data-label gates above.
        if "[SSN REDACTED]" in (resp.content or ""):
            gate_events.append(
                "🧹 OUTPUT SCANNER — SSN masked in the visible answer (pattern-based PII redaction)"
            )

        return {
            "response": resp.content or "",
            "taint": taint,
            "model_used": model_used,
            "gate_events": gate_events,
            "tools_called": tools_called,
        }

    async def _consume_stream(
        self, message: str, model: str, user_id: str, conversation_id: str, session_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """Consume ONE run_stream attempt on `model`. Yields live UI events
        (token / message / tool) and, as its last item, a ``__final__`` event
        carrying the turn's accumulated content, tools called, and tool-block
        gate messages. Raises ModelAccessDeniedError if the model-routing gate
        denies this model mid-run (the PHI taint tripped) — the caller catches
        it and re-streams on the on-prem model.

        Note: because the agent has an output scanner configured, the SDK
        suppresses raw token deltas (a token might still hold an un-redacted
        SSN) and emits one sanitized CONTENT_COMPLETE per turn — so the visible
        stream arrives per-turn (``message``), not per-token (``token``)."""
        self._agent.model = model
        content = ""
        tools_called: list[str] = []
        tool_blocks: list[str] = []
        async with aclosing(
            self._runner.run_stream(
                agent=self._agent,
                input=message,
                session_id=session_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        ) as stream:
            async for ev in stream:
                if ev.type == EventType.CONTENT_DELTA:
                    delta = ev.data.get("content", "")
                    content += delta
                    yield {"type": "token", "text": delta}
                elif ev.type == EventType.CONTENT_COMPLETE:
                    # Scanner active → full sanitized per-turn message; the final
                    # turn's CONTENT_COMPLETE is the visible answer.
                    content = ev.data.get("content", "")
                    yield {"type": "message", "text": content}
                elif ev.type == EventType.TOOL_CALL_START:
                    name = ev.data.get("tool_name", "")
                    if name:
                        tools_called.append(name)
                        yield {"type": "tool", "name": name}
                elif ev.type == EventType.TOOL_CALL_END:
                    result = str(ev.data.get("result", ""))
                    if "POLICY DENIED" in result:
                        msg = f"🛡️ TOOL — blocked: {result.strip()[:200]}"
                        tool_blocks.append(msg)
                        # Yield the block LIVE (not only in __final__): on the cloud
                        # attempt the model-routing deny raises before __final__ is
                        # reached, so a block captured only there would be lost on the
                        # reroute. Emitting it live lets chat_stream collect it before
                        # the exception propagates.
                        yield {"type": "gate", "text": msg}
        yield {
            "type": "__final__",
            "content": content,
            "tools_called": tools_called,
            "tool_blocks": tool_blocks,
        }

    async def chat_stream(
        self, message: str, user_id: str, conversation_id: str, scanner_on: bool = True
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming twin of chat(): the SAME enforcement, observed live.

        The cloud attempt streams until the PHI taint trips the model-routing
        deny (ModelAccessDeniedError) mid-run; we then emit a reroute event and
        re-stream on the on-prem model — the streaming analogue of chat()'s
        try/except fallback. The closing ``done`` event carries the same
        glassbox payload as chat() so the UI panels render identically.

        ``scanner_on`` toggles the output scanner: ON → sanitized per-turn
        ``message`` events (no token typing); OFF → live ``token`` deltas
        (visible typing, unredacted). See _apply_scanner."""
        if not self._initialized:
            await self.initialize()

        session_id = await self._ensure_session(user_id, conversation_id)
        self._apply_scanner(scanner_on)
        model_used = self.config.cloud_model
        final: dict[str, Any] = {"content": "", "tools_called": [], "tool_blocks": []}

        # Gate events and tools are collected across BOTH attempts (cloud, then
        # on-prem) and de-duplicated. A tool block detected on the cloud attempt
        # must survive the reroute even though that attempt raises before its
        # __final__ — so we fold in every block as we see it, from either run.
        gate_events: list[str] = []
        tools_called: list[str] = []
        seen_gates: set[str] = set()
        seen_tools: set[str] = set()

        def _add_gate(msg: str) -> None:
            if msg not in seen_gates:
                seen_gates.add(msg)
                gate_events.append(msg)

        async def _drain(model: str):
            nonlocal final
            async for ev in self._consume_stream(
                message, model, user_id, conversation_id, session_id
            ):
                if ev["type"] == "__final__":
                    final = ev
                    for b in ev["tool_blocks"]:
                        _add_gate(b)
                    for t in ev["tools_called"]:
                        if t not in seen_tools:
                            seen_tools.add(t)
                            tools_called.append(t)
                elif ev["type"] == "tool":
                    if ev["name"] not in seen_tools:
                        seen_tools.add(ev["name"])
                        tools_called.append(ev["name"])
                    yield ev
                elif ev["type"] == "gate":
                    _add_gate(ev["text"])  # tool block, collected live
                    yield ev
                else:
                    yield ev

        try:
            try:
                async for ev in _drain(self.config.cloud_model):
                    yield ev
            except ModelAccessDeniedError as e:
                # PHI taint tripped the cloud-model deny mid-run. Re-stream on the
                # PHI-approved on-prem model (the compliant fallback). Anything the
                # cloud attempt already surfaced (tool blocks, tools) is retained
                # above; de-dup keeps the on-prem re-run from doubling it.
                reroute = (
                    f"🛡️ MODEL ROUTING — cloud '{self.config.cloud_model}' DENIED for PHI "
                    f"(policy '{e.context.get('policy_name')}'). Re-routing to on-prem."
                )
                _add_gate(reroute)
                yield {"type": "reroute", "text": reroute}
                model_used = self.config.onprem_model
                async for ev in _drain(self.config.onprem_model):
                    yield ev
        finally:
            self._agent.model = self.config.cloud_model  # reset for next request
            self._apply_scanner(True)  # restore default for the next request

        # Aggregate the glassbox state (mirrors chat()). In this clinic the only
        # taint source is PHI, and any PHI taint denies the cloud model — so a
        # reroute to on-prem is exactly the signal that the run was tainted.
        tainted = model_used == self.config.onprem_model
        taint = [PHI] if tainted else []

        if tainted and self.memory_client() is not None:
            decision = self._policy_store.check(
                [self.config.agent_name, *taint], f"memory:{user_id}"
            )
            if not decision.allowed:
                gate_events.append(
                    "🛡️ MEMORY — write blocked: sensitive run not persisted to long-term memory"
                )

        if "[SSN REDACTED]" in (final["content"] or ""):
            gate_events.append(
                "🧹 OUTPUT SCANNER — SSN masked in the visible answer (pattern-based PII redaction)"
            )

        yield {
            "type": "done",
            "response": final["content"] or "",
            "taint": taint,
            "model_used": model_used,
            "gate_events": gate_events,
            "tools_called": tools_called,
        }

    @staticmethod
    def _iter_messages(resp: Any) -> list[Any]:
        """Best-effort access to a run's full message list across SDK versions."""
        return getattr(resp, "messages", None) or []

    @staticmethod
    def _mfield(m: Any, key: str) -> Any:
        """Read a field from a message that may be a dict or a ChatMessage."""
        if isinstance(m, dict):
            return m.get(key)
        return getattr(m, key, None)

    def memory_client(self):
        """The enabled long-term memory client, or None (needs a vector store)."""
        client = self._container.memory_client if self._container else None
        return client if (client and client.is_enabled) else None

    async def attempt_memory_write(
        self, text: str, labels: list[str], user_id: str = "u1"
    ) -> dict[str, Any]:
        """Demonstrate the MEMORY-WRITE gate: try to persist `text` to the USER
        scope carrying `labels`. With PHI the policy ``phi-never-persisted``
        (memory:*) denies it; without labels it is stored and becomes visible in
        the long-term-memory panel.

        Only meaningful when memory is enabled (needs mem0 + a vector store);
        otherwise we report skipped — the gate runs after _ensure_enabled().
        """
        client = self.memory_client()
        if client is None:
            return {
                "ok": False,
                "skipped": True,
                "reason": "memory not enabled (start Milvus + MEMORY_ENABLED=true)",
            }
        try:
            await client.add(
                text,
                user_id=user_id,
                policy_store=self._policy_store,
                subject=self.config.agent_name,
                data_labels=set(labels),
            )
            return {"ok": True, "denied": False, "stored": text}
        except MemoryAccessDeniedError as e:
            return {"ok": True, "denied": True, "policy_name": e.context.get("policy_name")}

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

    @property
    def policy_store(self):
        return self._policy_store
