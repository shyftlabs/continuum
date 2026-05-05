"""
Tool-attention router — selects which tool schemas to expose each LLM turn.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from orchestrator.logging import get_logger
from orchestrator.tools.tool_attention.config import ToolAttentionConfig
from orchestrator.tools.tool_attention.registry import ToolSummaryRegistry

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import RunContext

logger = get_logger(__name__)

_BUILTIN_ALWAYS_PROMOTE = {"think"}

# Last-run debug snapshot — updated each time apply_tool_attention fires.
# Playground/web debug endpoint reads from this.
_last_run_debug: dict = {}


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return tool.get("function", {}).get("name", "")
    return getattr(getattr(tool, "function", None), "name", "")


def _tool_one_liner(tool: Any) -> str:
    """Return 'name: description' for a single tool."""
    if isinstance(tool, dict):
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
    else:
        fn = tool.function
        name = fn.name
        desc = fn.description or ""
    return f"{name}: {desc}" if name else ""


def _build_summary_text(all_tools: list[Any]) -> str:
    """One-liner per tool — stable across turns for prompt caching."""
    lines = [_tool_one_liner(t) for t in all_tools]
    body = "\n".join(f"- {l}" for l in lines if l)
    return (
        "[Available tools]\n" + body + "\n\n"
        "If a tool you need is listed above but its parameters are not available, "
        "output: NEED_TOOL:<tool_name>"
    )


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude") or model.startswith("anthropic/")


def _extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Return the most recent user message text."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


class ToolAttentionRouter:
    """
    Per-agent router that filters tool schemas each turn via Milvus semantic search.

    Lifecycle:
        router = ToolAttentionRouter(config)
        await router.initialize(all_tools)   # once at agent startup
        filtered = router.route(messages, all_tools, context)  # each turn
    """

    def __init__(self, config: ToolAttentionConfig) -> None:
        self._config = config
        self._registry = ToolSummaryRegistry(config)
        self._initialized = False

    async def initialize(self, tool_defs: list[Any]) -> None:
        await self._registry.initialize(tool_defs)
        self._initialized = True

    def route(
        self,
        messages: list[dict[str, Any]],
        all_tools: list[dict[str, Any]],
        context: RunContext,
    ) -> list[dict[str, Any]] | None:
        """
        Select tools to expose this turn.

        Returns None when routing is skipped so the caller falls back to all tools.
        Never raises — errors return None (fail-open).
        """
        if not self._initialized or not self._registry.ready:
            return None

        total = len(all_tools)
        if total < self._config.min_tools:
            return None

        query = _extract_user_query(messages)
        if not query:
            return None

        k = min(self._config.k, total)

        # Semantic search
        routed = set(self._registry.search(query, k))

        # Always-promote: builtins + config + handoff tools (transfer_to_*)
        always: set[str] = _BUILTIN_ALWAYS_PROMOTE | set(self._config.always_promote)
        for name in (_tool_name(t) for t in all_tools):
            if name.startswith("transfer_to_"):
                always.add(name)

        promoted = routed | always

        # Persist for hallucination gate (tool_service reads this)
        # Also persist Phase 1 summary text (stable across turns)
        if context.metadata is not None:
            context.metadata["promoted_tools"] = promoted
            context.metadata["tool_summary_text"] = _build_summary_text(all_tools)

        result = [t for t in all_tools if _tool_name(t) in promoted]

        logger.info(
            "tool-attention: %d/%d tools promoted — query=%r routed=%s always=%s",
            len(result),
            total,
            query[:60],
            sorted(routed),
            sorted(always),
        )
        return result


async def apply_tool_attention(
    agent: BaseAgent,
    messages: list[dict[str, Any]],
    context: RunContext,
) -> list[dict[str, Any]] | None:
    """
    Entry point called by Executor and StreamExecutor each turn.

    Returns filtered tool list, or None to signal "use all tools".
    Never raises.
    """
    if not agent.config or not agent.config.tool_attention:
        return None

    # Lazy-init: create router once, store on agent.metadata so it survives turns
    router: ToolAttentionRouter | None = agent.metadata.get("_tool_attention_router")
    if router is None:
        router = ToolAttentionRouter(agent.config.tool_attention)
        all_tools = agent.get_tools_for_llm()
        if all_tools:
            await router.initialize(all_tools)
        agent.metadata["_tool_attention_router"] = router

    if not router._initialized:
        return None

    all_tools = agent.get_tools_for_llm()
    if not all_tools:
        return None

    try:
        filtered = await asyncio.to_thread(router.route, messages, all_tools, context)
    except Exception as e:
        logger.warning(f"tool-attention routing error (using all tools): {e}")
        return None

    # Phase 1: build summary message so the LLM sees all tool names each turn.
    # For Anthropic: add cache_control so the stable prefix gets prompt-cached.
    phase1_injected = False
    if filtered is not None and context.metadata is not None:
        summary_text = context.metadata.get("tool_summary_text", "")
        if summary_text:
            model = getattr(agent, "model", "") or ""
            msg: dict[str, Any] = {"role": "system", "content": summary_text}
            if _is_anthropic(model):
                msg["cache_control"] = {"type": "ephemeral"}
            context.metadata["tool_summary_message"] = msg
            phase1_injected = True

    # Update debug snapshot for playground inspection
    _last_run_debug.update({
        "router_active": filtered is not None,
        "total_tools": len(all_tools),
        "promoted_tools": sorted(t.get("function", {}).get("name", "") for t in (filtered or [])),
        "phase1_injected": phase1_injected,
        "phase1_preview": (context.metadata or {}).get("tool_summary_text", "")[:300] if phase1_injected else "",
        "phase1_has_cache_control": _is_anthropic(getattr(agent, "model", "") or ""),
    })

    return filtered
