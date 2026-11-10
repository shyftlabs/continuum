"""After-tool-round hook: let a product steer the agent once its tools have answered.

A common failure in tool-using agents: every lookup comes back empty, and the model
answers "nothing found" although another tool would have had the answer. The SDK
cannot know which results count as empty or which tool to try next -- that is the
product's knowledge -- so it offers the seam and the product supplies the rule::

    def dead_end(round: ToolRound) -> str | None:
        if all(r.is_empty(("no_results_found",)) for r in round.history):
            unused = {"vector_search", "sql_query"} - round.tools_used
            if unused:
                return f"Those lookups returned nothing usable. Call {', '.join(sorted(unused))} now."
        return None

    AgentConfig(after_tool_round=dead_end)

A returned string is added as a system message before the next model call, at most
``after_tool_round_max_nudges`` times per run (default 1). System messages are not
saved to the session, so the nudge steers this run without entering chat history.
A hook that raises is logged and ignored.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from continuum.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolRoundResult:
    """One tool call and what it returned."""

    name: str
    arguments: dict[str, Any]
    content: str

    def is_empty(self, markers: Iterable[str] = ()) -> bool:
        """True when the result is blank, ``[]``/``{}``, or contains one of ``markers``."""
        text = (self.content or "").strip()
        if text in ("", "[]", "{}", "null", "None"):
            return True
        return any(marker in text for marker in markers)


@dataclass
class ToolRound:
    """What the hook sees after a round of tool calls."""

    agent_name: str
    turn: int
    results: list[ToolRoundResult]
    history: list[ToolRoundResult] = field(default_factory=list)
    nudges_sent: int = 0

    @property
    def tools_used(self) -> set[str]:
        return {r.name for r in self.history}


AfterToolRound = Callable[[ToolRound], "str | None | Awaitable[str | None]"]


def _call_name_args_id(tc: Any) -> tuple[str, dict[str, Any], str]:
    if hasattr(tc, "function"):
        name, raw, tc_id = tc.function.name, tc.function.arguments, tc.id
    else:
        fn = tc.get("function", {})
        name, raw, tc_id = fn.get("name", ""), fn.get("arguments", "{}"), tc.get("id", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (ValueError, TypeError):
        args = {"_raw": raw}
    return name or "", args if isinstance(args, dict) else {"_raw": args}, tc_id or ""


class ToolRoundTracker:
    """Per-run state for the after-tool-round hook."""

    def __init__(self, agent: Any):
        config = getattr(agent, "config", None)
        self._agent_name = getattr(agent, "name", "")
        self._hook: AfterToolRound | None = getattr(config, "after_tool_round", None)
        self._max = int(getattr(config, "after_tool_round_max_nudges", 1) or 0)
        self._history: list[ToolRoundResult] = []
        self._nudges = 0

    async def after_round(
        self, turn: int, tool_calls: list[Any], messages: list[dict[str, Any]]
    ) -> str | None:
        """Record this round's results; if the hook returns guidance, append it to
        ``messages`` as a system message and return it."""
        if self._hook is None:
            return None
        by_id = {
            m.get("tool_call_id"): m
            for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        results = []
        for tc in tool_calls:
            name, args, tc_id = _call_name_args_id(tc)
            reply = by_id.get(tc_id)
            if reply is None:
                continue  # handoffs and internal calls have no tool result
            content = reply.get("content")
            results.append(
                ToolRoundResult(
                    name=name,
                    arguments=args,
                    content=content if isinstance(content, str) else json.dumps(content),
                )
            )
        if not results:
            return None
        self._history.extend(results)
        if self._nudges >= self._max:
            return None
        round_ = ToolRound(
            agent_name=self._agent_name,
            turn=turn,
            results=results,
            history=list(self._history),
            nudges_sent=self._nudges,
        )
        try:
            nudge = self._hook(round_)
            if inspect.isawaitable(nudge):
                nudge = await nudge
        except Exception as e:  # noqa: BLE001 — a product hook must not break the run
            logger.warning("after_tool_round hook failed for agent %s: %s", self._agent_name, e)
            return None
        if not nudge or not isinstance(nudge, str):
            return None
        self._nudges += 1
        messages.append({"role": "system", "content": nudge})
        logger.info(
            "after_tool_round: agent %s steered after turn %s (%s of %s)",
            self._agent_name,
            turn,
            self._nudges,
            self._max,
        )
        return nudge
