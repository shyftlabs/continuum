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
``after_tool_round_max_nudges`` times per run (default 1). The hook is still called
after every round once that cap is reached; what it returns is then ignored, so a
rule can keep observing the run (``nudges_sent`` says how many went in). System messages are not
saved to the session, so the nudge steers this run without entering chat history.
A hook that raises is logged and ignored.

The hook sees the same rounds in ``run`` and ``run_stream``: internal calls
(``think``, the Headroom retrieve, handoffs) are left out of a round, and a round
made only of them does not call the hook.
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


def _as_text(content: Any) -> str:
    """A tool result as text for the hook. Never raises: a result that is not
    JSON-serialisable must not fail the run."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except (TypeError, ValueError):  # e.g. a circular reference
        return str(content)


def _round_replies(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool messages of the latest round: those after the last assistant
    message with tool calls (every tool message if there is none)."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            start = i + 1
            break
    return [m for m in messages[start:] if isinstance(m, dict) and m.get("role") == "tool"]


def _is_internal_call(agent: Any, name: str) -> bool:
    """``think``, the Headroom retrieve and handoffs are plumbing, not lookups."""
    from continuum.llm.headroom.compressor import RETRIEVE_TOOL_NAME

    if name in ("think", RETRIEVE_TOOL_NAME):
        return True
    check = getattr(agent, "is_handoff_tool_call", None)
    if check is None:
        return False
    is_handoff, target = check(name)
    return bool(is_handoff and target)


class ToolRoundTracker:
    """Per-run state for the after-tool-round hook."""

    def __init__(self, agent: Any):
        config = getattr(agent, "config", None)
        self._agent = agent
        self._agent_name = getattr(agent, "name", "")
        self._hook: AfterToolRound | None = getattr(config, "after_tool_round", None)
        # Read only when a hook is set, so a bad value cannot fail agents that
        # do not use the hook.
        self._max = (
            int(getattr(config, "after_tool_round_max_nudges", 1) or 0)
            if self._hook is not None
            else 0
        )
        self._history: list[ToolRoundResult] = []
        self._nudges = 0

    async def after_round(
        self, turn: int, tool_calls: list[Any], messages: list[dict[str, Any]]
    ) -> str | None:
        """Record this round's results; if the hook returns guidance, append it to
        ``messages`` as a system message and return it."""
        if self._hook is None:
            return None
        # Pair each call with its own reply, so two calls that share an id (or
        # have none) each get theirs. Replies with one id are taken from the
        # end, last call first: results come in call order, and in run the
        # think / retrieve replies sit before the regular tools' results.
        by_id: dict[str, list[dict[str, Any]]] = {}
        for m in _round_replies(messages):
            by_id.setdefault(m.get("tool_call_id") or "", []).append(m)
        calls = [_call_name_args_id(tc) for tc in tool_calls]
        replies: list[dict[str, Any] | None] = [None] * len(calls)
        for i in range(len(calls) - 1, -1, -1):
            same_id = by_id.get(calls[i][2])
            if same_id:
                replies[i] = same_id.pop()
        results = []
        for (name, args, _tc_id), reply in zip(calls, replies, strict=True):
            # Left out here, once, so run (whose executor already drops them)
            # and run_stream (which passes every call) hand the hook the same
            # rounds: their results are not lookup results.
            if _is_internal_call(self._agent, name):
                continue
            if reply is None:
                continue  # no tool result (e.g. a call that was not executed)
            results.append(
                ToolRoundResult(name=name, arguments=args, content=_as_text(reply.get("content")))
            )
        if not results:
            return None
        self._history.extend(results)
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
        if self._nudges >= self._max:
            # Cap reached: the hook still observes every round, but its
            # guidance no longer goes in.
            logger.debug(
                "after_tool_round: agent %s nudge after turn %s ignored (cap %s reached)",
                self._agent_name,
                turn,
                self._max,
            )
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
