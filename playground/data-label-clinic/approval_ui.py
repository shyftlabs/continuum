"""The `ask` mode: an approval prompt that reaches the browser (finding F7).

The SDK's gate calls an async handler and waits for a decision. Getting that
prompt in front of a person, over HTTP, is the part an application has to solve,
and this is the smallest honest version of it.

HOW IT WORKS, AND WHY IT LOOKS LIKE THIS

``POST /chat`` is already in flight and blocked inside the tool executor when the
handler runs, so the decision cannot come back on that connection. The browser
opens a SECOND request: it polls ``/approval/pending`` while the chat request is
still open, renders whatever it finds, and posts the answer to
``/approval/decide``. The handler is parked on an ``asyncio.Future`` that the
decide endpoint resolves.

That is the shape of every blocking-approval UI over HTTP, and it is worth
seeing plainly because it is also the shape's limitation: the chat request stays
open the whole time. The SDK's ``approval_timeout`` (30s by default) is what
keeps that inside ordinary browser and proxy limits, and when it expires the
gate fails closed. A reviewer who needs longer needs Temporal or a
refuse-and-resume flow instead -- not a bigger timeout.

The pending map is process-local and deliberately not persisted. A restart loses
in-flight prompts, which is correct for a demo and exactly what a durable
implementation would have to fix.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from continuum.agent.approval import ToolApprovalDecision, ToolApprovalRequest

# request_id -> {"request": ..., "future": ...}
_PENDING: dict[str, dict[str, Any]] = {}


async def ui_approval_handler(request: ToolApprovalRequest) -> ToolApprovalDecision:
    """Park the call until the browser answers, or the SDK's timeout fires.

    No timeout of its own: ``request_approval`` already bounds this with
    ``approval_timeout`` and denies when it expires. A second timeout here would
    be a second answer to the same question, and the two would drift.
    """

    request_id = uuid.uuid4().hex[:8]
    future: asyncio.Future[ToolApprovalDecision] = asyncio.get_running_loop().create_future()
    _PENDING[request_id] = {"request": request, "future": future}
    try:
        return await future
    finally:
        # Also runs when the SDK cancels this on timeout, so an unanswered
        # prompt does not sit in the panel forever claiming to be live.
        _PENDING.pop(request_id, None)


def pending_approvals() -> list[dict[str, Any]]:
    """What the browser should show. Arguments included on purpose: a reviewer
    shown only a tool name is approving the name, not the action."""
    return [
        {
            "request_id": rid,
            "tool_name": entry["request"].tool_name,
            "arguments": entry["request"].arguments,
            "agent_name": entry["request"].agent_name,
            "data_labels": sorted(entry["request"].data_labels),
        }
        for rid, entry in _PENDING.items()
        if not entry["future"].done()
    ]


# ── refuse-and-resume: the queue ────────────────────────────────────────────
#
# `ask` blocks the turn while a reviewer answers, which needs the HTTP request
# to stay open and so needs the reviewer to be watching. `queue` is the other
# shape: the call is parked, the turn ends NOW telling the user it is pending,
# somebody answers whenever they get to it, and the user asks again.
#
# Keyed by (tool, arguments) rather than by request id, because the second turn
# is a different run asking the same question -- there is no id to carry over.
# The SDK deliberately remembers nothing between runs (a cached approval could
# authorise an execution nobody saw), so an app that wants resumption supplies
# the memory and decides its own rules. This is the smallest such store.

_QUEUE: dict[str, dict[str, Any]] = {}


def _queue_key(tool_name: str, arguments: dict[str, Any]) -> str:
    import json

    return f"{tool_name}::{json.dumps(arguments, sort_keys=True)}"


async def queue_approval_handler(request: ToolApprovalRequest) -> ToolApprovalDecision:
    """Defer on the first ask; honour the answer on a later one.

    Returns immediately either way, so the turn never holds a request open.
    """
    from continuum.agent.approval import ToolApprovalDecision

    key = _queue_key(request.tool_name, request.arguments)
    entry = _QUEUE.get(key)

    if entry is not None and entry.get("decided") is not None:
        approved = bool(entry["decided"])
        _QUEUE.pop(key, None)  # one answer authorises one execution, not a standing permit
        return ToolApprovalDecision(
            approved=approved,
            reviewer=entry.get("reviewer"),
            reason=None if approved else "A reviewer declined this action.",
        )

    _QUEUE.setdefault(
        key,
        {
            "tool_name": request.tool_name,
            "arguments": request.arguments,
            "data_labels": sorted(request.data_labels),
            "decided": None,
            "reviewer": None,
        },
    )
    return ToolApprovalDecision(
        approved=False,
        deferred=True,
        reason="Queued for a reviewer. Ask again once it has been answered.",
    )


def queued_approvals() -> list[dict[str, Any]]:
    return [
        {"key": k, **{x: v[x] for x in ("tool_name", "arguments", "data_labels", "decided")}}
        for k, v in _QUEUE.items()
    ]


def answer_queued(key: str, approved: bool, reviewer: str = "ui") -> bool:
    entry = _QUEUE.get(key)
    if entry is None:
        return False
    entry["decided"] = approved
    entry["reviewer"] = reviewer
    return True


def submit_decision(request_id: str, approved: bool, reviewer: str = "ui") -> bool:
    """Resolve a waiting handler. False when there is nothing to resolve --
    already answered, or already timed out and cleaned up."""
    entry = _PENDING.get(request_id)
    if entry is None or entry["future"].done():
        return False

    from continuum.agent.approval import ToolApprovalDecision

    entry["future"].set_result(
        ToolApprovalDecision(
            approved=approved,
            reviewer=reviewer,
            reason=None if approved else "A reviewer declined this action.",
        )
    )
    return True
