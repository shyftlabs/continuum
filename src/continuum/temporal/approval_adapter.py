"""Reach a Temporal reviewer from inside a tool call (security finding F7).

``AgentConfig.approval_handler`` is called when a declared tool is about to run,
and waits for a decision. The in-process handlers -- a CLI prompt, a web modal --
hold the caller's connection open, so they are bounded by whatever a browser or
proxy tolerates. This one is for a reviewer who will answer in an hour.

HOW IT WORKS

Under Temporal the tool call runs inside an ACTIVITY, which cannot touch workflow
state directly: signals and queries are the only doors. So the handler signals
``request_tool_approval`` to register the request, then polls
``get_approval_decision`` until someone answers, heartbeating so Temporal does
not kill the activity mid-wait.

The answer comes back through the workflow's existing ``submit_approval`` signal,
so the reviewer's side is unchanged -- the same UI, the same
``HumanInLoopManager``, the same allow-list check via ``is_authorized``. A tool
approval is not a second, weaker approval system living beside the first.

WHAT THIS BUYS, AND WHAT IT DOES NOT

The whole agent turn is one activity (``run_agent_activity`` calls
``runner.run()``), so the workflow cannot pause *between* the agent's own steps.
What this gives is that the activity blocks IN PLACE: work done before the gate
is not repeated, and the user does not have to ask again. It does NOT survive a
worker restart -- a retried activity starts from the beginning and redoes
everything.

That limit is why an unreachable workflow degrades to **deferred** rather than
denied. Deferred is the truthful outcome for "this did not happen and can be
asked again", and it is what the queue path already means; denied would tell the
user a reviewer refused them when none was ever reached.

Two things are deliberately NOT bounded here. There is no timeout of its own:
``request_approval`` already applies ``approval_timeout`` and denies when it
expires, and a second clock would be a second answer to the same question. And
there is no memory between runs: a retried run asks again, because a remembered
approval would authorise an execution the reviewer never saw.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from continuum.agent.approval import (
        ToolApprovalDecision,
        ToolApprovalHandler,
        ToolApprovalRequest,
    )

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 2.0


def _heartbeat(message: str) -> None:
    """Tell Temporal the activity is alive. A no-op outside one.

    Without this a wait longer than ``heartbeat_timeout`` (60s in the shipped
    workflow) has the activity killed and retried, so the reviewer's answer
    arrives to nobody and the turn starts over.
    """
    try:
        from temporalio import activity

        activity.heartbeat(message)
    except Exception:
        # Not in an activity, or temporalio absent. Waiting is still correct;
        # only the liveness signal is unavailable.
        pass


def _activity_workflow_id() -> str | None:
    """The workflow this activity belongs to, or None outside an activity."""
    try:
        from temporalio import activity

        return str(activity.info().workflow_id)
    except Exception:
        return None


def _temporal_client() -> Any:
    from continuum.temporal.client import get_temporal_client

    return get_temporal_client()


def temporal_tool_approval(
    *,
    approvers: list[str] | None = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> ToolApprovalHandler:
    """A handler that finds its own workflow at call time.

    An agent is built long before any workflow exists, so it cannot be handed a
    handle at construction. This resolves one per call: the id from
    ``activity.info()``, the handle from the global Temporal client.

    Outside an activity -- a local script, a test, an HTTP server -- it DEFERS.
    Not approves: the action would then proceed unreviewed by the very
    configuration that asked for review. Not denies either: nobody refused it,
    and deferring says truthfully that it did not happen and can be asked again.
    """

    async def handler(request: ToolApprovalRequest) -> ToolApprovalDecision:
        from continuum.agent.approval import ToolApprovalDecision

        workflow_id = _activity_workflow_id()
        if workflow_id is None:
            logger.error(
                "temporal_tool_approval is configured but this run is not inside a "
                "Temporal activity, so no workflow can be asked — deferring '%s'",
                request.tool_name,
            )
            return ToolApprovalDecision(
                approved=False,
                deferred=True,
                reason=(
                    "This run is not inside a Temporal activity, so no reviewer could "
                    "be reached. Use an in-process approval handler here."
                ),
            )

        try:
            handle = await _temporal_client().get_workflow_handle(workflow_id)
        except Exception as e:
            logger.error(
                "Could not reach the Temporal workflow %s (%s: %s) — deferring",
                workflow_id,
                type(e).__name__,
                e,
            )
            return ToolApprovalDecision(
                approved=False,
                deferred=True,
                reason=(
                    f"The approval workflow could not be reached ({e}). If this says "
                    "'Not connected', the GLOBAL Temporal client needs connecting — a "
                    "worker's own client is a different one: "
                    "await get_temporal_client().connect(host)."
                ),
            )

        inner = temporal_approval_handler(handle, approvers=approvers, poll_interval=poll_interval)
        return await inner(request)

    return handler


def temporal_approval_handler(
    workflow_handle: Any,
    *,
    approvers: list[str] | None = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    heartbeat: Callable[[str], None] = _heartbeat,
) -> ToolApprovalHandler:
    """Build a handler that asks a Temporal workflow's reviewers.

    ``workflow_handle`` is a Temporal client handle for the workflow running this
    agent -- whatever ``client.get_workflow_handle(workflow_id)`` returns.

    ``approvers`` is the allow-list. Empty means anyone may decide, matching the
    planned approval step's rule. It is enforced workflow-side, by the same
    ``is_authorized``: sending it from here is how the workflow learns it, not
    where it is checked.
    """
    if workflow_handle is None:
        # A handler that silently does nothing is the failure this finding is
        # about. Refuse at build time, where the message can name the cause,
        # rather than at 3am inside a tool call.
        raise ValueError(
            "temporal_approval_handler needs a workflow handle to signal. Pass "
            "client.get_workflow_handle(workflow_id) for the workflow running this agent."
        )

    allow_list = list(approvers or [])

    async def handler(request: ToolApprovalRequest) -> ToolApprovalDecision:
        from continuum.agent.approval import ToolApprovalDecision

        request_id = f"tool-{uuid.uuid4().hex[:12]}"
        try:
            arguments = json.dumps(request.arguments, default=str, sort_keys=True)
        except Exception:  # pragma: no cover - defensive; arguments came from JSON
            arguments = str(request.arguments)

        try:
            await workflow_handle.signal(
                "request_tool_approval",
                {
                    "request_id": request_id,
                    "description": request.tool_name,
                    # The ARGUMENTS. A reviewer shown only a tool name is
                    # approving the name, and neither the policy gate nor
                    # tool-trust can see this payload.
                    "context": arguments,
                    "approvers": allow_list,
                    "data_labels": sorted(request.data_labels),
                },
            )
        except Exception as e:
            logger.error(
                "Could not register tool approval with the workflow (%s: %s) — deferring",
                type(e).__name__,
                e,
            )
            return ToolApprovalDecision(
                approved=False,
                deferred=True,
                reason=f"The approval workflow could not be reached ({e}).",
            )

        # Poll until answered. No timeout here on purpose: request_approval
        # applies approval_timeout and cancels this, and a second clock would be
        # a second answer to the same question.
        while True:
            heartbeat(f"awaiting approval for {request.tool_name}")
            try:
                result = await workflow_handle.query("get_approval_decision", request_id)
            except asyncio.CancelledError:
                # The SDK's timeout firing. Must propagate: swallowing it would
                # leave this polling a workflow nobody is waiting on.
                raise
            except Exception as e:
                logger.error(
                    "Lost contact with the approval workflow (%s: %s) — deferring",
                    type(e).__name__,
                    e,
                )
                return ToolApprovalDecision(
                    approved=False,
                    deferred=True,
                    reason=f"Lost contact with the approval workflow ({e}).",
                )

            status = (result or {}).get("status")
            if status == "approved":
                return ToolApprovalDecision(
                    approved=True, reviewer=(result or {}).get("decided_by")
                )
            if status == "rejected":
                return ToolApprovalDecision(
                    approved=False,
                    reviewer=(result or {}).get("decided_by"),
                    reason=(result or {}).get("reason") or "A reviewer declined this action.",
                )
            if status == "unknown":
                # The workflow never saw the request -- a lost signal, or a
                # restarted workflow. Silence is not permission.
                logger.error(
                    "The workflow does not know approval request %s — deferring", request_id
                )
                return ToolApprovalDecision(
                    approved=False,
                    deferred=True,
                    reason="The approval request did not reach a reviewer.",
                )

            await asyncio.sleep(poll_interval)

    return handler
