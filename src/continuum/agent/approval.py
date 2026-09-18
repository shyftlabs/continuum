"""A human-in-the-loop gate before a declared tool call (security finding F7).

An approval workflow already existed -- ``temporal/human_in_loop.py``,
``ApprovalRequest``, anti-spoofing checks -- but it fired only for a Temporal
workflow step of type ``"approval"``. The default ``AgentRunner.run()`` path,
the one every quick-start uses, had no gate before any tool call.

WHY THE EXISTING CONTROLS DO NOT COVER IT

Tool-trust (F3) asks *is this server serving the catalogue I vetted?* It is
answered once, offline, per server. Once ``delete_records`` is approved it can
be called a thousand times with any arguments and tool-trust is satisfied,
because the catalogue has not changed -- which is all it ever claimed to check.

The policy gate (F5/F6) asks *is this run allowed this resource?* It checks
``tool:{name}`` and decides by rule, with no human. It is binary: allow always,
or deny always. It cannot express "allow, but ask someone first", which is the
whole point for an action that is legitimate but consequential.

So neither can tell these apart::

    transfer_funds(amount=5)            # fine
    transfer_funds(amount=5_000_000)    # ask someone

This gate is the only one that receives the ARGUMENTS. That is the capability
being added, not "a second opinion on the tool".

WHAT THIS MODULE DELIBERATELY DOES NOT DO

It ships no default list of risky tool names. ``delete_*``, ``send_*``,
``pay_*`` look like a reasonable default and are not one: a name-matcher blocks
a harmless ``send_receipt`` while missing ``wire_funds``, and an operator who
sees a gate they did not configure assumes more coverage than they have. The
operator declares which tools need a person; the runtime enforces it. Same
stance as ``tool_data_labels`` (no PII detector) and ``pre_store_filter`` (no
content classifier).

THREE LIMITS, STATED RATHER THAN DISCOVERED

*Blocking a run on a human breaks HTTP-shaped deployments.* The request stays
open while the reviewer thinks, and browsers, proxies and serverless platforms
all give up long before a person does. ``approval_timeout`` defaults to 30s so
it sits inside ordinary limits and fails closed; a wait measured in hours needs
Temporal (which is what ``human_in_loop.py`` was built for) or a refuse-and-
resume UI of the shape F6's ``on_labeled_recall="block"`` already uses.

*Approvals are serialised, not batched.* Tool calls run through
``asyncio.gather``, so without a lock two handlers fire at once -- two prompts
racing for one terminal. Serialising costs wall-clock when several calls in one
turn need approval, and it means a reviewer sees one call at a time rather than
"this turn wants to do these three things". Batching is the better shape and a
breaking change to the handler signature, so it is named here rather than
retrofitted later.

*Approvals are not remembered.* A retried run asks again. Idempotency sounds
obviously right and is a loaded gun: a cached "approve" authorises a second
execution the reviewer never saw, and whether a retry is the same action or a
new one depends on the tool -- transferring money and sending a reminder want
opposite answers. ``run_id`` and ``arguments`` are on the request so an app can
build the policy it needs; the SDK does not pick one, because half-built
idempotency looks like protection while quietly authorising repeats.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolApprovalRequest:
    """What a reviewer is shown before a call goes ahead.

    Frozen because it is handed to application code that may hold on to it: a
    handler must not be able to edit the arguments the gate is about to run.
    """

    tool_name: str
    arguments: dict[str, Any]
    agent_name: str
    run_id: str | None = None
    # The run's taint. "This run has read the public web" is exactly what a
    # reviewer wants to know before approving an outbound action, and it is not
    # visible from the tool name or the arguments.
    data_labels: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolApprovalDecision:
    """approved, refused, or deferred -- three outcomes, not two.

    ``deferred`` marks a call that was neither allowed nor refused: it was
    handed to someone who will answer later, the turn ends now, and the user is
    expected to ask again once it is answered. That is the refuse-and-resume
    pattern, and it needs to be distinguishable from a refusal or the model
    tells the user their request was DENIED when it is merely waiting -- and a
    user told they were refused does not go looking for an approver.

    Only a handler may defer. A timeout or a crash is a refusal: nobody answered
    and nothing is queued, so promising a resumption would be promising
    something nothing is going to deliver.
    """

    approved: bool
    reviewer: str | None = None
    reason: str | None = None
    deferred: bool = False

    def __post_init__(self) -> None:
        if self.approved and self.deferred:
            raise ValueError(
                "a decision cannot be both approved and deferred: either the call "
                "proceeds now or it does not, and allowing both would leave the "
                "executor guessing which was meant"
            )


ToolApprovalHandler = Callable[[ToolApprovalRequest], Awaitable[ToolApprovalDecision]]


@dataclass(frozen=True)
class ToolApprovalSettings:
    """What the tool executor needs to run the gate for one agent.

    Threaded into ``execute_tool_call`` as a parameter, the way ``policy_store``
    and ``data_labels`` already are, rather than stored on the executor: an
    executor is shared and these belong to the agent whose turn is running.
    """

    tools: frozenset[str] = frozenset()
    handler: ToolApprovalHandler | None = None
    timeout: float = 30.0
    agent_name: str = ""

    def covers(self, tool_name: str) -> bool:
        return needs_approval(tool_name, set(self.tools))


# One reviewer, one question at a time. Global rather than per-executor: the
# constraint is the human, and two agents sharing a terminal or a Slack channel
# collide exactly as two tool calls in one turn would.
#
# Keyed by event loop rather than created once at import. An asyncio.Lock binds
# to the loop that first awaits it and raises "bound to a different event loop"
# everywhere else -- so a module-level lock works in a server with one loop and
# breaks in anything that runs several (a test suite, a script calling
# asyncio.run twice, a worker that restarts its loop). The lock is per loop
# because that is the only scope in which it can function at all.
_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)

_WARNED_AGENTS: set[str] = set()


def _ask_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[loop] = lock
    return lock


def build_approval_settings(agent: Any) -> ToolApprovalSettings | None:
    """Read an agent's approval configuration into what the executor needs.

    Returns ``None`` when the agent declares no tools, so an app that never
    enabled this pays nothing per call.

    Deliberately returns settings when tools ARE declared but no handler is
    wired. That configuration must reach the gate precisely because it is
    broken: the gate refuses and says so. Returning ``None`` here would turn
    "declared but unwired" into "silently ungated", which is the failure this
    whole finding is about.
    """
    config = getattr(agent, "config", None)
    if config is None:
        return None
    declared = getattr(config, "tool_approval", None) or set()
    if not declared:
        return None

    handler = getattr(config, "approval_handler", None)
    agent_name = getattr(agent, "name", "")
    warn_if_approval_unwired(agent_name, set(declared), handler)
    return ToolApprovalSettings(
        tools=frozenset(declared),
        handler=handler,
        timeout=float(getattr(config, "approval_timeout", 30.0)),
        agent_name=agent_name,
    )


def needs_approval(tool_name: str, declared: set[str]) -> bool:
    """Is ``tool_name`` one the operator asked to be approved?

    fnmatch, matching how policy ``resources`` are written, so an operator
    learns one pattern syntax rather than two.
    """
    return any(fnmatch(tool_name, pattern) for pattern in declared)


async def request_approval(
    request: ToolApprovalRequest,
    handler: ToolApprovalHandler,
    timeout: float,
) -> ToolApprovalDecision:
    """Ask, and fail closed on anything that is not an explicit approval.

    Every failure path denies: a handler that raises, one that never answers,
    and one that returns something other than a decision. A gate that cannot
    answer has not approved, and the moment it stops working is precisely when
    proceeding would be worst.
    """
    async with _ask_lock():
        try:
            decision = await asyncio.wait_for(handler(request), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Tool approval timed out after %ss for '%s' — denying",
                timeout,
                request.tool_name,
            )
            return ToolApprovalDecision(
                approved=False,
                reason=f"No reviewer responded within {timeout}s (timed out).",
            )
        except Exception as e:
            logger.error(
                "Tool approval handler raised (%s: %s) for '%s' — denying",
                type(e).__name__,
                e,
                request.tool_name,
            )
            return ToolApprovalDecision(approved=False, reason=f"{type(e).__name__}: {e}")

    # Shape check outside the lock: app code returning a bare True, or None from
    # a function that forgot to return, must not read as approval.
    if not isinstance(decision, ToolApprovalDecision):
        logger.error(
            "Tool approval handler returned %s, not a ToolApprovalDecision, for '%s' — denying",
            type(decision).__name__,
            request.tool_name,
        )
        return ToolApprovalDecision(
            approved=False,
            reason="The approval handler did not return a ToolApprovalDecision.",
        )
    return decision


def warn_if_approval_unwired(
    agent_name: str,
    declared: set[str],
    handler: ToolApprovalHandler | None,
) -> None:
    """Say once that tools are declared for approval with nobody to ask.

    Declaring without wiring fails by doing nothing, which is the failure mode
    nobody notices: the operator believes every payment is reviewed while no
    prompt is ever raised. Reported the way an undeclared memory provenance is
    -- name the fields, say what is lost, then stay quiet.
    """
    if not declared or handler is not None:
        return
    if agent_name in _WARNED_AGENTS:
        return
    _WARNED_AGENTS.add(agent_name)
    logger.warning(
        "Agent '%s' declares tool_approval=%s but no approval_handler, so those tools "
        "run WITHOUT being approved. Set AgentConfig(approval_handler=...) or clear "
        "tool_approval — a gate wired to nobody is worse than none, because it reads "
        "as protection.",
        agent_name,
        sorted(declared),
    )
