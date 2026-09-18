"""AP6 — the durable approval route, end to end (security finding F7).

`CLINIC_APPROVAL=ask` holds an HTTP request open, so the reviewer has to be
watching. `queue` lets them answer whenever, but the turn ENDS and the user has
to ask again, redoing whatever happened before the gate. This is the third
shape: the turn **blocks in place** inside a Temporal activity and resumes when
answered, so work before the gate is not repeated and nobody asks twice.

It needs a workflow, which `web.py` does not run -- hence a separate driver
rather than another web mode. Everything else is the clinic's own agent, MCP
servers and policy store, so what you see gated here is the same call AP1-AP5
gate.

    # one terminal
    docker compose up -d temporal postgres-temporal temporal-ui
    python server.py
    python pharmacy_server.py

    # another
    python approval_temporal.py --auto approve     # scripted, both paths
    python approval_temporal.py --auto deny
    python approval_temporal.py                    # answer it yourself

Unanswered, it waits. Answer from the Temporal UI at localhost:8233 by sending
a `submit_approval` signal, or from another shell -- the request id is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

# The mode has to be set BEFORE config is imported: build_approval_handler and
# approval_timeout both read the environment at import time, so setting it after
# would leave the agent with no handler and the gate would refuse every call.
os.environ.setdefault("CLINIC_APPROVAL", "temporal")

from agent import ClinicAgent  # noqa: E402
from config import APPROVAL_TOOL, ClinicConfig  # noqa: E402

from continuum.temporal import (  # noqa: E402
    AgentWorkflow,
    WorkflowInput,
    get_agent_registry,
    get_temporal_client,
    get_worker_manager,
)
from continuum.temporal.types import ApprovalDecision  # noqa: E402

TASK_QUEUE = "clinic-f7-approval"
QUESTION = "Check for interactions between metformin and lisinopril."

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ap6")


async def _await_prompt(handle, timeout: float = 180.0) -> dict:
    """Poll the workflow until the tool approval shows up, or give up."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        pending = await handle.query("get_pending_approvals")
        if pending:
            return pending[0]
        await asyncio.sleep(1.0)
    raise TimeoutError(
        "no approval request appeared. The model may not have called the gated "
        f"tool ({APPROVAL_TOOL}) — check that both MCP servers are running."
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--auto",
        choices=("approve", "deny"),
        help="answer the prompt from here instead of waiting for a person",
    )
    ap.add_argument("--reviewer", default="tom", help="who the decision is recorded as")
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()

    # Line-buffer stdout. Without this, piping to a file or a pager holds the
    # prompt in a buffer while the turn blocks on the gate -- so the one thing
    # you are waiting to see is the one thing you cannot, and a correctly
    # blocked run is indistinguishable from a hung one.
    sys.stdout.reconfigure(line_buffering=True)

    host = os.environ.get("TEMPORAL_HOST", "localhost:7233")

    # THE LINE THAT COSTS FOUR ATTEMPTS IF MISSED. temporal_tool_approval()
    # resolves its handle through the GLOBAL client; a worker connects its OWN,
    # and they are different objects. Without this the worker runs activities
    # happily and every approval defers with "Not connected to Temporal server",
    # which reads as a network fault rather than a missing call.
    client = get_temporal_client()
    await client.connect(host)
    print(f"connected to temporal at {host}")

    clinic = ClinicAgent(ClinicConfig())
    await clinic.initialize()
    if APPROVAL_TOOL not in clinic._agent.config.tool_approval:
        print(f"ERROR: {APPROVAL_TOOL} is not declared for approval", file=sys.stderr)
        return 1
    print(f"gated tool: {APPROVAL_TOOL}")

    registry = get_agent_registry()
    registry.register(clinic._agent)
    registry.set_runner_factory(lambda: clinic._runner)

    # No register_workflow/register_activity here: WorkerManager.start already
    # registers AgentWorkflow and both built-in activities, and registering them
    # again is a hard failure -- "More than one activity named
    # run_agent_activity" -- not a duplicate that gets ignored.
    worker = get_worker_manager(client=client, registry=registry)
    await worker.start(task_queue=TASK_QUEUE)
    print(f"worker up on {TASK_QUEUE}")

    workflow_id = f"f7-{uuid.uuid4().hex[:8]}"
    try:
        handle = await client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                steps=[{"type": "agent", "agent_name": clinic._agent.name, "timeout": 900}],
                initial_input=args.question,
                user_id="ap6",
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(f"workflow: {workflow_id}\n")

        prompt = await _await_prompt(handle)
        print("PROMPT")
        print(f"  request_id: {prompt['request_id']}")
        print(f"  tool:       {prompt.get('description')}")
        print(f"  arguments:  {prompt.get('context')}")
        print()

        if args.auto:
            decision = "approved" if args.auto == "approve" else "rejected"
            await handle.signal(
                "submit_approval",
                ApprovalDecision(
                    request_id=prompt["request_id"],
                    decision=decision,
                    decided_by=args.reviewer,
                ),
            )
            print(f"{decision.upper()} by {args.reviewer}")
        else:
            print("waiting for a reviewer. Answer it with submit_approval —")
            print("  from the Temporal UI at http://localhost:8233, or:")
            print(
                f'  handle.signal("submit_approval", ApprovalDecision('
                f'request_id="{prompt["request_id"]}", decision="approved", '
                f'decided_by="you"))'
            )
            print()

        result = await handle.result()
        print(f"\nstatus:  {result.status}")
        print(f"answer:  {(result.content or '').strip()[:400]}")

        # The audit trail, and the claim AP6 actually makes: ONE run, one
        # workflow, answered mid-flight. The turn resumed in place -- the work
        # before the gate was not repeated and nobody had to ask again. Compare
        # the answer against AP5, where the turn ends and the user asks twice.
        for d in result.approval_decisions:
            print(f"decision: {d.request_id} {d.decision} by {d.decided_by}")
        return 0
    finally:
        await worker.stop()
        await clinic.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
