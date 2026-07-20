#!/usr/bin/env python3
"""End-to-end test for Continuum's Temporal durable execution.

Covers, against a live Temporal server (localhost:7233):
  1. Durable agent workflow:  agent -> wait -> agent  (proves jobs + retries)
  2. Human-in-the-loop:        agent -> approval -> agent  (query pending,
     signal submit_approval, await completion)

Run:  python playground/smoke/temporal_e2e.py   (Temporal infra must be up)
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root = nearest ancestor with pyproject.toml, so this works at any depth.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
load_dotenv(_ROOT / ".env", override=True)
# Force-enable Temporal regardless of .env so the client connects.
os.environ.setdefault("TEMPORAL_ENABLED", "true")

sys.path.insert(0, str(_ROOT / "src"))

from continuum.agent import BaseAgent  # noqa: E402
from continuum.agent.config import AgentMemoryConfig  # noqa: E402
from continuum.temporal import (  # noqa: E402
    ApprovalDecision,
    WorkflowInput,
    get_agent_registry,
    get_temporal_client,
    get_worker_manager,
)


def make_summarizer() -> BaseAgent:
    # Memory disabled to keep the durable-execution test isolated from mem0.
    return BaseAgent(
        name="summarizer",
        instructions="Summarize the input in one short sentence. Be terse.",
        model="gpt-4o-mini",
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


async def main() -> int:
    registry = get_agent_registry()
    registry.register(make_summarizer())

    client = get_temporal_client()
    await client.connect()
    worker = get_worker_manager(client, registry)
    await worker.start()  # spawns polling task, returns immediately
    print(f"[setup] worker running={worker.is_running} | connected={client.is_connected}")

    ok = True

    # ---- Test 1: durable sequential workflow (agent -> wait -> agent) ----
    print("\n=== TEST 1: durable workflow (agent -> wait 2s -> agent) ===")
    wid1 = f"e2e-seq-{os.getpid()}"
    handle1 = await client.run_agent_workflow(
        WorkflowInput(
            steps=[
                {"type": "agent", "agent_name": "summarizer"},
                {"type": "wait", "duration_seconds": 2},
                {"type": "agent", "agent_name": "summarizer"},
            ],
            initial_input="Temporal gives Continuum durable, crash-safe agent execution.",
        ),
        id=wid1,
    )
    res1 = await handle1.result()
    print(f"  status={res1.status}")
    print(f"  content={res1.content!r}")
    t1_ok = res1.status == "completed" and bool(res1.content)
    print(f"  [{'PASS' if t1_ok else 'FAIL'}] durable workflow completed")
    ok = ok and t1_ok

    # ---- Test 2: human-in-the-loop approval gate ----
    print("\n=== TEST 2: human-in-the-loop (agent -> approval -> agent) ===")
    wid2 = f"e2e-approval-{os.getpid()}"
    handle2 = await client.run_agent_workflow(
        WorkflowInput(
            steps=[
                {"type": "agent", "agent_name": "summarizer"},
                {"type": "approval", "description": "Review the draft", "approvers": ["alice"]},
                {"type": "agent", "agent_name": "summarizer"},
            ],
            initial_input="Approval gates pause a workflow until a human decides.",
        ),
        id=wid2,
    )

    # Poll until the workflow parks on a pending approval.
    request_id = None
    for _ in range(30):
        await asyncio.sleep(1)
        pending = await client.query_workflow(wid2, "get_pending_approvals")
        if pending:
            request_id = pending[0]["request_id"]
            break
    print(f"  pending approval request_id={request_id}")
    t2_pending_ok = request_id is not None
    print(f"  [{'PASS' if t2_pending_ok else 'FAIL'}] workflow parked awaiting approval")

    if request_id:
        await client.signal_workflow(
            wid2,
            "submit_approval",
            ApprovalDecision(request_id=request_id, decision="approved", decided_by="alice"),
        )
        res2 = await handle2.result()
        print(f"  status after approval={res2.status}")
        t2_done_ok = res2.status == "completed"
        print(f"  [{'PASS' if t2_done_ok else 'FAIL'}] workflow resumed & completed after approval")
        ok = ok and t2_pending_ok and t2_done_ok
    else:
        ok = False
        await client.cancel_workflow(wid2)

    await worker.stop()
    print("\n" + ("✅ TEMPORAL E2E PASSED" if ok else "❌ TEMPORAL E2E FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
