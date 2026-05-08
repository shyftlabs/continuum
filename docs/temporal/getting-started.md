# Temporal — Getting Started

[Temporal](https://temporal.io) gives Continuum **durable execution**:
agent workflows survive process crashes, can pause for human approval,
support automatic retries, and run for days or weeks without losing
state.

This guide walks through installing the optional dependency, starting
the Temporal infrastructure, and running your first agent workflow.

---

## 1 · Install the optional extra

```bash
pip install "shyftlabs-continuum[temporal]" --find-links wheels/
```

This adds `temporalio >= 1.23.0`.

---

## 2 · Start the Temporal services

The Temporal server, UI, and Postgres come up via the optional
`temporal` profile in the included `docker-compose.yml`:

```bash
docker compose --profile temporal up -d
```

| Service | Port |
|---|---|
| Temporal server (gRPC) | `localhost:7233` |
| Temporal Web UI | http://localhost:8080 |
| Temporal Postgres | internal |

---

## 3 · Configure

Add to `.env`:

```env
TEMPORAL_ENABLED=true
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
TEMPORAL_ENABLE_HUMAN_IN_LOOP=true
TEMPORAL_APPROVAL_TIMEOUT_SECONDS=86400
TEMPORAL_WORKFLOW_EXECUTION_TIMEOUT=604800
TEMPORAL_ACTIVITY_START_TO_CLOSE_TIMEOUT=300
TEMPORAL_ACTIVITY_RETRY_MAX_ATTEMPTS=3
```

---

## 4 · The 30-second mental model

Three actors, three lifecycles:

1. **Worker process** (long-running): polls the Temporal task queue,
   executes activities (each activity calls `AgentRunner.run()` for one
   registered agent).
2. **Workflow** (the durable plan): a declarative list of `AgentStep`,
   `ApprovalStep`, `ParallelStep`, `ConditionalStep`, `WaitStep`. Lives
   inside Temporal's database; resilient to crashes.
3. **Caller** (your app): submits a workflow with `WorkflowInput`, gets
   a handle, can `result()`, `signal()`, or `query()` it.

```text
caller ──▶ Temporal server ──▶ worker ──▶ AgentRunner ──▶ LLM/MCP
                       ▲                                    │
                       └──────── activity result ◀──────────┘
```

---

## 5 · Hello, durable workflow

```python
import asyncio
from orchestrator.agent import BaseAgent
from orchestrator.temporal import (
    get_temporal_client, get_worker_manager, get_agent_registry,
    WorkflowInput,
)

async def run_caller():
    # 5.1  Register at least one agent
    registry = get_agent_registry()
    registry.register(BaseAgent(
        name="summarizer",
        instructions="Summarize the input in two sentences.",
        model="gpt-4o-mini",
    ))

    # 5.2  Connect & start the worker
    #
    # `worker.start()` spawns the polling task and RETURNS IMMEDIATELY —
    # it does not block. To keep the worker alive in a hello-world
    # script, run the caller in a separate process, or `await
    # worker._worker_task` after submitting workflows.
    client = get_temporal_client()
    await client.connect()
    worker = get_worker_manager(client, registry)
    await worker.start()

    # 5.3  Submit a workflow
    handle = await client.run_agent_workflow(
        WorkflowInput(
            steps=[
                {"type": "agent", "agent_name": "summarizer"},
                {"type": "wait", "duration_seconds": 5},
                {"type": "agent", "agent_name": "summarizer"},
            ],
            initial_input="Temporal is a durable execution platform.",
        ),
        id="hello-temporal-001",
    )

    result = await handle.result()
    print(result.status, "=>", result.content)

asyncio.run(run_caller())
```

Open http://localhost:8080 — your workflow shows up there with full
event history.

---

## 6 · Anatomy of `WorkflowInput`

```python
from orchestrator.temporal import WorkflowInput

WorkflowInput(
    steps=[                                # list[dict] — parsed via parse_step()
        {"type": "agent",       "agent_name": "...", "input": "...", "timeout": 300, "retries": 3},
        {"type": "approval",    "description": "...", "approvers": ["alice"], "timeout": 86400},
        {"type": "parallel",    "agents": [{"type":"agent","agent_name":"a"}, {"type":"agent","agent_name":"b"}], "merge_strategy": "concatenate"},
        {"type": "conditional", "condition_agent": "is_done", "if_true": [...], "if_false": [...]},
        {"type": "wait",        "duration_seconds": 60},
    ],
    initial_input="…",
    session_id=None,
    user_id=None,
    metadata={},
)
```

Step types are validated up-front via `parse_step()`; invalid steps
fail fast before any LLM call.

---

## 7 · Production deployment outline

In production, run the worker as its own process:

```python
# worker_main.py
import asyncio
from orchestrator.temporal import (
    get_temporal_client, get_worker_manager, get_agent_registry,
)
from my_app.agents import all_agents

async def main():
    registry = get_agent_registry()
    registry.register_many(all_agents())
    client = get_temporal_client()
    await client.connect()
    worker = get_worker_manager(client, registry)
    await worker.start()
    # `start()` spawns the worker as an asyncio Task and returns. To
    # keep the process alive in a long-running worker, await the task
    # (or another forever-pending future).
    await worker._worker_task

asyncio.run(main())
```

Your API process is the **caller**:

```python
client = get_temporal_client()
await client.connect()
handle = await client.run_agent_workflow(WorkflowInput(...), id=request_id)
return {"workflow_id": handle.id}
```

---

## 8 · Where to go next

- [`workflow-patterns.md`](workflow-patterns.md) — every step type with
  examples
- [`custom-agents.md`](custom-agents.md) — registering your own agents
- [`custom-workflows.md`](custom-workflows.md) — writing custom
  `@workflow.defn` and `@activity.defn`
- [`human-in-loop.md`](human-in-loop.md) — approval gates,
  notifications, escalation
- [`docker.md`](docker.md) — what each Temporal service does
