---
name: continuum-temporal
description: Build durable agent workflows with Temporal — sequential/parallel/loop/conditional steps, human-in-the-loop approval gates, custom workflows and activities. Invoke when the user asks "long-running workflow", "approval gate", "human in the loop", "retry on failure", "workflow survives restart", or anything Temporal-related.
---

# Continuum Temporal Skill

Authoritative sources: [`docs/temporal/getting-started.md`](../../../docs/temporal/getting-started.md),
[`workflow-patterns.md`](../../../docs/temporal/workflow-patterns.md),
[`custom-agents.md`](../../../docs/temporal/custom-agents.md),
[`custom-workflows.md`](../../../docs/temporal/custom-workflows.md),
[`human-in-loop.md`](../../../docs/temporal/human-in-loop.md),
[`docker.md`](../../../docs/temporal/docker.md).

---

## Setup

```bash
pip install "shyftlabs-continuum[temporal]" --find-links wheels/
docker compose --profile temporal up -d
```

```env
TEMPORAL_ENABLED=true
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
TEMPORAL_ENABLE_HUMAN_IN_LOOP=true
```

---

## Three actors

1. **Worker** — long-running process; polls task queue, executes activities.
2. **Workflow** — declarative durable plan; lives inside Temporal.
3. **Caller** — your app; submits workflows, signals/queries them.

---

## Imports

```python
from orchestrator.temporal import (
    get_temporal_client, get_worker_manager, get_agent_registry,
    WorkflowInput, WorkflowResult,
    AgentStep, ApprovalStep, ParallelStep, ConditionalStep, WaitStep,
    AgentWorkflow, SequentialAgentWorkflow,
    ParallelAgentWorkflow, LoopAgentWorkflow,
    ApprovalDecision, ApprovalRequest,
    HumanInLoopManager, ApprovalNotificationConfig, NotificationParams,
)
```

---

## Hello-Temporal

```python
async def main():
    registry = get_agent_registry()
    registry.register(BaseAgent(name="summarizer", instructions="...", model="gpt-4o-mini"))

    client = get_temporal_client()
    await client.connect()
    await get_worker_manager(client, registry).start()         # blocks

    handle = await client.run_agent_workflow(
        WorkflowInput(
            steps=[
                {"type": "agent",    "agent_name": "summarizer"},
                {"type": "approval", "description": "Review", "approvers": ["alice"]},
                {"type": "agent",    "agent_name": "summarizer"},
            ],
            initial_input="...",
        ),
        id="hello-001",
    )
    result = await handle.result()
    print(result.status, result.content)
```

---

## Step types

| `type` | Schema |
|---|---|
| `"agent"` | `{agent_name, input?, timeout=300, retries=3, metadata}` |
| `"approval"` | `{description, approvers=[], timeout=86400}` |
| `"parallel"` | `{agents: [AgentStep…], merge_strategy: "concatenate"\|"first_success"\|"structured"}` |
| `"conditional"` | `{condition_agent, if_true: [...], if_false: [...]}` |
| `"wait"` | `{duration_seconds: 1..604800}` |

`parse_step(dict) -> WorkflowStep` validates a step dict; raises
`ValueError` on unknown `type`.

---

## Convenience workflows

```python
# Sequential
input_data = SequentialWorkflowInput(
    agent_names=["a", "b", "c"], initial_input="...",
    approval_between_steps=False,
)
await client.start_workflow(SequentialAgentWorkflow.run, input_data)

# Parallel
input_data = ParallelWorkflowInput(
    agent_names=["a", "b", "c"], initial_input="...",
    merge_strategy="structured",
)

# Loop
input_data = LoopWorkflowInput(
    agent_name="reviser", initial_input="...",
    max_iterations=10, termination_phrase="COMPLETE",
)
```

---

## Signals & queries on `AgentWorkflow`

| Signal | |
|---|---|
| `submit_approval(decision: ApprovalDecision)` | Approve / reject / escalate |
| `cancel_workflow()` | Cancel |
| `inject_input(data: dict)` | Inject data mid-flow |

| Query | Returns |
|---|---|
| `get_status()` | `{status, current_step_index, total_steps, completed_steps, cancelled}` |
| `get_pending_approvals()` | `list[dict]` |

---

## Human-in-the-loop

```python
async def notify(params: NotificationParams) -> None:
    if params.type == "approval_required":
        await send_email(...)

get_agent_registry().set_notification_handler(notify)

hlm = HumanInLoopManager(
    client,
    ApprovalNotificationConfig(
        handler=notify,
        timeout_seconds=86400,
        escalation_timeout=7200,
        escalation_handler=on_escalate,
    ),
)

await hlm.approve("workflow-id", request_id="...", decided_by="alice")
await hlm.reject("workflow-id",  request_id="...", decided_by="alice", reason="bad draft")
await hlm.escalate("workflow-id", request_id="...", escalate_to="bob")

pending = await hlm.get_pending_approvals("workflow-id")
status = await hlm.get_workflow_status("workflow-id")
```

---

## Custom workflow + activity

```python
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

@activity.defn(name="fetch_url_activity")
async def fetch_url_activity(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as c:
        return (await c.get(url)).text

@workflow.defn(sandboxed=False)
class FetchAndSummarizeWorkflow:
    @workflow.run
    async def run(self, url: str) -> WorkflowResult:
        text = await workflow.execute_activity(
            fetch_url_activity, url,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        result = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(agent_name="summarizer", input=text),
            start_to_close_timeout=timedelta(seconds=120),
        )
        return WorkflowResult(status="completed", content=result.content)

worker = get_worker_manager(client, registry)
worker.register_workflow(FetchAndSummarizeWorkflow)
worker.register_activity(fetch_url_activity)
await worker.start()
```

---

## Determinism rules (critical!)

In a `@workflow.defn` body:

- ❌ `random.random()`, `uuid.uuid4()`, `datetime.now()`, `time.time()`
- ❌ `requests.get()`, file/DB I/O, third-party SDKs
- ✅ `workflow.uuid4()`, `workflow.random()`, `workflow.now()`
- ✅ `workflow.execute_activity(...)` for I/O
- ✅ `workflow.sleep(...)`, `workflow.wait_condition(...)`
- ✅ `with workflow.unsafe.imports_passed_through():` for type-only imports

---

## Don't

- Don't run I/O in workflow code — push it into an activity.
- Don't reuse a workflow `id` — `WorkflowAlreadyStartedError`.
- Don't forget to register agents with the registry before submitting a
  workflow that references them — `AgentNotRegisteredError`.
- Don't forget `await client.connect()` before `start_workflow(...)`.
- Don't run the worker as a fire-and-forget — `await worker.start()`
  blocks; in production it's its own process.
