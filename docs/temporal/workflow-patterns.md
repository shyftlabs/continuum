# Temporal — Workflow Patterns

The Temporal integration ships **one general workflow** (`AgentWorkflow`)
that interprets a declarative step list, plus three convenience
workflows for common shapes.

`from orchestrator.temporal import (
    AgentWorkflow, SequentialAgentWorkflow,
    ParallelAgentWorkflow, LoopAgentWorkflow,
    AgentStep, ApprovalStep, ParallelStep, ConditionalStep, WaitStep,
    WorkflowInput, WorkflowResult, parse_step,
    AgentActivityParams, AgentActivityResult,
    NotificationParams,
    ApprovalRequest, ApprovalDecision,
    StepType,
)`

---

## 1 · `AgentWorkflow` (general purpose)

Decorated with `@workflow.defn(sandboxed=False)`. Takes a `WorkflowInput`
and runs each step in order.

### Step types

| `type` | Class | Purpose |
|---|---|---|
| `"agent"` | `AgentStep` | Run one registered agent |
| `"approval"` | `ApprovalStep` | Pause for human approval |
| `"parallel"` | `ParallelStep` | Run multiple agents concurrently, merge results |
| `"conditional"` | `ConditionalStep` | Branch on the output of a condition agent |
| `"wait"` | `WaitStep` | Sleep for a duration (1 s – 7 days) |

### Step schemas

```python
AgentStep(
    type="agent",
    agent_name: str,                       # required
    input: str | None = None,              # default: pass the previous step's output
    timeout: int = 300,
    retries: int = 3,
    metadata: dict[str, Any] = {},
)

ApprovalStep(
    type="approval",
    description: str,                      # required
    approvers: list[str] = [],
    timeout: int = 86400,                  # 24h default
    auto_approve_if: str | None = None,    # reserved
)

ParallelStep(
    type="parallel",
    agents: list[AgentStep],               # one AgentStep per branch
    merge_strategy: str = "concatenate",   # "concatenate" | "first_success" | "structured"
)

ConditionalStep(
    type="conditional",
    condition_agent: str,                  # agent that returns true/false
    condition_input: str | None = None,    # explicit input for the condition agent; default: last step's output
    if_true: list[dict[str, Any]] = [],    # nested steps
    if_false: list[dict[str, Any]] = [],
    timeout: int = 300,
    retries: int = 3,
    metadata: dict[str, Any] = {},
)

WaitStep(
    type="wait",
    duration_seconds: int,                 # 1 ≤ x ≤ 604800 (7 days)
)
```

`parse_step(data)` converts a `dict` to the appropriate model; it
raises `ValueError` for unknown `type`. The workflow validates *all*
steps upfront before running anything.

### Signals

| Signal | Purpose |
|---|---|
| `submit_approval(decision: ApprovalDecision)` | Human submits approval/rejection/escalation |
| `cancel_workflow()` | Cancel the run |
| `inject_input(data: dict)` | Inject data mid-workflow |

### Queries

| Query | Returns |
|---|---|
| `get_status()` | `{"status", "current_step_index", "total_steps", "completed_steps", "cancelled"}` |
| `get_pending_approvals()` | `list[dict]` — each with `request_id`, `description`, `context` |

### `WorkflowResult`

```python
WorkflowResult(
    status: str,                           # "completed" | "rejected" | "timed_out" | "failed" | "cancelled"
    content: str | None = None,            # final output
    step_results: list[AgentActivityResult] = [],
    approval_decisions: list[ApprovalDecision] = [],
    error: str | None = None,
)
```

### Example

```python
input_data = WorkflowInput(
    steps=[
        {"type": "agent", "agent_name": "drafter"},
        {"type": "approval", "description": "Review draft before publishing", "approvers": ["alice"]},
        {"type": "parallel",
         "agents": [{"type":"agent","agent_name":"seo"}, {"type":"agent","agent_name":"legal"}],
         "merge_strategy": "structured"},
        {"type": "agent", "agent_name": "publisher"},
    ],
    initial_input="Write a blog post about durable execution.",
)
handle = await client.run_agent_workflow(input_data, id="post-001")
result = await handle.result()
```

### Retry policy (per agent activity)

```python
RetryPolicy(
    maximum_attempts=step.retries,         # default 3
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
)
```

Activities also use `heartbeat_timeout=60s`; `run_agent_activity` calls
`activity.heartbeat()` so long-running agents don't get evicted.

### Parallel merge strategies

| Strategy | Behaviour |
|---|---|
| `"concatenate"` *(default)* | `"\n\n".join(results)` |
| `"first_success"` | Return the first non-error result |
| `"structured"` | JSON dict keyed by agent name |

### Conditional truthiness

`condition_agent`'s output is matched case-insensitively against:

```
true / yes / 1 / approved / continue
```

…to take the `if_true` branch. Anything else takes `if_false`.

---

## 2 · `SequentialAgentWorkflow`

Convenience workflow: run a list of agents in sequence, optionally with
approval gates between steps.

```python
from orchestrator.temporal import SequentialAgentWorkflow
from orchestrator.temporal.workflows.sequential_workflow import SequentialWorkflowInput

input_data = SequentialWorkflowInput(
    agent_names=["researcher", "writer", "editor"],
    initial_input="Topic: durable execution",
    session_id=None,
    user_id=None,
    approval_between_steps=False,          # if True, an ApprovalStep is inserted after each step
    approval_timeout=86400,
)
handle = await client.start_workflow(SequentialAgentWorkflow.run, input_data, id="seq-001")
```

Signals: `submit_approval`, `cancel_workflow`. Queries: `get_status`,
`get_pending_approvals`.

---

## 3 · `ParallelAgentWorkflow`

Run a list of agents concurrently against the same input.

```python
from orchestrator.temporal import ParallelAgentWorkflow
from orchestrator.temporal.workflows.parallel_workflow import ParallelWorkflowInput

input_data = ParallelWorkflowInput(
    agent_names=["seo_critic", "legal_critic", "tone_critic"],
    initial_input="<draft>",
    merge_strategy="structured",           # "concatenate" | "first_success" | "structured"
    timeout_per_agent=300,
)
handle = await client.start_workflow(ParallelAgentWorkflow.run, input_data)
```

Signal: `cancel_workflow`. Query: `get_status`.

---

## 4 · `LoopAgentWorkflow`

Run one agent in a loop until it emits a termination phrase.

```python
from orchestrator.temporal import LoopAgentWorkflow
from orchestrator.temporal.workflows.loop_workflow import LoopWorkflowInput

input_data = LoopWorkflowInput(
    agent_name="reviser",
    initial_input="<draft>",
    max_iterations=10,
    termination_phrase="COMPLETE",          # case-insensitive substring match
    approval_per_iteration=False,
    approval_timeout=86400,
)
handle = await client.start_workflow(LoopAgentWorkflow.run, input_data)
```

Signals: `submit_approval`, `cancel_workflow`. Query: `get_status` (with
iteration count), `get_pending_approvals`.

---

## 5 · `AgentActivityParams` / `AgentActivityResult`

Each agent step runs as the `run_agent_activity` activity:

```python
AgentActivityParams(
    agent_name: str,
    input: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] = {},
    tags: list[str] = [],
)

AgentActivityResult(
    content: str,
    status: str,                            # "completed" | "error"
    structured_output: dict[str, Any] | None = None,
    usage: dict[str, int] = {},             # prompt/completion/total tokens
    agents_used: list[str] = [],
    error: str | None = None,
)
```

`AgentActivityResult.from_agent_response(resp)` converts an
`AgentResponse` into the activity-shaped result for serialization.

---

## 6 · Notifications

`send_notification_activity` calls a registered handler:

```python
NotificationParams(
    type: str,                              # e.g. "approval_required"
    payload: dict[str, Any] = {},
)
```

Register the handler on the registry:

```python
registry.set_notification_handler(my_async_handler)
```

If no handler is registered, the activity logs a warning and returns
cleanly — the workflow does not fail.

---

## 7 · Tips

- **Validate steps locally** before submitting: `parse_step(my_dict)`
  raises `ValueError` on bad input.
- **Use `AgentWorkflow` for anything non-trivial.** The convenience
  workflows are easier to reason about for fixed shapes; the general
  workflow handles approval gates, branches, and waits in one place.
- **Sequential ↔ Parallel ↔ Loop** can be **nested** by using
  `AgentWorkflow` and putting a `ParallelStep` inside a step list, etc.
- **Token usage** is rolled up per step into `step_results[i].usage` —
  add them yourself for a workflow-level total.
