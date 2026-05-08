# Temporal — Human-in-the-Loop Approvals

Approval gates pause a workflow until a human submits an
`ApprovalDecision`. The framework provides a high-level
`HumanInLoopManager` so you don't have to wrangle signals manually.

`from orchestrator.temporal import (
    HumanInLoopManager, ApprovalNotificationConfig,
    ApprovalRequest, ApprovalDecision,
)`

---

## 1 · The flow

```
workflow ──▶ ApprovalStep
              │
              ├─▶ send_notification_activity ──▶ your handler (Slack/email/UI/…)
              │
              ▼
       wait_condition (with timeout)
              │
   external caller ──▶ HumanInLoopManager.approve(...)
                          │
                          └─▶ submit_approval signal ──▶ ApprovalDecision validated
                                                          │
                          ┌───────────────────────────────┘
                          ▼
              workflow resumes / rejects / times out
```

---

## 2 · Wire up notifications

The notification handler is responsible for **alerting a human** that
an approval is pending. You set it on the agent registry:

```python
from orchestrator.temporal import get_agent_registry
from orchestrator.temporal.types import NotificationParams

async def notify_slack(params: NotificationParams) -> None:
    if params.type == "approval_required":
        body = params.payload                  # {request_id, workflow_id, description, context, ...}
        await slack.post(channel="#approvals",
                         text=f"Approve workflow {body['workflow_id']}: {body['description']}",
                         attachments=[{"actions": [{"type": "button", "text": "Open",
                                                    "url": f"https://yourapp/approvals/{body['request_id']}"}]}])

get_agent_registry().set_notification_handler(notify_slack)
```

If no handler is registered, the activity logs a warning and the
workflow proceeds to wait — handy for tests, but in production wire one
up.

---

## 3 · Use `HumanInLoopManager`

```python
from orchestrator.temporal import HumanInLoopManager, ApprovalNotificationConfig

config = ApprovalNotificationConfig(
    handler=notify_slack,                        # also wire here for legacy reasons
    webhook_url=None,                            # reserved
    timeout_seconds=86400,                       # 1 day
    escalation_timeout=7200,                     # 2 hours before escalation
    escalation_handler=notify_oncall,            # async handler
    auto_approve_conditions=[],                  # list of (request) -> bool
)
hlm = HumanInLoopManager(client, config)
```

### Caller-side API

| Method | Behaviour |
|---|---|
| `await hlm.approve(workflow_id, request_id, decided_by, reason="")` | Submit `decision="approved"` |
| `await hlm.reject(workflow_id, request_id, decided_by, reason="")` | Submit `decision="rejected"` |
| `await hlm.escalate(workflow_id, request_id, escalate_to)` | Send escalation notification + signal `decision="escalated"` |
| `await hlm.submit_decision(workflow_id, decision)` | Low-level; the three above wrap it |
| `await hlm.get_pending_approvals(workflow_id)` | Query the workflow for pending requests |
| `await hlm.get_workflow_status(workflow_id)` | Status, current step, completed steps |

### `ApprovalRequest`

```python
ApprovalRequest(
    request_id: str,
    workflow_id: str,
    step_index: int,
    description: str,
    context: str | None = None,                  # last step's output
    approvers: list[str] = [],
    timeout: int = 86400,
    created_at: str,                             # ISO datetime
)
```

### `ApprovalDecision`

```python
ApprovalDecision(
    request_id: str,
    decision: str,                               # "approved" | "rejected" | "escalated"
    decided_by: str,
    reason: str | None = None,
    decided_at: str,                             # ISO datetime
)
```

The workflow's `submit_approval(decision)` signal validates that
`decision.request_id` matches a pending request before applying it.

---

## 4 · End-to-end example

### A) The workflow steps

```python
input_data = WorkflowInput(
    steps=[
        {"type": "agent",    "agent_name": "draft"},
        {"type": "approval", "description": "Review draft before publishing.",
                              "approvers": ["alice"], "timeout": 86400},
        {"type": "agent",    "agent_name": "publish"},
    ],
    initial_input="Topic: Q4 results",
)
handle = await client.run_agent_workflow(input_data, id="q4-001")
```

### B) The notification handler

```python
async def handler(params: NotificationParams) -> None:
    body = params.payload
    if params.type == "approval_required":
        await send_email(
            to="alice@example.com",
            subject=f"Approve {body['workflow_id']}",
            body=(f"Description: {body['description']}\n"
                  f"Context:\n{body['context']}\n\n"
                  f"Approve: https://app/api/approve/{body['workflow_id']}/{body['request_id']}\n"
                  f"Reject:  https://app/api/reject/{body['workflow_id']}/{body['request_id']}\n"),
        )

get_agent_registry().set_notification_handler(handler)
```

### C) The approve / reject endpoints

```python
@app.post("/api/approve/{workflow_id}/{request_id}")
async def approve_endpoint(workflow_id: str, request_id: str, decided_by: str):
    await hlm.approve(workflow_id, request_id, decided_by=decided_by)
    return {"ok": True}

@app.post("/api/reject/{workflow_id}/{request_id}")
async def reject_endpoint(workflow_id: str, request_id: str, decided_by: str, reason: str):
    await hlm.reject(workflow_id, request_id, decided_by=decided_by, reason=reason)
    return {"ok": True}
```

### D) Polling pending approvals (e.g. dashboard)

```python
status = await hlm.get_workflow_status("q4-001")
# {"status": "running", "current_step_index": 1, "total_steps": 3, "completed_steps": 1}

pending = await hlm.get_pending_approvals("q4-001")
# [{"request_id": "...", "description": "Review draft before publishing.", ...}]
```

---

## 5 · Behaviour reference

| Scenario | Workflow result |
|---|---|
| Approver submits `approved` before timeout | Continues with the next step |
| Approver submits `rejected` | Workflow returns `WorkflowResult(status="rejected", content=<last output>, step_results=..., approval_decisions=[<the rejection>])`. `error` is `None`; the rejection reason is on `approval_decisions[-1].reason` |
| Approver submits `escalated` | `escalation_handler` invoked; workflow keeps waiting until a non-`escalated` decision arrives or timeout fires |
| No decision before `timeout_seconds` | Workflow returns `WorkflowResult(status="timed_out", content=<last output>, step_results=..., approval_decisions=...)`. `error` is `None` |
| Caller calls `client.cancel_workflow(...)` while waiting | Workflow returns `WorkflowResult(status="cancelled")` (Temporal-level cancel; raises inside the workflow) |

---

## 6 · Auto-approval

`ApprovalNotificationConfig.auto_approve_conditions` accepts a list of
predicates `(ApprovalRequest) -> bool`. If any returns `True`, the
manager auto-approves on the worker side without notifying a human.

```python
config = ApprovalNotificationConfig(
    auto_approve_conditions=[
        lambda req: "<small refund>" in req.description,
    ],
)
```

Use sparingly — auto-approvals defeat the point of an approval gate.

---

## 7 · Exceptions

`from orchestrator.temporal import ApprovalTimeoutError, WorkflowCancelledError`

`ApprovalTimeoutError(message, *, request_id=None, timeout_seconds=None, **kwargs)` carries those fields in its context dict.
`WorkflowCancelledError(message, *, workflow_id=None, **kwargs)` likewise.

---

## 8 · Tips

- **Encode the URL in the notification.** Approvers click a link →
  your endpoint calls `hlm.approve()` → workflow resumes. No polling
  needed.
- **Set a `request_id` based on `workflow_id + step_index`** if you
  need idempotency on the approve/reject endpoint — a duplicate signal
  is a no-op once a decision is recorded.
- **Use queries for dashboards.** `get_workflow_status()` and
  `get_pending_approvals()` are cheap and don't replay the workflow.
- **Reasonable timeouts.** Default is 24h. Longer for human-driven
  flows; shorter for time-sensitive ones.
