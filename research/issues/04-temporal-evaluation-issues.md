# Temporal Workflows & Evaluation Issues

---

## CRITICAL

### 1. Race Condition in Approval Signal Handling — No request_id Validation
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 221-246)

When an approval decision is received, there's no validation that `decision.request_id` matches the current pending approval's `request_id`. An attacker or misconfigured caller can submit an approval for the wrong step, and the workflow accepts it.

```python
# Decision received could be for a DIFFERENT approval request
decision = self._pending_decision
self._pending_decision = None  # Cleared without validation
```

**Impact:** Workflow integrity compromised. Wrong approvals applied to wrong steps. Critical for financial or compliance workflows.

---

### 2. Non-Deterministic Conditional Workflow Logic
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 315-324)

The conditional step relies on LLM string output matched against a list: `"true", "yes", "1", "approved", "continue"`. This is NOT deterministic — if the LLM produces different output on replay (which it will), Temporal's event history becomes invalid and the workflow corrupts.

Additionally, only `AgentStep` is handled in branch execution. Other step types (ApprovalStep, ParallelStep, WaitStep, ConditionalStep) are silently dropped.

**Impact:** Workflow replay corruption. Silent data loss in conditional branches.

---

### 3. Unbounded Retry Policy with No Backoff Ceiling
**Files:** All workflow files (agent_workflow.py, sequential_workflow.py, parallel_workflow.py, loop_workflow.py)

All workflows use `RetryPolicy(maximum_attempts=step.retries)` with NO `initial_interval`, `backoff_coefficient`, or `max_interval`. Temporal defaults apply but are implicit and may not be appropriate.

```python
# DANGEROUS — no backoff ceiling
retry_policy=RetryPolicy(maximum_attempts=step.retries)

# CORRECT
retry_policy=RetryPolicy(
    maximum_attempts=step.retries,
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    max_interval=timedelta(seconds=60),
)
```

**Impact:** Failed activities hammer the system with unbounded exponential backoff.

---

### 4. Silent Data Loss in ParallelStep Merge — Key Collision
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 280-293)

In "structured" merge mode, if multiple agents share the same name, dictionary keys collide and results are silently overwritten:

```python
parts = {
    r.agents_used[0] if r.agents_used else f"agent-{i}": r.content
    for i, r in enumerate(results)
}
self._last_output = str(parts)  # str() is NOT reversible!
```

**Impact:** Agent results silently lost. `str()` conversion loses structured data permanently.

---

## HIGH

### 5. Approval Timeout Doesn't Update Overall Workflow Status
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 220-230)

When an approval times out, `self._status = "timed_out"` is set, but then execution continues to the next step. The next step's completion overwrites the status. The timeout is lost.

**Impact:** Timed-out approvals don't block workflow progression. Audit trail shows "completed" instead of "timed_out."

---

### 6. No Upfront Schema Validation of Workflow Steps
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 103-114)

Steps are parsed one-by-one at runtime via `parse_step()`. If a step dict has an unknown type, `ValueError` is raised mid-workflow (not at submission time). The workflow has already executed previous steps before discovering the error.

**Impact:** Partially-executed workflows that can't be recovered. Bad step definitions discovered too late.

---

### 7. Hardcoded Timeout/Retries for Conditional Agent Execution
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 299-310)

Conditional step always uses `start_to_close_timeout=300s`, `retry_policy=max_attempts=3`, `heartbeat_timeout=60s`. These are hardcoded. The `ConditionalStep` type has no timeout/retry fields, so it can't be customized.

---

### 8. JSON Parsing Failure in EvaluatorAgent Silently Defaults to 0
**File:** `src/orchestrator/evaluation/evaluator_agent.py` (~lines 293-315)

If the LLM fails to produce valid JSON, the parser returns `{}`. Then `score_val = float(raw.get("score", 0.0))` defaults to 0.0. A failed evaluation looks identical to a genuine zero score. No error is logged.

**Impact:** Evaluation results are untrustworthy. LLM failures indistinguishable from actual poor performance.

---

### 9. Hardcoded 0.7 Pass Threshold in RAGAS Evaluator
**File:** `src/orchestrator/evaluation/ragas_eval.py` (~line 230)

All RAGAS metric scores use `passed=raw_score >= 0.7`. No configurable threshold. Different metrics need different thresholds (faithfulness critical at 0.8+, context_precision acceptable at 0.6+).

---

### 10. Unbounded Recursion in Secret Sanitization
**File:** `src/orchestrator/utils/secrets.py` (~lines 33-56)

`redact_dict()` has `max_depth` parameter, but when depth is exceeded, it returns the ORIGINAL unredacted data (not a partially-redacted result). Circular references cause infinite recursion since Python dicts can self-reference.

**Impact:** Sensitive data leaks at deep nesting levels. Stack overflow on circular structures.

---

### 11. Agent Registry Not Thread-Safe for Read Operations
**File:** `src/orchestrator/temporal/registry.py` (~lines 46-55)

`get()` reads from `_agents` dict without holding the lock. If `clear()` is called from another thread between `.get(name)` and `list(self._agents.keys())`, the available_agents list may be stale.

---

## MEDIUM

### 12. Activity Heartbeat Timing Mismatch
**Files:** All workflow files

Heartbeat timeout is 60 seconds, activity timeout is 300 seconds. If an activity hangs, there's a 60-second dead period before Temporal detects it. Should be configurable and typically shorter (30 seconds).

---

### 13. DeepEval Metrics Extracted by Positional Index
**File:** `src/orchestrator/evaluation/deepeval_eval.py` (~lines 138-161)

Results mapped by index position, not metric name. If DeepEval reorders metrics_data or one metric fails to produce output, scores are attributed to wrong metrics.

---

### 14. No Type Validation of EvalCase.context
**File:** `src/orchestrator/evaluation/types.py` (~lines 39-42)

`EvalCase(context="single string")` is accepted (passing string instead of list). Later code that assumes list type will fail.

---

### 15. RAGAS Hardcodes OpenAI — Not Configurable
**File:** `src/orchestrator/evaluation/ragas_eval.py` (~lines 87-88, 169-177)

RAGAS evaluation forces OpenAI LLM and embeddings. No option to use Anthropic, local LLMs, etc. API key check happens at eval time, not initialization — late failure.

---

### 16. Notification Activity Failure Silently Swallowed
**Files:** agent_workflow.py, sequential_workflow.py, loop_workflow.py

All approval notification failures are caught with `except Exception: pass`. No logging or observability. If all notifications fail, no one notices.

---

### 17. Missing Null Check in AgentActivityResult.from_agent_response
**File:** `src/orchestrator/temporal/types.py` (~lines 107-127)

`resp.usage` null check is in a ternary that evaluates the dict comprehension first. Safer to check None upfront.

---

### 18. No Validation of WaitStep Duration
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 132-133)

`duration_seconds` not validated. Negative, zero, or extremely large values (10^9 = 31 years) cause unexpected behavior.

---

### 19. WorkflowStep Union Type Not Extensible
**File:** `src/orchestrator/temporal/types.py` (~lines 65, 68-82)

Adding a new step type requires modifying both the union type and the parse function. Users cannot register custom step types. Should use a registry pattern.

---

## LOW

### 20. Silent Failure in HumanInLoopManager.escalate
**File:** `src/orchestrator/temporal/human_in_loop.py` (~lines 100-123)

If request_id not found in pending approvals, escalation is silently skipped. No warning logged.

---

### 21. Langfuse Client Access Via Private Attribute
**File:** `src/orchestrator/evaluation/langfuse_datasets.py` (~line 208)

`getattr(client, "_client", None)` accesses internal implementation detail. Will break if wrapper internals change.

---

### 22. DeepEval Display Config Hardcoded
**File:** `src/orchestrator/evaluation/deepeval_eval.py` (~lines 112-120)

`DisplayConfig(print_results=False, show_indicator=False)` is hardcoded. No option to enable for debugging.

---

### 23. Conditional Branching Violates Temporal Determinism
**File:** `src/orchestrator/temporal/workflows/agent_workflow.py` (~lines 295-324)

Conditional branching appends results to `_step_results` even if branch condition produces different output on replay. Step result counts become inconsistent across replays.

---

### 24. No Maximum Payload Size Validation
**File:** `src/orchestrator/temporal/types.py` (~lines 164-169)

`WorkflowInput` has no size limits. 100MB+ inputs will fail Temporal's gRPC serialization with cryptic errors.
