# Agent Layer Issues — BaseAgent, AgentRunner, Handoffs, Workflows

---

## CRITICAL

### 1. Race Condition in CircuitBreaker State Transitions
**File:** `src/orchestrator/agent/utils/circuit_breaker.py` (~lines 43-49)

The `state` property checks `self._state` and potentially modifies it without holding the lock for the entire check-and-modify duration. Between reading `self._state == CircuitState.OPEN` and checking the timeout, another thread could call `record_failure()` or `record_success()`, creating a TOCTOU (Time-of-Check-Time-of-Use) inconsistency.

**Impact:** Rate limiting becomes unreliable under concurrent load. Agents can hammer failing services.

---

### 2. Potential Null Pointer in HandoffExecutor
**File:** `src/orchestrator/agent/execution/handoff_executor.py` (~lines 86-93)

If `_handoff_manager` is None, the function returns a `HandoffResult` with an empty handoff_id string (`""`). Downstream code expecting valid UUIDs will misbehave. The initialization failure is not properly surfaced.

**Impact:** Handoff failures silently produce empty IDs, causing confusing errors downstream.

---

## HIGH

### 3. Missing Deepcopy in BaseAgent.clone()
**File:** `src/orchestrator/agent/base.py` (~lines 370-414)

`clone()` uses `list()` and `dict()` for shallow copies of `tools`, `mcp_servers`, `handoffs`, `metadata`, `tags`, `template_vars`, and `examples`. Nested structures are shared between original and clone.

```python
# This modifies BOTH original and clone:
agent_clone.handoffs[0].description = "modified"
```

**Impact:** Silent data corruption in multi-agent systems where agents are cloned and customized.

---

### 4. Exception Swallowing in resolve_system_prompt()
**File:** `src/orchestrator/agent/base.py` (~lines 269-290)

Two `except Exception: pass` blocks silently swallow ALL errors in template rendering and instruction modifier application. TypeError, AttributeError, and other real bugs are hidden.

**Impact:** Broken system prompts silently degrade agent quality. No logging, no indication of failure.

---

### 5. RunState.agent_stack Mutation Without Synchronization
**File:** `src/orchestrator/agent/types.py` (~line 343)

`run_state.agent_stack` is modified from multiple locations (executor.py, handoff_executor.py) without synchronization. In parallel workflows, concurrent agents could corrupt the stack via simultaneous append/pop operations.

**Impact:** Agent routing becomes non-deterministic in parallel execution scenarios.

---

### 6. Circular Dependency Pattern in HandoffExecutor ↔ Executor
**File:** `src/orchestrator/agent/execution/handoff_executor.py` (~lines 50-52)

HandoffExecutor stores a reference to Executor, which stores a reference back. This is set after initialization via `set_executor()`. If the executor isn't set before `execute_handoff()` is called, it fails silently with None.

**Impact:** Order-dependent initialization makes testing harder and creates fragile coupling.

---

### 7. Session History Filtering Fragility
**File:** `src/orchestrator/agent/services/session_service.py` (~lines 94-125)

Message filtering assumes a specific sequence: user → system → assistant → tool → assistant. If two assistant messages occur in a row (no tool call), both get saved instead of just the final one. This could happen when the LLM returns a plain response on one turn, then makes tool calls on the next.

**Impact:** Session history grows incorrectly; irrelevant intermediate messages are persisted.

---

### 8. Missing Validation for ToolContextState Injection
**File:** `src/orchestrator/agent/execution/message_builder.py` (~lines 136-142)

Tool context state is loaded from persistence but never validated. Corrupted or missing fields cause silent failures during injection.

**Impact:** Tool execution may receive wrong context, producing incorrect results.

---

## MEDIUM

### 9. Implicit State Mutation in HandoffData Serialization
**File:** `src/orchestrator/agent/types.py` (~lines 264-295)

`HandoffData.from_dict()` silently creates a new timestamp via `datetime.now(UTC)` if the timestamp field is missing from the data dict, rather than raising an error. Timestamp data loss is hidden.

---

### 10. Off-by-One in SessionService Message Filtering
**File:** `src/orchestrator/agent/services/session_service.py` (~lines 86-89)

`start_index = max(0, initial_count - 1)` offset is correct but fragile. The `-1` adjustment is non-obvious and error-prone for future modifications.

---

### 11. Hardcoded Timeout Values Throughout
**Files:** `runner.py`, `circuit_breaker.py`, `persistence/state.py`, `config.py`

CircuitBreaker defaults (5 threshold, 60s cooldown), session TTL (24h), and other values are buried in code. No easy way to tune without code changes.

---

### 12. Missing Handoff Depth Check in Handler
**File:** `src/orchestrator/agent/execution/handoff_executor.py`

HandoffExecutor validates cycles but doesn't check depth against `_handoff_manager._max_depth` before cycle check. A chain could exceed depth without being detected as a cycle.

---

### 13. Incorrect Error Logging in HandoffExecutor
**File:** `src/orchestrator/agent/execution/handoff_executor.py` (~lines 100-116)

If `target_agent` is None but `handoff_def` exists, it logs "Agent exists but not in registry" and returns failure. It should attempt to use the definition instead of failing immediately.

---

## LOW

### 14. Missing Input Validation in Tool Handler
**File:** `src/orchestrator/agent/services/tool_service.py` (~lines 84-90)

Malformed JSON tool arguments silently produce empty dicts (`{}`). Tool calls proceed with empty args instead of failing loudly.

---

### 15. Incomplete Span Context in Executor
**File:** `src/orchestrator/agent/execution/executor.py` (~lines 118-132)

Tracing spans for tool calls and handoffs don't have parent-child relationships. Observability becomes fragmented across turn boundaries.

---

### 16. Memory Leak Risk in Executor
**File:** `src/orchestrator/agent/execution/executor.py` (~lines 95-96)

`all_tool_summaries` list grows unbounded across turns. In long-running agents (max_turns=25), memory grows linearly with no cleanup.

---

### 17. Inconsistent Error Response Creation
**Files:** `runner.py`, `executor.py`, `workflow/` files

Different error responses use different status codes and formats. No standard error response factory makes caller error handling inconsistent.

---

### 18. Potential None Dereference in ValidationError Handling
**File:** `src/orchestrator/agent/utils/validation_utils.py` (~lines 80-81)

Direct dict access (`err['loc']`, `err['msg']`) without null checking. Assumes Pydantic error structure won't change.

---

### 19. Implicit Type Conversion in TokenUsage.add()
**File:** `src/orchestrator/agent/types.py` (~lines 162-182)

Assumes dict values are ints. None values or non-numeric types from model_usage dicts could cause addition errors.

---

### 20. Missing Context Reset Between Concurrent Runs
**File:** `src/orchestrator/agent/runner.py` (~lines 250-253)

`clear_run_artifacts()` isn't synchronized. Concurrent runs on the same runner instance could race on clearing shared state.
