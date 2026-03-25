# Tools (MCP), Observability & Infrastructure Issues

---

## CRITICAL

### 1. Unvalidated Tool Argument Injection via Context Variables
**File:** `src/orchestrator/tools/executor.py` (~lines 244-319)

`_inject_context_variables()` injects values from `_context_state` into tool arguments based only on parameter name matching — no schema validation, no type checking. An attacker who compromises an MCP server can inject malicious values into subsequent tool calls.

```python
# Attack scenario:
# 1. Attacker-controlled MCP server returns: {"session_id": "; rm -rf /"}
# 2. Value captured as context variable
# 3. Injected into next tool call: server.call_tool("dangerous_tool", {"session_id": "; rm -rf /"})
```

**Impact:** Remote code execution if any downstream tool passes arguments to system calls.

**Fix:** Validate injected values against parameter schemas. Type-check all context variable captures.

---

### 2. MCP Server Cleanup Race Condition
**File:** `src/orchestrator/tools/mcp.py` (~lines 425-467)

The `cleanup()` method has a double-check lock, but there's a race window where multiple async tasks can pass the outer `if` check before the lock is acquired. More critically, `exit_stack.aclose()` is called while another task may still be using `self.session`, tearing down the connection mid-call.

**Impact:** MCP connections terminated during active tool calls. Cascading failures across tool executions.

**Fix:** Use proper shutdown sequence: stop accepting new calls → wait for in-flight calls → cleanup.

---

### 3. Uncaught Exception in Concurrent Tool Execution
**File:** `src/orchestrator/tools/executor.py` (~lines 599-606)

`asyncio.gather(*tasks)` is called without `return_exceptions=True`. While individual tool calls have try/except, if any raises an uncaught exception (from the error handling path itself), the gather raises and terminates the entire batch. Semaphore and rate limiter tokens may leak.

**Impact:** Batch tool execution fails unpredictably. Resource leaks under error conditions.

---

### 4. Incomplete Shutdown of Resources in Container
**File:** `src/orchestrator/core/container.py` (~lines 431-521)

`shutdown()` wraps resource cleanup in `asyncio.wait_for()` with 3-second timeout, but doesn't force-close resources if timeout is hit. MCP server cleanup can hang (see issue #2), leaving Redis connections, Qdrant connections, and HTTP sessions in partially-closed states.

**Impact:** Connection leaks. Resource exhaustion on subsequent restarts.

---

## HIGH

### 5. Fragile JSON Error Detection via String Matching
**File:** `src/orchestrator/tools/mcp.py` (~lines 372-401), `util.py` (~lines 533-540)

JSON errors detected by checking strings like `"JSON"`, `"Expecting value"`, `"JSONDecodeError"`. Different JSON libraries and Python versions have different error messages. Should use `isinstance(e, json.JSONDecodeError)` instead.

**Impact:** JSON parsing errors from MCP servers misclassified, leading to incorrect error handling.

---

### 6. Race Condition in Global Container Singleton
**File:** `src/orchestrator/core/container.py` (~lines 593-618)

Double-checked locking for global container, but if two threads call `get_container()` with different configs simultaneously, only the first config wins. The second config is silently ignored with no warning.

**Impact:** Silent misconfiguration in multi-threaded initialization.

---

### 7. Missing Validation of Schema Normalization Results
**File:** `src/orchestrator/tools/util.py` (~lines 164-178)

When schema normalization fails, the code falls back to a minimal fix (adding `properties: {}`) but doesn't validate the result. Tools may be registered with invalid schemas that LLM providers reject at runtime.

**Impact:** Tool calls fail at execution time instead of at schema registration time.

---

### 8. Health Check Thread Pool Saturation
**File:** `src/orchestrator/core/health.py` (~lines 294-306, 371-381)

Health checks use `asyncio.to_thread()` for blocking operations. Under high concurrency, if the thread pool is saturated, health checks timeout even though services are healthy. No timeout set at the thread level, only at the overall check level.

**Impact:** False-negative health checks. Startup/readiness probes incorrectly report failure.

---

### 9. Silent Context Variable Capture Failure
**File:** `src/orchestrator/tools/executor.py` (~lines 321-384)

If a tool result is not valid JSON, `_capture_context_variables()` logs a warning and returns without capturing. The caller doesn't know capture failed. Session state variables may be lost, breaking stateful tool interactions.

**Impact:** Stateful multi-tool workflows silently lose context between calls.

---

## MEDIUM

### 10. Unprotected Tool Registry Modification
**File:** `src/orchestrator/tools/executor.py` (~lines 197, 403-405, 612-615)

`tool_registry` dict is modified without synchronization. Concurrent `refresh_registry()` and `execute_tool_call()` can race — the registry could be cleared while a tool call is looking up a tool.

**Impact:** `KeyError` during tool execution. Tools temporarily unavailable during refresh.

---

### 11. Langfuse Trace ID Not Propagated in Batch Execution
**File:** `src/orchestrator/tools/executor.py` (~lines 579-606)

`@trace_tool()` decorator creates a parent span, but individual tool calls may not inherit it if `trace_id` is None. Batch tool execution traces become fragmented in Langfuse.

---

### 12. Missing Error Context in MCP Tool Calls
**File:** `src/orchestrator/tools/executor.py` (~lines 527-577)

When tool calls fail, the error is serialized as JSON but the original exception object and traceback are lost. Debugging tool failures in production becomes very difficult.

---

### 13. Rate Limiter Lock Contention / Potential Deadlock
**File:** `src/orchestrator/tools/executor.py` (~lines 84-132)

The rate limiter holds a lock during `asyncio.sleep()`. Under high concurrency, all tasks serialize through the lock during sleep, defeating the rate limiting purpose. The sleep should be outside the lock.

---

### 14. Tool Filter Result Not Validated
**File:** `src/orchestrator/tools/mcp.py` (~lines 220-258)

Dynamic tool filters can return any value (strings, lists, etc.), evaluated as truthy/falsy. A buggy filter returning a string will silently include/exclude tools based on string truthiness.

---

### 15. Silent Tool Filter Failure Excludes Tools
**File:** `src/orchestrator/tools/mcp.py` (~lines 251-256)

If a tool filter raises an exception, the tool is silently excluded with only a log message. Developers won't realize their filter is broken — tools just mysteriously disappear.

---

### 16. Non-Atomic Context Variable Set Operations
**File:** `src/orchestrator/tools/types.py` (~lines 403-419)

`set()` modifies `_variables` and `_metadata` separately. If an exception occurs between the two updates, state becomes inconsistent.

---

### 17. Overly Broad Except in Lifecycle Initialization
**File:** `src/orchestrator/core/lifecycle.py` (~lines 405-415)

`except Exception as e:` catches everything including `KeyboardInterrupt` and `SystemExit`. User interruptions during initialization are treated as regular errors.

---

## LOW

### 18. Inconsistent Error Logging in Observability Provider Manager
**File:** `src/orchestrator/observability/provider_manager.py` (~lines 108-112)

Provider failures logged with `logger.warning()` and `exc_info=False`, suppressing tracebacks. Other places use `logger.error()` with `exc_info=True`.

---

### 19. Hardcoded Shutdown Timeout Values
**File:** `src/orchestrator/core/container.py` (~lines 484, 499, 512)

All shutdown timeouts hardcoded to 3 seconds. Not configurable per deployment.

---

### 20. Unclear Context Variable Scope Documentation
**File:** `src/orchestrator/tools/types.py` (~lines 260-265)

The `scope` parameter ("session" vs "run") semantics are unclear when multiple agents share the same session. Undocumented interaction with distributed scenarios.
