# Orla Integration Update — 2026-05-01

Changes introduced in `tomli-dev-orla` relative to `tomli-dev`.

---

## Bug Fixes

### 1. Shared Agent Config Mutation in Workflows

**Files:** `workflow/sequential.py`, `parallel.py`, `scatter.py`, `loop.py`, `reflection.py`, `debate.py`, `planner.py`, `supervised.py`

Every workflow suppressed sub-agent session saves by mutating `agent.config.log_to_session = False` on the shared agent object, then restoring it in a `finally` block. If the same agent was used in two concurrent requests, one request could silently suppress logging for the other.

**Fix:** Added `suppress_session_log: bool` to `RunContext` (per-request, not shared). `RunFinalizer` checks `context.suppress_session_log` before writing to Redis. Workflows now set `context.suppress_session_log = True` at the top — no shared-state mutation, no fragile try/finally restore.

---

### 2. Parallel Context Isolation

**Files:** `workflow/parallel.py`, `workflow/scatter.py`, `agent/types.py`

`ParallelAgent` and `ScatterAgent` passed the *same* `RunContext` object to all concurrent branch tasks. Branches writing to `context.metadata`, `context.retrieved_memories`, `context.agent_stack`, etc. would stomp on each other.

**Fix:** Added `RunContext.branch_copy()` method that shallow-copies all mutable collections (`metadata`, `retrieved_memories`, `agent_stack`, `handoff_chain`, `tags`, `data_labels`, `usage`) into isolated copies. Both agents now call `context.branch_copy()` for each branch task.

---

### 3. Handoff Bugs

**Files:** `execution/handoff_executor.py`, `agent/runner.py`

Three separate fixes:

- **`max_turns=0`**: `context.max_turns - run_state.turn_count` could reach zero when passed to a handoff sub-run, causing it to immediately refuse to execute. Fixed with `max(1, context.max_turns - run_state.turn_count)`.
- **`data_labels` not propagated**: Orla-style data sensitivity labels on `RunContext` were dropped across handoffs. Fixed by copying `context.data_labels` in `HandoffExecutor`.
- **Streaming handoffs were fake**: In `run_stream()`, detecting a handoff only appended `"Handoff to {target} initiated. Note: Full handoff support requires non-streaming mode."` — the handoff never actually ran. Now calls `self._handoff_executor.execute_handoff()` and emits real `HANDOFF_END` / `HANDOFF_RETURN` / `CONTENT_COMPLETE` events with the actual response.

---

### 4. `NoneType` on `agent.memory_config`

**File:** `execution/message_builder.py`

`agent.memory_config.search_memories` raised `AttributeError` when `agent.memory_config` was `None`.

**Fix:** Added null guard: `if agent.memory_config and agent.memory_config.search_memories`.

---

### 5. Wrong Attribute on Tool List in Error Metadata

**File:** `execution/run_lifecycle.py`

Error metadata extraction called `t.name` on `ToolDefinition` objects, but names live at `t.function.name`. This silently produced empty strings or crashed.

**Fix:** `t.function.name if hasattr(t, "function") else t.get("function", {}).get("name", "")` — handles both typed objects and raw dicts.

---

### 6. Reasoning Pass Missing Full Context

**File:** `execution/executor.py`

`_run_reasoning_pass()` only accepted `session_id`, so `priority` and `stage_priority` could not be forwarded to the LLM call.

**Fix:** Signature changed from `session_id: str | None` to `context: RunContext`. Priority values now flow through the reasoning pass the same way they do for main-turn calls.

---

### 7. MCP `asyncio.CancelledError` Not Caught on Connect

**File:** `tools/mcp.py`

`asyncio.CancelledError` is not a subclass of `Exception` in Python 3.8+, so the `except Exception` block in `connect()` missed it, leaving connections partially initialised without cleanup.

**Fix:** Changed to `except (Exception, asyncio.CancelledError)`.

---

### 8. MCP `isError` Not Wrapped for LLM

**File:** `tools/executor.py`

When an MCP server returned `isError=True` in `CallToolResult`, the raw result text was passed to the LLM with no structure or signal that it was an error.

**Fix:** When `artifact.is_error` is `True`, the result is now wrapped as `{"error": ..., "error_type": "MCPToolError", "is_error": True}` so the LLM receives a consistent error envelope.

---

### 9. MCP Cleanup Tears Down Connection While Calls Are In-Flight

**File:** `tools/mcp.py`

When `cleanup()` was called during agent teardown, any in-flight `call_tool()` calls failed mid-execution because the session was torn down underneath them.

**Fix:** Added `_active_calls` counter and `_no_active_calls: asyncio.Event`. `call_tool()` increments the counter before the call and decrements in `finally`. `cleanup()` waits for `_no_active_calls` with a 30-second timeout before tearing down (Orla-style drain → close → signal done).

---

### 10. `structuredContent` Used Unconditionally

**File:** `tools/util.py`

Tool result `structuredContent` was always preferred over `content` when present, regardless of whether the server intended it to be used by the LLM.

**Fix:** Now guarded by `server.use_structured_content` flag.

---

### 11. Tool Fallback Was Silent

**File:** `agent/services/tool_service.py`

When a tool fell back from the agent's executor to the global executor, nothing was logged, making it very hard to diagnose unexpected tool routing.

**Fix:** Added `logger.warning("⚠️ TOOL FALLBACK: ...")` when fallback occurs.

---

### 12. `mcp_servers` Without `tool_executor` Was Silent

**File:** `agent/runner.py`

Setting `agent.mcp_servers` without a `tool_executor` silently did nothing — the MCP servers were never connected or used.

**Fix:** `AgentRunner._prepare_run()` now raises `AgentConfigurationError` with a clear message and example setup if `mcp_servers` is set but `tool_executor` is `None`.

---

## New Features

### 13. Priority Dispatch System

**Files:** `llm/dispatcher.py` (new), `llm/client.py`, `agent/config.py`, `agent/types.py`, `workflow/router.py`

Two new dispatchers for LLM call scheduling under load:

- **`PriorityDispatcher`** — for external APIs (Anthropic, OpenAI, Bedrock). Runs `max_concurrent` worker coroutines. When all workers are busy, queued calls are served highest-priority-first. Equal priorities are FIFO.
- **`TwoLevelDispatcher`** — for internal/self-hosted models (vLLM, SGLang). Composite priority key `(-stage_priority, -request_priority, seq)` matching Orla's two-level scheduler:
  - **Stage level** (`AgentConfig.stage_priority`): static weight of the agent type — a "reply" agent outranks a "summarize" agent regardless of request urgency.
  - **Request level** (`RunContext.priority`): runtime weight of this specific request — set by the router based on user tier or query urgency.

New fields:
- `RunContext.priority` (int, 1–10, default 5): per-request dispatch priority.
- `AgentConfig.stage_priority` (int, 1–10, default 5): per-agent-type static weight.
- `Route.dispatch_priority` (int, default 5): stamped onto `RunContext.priority` by `RouterAgent` when a route is selected.

Priority values flow through all LLM call paths: main turn, reasoning pass, and structured output pass.

`LLMClient` accepts an optional `dispatcher` argument and forwards `priority` / `stage_priority` on every `chat()` call.

---

### 14. Access Control Policy Engine

**Files:** `security/__init__.py` (new), `security/policy.py` (new), `tools/executor.py`, `memory/client.py`, `agent/base.py`, `agent/exceptions.py`, `exceptions.py`

New `orchestrator.security` module implementing Orla-style deny-overrides ABAC:

- **`AccessPolicy`**: rule with `subjects` (glob patterns against caller identity), `resources` (glob patterns against resource name), `effect` (`"allow"` or `"deny"`).
- **`PolicyStore`**: in-memory store. Evaluation order: explicit deny → explicit allow → open default (allow).
- Resource prefixes: `tool:<name>`, `memory:<scope>`, `data:<label>`.

**Enforcement points:**
- `ToolExecutor.execute_tool_call()` checks `tool:<tool_name>` before calling the MCP server. Denials produce a `POLICY DENIED: ...` message to the LLM (not an exception) and are logged at INFO.
- `MemoryClient.add()` and `search()` check `memory:<scope>` before reading or writing.
- `BaseAgent` gains `policy_store: PolicyStore | None` field, propagated through `clone_for_handoff()`.
- `ToolService` passes the agent's `policy_store` and `agent.name` as `subject` when calling `execute_tool_calls`.

**New exceptions:** `ToolAccessDeniedError`, `MemoryAccessDeniedError`, `InputBlockedError`.

**Example:**
```python
store = PolicyStore()
store.add_policy(AccessPolicy(
    name="no-delete",
    subjects=["billing_agent"],
    resources=["tool:delete_*"],
    effect="deny",
    denial_message="Deletion is not permitted from the billing agent.",
))
agent = BaseAgent(..., policy_store=store)
```

Now when billing_agent tries to call `delete_order`:                                                                                       
billing_agent calls `delete_order()`                        
→ PolicyStore checks: "billing_agent" + "tool:delete_order" → `DENY`                                                                       
→ LLM gets back: "Deletion is not permitted from the billing agent."
→ Tool never actually runs 

---

### 15. DAGAgent — Dependency-Aware Parallel Workflow

**File:** `workflow/dag.py` (new)

New workflow type. Unlike `ParallelAgent` (same input to all agents), DAG stages declare dependencies on other stages' outputs and receive those outputs as their input.

```python
dag = create_dag_agent(
    name="research-pipeline",
    stages=[
        ("fetch_a",    fetch_a_agent,    []),
        ("fetch_b",    fetch_b_agent,    []),
        ("synthesize", synthesize_agent, ["fetch_a", "fetch_b"]),
        ("format",     format_agent,     ["synthesize"]),
    ],
)
result = await runner.run(dag, "Research topic X")
```

- Stages with no dependencies start immediately; a stage starts as soon as all its dependencies complete.
- Uses `asyncio.Event` per stage for dependency gating. A shared `abort` event stops waiting stages early on `FAIL_FAST`.
- DFS cycle detection raises `DAGCycleError` before execution begins.
- `MergeStrategy.CONCATENATE` (default) or `STRUCTURED` (JSON dict) for combining predecessor outputs.
- `FailStrategy.FAIL_FAST` (default) or `REQUIRE_ALL`.

---

### 16. In-Process Function Tools (`MCPServerFunction`)

**File:** `tools/mcp.py`

New `MCPServerFunction` wraps plain Python callables as MCP-compatible tools with no subprocess or network connection — direct in-process execution.

```python
@function_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

server = MCPServerFunction("math", [add])
executor = ToolExecutor({server: None})
await executor.initialize()
```

Input schema is auto-generated from type hints (`str`, `int`, `float`, `bool`, `list`, `dict`, `Optional[X]`). Accepts four formats: `FunctionTool` dataclass, `@function_tool` decorated function, plain callable, or dict with `"fn"` key.

---

### 17. Tool Namespace Support

**Files:** `tools/executor.py`, `tools/util.py`

Previously, if two MCP servers exposed tools with the same name, a hard `MCPError` was raised with no workaround.

**Fix:** `namespace_tools=True` option on both `ToolExecutor` and `MCPUtil.get_all_function_tools()` prefixes names as `server_name__tool_name`. Both must use the same setting so LLM-facing names align with registry lookup.

---

### 18. `ToolExecutor.get_tool_definitions()`

**File:** `tools/executor.py`

New method that derives LLM-facing tool definitions from the already-built registry, avoiding a second `list_tools()` round-trip to the MCP server after `initialize()` already fetched them.

```python
executor = ToolExecutor({server: None})
await executor.initialize()
tools = executor.get_tool_definitions()  # no extra round-trip
agent = BaseAgent(..., tool_executor=executor, tools=tools)
```

---

### 19. JSON Path Variable Capture

**File:** `tools/executor.py`

`ContextConfig.variables` now supports `var.json_path` — a dot-notation path for extracting nested values from tool results (e.g. `"user.profile.id"`). Previously only top-level keys were capturable.

---

### 20. Tool List Debug Logging

**File:** `execution/message_builder.py`

Tool names and parameter summaries are now logged alongside the final prompt at debug level, making it easier to verify which tools were available for each agent turn.

---

### 21. `StreamExecutor` Deprecation Warning

**File:** `execution/stream_executor.py`

`StreamExecutor.execute_stream()` now emits a `DeprecationWarning` at call time. The class does not execute tools — it emits placeholder events only. Use `AgentRunner.run_stream()` instead.

---

## Summary

| Category | Count |
|---|---|
| Bug fixes | 12 |
| New features | 9 |

The most architecturally significant changes are the **session log suppression refactor** (fixes a concurrency correctness bug across every workflow), **parallel context isolation** (fixes data corruption in concurrent branches), the **priority dispatch system** (new infrastructure for load-based scheduling), and the **access control policy engine** (new security layer for tool and memory access).
