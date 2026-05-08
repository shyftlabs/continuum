# Tools / MCP Module

Continuum is **MCP-native** — every tool surface is exposed via the
[Model Context Protocol](https://modelcontextprotocol.io). Three
transports ship out of the box (Stdio, SSE, StreamableHTTP), all sharing
the same agent-facing API.

What this module gives you:
- `MCPServer` (abstract) and three concrete transports
- `ToolExecutor` — concurrency, rate-limiting, context capture/injection,
  artifact collection
- `MCPUtil` — discover tools from servers, normalize their schemas for
  any LLM provider
- A **tool-context** mechanism that captures things like `session_id` /
  `auth_token` from one tool's output and injects them into subsequent
  tool calls (so the LLM doesn't have to thread them around)
- A **run-artifact** mechanism that captures structured tool output
  (UI widgets, tables, charts) separately from the text the LLM sees

---

## 1 · Quick start

```python
from orchestrator.tools import (
    MCPServerStreamableHttp, ToolExecutor, MCPUtil,
)
from orchestrator.agent import BaseAgent, AgentRunner

server = MCPServerStreamableHttp(
    {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer …"}},
    name="example",
)
await server.connect()

executor = ToolExecutor({server: None})            # None = expose all tools
await executor.initialize()

agent = BaseAgent(
    name="tool-agent",
    instructions="Use the available tools to answer.",
    mcp_servers=[server],
)
resp = await AgentRunner().run(agent, "What's in the latest report?")
```

---

## 2 · MCP Server classes

All three inherit from the abstract `MCPServer` and share a common
constructor shape (only the `params` payload differs).

### Common constructor parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `params` | TypedDict | required | See per-transport details below |
| `cache_tools_list` | `bool` | `False` | Cache `list_tools()` across calls; invalidate via `server.invalidate_tools_cache()` |
| `name` | `str \| None` | auto | Human-readable name (auto-generated from command/url if `None`) |
| `client_session_timeout_seconds` | `float \| None` | `5` | Read timeout for the MCP `ClientSession` |
| `tool_filter` | `ToolFilter \| None` | `None` | See Section 5 |
| `use_structured_content` | `bool` | `False` | Prefer `tool_result.structured_content` over text content |
| `max_retry_attempts` | `int` | `0` | Retries for transient failures |
| `retry_backoff_seconds_base` | `float` | `1.0` | Exponential backoff base |
| `message_handler` | `MessageHandlerFnT \| None` | `None` | Hook for raw MCP messages |
| `context_config` | `ToolContextConfig \| None` | `None` | See Section 4 |
| `validate_on_connect` | `bool` | `False` | If `True`, calls `list_tools()` after connect to fail fast on a broken server |

### Common methods

| Method | Description |
|---|---|
| `await server.connect()` | Open the transport |
| `await server.cleanup()` | Close cleanly |
| `await server.list_tools(metadata=None)` | Discover available tools |
| `await server.call_tool(tool_name, arguments)` | Invoke a tool directly |
| `await server.list_prompts()` / `get_prompt(name, arguments)` | MCP prompt API |
| `server.invalidate_tools_cache()` | Force a re-fetch of `list_tools()` |
| `async with server: ...` | Context manager calls connect/cleanup automatically |

### `MCPServerStdio`

`from orchestrator.tools import MCPServerStdio`

```python
mcp = MCPServerStdio({
    "command": "npx",                     # required
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"],
    "env": {"NODE_ENV": "production"},
    "cwd": "/path/to/cwd",
    "encoding": "utf-8",
    "encoding_error_handler": "strict",   # "strict" | "ignore" | "replace"
})
```

### `MCPServerSse`

`from orchestrator.tools import MCPServerSse`

```python
mcp = MCPServerSse({
    "url": "https://example.com/sse",     # required
    "headers": {"Authorization": "Bearer …"},
    "timeout": 5.0,
    "sse_read_timeout": 300.0,
})
```

### `MCPServerStreamableHttp` *(recommended for remote)*

`from orchestrator.tools import MCPServerStreamableHttp`

```python
from datetime import timedelta

mcp = MCPServerStreamableHttp({
    "url": "https://example.com/mcp",     # required
    "headers": {"Authorization": "Bearer …"},
    "timeout": timedelta(seconds=10),     # or float
    "sse_read_timeout": timedelta(minutes=5),
    "terminate_on_close": True,
    # "httpx_client_factory": custom_factory,
})
```

---

## 3 · `ToolExecutor`

`from orchestrator.tools import ToolExecutor, ToolExecutorConfig`

```python
executor = ToolExecutor(
    tool_registry={                       # dict[MCPServer, list[str] | None]
        local_server: None,               # None = expose all of this server's tools
        remote_server: ["search", "ingest"],   # restrict to these tools
    },
    config=ToolExecutorConfig(
        max_concurrent_calls=5,
        rate_limit_per_second=10.0,       # 0 disables
        timeout_seconds=30.0,
    ),
    context_state=None,
)
await executor.initialize()
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `await initialize()` | `None` | Build the internal `tool_name → (server, tool)` registry. **Required** if you constructed with a `tool_registry` |
| `await execute_tool_call(tool_call, trace_id=None, span_id=None, metadata=None)` | `ChatMessage` (role=`tool`) | Run one tool call with context injection/capture, rate-limiting, timeout |
| `await execute_tool_calls(tool_calls, trace_id=None, span_id=None, metadata=None)` | `list[ChatMessage]` | Concurrent execution; one failure does not cancel others |
| `clear_run_artifacts(run_id=None)` | `None` | Reset captured artifacts at the start of a run |
| `get_available_tools()` | `list[str]` | Names known to this executor |
| `await refresh_registry(tool_registry)` | `None` | Atomic rebuild — keeps the old registry until the new one validates |

### Properties

- `context_state: ToolContextState` (read/write)
- `run_artifacts: RunArtifacts` (read-only)

---

## 4 · Tool context (capture & inject)

A common MCP pattern: the first tool call returns a `session_id` (or
`auth_token`, `merchant_id`, etc.) that every subsequent call must
include. Continuum captures this automatically and re-injects it.

### `ToolContextConfig`

```python
from orchestrator.tools import ToolContextConfig, ToolContextVariable

ctx_cfg = ToolContextConfig(
    variables=[
        ToolContextVariable(
            name="session_id",
            capture_from=["create_session"],
            inject_into=None,                     # all tools with `session_id` param
            json_path=None,
            scope="session",                      # "session" | "run"
            override_llm_value=True,
            required=False,
            sensitive=False,
        ),
        ToolContextVariable(name="auth_token", scope="session", sensitive=True),
    ],
    auto_capture_common=True,                     # session_id, auth_token, user_id, merchant_id, store_id, …
    namespace="my-server",
    inject_into_system_prompt=True,
)

server = MCPServerStreamableHttp({"url": "..."}, context_config=ctx_cfg)
```

`ToolContextVariable` fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Variable to capture/inject |
| `capture_from` | `list[str] \| None` | `None` | Tool names to capture from (None = all) |
| `inject_into` | `list[str] \| None` | `None` | Tool names to inject into (None = all matching) |
| `json_path` | `str \| None` | `None` | JSONPath; default uses `name` as a top-level key |
| `scope` | `Literal["session","run"]` | `"session"` | Run-scoped values are cleared between runs |
| `override_llm_value` | `bool` | `True` | Override an LLM-provided value with the captured one |
| `required` | `bool` | `False` | If `True`, the call fails when the variable is missing |
| `sensitive` | `bool` | `False` | Mask in logs and `to_dict()` |

### `ToolContextState`

Where captured values live. Thread-safe; persists across runs at session
scope. Methods include `get`, `set`, `get_all`, `get_all_namespaces`,
`has`, `clear_namespace`, `clear_run_scoped`, `merge_from`,
`to_prompt_context`, `is_empty`, `to_dict`, `from_dict`.

### Auto-captured names

```
session_id, sessionId, session,
auth_token, token, access_token, authToken,
user_id, userId,
merchant_id, merchantId,
store_id, storeId
```

Sensitive (always masked in logs):

```
auth_token, token, access_token, authToken, bearer
```

---

## 5 · Tool filtering

```python
from orchestrator.tools import create_static_tool_filter, ToolFilterContext

# Static (allowlist / blocklist)
server = MCPServerStreamableHttp(
    {"url": "..."},
    tool_filter=create_static_tool_filter(allowed_tool_names=["search", "fetch"]),
)

# Dynamic (sync or async callable)
async def admin_only(context: ToolFilterContext, tool) -> bool:
    return context.metadata.get("role") == "admin"

server = MCPServerStreamableHttp({"url": "..."}, tool_filter=admin_only)
await server.list_tools(metadata={"role": "admin"})
```

---

## 6 · Run artifacts

Tools often produce **two** outputs: text (for the LLM) and structured
data (for the UI). Continuum captures both and exposes the structured
data via `AgentResponse.run_artifacts`.

### `MCPToolArtifact`

| Field | Description |
|---|---|
| `tool_name`, `server_name` | Identification |
| `meta: dict \| None` | MCP `_meta` (widget templates, etc.) |
| `structured_content: dict \| None` | The data for rendering |
| `text_content: str \| None` | The LLM-facing text |
| `raw_content: list[dict] \| None` | Raw MCP `content` items |
| `is_error: bool` | |
| `timestamp`, `latency_ms` | |

Methods: `has_widget()`, `get_widget_template()`, `to_dict()`, `from_dict(data)`.

### `RunArtifacts`

| Method | Returns |
|---|---|
| `add_artifact(a)` | — |
| `clear()` / `is_empty()` | `None` / `bool` |
| `get_by_tool(tool_name)` | `list[MCPToolArtifact]` |
| `get_latest_by_tool(tool_name)` | `MCPToolArtifact \| None` |
| `get_widgets()` | `list[MCPToolArtifact]` |
| `get_structured_data()` | merged `dict` |
| `to_dict()` / `from_dict(data)` | round-trip |

Access at the application level via `response.run_artifacts`. See
`playground/commerce-chat/multi_agent.py` (in the framework repo) for
a real-world pattern that injects an MCP `session_id` into each
captured widget before forwarding to a frontend.

---

## 7 · `MCPUtil`

`from orchestrator.tools import MCPUtil`

| Method | Returns | Notes |
|---|---|---|
| `await MCPUtil.get_function_tools(server, normalize_schemas=True, strict_mode=False, metadata=None)` | `list[ToolDefinition]` | LLM-shaped tools from one server |
| `await MCPUtil.get_all_function_tools(servers, normalize_schemas=True, strict_mode=False, metadata=None)` | `list[ToolDefinition]` | Across multiple servers; raises `MCPError` on duplicate tool names |
| `MCPUtil.to_function_tool(tool, server, normalize_schemas=True, strict_mode=False)` | `ToolDefinition` | Convert one MCP tool |
| `await MCPUtil.invoke_mcp_tool(server, tool, input_json, trace_id=None, span_id=None, metadata=None)` | `str` | JSON-string result |
| `await MCPUtil.invoke_mcp_tool_with_artifact(server, tool, input_json, ...)` | `tuple[str, MCPToolArtifact]` | Result text + full artifact |

Schema helpers:

```python
from orchestrator.tools import normalize_schema_for_llm, ensure_strict_json_schema
```

These fix common MCP schema oddities (arrays without `items`, objects
without `properties`, missing `type`) so that strict OpenAI/Gemini JSON
schemas don't reject them.

---

## 8 · Exceptions

`from orchestrator.tools.exceptions import (
    ToolError, MCPError, MCPConnectionError, MCPToolError,
)`

`MCPError` constructors carry `server_name` and `tool_name` in their
context dict.

---

## 9 · Common patterns

### Multiple servers in one agent

```python
local = MCPServerStdio({"command": "python", "args": ["my_server.py"]}, name="local")
remote = MCPServerStreamableHttp({"url": "https://api.example.com/mcp"}, name="remote")
await local.connect(); await remote.connect()

agent = BaseAgent(name="multi", mcp_servers=[local, remote], instructions="...")
```

### Restricting tools per-agent (security)

```python
plan_tool_names = {step.tool_name for step in plan.steps}
filtered = [t for t in all_tools
            if t.get("function", {}).get("name") in plan_tool_names]
agent = BaseAgent(name="executor", tools=filtered, instructions="...")
```

### Capturing a session id from `create_session`

```python
ctx = ToolContextConfig(
    variables=[ToolContextVariable(name="session_id", capture_from=["create_session"])],
    auto_capture_common=False,
)
server = MCPServerStreamableHttp({"url": "..."}, context_config=ctx)
```

The first time `create_session` runs, its output's `session_id` is
captured. Every subsequent tool call that has a `session_id` parameter
gets the captured value injected automatically.

---

## 10 · Gotchas

- **`await server.connect()` first.** Forgetting this is the #1 source
  of "no tools available" reports.
- **`ToolExecutor.initialize()` is required** when you build it with a
  `tool_registry` argument.
- **Tool names must be unique across servers** when using
  `MCPUtil.get_all_function_tools(...)`.
- **`use_structured_content=True`** changes what gets sent to the LLM —
  prefer leaving it `False` unless you specifically want the structured
  payload as text.
- **Schemas:** if the LLM rejects a tool's schema, `normalize_schemas=True`
  (the default) usually fixes it. Add `strict_mode=True` if your provider
  needs strict OpenAI-style schemas.
- **`run`-scoped vs `session`-scoped** context variables behave
  differently — `run` values are wiped between runs, `session` values
  persist as long as the session.
