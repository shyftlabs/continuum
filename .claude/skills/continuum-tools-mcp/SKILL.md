---
name: continuum-tools-mcp
description: Connect MCP servers (Stdio/SSE/StreamableHTTP) to a Continuum agent, configure tool filtering, set up tool-context capture/injection (e.g. session_id), and read run artifacts (UI widgets, structured tool data). Invoke when the user asks "connect MCP", "filesystem tool", "remote API tool", "auto-capture session_id", "agent uses too many tools", or "expose widget data".
---

# Continuum MCP / Tools Skill

Authoritative source: [`docs/tools.md`](../../../docs/tools.md).

---

## Imports

```python
from orchestrator.tools import (
    MCPServerStdio, MCPServerSse, MCPServerStreamableHttp,
    ToolExecutor, MCPUtil,
    ToolContextConfig, ToolContextVariable,
    create_static_tool_filter, ToolFilterContext,
    MCPToolArtifact, RunArtifacts,
)
# `ToolExecutorConfig` is not in the `orchestrator.tools` namespace —
# import it from the executor module directly.
from orchestrator.tools.executor import ToolExecutorConfig
```

---

## Three transports

| Transport | When |
|---|---|
| `MCPServerStdio` | Local subprocess MCP server |
| `MCPServerSse` | Legacy SSE-based remote |
| `MCPServerStreamableHttp` | **Recommended** for any modern remote |

```python
local = MCPServerStdio(
    {"command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]},
    name="local",
)
await local.connect()                       # ALWAYS connect first

remote = MCPServerStreamableHttp(
    {"url": "https://example.com/mcp",
     "headers": {"Authorization": "Bearer …"}},
    name="remote",
)
await remote.connect()
```

---

## Quickest agent wiring

```python
from orchestrator.agent import BaseAgent, AgentRunner

agent = BaseAgent(
    name="tool-agent",
    instructions="Use the tools to answer.",
    mcp_servers=[local, remote],            # tool discovery is automatic
)
resp = await AgentRunner().run(agent, "...")
```

---

## Manual ToolExecutor (for shared executors / restrictions)

```python
executor = ToolExecutor(
    tool_registry={
        local: None,                        # None = expose all of this server's tools
        remote: ["search", "ingest"],       # restrict
    },
    config=ToolExecutorConfig(
        max_concurrent_calls=5,
        rate_limit_per_second=10.0,
        timeout_seconds=30.0,
    ),
)
await executor.initialize()                  # REQUIRED when constructed with tool_registry

agent = BaseAgent(name="…", instructions="…", tool_executor=executor)
```

---

## Tool filtering

```python
# Static
server = MCPServerStreamableHttp(
    {"url": "..."},
    tool_filter=create_static_tool_filter(allowed_tool_names=["search", "fetch"]),
)

# Dynamic (sync or async)
async def admin_only(ctx: ToolFilterContext, tool) -> bool:
    return ctx.metadata.get("role") == "admin"

server = MCPServerStreamableHttp({"url": "..."}, tool_filter=admin_only)
await server.list_tools(metadata={"role": "admin"})
```

---

## Tool context (capture + inject)

When tool A returns a `session_id` (or `auth_token`, etc.) that tool B
needs, the framework can capture and re-inject automatically:

```python
ctx_cfg = ToolContextConfig(
    variables=[
        ToolContextVariable(
            name="session_id",
            capture_from=["create_session"],     # only from this tool
            inject_into=None,                    # any tool with `session_id` param
            scope="session",                     # persists across runs in the session
            sensitive=False,
        ),
        ToolContextVariable(name="auth_token", scope="session", sensitive=True),
    ],
    auto_capture_common=True,                    # session_id, auth_token, user_id, …
    namespace=None,                              # defaults to MCP server name
    inject_into_system_prompt=True,
)
server = MCPServerStreamableHttp({"url": "..."}, context_config=ctx_cfg)
```

---

## Run artifacts (widgets, structured tool data)

```python
resp = await runner.run(agent, "...")
artifacts = resp.run_artifacts                  # dict-shaped if any captured
```

Tools often return both text (for the LLM) and structured payloads
(for a UI). The framework captures both — text goes into the model
context, structured data lands in `run_artifacts`.

---

## Schema utilities

```python
from orchestrator.tools import normalize_schema_for_llm, ensure_strict_json_schema

# Most users don't call these directly — MCPUtil.get_function_tools handles
# normalization. Reach for them if a model rejects an MCP tool's schema.
```

---

## MCPUtil

```python
tools = await MCPUtil.get_function_tools(server)
all_tools = await MCPUtil.get_all_function_tools([s1, s2])  # raises on duplicate names
text, art = await MCPUtil.invoke_mcp_tool_with_artifact(server, tool, '{"k":"v"}')
```

---

## Don't

- Don't forget `await server.connect()` — top cause of "no tools".
- Don't forget `await executor.initialize()` if you build the executor
  with a `tool_registry`.
- Don't have duplicate tool names across servers if you use
  `get_all_function_tools()` — it raises `MCPError`.
- Don't change `use_structured_content=True` casually — it changes what
  the LLM sees.
- Don't expose unsafe tools to a low-trust agent — use `tool_filter` or
  build a per-call filtered list (see `playground/commerce-chat` in the
  framework repo).
