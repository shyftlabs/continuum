---
name: continuum-tools-mcp
description: Connect MCP servers (Stdio/SSE/StreamableHTTP) to a Continuum agent, configure tool filtering, set up tool-context capture/injection (e.g. session_id), and read run artifacts (UI widgets, structured tool data). Invoke when the user asks "connect MCP", "filesystem tool", "remote API tool", "auto-capture session_id", "agent uses too many tools", or "expose widget data".
---

# Continuum MCP / Tools Skill

Authoritative source: [`docs/tools.md`](../../../docs/tools.md).

---

## Imports

```python
from continuum.tools import (
    MCPServerStdio, MCPServerSse, MCPServerStreamableHttp,
    ToolExecutor, MCPUtil,
    ToolContextConfig, ToolContextVariable,
    create_static_tool_filter, ToolFilterContext,
    MCPToolArtifact, RunArtifacts,
)
# `ToolExecutorConfig` is not in the `continuum.tools` namespace —
# import it from the executor module directly.
from continuum.tools.executor import ToolExecutorConfig
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
from continuum.agent import BaseAgent, AgentRunner

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
from continuum.tools import normalize_schema_for_llm, ensure_strict_json_schema

# Most users don't call these directly — MCPUtil.get_function_tools handles
# normalization. Reach for them if a model rejects an MCP tool's schema.
```

---

## MCPUtil

```python
tools = await MCPUtil.get_function_tools(server)
all_tools = await MCPUtil.get_all_function_tools([s1, s2])
text, art = await MCPUtil.invoke_mcp_tool_with_artifact(server, tool, '{"k":"v"}')
```

---

## Server trust

A third-party server's tool `description` / `inputSchema` reach the model's
prompt verbatim and steer its behaviour — treat adding a server like adding a
dependency.

Continuum strips invisible characters from fetched catalogues and invalidates the
tools cache on `connect()`. It does **not** filter description wording (a
description is legitimately instructional; filtering it breaks real tools).

What to do: read the descriptions before trusting a server; restrict with
`tool_filter=create_static_tool_filter(allowed_tool_names=[...])`; and bound the
damage with `PolicyStore.default_deny()` — the only control that helps against a
server you cannot vet. `tool_filter` matches `tool.name`, and a poisoned tool
keeps an innocent name.

```bash
continuum mcp inspect URL --name weather                   # descriptions + schemas
continuum mcp inspect URL --name weather --write-pins PATH # ...then record digests
continuum mcp diff weather --pins PATH                     # exit 1 while changed
continuum mcp approve weather --pins PATH --tool NAME      # per tool, merges
continuum mcp approve weather --pins PATH --all
continuum mcp rename OLD NEW --pins PATH                   # server name moved
```

`diff` / `approve` / `rename` work from files alone — you review the text the
agent saw, not whatever the server says now, and they behave identically for
every transport. Add `--record PATH` if the app sets `record_path`.

`mcp inspect` passes a **bare URL**, so it only reaches unauthenticated
streamable HTTP. For stdio, SSE, or an HTTP server behind an `Authorization`
header, review the object instead:

```python
from continuum.tools import review_server

await review_server(build_my_server())            # prints the same catalogue
await review_server(server, write_pins=PATH)      # ...and records it
```

Taking the object rather than CLI flags means headers, env, cwd and transport
are right by construction — a retyped `--cwd` reviews a different server than
the one your agent runs, and a pin file then vouches for something nobody read.
Reviewing a stdio server *launches* it; pinning bounds what the model is told to
do, not what a subprocess does at import.

```python
from continuum.tools import ToolTrustConfig

MCPServerStreamableHttp({"url": ...}, name="weather",
                        trust_config=ToolTrustConfig(pin_path=PATH))
```

| Field | Default | Meaning |
|---|---|---|
| `pin_path` | `None` | The approved catalogue. **Required for enforcement** — without it both settings below only report |
| `record_path` | sibling `.tool-pins-last-seen.json` | Where the runtime logs what was served |
| `on_unreviewed` | `"block"` | No entry in the approved catalogue. No false positives, and the one case pinning cannot defend alone |
| `on_drift` | `"warn"` | Approved tool whose text changed. Usually a typo fix; blocking by default gets the feature switched off |
| `on_change` | `None` | `Callable[[ToolChangeEvent], None]` — page oncall, fail CI |

`block` and `warn` both log a WARNING; only `allow` is silent. The mode decides
keep-vs-drop, not whether you are told. Env overrides: `MCP_ON_UNREVIEWED`,
`MCP_ON_DRIFT`.

**Two files, one writer each.** `tool-pins.json` is written only by a human
command (commit it); `.tool-pins-last-seen.json` only by the runtime (gitignore
it). When one file did both, the tripwire rewrote what the gate read and a
poisoned catalogue became "approved" one restart later. In production mount the
approval read-only and point `record_path` somewhere writable.

A pin means *unchanged since you looked*, never *safe* — review first.
Each pin holds two digests over description + `inputSchema`: `raw` (as sent —
what the tripwire compares) and `effective` (after invisible chars are stripped
— what the model reads and what the gate compares).

**Two boundaries.** Pinning covers the *catalogue*, not tool *results* — a
server with pristine descriptions can still inject through what it returns;
that is what `AgentConfig(tool_data_labels=...)` plus a `PolicyStore` rule is
for. And the gate runs on each `list_tools()` fetch, not on each tool call, so
an app that builds its registry once at startup gets one check per process.

**Always pass `name=` to a server.** It becomes the `<server>__<tool>` prefix
the model sees and that `PolicyStore` resources match, **and** it is the key
your approvals are filed under. Without it both are derived from the URL
(`tool:sse_https_db_internal_example_com_mcp__delete_user`) and change whenever
the URL does — silently breaking policy rules, and loudly orphaning every
approval. Continuum warns once per server when a derived name is namespaced or
pinned. `mcp inspect` prints the `policy resource:` strings to use.

---

## Tool namespacing

`namespace_tools=True` is the **default** on both `ToolExecutor` and
`MCPUtil.get_*_function_tools()`. Tools reach the LLM as `<server>__<tool>`
(e.g. `weather__get_forecast`), so two servers can expose the same tool name.
The prefix is sanitized to the provider's `^[a-zA-Z0-9_-]{1,64}$`.

Which name a setting matches:

| Setting | Name |
|---|---|
| `tool_filter` allow/block lists | **raw** (`read_file`) |
| `ToolExecutor(tool_registry={server: [...]})` | **raw** |
| `ToolContextVariable(capture_from=, inject_into=)` | **raw** |
| `tool-pins.json` tool keys | **raw** (under a server-name key) |
| `AgentConfig(tool_data_labels=)` | **either** — exact match wins |
| `PolicyStore` resources | **namespaced** (`tool:weather__read_file`) |
| `ToolAttentionConfig(always_promote=)` | **namespaced** |

Rule of thumb: scoped to one server → raw; operating on the merged LLM-facing
list → namespaced.

Pin files are raw-keyed because the trust gate runs inside `list_tools()`,
before any prefix is applied — so flipping `namespace_tools` does not
invalidate your approvals. `tool_data_labels` takes either spelling, and an
entry matching no tool (or several) is logged once per agent.

---

## Don't

- Don't forget `await server.connect()` — top cause of "no tools".
- Don't forget `await executor.initialize()` if you build the executor
  with a `tool_registry`.
- Don't mix `namespace_tools` settings between `ToolExecutor` and
  `MCPUtil.get_*_function_tools()` — the model would call names the
  registry can't resolve. Both default to `True`.
- Don't use a dynamic `tool_filter` without passing `metadata` to whatever
  builds the tool list — `initialize(metadata=...)` /
  `refresh_registry(..., metadata=...)` on `ToolExecutor`, or
  `get_all_function_tools(..., metadata=...)`. Omit it and the filter sees
  `None`, every tool is excluded "for safety", and the agent silently gets
  zero tools.
- Don't filter the `ToolExecutor` registry per caller — it is built once and
  shared, and must contain everything dispatchable. Per-caller lists come
  from `MCPUtil.get_all_function_tools(metadata=...)`; per-turn narrowing
  comes from tool-attention.
- Don't have duplicate tool names across servers when
  `namespace_tools=False` — `ToolExecutor.initialize()` and
  `get_all_function_tools()` both raise `MCPError`.
- Don't write `PolicyStore` rules against bare tool names — resources match
  the namespaced key (`tool:weather__read_file`). With
  `namespace_tools=False` it is the bare name; `mcp inspect` prints both
  because it cannot know which you use.
- Don't set `on_unreviewed="block"` without a `pin_path` — there is nowhere an
  approval can live, so nothing is enforced. Continuum warns when you ask for
  it explicitly.
- Don't re-approve a server whose *name* changed — that re-blesses whatever it
  serves now, without reading. `continuum mcp rename` moves the entry instead.
- Don't change `use_structured_content=True` casually — it changes what
  the LLM sees.
- Don't expose unsafe tools to a low-trust agent — use `tool_filter` or
  build a per-call filtered list (see `playground/commerce-chat` in the
  framework repo).
