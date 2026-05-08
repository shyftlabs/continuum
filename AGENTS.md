# Continuum — AI Assistant Knowledge Pack

> Loaded automatically by **OpenAI Codex CLI**, **Claude Code** (via
> `CLAUDE.md` import), **Cursor** (via `.cursor/rules/continuum.mdc`),
> and other agents that respect the `AGENTS.md` convention.
>
> Goal: give the assistant enough context to write correct Continuum
> code on the first try.

---

## What is Continuum?

A Python 3.13 agentic framework by ShyftLabs. Provides:

- `BaseAgent` (dataclass) + `AgentRunner` (executor)
- Multi-LLM via direct provider SDKs (OpenAI / Anthropic / Gemini — **not LiteLLM**)
- Two-tier memory: short-term **Redis sessions** + long-term **mem0 + Qdrant**
- **MCP-native** tool integration (Stdio / SSE / StreamableHTTP)
- 9 **workflow agents** (Router, Sequential, Parallel, Loop, Reflection, Planner, Debate, Scatter, SupervisedSequential)
- Optional **Temporal** durable workflows with human-in-the-loop approval gates
- **Langfuse** observability and tracing

Distributed as a Python package — `pip install shyftlabs-continuum`
(or `pip install -e .` from a checkout). Importable as
`from orchestrator import ...` — the wheel name on disk is
`shyftlabs-continuum`, the import name is `orchestrator`.

---

## Repository layout

```
continuum/
├── AGENTS.md / .cursor/rules/continuum.mdc          # this file
├── .claude/skills/continuum-*                        # 13 invocable skills
├── README.md                                         # project overview
├── pyproject.toml                                    # package metadata
├── src/orchestrator/                                 # framework source
│   ├── agent/, llm/, memory/, session/, tools/,
│   │   observability/, core/, evaluation/, temporal/
│   ├── config.py, protocols.py, exceptions.py
├── docs/                                             # API reference
│   ├── agent.md, llm.md, memory.md, session.md, tools.md,
│   │   observability.md, core.md, installation.md
│   └── temporal/*.md
├── tests/                                            # pytest suite (unit / integration / e2e)
├── playground/                                       # runnable example apps
│   ├── memory-modes-demo/, sdk_feature_test/,
│   │   commerce-chat/, fetch-agent/, assortment/
├── docker-compose.yml                                # Redis + Qdrant + Langfuse stack
└── .env.template                                     # API keys go here
```

A separate hackathon-kit repo ships a pre-built wheel and minimal
starter examples; this repo is the source of truth.

---

## Setup invariants (do not violate)

- **Python 3.13** is required. `python3.13 -m venv .venv && source .venv/bin/activate`.
- `OPENAI_API_KEY` is required even if you don't use OpenAI models, because
  mem0 (long-term memory) initializes its OpenAI embedder at startup.
- Redis runs on host port **6380** (mapped to container 6379).
- Qdrant runs on **6333** (REST) and **6334** (gRPC).
- Always `load_dotenv()` at the top of any example script.
- All public APIs are **async** — wrap in `asyncio.run(main())`.

---

## Core API

### Imports

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentConfig, AgentMemoryConfig, RunnerConfig
from orchestrator.agent.types import Handoff, MemoryScope, RunContext, EventType
from orchestrator.agent.workflow import (
    RouterAgent, SequentialAgent, ParallelAgent, LoopAgent,
    ReflectionAgent, PlannerAgent, DebateAgent, ScatterAgent,
    SupervisedSequentialAgent,
    create_sequential_agent, create_reflection_agent,
    create_planner_agent, create_debate_agent,
    create_scatter_agent, create_supervised_agent,
)
from orchestrator.core.container import Container, ContainerConfig, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager
from orchestrator.tools import (
    MCPServerStdio, MCPServerSse, MCPServerStreamableHttp,
    ToolExecutor, MCPUtil,
)
from orchestrator.observability import observe
```

### BaseAgent

```python
agent = BaseAgent(
    name="my-agent",                          # required, [a-zA-Z0-9_-]+
    instructions="You are ... {company}.",    # supports {slot} templates
    model="gpt-4o-mini",                      # or "claude-sonnet-4-20250514", "gemini/gemini-2.5-flash"
    temperature=0.3,
    tools=[...],                              # tool definitions (LLM-shaped)
    mcp_servers=[mcp1, mcp2],                 # auto-discovered tool sources
    handoffs=[Handoff(target_agent="other", description="...")],
    memory_config=AgentMemoryConfig(...),
    output_schema=MyPydanticModel,            # optional structured output
    template_vars={"company": "Acme"},
    examples=[{"input": "...", "output": "..."}],
    instruction_modifiers=[my_callable],      # (prompt, ctx) -> prompt
    on_start=hook, on_end=hook, on_error=hook,
    on_tool_call=hook, on_handoff=hook,
    config=AgentConfig(max_turns=10, reasoning_mode=False, react_mode=False),
)
```

### AgentRunner

```python
runner = AgentRunner()                                # uses global container
# or: AgentRunner(container=Container(ContainerConfig(...)))

# Non-streaming
resp = await runner.run(
    agent,
    "user message",
    user_id="user-123",         # used by memory scoping
    session_id="session-456",   # Redis history key
    metadata={"tier": "pro"},
    tags=["beta"],
)
print(resp.content)
print(resp.structured_output)   # if output_schema set
print(resp.run_artifacts)       # if tools captured artifacts (e.g. widgets)

# Streaming
async for event in runner.run_stream(agent, "..."):
    if event.type == EventType.CONTENT_DELTA:
        print(event.data["content"], end="")
```

### Memory scopes (`MemoryScope`)

| Scope | Visibility |
|---|---|
| `SHARED` | All agents, all users — global knowledge |
| `USER` | One user across all agents (default) |
| `AGENT` | One agent across all users |
| `RUN` | Single run only — ephemeral |

```python
memory_config=AgentMemoryConfig(
    search_memories=True,
    store_memories=True,
    search_scope=MemoryScope.USER,
    store_scope=MemoryScope.USER,
    search_limit=5,
)
```

### MCP tools

```python
local = MCPServerStdio(
    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]}
)
remote = MCPServerStreamableHttp(
    {"url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
)
await local.connect(); await remote.connect()
tools = await MCPUtil.get_function_tools(local)

agent = BaseAgent(name="tool-agent", instructions="...", mcp_servers=[local, remote])
# or: agent.tools=tools + agent.tool_executor=ToolExecutor({local: None, remote: None})
```

### Handoffs

```python
triage = BaseAgent(
    name="triage",
    instructions="Route to the right specialist.",
    handoffs=[
        Handoff(target_agent="billing",   description="Billing/payment issues"),
        Handoff(target_agent="technical", description="Technical support"),
    ],
)
runner.register_agent(billing_agent)   # required so the runner can resolve targets
runner.register_agent(technical_agent)
```

### Structured outputs

```python
from pydantic import BaseModel

class Plan(BaseModel):
    intent: str
    steps: list[str]

agent = BaseAgent(name="planner", instructions="...", output_schema=Plan)
resp = await runner.run(agent, "...")
plan: Plan = resp.structured_output
```

### Workflow agents

```python
pipeline = create_sequential_agent(
    name="content-pipeline",
    agents=[researcher, writer, editor],   # output of each feeds the next
)
debate = create_debate_agent(
    name="debate",
    pro_stance="Argue in favor.",            # strings, not agent instances
    con_stance="Argue against.",
    judge_instructions=None,                 # default judge prompt
)
reflect = create_reflection_agent(
    name="self-improve",
    agent=writer,
    max_reflections=2,                       # NOT `max_iterations`; factory has no `critic` kwarg
)
```

All workflow agents are themselves `BaseAgent` subclasses, so they nest.

---

## Common gotchas

- **`OPENAI_API_KEY` is mandatory** at framework startup, even if you only
  use Anthropic for the LLM, because mem0 instantiates an OpenAI embedder
  by default. To opt out of long-term memory entirely, set
  `MEMORY_ENABLED=false` in `.env` (and `enable_memory=False` in the
  `ContainerConfig` if constructing manually).
- `Container.memory_client` is **lazy** — it only initializes on first
  access. Don't be surprised if logs show the mem0 init firing midway
  through your script.
- The runner expects an **async event loop**. Use `asyncio.run(main())`.
- `BaseAgent.name` must match `[A-Za-z0-9_-]+` — handoff resolution uses it.
- For MCP servers, **always `await server.connect()` before passing to an agent**.
- The provider router in `orchestrator.llm.providers.get_provider` keys off
  the model string prefix:
  - `gemini/` or `google/` → Gemini
  - `claude/`, `anthropic/`, or starts with `claude-` → Anthropic
  - everything else → OpenAI (handles `gpt-*`, `azure/...`, `openai/...`)

---

## Doc map

When the participant asks a topical question, point them at the right doc.

| Topic | File |
|---|---|
| Agent dataclass, runner, lifecycle hooks | `docs/agent.md` |
| LLM client, providers, fallback, context compression | `docs/llm.md` |
| Memory architecture, mem0 + Qdrant | `docs/memory.md` |
| Sessions / Redis | `docs/session.md` |
| MCP tools, transports, filtering | `docs/tools.md` |
| Langfuse tracing, metrics, `@observe` | `docs/observability.md` |
| DI Container, lifecycle, health checks | `docs/core.md` |
| Setup, env vars, troubleshooting | `docs/installation.md` |
| Temporal: getting started, workflow patterns, custom agents/workflows, approvals | `docs/temporal/*.md` |

---

## Hackathon-specific tips

- Start from `playground/memory-modes-demo/` or `playground/sdk_feature_test/`
  and incrementally add memory/tools/handoffs.
- Use `docker compose ps` to confirm Redis + Qdrant are healthy before running memory examples.
- For tracing, run `docker compose up -d langfuse` (or full Langfuse stack) and visit `http://localhost:3000`.
- For durable workflows, install the temporal extra (`pip install -e ".[temporal]"`),
  run `docker compose --profile temporal up -d`, and visit `http://localhost:8080`.
- If you change `.env`, re-source / restart the venv shell — pydantic-settings reads on import.
- Editable install for development: `pip install -e ".[dev,temporal,eval]"`.
- Run the test suite with `pytest -m unit` (or `-m integration` once infra is up).

---

## Streaming events (`run_stream`)

`runner.run_stream(agent, input)` yields `AgentEvent` objects with the
following `EventType` values. Match on `event.type`; payload lives in
`event.data`.

| `EventType` | When | `event.data` keys |
|---|---|---|
| `RUN_START` | Run begins | `agent_name`, `input_preview` |
| `RUN_END` | Run completes | `status`, `latency_ms`, `usage` |
| `RUN_ERROR` | Run fails | `error`, `error_type` |
| `AGENT_START` / `AGENT_END` | Wrapping each agent (incl. handoff targets) | `agent_name` |
| `CONTENT_DELTA` | Token-level stream from the LLM | `content` (partial text) |
| `CONTENT_COMPLETE` | LLM emitted a full assistant message | `content` |
| `TOOL_CALL_START` / `TOOL_CALL_END` | Around each tool invocation | `tool_name`, `arguments`, `result` (on _END) |
| `TOOL_CALL_ERROR` | Tool raised | `tool_name`, `error` |
| `HANDOFF_START` / `HANDOFF_END` / `HANDOFF_RETURN` | Around agent transitions | `from_agent`, `to_agent`, `reason` |
| `MEMORY_RETRIEVAL` | Memory injected into prompt | `count`, `query` |
| `MEMORY_STORAGE` | New memories stored after the turn | `count` |
| `WORKFLOW_STEP` | Workflow agent transitioned to a new step | `step`, `agent_name` |
| `LOOP_ITERATION` | LoopAgent completed an iteration | `iteration`, `output` |

```python
from orchestrator.agent.types import EventType

async for ev in runner.run_stream(agent, "..."):
    if ev.type == EventType.CONTENT_DELTA:
        print(ev.data["content"], end="", flush=True)
    elif ev.type == EventType.TOOL_CALL_START:
        print(f"\n[tool: {ev.data['tool_name']}]")
```

---

## Memory CRUD reference

```python
from orchestrator.memory import MemoryClient

client = MemoryClient()                                  # uses env defaults

# Add (LLM-extracted facts)
await client.add(messages=[{"role":"user","content":"I'm vegetarian"}], user_id="u1")
await client.add("plain string also works", user_id="u1")

# Search (semantic)
result = await client.search("dietary preferences", user_id="u1", limit=5)
for entry in result.results:
    print(entry.memory, entry.score)

# CRUD by id
entry = await client.get(memory_id)
all_entries = await client.get_all(user_id="u1", limit=100)
await client.update(memory_id, "updated text")
await client.delete(memory_id)
await client.delete_all(user_id="u1")                    # wipe one user

# Versioning / housekeeping
versions = await client.history(memory_id)               # list[dict] of past versions
await client.reset()                                     # WIPE EVERYTHING — destructive
```

Sync mirrors of every method exist (`add_sync`, `search_sync`, …).

---

## Lifecycle hooks (sync callables)

```python
def on_start(agent, ctx):       ...   # ctx = {"context": RunContext, "input": Any}
def on_end(agent, ctx):         ...
def on_error(agent, exc, ctx):  ...
def on_tool_call(agent, name, args): ...
def on_handoff(agent, target, data): ...   # data = HandoffData

agent = BaseAgent(name="…", instructions="…",
                  on_start=on_start, on_end=on_end,
                  on_error=on_error, on_tool_call=on_tool_call,
                  on_handoff=on_handoff)
```

Hooks are sync. Async hooks are not awaited.

---

## Provider-specific quirks (handled automatically)

| Provider | Quirk | Framework handling |
|---|---|---|
| Anthropic | System messages must be top-level; tool results need wrapping | Provider auto-translates the OpenAI message format |
| Anthropic | `max_tokens` is required by the SDK | Defaults to `4096` if you don't set one |
| Anthropic | No native `response_format` | `check_response_format_support()` returns `False`; instruct via system prompt |
| Gemini | OpenAI-compat endpoint (no `google-generativeai` SDK) | `GeminiProvider` uses `OpenAI(base_url="…/v1beta/openai/")` |
| Gemini / Vertex | Tools + JSON mode are mutually exclusive | Framework auto-disables `json_mode` when `tools` are present |

---

## Common error patterns

```python
from orchestrator.agent import (
    AgentExecutionError, MaxTurnsExceededError, AgentToolError,
    HandoffNotAllowedError, HandoffDepthExceededError,
    HandoffTargetNotFoundError,
)
from orchestrator.llm import (
    LLMAuthenticationError, LLMRateLimitError, LLMTimeoutError,
    LLMContextLengthError, LLMServiceUnavailableError,
    LLMFallbackExhaustedError,
)
from orchestrator.memory import MemoryConfigurationError
from orchestrator.session import SessionMessageLimitError

try:
    resp = await runner.run(agent, "...", user_id="u1", session_id="s1")
except LLMAuthenticationError:
    ...   # bad / missing API key
except LLMRateLimitError:
    ...   # 429 — provider tier limit
except LLMContextLengthError:
    ...   # input + history exceeded model window
except MaxTurnsExceededError:
    ...   # tool/handoff loop didn't converge in `max_turns`
except HandoffTargetNotFoundError:
    ...   # forgot runner.register_agent(target)
except AgentExecutionError as e:
    ...   # general LLM/tool execution failure
```

`HandoffCycleDetectedError` exists but is **not** re-exported from
`orchestrator.agent` — import from `orchestrator.agent.exceptions`
directly when needed.

---

## Context window compression

Auto-runs before each LLM call when context approaches the threshold.
Override per-agent:

```python
from orchestrator.llm.context_management import (
    ContextManagementConfig, CompressionStrategy,
)
from orchestrator.agent.config import AgentConfig

agent = BaseAgent(
    name="long-context",
    instructions="…",
    config=AgentConfig(context_management=ContextManagementConfig(
        enabled=True,
        compression_threshold=0.8,                 # 80 % of model window
        summarization_model="gpt-4o-mini",
        keep_recent_messages=10,
        compression_strategy=CompressionStrategy.SMART,
    )),
)
```

---

## Few-shot + instruction modifiers

```python
def add_tier_note(prompt: str, ctx) -> str:
    if ctx.metadata.get("tier") == "enterprise":
        return prompt + "\nThis is an enterprise account — SLA priority."
    return prompt

agent = BaseAgent(
    name="adaptive",
    instructions="You are helping {user_name}. Today is {date}.",
    template_vars={"user_name": "Alice"},   # {date} is auto-resolved
    examples=[
        {"input": "Reset password", "output": "Settings → Security → Reset"},
        {"input": "Cancel subscription", "output": "Account → Plan → Cancel"},
    ],
    instruction_modifiers=[add_tier_note],
)
await runner.run(agent, "...", metadata={"tier": "enterprise"})
```

---

## Banned APIs (these will mislead participants)

- `from litellm import …` — LiteLLM was removed in commit `657607a`. Use `LLMClient`.
- `SessionClient.add_message(session_id, role=..., content=...)` — wrong; pass `ChatMessage(role=, content=)` as the second positional arg.
- `Route(target=..., description=...)` — the field is `agent_name=...`.
- `create_debate_agent(pro_agent=, con_agent=, judge_agent=)` — factory takes string `pro_stance` / `con_stance` / `judge_instructions`.
- `create_reflection_agent(critic=…, max_iterations=…)` — no `critic` kwarg; use `max_reflections`.
- `create_debate_agent`, `create_scatter_agent`, `create_supervised_agent` from `orchestrator.agent` — they're in `orchestrator.agent.workflow`.
- `ToolExecutorConfig` from `orchestrator.tools` — import from `orchestrator.tools.executor`.
- `report_error(e, context={...})` — `context` is a short string; dict goes in `metadata=`.
- `ObservationContext` with `async with` or `capture_output=True` — sync `with` only; no such kwarg.
- `obs.update_metadata(...)` — method is `add_metadata(...)`.
- `transfer_to_<target>` (handoff tool name) — actual prefix is `handoff_to_<target>`.
