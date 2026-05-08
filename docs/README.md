# Continuum Documentation

Reference documentation for the **Continuum** agentic framework
(`shyftlabs-continuum` on PyPI, importable as `orchestrator`).

> **Note**: every example in these docs has been verified against the
> actual source under `src/orchestrator/`. Imports, parameter names,
> defaults, and signatures match the codebase as of the latest
> release.

---

## Where to start

If you've never used Continuum before, read in this order:

1. [`installation.md`](installation.md) — Python 3.13 setup, env vars,
   troubleshooting
2. [`agent.md`](agent.md) — the `BaseAgent` / `AgentRunner` mental model;
   the most important file
3. [`llm.md`](llm.md) — provider routing (no LiteLLM), structured
   outputs, context compression
4. [`tools.md`](tools.md) — connecting MCP servers, tool-context capture,
   run artifacts
5. [`memory.md`](memory.md) and [`session.md`](session.md) — persistent
   memory and conversation history
6. [`observability.md`](observability.md) — tracing, metrics, error
   reporting
7. [`core.md`](core.md) — the DI container, lifecycle, health checks,
   protocols
8. [`temporal/`](temporal/) — durable workflows with approval gates

---

## Module index

| Module | Doc | What it gives you |
|---|---|---|
| `orchestrator.agent` | [agent.md](agent.md) | `BaseAgent`, `AgentRunner`, 9 workflow patterns, handoffs |
| `orchestrator.llm` | [llm.md](llm.md) | `LLMClient`, provider routing, structured outputs, compression |
| `orchestrator.memory` | [memory.md](memory.md) | mem0 + Qdrant long-term memory; `IntelligentMemoryClient` |
| `orchestrator.session` | [session.md](session.md) | Redis-backed conversation history |
| `orchestrator.tools` | [tools.md](tools.md) | MCP servers (Stdio/SSE/StreamableHTTP), `ToolExecutor`, artifacts |
| `orchestrator.observability` | [observability.md](observability.md) | Langfuse tracing, metrics, error reporter |
| `orchestrator.core` | [core.md](core.md) | `Container`, `OrchestratorLifecycle`, health checks |
| `orchestrator.temporal` | [temporal/](temporal/) | Durable workflows, approval gates |
| `orchestrator.config`, `protocols`, `exceptions` | [core.md §7-§8](core.md) | Settings, runtime-checkable protocols, exception hierarchy |

---

## Conventions

- **Async-first** — every public API is `async`; wrap in
  `asyncio.run(main())`. The few sync entry points are clearly marked
  (`*_sync` methods).
- **Direct provider SDKs** — Continuum calls OpenAI / Anthropic /
  Gemini directly. **No LiteLLM** anywhere.
- **Provider routing by model prefix** — `gemini/...` →  Gemini,
  `claude-…` / `anthropic/...` → Anthropic, everything else → OpenAI.
- **MCP-native tooling** — every tool is exposed via the Model Context
  Protocol; no bespoke adapters.
- **Three IDs that travel together** — `trace_id` (Langfuse),
  `session_id` (Redis), and `run_id` (mem0). The framework keeps
  them synchronized via `contextvars`.
- **`OPENAI_API_KEY` is required at startup** — mem0's default embedder
  is OpenAI. Set the key, change the embedder provider, or disable
  memory entirely.

---

## Runnable example apps

The repository ships a `playground/` tree with end-to-end example apps
that exercise different feature combinations:

| Path | Demonstrates |
|---|---|
| [`../playground/sdk_feature_test/`](../playground/sdk_feature_test/) | SDK-feature smoke test |
| [`../playground/memory-modes-demo/`](../playground/memory-modes-demo/) | All four memory scopes (USER / AGENT / RUN / SHARED) |
| [`../playground/commerce-chat/`](../playground/commerce-chat/) | Plan-and-execute multi-agent + MCP tool restriction |
| [`../playground/fetch-agent/`](../playground/fetch-agent/) | Tool-using agent with MCP fetch server |
| [`../playground/assortment/`](../playground/assortment/) | FastAPI-served agent endpoint |

A separate `continuum-hackathon` kit packages a pre-built wheel and
minimal starter examples for downstream users; this repository is the
source of truth.

---

## Working with AI assistants

The repo ships AI-assistant configuration so Claude Code / Codex /
Cursor can write Continuum code correctly without prompting:

- [`../AGENTS.md`](../AGENTS.md) — canonical knowledge pack
- [`../CLAUDE.md`](../CLAUDE.md) — Claude Code project rules
- [`../.cursor/rules/continuum.mdc`](../.cursor/rules/continuum.mdc) — Cursor always-on rule
- [`../.claude/skills/`](../.claude/skills/) — invocable per-topic skills

---

## Out-of-date references in legacy material

If you stumble across older docs, READMEs, or example code that
mentions any of the following, they are stale:

- **LiteLLM** — removed in framework commit `657607a`. Use `LLMClient`
  with direct provider SDKs.
- **`SessionClient.add_message(session_id, role=..., content=...)`** —
  the real signature is `add_message(session_id, message: ChatMessage)`.
- **`from orchestrator.agent import RouterAgent, create_router_agent`
  with a `Route(target=...)`** — the field is `Route(agent_name=..., description=...)`.
- **`from orchestrator.memory import MemoryClient` with
  `client.add(text, user_id=...)`** — works, but `add()` accepts strings,
  lists of strings, OR lists of `{role,content}` dicts; the messages-style
  is the recommended path.
