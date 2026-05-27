---
name: continuum-codex
description: Set up OpenAI Codex CLI for a Continuum project — AGENTS.md content, project conventions, and what Codex needs to write correct Continuum code. Invoke when the user asks "set up Codex for my project", "AGENTS.md for Continuum", or "configure Codex CLI".
---

# Continuum + Codex CLI Skill

Codex CLI reads `AGENTS.md` from the project root automatically.
The Continuum repo ships a ready-made `AGENTS.md` — copy or symlink
it into any downstream project.

---

## Minimal AGENTS.md for a Continuum project

```markdown
# My Project — Agent Knowledge Pack

@/path/to/continuum/AGENTS.md

## Project-specific conventions

- Package: `my_app` (imports as `from my_app import ...`)
- Entry point: `src/my_app/main.py`
- Agents are defined in `src/my_app/agents.py`
- Tests: `pytest tests/` with `pytest-asyncio`

## Key agents in this project

| Agent | File | Purpose |
|---|---|---|
| `SupportAgent` | `agents/support.py` | Customer support triage |
| `BillingAgent` | `agents/billing.py` | Billing inquiries |
```

The `@import` pulls in the full Continuum API reference so Codex knows
every class, field, and pattern.

---

## What AGENTS.md must include for Continuum projects

Codex needs these facts to write correct code on the first try:

1. **Import name**: `orchestrator` (not `continuum` or `shyftlabs_continuum`)
2. **Async**: all public APIs are async — entrypoints need `asyncio.run()`
3. **Provider routing**: by model-string prefix, or via `SMART_GATEWAY_URL`
4. **Infrastructure**: Redis on `:6380`, Milvus/Qdrant for memory
5. **No LiteLLM**: direct provider SDKs only

Minimal addition to any `AGENTS.md`:

```markdown
## Continuum conventions

- Import: `from orchestrator.agent import BaseAgent, AgentRunner`
- All APIs async. Use `asyncio.run(main())` at entrypoints.
- Model routing: prefix-based (`gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini/*` → Gemini)
- Gateway: set `SMART_GATEWAY_URL` to route all models through the Smart Gateway
- Memory: Redis (:6380) for sessions, Milvus (:19530) or Qdrant (:6333) for long-term
- Load dotenv: always `load_dotenv()` before importing orchestrator settings
```

---

## Running Codex against a Continuum project

```bash
# Point Codex at the project
cd my-continuum-project
codex "add a handoff from SupportAgent to BillingAgent for billing questions"
```

Codex will read `AGENTS.md`, understand the Continuum API, and generate
correct `Handoff(target_agent=..., description=...)` code.

---

## Don't

- Don't omit `load_dotenv()` — Codex-generated scripts often miss it and settings fail to load.
- Don't tell Codex to use `litellm` — it's removed from Continuum.
- Don't let Codex guess at field names — always have `AGENTS.md` present so it reads the authoritative API.
