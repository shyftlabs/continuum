---
name: continuum-cursor
description: Set up Cursor for a Continuum project — .cursor/rules/ MDC file, glob patterns, and what to include so Cursor generates correct Continuum code. Invoke when the user asks "set up Cursor for my project", "Cursor rules for Continuum", ".cursor/rules", or "MDC file for Continuum".
---

# Continuum + Cursor Skill

Cursor reads `.cursor/rules/*.mdc` files. The Continuum repo ships
`.cursor/rules/continuum.mdc` — copy it into any downstream project.

---

## Minimal MDC file for a Continuum project

Create `.cursor/rules/continuum.mdc`:

```markdown
---
description: Continuum agent framework conventions for this project
globs: ["src/**/*.py", "tests/**/*.py", "playground/**/*.py"]
alwaysApply: true
---

# Continuum Project Rules

This project uses **Continuum** (import: `orchestrator`, package: `shyftlabs-continuum`).

## Setup invariants

- Python 3.13. Venv: `python3.13 -m venv .venv`
- All public APIs are **async**. Entrypoints: `asyncio.run(main())`
- Always `load_dotenv()` before importing orchestrator settings
- `OPENAI_API_KEY` required even when using Anthropic/Gemini (mem0 embedder)
- Infra via docker compose: Redis :6380, Milvus :19530 (or Qdrant :6333)

## Core imports

\`\`\`python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentConfig, AgentMemoryConfig, RunnerConfig
from orchestrator.agent.types import Handoff, MemoryScope, RunContext, EventType
from orchestrator.llm import LLMClient, LLMConfig
from orchestrator.memory import MemoryClient
from orchestrator.session import SessionClient
\`\`\`

## Provider routing

Model string prefix determines provider (when gateway is not set):
- `gpt-*`, `openai/*` → OpenAI SDK
- `claude-*`, `anthropic/*` → Anthropic SDK  
- `gemini/*`, `google/*` → Gemini via OpenAI-compat

When `SMART_GATEWAY_URL` is set, all models route through GatewayProvider automatically.
Use `gateway_mode="strict"|"modest"|"quality"` on BaseAgent to control routing tier.

## Critical don'ts

- No `litellm` — removed entirely
- No `response_format` on Anthropic — instruct via system prompt and parse manually
- No JSON mode + tools on Gemini simultaneously — pick one
- MemoryScope enum: `from orchestrator.agent.types import MemoryScope` (not from `orchestrator.memory`)
- SessionClient.add_message takes `ChatMessage(...)` not `role=` kwargs
```

---

## Glob patterns

Match these globs to ensure rules apply to all relevant files:

```yaml
globs:
  - "src/**/*.py"          # application source
  - "tests/**/*.py"        # tests
  - "playground/**/*.py"   # playground scripts
  - "*.py"                 # root-level scripts
```

Use `alwaysApply: true` for framework-level rules that apply regardless
of which file is open.

---

## Multiple rule files (recommended)

Split into focused files for large projects:

```
.cursor/rules/
├── continuum.mdc        # framework API — alwaysApply: true
├── agents.mdc           # project-specific agent descriptions
├── testing.mdc          # test conventions — globs: tests/**
└── api.mdc              # FastAPI/endpoint conventions
```

---

## Don't

- Don't put the full Continuum API reference inline — reference `AGENTS.md` or docs instead.
- Don't use `alwaysApply: true` for files with large content — it inflates every request's context.
- Don't duplicate rules across files — Cursor merges all matching MDC files per request.
