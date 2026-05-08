---
name: continuum-quickstart
description: Get a Continuum agent up and running — Python 3.13 venv, infra via docker compose, smallest possible BaseAgent + AgentRunner example. Invoke when the user asks "how do I start", "set up Continuum", "run my first agent", or is at the very beginning of a project.
---

# Continuum Quickstart Skill

Use this skill when the user is starting from scratch. The goal is to
get them to a running agent in a few minutes.

---

## Path A — Library consumer (5 commands)

```bash
# 1. Configure infra & secrets
cp .env.template .env
# Edit .env and set OPENAI_API_KEY

# 2. Start infra (Redis :6380, Qdrant :6333)
docker compose up -d

# 3. Create venv (Python 3.13 required)
python3.13 -m venv .venv && source .venv/bin/activate

# 4. Install the package
pip install shyftlabs-continuum

# 5. Run the smallest example (see snippet below)
python my_first_agent.py
```

## Path B — Framework contributor (clone + editable install)

```bash
git clone https://github.com/bhavik-shyftlabs/continuum.git
cd continuum
cp .env.template .env                       # add OPENAI_API_KEY
docker compose up -d                        # Redis + Qdrant
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,temporal,eval]"      # editable install with all extras

# Smoke-test against a runnable example app
python -m playground.sdk_feature_test
```

---

## Smallest possible agent

```python
import asyncio
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentMemoryConfig
from orchestrator.core.container import Container, ContainerConfig

async def main():
    # No infra needed for this example
    container = Container(ContainerConfig(
        enable_memory=False, enable_session=False, enable_langfuse=False,
    ))
    agent = BaseAgent(
        name="hello",
        instructions="Reply in one short sentence.",
        model="gpt-4o-mini",
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )
    response = await AgentRunner(container=container).run(agent, "Say hi.")
    print(response.content)

asyncio.run(main())
```

---

## Quick checks if something fails

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: orchestrator` | `source .venv/bin/activate` |
| `Failed to initialize mem0: Missing credentials` | Set `OPENAI_API_KEY` in `.env` (mem0 needs it for embeddings, even if you use Anthropic/Gemini for chat) |
| `redis ConnectionError` on port 6380 | `docker compose ps` — make sure the SDK Redis service is healthy |
| `ImportError` on Python startup | Wrong Python — must be 3.13 |
| `pip install -e .` fails on Python <3.13 | Switch to Python 3.13 (pyenv / uv) |

---

## Next steps to suggest

After "hello world" works, point them at one of:

- `playground/memory-modes-demo/` — all four memory scopes
- `playground/commerce-chat/` — plan-and-execute multi-agent + MCP
- `docs/agent.md` — full `BaseAgent` API reference
- `.claude/skills/continuum-agent/` — the agent-building skill

---

## What NOT to do

- Don't suggest `pip install litellm` or any LiteLLM code paths — it
  was removed.
- Don't change the default Redis (6380) or Qdrant (6333) ports —
  they're wired into `Settings`.
- Don't add new infra services to `docker-compose.yml` casually —
  Redis, Qdrant, and the Langfuse stack are the canonical set.
- Don't write to `src/orchestrator/` if you only intended to consume
  the library — install via pip and write your code outside the source
  tree.
