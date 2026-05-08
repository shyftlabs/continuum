---
name: continuum-quickstart
description: Get a Continuum agent up and running in this hackathon kit — Python 3.13 venv, infra via docker compose, smallest possible BaseAgent + AgentRunner example. Invoke when the user asks "how do I start", "set up Continuum", "run my first agent", or is at the very beginning of a project.
---

# Continuum Quickstart Skill

Use this skill when the user is starting from scratch with the
hackathon kit. The goal is to get them to a running agent in 5
commands.

---

## The 5-command path

```bash
# 1. Configure
cp .env.template .env
# Edit .env and set OPENAI_API_KEY

# 2. Start infra (Redis :6380, Qdrant :6333)
docker compose up -d

# 3. Create venv (Python 3.13 is required)
python3.13 -m venv .venv && source .venv/bin/activate

# 4. Install the framework wheel
pip install -r requirements.txt

# 5. Run the smallest example
python examples/01_hello_agent.py
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
| `redis ConnectionError` | `docker compose ps` — make sure `continuum-redis-sdk` is healthy |
| `ImportError` on Python startup | Wrong Python — must be 3.13 |

---

## Next steps to suggest

After "hello world" works, point them at one of:

- `examples/02_memory_session.py` — memory + session demo
- `examples/03_workflow_sequential.py` — chain agents together
- `docs/agent.md` — full `BaseAgent` API reference
- `.claude/skills/continuum-agent/` — the agent-building skill (this kit)

---

## What NOT to do

- Don't suggest `pip install litellm` or any LiteLLM code paths — it
  was removed.
- Don't tell them to clone the framework repo — they don't need to.
- Don't change Redis/Qdrant ports — defaults are wired.
- Don't write to `wheels/` — it's a sealed binary distribution.
