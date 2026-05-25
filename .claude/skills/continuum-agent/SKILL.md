---
name: continuum-agent
description: Build BaseAgent instances and run them with AgentRunner — covers fields, lifecycle hooks, structured outputs, ReAct mode, instruction modifiers, and the full execution flow. Invoke when the user asks "create an agent", "configure max_turns", "add lifecycle hooks", "structured output with Pydantic", or anything around the core agent abstraction.
---

# Continuum Agent Skill

Authoritative source: [`docs/agent.md`](../../../docs/agent.md).

---

## Core API

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentConfig, AgentMemoryConfig, RunnerConfig
from orchestrator.agent.types import (
    Handoff, MemoryScope, RunContext, AgentResponse, EventType,
)
```

## Minimal agent

```python
agent = BaseAgent(
    name="my-agent",                    # required, [A-Za-z0-9_-]+
    instructions="...",                 # supports {slot} templates
    model="gpt-4o-mini",                # provider chosen by prefix
    temperature=0.7,
    max_tokens=None,
)

runner = AgentRunner()
resp = await runner.run(agent, "user input", user_id="u1", session_id="s1")
print(resp.content)
print(resp.usage.total_tokens)
print(resp.structured_output)           # populated if output_schema set
```

## Important `BaseAgent` parameters

| Parameter | Default | Notes |
|---|---|---|
| `name` | required | unique id |
| `instructions` | `""` | supports `{template_vars}` |
| `model` | `settings.default_llm_model` | prefix routes provider |
| `tools` | `[]` | LLM-shaped dicts |
| `mcp_servers` | `[]` | Auto-discovered tool sources |
| `handoffs` | `[]` | `[Handoff(target_agent=..., description=...)]` |
| `memory_config` | `AgentMemoryConfig()` | search_memories=True / store_memories=True |
| `output_schema` | `None` | Pydantic model |
| `template_vars` | `{}` | static slots |
| `examples` | `[]` | few-shot |
| `instruction_modifiers` | `[]` | dynamic prompt rewriters |
| `on_start` / `on_end` / `on_error` / `on_tool_call` / `on_handoff` | `None` | sync callables |
| `config` | `AgentConfig()` | max_turns=25, react_mode, reasoning_mode, scanners, … |
| `gateway_mode` | `None` | Smart Gateway routing: `"strict"` / `"modest"` / `"quality"` |
| `policy_store` | `None` | Security policy store for request validation |

## Structured output

```python
from pydantic import BaseModel

class Plan(BaseModel):
    intent: str
    steps: list[str]

agent = BaseAgent(name="planner", instructions="...", output_schema=Plan)
resp = await runner.run(agent, "...")
plan: Plan = resp.structured_output
```

## Streaming

```python
from orchestrator.agent.types import EventType

async for ev in runner.run_stream(agent, "..."):
    if ev.type == EventType.CONTENT_DELTA:
        print(ev.data["content"], end="", flush=True)
```

## Memory scope (the agent enum, not the dataclass!)

```python
from orchestrator.agent.types import MemoryScope
# MemoryScope.SHARED / USER / AGENT / RUN / CONVERSATION  — string enum
```

## ReAct + reasoning modes

```python
agent = BaseAgent(
    name="thinker",
    instructions="...",
    config=AgentConfig(react_mode=True, reasoning_mode=False),
)
```

`react_mode` injects a hidden `think` tool the LLM must call before
producing a final answer. `reasoning_mode` does a silent "think first"
pass before the main loop.

## RAG context

```python
agent = BaseAgent(
    name="rag-agent",
    instructions="Answer using PROVIDED CONTEXT.",
    config=AgentConfig(rag_context=retrieved_docs, require_context=True),
)
```

## Hooks

```python
def on_tool_call(agent, tool_name, args):
    print(f"{agent.name} -> {tool_name}({args})")

agent = BaseAgent(name="...", instructions="...", on_tool_call=on_tool_call)
```

Hooks are sync. Async hooks are not awaited.

---

## Execution flow (what `runner.run()` does)

1. Validate input against `agent.input_schema` (if any)
2. Build `RunContext` and `RunState`; start trace
3. Load tool-context state from session
4. Assemble messages: system prompt → ReAct scaffold → tool context →
   memory facts (Qdrant) → session history (Redis) → optional RAG →
   sanitized user input → optional context compression
5. Optional reasoning pass (silent)
6. Loop up to `max_turns`: LLM call → run tools / handoffs → feed results back
7. Persist session, store memories, return `AgentResponse`

---

## Don't

- Don't pass `role=` / `content=` to `SessionClient.add_message` — pass
  a `ChatMessage` object.
- Don't import `MemoryScope` from `orchestrator.memory` and then pass
  it to `AgentMemoryConfig` — that's the dataclass; the agent module
  re-exports the **enum** of the same name.
- Don't make hooks async — they're sync callables.
- Don't change `max_turns` to a huge number; if a loop fails to
  converge in 25 turns the prompt is the problem.
