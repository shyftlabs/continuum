# Temporal — Registering Custom Agents

Workflows look up agents **by name** from a registry at activity
execution time. Any `BaseAgent` works — including workflow agents like
`SequentialAgent` or `RouterAgent` — as long as it's registered before
the worker spins up an activity.

`from orchestrator.temporal import (
    AgentRegistry, get_agent_registry, reset_agent_registry,
)`

---

## 1 · Register agents

```python
from orchestrator.agent import BaseAgent
from orchestrator.temporal import get_agent_registry

registry = get_agent_registry()                    # thread-safe singleton

registry.register(BaseAgent(
    name="summarizer",
    instructions="Summarize the input.",
    model="gpt-4o-mini",
))
registry.register(BaseAgent(
    name="translator",
    instructions="Translate to French.",
))
```

Or batch:

```python
registry.register_many([summarizer, translator, reviewer])
```

The registry uses `agent.name` as the lookup key — names must be
unique across all registered agents.

---

## 2 · `AgentRegistry` API

| Method | Returns | Notes |
|---|---|---|
| `register(agent)` | `None` | Add (or overwrite) an agent |
| `register_many(agents)` | `None` | Bulk register |
| `get(name)` | `BaseAgent` | Raises `AgentNotRegisteredError` if missing |
| `list_agents()` | `list[str]` | Names |
| `set_runner_factory(factory)` | `None` | Custom factory `() -> AgentRunner` (used in activities) |
| `get_runner()` | `AgentRunner` | Returns the factory result, or builds a default runner |
| `set_notification_handler(handler)` | `None` | `async def handler(params: NotificationParams) -> None` |
| `get_notification_handler()` | `Callable \| None` | |
| `clear()` | `None` | Wipe the registry (testing) |

---

## 3 · Wiring shared services to workflow runs

If you want activities to share a memory client, session client, or
custom container, supply a `runner_factory`:

```python
from orchestrator.agent import AgentRunner
from orchestrator.core.container import Container, ContainerConfig

container = Container(ContainerConfig(enable_memory=True, enable_session=True))

def make_runner() -> AgentRunner:
    return AgentRunner(container=container)

registry.set_runner_factory(make_runner)
```

Without a factory, the activity falls back to a default `AgentRunner()`
which uses the global `Container` singleton — that's usually fine for
most projects.

---

## 4 · Workflow agents inside Temporal

Workflow agents (`SequentialAgent`, `RouterAgent`, etc.) are themselves
`BaseAgent` subclasses, so they register the same way:

```python
from orchestrator.agent import create_sequential_agent, BaseAgent

researcher = BaseAgent(name="researcher", instructions="…")
writer     = BaseAgent(name="writer",     instructions="…")
editor     = BaseAgent(name="editor",     instructions="…")

pipeline = create_sequential_agent(
    name="content-pipeline",
    agents=[researcher, writer, editor],
)

registry.register_many([researcher, writer, editor, pipeline])

# In WorkflowInput:
WorkflowInput(steps=[{"type": "agent", "agent_name": "content-pipeline"}], initial_input="…")
```

The pipeline runs as one Temporal activity — durability is at the
**Temporal step** boundary, not at the inner sub-agent boundary.

---

## 5 · Inside `run_agent_activity`

For reference, this is what the activity does (so you understand what
becomes durable):

1. `agent = registry.get(params.agent_name)`
2. `runner = registry.get_runner()`
3. `activity.heartbeat(f"Running agent: {params.agent_name}")`
4. `resp = await runner.run(agent, params.input,
                            session_id=params.session_id,
                            user_id=params.user_id,
                            metadata=params.metadata, tags=params.tags)`
5. Return `AgentActivityResult.from_agent_response(resp)`

If `run()` raises, the activity catches it and returns a result with
`status="error"` and `error=str(exc)`. The workflow then decides what
to do based on its retry policy and step config.

---

## 6 · Exception: `AgentNotRegisteredError`

`from orchestrator.temporal import AgentNotRegisteredError`

Raised when a workflow references an unknown `agent_name`. Carries
`agent_name` and `available_agents` in its context dict for fast
debugging.

---

## 7 · Patterns

### Per-environment agents

```python
def production_agents():
    return [
        BaseAgent(name="summarizer", model="gpt-4o", ...),
        BaseAgent(name="translator", model="claude-sonnet-4-20250514", ...),
    ]

def dev_agents():
    return [
        BaseAgent(name="summarizer", model="gpt-4o-mini", ...),
        BaseAgent(name="translator", model="gpt-4o-mini", ...),
    ]

registry.register_many(production_agents() if env == "prod" else dev_agents())
```

### Hot-replacing an agent

```python
registry.register(BaseAgent(name="summarizer", instructions="<v2 prompt>"))
```

`register()` overwrites in place — new workflow runs pick up the change
instantly. Workflows already in flight continue with the original
because Temporal replays history deterministically.

### Tests

```python
from orchestrator.temporal import reset_agent_registry

def setup_function():
    reset_agent_registry()
```
