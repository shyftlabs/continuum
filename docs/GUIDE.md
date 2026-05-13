# Continuum Developer Guide

---

## 1. BaseAgent

### Default configuration (`AgentConfig` and `AgentMemoryConfig`)

```python
BaseAgent(
    name="my-agent",
    instructions="...",
    memory_config=AgentMemoryConfig(search_memories=True, store_memories=True),
    config=AgentConfig(log_to_session=True, session_history_turns=None),
)
```

**Memory (default):**

- `search_memories=True` — looks up `long-term memories` before responding
- `store_memories=True` — saves `long-term memories` after responding

**Session (default):**

- `log_to_session=True` — saves to session history (`short-term memory`)
- `session_history_turns=None` — loads last 20 turns of history (`short-term memory`)

### `session_history_turns` behaviour


| Value            | Behavior                                    |
| ---------------- | ------------------------------------------- |
| `None` (default) | load last **20** turns from Redis           |
| `0`              | **skip** Redis call entirely — load nothing |
| `5`              | load last **5** turns from Redis            |


---

## 2. Run an agent

### You must create a session before calling `runner.run()`

If you want session history to work (load prior turns, save new ones), you must create the session first. Without it, the agent runs statelessly — messages silently fail to save and history is not loaded.

```python
# Step 1: create session
session_id = await session_client.get_or_create_session(
    user_id="user-123",
    conversation_id="conv-456",   # optional — see below
)

# Step 2: run
response = await runner.run(
    agent=agent,
    input="Hello!",
    session_id=session_id,
    user_id="user-123",
)
```

> If you pass a `session_id` that was never created, the runner will not crash — but messages will silently fail to save and history will not load.

### How `session_id` is computed

`get_or_create_session()` computes a deterministic session ID based on what you pass:


| Arguments                     | Computed `session_id`             |
| ----------------------------- | --------------------------------- |
| explicit `session_id`         | used as-is                        |
| `conversation_id` + `user_id` | `c:{conversation_id}:u:{user_id}` |
| `user_id` only                | `u:{user_id}`                     |
| neither                       | random UUID                       |


**Effect of** `conversation_id` **:** 

- Without `conversation_id`: one session per user (`u:{user_id}`) — all conversations share the same history
- With `conversation_id`: one session per conversation (`c:{conversation_id}:u:{user_id}`) — each conversation has its own isolated history

Use `conversation_id` when you want to keep separate chat threads per user (e.g. multiple chat windows).

---

## 3. Workflow Agents

### Session saving

Every workflow agent (Sequential, Parallel, Loop, Reflection) calls `runner.run()` one or more times internally — one per sub-agent or iteration. By default, `runner.run()` auto-saves each turn to session history, which would result in noisy intermediate turns the user never saw.

To prevent this, every workflow agent must:

**1. Set** `suppress_session_log = True` **at the start of** `execute()`  ****

```python
async def execute(self, input_text, runner, context) -> AgentResponse:
    context.suppress_session_log = True  # blocks auto-saving for all sub-agent runs
    ...
    response = await runner.run(
        agent=agent,
        input=current_input,
        context=context,   # same context passed to every runner.run()
    )
```

The same `context` object is passed to every `runner.run()` call. Inside `run_finalizer.py`, each run checks `context.suppress_session_log` and skips saving if `True`.

**2. Call `save_turn()` once at the end**

```python
await runner.save_turn(
    session_id=context.session_id,
    user_message=input_text,           # what user originally sent
    assistant_message=final_output,    # what user actually sees
)
```

This saves exactly one clean turn — the original input and the final output — to session history.

> **If you build a custom workflow agent, you must follow the same pattern.** Forgetting `suppress_session_log = True` will cause every sub-agent turn to be saved to session history.

### Built-in workflow implementations

Built-in workflow agents are provided in `src/orchestrator/agent/workflow/`:

```
sequential.py  parallel.py  loop.py  reflection.py
planner.py     router.py    scatter.py  supervised.py  debate.py
```

You can refer to `playground/multi-agent-shop/workflows.py` and `playground/multi-agent-shop/agents.py` as usage examples.

### Build your own workflow instead of using built-ins

We recommend defining and building your own workflows using `BaseAgent` directly, rather than relying solely on the built-in workflow agents, because workflows are closely tied to your project's business logic — a custom agent gives you full control over the flow, session saving, and memory behaviour.

**Example: `ParallelCoordinatorAgent` in `playground/multi-agent-shop/workflows.py`**

Instead of using `ParallelAgent` directly, the pet shop defines a custom `BaseAgent` subclass that orchestrates the parallel search and synthesis steps manually:

```python
class ParallelCoordinatorAgent(BaseAgent):
    synthesiser: BaseAgent | None = None
    parallel: ParallelAgent | None = None

    async def execute(self, input_text, runner, context, llm_client=None) -> AgentResponse:
        context.suppress_session_log = True

        # Step 1: run parallel workers with a fresh stateless context (no history, no save)
        parallel_ctx = create_run_context(user_id=context.user_id, conversation_id=context.conversation_id)
        parallel_result = await self.parallel.execute(input_text, runner, parallel_ctx)

        # Step 2: synthesiser builds the user-facing reply using session history + memory
        # suppress_session_log=True blocks auto-save; context carries session_id so
        # message_builder loads history and memory normally.
        final = await runner.run(agent=self.synthesiser, input=synthesis_input, context=context)

        # Step 3: save one clean turn
        await runner.save_turn(session_id=context.session_id, user_message=input_text, assistant_message=final.content)
```

This gives the parallel workers (Parallel workflow based on several Base Agents) a clean stateless context (no Redis calls) while the synthesiser (based on a Base Agent) still loads session history and memory — something the built-in `ParallelAgent` cannot do out of the box.

---

## 4. Long-term Memory Management (mem0)

### Controlling what gets stored

`**store_memories**` — master on/off switch. If `False`, nothing goes to mem0.

`**extraction_prompt**` — tells mem0 what facts to extract from the conversation. By default mem0 decides. Override it to be specific:

```python
AgentMemoryConfig(
    store_memories=True,
    extraction_prompt=(
        "Only extract long-term facts about the user's pets, animal preferences, "
        "and dietary needs. Do NOT store transient actions like adding to cart or searches."
    ),
)
```

`**pre_store_filter**` — runs after mem0 stores facts. Facts not returned by the filter are **deleted from mem0**. Use it to remove PII or irrelevant facts that slipped through extraction:

```python
def my_filter(facts: list[str]) -> list[str]:
    return [f for f in facts if "credit card" not in f]

AgentMemoryConfig(
    store_memories=True,
    pre_store_filter=my_filter,
)
```

The full flow:

```
conversation → extraction_prompt extracts facts → stored in mem0
             → pre_store_filter runs → blocked facts deleted from mem0
```

### Memory management

Get `memory_client` from the container:

```python
from orchestrator.core.container import get_container
memory_client = get_container().memory_client
```

**View memories:**

```python
# get all memories for a user
memories = await memory_client.get_all(user_id="user-123")
for m in memories:
    print(m.memory, m.id)

# search by query
results = await memory_client.search("pet preferences", user_id="user-123")
```

**Delete memories:**

```python
# delete a specific memory
await memory_client.delete(memory_id="abc-123")

# delete ALL memories for a user
await memory_client.delete_all(user_id="user-123")
```

> Consider exposing memory management to your frontend — let users view and delete what the AI remembers about them. This is important for privacy compliance (GDPR "right to be forgotten") and user trust.

