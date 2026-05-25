---
name: continuum-memory
description: Configure and use Continuum's two-tier memory system — mem0+Qdrant/Milvus for long-term facts, Redis for short-term sessions, with multi-tenant scopes (USER / AGENT / SHARED / RUN / CONVERSATION). Invoke when the user asks about "remember", "user preferences", "long-term memory", "vector search over memories", "multi-tenant isolation", "PII filtering on memory writes", "Milvus", "Qdrant", or memory-related errors.
---

# Continuum Memory Skill

Authoritative sources: [`docs/memory.md`](../../../docs/memory.md) and
[`docs/session.md`](../../../docs/session.md).

---

## Two layers

| Layer | Class | Backend | Purpose |
|---|---|---|---|
| Short-term | `SessionClient` | Redis (port 6380) | Conversation history this session |
| Long-term | `MemoryClient` | mem0 + Qdrant or Milvus | Facts extracted across sessions |

`AgentRunner` uses both automatically when `user_id` and `session_id`
are passed.

### Vector store selection

Default is **Milvus**. Switch via env var:

```env
VECTOR_STORE_PROVIDER=milvus      # default — Milvus (port 19530)
VECTOR_STORE_PROVIDER=qdrant      # Qdrant (port 6333)
```

Milvus config:
```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=                     # for Zilliz Cloud
MILVUS_COLLECTION=orchestrator_memories
```

Qdrant config:
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=                   # for Qdrant Cloud
QDRANT_COLLECTION=orchestrator_memories
```

---

## Quick agent setup

```python
from orchestrator.agent import BaseAgent
from orchestrator.agent.config import AgentMemoryConfig
from orchestrator.agent.types import MemoryScope

agent = BaseAgent(
    name="assistant",
    instructions="...",
    memory_config=AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,        # ENUM, not the dataclass
        store_scope=MemoryScope.USER,
        search_limit=5,
    ),
)
resp = await runner.run(agent, "...", user_id="u1", session_id="s1")
```

---

## Direct memory access

```python
from orchestrator.memory import MemoryClient

client = MemoryClient()                       # uses env defaults

# add (fact extraction via LLM)
await client.add(
    messages=[{"role": "user", "content": "I'm vegetarian"}],
    user_id="u1",
)

# semantic search
result = await client.search("dietary preferences", user_id="u1", limit=5)
for entry in result.results:
    print(entry.memory)

# CRUD
entry = await client.get(memory_id)
all_entries = await client.get_all(user_id="u1")
await client.update(memory_id, "Updated text")
await client.delete(memory_id)
await client.delete_all(user_id="u1")         # wipe a user's memories
```

Every async method has a `*_sync` counterpart.

---

## Scopes (the two `MemoryScope` types — read carefully)

### Agent-side enum (use here)

```python
from orchestrator.agent.types import MemoryScope     # str-Enum
MemoryScope.SHARED / USER / AGENT / RUN / CONVERSATION
# Pass to AgentMemoryConfig
```

### Memory-side dataclass (different!)

```python
from orchestrator.memory.scopes import MemoryScope   # dataclass
MemoryScope.user("u1")
MemoryScope.shared()
MemoryScope.agent("billing")
MemoryScope.run("run_abc")
# Used internally; rarely passed by user code
```

| Scope | Visible to |
|---|---|
| `SHARED` | All agents, all users |
| `USER` | One user, all agents (default) |
| `AGENT` | One agent, all users |
| `RUN` | One run only — ephemeral |
| `CONVERSATION` | One session only — scoped to session_id |

---

## Sessions

```python
from orchestrator.session import SessionClient
from orchestrator.llm.types import ChatMessage

client = SessionClient()
sid = await client.get_or_create_session(user_id="u1", agent_id="support")
await client.add_message(sid, ChatMessage(role="user", content="Hi"))   # NOT role= kwarg
history = await client.get_conversation_history(sid)
```

Sessions cascade to long-term memory by default
(`store_in_memory=True`); set `False` for ephemeral chats.

---

## PII / extraction hooks

```python
agent = BaseAgent(
    name="assistant",
    memory_config=AgentMemoryConfig(
        store_memories=True,
        pre_store_filter=lambda text: redact(text),       # sanitize before mem0
        on_stored=lambda items: log.info(f"stored {len(items)} memories"),
        extraction_prompt="Extract preferences and facts only.",
    ),
)
```

---

## IntelligentMemoryClient (richer)

Adds importance scoring, time decay, entity memory, and user profiles.

```python
from orchestrator.memory import IntelligentMemoryClient, IntelligenceConfig

client = IntelligentMemoryClient(
    intelligence_config=IntelligenceConfig(
        enable_entity_memory=True,
        enable_user_profiles=True,
        enable_scoring=True,
        enable_decay=True,
        prune_threshold=0.15,
    ),
)
profile = await client.get_user_profile("u1")
entities = await client.search_entities("Acme Corp", user_id="u1", limit=5)
removed = await client.prune(user_id="u1", threshold=0.15)
```

Wire into the container:

```python
from orchestrator.core.container import get_container
get_container().set_memory_client(IntelligentMemoryClient())
```

---

## Disable memory entirely

```env
MEMORY_ENABLED=false
```

```python
from orchestrator.core.container import Container, ContainerConfig
container = Container(ContainerConfig(enable_memory=False))
```

---

## Don't

- Don't pass `role=` / `content=` to `SessionClient.add_message` — use
  `ChatMessage(...)`.
- Don't expect memory to work without `OPENAI_API_KEY` set — the
  default mem0 embedder is OpenAI's. Either set the key, change
  `EMBEDDER_PROVIDER`, or disable memory.
- Don't mix the two `MemoryScope` types — the **enum** goes into
  `AgentMemoryConfig`; the **dataclass** is internal.
- Don't use `client.reset()` casually — it wipes the entire vector
  store. Use `delete_all(user_id=...)` for per-user resets.
