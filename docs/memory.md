# Memory Module

Long-term semantic memory backed by **mem0** (fact extraction) +
**Qdrant or Milvus** (vector search; default is Milvus). Memories are stored across sessions and
retrieved automatically by `AgentRunner` before each LLM call.

This doc covers:
- `MemoryClient` — the public async/sync API
- `MemoryConfig` — full configuration reference
- Memory **scopes** for multi-tenant isolation
- The richer `IntelligentMemoryClient` (importance scoring, decay,
  entity memory, user profiles)
- Provider system

---

## 1 · Quick start

```python
from orchestrator.memory import MemoryClient, MemoryConfig

client = MemoryClient(config=MemoryConfig())     # uses env defaults

# Add memories from a conversation (mem0 extracts facts via LLM)
await client.add(
    messages=[
        {"role": "user", "content": "I'm vegetarian"},
        {"role": "assistant", "content": "Got it — I'll suggest vegetarian options."},
    ],
    user_id="user-123",
)

# Retrieve relevant memories
result = await client.search("dietary preferences", user_id="user-123", limit=5)
for entry in result.results:
    print(entry.memory)                          # str
```

In an agent:

```python
from orchestrator.agent import BaseAgent, AgentMemoryConfig, MemoryScope

agent = BaseAgent(
    name="assistant",
    instructions="...",
    memory_config=AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,
        store_scope=MemoryScope.USER,
        search_limit=5,
    ),
)
```

When `AgentRunner.run(..., user_id="u1")` executes, the runner queries
mem0 for `user_id="u1"`, injects the top-5 results into the system
prompt as a "User profile" message, then after the turn writes any new
facts back.

---

## 2 · `MemoryClient`

`from orchestrator.memory import MemoryClient`

```python
MemoryClient(
    config: MemoryConfig | None = None,
    provider: BaseMemoryProvider | None = None,
    auto_initialize: bool = True,
)
```

### Properties

- `config: MemoryConfig` — current configuration
- `provider: BaseMemoryProvider | None` — concrete provider (defaults to mem0)
- `is_enabled: bool` — `True` only if enabled in config and provider initialized

### Async methods

| Method | Returns | Notes |
|---|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, custom_prompt=None, infer=True)` | `MemoryAddResult` | Run mem0 fact extraction over `messages` (string, list of strings, or list of `{role,content}` dicts) |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, filters=None)` | `MemorySearchResult` | Semantic search; `limit` defaults to `MemoryConfig.search_limit` |
| `get(memory_id)` | `MemoryEntry \| None` | |
| `get_all(*, user_id=None, agent_id=None, run_id=None, limit=None)` | `list[MemoryEntry]` | |
| `delete(memory_id)` | `bool` | |
| `delete_all(*, user_id=None, agent_id=None, run_id=None)` | `bool` | Wipes a scope |
| `update(memory_id, data, *, custom_prompt=None)` | `MemoryEntry` | Replace memory text |
| `history(memory_id)` | `list[dict]` | All versions of a memory |
| `reset()` | `bool` | **DESTRUCTIVE** — wipes the entire store |
| `close()` | `None` | Release vector store client |

Every async method has a `*_sync` counterpart (`add_sync`, `search_sync`,
…) that wraps the async method via `asyncio.to_thread`.

### Context manager

```python
async with MemoryClient() as client:
    await client.add("Note about user", user_id="u1")
```

### Global helpers

```python
from orchestrator.memory import (
    initialize_global_memory, get_global_memory_client,
)
from orchestrator.memory.client import reset_global_memory

initialize_global_memory()                  # uses env defaults; returns bool
client = get_global_memory_client()         # auto-inits on first call
reset_global_memory()                       # for tests
```

---

## 3 · `MemoryConfig`

`from orchestrator.memory import MemoryConfig`

A Pydantic model. Fields with their defaults sourced from `Settings`.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"mem0"` | Currently `mem0` is the only built-in provider |
| `enabled` | `bool` | `settings.memory_enabled` | Master switch |
| `vector_store_provider` | `Literal["qdrant", "milvus"]` | `"milvus"` | Vector store backend |
| `qdrant_host` | `str` | `settings.qdrant_host` | Used when `vector_store_provider="qdrant"` |
| `qdrant_port` | `int` | `settings.qdrant_port` | |
| `qdrant_api_key` | `str \| None` | `settings.qdrant_api_key` | For Qdrant Cloud |
| `qdrant_collection` | `str` | `settings.qdrant_collection` | Default `"orchestrator_memories"` |
| `milvus_host` | `str` | `settings.milvus_host` | Used when `vector_store_provider="milvus"` |
| `milvus_port` | `int` | `settings.milvus_port` | Default `19530` |
| `milvus_token` | `str \| None` | `settings.milvus_token` | For Zilliz Cloud |
| `milvus_collection` | `str` | `settings.milvus_collection` | Default `"orchestrator_memories"` |
| `memory_llm_model` | `str` | `settings.memory_llm_model` | LLM that does fact extraction |
| `memory_llm_temperature` | `float` | `settings.memory_llm_temperature` | |
| `embedder_provider` | `str` | `settings.embedder_provider` | `openai`, `azure_openai`, `huggingface`, `ollama`, `gemini`, `vertexai`, `cohere` |
| `embedder_model` | `str` | `settings.embedder_model` | e.g. `text-embedding-3-small` |
| `embedding_dims` | `int` | `settings.embedding_dims` | Must match the embedder output |
| `embedder_api_key` | `str \| None` | `settings.embedder_api_key` | Override env-supplied key |
| `embedder_api_base` | `str \| None` | `settings.embedder_api_base` | Self-hosted / Azure |
| `history_db_path` | `str` | `settings.memory_history_db_path` | SQLite history file |
| `memory_isolation` | `Literal["shared","user","agent","conversation"]` | `settings.memory_isolation` | Default scope |
| `search_limit` | `int` | `settings.memory_search_limit` | Default top-K |
| `reranker_enabled` | `bool` | `False` | Enable mem0 reranker |
| `custom_config` | `dict` | `{}` | Advanced mem0 overrides |

### Methods

- `is_configured() -> bool` — sanity check
- `to_mem0_config() -> dict` — produces the dict mem0's `Memory.from_config(...)` expects

---

## 4 · Scopes (multi-tenant isolation)

There are **two distinct things called `MemoryScope`** in the codebase
— pay attention to the import path.

### 4.1 The enum used by agents

`from orchestrator.agent import MemoryScope`

```python
class MemoryScope(str, Enum):
    SHARED = "shared"
    USER = "user"
    AGENT = "agent"
    CONVERSATION = "conversation"
```

This is what you pass to `AgentMemoryConfig(search_scope=MemoryScope.USER)`.

| Scope | Visibility |
|---|---|
| `SHARED` | All agents and users — global knowledge base |
| `USER` | One user across all agents — default |
| `AGENT` | One agent across all users |
| `CONVERSATION` | Single conversation only — ephemeral |

### 4.2 The dataclass used at the memory layer

`from orchestrator.memory import MemoryScope, MemoryIsolationLevel`

The memory module's `MemoryScope` is a **dataclass** carrying real
identifiers, not an enum.

```python
@dataclass
class MemoryScope:
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    custom_identifiers: dict[str, str] = {}
```

Factories:

```python
MemoryScope.shared()                    # agent_id="shared"
MemoryScope.user("user-123")
MemoryScope.agent("billing-agent")
MemoryScope.run("run_abc123")
MemoryScope.from_isolation_mode("user", user_id="u1")
MemoryScope.from_identifiers(user_id="u1", agent_id="billing")
```

Conversions:

- `to_identifiers() -> dict[str, str]` — only non-None values
- `to_dict() -> dict`
- `to_metadata() -> dict[str, Any]` — keys prefixed with `_`
- `validate_for_mode(mode) -> tuple[bool, str | None]` — does this scope satisfy the required identifier for a given isolation mode?
- `is_empty() -> bool`
- `get_primary_identifier(mode) -> dict[str, str]`

`MemoryIsolationLevel` is the underlying enum: `SHARED`, `USER`,
`AGENT`, `CONVERSATION` (string-valued).

### Custom scope registry

```python
from orchestrator.memory import (
    register_scope, get_scope_definition, list_scopes, is_scope_registered,
)

register_scope(
    name="organization",
    required_field="org_id",
    description="Organization-scoped memories",
    default_identifier={"org_id": "default"},
)
```

---

## 5 · Types

`from orchestrator.memory import (
    MemoryEntry, MemorySearchResult, MemoryAddResult, MemoryMetadata, MemoryFilter,
)`

### `MemoryEntry`
- `id: str`
- `memory: str` — the actual fact text
- `hash: str | None`
- `user_id`, `agent_id`, `run_id` — scope identifiers
- `metadata: dict`
- `created_at`, `updated_at: datetime | None`
- `score: float | None` — only populated by `search()`

### `MemorySearchResult`
- `results: list[MemoryEntry]`
- `query: str`
- `limit: int`
- `total_results: int | None`
- `get_memory_strings() -> list[str]`
- `get_top_k(k) -> list[MemoryEntry]`

### `MemoryAddResult`
- `message: str`
- `results: list[dict]` — extracted memories, mem0-shaped
- `relations: list[dict]` — graph edges between memories

### `MemoryMetadata`
Free-form structured metadata: `category`, `tags`, `source`,
`confidence`, `created_at`, `updated_at`, `custom`.

### `MemoryFilter`
For `search(filters=...)`: `user_id`, `agent_id`, `run_id`, `category`,
`tags`, `metadata`. Use `to_mem0_filter()` to translate.

---

## 6 · Intelligent memory layer

`from orchestrator.memory import IntelligentMemoryClient, IntelligenceConfig`

Adds four behaviours on top of mem0:

1. **Importance scoring** — every memory gets an LLM-assigned 0-1 score at write time.
2. **Time decay** — search re-ranks blending semantic similarity, importance, and recency.
3. **Entity memory** — extracts named entities (people, orgs, products) and stores them as tagged memories you can query separately.
4. **User profiles** — maintains a structured profile of each user
   (preferences, employer, expertise, topics) across sessions.

### Config

```python
IntelligenceConfig(
    enable_entity_memory=True,
    enable_user_profiles=True,
    enable_scoring=True,
    enable_decay=True,
    semantic_weight=0.6,
    importance_weight=0.3,
    decay_weight=0.1,
    intelligence_model=None,         # defaults to MemoryConfig.memory_llm_model
    prune_threshold=0.15,
)
```

### Extra methods (over `MemoryClient`)

- `await client.search_entities(query, *, user_id, limit=5)` — entity-only search; `user_id` is **required**
- `await client.get_user_profile(user_id)` — structured profile dict (or `None`)
- `await client.prune(*, user_id, threshold=None) -> int` — delete memories below `importance + decay` threshold; `user_id` is **required**

### Use it

```python
client = IntelligentMemoryClient(
    config=MemoryConfig(),
    intelligence_config=IntelligenceConfig(),
)
```

You can wire it into the container in place of the default `MemoryClient`:

```python
from orchestrator.core.container import get_container
get_container().set_memory_client(IntelligentMemoryClient())
```

---

## 7 · Provider system

`from orchestrator.memory import (
    BaseMemoryProvider, register_provider, get_provider_class,
    create_provider, list_providers,
)`

```python
print(list_providers())              # ["mem0"]
provider = create_provider("mem0", MemoryConfig())
```

Register custom providers (e.g., Pinecone, in-memory test stub):

```python
class MyProvider(BaseMemoryProvider):
    @property
    def provider_name(self): return "my"
    @property
    def is_initialized(self): return True
    # implement async + sync versions of add/search/get/get_all/delete/...

register_provider("my", MyProvider)
client = MemoryClient(config=MemoryConfig(provider="my"))
```

The mem0 provider uses `asyncio.to_thread()` for sync mem0 ops — your
custom provider can be natively async.

---

## 8 · Exceptions

`from orchestrator.memory import (
    MemoryError, MemoryConfigurationError, MemoryNotEnabledError,
    MemoryConnectionError, MemorySearchError, MemoryAddError,
    MemoryDeleteError, MemoryUpdateError, MemoryIdentifierError,
)`

All inherit from `OrchestratorError`. The most common one you'll see in
local development is `MemoryConfigurationError: Failed to initialize
mem0: Missing credentials` — that's the OpenAI embedder demanding
`OPENAI_API_KEY`.

---

## 9 · Common patterns

### Disable long-term memory entirely

For a fully offline test or a non-AI use case:

```bash
# .env
MEMORY_ENABLED=false
```

```python
from orchestrator.core.container import Container, ContainerConfig
container = Container(ContainerConfig(enable_memory=False))
runner = AgentRunner(container=container)
```

### Cross-agent shared knowledge

```python
shared_agent = BaseAgent(
    name="globals",
    instructions="...",
    memory_config=AgentMemoryConfig(
        search_scope=MemoryScope.SHARED,
        store_scope=MemoryScope.SHARED,
    ),
)
```

### PII-aware storage

```python
def redact_pii(text: str) -> str:
    return re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN]", text)

agent = BaseAgent(
    name="assistant",
    memory_config=AgentMemoryConfig(
        store_memories=True,
        pre_store_filter=redact_pii,
        on_stored=lambda items: log.info("stored %d", len(items)),
    ),
)
```

---

## 10 · Gotchas

- **`OPENAI_API_KEY` is required at startup** when memory is enabled,
  because the default mem0 embedder is OpenAI. Either provide a key,
  switch `EMBEDDER_PROVIDER` (e.g. `huggingface` for local models), or
  set `MEMORY_ENABLED=false`.
- **`embedding_dims` must match the embedder's output**. `text-embedding-3-small` → 1536; `text-embedding-3-large` → 3072.
  Mismatch → Qdrant insert errors.
- **Two `MemoryScope` types** exist with the same name (the agent enum
  vs the memory dataclass). Make sure your import path matches the
  expected type at the call site.
- **Wiping memory**: `client.reset()` is destructive — there is no
  per-collection reset, it nukes everything. For per-user wipes use
  `delete_all(user_id=...)`.
- **`add()` does fact extraction** by calling the configured LLM —
  expect a noticeable latency hit on chatty agents. Set `infer=False` to
  store raw text without extraction.
