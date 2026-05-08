# Session Module

Short-term, Redis-backed conversation history with TTL, message limits,
and multi-tenant key-prefix isolation. `AgentRunner` uses sessions
automatically when you pass a `session_id` — most app code never calls
`SessionClient` directly.

---

## 1 · Quick start

```python
from orchestrator.session import SessionClient
from orchestrator.llm.types import ChatMessage

client = SessionClient()                                     # uses env defaults
sid = await client.get_or_create_session(
    user_id="user-123", agent_id="support-agent",
)

await client.add_message(sid, ChatMessage(role="user", content="Hello"))
await client.add_message(sid, ChatMessage(role="assistant", content="Hi!"))

history: list[ChatMessage] = await client.get_conversation_history(sid)
```

> ⚠️ `add_message()` takes a `ChatMessage` object, **not** `role=...`,
> `content=...` keyword arguments. Earlier docs had this wrong.

---

## 2 · `SessionClient`

`from orchestrator.session import SessionClient`

```python
SessionClient(
    session_config: SessionConfig | None = None,
    memory_client: MemoryClient | None = None,
    provider: BaseSessionProvider | None = None,
    auto_initialize: bool = True,
)
```

The optional `memory_client` lets the session client double-write
messages to long-term memory — see Section 5.

### Properties

- `provider: BaseSessionProvider` — concrete provider (Redis by default)
- `config: SessionConfig`
- `memory_client: MemoryClient` — pulled from the global container
- `is_enabled: bool`

### Methods

All public methods are async and decorated with `@observe` for tracing.

| Method | Returns | Notes |
|---|---|---|
| `initialize()` | `bool` | Thread-safe, runs once. Most callers don't need this — `auto_initialize=True` covers it |
| `set_provider(provider)` | `None` | Swap providers at runtime |
| `get_or_create_session(session_id=None, user_id=None, agent_id=None)` | `str` | Returns the session id (creates if missing); maps to mem0's `run_id` |
| `add_message(session_id, message: ChatMessage, *, metadata=None, store_in_memory=True, extraction_prompt=None, pre_store_filter=None, on_stored=None)` | `None` | Append message; optionally cascade to mem0 with custom extraction prompt and PII filter |
| `get_conversation_history(session_id, limit=None)` | `list[ChatMessage]` | |
| `get_relevant_memories(session_id, query, limit=None)` | `list[Any]` | Semantic search over mem0 long-term memory using this session's scope |
| `clear_session(session_id)` | `bool` | Drop messages, keep metadata |
| `delete_session(session_id)` | `bool` | Drop everything |
| `get_session_metadata(session_id)` | `SessionMetadata \| None` | |
| `update_session_metadata(session_id, metadata)` | `bool` | Redis-provider-specific |

### Global helpers

```python
from orchestrator.session import (
    initialize_global_session_client, get_global_session_client,
)
from orchestrator.session.client import reset_global_session

initialize_global_session_client()                # bool
client = get_global_session_client()              # singleton, lazy
reset_global_session()                            # for tests
```

---

## 3 · `SessionConfig`

`from orchestrator.session import SessionConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"redis"` | Currently the only built-in provider |
| `enabled` | `bool` | `settings.session_enabled` | Master switch |
| `redis_host` | `str` | `settings.session_redis_host` | |
| `redis_port` | `int` | `settings.session_redis_port` | **Default `6380`**, not `6379` |
| `redis_password` | `str \| None` | `settings.session_redis_password` | |
| `redis_db` | `int` | `settings.session_redis_db` | |
| `redis_ssl` | `bool` | `settings.session_redis_ssl` | |
| `redis_max_connections` | `int` | `10` | Pool size |
| `ttl_seconds` | `int` | `settings.session_ttl_seconds` | Default 7 days |
| `max_messages` | `int` | `settings.session_max_messages` | Default 1000 |
| `key_prefix` | `str` | `settings.session_key_prefix` | Default `"orchestrator:session"` |
| `message_limit_strategy` | `Literal["error","sliding_window"]` | `"sliding_window"` | When `max_messages` is hit |
| `sliding_window_trim_count` | `int` | `100` | How many oldest messages to drop |

Methods:
- `is_configured() -> bool`
- `get_redis_url() -> str` — useful for plumbing third-party libs at the same instance

---

## 4 · Types

`from orchestrator.session import (
    Session, SessionMetadata, SessionMessage, generate_session_id,
)`

### `Session`
- `session_id: str`
- `metadata: SessionMetadata`
- `messages: list[ChatMessage]`

### `SessionMetadata`
- `session_id: str`
- `user_id: str | None`
- `agent_id: str | None`
- `created_at: datetime`
- `last_accessed_at: datetime`
- `message_count: int = 0`
- `custom: dict[str, Any]`

`to_dict()` and `from_dict(data)` for Redis (de)serialization.

### `SessionMessage`
- `message: ChatMessage`
- `timestamp: datetime`
- `metadata: dict[str, Any]` — `trace_id`/`span_id` are managed by `@observe`, no need to set them by hand

### `generate_session_id() -> str`
Returns a UUID-based session id; the framework calls this for you when
you don't provide one.

---

## 5 · Long-term memory cascade

When you call `add_message(..., store_in_memory=True)`, the session
client also writes to mem0 with the same scope (`run_id` = session id,
`user_id`, `agent_id`). Three optional hooks let you customize:

| Argument | Purpose |
|---|---|
| `extraction_prompt: str` | Custom prompt for mem0's fact extraction LLM |
| `pre_store_filter: Callable[[str], str]` | Sanitize the message before mem0 sees it (PII redaction, etc.) |
| `on_stored: Callable[[list[dict]], None]` | Callback after mem0 returns extracted memories |

Set `store_in_memory=False` to keep a session purely short-term and skip
the mem0 write — useful for ephemeral flows or when running without
Qdrant.

---

## 6 · Provider system

`from orchestrator.session import (
    BaseSessionProvider, register_provider, create_provider,
    get_provider_class, list_providers,
)`

`BaseSessionProvider` defines the abstract async API:
`get_or_create_session`, `add_message`, `get_messages`,
`get_session_metadata`, `clear_session`, `delete_session`, `close`. The
built-in `RedisSessionProvider` implements this with:

- Redis Lists for messages (chronological ordering)
- JSON-encoded metadata
- TTL via `EXPIRE`
- Sliding-window trim on overflow
- `(user_id, agent_id) -> session_id` lookup keys for
  `get_or_create_session`

`is_redis_available()` returns `False` if `redis` isn't installed
(unlikely — `redis` is a hard runtime dependency).

To swap in a custom provider:

```python
from orchestrator.session import register_provider, SessionClient, SessionConfig

class MyProvider(BaseSessionProvider):
    @property
    def provider_name(self): return "my"
    @property
    def is_initialized(self): return True
    # implement the async interface...

register_provider("my", MyProvider)
client = SessionClient(SessionConfig(provider="my"))
```

---

## 7 · Exceptions

`from orchestrator.session import (
    SessionError, SessionConfigurationError, SessionNotEnabledError,
    SessionConnectionError, SessionNotFoundError, SessionMessageLimitError,
)`

Constructors share a common shape `(message, session_id=None, original_error=None)`. `SessionMessageLimitError` adds `current_count` and `max_messages`.

---

## 8 · Common patterns

### Add a system event without a user message

```python
from orchestrator.llm.types import ChatMessage
await client.add_message(
    sid,
    ChatMessage(role="system", content="User upgraded to Pro tier."),
    store_in_memory=False,
)
```

### Resume an existing conversation

```python
sid = await client.get_or_create_session(user_id="u1", agent_id="support")
history = await client.get_conversation_history(sid, limit=20)
resp = await runner.run(agent, "Hi again", session_id=sid, user_id="u1")
# The runner reloads history from Redis automatically — you don't need
# to inject `history` into the input.
```

### Manual cleanup

```python
await client.delete_session(sid)
```

---

## 9 · Gotchas

- **`add_message(session_id, message=ChatMessage(...))`** — pass a
  `ChatMessage` object, not `role=` / `content=` kwargs.
- **Redis port is `6380`** in this kit (mapped from container `6379`)
  to avoid clashes with any other Redis on your machine. If you change
  it, update both `docker-compose.yml` *and* `.env`'s
  `SESSION_REDIS_PORT`.
- **`session_id == run_id`** in mem0. The framework standardizes on this
  — don't pass different values to memory and session APIs for the same
  conversation.
- **TTL defaults to 7 days**. After that, sessions vanish. Bump
  `SESSION_TTL_SECONDS` if you need longer retention.
- **Sliding-window trim drops 100 oldest messages by default** when
  `max_messages` is hit. If you'd rather error out, set
  `message_limit_strategy="error"` and catch `SessionMessageLimitError`.
