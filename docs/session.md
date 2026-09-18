# Session Module

Short-term, Redis-backed conversation history with TTL, message limits,
and multi-tenant key-prefix isolation. `AgentRunner` uses sessions
automatically when you pass a `session_id` — most app code never calls
`SessionClient` directly.

Persistence is initialized **lazily** (no connection until the first session
op) and degrades cleanly to a non-durable in-memory store if Redis is
unreachable — see [§10](#10--lazy-initialization--in-memory-fallback).

---

## 1 · Quick start

```python
from continuum.session import SessionClient
from continuum.llm.types import ChatMessage

from continuum.session import bind_principal

client = SessionClient()                                     # uses env defaults

# Bind the caller's identity once, where your app has just authenticated the
# request. Everything below inherits it — see §11.
with bind_principal("user-123"):
    sid = await client.get_or_create_session(
        user_id="user-123", conversation_id="conv-456",
    )

    await client.add_message(sid, ChatMessage(role="user", content="Hello"))
    await client.add_message(sid, ChatMessage(role="assistant", content="Hi!"))

    history: list[ChatMessage] = await client.get_conversation_history(sid)
```

> ⚠️ `add_message()` takes a `ChatMessage` object, **not** `role=...`,
> `content=...` keyword arguments. Earlier docs had this wrong.

> ⚠️ Without `bind_principal`, every call above raises
> `SessionOwnershipError`: a session that records an owner is not readable or
> writable by a caller who has not said who they are. Sessions created with no
> `user_id` have no owner and need no principal. See §11.

---

## 2 · `SessionClient`

`from continuum.session import SessionClient`

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
- `persistence_degraded: bool` — `True` once the client has fallen back from
  Redis to the in-memory store at runtime (see §10). Stays `False` when
  `"memory"` was chosen on purpose — it flags an *unplanned* loss of durability,
  not in-memory use itself.

### Methods

All public methods are async and decorated with `@observe` for tracing.

| Method | Returns | Notes |
|---|---|---|
| `initialize()` | `bool` | Thread-safe, runs once. Most callers don't need this — `auto_initialize=True` covers it |
| `set_provider(provider)` | `None` | Swap providers at runtime |
| `get_or_create_session(session_id=None, user_id=None, conversation_id=None)` | `str` | Returns the session id (creates if missing); maps to mem0's `run_id` |
| `add_message(session_id, message: ChatMessage, *, metadata=None, store_in_memory=True, extraction_prompt=None, pre_store_filter=None, on_stored=None)` | `None` | 🔒 Append message; optionally cascade to mem0 with custom extraction prompt and PII filter |
| `get_conversation_history(session_id, limit=None)` | `list[ChatMessage]` | 🔒 |
| `get_relevant_memories(session_id, query, limit=None)` | `list[Any]` | 🔒 Semantic search over mem0 long-term memory using this session's scope |
| `clear_session(session_id)` | `bool` | 🔒 Drop messages, keep metadata |
| `delete_session(session_id)` | `bool` | 🔒 Drop everything |
| `get_session_metadata(session_id)` | `SessionMetadata \| None` | 🔒 |
| `update_session_metadata(session_id, metadata)` | `bool` | 🔒 Redis-provider-specific |

🔒 = ownership-checked. When the stored session records an owner, the caller
must have bound a matching principal or the call raises
`SessionOwnershipError`. See §11.

### Global helpers

```python
from continuum.session import (
    initialize_global_session_client, get_global_session_client,
)
from continuum.session.client import reset_global_session

initialize_global_session_client()                # bool
client = get_global_session_client()              # singleton, lazy
reset_global_session()                            # for tests
```

---

## 3 · `SessionConfig`

`from continuum.session import SessionConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"redis"` | Built-in: `"redis"` (durable) and `"memory"` (in-process, non-durable) |
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
| `fallback_mode` | `Literal["degrade","fail"]` | `settings.session_fallback_mode` (`"degrade"`) | What to do when Redis is unreachable — see §10 |
| `session_ownership` | `Literal["open","audit","enforce"]` | `settings.session_ownership` (`"enforce"`) | How to react when a caller touches a session owned by someone else — see §11 |
| `require_principal` | `bool` | `settings.session_require_principal` (`False`) | Treat "no principal bound" as an ownership problem — see §11 |
| `hash_session_ids` | `bool` | `settings.session_hash_ids` (`False`) | Derive session ids as an HMAC instead of plaintext — see §11 |
| `session_id_secret` | `str \| None` | `settings.session_id_secret` | HMAC key; required when `hash_session_ids` is on |

Methods:
- `is_configured() -> bool`
- `get_redis_url() -> str` — useful for plumbing third-party libs at the same instance

---

## 4 · Types

`from continuum.session import (
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

`from continuum.session import (
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

The second built-in is `MemorySessionProvider` (`provider="memory"`): an
in-process store that mirrors the same semantics (deterministic session ids,
sliding-window trim, metadata) but holds everything in a plain dict. It is
**non-durable** — state is lost when the process exits — and it never raises a
connection error. It serves two roles: an explicit zero-dependency provider for
ephemeral or test flows, and the automatic fallback target when Redis is
unreachable (see §10).

To swap in a custom provider:

```python
from continuum.session import register_provider, SessionClient, SessionConfig

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

`from continuum.session import (
    SessionError, SessionConfigurationError, SessionNotEnabledError,
    SessionConnectionError, SessionNotFoundError, SessionMessageLimitError,
    SessionOwnershipError,
)`

`SessionOwnershipError` is raised when the caller does not own the session they
named — see §11. It is the one exception `AgentRunner` does **not** swallow: a
refused read or write reaches your code instead of becoming a log line.

Constructors share a common shape `(message, session_id=None, original_error=None)`. `SessionMessageLimitError` adds `current_count` and `max_messages`.

---

## 8 · Common patterns

### Add a system event without a user message

```python
from continuum.llm.types import ChatMessage
await client.add_message(
    sid,
    ChatMessage(role="system", content="User upgraded to Pro tier."),
    store_in_memory=False,
)
```

### Resume an existing conversation

```python
with bind_principal("u1"):
    sid = await client.get_or_create_session(user_id="u1", conversation_id="c1")
    history = await client.get_conversation_history(sid, limit=20)
    resp = await runner.run(agent, "Hi again", session_id=sid, user_id="u1")
# The runner reloads history from Redis automatically — you don't need
# to inject `history` into the input.
```

### Manual cleanup

```python
with bind_principal("u1"):
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

---

## 10 · Lazy initialization & in-memory fallback

The session client **never connects to Redis at construction time**. The
provider is resolved lazily on the first session operation. Two consequences:

- **Disabled or unconfigured persistence costs nothing.** If sessions are off,
  or no Redis host is set, no connection is attempted and no warnings are
  logged — the feature is silent when it isn't in use.
- **Startup never blocks** on a slow or absent Redis.

### What happens when Redis is unreachable

There are two moments persistence can fail, both governed by `fallback_mode`:

1. **At first use** — the lazy connect/ping fails (Redis down or misconfigured).
2. **Mid-session** — Redis was reachable, then drops out partway through a
   conversation. Any session op that raises `SessionConnectionError` triggers
   the same handling.

| `fallback_mode` | Behavior on failure |
|---|---|
| `"degrade"` *(default)* | Swap to `MemorySessionProvider` and keep serving. Durability is lost, but the request succeeds. The client sets `persistence_degraded = True` and emits the `session_persistence_degraded` gauge. **One** warning is logged for the transition — not one error per request. |
| `"fail"` | Raise `SessionConnectionError` instead of silently degrading. Use this when a session write *must* be durable and you'd rather surface the outage. |

Set it via the `SESSION_FALLBACK_MODE` environment variable (`degrade` | `fail`)
or per-client with `SessionConfig(fallback_mode=...)`.

### Observing the degraded state

Degradation is intentionally quiet on the request path, so it is surfaced for
operators two ways (see [observability.md](observability.md)):

- **Health check** `session_persistence` flips from `healthy` to `degraded`.
- **Metric** `session_persistence_degraded` is a gauge: `1.0` while degraded,
  `0.0` when healthy.

```python
client = SessionClient(session_config=SessionConfig(fallback_mode="degrade"))
sid = await client.get_or_create_session(user_id="u1")  # Redis down → in-memory
assert client.persistence_degraded is True               # durability lost, still serving
```

> **`persistence_degraded` vs. `provider="memory"`** — choosing the in-memory
> provider on purpose is *not* a degradation, so the flag stays `False`. The
> flag means specifically "we wanted Redis and couldn't get it."

---

## 11 · Session ownership

A session id names storage. It is not proof of who is calling.

That distinction used to be theoretical, because ids were built in plaintext
from the identifiers they scope — `u:{user_id}`, or
`c:{conversation_id}:u:{user_id}`. Anyone who knew a user id could *construct*
that user's session id and read their history. No leak required.

Two things changed. Ids can now be derived through an HMAC, so they cannot be
constructed; and a session that records an owner is only reachable by a caller
who has said who they are.

### Binding the caller

The framework never derives an identity for you, and deliberately offers no way
to set one from request data: from inside `run()`, a validated user id and a
string someone typed are both just `str`. Only your application knows which of
its values came out of a credential it checked.

So you bind it, once, at that boundary:

```python
from continuum.session import bind_principal

user = verify_jwt(request.headers["Authorization"])   # your own auth
with bind_principal(user.id):
    await runner.run(agent, message, session_id=sid, user_id=user.id)
```

Everything below inherits it — `AgentRunner`, `LLMClient`, and any direct
`SessionClient` call — without an argument being threaded through.

> ⚠️ Bind an id you **verified**, not one the caller supplied. Binding
> `request.json["user_id"]` compares an attacker-controlled value against itself
> and protects nothing. For the same reason, do not reuse
> `continuum.core.context`'s `user_id`: that one is populated by tracing from a
> caller-supplied argument.

Streaming needs the binding *inside* the generator — a `with` around the call
exits before the response body is produced:

```python
async def stream():
    with bind_principal(user.id):
        async for chunk in runner.run_stream(agent, message, session_id=sid):
            yield chunk
```

### What is checked

| stored owner | principal bound | result |
|---|---|---|
| none | anything | allowed — nothing to protect |
| set | matches | allowed |
| set | differs | `SessionOwnershipError` |
| set | none bound | allowed by default; `SessionOwnershipError` when `require_principal=True` |

### What the defaults do not stop

`require_principal` ships **off**, so a caller who names nobody is let through —
the compatibility choice, since every application written before
`bind_principal` existed is in exactly that position.

`session_ownership="enforce"` does not cover this. It refuses a caller who gives
the *wrong* identity, not one who gives *none*. And with `hash_session_ids` also
off, the id is derived in plaintext from the user id it scopes, so anyone who
knows a user id can construct their session id and present it while naming
nobody.

A multi-tenant deployment needs one of these two, and either is sufficient:

```bash
SESSION_REQUIRE_PRINCIPAL=true   # the caller must say who they are
SESSION_HASH_IDS=true            # the id can no longer be derived
```


Sessions created without a `user_id` have no owner, so anonymous and
single-user deployments are unaffected.

### Settings

| Variable | Value | What it does |
|---|---|---|
| `SESSION_OWNERSHIP` | `open` | Log the problem quietly, allow the call |
| | `audit` | Log a warning and emit a metric, allow the call — measure before enforcing |
| | `enforce` *(default)* | Raise `SessionOwnershipError` |
| `SESSION_REQUIRE_PRINCIPAL` | `true` | A caller who names nobody is refused |
| | `false` *(default)* | Holding the session id is enough |
| `SESSION_HASH_IDS` | `true` | Ids are `s_<hex>` — not derivable, and no user ids in Redis keys or logs |
| | `false` *(default)* | Ids are `u:{user_id}` — readable, and constructible by anyone who knows a user id |
| `SESSION_ID_SECRET` | a random string | HMAC key. Required when hashing is on; short, memorable and placeholder values are refused at startup |

The defaults are the compatible ones, so that upgrading an existing application
does not break it. For a multi-tenant deployment, set all three:

```bash
SESSION_OWNERSHIP=enforce           # refuse a caller whose identity does not match
SESSION_REQUIRE_PRINCIPAL=true      # and one who gives no identity at all
SESSION_HASH_IDS=true               # ids become s_<hex>; re-derives existing keys

# generate with: openssl rand -hex 32
SESSION_ID_SECRET=…
```

### Upgrading an existing application

`require_principal` ships off, so an application that does not yet call
`bind_principal` keeps working on upgrade. What it will hit is the *mismatch*
case: a caller that binds an identity not matching the stored owner is refused.

To see that coming rather than discover it in production:

```bash
SESSION_OWNERSHIP=audit
```

`audit` allows the call and reports it — a log line plus a
`session_ownership_*` metric — so you can measure what enforcement would break
before it breaks anything. Adopt `bind_principal` at your auth boundary, watch
the metric fall to zero, then return to `enforce`.

Then close the omission path, which the defaults leave open:

```bash
SESSION_REQUIRE_PRINCIPAL=true   # or SESSION_HASH_IDS=true — either one
```

### Hashing the ids

Off by default, because turning it on changes every derived key.

```bash
SESSION_HASH_IDS=true
SESSION_ID_SECRET=$(openssl rand -hex 32)
```

Determinism is preserved — the same identifiers still resolve to the same
session — but the id can no longer be computed from a user id, and user
identifiers stop appearing in Redis keys, log lines and traces.

The secret must be **identical in every process and stable across restarts**.
It is a derivation parameter, not a per-process random: a different value per
worker splits one user's history across workers, and changing it re-derives
every id. Existing plaintext sessions migrate to their new keys on first
access.

It must also be **random**. A guessable value defeats the feature entirely:
every user of your system holds one matched pair — their own user id and their
own session id — and can brute-force the secret offline, with no rate limit and
no logs. Weak, short and placeholder values are refused at startup;
`CONTINUUM_ALLOW_INSECURE=1` downgrades that to a warning for local work.

### What this does not cover

Long-term memory (mem0) scopes by `user_id`, not by session, so it is unaffected
by ownership checks and by changes to the id scheme. Changing
`SESSION_ID_SECRET` costs you in-flight conversation history, never stored
memories.
