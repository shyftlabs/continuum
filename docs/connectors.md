# Connectors Module

A uniform, pluggable layer for connecting to external services — Redis, the
vector store (Milvus/Qdrant), Temporal, and Langfuse. Every connector exposes
the **same interface** (enabled / configured / mode / connect / aping /
describe) and registers in a shared registry, so connections are configured and
probed consistently — via API keys, local Docker, or custom hosts — and a new
service is one file plus one registration line.

Most application code never touches connectors directly: the session, memory,
temporal, and observability layers consume them internally. You reach for this
module to **inspect** how each service is configured, **probe** liveness, or
**add** a new service.

---

## 1 · Quick start

```python
from continuum.connectors import get_connector, health_check_all, list_connectors

list_connectors()                 # ['langfuse', 'redis', 'temporal', 'vector_store']

redis = get_connector("redis")
redis.describe()                  # {'name': 'redis', 'enabled': True, 'mode': 'local_docker', 'host': ...}
await redis.aping()               # True / False — never raises

report = await health_check_all() # probe every enabled connector at once
```

From inside an app you can also reach the registry through the container, which
registers the built-ins on first access:

```python
from continuum.core.container import get_container

connectors = get_container().connectors   # dict[str, BaseConnector]
```

---

## 2 · Connection modes

Every connector reports a `ConnectionMode`, **inferred from configuration** — no
manual flag needed:

| Mode | When | Inferred from |
|---|---|---|
| `LOCAL_DOCKER` | Default host/ports (e.g. `continuum up`) | Host is `localhost` / `127.0.0.1` / the compose service name |
| `CLOUD` | Managed / API-key endpoint | An API key / token is set, or TLS is on |
| `CUSTOM` | An explicit non-default host with no cloud credential | Host is set but matches neither of the above |
| `DISABLED` | Turned off | The service's `enabled` flag is `False` |

`from continuum.connectors import ConnectionMode`

This is what makes "connect via API keys, local docker, etc." work: you set the
relevant env vars and the connector picks the right mode. For example, Redis is
`LOCAL_DOCKER` against `localhost`, but flips to `CLOUD` once `SESSION_REDIS_SSL`
is on; Temporal flips to `CLOUD` when `TEMPORAL_TLS` or `TEMPORAL_API_KEY` is set.

---

## 3 · The `BaseConnector` interface

`from continuum.connectors import BaseConnector`

```python
class BaseConnector[T](ABC):
    name: str

    @property
    def is_enabled(self) -> bool: ...          # turned on by config
    def is_configured(self) -> bool: ...        # minimum settings to connect are present
    @property
    def mode(self) -> ConnectionMode: ...       # inferred connection mode
    async def connect(self) -> T: ...           # open & return a client; may raise

    async def aping(self) -> bool: ...          # liveness probe; never raises (default: try connect())
    def describe(self) -> dict[str, Any]: ...   # diagnostics, secrets masked
```

There are two flavors, both behind this one interface:

- **We own the client** — `connect()` returns the live client and the connector
  *is* the connection. Redis, Temporal, and Langfuse work this way.
- **A library owns the client** — e.g. mem0 builds its own vector-store client
  from a config dict. Here `connect()` returns a standalone *probe* client used
  only for health checks, and the connector additionally exposes a
  config-producing method the library consumes (`VectorStoreConnector.to_mem0_block()`).
  The connector stays the single source of truth for connection params and mode.

`aping()` and `describe()` have defaults; `describe()` is overridden by each
built-in to add masked, service-specific fields.

---

## 4 · Built-in connectors

| Name | Owns client? | Modes hinge on | Key `describe()` fields |
|---|---|---|---|
| `redis` | yes | `SESSION_REDIS_SSL` → cloud; localhost → local-docker | `host`, `port`, `db`, `ssl`, `password` (masked) |
| `vector_store` | no (mem0 owns it) | API key/token → cloud; localhost → local-docker | `provider` (`milvus`/`qdrant`), `host`, `port`, `collection`, `token`/`api_key` (masked) |
| `temporal` | yes | `TEMPORAL_TLS` / `TEMPORAL_API_KEY` → cloud | `host`, `namespace`, `tls`, `api_key` (masked) |
| `langfuse` | yes | local host hint → local-docker; else cloud | `host`, `public_key` / `secret_key` (masked) |

### `redis`
The single place that builds the async Redis client (a `BlockingConnectionPool`
with socket + pool-acquire timeouts). `aping()` issues a real `PING`. Backed by
`SessionConfig`.

### `vector_store`
Wraps the Milvus (default) or Qdrant configuration from `MemoryConfig`.
`to_mem0_block()` is the **single source of truth** for the mem0 `vector_store`
config block that `MemoryConfig.to_mem0_config()` consumes — so connection
params live in exactly one place. `connect()` returns a standalone probe client
(not the one mem0 uses); `aping()` lists collections.

### `temporal`
Opens a Temporal client with optional TLS and API key (Temporal Cloud).
`connect()` accepts optional `host` / `namespace` overrides. Backed by
`TemporalConfig`.

### `langfuse`
Brings observability under the same interface. Authenticates with a
public/secret key pair; `aping()` runs Langfuse's `auth_check()` in a worker
thread (the Langfuse client is synchronous).

> **Why no `llm` connector?** LLM providers are intentionally *not* modeled as a
> connector. They are a per-request **router** (provider chosen by model prefix,
> optional Smart Gateway, fallback chains), not a persistent connection — there
> is no single client to own. They keep their existing routing layer.

---

## 5 · The registry

`from continuum.connectors import (
    register_connector, get_connector, list_connectors, all_connectors,
    unregister_connector, register_default_connectors, health_check_all,
)`

| Function | Returns | Notes |
|---|---|---|
| `register_connector(name, connector, *, replace=False)` | `None` | Raises `ValueError` on duplicate unless `replace=True` |
| `get_connector(name)` | `BaseConnector` | Raises `KeyError` (lists available names) if absent |
| `list_connectors()` | `list[str]` | Sorted names |
| `all_connectors()` | `dict[str, BaseConnector]` | Snapshot |
| `register_default_connectors(*, replace=True)` | `None` | Registers the four built-ins; idempotent |
| `unregister_connector(name)` | `None` | No error if absent (for tests) |
| `health_check_all()` | `dict[str, dict]` | Probes every **enabled** connector; disabled ones are reported as `disabled` and never probed |

`health_check_all()` output per connector:

```python
{
  "redis":        {"status": "healthy",  "name": "redis", "mode": "local_docker", "host": ...},
  "temporal":     {"status": "unhealthy", ...},
  "vector_store": {"status": "disabled",  "mode": "disabled"},
}
```

A disabled service costs **zero** connection attempts — it short-circuits before
`aping()`.

---

## 6 · Adding a new service

The cost is always **config first, then one connector file, then one
registration line** — it does not grow as more connectors are added. Worked
example: an S3 connector.

### Step 1 — Add the settings (the fields)

Put the connection fields on the global `Settings` (`config.py`) so they are
env-configurable:

```python
# src/continuum/config.py → class Settings
s3_enabled: bool = False
s3_endpoint: str = "http://localhost:9000"   # local docker (minio)
s3_access_key: str | None = None              # api key → cloud
s3_secret_key: str | None = None
s3_bucket: str = "continuum"
```

That's the minimum. If the service has lots of structured config you can add a
small `S3Config(BaseModel)` like `SessionConfig` — but it's optional; the
connector can read `settings` directly.

### Step 2 — Write the connector

Create `connectors/<service>.py` with a `BaseConnector` subclass. Set `name` and
implement `is_enabled` / `is_configured` / `mode` / `connect`; override `aping` /
`describe` when a cheaper probe or richer (masked) diagnostics exist.

```python
# src/continuum/connectors/s3.py
from typing import Any

from continuum.config import settings
from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.utils.secrets import mask_value


class S3Connector(BaseConnector[Any]):
    name = "s3"

    @property
    def is_enabled(self) -> bool:
        return settings.s3_enabled

    def is_configured(self) -> bool:
        return bool(settings.s3_endpoint)

    @property
    def mode(self) -> ConnectionMode:
        if not settings.s3_enabled:
            return ConnectionMode.DISABLED
        if settings.s3_access_key:               # api key → cloud
            return ConnectionMode.CLOUD
        if "localhost" in settings.s3_endpoint:  # local docker (minio)
            return ConnectionMode.LOCAL_DOCKER
        return ConnectionMode.CUSTOM

    async def connect(self) -> Any:
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )

    async def aping(self) -> bool:
        try:
            (await self.connect()).list_buckets()
            return True
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            endpoint=settings.s3_endpoint,
            access_key=mask_value(settings.s3_access_key) if settings.s3_access_key else None,
        )
        return d
```

### Step 3 — Register it (one line)

```python
# src/continuum/connectors/registry.py → register_default_connectors()
from continuum.connectors.s3 import S3Connector
register_connector("s3", S3Connector(), replace=replace)
```

### Step 4 — Use it

```python
from continuum.core.container import get_container

s3 = get_container().connectors["s3"]   # or get_connector("s3")
client = await s3.connect()              # live client (you own it)
```

It is now automatically in `health_check_all()`, gets uniform
mode/describe/secret-masking, and is configurable via env (api key → cloud,
localhost → docker).

### Which flavor?

- **You own the client** (S3, Postgres, Kafka, Redis, Temporal) → implement
  `connect()` returning the live client; consumers call it. *(the Redis/Temporal
  pattern)*
- **A library owns the client** (like mem0) → implement a `to_<lib>_block()` /
  config method the library consumes, and use `connect()` only for a
  health-probe client. *(the VectorStore pattern)*

In short: **config first** (even if it's just a few `Settings` fields), **then
the connector, then one registration line.** The dedicated config class is
optional — add one only when the service's settings are complex enough to
warrant it.

---

## 7 · Gotchas

- **`aping()` never raises** — it returns `False` on any failure (timeout, auth,
  service down). Build dashboards on the boolean, not on exceptions.
- **`describe()` masks secrets** — passwords, tokens, and API keys come back
  masked, so it is safe to log or print (the `/connectors` playground command
  does exactly this).
- **`mode` is inferred, not stored** — change the underlying config (host, TLS,
  API key) and the mode follows automatically. There is no separate "mode"
  setting to keep in sync.
- **Vector store config has one home** — never hand-build the mem0
  `vector_store` block; call `VectorStoreConnector(...).to_mem0_block()` so
  Milvus/Qdrant params stay consistent.
- **Disabled ≠ unhealthy** — a turned-off connector reports `disabled` in
  `health_check_all()` and is never probed; don't alert on it.
