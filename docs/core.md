# Core Module

The "plumbing" layer. The core module owns:

- **`Container`** — the dependency-injection container that lazily
  creates and shares `LLMClient`, `MemoryClient`, `SessionClient`,
  `TracingManager`, the Langfuse client, and the global `ToolExecutor`.
- **`OrchestratorLifecycle`** — orderly startup/shutdown with health
  checks, signal handling, and graceful resource cleanup.
- **`HealthCheck`** — pluggable health probes for Redis, Qdrant,
  Langfuse, the LLM provider, and Temporal.
- **`ContextManager`** — async-safe execution-context propagation
  via `contextvars`.

`from orchestrator.core import (
    Container, ContainerConfig, get_container, reset_container,
    OrchestratorLifecycle, get_lifecycle_manager,
    initialize_orchestrator, shutdown_orchestrator,
    validate_configuration, ConfigurationError,
    HealthCheck, HealthCheckResult, HealthStatus,
    get_health_checker, check_all_health,
)`

---

## 1 · Quick start

```python
from orchestrator.core.lifecycle import get_lifecycle_manager

async def main():
    lifecycle = get_lifecycle_manager()
    result = await lifecycle.initialize()
    if not result.success:
        print("startup errors:", result.errors)
        return
    # … run your agents
    await lifecycle.shutdown()
```

Or as a context manager:

```python
async with get_lifecycle_manager() as lifecycle:
    ...
```

---

## 2 · `Container` (Dependency Injection)

`from orchestrator.core.container import Container, ContainerConfig, get_container, reset_container`

A **thread-safe singleton** that lazily initializes every shared client
on first access. `AgentRunner()` reaches for it when no container is
passed explicitly.

### `ContainerConfig`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `auto_initialize` | `bool` | `True` | If `False`, properties don't auto-init (test injection) |
| `enable_memory` | `bool` | `settings.memory_enabled` | If `False`, `memory_client` returns `None` |
| `enable_session` | `bool` | `settings.session_enabled` | If `False`, `session_client` returns `None` |
| `enable_langfuse` | `bool` | `settings.langfuse_enabled` | If `False`, `langfuse_client`/`tracing_manager` return `None` |
| `llm_config` | `dict \| None` | `None` | Forwarded to `LLMClient(**)` |
| `memory_config` | `dict \| None` | `None` | Forwarded to `MemoryConfig(**)` |
| `session_config` | `dict \| None` | `None` | Forwarded to `SessionConfig(**)` |
| `langfuse_config` | `dict \| None` | `None` | Forwarded to `ObservabilityConfig(**)` |

### Properties (lazy)

| Property | Returns | When |
|---|---|---|
| `llm_client` | `LLMClient` | Always |
| `memory_client` | `MemoryClient \| None` | `None` if memory disabled |
| `session_client` | `SessionClient \| None` | `None` if session disabled |
| `session_provider` | `BaseSessionProvider \| None` | Underlying provider |
| `langfuse_client` | `Any \| None` | `None` if Langfuse disabled |
| `tracing_manager` | `TracingManager \| None` | `None` if Langfuse disabled |
| `tool_executor` | `ToolExecutor \| None` | Set by you via `set_tool_executor()` |

### Setters (for tests / custom wiring)

`set_llm_client(client)`, `set_memory_client(client)`,
`set_session_client(client, provider=None)`, `set_langfuse_client(client)`,
`set_tracing_manager(manager)`, `set_tool_executor(executor)`. Each
setter accepts the concrete class **or** the matching protocol
(`ILLMClient`, `IMemoryClient`, `ISessionClient`).

### Predicates

`has_llm_client()`, `has_memory_client()`, `has_session_client()`,
`has_langfuse_client()`, `has_tool_executor()`.

### Lifecycle methods

- `initialize_all() -> dict[str, bool]` — eagerly init everything; returns per-client success
- `reset()` — clear all clients (testing only)
- `await shutdown()` — flush Langfuse, close LLM/memory/session with timeouts; respects `settings.shared_services_enabled`
- `async with container: ...` — context manager calls `shutdown()` on exit

### Global access

```python
from orchestrator.core.container import get_container, reset_container

container = get_container()                       # default singleton
container = get_container(config=ContainerConfig(enable_memory=False))   # first call only
reset_container()                                 # tests
```

---

## 3 · `OrchestratorLifecycle`

`from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager`

```python
lifecycle = OrchestratorLifecycle(
    shutdown_timeout=10.0,
    fail_on_unhealthy=False,
    verify_connections=True,
    enable_signal_handlers=True,
)
result = await lifecycle.initialize()
```

| Param | Default | Purpose |
|---|---|---|
| `shutdown_timeout` | `10.0` | Seconds for graceful shutdown |
| `fail_on_unhealthy` | `False` | Abort init when health checks fail |
| `verify_connections` | `True` | Eager connect at startup (Redis/Qdrant/Langfuse) |
| `enable_signal_handlers` | `True` | Register SIGINT/SIGTERM. **Set `False` in FastAPI/uvicorn lifespans** |

### Properties

- `state: LifecycleState` — `NOT_INITIALIZED` → `INITIALIZING` → `RUNNING` → `SHUTTING_DOWN` → `SHUTDOWN` (or `FAILED`)
- `is_running: bool`

### Methods

- `register_shutdown_callback(coro)` — coroutine to run during shutdown
- `await initialize() -> InitializationResult`:
  1. `validate_configuration()`
  2. Initialize the global `Container`
  3. Run health checks
  4. Abort if `fail_on_unhealthy=True` and any check fails
  5. Register signal handlers (if enabled)
- `await shutdown()` — invoke registered callbacks, then `container.shutdown()`
- `await get_health() -> OverallHealthResult`
- `async with lifecycle: ...`

### `InitializationResult`

| Field | Type | Notes |
|---|---|---|
| `success` | `bool` | |
| `state` | `LifecycleState` | |
| `health` | `OverallHealthResult \| None` | |
| `errors` | `list[str]` | Configuration / startup errors |
| `warnings` | `list[str]` | Non-fatal |
| `initialized_at` | `datetime` | |

`to_dict()` for serialization.

### Convenience module-level functions

```python
from orchestrator.core.lifecycle import (
    initialize_orchestrator, shutdown_orchestrator, validate_configuration,
)

result = await initialize_orchestrator(fail_on_unhealthy=False, verify_connections=True)
await shutdown_orchestrator()

errors, warnings = validate_configuration()       # both list[ConfigurationError]
```

`ConfigurationError` here is a **dataclass** (`field`, `message`,
`severity`) — distinct from the *exception* of the same name in
`orchestrator.exceptions`.

---

## 4 · Health checks

`from orchestrator.core.health import (
    HealthCheck, HealthCheckResult, HealthStatus, OverallHealthResult,
    get_health_checker, check_all_health,
)`

### `HealthStatus`
```python
class HealthStatus(str, Enum):
    HEALTHY / UNHEALTHY / DEGRADED / UNKNOWN
```

### `HealthCheckResult`
- `name`, `status`, `message`, `latency_ms`, `details`, `checked_at`

### `OverallHealthResult`
- `status: HealthStatus` — worst of all child statuses
- `checks: list[HealthCheckResult]`
- `total_latency_ms`
- `to_dict()` keys results by check name

### Built-in checks

`HealthCheck` automatically registers: `redis`, `qdrant`, `langfuse`,
`llm`, `temporal`.

```python
checker = get_health_checker()
result = await checker.check_all(timeout=10.0)
print(result.status, [c.name for c in result.checks])
```

### Custom checks

```python
async def check_my_api() -> HealthCheckResult:
    start = time.monotonic()
    try:
        await my_api.ping()
        return HealthCheckResult(name="my_api", status=HealthStatus.HEALTHY,
                                 latency_ms=(time.monotonic() - start) * 1000)
    except Exception as e:
        return HealthCheckResult(name="my_api", status=HealthStatus.UNHEALTHY,
                                 message=str(e))

get_health_checker().register_check("my_api", check_my_api)
```

`check_all_health(timeout=10.0)` is a module-level shorthand for
`get_health_checker().check_all(timeout)`.

---

## 5 · Execution context

`from orchestrator.core.context import (
    ContextManager, ExecutionContext, ContextScope, ContextToken,
    get_context_manager, get_current_context,
    get_trace_id, get_span_id, get_user_id, get_session_id,
    get_run_id, get_agent_name,
)`

A thin wrapper around Python `contextvars`. The agent runner manages
this for you; reach for it directly only when writing custom executors.

```python
mgr = get_context_manager()

with mgr.context(trace_id="abc", user_id="u1") as ctx:
    print(ctx.trace_id)
    print(get_user_id())             # "u1"
```

`with mgr.span_context(span_id="xyz")` and
`mgr.agent_context(agent_name="planner")` are convenience scopes.

---

## 6 · Protocols

`from orchestrator.protocols import ILLMClient, IMemoryClient, ISessionClient`

Runtime-checkable protocols for swapping in mocks or custom clients
without inheriting from the framework's classes.

- `ILLMClient`: `async chat()`, `async chat_stream()`, `count_tokens()`
- `IMemoryClient`: `is_enabled` property, `async search()`, `async add()`
- `ISessionClient`: `is_enabled` property, `async get_conversation_history()`, `async add_message()`

`Container.set_*_client()` accepts any object satisfying the
corresponding protocol.

---

## 7 · `Settings` & environment

`from orchestrator.config import settings, Settings, get_settings`

A pydantic-settings `BaseSettings`. See [`installation.md`](installation.md)
for the full env-var reference.

`__repr__()` masks all secret/key/password fields, so it's safe to log
`settings`. `settings = get_settings()` is the singleton;
`get_settings()` is LRU-cached.

---

## 8 · Exception model

`from orchestrator.exceptions import (
    OrchestratorError, ConfigurationError, ValidationError,
    ObservabilityError, LangfuseError, TracingError,
    NetworkError, ProviderError,
    ErrorSeverity, ErrorCategory,
    wrap_exception, set_error_reporter, get_error_reporter,
)`

Every typed exception in the SDK descends from `OrchestratorError`,
which carries:

- `message`, `error_code`
- `category: ErrorCategory` — `CONFIGURATION`, `AUTHENTICATION`, `RATE_LIMIT`, `TIMEOUT`, `VALIDATION`, `NETWORK`, `PROVIDER`, `INTERNAL`, `OBSERVABILITY`, `UNKNOWN`
- `severity: ErrorSeverity` — `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- `context: dict` — extras (auto-redacts sensitive keys)
- `original_error: Exception | None`
- `trace_id`, `span_id`
- `should_report: bool`

`to_dict()` and `__str__()` redact sensitive context.
`wrap_exception(error, error_class, message=None, **kwargs)` is a
helper for converting third-party exceptions while preserving the
original.

---

## 9 · Patterns

### Test container

```python
container = Container(ContainerConfig(auto_initialize=False))
container.set_llm_client(MockLLM())
container.set_memory_client(None)
container.set_session_client(None)
runner = AgentRunner(container=container)
```

### FastAPI lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from orchestrator.core.lifecycle import OrchestratorLifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    lc = OrchestratorLifecycle(enable_signal_handlers=False)
    await lc.initialize()
    yield
    await lc.shutdown()

app = FastAPI(lifespan=lifespan)
```

### Disable everything but the LLM

```python
container = Container(ContainerConfig(
    enable_memory=False, enable_session=False, enable_langfuse=False,
))
```

### Health-gated startup

```python
result = await get_lifecycle_manager(fail_on_unhealthy=True).initialize()
if not result.success:
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(1)
```

---

## 10 · Gotchas

- **`Container` is a singleton.** First call to `get_container()` wins
  the configuration. `reset_container()` to re-configure.
- **Lifecycle signal handlers conflict in notebooks**. Pass
  `enable_signal_handlers=False` in Jupyter / FastAPI / embedded contexts.
- **`shared_services_enabled=True`** (the default) means
  `container.shutdown()` does *not* close Redis or flush Langfuse — set
  `False` if your app is the sole owner.
- **`ConfigurationError`** exists twice: data class in `core.lifecycle`,
  exception in `orchestrator.exceptions`.
- **`auto_initialize=False`** containers won't connect to anything until
  you call `initialize_all()` or set every client by hand.
