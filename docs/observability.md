# Observability Module

Tracing, metrics, error reporting, and a multi-provider abstraction.
The default provider is **Langfuse** (self-hosted via the included
docker-compose profile, or Langfuse Cloud), with the architecture
designed to plug in alternatives like Vertex or Datadog.

What you get:
- `TracingManager` with `Trace`, `Span`, `GenerationSpan`
- `@observe`, `@trace_tool`, `@trace_agent` decorators
- `MetricsCollector` for latency, token usage, errors, and cost estimates
- `ErrorReporter` with thread-safe queue and auto-flush
- Async-safe context propagation via `contextvars`
- A provider registry pattern for plugging in additional backends

`from orchestrator.observability import (
    ObservabilityConfig, TracingManager, Trace, Span, GenerationSpan, SpanLevel,
    observe, trace_tool, trace_agent, ObservationContext,
    MetricsCollector, get_metrics_collector,
    ErrorReporter, get_error_reporter, report_error, report_exception,
    initialize_observability, is_initialized, reset_initialization,
    ProviderManager, ProviderRegistry, ObservabilityProvider,
    get_provider, get_provider_manager, get_provider_registry, register_provider,
    # …trace context helpers (§5)
)`

---

## 1 · Quick start

Tracing is on by default. With Langfuse running on `localhost:3000` and
the public/secret keys in `.env`, every LLM call, tool call, and agent
run is traced automatically.

To trace your own functions:

```python
from orchestrator.observability import observe

@observe(name="my-pipeline", capture_output=True)
async def run_pipeline(data):
    ...
```

Turn observability off entirely with `LANGFUSE_ENABLED=false` — every
tracing call no-ops.

---

## 2 · `ObservabilityConfig`

`from orchestrator.observability import ObservabilityConfig`

A pydantic model. Defaults read from the global `Settings`.

| Field | Type | Default | Description |
|---|---|---|---|
| `providers` | `list[str]` | `[]` | Provider names (e.g. `["langfuse"]`); empty = legacy auto-detect |
| `provider_configs` | `dict[str, dict]` | `{}` | Per-provider config |
| `enabled` | `bool` | `settings.langfuse_enabled` | Legacy switch |
| `public_key` | `str \| None` | `settings.langfuse_public_key` | |
| `secret_key` | `str \| None` | `settings.langfuse_secret_key` | |
| `host` | `str` | `settings.langfuse_host` | |
| `sample_rate` | `float` | `settings.langfuse_sample_rate` | 0.0–1.0 |
| `flush_interval` | `int` | `settings.langfuse_flush_interval` | Seconds |
| `flush_at` | `int` | `settings.langfuse_flush_at` | Event count threshold |
| `debug` | `bool` | `settings.langfuse_debug` | Verbose tracing logs |
| `release` | `str \| None` | `settings.langfuse_release` | |
| `environment` | `str` | `settings.environment` | `development` / `staging` / `production` |
| `default_tags` | `list[str]` | `[]` | Applied to every trace |
| `default_metadata` | `dict` | `{}` | Applied to every trace |

Methods: `is_configured()`, `get_provider_config(name)`,
`to_langfuse_kwargs()`, `to_dict()`.

---

## 3 · `TracingManager`, `Trace`, `Span`, `GenerationSpan`

```python
mgr = get_container().tracing_manager

with mgr.trace(name="customer-flow",
               user_id="u1", session_id="s1",
               metadata={"tier": "pro"},
               tags=["beta"]) as trace:
    with mgr.span(name="step-1", input={"foo": 1}) as span:
        span.event(name="cache-hit")
```

### `Trace`
- `id`, `name`, `langfuse_trace`
- `update(...)`
- `span(name, *, input=None, metadata=None, level=DEFAULT) -> Span`
- `generation(name, *, model=None, model_parameters=None, input=None, metadata=None) -> GenerationSpan`
- `event(name, *, input=None, output=None, metadata=None, level=DEFAULT)`
- `score(name, value, *, comment=None, data_type=None)`
- `get_trace_url() -> str | None` — direct link in the Langfuse UI

### `Span`
- `update(...)`, `end(...)`
- `span(...)` — child spans nest naturally
- `generation(...)` — for LLM calls within the span
- `event(...)`, `score(...)`

### `GenerationSpan` *(LLM calls)*
- `update(...)` and `end(...)` accept
  `usage_prompt_tokens`, `usage_completion_tokens`, `usage_total_tokens`
  for token accounting.
- `score(...)` for quality scores.

### `SpanLevel`
`DEBUG`, `DEFAULT`, `WARNING`, `ERROR`.

---

## 4 · Decorators

`from orchestrator.observability import observe, trace_tool, trace_agent, ObservationContext`

### `@observe`

```python
@observe(
    name=None,                     # defaults to function name
    capture_input=True,
    capture_output=True,
    metadata=None,
    level=SpanLevel.DEFAULT,
    manager=None,                  # custom TracingManager
)
async def my_func(...): ...
```

Works on sync and async functions. Captures input args, return value,
exceptions, and timing. Nests under any active span via `SpanScope`.

### `@trace_tool`

```python
@trace_tool(
    name=None,
    tool_type="function",          # "function" | "api" | "database" | …
    capture_input=True,
    capture_output=True,
    metadata=None,
    manager=None,
)
def my_tool(...): ...
```

### `@trace_agent`

Wraps an entire agent invocation as one trace. The `AgentRunner` adds
this automatically — apply manually only when you bypass the runner.

### `ObservationContext`

`ObservationContext(name, *, span_type="span", metadata=None, manager=None)`
is a **synchronous** context manager. Use `with`, not `async with`.
Methods: `set_input(value)`, `set_output(value)`, `add_metadata(d)`,
`set_level(level)`, `set_error(exc)`.

```python
with ObservationContext(name="pipeline.stage-3", metadata={"v": 2}) as obs:
    obs.add_metadata({"items": 42})
    result = await do_work()
    obs.set_output(result)
```

---

## 5 · Trace context (async-safe)

`from orchestrator.observability import (
    set_trace_context, restore_trace_context, clear_trace_context,
    get_current_trace_id, get_current_span_id,
    get_current_user_id, get_current_session_id,
    get_current_agent_name, get_current_run_id,
    get_current_trace_client, get_current_span_client,
    get_parent_observation_id, get_parent_client,
    get_trace_metadata, build_langfuse_metadata,
    truncate_data, traced_operation,
    TraceScope, SpanScope, TraceContextToken,
)`

Everything is built on `contextvars`, so contexts propagate naturally
across `await` and `asyncio.gather()`.

```python
token = set_trace_context(trace_id="abc", user_id="u1")
try:
    ...
finally:
    restore_trace_context(token)
```

`get_parent_observation_id()` returns the current span id if you're
inside one, otherwise the trace id — exactly what Langfuse wants for
nesting child observations.

`truncate_data(data, max_size=10240)` clips bulky payloads. Default cap
is **10 KB** per field.

---

## 6 · Metrics

`from orchestrator.observability import (
    MetricsCollector, get_metrics_collector, get_metrics_summary,
    initialize_metrics_collector, reset_metrics,
)`

### Latency

```python
mc = get_metrics_collector()

with mc.track_latency("db.query", metadata={"table": "users"}) as m:
    rows = await db.fetch_all(...)
print(m.duration_ms)
```

Or:

```python
mc.record_latency("db.query", 12.4, metadata={"table": "users"})
```

### Token usage

`TokenUsageMetric` carries `prompt_tokens`, `completion_tokens`,
`total_tokens`, `model`, `timestamp`. Its `cost_estimate` property
returns a USD estimate based on a hardcoded pricing table.

### Errors

`ErrorMetric`: `name`, `error_type`, `error_message`, `timestamp`,
`metadata`. Independent of error reporting (§7).

### Summary

```python
summary = get_metrics_summary()
# {"latency": {...}, "tokens": {...}, "errors": {...}}
```

`reset_metrics()` between tests.

---

## 7 · Error reporting

`from orchestrator.observability import (
    ErrorReporter, ErrorReportingContext,
    get_error_reporter, report_error, report_exception,
    enable_error_reporting, disable_error_reporting, flush_errors,
)`

Every `OrchestratorError` with `should_report=True` is auto-reported.
Manual:

```python
try:
    ...
except SomeError as e:
    report_error(e, context={"user_id": "u1"})
```

Module-level helpers:

- `report_error(error, *, context=None, trace_id=None, span_id=None, user_id=None, session_id=None, metadata=None, immediate=False)`
- `report_exception(context: str, *, trace_id=None, span_id=None, user_id=None, metadata=None)` — call inside an `except` block; the live exception is auto-pulled from `sys.exc_info()`. The first argument is a **context string**, not the exception itself.
- `enable_error_reporting()` / `disable_error_reporting()`
- `flush_errors()` — force the queue to drain (called by lifecycle shutdown)

The reporter has a thread-safe queue (`maxlen=1000`) and an
`auto_flush_interval=5.0` second background flush.

---

## 8 · Provider system

`from orchestrator.observability import (
    ObservabilityProvider, ProviderCapabilities, ProviderManager,
    ProviderRegistry, LangfuseProvider,
    get_provider, get_provider_manager, get_provider_registry, register_provider,
    initialize_observability, is_initialized, reset_initialization,
)`

### `ProviderCapabilities`
```
TRACE / SPAN / GENERATION / EVENT / SCORE
PROMPT_MANAGEMENT / DATASET_MANAGEMENT
ERROR_REPORTING / METRICS / STREAMING / BATCH_FLUSH
```

### Built-in

`LangfuseProvider` — the default.

### Registering a custom provider

```python
class MyProvider(ObservabilityProvider):
    def __init__(self):
        super().__init__(name="my", config={})
    def supports_feature(self, feat): return feat == ProviderCapabilities.TRACE
    def trace(self, name, *, user_id=None, **kwargs): ...
    def span(self, *, trace_id, name, **kwargs): ...
    # …generation, event, score, flush, shutdown

register_provider("my", MyProvider())
```

### `ProviderManager`

Aggregates all enabled providers. `mgr.is_enabled` is `True` if any
provider is enabled. `trace`, `span`, `generation`, `event`, `score`
fan out; the first non-None result is returned.

### Initialization

```python
from orchestrator.observability import initialize_observability, ObservabilityConfig

mgr = initialize_observability(ObservabilityConfig(providers=["langfuse"]))
print(is_initialized())          # True
reset_initialization()           # tests
```

The lifecycle manager calls `initialize_observability()` for you.

---

## 9 · Common patterns

### Trace a custom flow that spans multiple agents

```python
mgr = get_container().tracing_manager

with mgr.trace(name="onboarding", user_id="u1", session_id="s1") as trace:
    r1 = await runner.run(intake_agent, "...", trace_id=trace.id)
    r2 = await runner.run(verifier, r1.content, trace_id=trace.id)
    trace.update(output=r2.content)
```

### Score an LLM output

```python
mgr.get_current_trace().score(name="answer.correctness", value=0.92,
                              comment="auto-graded by judge agent")
```

### Capture latency

```python
with get_metrics_collector().track_latency("rag.retrieve") as m:
    docs = await retriever.search(query)
```

### Scope a block to specific trace context

```python
from orchestrator.observability import set_trace_context, restore_trace_context

token = set_trace_context(trace_id=tid, user_id="u1", session_id="s1")
try:
    await client.chat(messages)
finally:
    restore_trace_context(token)
```

---

## 10 · Gotchas

- **Langfuse is shared infrastructure by default**
  (`shared_services_enabled=True`). `Container.shutdown()` does **not**
  flush or close Langfuse. If your process is the sole owner, set
  `SHARED_SERVICES_ENABLED=false`.
- **Decorators capture inputs and outputs by default.** Disable for
  privacy / payload reasons. The 10 KB cap from `truncate_data` applies
  regardless.
- **`@observe` on sync functions** works, but trace integration is best
  in async code where contextvars survive across `await`.
- **`SpanLevel.ERROR`** does not raise — it just tags the span.
- **No traces in the UI?** Three usual culprits:
  `LANGFUSE_ENABLED=false`, missing public/secret keys, or the process
  exited without flushing.
