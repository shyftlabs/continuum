---
name: continuum-observability
description: Trace agent runs with Langfuse, decorate functions with @observe, collect latency/token/error metrics, and report errors. Invoke when the user asks about "see what the LLM was prompted with", "Langfuse traces", "track latency", "metrics dashboard", "error reporting", or "instrument my function".
---

# Continuum Observability Skill

Authoritative source: [`docs/observability.md`](../../../docs/observability.md).

---

## Quick start

Tracing is on by default if Langfuse is configured.

```bash
docker compose --profile observability up -d              # brings up Langfuse on :3000
```

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=…                                      # from Langfuse UI
LANGFUSE_SECRET_KEY=…
```

Open http://localhost:3000 — every agent run, LLM call, and tool call
shows up automatically.

To disable: `LANGFUSE_ENABLED=false`.

---

## Trace your own functions

```python
from continuum.observability import observe, SpanLevel

@observe(name="my-pipeline", capture_input=True, capture_output=True,
         metadata={"version": "v2"}, level=SpanLevel.DEFAULT)
async def run_pipeline(data):
    ...
```

Specialized variants:

```python
from continuum.observability import trace_tool, trace_agent

@trace_tool(name="search_db", tool_type="database")
def search_db(query): ...

@trace_agent
async def my_custom_agent_runner(...): ...
```

---

## Manual spans

```python
from continuum.core.container import get_container

mgr = get_container().tracing_manager

with mgr.trace(name="onboarding", user_id="u1", session_id="s1",
               metadata={"tier": "pro"}, tags=["beta"]) as trace:
    with mgr.span(name="step-1", input={"foo": 1}) as span:
        span.event(name="cache-hit")
        span.score(name="quality", value=0.92)
```

Generation spans (LLM calls):

```python
with mgr.span(name="step") as span:
    gen = span.generation(name="llm-call", model="gpt-4o-mini",
                          input=messages)
    # … call the LLM …
    gen.end(output=response_text,
            usage_prompt_tokens=120, usage_completion_tokens=80, usage_total_tokens=200)
```

---

## Async-safe trace context

```python
from continuum.observability import (
    set_trace_context, restore_trace_context,
    get_current_trace_id, get_current_session_id, get_current_user_id,
)

token = set_trace_context(trace_id="abc", user_id="u1", session_id="s1")
try:
    await client.chat(messages)                # inherits the trace context
finally:
    restore_trace_context(token)
```

---

## Metrics

```python
from continuum.observability import (
    get_metrics_collector, get_metrics_summary, reset_metrics,
)

mc = get_metrics_collector()

# Latency
with mc.track_latency("rag.retrieve") as m:
    docs = await retrieve(query)
print(m.duration_ms)

# Or record directly
mc.record_latency("db.query", 12.4, metadata={"table": "users"})

summary = get_metrics_summary()      # {latency: {...}, tokens: {...}, errors: {...}}
```

---

## Error reporting

```python
from continuum.observability import (
    report_error, report_exception, flush_errors,
    enable_error_reporting, disable_error_reporting,
)

try:
    ...
except SomeError as e:
    # `context` is a short label string; structured data goes in `metadata=`.
    report_error(e, context="db_query", user_id="u1", metadata={"table": "users"})
```

Every `OrchestratorError` with `should_report=True` is reported
automatically. The reporter has a thread-safe queue (max 1000 entries)
and an auto-flush every 5 seconds.

---

## Debug tip: log the assembled prompt

```env
LOG_PROMPT_CONTENT=true
```

By default (`false`) the log says which agent built a prompt and how big
it was (`FINAL PROMPT [shop] <4383 chars>`), and identifiers show as
`id#…` pseudonyms. With `true` it prints the whole message list sent to
the LLM, including memories, tool results, model output and the real ids.
That's what you need to debug memory / RAG / handoff flows. Turn it on
locally only, never where logs are shipped. `LOG_FULL_PROMPT` is gone.

---

## Writing a log line in Continuum code

Pass values as `%s` arguments. Ruff `G004` rejects f-strings in logging
calls, because an f-string line cannot be redacted. Then wrap each value
by **whose data it is**, not by whether it looks sensitive:

```python
from continuum.logging import log_content, log_id

logger.info("TOOL RESULT: %s -> %s", tool_name, log_content(result))
logger.info("Session ready: %s", log_id(session_id))
```

| value | wrapper |
|---|---|
| prompt, memory, tool argument or result, model output | `log_content()` |
| `user_id`, `session_id`, `memory_id` | `log_id()` |
| agent/tool names, schemas, model ids, counts, `trace_id`, paths, commands | bare |

- Exceptions stay bare, unless the text quotes input: a pydantic
  `ValidationError` includes `input_value='…'`, so wrap it.
- `extra={...}` values follow the same rules. Third-party handlers
  serialise them even though Continuum's formatters don't.
- Never slice a logged value (`[:8]`, `[:200]`). Pass it whole: the
  wrappers handle it.
- `log_id(x) if x else "none"`, not `log_id(x or "none")`.
- Add a path to `tests/unit/test_log_canary.py` when you add one. It is
  the only check that a wrapper wasn't forgotten.

Full rules: [`docs/installation.md`](../../../docs/installation.md), "Adding a log line".

---

## ObservabilityConfig (when wiring providers manually)

```python
from continuum.observability import ObservabilityConfig, initialize_observability

cfg = ObservabilityConfig(
    providers=["langfuse"],
    enabled=True,
    public_key="pk-…", secret_key="sk-…",
    host="http://localhost:3000",
    sample_rate=1.0, flush_interval=1, flush_at=15,
    environment="production", default_tags=["app:billing"],
)
mgr = initialize_observability(cfg)
```

The lifecycle manager normally calls this for you.

---

## Custom provider

```python
from continuum.observability import (
    ObservabilityProvider, ProviderCapabilities, register_provider,
)

class MyProvider(ObservabilityProvider):
    def __init__(self): super().__init__(name="my", config={})
    def supports_feature(self, feat): return feat == ProviderCapabilities.TRACE
    def trace(self, name, **kw): ...
    def span(self, *, trace_id, name, **kw): ...
    # …generation, event, score, flush, shutdown

register_provider("my", MyProvider())
```

---

## Don't

- Don't expect to see traces if `LANGFUSE_ENABLED=false` or the keys
  are missing.
- Don't rely on `@observe` to capture huge payloads — `truncate_data`
  caps each field at 10 KB by default.
- Don't make `@observe`-decorated functions sync if the rest of your
  code is async — the contextvars magic works best in fully-async code.
- Don't `flush_langfuse()` in the middle of a request loop — it's
  expensive. Lifecycle shutdown handles it.
- Don't forget to set `SHARED_SERVICES_ENABLED=false` if your process
  is the sole owner of Langfuse — otherwise traces don't flush on
  shutdown.
