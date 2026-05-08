# LLM Module

Unified interface for LLM completions across **OpenAI**, **Anthropic**,
and **Google Gemini**. Calls the provider SDKs directly — **LiteLLM has
been removed** as of commit `657607a`.

What this module gives you:

- A single `LLMClient` with sync, async, streaming, and tool-calling APIs
- Provider routing by model-string prefix
- Automatic structured outputs (JSON mode + Pydantic schemas)
- Proactive context-window compression
- Token-bucket rate limiting
- Automatic Langfuse tracing via the `@observe` decorator
- Auto-load and auto-save conversation history when a `session_id` is in scope

> Most application code never touches `LLMClient` directly — `AgentRunner`
> uses it under the hood. Reach for it directly when you need raw
> completions outside the agent abstraction.

---

## 1 · Provider routing

`from orchestrator.llm.providers import get_provider`

The router picks a provider purely from the **model string prefix**, in
this order:

| Prefix | Provider | SDK |
|---|---|---|
| `gemini/`, `google/` | `GeminiProvider` | OpenAI SDK against the Gemini OpenAI-compat endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`) |
| `claude/`, `anthropic/`, or starts with `claude-` | `AnthropicProvider` | Anthropic SDK |
| anything else (`gpt-*`, `azure/...`, `openai/...`, …) | `OpenAIProvider` | OpenAI SDK |

Each provider implements:

```python
class BaseProvider(ABC):
    def complete(messages, config, tools=None, tool_choice=None) -> LLMResponse
    async def acomplete(messages, config, tools=None, tool_choice=None) -> LLMResponse
    def stream(messages, config, tools=None, tool_choice=None) -> Iterator[StreamChunk]
    async def astream(messages, config, tools=None, tool_choice=None) -> AsyncIterator[StreamChunk]
```

### Provider quirks (handled for you)

- **Gemini** does not support tools and JSON mode simultaneously — the
  framework auto-disables JSON mode when tools are present.
- **Anthropic** uses a different message format (`system` is a top-level
  parameter, tool results are wrapped in user-role messages) — the
  provider transparently translates.
- **Anthropic** requires `max_tokens` — defaults to `4096` if you don't
  set one.

---

## 2 · `LLMClient`

`from orchestrator.llm import LLMClient`

```python
client = LLMClient(
    config=None,                  # default LLMConfig() if None
    enable_langfuse=True,         # auto-trace via @observe
)
```

### Async API (primary)

```python
response: LLMResponse = await client.chat(
    messages,                     # list[ChatMessage] | list[dict]
    config=None,                  # override default LLMConfig
    tools=None,                   # list[ToolDefinition] | list[dict]
    tool_choice=None,             # "auto" | "required" | "none" | {"function": {"name": ...}}
    *,
    session_id=None,              # auto-load history & save messages
    trace_metadata=None,          # forwarded to Langfuse
    auto_session=True,            # set False to skip session integration
)
```

```python
async for chunk in client.chat_stream(messages, config=None, tools=None, tool_choice=None):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Sync API

```python
response = client.chat_sync(messages, config=None, tools=None, tool_choice=None, trace_metadata=None)

for chunk in client.chat_stream_sync(messages, config=None, tools=None, tool_choice=None):
    print(chunk.content, end="")
```

### Aliases (legacy / convenience)

| Alias | Equivalent |
|---|---|
| `complete()` | `chat_sync()` |
| `stream()` | `chat_stream_sync()` |
| `acomplete()` | `chat()` |

### Utility methods

- `get_model_info(model=None) -> dict` — context-window info (`max_tokens`, `max_input_tokens`, `max_output_tokens`, `effective_input_limit`)
- `get_supported_models() -> list[str]` — known-good model names
- `count_tokens(messages, model=None) -> int` — tiktoken-backed count

### Auto-session behaviour

When `auto_session=True` (the default) **and** a `session_id` is
available (either passed explicitly or set via `trace_context`), the
client:

1. Loads the conversation history from Redis and prepends it to your
   `messages` list.
2. After the response completes, persists both the user input and the
   assistant output back to the session.

Set `auto_session=False` to opt out of this for a specific call.

---

## 3 · `LLMConfig`

`from orchestrator.llm import LLMConfig`

A Pydantic model with full provider-agnostic settings.

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `str` | `settings.default_llm_model` | Routes to provider by prefix |
| `fallback_models` | `list[str]` | `[]` | Tried in order if primary fails (when `enable_fallback=True`) |
| `temperature` | `float` | `settings.default_llm_temperature` | |
| `max_tokens` | `int \| None` | `settings.default_llm_max_tokens` | |
| `top_p` | `float \| None` | `None` | |
| `frequency_penalty` | `float \| None` | `None` | OpenAI-only |
| `presence_penalty` | `float \| None` | `None` | OpenAI-only |
| `stop` | `list[str] \| str \| None` | `None` | Stop sequences |
| `seed` | `int \| None` | `None` | Determinism |
| `timeout` | `int` | `settings.llm_request_timeout` | Seconds |
| `max_retries` | `int` | `settings.llm_max_retries` | |
| `enable_fallback` | `bool` | `settings.llm_enable_fallback` | |
| `response_format` | `dict \| type[BaseModel] \| None` | `None` | Structured output |
| `json_mode` | `bool` | `False` | Simple `json_object` mode |
| `user` | `str \| None` | `None` | Forwarded for billing |
| `metadata` | `dict[str, Any]` | `{}` | Trace metadata |
| `api_base` | `str \| None` | `None` | Custom endpoint |
| `api_key` | `str \| None` | `None` | Override env-supplied key |
| `api_version` | `str \| None` | `None` | Azure |
| `custom_llm_provider` | `str \| None` | `None` | Override prefix-based routing |
| `rate_limit_rpm` | `int \| None` | `None` | Activates token-bucket limiter |
| `cache` | `bool` | `False` | Provider-side cache hint |
| `cache_ttl` | `int \| None` | `None` | |

### Methods

- `to_kwargs() -> dict` — convert to provider SDK kwargs
- `with_overrides(**kwargs) -> LLMConfig` — copy with patches
- `LLMConfig.from_agent_config(agent) -> LLMConfig` — derive from a `BaseAgent`, including JSON-mode setup

### Rate limiting

```python
LLMConfig(model="gpt-4o-mini", rate_limit_rpm=60)   # ≤ 60 RPM
```

Internally `_LLMRateLimiter` does token-bucket math; if the bucket is
empty, `client.chat()` blocks until a token frees up. Useful for
respecting tier limits without trial-and-error 429s.

---

## 4 · Types

`from orchestrator.llm import ChatMessage, LLMResponse, StreamChunk, Usage,
                              FunctionCall, FunctionDefinition,
                              ToolCall, ToolDefinition`

### `ChatMessage`
```python
ChatMessage(
    role="system" | "user" | "assistant" | "tool" | "function",
    content: str | None = None,
    name: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    tool_call_id: str | None = None,
    function_call: FunctionCall | None = None,
)
```

### `LLMResponse`
- `id`, `model`, `content`, `role`
- `tool_calls: list[ToolCall] | None`
- `function_call: FunctionCall | None`
- `usage: Usage | None`
- `finish_reason: str | None`
- `raw_response: dict | None` — provider-native payload

### `StreamChunk`
- `id`, `model`, `content`, `role`
- `tool_calls: list[ToolCall] | None`
- `finish_reason: str | None`
- `is_finished: bool`

### `ToolDefinition`
```python
ToolDefinition(
    type="function",
    function=FunctionDefinition(name=..., description=..., parameters={...}),
)
```

### `Usage`
`prompt_tokens`, `completion_tokens`, `total_tokens`.

---

## 5 · Structured outputs

Three ways to ask for structured output, in increasing strictness:

### 5.1 Plain JSON mode
```python
config = LLMConfig(model="gpt-4o-mini", json_mode=True)
resp = await client.chat(messages, config=config)
# resp.content is valid JSON; you parse it.
```

### 5.2 JSON Schema (dict)
```python
schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
config = LLMConfig(model="gpt-4o-mini", response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}})
resp = await client.chat(messages, config=config)
```

### 5.3 Pydantic model (recommended)
```python
from pydantic import BaseModel

class Result(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

config = LLMConfig(model="gpt-4o-mini", response_format=Result)
resp = await client.chat(messages, config=config)
parsed = Result.model_validate_json(resp.content)
```

When using `BaseAgent`, `output_schema=Result` is the equivalent —
`AgentRunner` parses the response for you and returns it on
`response.structured_output`.

### Capability checks

```python
from orchestrator.llm.utils import (
    check_response_format_support, check_json_schema_support,
    supports_tools_with_json_mode,
)
```

- `check_response_format_support(model, custom_llm_provider=None)` — does this model accept any `response_format`?
- `check_json_schema_support(model, custom_llm_provider=None)` — does it accept *strict* JSON schemas?
- `supports_tools_with_json_mode(model, custom_llm_provider=None)` — `False` for Gemini and Vertex (mutually exclusive features).

---

## 6 · Context window management

`from orchestrator.llm import (
    ContextWindowManager, ModelLimits, TruncationStrategy,
    get_context_window_manager,
)`

### `ContextWindowManager`

Hardcoded limits per model. Usage:

```python
mgr = get_context_window_manager()
limits = mgr.get_model_limits("gpt-4o-mini")          # ModelLimits
mgr.count_tokens(messages, "gpt-4o-mini")             # int
mgr.will_exceed_limit(messages, "gpt-4o-mini")        # bool
truncated, result = mgr.truncate_messages(messages, "gpt-4o-mini",
                                          strategy=TruncationStrategy.SMART)
```

Strategies: `OLDEST_FIRST`, `KEEP_SYSTEM_AND_RECENT`, `SMART`, `NONE`.

Built-in limits include GPT-4o (128K), Claude Sonnet/Opus/Haiku (200K),
Gemini 2.5 family (1M), older Gemini Pro (32K), GPT-3.5 (16K), GPT-4
(8K). 25% of the window is reserved for the response by default
(`response_buffer_percent=0.25`).

### Proactive compression — `ProgressiveContextManager`

`from orchestrator.llm import (
    ContextManagementConfig, ProgressiveContextManager,
    CompressionStrategy, CompressionResult,
    get_progressive_context_manager,
)`

The agent runner automatically calls `compress_if_needed()` before each
LLM call when context approaches the threshold.

```python
from orchestrator.llm.context_management import ContextManagementConfig

cfg = ContextManagementConfig(
    enabled=True,
    compression_threshold=0.8,           # at 80% of window
    summarization_model="gpt-4o-mini",
    summarization_temperature=0.1,
    summarization_timeout=30,
    summarization_max_retries=2,
    keep_recent_messages=10,
    compression_strategy=CompressionStrategy.SMART,   # SUMMARIZE_OLD | TRUNCATE_OLDEST | SMART
    enable_caching=True,
    cache_ttl_seconds=3600,
)

mgr = get_progressive_context_manager(config=cfg)
new_messages, result = await mgr.compress_if_needed(messages, model="gpt-4o-mini")
```

`CompressionResult` reports `original_token_count`,
`compressed_token_count`, `was_compressed`, `strategy_used`,
`compression_ratio`, `latency_ms`, `cache_hit`, etc.

To override per-agent: set `agent.config.context_management = cfg`.

---

## 7 · Tracing & callbacks

`from orchestrator.llm.callbacks import (
    setup_langfuse, get_langfuse_callback, get_langfuse_metadata,
    trace_context, set_trace_context, get_trace_context, clear_trace_context,
    flush_langfuse, shutdown_langfuse,
    LangfuseTraceContext,
)`

Every LLM call is wrapped in `@observe(name="llm_chat")` and
automatically nests under any active span. To explicitly scope a series
of calls to a single trace:

```python
from orchestrator.llm.callbacks import trace_context

with trace_context(trace_id="my-trace-id"):
    a = await client.chat(messages_a)
    b = await client.chat(messages_b)
```

`LangfuseTraceContext` wraps a higher-level Langfuse trace:

```python
with LangfuseTraceContext(name="customer-support-flow",
                          user_id="u1", session_id="s1",
                          metadata={"tier": "pro"}):
    ...
```

`flush_langfuse()` and `shutdown_langfuse()` are normally handled by
`OrchestratorLifecycle.shutdown()` — call them yourself only if you
manage observability outside the lifecycle manager.

---

## 8 · Exception hierarchy

`from orchestrator.llm import (
    LLMError, LLMAuthenticationError, LLMRateLimitError,
    LLMTimeoutError, LLMContextLengthError, LLMInvalidRequestError,
    LLMServiceUnavailableError, LLMFallbackExhaustedError,
    LLMToolCallError, LLMStreamingError, LLMContentFilterError,
)`

Each provider maps native SDK exceptions into this hierarchy, so your
error-handling code is provider-agnostic.

---

## 9 · Examples

### 9.1 Direct chat (no agent)

```python
import asyncio
from orchestrator.llm import LLMClient, LLMConfig, ChatMessage

async def main():
    client = LLMClient()
    resp = await client.chat(
        [ChatMessage(role="user", content="Capital of France?")],
        config=LLMConfig(model="gpt-4o-mini", temperature=0.1),
    )
    print(resp.content, "—", resp.usage.total_tokens, "tokens")

asyncio.run(main())
```

### 9.2 Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

resp = await client.chat(messages, tools=tools, tool_choice="auto")
if resp.tool_calls:
    for tc in resp.tool_calls:
        print(tc.function.name, tc.function.arguments)
```

### 9.3 Streaming

```python
async for chunk in client.chat_stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.is_finished:
        print()
```

### 9.4 Per-call rate limit

```python
cfg = LLMConfig(model="gpt-4o-mini", rate_limit_rpm=10)
for prompt in big_batch:
    await client.chat([ChatMessage(role="user", content=prompt)], config=cfg)
```

---

## 10 · Gotchas

- **No LiteLLM imports anywhere** — searches like `from litellm import ...`
  will fail. Use `LLMClient` (or call the provider SDKs directly if you
  must).
- **`OPENAI_API_KEY` is required at framework startup** even when using
  Anthropic or Gemini, because mem0 instantiates an OpenAI embedder by
  default. To avoid this, disable long-term memory: see
  [`memory.md`](memory.md).
- **Anthropic + JSON mode**: Claude doesn't have a native `response_format`
  — `check_response_format_support()` returns `False` for Claude. If you
  need JSON, instruct it via the system prompt and parse the result.
- **Gemini + tools + JSON**: not supported simultaneously; the framework
  drops `json_mode` automatically when `tools` are present.
- **`session_id` in `chat()`** triggers automatic Redis history loading
  *and* persistence. Pass `auto_session=False` if you want to manage
  history yourself.
