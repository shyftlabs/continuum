---
name: continuum-llm-providers
description: Pick the right LLM provider, configure structured outputs, control context-window compression, and use the LLMClient directly. Provider routing is by model-string prefix; LiteLLM has been removed. Invoke when the user asks about "switch to Claude", "Gemini structured output", "context length error", "rate limiting", "JSON schema", or "how does the LLM client work".
---

# Continuum LLM Providers Skill

Authoritative source: [`docs/llm.md`](../../../docs/llm.md).

> ⚠️ **LiteLLM is removed.** All providers are called directly via
> their official SDKs. Don't suggest `litellm.*` imports or any LiteLLM
> code paths.

---

## Provider routing (by model-string prefix)

| Prefix | Provider | SDK |
|---|---|---|
| `gemini/...`, `google/...` | Gemini | OpenAI SDK against Gemini's OpenAI-compat endpoint |
| `claude/...`, `anthropic/...`, `claude-...` | Anthropic | Anthropic SDK |
| anything else (`gpt-*`, `azure/...`, `openai/...`) | OpenAI | OpenAI SDK |

```python
agent.model = "gpt-4o-mini"                           # OpenAI
agent.model = "claude-sonnet-4-20250514"              # Anthropic
agent.model = "gemini/gemini-2.5-flash"               # Gemini
```

---

## LLMClient (direct usage)

```python
from orchestrator.llm import LLMClient, LLMConfig, ChatMessage

client = LLMClient()                          # default config

# Async (primary)
resp = await client.chat(
    [ChatMessage(role="user", content="Hi")],
    config=LLMConfig(model="gpt-4o-mini", temperature=0.3),
    tools=None, tool_choice=None,
    session_id=None,                          # auto-loads history if set
    auto_session=True,
)

# Stream
async for chunk in client.chat_stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

Sync mirrors: `chat_sync`, `chat_stream_sync`. Aliases: `complete`,
`stream`, `acomplete`.

---

## LLMConfig essentials

```python
LLMConfig(
    model="gpt-4o-mini",
    fallback_models=["gemini/gemini-1.5-flash"],
    temperature=0.7,
    max_tokens=4096,
    timeout=60,
    max_retries=3,
    enable_fallback=True,

    # Structured output
    response_format=None,                     # dict | type[BaseModel] | None
    json_mode=False,                          # plain JSON object mode

    # Rate limiting
    rate_limit_rpm=None,                      # token-bucket if set

    # Tracing metadata
    metadata={},
)
```

---

## Structured outputs (3 patterns)

### Plain JSON

```python
config = LLMConfig(model="gpt-4o-mini", json_mode=True)
resp = await client.chat(messages, config=config)
# resp.content is valid JSON
```

### Pydantic schema (recommended)

```python
from pydantic import BaseModel

class Result(BaseModel):
    sentiment: str
    confidence: float

config = LLMConfig(model="gpt-4o-mini", response_format=Result)
resp = await client.chat(messages, config=config)
parsed = Result.model_validate_json(resp.content)
```

### Via BaseAgent (agent runner parses for you)

```python
agent = BaseAgent(name="...", instructions="...", output_schema=Result)
resp = await runner.run(agent, "...")
parsed: Result = resp.structured_output
```

---

## Capability checks

```python
from orchestrator.llm.utils import (
    check_response_format_support, check_json_schema_support,
    supports_tools_with_json_mode,
)

check_response_format_support("claude-sonnet-4-20250514")     # False
supports_tools_with_json_mode("gemini/gemini-2.5-pro")        # False — Gemini can't do both
```

---

## Context window management

Hardcoded limits per model (GPT-4o = 128K, Claude family = 200K, Gemini
2.5 = 1M, …). 25% of the window is reserved for the response by
default.

```python
from orchestrator.llm import (
    ContextWindowManager, TruncationStrategy, get_context_window_manager,
    ContextManagementConfig, get_progressive_context_manager, CompressionStrategy,
)

mgr = get_context_window_manager()
limits = mgr.get_model_limits("gpt-4o-mini")
mgr.will_exceed_limit(messages, "gpt-4o-mini")                 # bool
truncated, result = mgr.truncate_messages(messages, "gpt-4o-mini",
                                          strategy=TruncationStrategy.SMART)
```

Proactive compression (the runner does this for you):

```python
cfg = ContextManagementConfig(
    enabled=True, compression_threshold=0.8,
    summarization_model="gpt-4o-mini", keep_recent_messages=10,
    compression_strategy=CompressionStrategy.SMART,
)
mgr = get_progressive_context_manager(config=cfg)
new_msgs, result = await mgr.compress_if_needed(messages, model="gpt-4o-mini")
```

---

## Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type":"string"}},
                       "required": ["city"]},
    },
}]
resp = await client.chat(messages, tools=tools, tool_choice="auto")
# tool_choice: "auto" | "required" | "none" | {"function": {"name": "..."}}
```

---

## Provider quirks (handled automatically)

- **Gemini + tools + JSON mode**: not supported simultaneously; framework
  drops `json_mode` if tools are present.
- **Anthropic message format**: system messages extracted, tool results
  wrapped — done transparently.
- **Anthropic `max_tokens`**: required by SDK; defaults to 4096 when
  unset.

---

## Don't

- Don't suggest LiteLLM (`from litellm import …`) — it's gone.
- Don't expect Claude to honour `response_format` natively — it doesn't.
  Instruct it via the system prompt and parse manually.
- Don't run JSON mode + tools on Gemini and expect both to work — pick
  one.
- Don't hammer the API — use `LLMConfig.rate_limit_rpm=N` for predictable
  pacing under tier limits.
