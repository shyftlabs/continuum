---
name: continuum-testing
description: Write tests for Continuum agents — mock LLM and memory clients via the DI Container, use fakeredis for sessions, snapshot agent responses, and run pytest-asyncio. Invoke when the user asks "test my agent", "mock the LLM", "fakeredis", "container injection", "pytest", or wants their CI to validate agent behavior without burning real API tokens.
---

# Continuum Testing Skill

Continuum's test-friendly seams:

1. `Container(ContainerConfig(auto_initialize=False))` — opt out of auto-init and inject mocks.
2. `Container.set_*_client(...)` accepts the real client, a custom subclass, or any object satisfying the `ILLMClient` / `IMemoryClient` / `ISessionClient` protocols.
3. `fakeredis` (a dev dependency) drops in for the Redis session provider.
4. `reset_container()` and `reset_global_memory()` / `reset_global_session()` cleanly nuke globals between tests.

---

## Imports

```python
import pytest
from orchestrator.agent import AgentRunner, BaseAgent
from orchestrator.core.container import Container, ContainerConfig, reset_container
from orchestrator.llm.types import LLMResponse, StreamChunk, Usage
from orchestrator.protocols import ILLMClient
```

---

## 1. Mock LLM that satisfies `ILLMClient`

```python
class MockLLM:
    is_enabled = True

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    async def chat(self, messages, **kw):
        text = self._replies.pop(0) if self._replies else "mock reply"
        return LLMResponse(model="mock", content=text, role="assistant",
                           usage=Usage(prompt_tokens=10, completion_tokens=5,
                                       total_tokens=15))

    async def chat_stream(self, messages, **kw):
        text = self._replies.pop(0) if self._replies else "mock reply"
        for ch in text:
            yield StreamChunk(model="mock", content=ch)
        yield StreamChunk(model="mock", is_finished=True)

    def count_tokens(self, messages, model=None) -> int:
        return sum(len(m.get("content", "")) // 4 for m in messages)
```

Wire it into the container:

```python
container = Container(ContainerConfig(auto_initialize=False))
container.set_llm_client(MockLLM(["Hello, world!"]))
container.set_memory_client(None)
container.set_session_client(None)

runner = AgentRunner(container=container)
agent = BaseAgent(name="t", instructions="...")
resp = await runner.run(agent, "anything")
assert resp.content == "Hello, world!"
```

---

## 2. pytest fixtures

```python
import pytest
from orchestrator.core.container import reset_container

@pytest.fixture(autouse=True)
def clean_globals():
    reset_container()
    try:
        from orchestrator.memory.client import reset_global_memory
        reset_global_memory()
    except ImportError:
        pass
    try:
        from orchestrator.session.client import reset_global_session
        reset_global_session()
    except ImportError:
        pass
    yield

@pytest.fixture
def mock_runner():
    container = Container(ContainerConfig(auto_initialize=False))
    container.set_llm_client(MockLLM(["mocked"]))
    container.set_memory_client(None)
    container.set_session_client(None)
    return AgentRunner(container=container)
```

```python
@pytest.mark.asyncio
async def test_agent_replies(mock_runner):
    agent = BaseAgent(name="echo", instructions="...")
    resp = await mock_runner.run(agent, "ping")
    assert resp.content == "mocked"
```

`pytest-asyncio` is in the `[dev]` extra. With `asyncio_mode = "auto"`
in `pyproject.toml`, you don't need the `@pytest.mark.asyncio`
decorator.

---

## 3. fakeredis for session tests

```python
import pytest, fakeredis.aioredis
from orchestrator.session import SessionClient, SessionConfig
from orchestrator.session.providers.redis import RedisSessionProvider

@pytest.fixture
async def session_client(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis()
    cfg = SessionConfig(provider="redis", enabled=True,
                        redis_host="ignored", redis_port=0)
    provider = RedisSessionProvider(cfg, auto_initialize=False)
    provider._redis = fake                       # bypass real Redis connect
    provider._initialized = True
    return SessionClient(session_config=cfg, provider=provider, auto_initialize=False)
```

Now `await session_client.add_message(sid, ChatMessage(...))` round-trips
through fakeredis without a Docker container.

---

## 4. Snapshot-style tests against structured output

```python
from pydantic import BaseModel

class Result(BaseModel):
    sentiment: str
    confidence: float

async def test_sentiment_schema(mock_runner):
    mock_runner._container.set_llm_client(
        MockLLM(['{"sentiment":"positive","confidence":0.95}'])
    )
    agent = BaseAgent(name="s", instructions="...", output_schema=Result)
    resp = await mock_runner.run(agent, "I love it!")
    assert isinstance(resp.structured_output, Result)
    assert resp.structured_output.sentiment == "positive"
```

---

## 5. Testing tool calls

If your `MockLLM` needs to emit tool calls, populate `tool_calls` on the
`LLMResponse`:

```python
from orchestrator.llm.types import LLMResponse, ToolCall, FunctionCall

class ToolCallingMock:
    is_enabled = True
    def __init__(self):
        self._step = 0
    async def chat(self, messages, **kw):
        self._step += 1
        if self._step == 1:
            return LLMResponse(
                model="mock", content=None, role="assistant",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(
                    name="get_weather", arguments='{"city": "Paris"}'))])
        return LLMResponse(model="mock", content="It's sunny in Paris.",
                           role="assistant")
    async def chat_stream(self, *a, **kw):
        yield StreamChunk(model="mock", is_finished=True)
    def count_tokens(self, *a, **kw): return 0
```

Couple this with a fake `ToolExecutor` to test full tool-loop behaviour
without real MCP servers.

---

## 6. Asserting on streaming events

```python
from orchestrator.agent.types import EventType

async def test_streaming_emits_deltas(mock_runner):
    agent = BaseAgent(name="s", instructions="...")
    types = [ev.type async for ev in mock_runner.run_stream(agent, "hi")]
    assert EventType.RUN_START in types
    assert EventType.CONTENT_DELTA in types
    assert EventType.RUN_END in types
```

---

## 7. End-to-end with a live LLM but no infra

For "happy path" tests against a real LLM but without Redis/Qdrant:

```python
container = Container(ContainerConfig(
    enable_memory=False, enable_session=False, enable_langfuse=False,
))
agent = BaseAgent(
    name="e2e",
    instructions="...",
    memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
)
runner = AgentRunner(container=container)
```

Combine with `pytest -m integration` markers to gate them behind a key.

---

## Don't

- Don't `await runner.run_stream(...)` — it's an async generator;
  use `async for`.
- Don't mock `BaseAgent` itself — it's a dataclass; mock the LLM client
  underneath instead.
- Don't reuse `Container` across tests without `reset_container()` —
  the singleton survives between tests and leaks state.
- Don't try to test prompt-injection scanning by passing raw HTML —
  the scanners run on user input, not arbitrary strings; pass via
  `runner.run(agent, payload)`.
- Don't real-Redis your tests in CI without isolation — use `fakeredis`
  or a per-test Redis db (`SESSION_REDIS_DB`).
