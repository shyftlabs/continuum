---
name: continuum-recipes
description: Copy-pasteable Continuum patterns — RAG, plan-and-execute, ReAct, multi-tenant agents, FastAPI integration, structured output, prompt-injection scanning, custom containers. Invoke when the user asks "how do I do X with Continuum" and X is a common app pattern rather than a single API question.
---

# Continuum Recipes Skill

Common, ready-to-paste patterns. Each is verified against framework
0.2.0.

---

## 1. RAG-augmented agent

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentConfig

retrieved_docs = await my_retriever.search(query)
rag_text = "\n\n".join(d.text for d in retrieved_docs[:5])

agent = BaseAgent(
    name="rag-agent",
    instructions="Answer ONLY using the PROVIDED CONTEXT. If unsure, say so.",
    config=AgentConfig(rag_context=rag_text, require_context=True),
)
resp = await AgentRunner().run(agent, query, user_id="u1")
```

---

## 2. Plan-and-execute (orchestrator + executor)

```python
from pydantic import BaseModel
from typing import Literal

class ToolStep(BaseModel):
    step_id: str
    tool_name: str
    parameters: dict
    instruction: str
    depends_on: list[str] | None = None

class ExecutionPlan(BaseModel):
    intent: Literal["search", "checkout", "support", "other"]
    respond_directly: bool = False
    direct_response: str | None = None
    steps: list[ToolStep] = []
    user_context: str | None = None
    response_instructions: str = "Be concise."

orchestrator = BaseAgent(
    name="orchestrator",
    instructions=("Analyze the request and emit an ExecutionPlan as JSON. "
                  "If you can answer directly, set respond_directly=true."),
    output_schema=ExecutionPlan,
    model="gpt-4o-mini",
)
executor = BaseAgent(
    name="executor",
    instructions="Execute the plan steps in order using the available tools.",
    mcp_servers=[mcp],
    model="gpt-4o-mini",
)

runner = AgentRunner(agent_registry={"orchestrator": orchestrator, "executor": executor})
plan_resp = await runner.run(orchestrator, user_msg, session_id=sid, user_id=uid)
plan: ExecutionPlan = plan_resp.structured_output

if plan.respond_directly:
    return plan.direct_response
exec_resp = await runner.run(executor, format_plan_for_executor(plan),
                             session_id=sid, user_id=uid)
return exec_resp.content
```

---

## 3. ReAct (think-then-act)

```python
from orchestrator.agent.config import AgentConfig

agent = BaseAgent(
    name="react-agent",
    instructions="...",
    mcp_servers=[mcp],
    config=AgentConfig(react_mode=True),
)
```

`react_mode=True` injects a hidden `think` tool — the LLM must call it
before producing a final answer or calling another tool.

---

## 4. Self-improving via reflection

```python
from orchestrator.agent import create_reflection_agent

writer = BaseAgent(name="writer", instructions="Draft the email.")
reflective_writer = create_reflection_agent(
    name="reflective-writer", agent=writer, max_reflections=2,
)
resp = await runner.run(reflective_writer, "Email about Q4 results")
```

---

## 5. Multi-tenant isolation

```python
from orchestrator.agent.types import MemoryScope

agent = BaseAgent(
    name="tenant-aware",
    instructions="...",
    memory_config=AgentMemoryConfig(
        search_scope=MemoryScope.USER,
        store_scope=MemoryScope.USER,
    ),
)

# All memory access is automatically scoped to user_id
await runner.run(agent, "...", user_id=current_user.id)
```

For org-level isolation, register a custom scope:

```python
from orchestrator.memory import register_scope
register_scope(name="organization", required_field="org_id",
               description="Org-scoped memories")
```

---

## 6. FastAPI integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from orchestrator.core.lifecycle import OrchestratorLifecycle
from orchestrator.core.container import get_container

@asynccontextmanager
async def lifespan(app: FastAPI):
    lc = OrchestratorLifecycle(enable_signal_handlers=False)
    result = await lc.initialize()
    if not result.success:
        raise RuntimeError(f"Continuum init failed: {result.errors}")
    yield
    await lc.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(message: str, user_id: str = Depends(get_user_id)):
    runner = AgentRunner()
    resp = await runner.run(my_agent, message, user_id=user_id, session_id=user_id)
    return {"reply": resp.content, "tokens": resp.usage.total_tokens}
```

---

## 7. Structured output with validation

```python
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

agent = BaseAgent(
    name="sentiment",
    instructions="Classify the sentiment.",
    output_schema=Sentiment,
)
resp = await runner.run(agent, "I love this product!")
print(resp.structured_output.label, resp.structured_output.confidence)
```

---

## 8. Prompt-injection scanning

```python
from orchestrator.agent.config import AgentConfig

def my_scanner(text: str) -> tuple[str, bool, str]:
    if "ignore previous instructions" in text.lower():
        return text, False, "prompt_injection_attempt"
    return text, True, ""

agent = BaseAgent(
    name="safe-agent",
    instructions="...",
    config=AgentConfig(input_scanners=[my_scanner],
                       input_sanitization=True,
                       injection_detection=True),
)

# Blocked input raises orchestrator.exceptions.InputBlockedError
```

---

## 9. Custom container for tests

```python
from orchestrator.core.container import Container, ContainerConfig

class MockLLM:
    is_enabled = True
    async def chat(self, messages, **kw):
        return LLMResponse(content="mocked", model="test")
    async def chat_stream(self, messages, **kw):
        yield StreamChunk(content="mocked", is_finished=True)
    def count_tokens(self, messages, model=None): return 0

container = Container(ContainerConfig(auto_initialize=False))
container.set_llm_client(MockLLM())
container.set_memory_client(None)
container.set_session_client(None)

runner = AgentRunner(container=container)
resp = await runner.run(agent, "test")
```

---

## 10. Two-step handoff

```python
from orchestrator.agent.types import Handoff

triage = BaseAgent(
    name="triage",
    instructions="Route to the right specialist.",
    handoffs=[
        Handoff(target_agent="billing",   description="billing issues"),
        Handoff(target_agent="technical", description="technical support"),
    ],
)
billing   = BaseAgent(name="billing",   instructions="...")
technical = BaseAgent(name="technical", instructions="...")

runner = AgentRunner(agent_registry={
    "triage": triage, "billing": billing, "technical": technical,
})
resp = await runner.run(triage, "I want a refund for invoice 1234",
                        user_id="u1", session_id="s1")
print(resp.handoff_chain)            # ["triage", "billing"]
print(resp.content)
```

---

## 11. Disable everything but the LLM (offline test)

```python
from orchestrator.core.container import Container, ContainerConfig

container = Container(ContainerConfig(
    enable_memory=False, enable_session=False, enable_langfuse=False,
))
agent = BaseAgent(
    name="offline",
    instructions="...",
    memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
)
resp = await AgentRunner(container=container).run(agent, "...")
```

This skips Redis, Qdrant, mem0, and Langfuse entirely — useful for
unit tests when an API key isn't available.

---

## 12. Streaming to a websocket

```python
from orchestrator.agent.types import EventType

async def stream_to_ws(ws, agent, user_msg, user_id):
    runner = AgentRunner()
    async for ev in runner.run_stream(agent, user_msg, user_id=user_id):
        if ev.type == EventType.CONTENT_DELTA:
            await ws.send_text(ev.data["content"])
        elif ev.type == EventType.TOOL_CALL_START:
            await ws.send_json({"event": "tool_start",
                                "tool": ev.data["tool_name"]})
        elif ev.type == EventType.RUN_END:
            await ws.send_json({"event": "done"})
```
