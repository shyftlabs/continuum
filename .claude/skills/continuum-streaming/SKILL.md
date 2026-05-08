---
name: continuum-streaming
description: Stream tokens, tool calls, handoffs, and memory events out of a Continuum agent in real time using `runner.run_stream()` and the `EventType` enum. Invoke when the user asks "stream tokens to UI", "websocket chat", "live progress", "see tool execution as it happens", or anything that needs token-by-token output.
---

# Continuum Streaming Skill

Authoritative sources: [`docs/agent.md`](../../../docs/agent.md) §3 and
the `EventType` enum in `orchestrator/agent/types.py`.

---

## Imports

```python
from orchestrator.agent import AgentRunner
from orchestrator.agent.types import EventType, AgentEvent
```

---

## Smallest possible stream

```python
runner = AgentRunner()

async for ev in runner.run_stream(agent, "Tell me a story", user_id="u1", session_id="s1"):
    if ev.type == EventType.CONTENT_DELTA:
        print(ev.data["content"], end="", flush=True)
```

`run_stream()` returns an `AsyncIterator[AgentEvent]`. Every event
carries: `type: EventType`, `agent_name: str`, `run_id: str`,
`data: dict`, `timestamp`, `trace_id`, `span_id`.

---

## Full event reference

| `EventType` | Fires | `event.data` keys |
|---|---|---|
| `RUN_START` | Run begins | `agent_name`, `input_preview` |
| `RUN_END` | Run completes | `status`, `latency_ms`, `usage` |
| `RUN_ERROR` | Run fails | `error`, `error_type` |
| `AGENT_START` | Each agent (incl. handoff target) starts | `agent_name` |
| `AGENT_END` | Each agent ends | `agent_name`, `status` |
| `CONTENT_DELTA` | LLM token chunks | `content` (partial text) |
| `CONTENT_COMPLETE` | LLM emits a full assistant message | `content` |
| `TOOL_CALL_START` | A tool is about to run | `tool_name`, `arguments` |
| `TOOL_CALL_END` | Tool returned successfully | `tool_name`, `result` |
| `TOOL_CALL_ERROR` | Tool raised | `tool_name`, `error` |
| `HANDOFF_START` | Source agent invoking the handoff tool | `from_agent`, `to_agent`, `reason` |
| `HANDOFF_END` | Target agent finished | `from_agent`, `to_agent` |
| `HANDOFF_RETURN` | Control returned to source (`return_to_parent=True`) | `from_agent`, `to_agent` |
| `MEMORY_RETRIEVAL` | Long-term memories injected into prompt | `count`, `query` |
| `MEMORY_STORAGE` | New memories stored after the turn | `count` |
| `WORKFLOW_STEP` | Workflow agent advanced | `step`, `agent_name` |
| `LOOP_ITERATION` | LoopAgent completed an iteration | `iteration`, `output` |

---

## Common patterns

### Plain console with tool indicators

```python
async for ev in runner.run_stream(agent, "..."):
    if ev.type == EventType.CONTENT_DELTA:
        print(ev.data["content"], end="", flush=True)
    elif ev.type == EventType.TOOL_CALL_START:
        print(f"\n[tool: {ev.data['tool_name']} ...]", flush=True)
    elif ev.type == EventType.TOOL_CALL_END:
        print(" ✓", flush=True)
    elif ev.type == EventType.RUN_END:
        print()
```

### Stream to a websocket

```python
async def stream_to_ws(ws, agent, user_msg, user_id):
    async for ev in AgentRunner().run_stream(agent, user_msg, user_id=user_id):
        if ev.type == EventType.CONTENT_DELTA:
            await ws.send_text(ev.data["content"])
        elif ev.type == EventType.TOOL_CALL_START:
            await ws.send_json({"event": "tool_start", "tool": ev.data["tool_name"]})
        elif ev.type == EventType.TOOL_CALL_END:
            await ws.send_json({"event": "tool_end", "tool": ev.data["tool_name"]})
        elif ev.type == EventType.HANDOFF_START:
            await ws.send_json({"event": "handoff",
                                "from": ev.data["from_agent"],
                                "to":   ev.data["to_agent"]})
        elif ev.type == EventType.RUN_END:
            await ws.send_json({"event": "done",
                                "usage": ev.data.get("usage", {})})
```

### Stream as Server-Sent Events (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(message: str, user_id: str):
    async def gen():
        async for ev in AgentRunner().run_stream(agent, message, user_id=user_id):
            if ev.type == EventType.CONTENT_DELTA:
                yield f"data: {ev.data['content']}\n\n"
            elif ev.type == EventType.RUN_END:
                yield "event: done\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### Collecting full content from a stream

```python
buf = []
async for ev in runner.run_stream(agent, "..."):
    if ev.type == EventType.CONTENT_DELTA:
        buf.append(ev.data["content"])
content = "".join(buf)
```

---

## Behaviour notes

- During a tool call, `CONTENT_DELTA` pauses. The next chunks will fire
  after `TOOL_CALL_END` — when the LLM resumes generation.
- During a handoff, the target agent emits its own `AGENT_START` and
  `CONTENT_DELTA` events under the same `run_id`. Use `agent_name` on
  the event (or your own state) to attribute output.
- Streaming respects context-window compression — by the time
  `CONTENT_DELTA` fires, the prompt has already been compressed if
  needed.

---

## Don't

- Don't expect `await runner.run_stream(agent, ...)` — `run_stream`
  is an async generator. Use `async for ev in runner.run_stream(...)`.
- Don't accumulate `RUN_END.data["usage"]` from individual deltas — it's
  reported once at the end.
- Don't write to the websocket inside `CONTENT_DELTA` without a flush
  on every chunk; ws frames buffer otherwise.
- Don't rely on `event.timestamp` ordering across events from different
  agents in a workflow — they may interleave.
