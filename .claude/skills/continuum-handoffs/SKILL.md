---
name: continuum-handoffs
description: Build agent-to-agent transitions with Continuum's `Handoff` system — triage routing, history summarization modes (FULL/SUMMARY/RECENT_N/HYBRID), cycle detection, depth tracking, return-to-parent. Invoke when the user asks "route customer requests to specialists", "agent that can transfer to another", "summarize history before handing off", or anything multi-agent that's a *transition* (not a *workflow*).
---

# Continuum Handoffs Skill

Authoritative source: [`docs/agent.md`](../../../docs/agent.md), §7.

> Handoffs are agent-to-agent **transitions** ("triage → billing"),
> distinct from workflow agents which compose multiple agents into one
> (Sequential, Parallel, etc.). Use a `RouterAgent` for stateless
> routing; use handoffs when the source agent should be able to decide
> at any turn that the target should take over.

---

## Imports

```python
from orchestrator.agent import BaseAgent, AgentRunner, Handoff
from orchestrator.agent.types import HistorySummarizationMode, HandoffData, HandoffResult
from orchestrator.agent.exceptions import (
    HandoffNotAllowedError, HandoffDepthExceededError,
    HandoffTargetNotFoundError,
)
# HandoffCycleDetectedError is NOT re-exported from `orchestrator.agent`:
from orchestrator.agent.exceptions import HandoffCycleDetectedError
```

---

## Define a triage agent

```python
billing   = BaseAgent(name="billing",   instructions="Help with invoices and refunds.")
technical = BaseAgent(name="technical", instructions="Help with bugs and outages.")

triage = BaseAgent(
    name="triage",
    instructions="Route the customer to the right specialist.",
    handoffs=[
        Handoff(target_agent="billing",
                description="Billing, payments, refunds, invoices"),
        Handoff(target_agent="technical",
                description="Bugs, errors, outages, integration issues"),
    ],
)

runner = AgentRunner(agent_registry={
    "triage": triage, "billing": billing, "technical": technical,
})
resp = await runner.run(triage, "I want a refund for invoice 1234",
                        user_id="u1", session_id="s1")
print(resp.handoff_chain)            # ["triage", "billing"]
print(resp.content)                  # the billing agent's reply
```

---

## `Handoff` fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `target_agent` | `str` | required | Name of the target agent |
| `description` | `str` | required | When this handoff should fire — read by the LLM |
| `condition` | `str \| Callable \| None` | `None` | Extra guard (rarely used) |
| `transfer_history` | `bool` | `True` | Pass conversation history to the target |
| `summarize_history` | `bool` | `True` | Run an LLM summarization before transferring |
| `summarization_mode` | `HistorySummarizationMode` | `HYBRID` | See table below |
| `recent_messages` | `int` | `5` | Used by `RECENT_N` and `HYBRID` |
| `return_to_parent` | `bool` | `True` | After the target finishes, control returns to the source |

---

## History summarization modes

| Mode | Behaviour |
|---|---|
| `FULL` | Pass the entire conversation verbatim — no summarization |
| `SUMMARY` | Replace history with a single LLM-generated summary |
| `RECENT_N` | Pass only the last `recent_messages` messages |
| `HYBRID` *(default)* | LLM summary + last `recent_messages` raw messages |

```python
from orchestrator.agent.types import HistorySummarizationMode

Handoff(
    target_agent="billing",
    description="...",
    summarization_mode=HistorySummarizationMode.RECENT_N,
    recent_messages=8,
)
```

---

## Tool prefix (internal detail)

The framework injects a hidden tool per handoff named
**`handoff_to_<target>`** (NOT `transfer_to_…`). The LLM calls this
tool to trigger the transition; you never write the tool yourself.

`agent.is_handoff_tool_call(tool_name) -> tuple[bool, str | None]`
detects whether a tool call is actually a handoff invocation.

---

## Multi-level handoffs (depth + cycles)

`HandoffConfig.max_handoff_depth` defaults to `10`. Beyond that:
`HandoffDepthExceededError(current_depth, max_depth, ...)`.

Cycles (e.g. `A → B → A`) are detected before the call — they raise
`HandoffCycleDetectedError(from_agent, to_agent, agent_stack, ...)`.
Import path: `orchestrator.agent.exceptions` (not re-exported at the
package root).

---

## Inspecting the result

```python
resp = await runner.run(triage, "...")
resp.agents_used      # ["triage", "billing"]
resp.handoff_chain    # ["triage", "billing"]
resp.handoff          # HandoffData of the most recent handoff (or None)
resp.handoff_result   # HandoffResult with success / error / returned_to_parent
```

Streaming exposes `EventType.HANDOFF_START` / `HANDOFF_END` /
`HANDOFF_RETURN` events with `event.data = {"from_agent": ..., "to_agent": ..., "reason": ...}`.

---

## Driving handoffs manually with `HandoffManager`

```python
from orchestrator.agent import HandoffManager

mgr = HandoffManager(llm_client=runner.llm_client, tracing_manager=None, max_depth=10)

mgr.validate_handoff(from_agent=triage, to_agent_name="billing")
data = await mgr.prepare_handoff(
    from_agent=triage, to_agent=billing,
    reason="Refund request",
    messages=current_messages,
    handoff_config=triage.get_handoff("billing"),
)
```

Most apps don't need this — let the runner drive it.

---

## RouterAgent vs Handoff

| Use case | Pick |
|---|---|
| LLM picks ONE specialist for the whole turn | `RouterAgent` (`create_router_agent`) |
| Source agent might decide *during the conversation* to transfer | `Handoff` |
| Source agent should regain control after the specialist finishes | `Handoff` with `return_to_parent=True` |
| Stateless mapping of intent → agent | `RouterAgent` |

A `RouterAgent` never looks back; a handoff chain can. Use whichever
matches the conversation shape.

---

## Don't

- Don't forget to register the target via `runner.register_agent(...)`
  or `agent_registry={...}` — `HandoffTargetNotFoundError`.
- Don't reference the handoff tool by `transfer_to_<target>` — actual
  prefix is `handoff_to_<target>`.
- Don't try to import `HandoffCycleDetectedError` from
  `orchestrator.agent` — only from `orchestrator.agent.exceptions`.
- Don't crank `max_handoff_depth` to skip cycle errors — fix the routing
  logic instead. Cycles indicate either bad descriptions or a missing
  fallback agent.
- Don't pass async hooks for `on_handoff` — hooks are sync.
