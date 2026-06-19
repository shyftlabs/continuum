"""
Shared decision-trace capture helpers.

Both the non-streaming executor (``execution/executor.py``) and the streaming run
loop (``AgentRunner.run_stream``) record the same decision steps and message
checkpoints, so a streamed run produces the *same* trace — and is equally
forkable — as a non-streamed one. This module is the single definition of that
capture so the two paths can't drift apart.

* :func:`capture_snapshot` — deep-copy the exact messages sent this turn (the
  fork resume point), gated on a checkpoint-enabled recorder.
* :func:`record_llm_turn` — record this turn's LLM decision step, returning the
  step id to nest the turn's tool/reasoning/handoff steps under.
* :func:`record_tool_steps` — one ``TOOL_CALL`` step per executed tool, paired
  with its result by ``tool_call_id``.
"""

from __future__ import annotations

import copy
from typing import Any


def capture_snapshot(recorder: Any, llm_messages: list[Any]) -> list[dict[str, Any]] | None:
    """Deep-copy the exact messages sent to the model this turn — the true resume
    point for :meth:`AgentRunner.fork`.

    Returns ``None`` when there is no recorder or checkpoints are disabled. The
    deep copy is required because ``message_to_dict`` returns dicts by reference,
    so a later in-place edit of a message would otherwise corrupt the stored
    snapshot (and any fork replayed from it). Must be called BEFORE this turn's
    assistant message is appended, or the snapshot would contain the turn's own
    answer and a fork would replay an already-finished conversation.
    """
    if recorder is None or not getattr(recorder, "checkpoint", False):
        return None
    from continuum.agent.utils.message_utils import message_to_dict

    return copy.deepcopy([message_to_dict(m) for m in llm_messages])


def record_llm_turn(
    recorder: Any,
    agent_name: str,
    turn: int,
    *,
    content: str | None,
    has_tool_calls: bool,
    usage: Any,
    snapshot: list[dict[str, Any]] | None,
    agent_stack: list[str] | None,
    data_labels: set[str] | list[str] | None = None,
) -> str | None:
    """Record this turn's LLM decision step; returns its id so the turn's tool /
    reasoning / handoff steps can nest under it. No-op (``None``) without a
    recorder. ``usage`` may be ``None`` (e.g. streaming) → zero token counts.

    ``data_labels`` is the run's taint active at this step — stored so fork()
    can seed the resumed run's context and keep gating it the same way.
    """
    if recorder is None:
        return None
    return recorder.record_llm_call(
        agent_name,
        turn,
        output=content or "",
        decision="tool_call" if has_tool_calls else "final_answer",
        prompt_tokens=(usage.prompt_tokens or 0) if usage else 0,
        completion_tokens=(usage.completion_tokens or 0) if usage else 0,
        total_tokens=(usage.total_tokens or 0) if usage else 0,
        messages_snapshot=snapshot,
        agent_stack=agent_stack,
        data_labels=data_labels,
    )


def step_event_payload(step: Any) -> dict[str, Any]:
    """Compact, serializable summary of a recorded step for a ``DECISION_STEP``
    streaming event (S2) — so a client can watch the trace build live."""
    return {
        "step_id": step.step_id,
        "kind": step.kind.value,
        "agent_name": step.agent_name,
        "decision": step.decision,
        "output_preview": (str(step.output)[:200] if step.output else None),
        "forkable": bool(step.messages_snapshot),
        "agent_stack": list(step.agent_stack or []),
    }


def latest_step_payload(recorder: Any) -> dict[str, Any] | None:
    """Payload for the most recently recorded step, or ``None`` when there's no
    recorder or no steps yet. Call right after a ``record_*`` to emit its event."""
    if recorder is None:
        return None
    steps = recorder.trace.steps
    if not steps:
        return None
    return step_event_payload(steps[-1])


def record_tool_steps(
    recorder: Any,
    agent_name: str,
    turn: int,
    tool_calls: list[Any],
    tool_results: list[dict[str, Any]],
    parent_id: str | None,
    agent_stack: list[str] | None = None,
) -> None:
    """Record one TOOL_CALL decision step per executed tool, pairing each call
    with its result by ``tool_call_id`` (best-effort)."""
    import json as _json

    results_by_id: dict[str, Any] = {}
    for r in tool_results:
        rid = r.get("tool_call_id") if isinstance(r, dict) else None
        if rid:
            results_by_id[rid] = r.get("content")
    for tc in tool_calls:
        name = (
            tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
        )
        raw_args = (
            tc.function.arguments
            if hasattr(tc, "function")
            else tc.get("function", {}).get("arguments", "{}")
        )
        try:
            args = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = raw_args
        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
        recorder.record_tool_call(
            agent_name,
            turn,
            name,
            args,
            results_by_id.get(tc_id),
            parent_id=parent_id,
            agent_stack=agent_stack,
        )
