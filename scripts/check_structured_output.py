"""
Live-provider acceptance check for structured output (on-demand / pre-release).

This is the REAL-provider counterpart to the mocked, every-PR CI matrix in
``tests/unit/llm/test_provider_matrix_structured_output.py``. The unit matrix
proves the parsing/retry LOGIC across provider quirks with fakes (free, no
keys); this script actually calls OpenAI, Anthropic and Gemini to prove the
end-to-end behavior still holds against the live APIs. It is deliberately NOT a
pytest test and NOT wired into CI — run it by hand before a release (or when a
provider SDK / model changes) so it never adds API cost or flaky-CI noise to the
normal pipeline.

It reproduces the two original structured-output bug reports:

    class Review(BaseModel):
        sentiment: str
        score: float
        summary: str

    agent = BaseAgent(name="reviewer", instructions=..., output_schema=Review)
    response = await runner.run(agent, "The hotel was fantastic but expensive.")
    review: Review = response.structured_output   # must NOT be None

Bug #1 — "structured_output is always None":
    For each provider we assert response.structured_output is a populated Review
    (not None) and that review.score is a float.

Bug #2 — "unreliable across the 100+ models":
    We run the SAME prompt + schema against OpenAI, Anthropic and Gemini and
    report, per provider, whether the parse succeeded — so a model that behaves
    differently is visible at a glance.

Both code paths are exercised (they are NOT the same in the framework):
    * runner.run()        -> non-streaming; through executor.execute_loop (the
                             path the docs show).
    * runner.run_stream() -> streaming; structured_output arrives on the final
                             event as a serialized dict.

Run it
------
    # from repo root, with .env containing your provider keys:
    python scripts/check_structured_output.py

    # limit to specific models:
    python scripts/check_structured_output.py gpt-4o-mini "gemini/gemini-2.5-flash"

A model with no API key in .env is reported as SKIPPED, not failed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel

# --- Load .env from the repo root --------------------------------------------
# scripts/check_structured_output.py -> parents[1] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _REPO_ROOT / ".env"
try:
    from dotenv import dotenv_values, load_dotenv

    load_dotenv(_ENV_PATH, override=True)

    # load_dotenv(override=True) sets vars that ARE in .env, but it does NOT clear
    # vars that are absent/commented in .env yet still live in os.environ as stale
    # shell exports (e.g. a `export SMART_GATEWAY_URL=...` from an earlier session).
    # Those stale exports would silently route every call through the Smart Gateway
    # — which is NOT what we want here: this check must hit the providers directly.
    # So: for the gateway vars, if .env does not define them, drop any stale export.
    _file_env = dotenv_values(_ENV_PATH)
    for _var in (
        "SMART_GATEWAY_URL",
        "SMART_GATEWAY_API_KEY",
        "SMART_GATEWAY_DEFAULT_MODE",
        "EMBEDDER_API_BASE",
        "EMBEDDER_API_KEY",
    ):
        if _var not in _file_env and _var in os.environ:
            print(f"  [setup] dropping stale shell export {_var} (not set in .env)")
            os.environ.pop(_var, None)
except Exception:  # dotenv is optional; keys may already be in the environment
    pass

from continuum import (  # noqa: E402  (must import after .env setup above)
    AgentConfig,
    AgentMemoryConfig,
    AgentRunner,
    BaseAgent,
    RunContext,
    RunnerConfig,
)
from continuum.agent.types import EventType, generate_run_id  # noqa: E402

# ---------------------------------------------------------------------------
# The schema + prompt, copied verbatim from the docs example.
# ---------------------------------------------------------------------------


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


PROMPT = "The hotel was fantastic but expensive."
INSTRUCTIONS = "Analyze the review and return structured output."

# Default models — one per provider — to expose any cross-provider differences.
# Override by passing model names on the command line.
DEFAULT_MODELS = [
    "gpt-4o-mini",  # OpenAI
    "claude-haiku-4-5-20251001",  # Anthropic (full API id; "claude-haiku-4.5" 404s)
    "gemini/gemini-2.5-flash",  # Google Gemini
]

# Memory + session disabled so the script runs with no DB/session setup. This
# is unrelated to structured output — it only keeps the check self-contained.
_NO_MEMORY = AgentMemoryConfig(search_memories=False, store_memories=False)
_NO_SESSION = AgentConfig(log_to_session=False, input_sanitization=False)


def _make_agent(model: str) -> BaseAgent:
    # Faithful to the docs: the ONLY structured-output knob set is output_schema.
    # We deliberately do NOT set enable_json_mode — the docs never mention it, and
    # bug #1 was that the parse used to require it.
    return BaseAgent(
        name="reviewer",
        instructions=INSTRUCTIONS,
        model=model,
        output_schema=Review,
        memory_config=_NO_MEMORY,
        config=_NO_SESSION,
    )


def _verdict(label: str, ok: bool, detail: str) -> str:
    mark = "✅ PASS" if ok else "❌ FAIL"
    return f"    {mark}  {label:<14} {detail}"


def _describe(obj: object) -> str:
    if obj is None:
        return "structured_output = None"
    if isinstance(obj, Review):
        return (
            f"Review(sentiment={obj.sentiment!r}, score={obj.score}, summary={obj.summary[:40]!r})"
        )
    return f"{type(obj).__name__} (NOT a Review!) -> {obj!r}"


async def _check_non_streaming(runner: AgentRunner, model: str) -> bool:
    """Doc path: runner.run() should return a populated Review."""
    agent = _make_agent(model)
    ctx = RunContext(run_id=generate_run_id(), session_id=None)
    resp = await runner.run(agent=agent, input=PROMPT, context=ctx)

    so = resp.structured_output
    print(f"    raw content : {(resp.content or '')[:80]!r}")
    print(f"    parsed      : {_describe(so)}")

    if so is None:
        return False  # bug #1: arrived as text but never parsed
    if not isinstance(so, Review):
        return False  # bug #2: parsed into the wrong shape
    # The docs' final line: review.score must be usable as a float.
    return isinstance(so.score, float)


async def _check_streaming(runner: AgentRunner, model: str) -> bool:
    """Streaming path: does any event expose a parsed structured object?"""
    agent = _make_agent(model)
    # NOTE: run_stream() does NOT accept a `context=` kwarg (unlike run()); it
    # builds its own run internally. Pass only agent + input.
    streamed = ""
    structured_from_stream: object = None
    async for event in runner.run_stream(agent=agent, input=PROMPT):
        if event.type == EventType.CONTENT_DELTA:
            streamed += (event.data or {}).get("delta", "") or ""
        # Look for a structured object on any event's data, however it's named.
        data = event.data or {}
        for key in ("structured_output", "structured", "parsed"):
            if data.get(key) is not None:
                structured_from_stream = data[key]

    # Streaming events are JSON-serialized for transport, so structured_output
    # arrives as a DICT (model_dump), not a Review instance. Rehydrate + validate
    # it against the schema — success = the dict matches Review.
    obj = structured_from_stream
    if isinstance(obj, dict):
        try:
            obj = Review.model_validate(obj)
        except Exception:
            pass

    print(f"    streamed    : {streamed[:80]!r}")
    print(f"    parsed      : {_describe(obj)}")
    return isinstance(obj, Review)


async def run_for_model(model: str) -> tuple[str, bool, bool, str | None]:
    """Returns (model, non_streaming_ok, streaming_ok, skip_reason)."""
    print(f"\n{'─' * 70}")
    print(f"  MODEL: {model}")
    print(f"{'─' * 70}")

    runner = AgentRunner(config=RunnerConfig(persist_state=False, default_max_turns=3))

    try:
        print("  [non-streaming]  runner.run()")
        ns_ok = await _check_non_streaming(runner, model)
        print(_verdict("run()", ns_ok, "structured_output populated as Review"))
    except Exception as exc:  # e.g. missing API key for this provider
        msg = str(exc)
        if "api" in msg.lower() and "key" in msg.lower():
            print(f"    ⏭️  SKIPPED — no API key for this provider ({msg[:60]})")
            return model, False, False, "no api key"
        print(f"    ❌ ERROR — {type(exc).__name__}: {msg[:120]}")
        return model, False, False, f"error: {type(exc).__name__}"

    try:
        print("  [streaming]      runner.run_stream()")
        s_ok = await _check_streaming(runner, model)
        print(_verdict("run_stream()", s_ok, "structured object exposed in stream"))
    except Exception as exc:
        print(f"    ❌ ERROR — {type(exc).__name__}: {str(exc)[:120]}")
        s_ok = False

    return model, ns_ok, s_ok, None


async def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS

    print("=" * 70)
    print("  STRUCTURED OUTPUT — live-provider acceptance (docs example)")
    print("=" * 70)
    print(f"  prompt : {PROMPT!r}")
    print("  schema : Review(sentiment: str, score: float, summary: str)")
    print(f"  models : {', '.join(models)}")

    results = [await run_for_model(m) for m in models]

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'model':<28} {'run()':<10} {'run_stream()':<14}")
    any_failed = False
    for model, ns_ok, s_ok, skip in results:
        if skip:
            print(f"  {model:<28} {'SKIPPED':<10} {'SKIPPED':<14} ({skip})")
        else:
            print(f"  {model:<28} {('✅' if ns_ok else '❌'):<10} {('✅' if s_ok else '❌'):<14}")
            if not (ns_ok and s_ok):
                any_failed = True
    print()
    print("  Legend: ✅ = structured_output is a populated Review")
    print("          ❌ = None or wrong shape (the reported bug)")

    # Non-zero exit on any real failure so this can gate a release step.
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
