#!/usr/bin/env python3
"""
Headroom × multi-agent — headless proof that the per-run compressor isolation
holds across Continuum's multi-agent execution boundaries.

This is the coverage gap the incident-desk rig (single-agent) and this
gateway-multi-agent-shop rig (built for the Smart Gateway, not Headroom) leave
open: does the CCR retrieve-authorization boundary
(`HeadroomCompressor._issued_hashes`) behave correctly when agents fan out
(parallel/scatter) or hand off?

Two tiers:

  TIER 1 — deterministic mechanism proof (default; needs only the sidecar).
    Exercises the REAL compressor / contextvar functions the runner uses
    (`enter_run_compressor`, `get_headroom_compressor`, `apply`,
    `resolve_retrieve`) in the exact task shapes the workflow engine uses:
      * parallel  = `asyncio.create_task` fan-out (parallel.py:136 / scatter.py:271)
      * handoff   = same-task direct `await` re-entry (executor.py:624 → :131)
    No LLM in the loop, so every assertion is provable, not probabilistic.

  TIER 2 — true end-to-end (opt-in: E2E=1; needs MCP server :8890 + LLM keys).
    Instruments `new_run_compressor` and drives the REAL ParallelAgent and the
    REAL handoff via the rig's create_workflow(...).chat(...), asserting the
    parallel run creates a distinct compressor per branch and the handoff run
    shares one across the chain.

PORT NOTE: the Headroom sidecar and the Smart Gateway both default to :8787.
This test is about Headroom, so do NOT run the gateway — leave SMART_GATEWAY_URL
unset (route direct to the provider) and give :8787 to the sidecar.

Run from this directory:

    # sidecar:  cd extensions/headroom && HEADROOM_CCR_BACKEND=memory \
    #           HEADROOM_OFFLINE=1 HF_HUB_OFFLINE=1 uv run headroom proxy --port 8787
    python headroom_multiagent_test.py           # tier 1
    E2E=1 python headroom_multiagent_test.py      # tier 1 + tier 2 (also start server.py)
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---- config guard: repo-root .env authoritative, Headroom FORCED on --------- #
_ROOT = Path(__file__).resolve().parents[2]  # repo root (playground/gateway-.../ -> continuum)
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env", override=True)
os.environ["HEADROOM_ENABLED"] = "true"  # the whole point of this rig

SIDECAR_BASE = os.environ.get("HEADROOM_API_BASE", "http://127.0.0.1:8787")
HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"  [SKIP] {name} — {why}")


def sidecar_up() -> bool:
    import httpx

    try:
        httpx.get(f"{SIDECAR_BASE}/health", timeout=3.0)
        return True
    except Exception:
        return False


# A read(config) → write(config) message list: the read is provably STALE (a
# write on the same path follows), so the sidecar's read-lifecycle transform
# MUST crush it to a marker and issue a CCR retrieve hash. Big content so the
# crush is unambiguous. Deterministic — no LLM decides anything here.
_BIG_CONFIG = "\n".join(
    f"service_{i}:\n  replicas: {i}\n  pool_max_size: {10 + i}\n  "
    f"timeout_ms: {1000 + i * 7}\n  region: us-east-{i % 4}\n  notes: "
    f"'auto-generated line {i} — filler to make the payload worth crushing'"
    for i in range(120)
)

STALE_READ_WRITE = [
    {"role": "user", "content": "Read cluster.yaml, then bump every pool size and save it."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "cluster.yaml"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_read", "content": _BIG_CONFIG},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_write",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"path": "cluster.yaml", "content": "pool_max_size: 50"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_write", "content": "Wrote 17 bytes to cluster.yaml."},
    {"role": "user", "content": "Confirm the new pool sizes."},
]


async def _issue_a_hash(comp) -> str | None:
    """Run a real compress via the given per-run compressor and return one
    freshly-issued CCR hash (or None if none was issued)."""
    await comp.apply(STALE_READ_WRITE, model="gpt-4o-mini")
    hashes = list(comp.issued_hashes)
    return hashes[0] if hashes else None


# --------------------------------------------------------------------------- #
# TIER 1 — deterministic mechanism proof
# --------------------------------------------------------------------------- #
async def t1_parallel_isolates_compressors() -> None:
    """The create_task fan-out that ParallelAgent/ScatterAgent use gives each
    branch its OWN per-run compressor (distinct instance + distinct issued-hash
    set). Uses the real enter_run_compressor / get_headroom_compressor."""
    from continuum.llm.headroom.compressor import (
        enter_run_compressor,
        exit_run_compressor,
        get_headroom_compressor,
    )

    captured: dict[int, object] = {}
    ready = asyncio.Barrier(2)  # force genuine interleave, like real branches

    async def branch(i: int) -> None:
        token = enter_run_compressor()  # exactly what runner.run does (runner.py:482)
        try:
            await ready.wait()
            captured[i] = get_headroom_compressor()
        finally:
            exit_run_compressor(token)

    # Fan out via create_task — the isolation-critical path (parallel.py:136).
    await asyncio.gather(
        asyncio.create_task(branch(0)),
        asyncio.create_task(branch(1)),
    )

    a, b = captured.get(0), captured.get(1)
    record(
        "T1 parallel: each branch got its OWN compressor instance",
        a is not None and b is not None and a is not b,
        f"id(A)={id(a) & 0xFFFF:#06x} id(B)={id(b) & 0xFFFF:#06x}",
    )
    record(
        "T1 parallel: branches have separate issued-hash sets",
        a is not None and b is not None and a.issued_hashes is not b.issued_hashes,
        "distinct _issued_hashes",
    )


async def t1_handoff_shares_compressor() -> None:
    """A handoff is a same-task direct await (executor.py:624), and the target's
    execute_loop re-enters use_run_compressor_if_enabled (executor.py:131). The
    nested enter must no-op (guard at compressor.py:436) so the target INHERITS
    the source's compressor and its pre-handoff issued hashes."""
    from continuum.llm.headroom.compressor import (
        enter_run_compressor,
        exit_run_compressor,
        get_headroom_compressor,
    )

    top = enter_run_compressor()  # top-level runner.run bind
    try:
        source_comp = get_headroom_compressor()
        # Handoff → target execute_loop re-entry, same task:
        nested = enter_run_compressor()  # executor.py:131 nested bind
        target_comp = get_headroom_compressor()
        try:
            record(
                "T1 handoff: nested bind no-ops (returns None token)",
                nested is None,
                "guard compressor.py:436",
            )
            record(
                "T1 handoff: target inherits the SAME compressor instance",
                source_comp is target_comp,
                f"id={id(source_comp) & 0xFFFF:#06x}",
            )
        finally:
            exit_run_compressor(nested)
    finally:
        exit_run_compressor(top)


async def t1_cross_run_hash_rejected() -> None:
    """THE security property: a hash issued in run A must NOT authorize a
    retrieve in a concurrent, unrelated run B. Real sidecar, no LLM."""
    from continuum.llm.headroom.compressor import (
        enter_run_compressor,
        exit_run_compressor,
        get_headroom_compressor,
    )

    if not sidecar_up():
        skip("T1 cross-run hash rejected", "sidecar down — needs real /v1/compress")
        return

    hash_a_box: dict[str, str | None] = {}

    async def run_a_then_wait(gate: asyncio.Event, done: asyncio.Event) -> None:
        token = enter_run_compressor()
        try:
            comp_a = get_headroom_compressor()
            hash_a_box["h"] = await _issue_a_hash(comp_a)
            # positive control: run A can retrieve its OWN hash
            if hash_a_box["h"]:
                served = await comp_a.resolve_retrieve(hash_a_box["h"])
                record(
                    "T1 same-run: run A retrieves its OWN issued hash",
                    "not issued" not in served and len(served) > 40,
                    f"{len(served)} chars restored",
                )
            gate.set()  # let run B try A's hash while A's compressor is still live
            await done.wait()
        finally:
            exit_run_compressor(token)

    async def run_b_tries_a(gate: asyncio.Event, done: asyncio.Event) -> None:
        token = enter_run_compressor()  # fresh context (separate task) → fresh compressor
        try:
            await gate.wait()
            h = hash_a_box.get("h")
            if not h:
                skip("T1 cross-run hash rejected", "sidecar issued no CCR hash for the payload")
                done.set()
                return
            comp_b = get_headroom_compressor()
            result = await comp_b.resolve_retrieve(h)
            record(
                "T1 cross-run: run B is DENIED run A's hash (anti-forgery)",
                "not issued" in result,
                f"hash={h[:12]}… rejected",
            )
            done.set()
        finally:
            exit_run_compressor(token)

    gate, done = asyncio.Event(), asyncio.Event()
    await asyncio.gather(
        asyncio.create_task(run_a_then_wait(gate, done)),
        asyncio.create_task(run_b_tries_a(gate, done)),
    )


async def t1_pre_handoff_hash_survives() -> None:
    """A CCR hash issued BEFORE a handoff must remain retrievable AFTER it,
    because the target shares the compressor. Real sidecar, no LLM."""
    from continuum.llm.headroom.compressor import (
        enter_run_compressor,
        exit_run_compressor,
        get_headroom_compressor,
    )

    if not sidecar_up():
        skip("T1 pre-handoff hash survives", "sidecar down — needs real /v1/compress")
        return

    top = enter_run_compressor()
    try:
        source_comp = get_headroom_compressor()
        h = await _issue_a_hash(source_comp)  # source agent compresses → issues hash
        if not h:
            skip("T1 pre-handoff hash survives", "sidecar issued no CCR hash for the payload")
            return
        # handoff → target execute_loop re-entry (same task, nested no-op bind):
        nested = enter_run_compressor()
        try:
            target_comp = get_headroom_compressor()
            served = await target_comp.resolve_retrieve(h)
            record(
                "T1 handoff: target retrieves a hash issued PRE-handoff",
                "not issued" not in served and len(served) > 40,
                f"{len(served)} chars restored post-handoff",
            )
        finally:
            exit_run_compressor(nested)
    finally:
        exit_run_compressor(top)


# --------------------------------------------------------------------------- #
# TIER 2 — true end-to-end (opt-in)
# --------------------------------------------------------------------------- #
def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_mcp_server() -> subprocess.Popen | None:
    if _port_open(8890):
        return None  # already running
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "server.py")],
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if _port_open(8890):
            return proc
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("MCP server on :8890 did not come up")


async def t2_real_workflows() -> None:
    """Drive the REAL ParallelAgent and REAL handoff via the rig, instrumenting
    new_run_compressor to see how many distinct per-run compressors each creates:
    parallel (direct execute) → one per branch; handoff (runner.run) → one shared."""
    from workflows import create_workflow  # rig harness

    import continuum.llm.headroom.compressor as hc

    created: list[int] = []
    real_new = hc.new_run_compressor

    def spy() -> object:
        comp = real_new()
        created.append(id(comp))
        return comp

    async def run_mode(mode: str, query: str) -> tuple[int, str]:
        created.clear()
        hc.new_run_compressor = spy  # type: ignore[assignment]
        # runner.run imports the symbol at call time
        # (from continuum.llm.headroom.compressor import ...), so patching the
        # source module is enough.
        try:
            wf = create_workflow(mode)
            await wf.initialize()
            reply = await wf.chat(query, user_id="e2e", conversation_id=f"e2e-{mode}")
        finally:
            hc.new_run_compressor = real_new  # type: ignore[assignment]
        return len(set(created)), reply

    n_par, par_reply = await run_mode("parallel", "Find good food for dogs and for cats.")
    record(
        "T2 parallel: real ParallelAgent created >=2 distinct compressors (per-branch)",
        n_par >= 2,
        f"{n_par} distinct per-run compressors; reply {len(par_reply)} chars",
    )

    n_ho, ho_reply = await run_mode("handoff", "Find a cat toy under $15 and add it to my cart.")
    record(
        "T2 handoff: real handoff shared ONE compressor across the chain",
        n_ho == 1,
        f"{n_ho} distinct per-run compressor(s); reply {len(ho_reply)} chars",
    )


# --------------------------------------------------------------------------- #
async def main() -> int:
    up = sidecar_up()
    print(f"{'✓ sidecar healthy' if up else '⚠️  sidecar DOWN'} at {SIDECAR_BASE}")
    if not up:
        print(
            "   Restart: cd extensions/headroom && HEADROOM_CCR_BACKEND=memory "
            "HEADROOM_OFFLINE=1 HF_HUB_OFFLINE=1 uv run headroom proxy --port 8787"
        )

    print("\n── TIER 1 — deterministic mechanism proof ──")
    await t1_parallel_isolates_compressors()
    await t1_handoff_shares_compressor()
    await t1_cross_run_hash_rejected()
    await t1_pre_handoff_hash_survives()

    if os.environ.get("E2E") == "1":
        print("\n── TIER 2 — true end-to-end (real workflows) ──")
        server = start_mcp_server()
        try:
            await t2_real_workflows()
        except Exception as e:  # noqa: BLE001
            record("T2 real workflows", False, f"error: {type(e).__name__}: {e}")
        finally:
            if server is not None:
                server.terminate()
    else:
        print("\n── TIER 2 skipped (set E2E=1 + start server.py to run real workflows) ──")

    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for s, _, _ in RESULTS if s == "SKIP")
    print(f"\n{'=' * 60}\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n{'=' * 60}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
