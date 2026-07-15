#!/usr/bin/env python3
"""
Incident Desk — headless end-to-end proof of the Headroom integration.

Spawns the MCP tool server itself, runs every scenario against the REAL
sidecar (:8787) and a REAL LLM (gpt-4o-mini), and hard-asserts on ground
truth from data.py. Run from this directory:

    python e2e_test.py              # scenarios 1–8
    KOMPRESS=1 python e2e_test.py   # + scenario 9 (Kompress characterization)

Scenario map (what each one PROVES):
  1 DB rows      SmartCrusher lossless — correct answer FROM the compressed view
  2 Logs + CCR   lossy crush → model calls continuum_headroom_retrieve → needle answered
                 (a correct answer also proves the anti-doom-loop restore:
                 compression runs again between the retrieve and the answer)
  3 Search/RAG   SearchCompressor — right runbook cited from compressed results
  4 `read` tool  Headroom's exclusion list — payload passes through untouched
  5 Fail-open    dead sidecar → run still succeeds, zero compression
  6 Anti-forgery fabricated hash rejected without contacting the sidecar
  7 Streaming    scenario-2 shape through run_stream (runner interception path)
  8 Multi-tool   two crushed payloads in one run, both needles recovered
  9 Kompress     (opt-in) prose passes through cold, compresses ~6% once warm
 10 RAG context  the SAME bytes that scenario 3 crushes ~98% as a tool result
                 pass through UNCOMPRESSED when injected via Continuum's native
                 rag_context slot — because that's a system message and
                 Headroom protects system/developer messages by default
 11 File read    a read(service.yaml) made STALE by a later write(service.yaml)
                 is replaced by a marker + CCR hash (ReadLifecycleManager).
                 Deterministic: a crafted read→write message list is sent
                 straight to the sidecar's /v1/compress — no LLM in the loop,
                 so the STALE classification is provable, not probabilistic.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

import config  # noqa: F401  — MUST be first: .env guard + HEADROOM_ENABLED=true
from agent import IncidentAgent
from config import sidecar_health
from data import GROUND_TRUTH

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"  [SKIP] {name} — {why}")


def norm(s: str) -> str:
    return (s or "").replace(",", "").replace("$", "")


def show_box(box: dict) -> str:
    lc = box.get("last_call")
    core = (
        f"{lc['tokens_before']}→{lc['tokens_after']} tok ({lc['pct_saved']}% saved)"
        if lc
        else "no compression stats (fail-open?)"
    )
    d = box.get("sidecar_delta", {})
    return f"{core}; run removed {d.get('tokens_removed', 0)} tok over {d.get('llm_calls_compressed', 0)} calls"


# --------------------------------------------------------------------------- #

async def s1_db_lossless(agent: IncidentAgent) -> None:
    gt = GROUND_TRUTH["orders"]
    r = await agent.chat(
        "Query the orders database for failed orders. How many orders failed, "
        "and which single order has the largest refund? Give its order id and "
        "the exact refund amount."
    )
    ans = norm(r["response"])
    box = r["headroom"]
    print(f"    headroom: {show_box(box)}")
    print(f"    answer: {r['response'][:200]}")
    ok_count = str(gt["failed_count"]) in ans
    ok_max = gt["largest_refund_order"] in ans and "2499" in ans
    lc = box.get("last_call") or {}
    # % is of the WHOLE message list (system + user + tool payload), so the
    # bar is proportional to the payload's share of the context, not the
    # 50-60% Headroom achieves on the JSON block itself.
    ok_savings = (lc.get("pct_saved") or 0) >= 15
    record("1 DB SmartCrusher: correct count from compressed view", ok_count, f"expect {gt['failed_count']}")
    record("1 DB SmartCrusher: correct max-refund extraction", ok_max, f"expect {gt['largest_refund_order']} / 2499.00")
    record("1 DB SmartCrusher: ≥15% saved on the tool-result call", ok_savings, f"got {lc.get('pct_saved')}%")


async def s2_logs_ccr(agent: IncidentAgent) -> None:
    token = GROUND_TRUTH["tokens"]["checkout-api"]
    r = await agent.chat(
        "Fetch the checkout-api logs and tell me the exact incident reference "
        "token recorded in the audit trail."
    )
    box = r["headroom"]
    print(f"    headroom: {show_box(box)}")
    print(f"    retrieves: {r['retrieve_calls']}  new_hashes: {len(box['new_hashes'])}")
    print(f"    answer: {r['response'][:200]}")
    record("2 CCR: sidecar issued retrieve marker(s)", len(box["new_hashes"]) >= 1)
    record("2 CCR: model called continuum_headroom_retrieve", len(r["retrieve_calls"]) >= 1)
    big = any(rc["chars"] > 50_000 for rc in r["retrieve_calls"])
    record("2 CCR: retrieve returned the full original (>50k chars)", big,
           str([rc["chars"] for rc in r["retrieve_calls"]]))
    record("2 CCR: needle answered (anti-doom-loop held)", token in r["response"], f"expect {token}")


async def s3_search(agent: IncidentAgent) -> None:
    r = await agent.chat(
        "Search the runbooks for guidance on database connection pool "
        "exhaustion. Which runbook applies? Give its ID."
    )
    box = r["headroom"]
    lc = box.get("last_call") or {}
    print(f"    headroom: {show_box(box)}")
    print(f"    transforms: {lc.get('transforms')}")
    print(f"    answer: {r['response'][:200]}")
    record("3 Search: right runbook from compressed results",
           GROUND_TRUTH["runbook_id"] in r["response"])
    record("3 Search: results actually compressed", (lc.get("pct_saved") or 0) > 0,
           f"got {lc.get('pct_saved')}% — payload too small for the floor?")


async def s4_read_excluded(agent: IncidentAgent) -> None:
    r = await agent.chat(
        "Use the read tool to read service.yaml, then report the exact values "
        "of database.pool_max_size and workers.refund_worker_concurrency."
    )
    box = r["headroom"]
    lc = box.get("last_call") or {}
    transforms = lc.get("transforms", [])
    print(f"    transforms: {transforms}")
    print(f"    answer: {r['response'][:200]}")
    ans = norm(r["response"])
    record("4 read-exclusion: config values reported correctly",
           "20" in ans and "7" in ans)
    excluded = any("exclude" in t for t in transforms)
    record("4 read-exclusion: router marked the payload excluded", excluded,
           f"transforms={transforms}")


async def s5_fail_open(agent: IncidentAgent) -> None:
    agent.point_sidecar("http://127.0.0.1:9")  # nothing listens here
    try:
        r = await agent.chat(
            "Query the orders database for failed orders and report just the count."
        )
        ans = norm(r["response"])
        box = r["headroom"]
        print(f"    answer: {r['response'][:160]}")
        record("5 fail-open: run succeeded with sidecar dead",
               str(GROUND_TRUTH["orders"]["failed_count"]) in ans)
        record("5 fail-open: nothing compressed", box.get("last_call") is None)
    finally:
        agent.point_sidecar(None)


async def s6_anti_forgery(agent: IncidentAgent) -> None:
    result = await agent.forge_retrieve("f" * 24)
    print(f"    result: {result[:160]}")
    record("6 anti-forgery: fabricated hash rejected", "not issued" in result)


async def s7_streaming(agent: IncidentAgent) -> None:
    token = GROUND_TRUTH["tokens"]["payments-svc"]
    events: list[str] = []
    done: dict = {}
    async for ev in agent.chat_stream(
        "Fetch the payments-svc logs and tell me the exact incident reference "
        "token recorded in the audit trail."
    ):
        events.append(ev["type"])
        if ev["type"] == "done":
            done = ev
    box = done.get("headroom", {})
    print(f"    events: {len(events)} ({', '.join(sorted(set(events)))})")
    print(f"    headroom: {show_box(box)}")
    print(f"    answer: {done.get('response', '')[:200]}")
    record("7 streaming: live events emitted", "token" in events or "message" in events)
    record("7 streaming: needle answered via runner interception",
           token in done.get("response", ""), f"expect {token}")


async def s8_multi_payload(agent: IncidentAgent) -> None:
    t1 = GROUND_TRUTH["tokens"]["checkout-api"]
    t2 = GROUND_TRUTH["tokens"]["payments-svc"]
    r = await agent.chat(
        "Fetch the logs for BOTH checkout-api and payments-svc. Then report "
        "the exact incident reference token recorded in each service's audit trail."
    )
    print(f"    headroom: {show_box(r['headroom'])}")
    print(f"    retrieves: {r['retrieve_calls']}")
    print(f"    answer: {r['response'][:300]}")
    record("8 multi-payload: retrieve used", len(r["retrieve_calls"]) >= 1)
    record("8 multi-payload: checkout-api needle", t1 in r["response"], f"expect {t1}")
    record("8 multi-payload: payments-svc needle", t2 in r["response"], f"expect {t2}")


def _kompress_fired(transforms: list[str]) -> bool:
    """Warm Kompress shows up as `router:text:<ratio>` with a compressive
    ratio (probed live on v0.29.0 — the label is NOT 'kompress'). A cold
    model is a pure passthrough and emits NO router entry for the block."""
    for t in transforms:
        if t.startswith("router:text:"):
            try:
                return float(t.rsplit(":", 1)[1]) < 0.95
            except ValueError:
                return True
    return False


async def s9_kompress(agent: IncidentAgent) -> None:
    if os.environ.get("KOMPRESS") != "1":
        skip("9 Kompress", "opt-in only — rerun with KOMPRESS=1")
        return
    question = (
        "Fetch the postmortem for INC-2417 and state the exact total number "
        "of checkout attempts that failed."
    )
    first_transforms: list[str] | None = None
    for attempt in range(10):
        r = await agent.chat(question)
        lc = (r["headroom"].get("last_call") or {})
        transforms = lc.get("transforms", [])
        if first_transforms is None:
            first_transforms = transforms
        if _kompress_fired(transforms):
            fact_ok = "4127" in norm(r["response"])
            print(f"    warm after {attempt + 1} attempt(s): {show_box(r['headroom'])}")
            print(f"    answer: {r['response'][:200]}")
            record("9 Kompress: prose routed to ML once warm", True, str(transforms))
            record("9 Kompress: impact figure still answered", fact_ok)
            # cold-phase evidence, when this run happened to observe it
            if attempt > 0 and not _kompress_fired(first_transforms):
                record("9 Kompress: cold start passed through (warm-gate)", True,
                       str(first_transforms))
            return
        print(f"    attempt {attempt + 1}: not routed yet (transforms={transforms}); warming…")
        await asyncio.sleep(10)
    skip("9 Kompress", f"never routed after 10 attempts (first transforms={first_transforms}) — "
         "ML extra not installed on this sidecar, or model failed to load")


def _rag_block(msgs: list[dict] | None) -> dict | None:
    """The rag_context system message Continuum injects at position 7."""
    for m in msgs or []:
        if m.get("role") == "system" and "PROVIDED CONTEXT" in (m.get("content") or ""):
            return m
    return None


async def s10_rag_context_protected(agent: IncidentAgent) -> None:
    from data import format_runbook_results

    # Identical generator to search_runbooks (scenario 3): grep path:line:text.
    # As a tool result it crushes ~98%; here it rides in rag_context instead.
    payload = format_runbook_results("database connection pool exhaustion")
    r = await agent.chat_with_rag_context(
        "Using ONLY the provided context, which runbook covers database "
        "connection pool exhaustion? Give its ID.",
        rag_context=payload,
    )
    box = r["headroom"]
    lc = box.get("last_call") or {}
    transforms = lc.get("transforms", [])
    rb = _rag_block(box.get("messages_before"))
    ra = _rag_block(box.get("messages_after"))
    print(f"    transforms: {transforms}")
    print(
        f"    rag block: before={rb and rb.get('content_len')} "
        f"after={ra and ra.get('content_len')} chars  (tools_called={r['tools_called']})"
    )
    print(f"    answer: {r['response'][:200]}")

    record("10 rag_context: payload injected at position 7 (system msg)", rb is not None)
    # The discriminator: same bytes in and out => Headroom did NOT compress it.
    protected = rb is not None and ra is not None and rb.get("content") == ra.get("content")
    record(
        "10 rag_context: system-message payload passed through UNCOMPRESSED",
        protected,
        "before/after bytes identical" if protected else "block was altered — unexpected",
    )
    record(
        "10 rag_context: router marked it protected:system",
        any("protected:system" in t for t in transforms),
        f"transforms={transforms}",
    )
    record(
        "10 rag_context: answered the right runbook from provided context",
        GROUND_TRUTH["runbook_id"] in r["response"],
        f"expect {GROUND_TRUTH['runbook_id']}",
    )


async def s11_file_read_stale(agent: IncidentAgent) -> None:
    """File-read lifecycle: a Read made STALE by a later Write is compressed.

    Unlike every other scenario this one does NOT go through the LLM — the
    STALE classification depends only on the shape of the message list, so we
    craft that list by hand (read service.yaml → write service.yaml) and send
    it straight to the sidecar. That makes the result deterministic: the read
    is provably stale (a write on the same path follows it), so if the
    lifecycle manager is wired and enabled, it MUST fire.
    """
    from config import SIDECAR_BASE
    from continuum.llm.headroom.client import HeadroomClient
    from data import SERVICE_YAML

    if not sidecar_health()["up"]:
        skip("11 file-read STALE", "sidecar down — lifecycle needs the real /v1/compress")
        return

    # read(service.yaml) → write(service.yaml): the read is now STALE.
    messages = [
        {"role": "user", "content": "Read service.yaml, then raise the DB pool size and save it."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_read", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "service.yaml"}'}}]},
        {"role": "tool", "tool_call_id": "call_read", "content": SERVICE_YAML},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_write", "type": "function",
             "function": {"name": "write",
                          "arguments": '{"path": "service.yaml", "content": "pool_max_size: 50"}'}}]},
        {"role": "tool", "tool_call_id": "call_write", "content": "Wrote 17 bytes to service.yaml. Change applied."},
        {"role": "user", "content": "Confirm the pool size is now 50."},
    ]

    client = HeadroomClient(api_base=SIDECAR_BASE)
    try:
        compressed, stats, hashes = await client.compress(messages, model="gpt-4o-mini")
    finally:
        await client.aclose()

    transforms = stats.transforms_applied
    read_msg = next((m for m in compressed if m.get("tool_call_id") == "call_read"), None)
    new_content = (read_msg or {}).get("content", "") or ""
    print(f"    transforms: {transforms}")
    print(f"    read result: {len(SERVICE_YAML)} chars → {len(new_content)} chars")
    print(f"    marker: {new_content[:140]}")

    record("11 file-read STALE: read classified stale by later write",
           any("read_lifecycle:stale" in t for t in transforms),
           f"transforms={transforms}")
    record("11 file-read STALE: stale read replaced by a shorter marker",
           len(new_content) < len(SERVICE_YAML),
           f"{len(SERVICE_YAML)} → {len(new_content)} chars")
    record("11 file-read STALE: marker carries a CCR retrieve hash",
           "Retrieve original: hash=" in new_content and len(hashes) >= 1,
           f"hashes={hashes}")


# --------------------------------------------------------------------------- #

def start_mcp_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "server.py")],
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", 8921)) == 0:
                return proc
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("MCP server on :8921 did not come up")


async def main() -> int:
    health = sidecar_health()
    if not health["up"]:
        print(f"⚠️  Headroom sidecar is DOWN ({health['error']}).")
        print(f"    Restart it: {health['restart']}")
        print("    Continuing — everything will run fail-open, but only scenario 5/6 can pass.")
    else:
        print(f"✓ Sidecar healthy (v{health['version']})")

    server = start_mcp_server()
    print("✓ MCP tool server up on :8921")
    agent = IncidentAgent()
    try:
        await agent.initialize()
        for scenario in (
            s1_db_lossless,
            s2_logs_ccr,
            s3_search,
            s4_read_excluded,
            s5_fail_open,
            s6_anti_forgery,
            s7_streaming,
            s8_multi_payload,
            s9_kompress,
            s10_rag_context_protected,
            s11_file_read_stale,
        ):
            print(f"\n▶ {scenario.__doc__ or scenario.__name__}")
            try:
                await scenario(agent)
            except Exception as e:
                record(scenario.__name__, False, f"raised {type(e).__name__}: {e}")
    finally:
        await agent.close()
        server.kill()

    print("\n" + "=" * 72)
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    for status, name, detail in RESULTS:
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[status]
        print(f"{mark} {name}" + (f" — {detail}" if status != "PASS" and detail else ""))
    print(f"\n{len([r for r in RESULTS if r[0] == 'PASS'])} passed, "
          f"{len(fails)} failed, {len([r for r in RESULTS if r[0] == 'SKIP'])} skipped")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
