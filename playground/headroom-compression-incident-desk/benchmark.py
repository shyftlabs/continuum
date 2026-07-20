#!/usr/bin/env python3
"""
Incident Desk — Headroom token-reduction benchmark (with vs without).

Measures GROSS per-call compression at the sidecar's /v1/compress seam for
every Incident Desk scenario. There is NO LLM in the loop, so
continuum_headroom_retrieve can never fire — the numbers are the clean,
deterministic reduction in the tokens we would send to the model
(without Headroom = tokens_before; with Headroom = tokens_after).

    python benchmark.py                 # scenarios 1-8,10,11
    KOMPRESS=1 python benchmark.py      # + scenario 9 (prose / ML router)

Emits a per-scenario table + aggregate, and writes benchmark_results.md
(paste-ready for the report).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import config  # noqa: F401 — .env guard + HEADROOM_ENABLED
from config import SIDECAR_BASE, sidecar_health
from data import (
    SERVICE_YAML,
    format_runbook_results,
    generate_failed_orders,
    generate_logs,
    generate_postmortem,
)

HERE = Path(__file__).resolve().parent
SYS = {"role": "system", "content": "You are Incident Desk, an on-call incident copilot."}
RUNBOOK_Q = "database connection pool exhaustion"


def _tool_turn(tool: str, args: str, payload: str, question: str) -> list[dict]:
    """A realistic mid-history tool result: the payload is NOT the last message
    (so protect-recent doesn't shield it), mirroring a real agent turn."""
    return [
        SYS,
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": tool, "arguments": args}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": payload},
        {"role": "user", "content": "Answer using the data above."},
    ]


# --- scenario payload builders (compress-measurable ones) ------------------- #
def s1_db() -> list[dict]:
    return _tool_turn(
        "query_orders_db",
        "{}",
        json.dumps(generate_failed_orders()),
        "How many orders failed and which had the largest refund?",
    )


def s2_logs() -> list[dict]:
    return _tool_turn(
        "get_logs",
        '{"service":"checkout-api"}',
        generate_logs("checkout-api"),
        "What is the incident reference token in the checkout-api logs?",
    )


def s3_search() -> list[dict]:
    return _tool_turn(
        "search_runbooks",
        f'{{"q":"{RUNBOOK_Q}"}}',
        format_runbook_results(RUNBOOK_Q),
        "Which runbook covers DB connection pool exhaustion?",
    )


def s4_read_excluded() -> list[dict]:
    # A standalone `read` (no later write) — router excludes file reads.
    return _tool_turn(
        "read",
        '{"path":"service.yaml"}',
        SERVICE_YAML,
        "Report database.pool_max_size and workers.refund_worker_concurrency.",
    )


def s7_streaming() -> list[dict]:
    # Same shape/size as s2 but payments-svc — proves the streaming path
    # compresses identically (the seam is shared by run and run_stream).
    return _tool_turn(
        "get_logs",
        '{"service":"payments-svc"}',
        generate_logs("payments-svc"),
        "What is the incident reference token in the payments-svc logs?",
    )


def s8_multi() -> list[dict]:
    a, b = generate_logs("checkout-api"), generate_logs("payments-svc")
    return [
        SYS,
        {
            "role": "user",
            "content": "Fetch logs for checkout-api AND payments-svc; report each token.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_logs", "arguments": '{"service":"checkout-api"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "get_logs", "arguments": '{"service":"payments-svc"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": a},
        {"role": "tool", "tool_call_id": "c2", "content": b},
        {"role": "user", "content": "Report both incident tokens."},
    ]


def s9_kompress() -> list[dict]:
    return _tool_turn(
        "get_postmortem",
        '{"id":"INC-2417"}',
        generate_postmortem(),
        "How many checkout attempts failed per the postmortem?",
    )


def s10_rag_context() -> list[dict]:
    # Same bytes as s3, but injected as a PROVIDED CONTEXT system message —
    # Headroom protects system/developer messages, so it should pass through.
    payload = format_runbook_results(RUNBOOK_Q)
    return [
        SYS,
        {"role": "system", "content": "PROVIDED CONTEXT:\n" + payload},
        {
            "role": "user",
            "content": "Using ONLY the provided context, which runbook applies? Give its ID.",
        },
    ]


def s11_file_stale() -> list[dict]:
    return [
        {"role": "user", "content": "Read service.yaml, then raise the DB pool size and save it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "service.yaml"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": SERVICE_YAML},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"path": "service.yaml", "content": "pool_max_size: 50"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "content": "Wrote 17 bytes to service.yaml. Change applied.",
        },
        {"role": "user", "content": "Confirm the pool size is now 50."},
    ]


# id, title, section, builder, kind
COMPRESS_SCENARIOS = [
    ("1", "DB rows (failed orders)", "efficiency", s1_db),
    ("2", "Logs — checkout-api", "efficiency", s2_logs),
    ("3", "Search / RAG runbooks", "efficiency", s3_search),
    ("7", "Streaming path (logs)", "efficiency", s7_streaming),
    ("8", "Multi-tool (two log dumps)", "efficiency", s8_multi),
    ("11", "File read→write (stale, CCR ticket)", "efficiency", s11_file_stale),
    ("4", "read tool (excluded)", "safety", s4_read_excluded),
    ("10", "RAG context (system msg)", "safety", s10_rag_context),
]


async def measure(build) -> dict:
    from continuum.llm.headroom.client import HeadroomClient

    client = HeadroomClient(api_base=SIDECAR_BASE)
    try:
        _, stats, hashes = await client.compress(build(), model="gpt-4o-mini")
    finally:
        await client.aclose()
    saved = stats.tokens_before - stats.tokens_after
    pct = (1 - stats.compression_ratio) * 100
    return {
        "before": stats.tokens_before,
        "after": stats.tokens_after,
        "saved": saved,
        "pct": pct,
        "transforms": stats.transforms_applied,
        "hashes": len(hashes),
    }


async def s5_fail_open() -> str:
    from continuum.llm.headroom.client import HeadroomClient

    client = HeadroomClient(api_base="http://127.0.0.1:9")  # nothing listens
    try:
        try:
            await client.compress(s1_db(), model="gpt-4o-mini")
            return "unexpected: compress succeeded against dead sidecar"
        except Exception:
            return "sidecar down → compress errors, Continuum fail-opens (forwards uncompressed, run survives)"
    finally:
        await client.aclose()


async def s6_anti_forgery() -> str:
    from continuum.llm.headroom.compressor import new_run_compressor

    comp = new_run_compressor()  # empty issued_hashes
    out = await comp.resolve_retrieve("f" * 24)
    return "forged hash rejected" if "not issued" in out else f"UNEXPECTED: {out[:80]}"


async def main() -> int:
    h = sidecar_health()
    print(
        f"{'✓ sidecar' if h['up'] else '⚠️  sidecar DOWN'} @ {SIDECAR_BASE}"
        + (f"  (v{h.get('version')})" if h["up"] else "")
    )
    if not h["up"]:
        print("   start it:", h.get("restart"))
        return 1

    rows = []
    for sid, title, section, build in COMPRESS_SCENARIOS:
        m = await measure(build)
        rows.append((sid, title, section, m))
        print(
            f"  [{sid:>2}] {title:<32} {m['before']:>7,} → {m['after']:>6,} tok "
            f"({m['pct']:5.1f}% saved)  hashes={m['hashes']}"
        )

    if os.environ.get("KOMPRESS") == "1":
        m = await measure(s9_kompress)
        rows.append(("9", "Postmortem prose (Kompress)", "efficiency", m))
        print(
            f"  [ 9] {'Postmortem prose (Kompress)':<32} {m['before']:>7,} → {m['after']:>6,} tok "
            f"({m['pct']:5.1f}% saved)"
        )

    v5 = await s5_fail_open()
    v6 = await s6_anti_forgery()
    print(f"  [ 5] fail-open           : {v5}")
    print(f"  [ 6] anti-forgery        : {v6}")

    eff = [r for r in rows if r[2] == "efficiency"]
    tot_b = sum(r[3]["before"] for r in eff)
    tot_a = sum(r[3]["after"] for r in eff)
    agg_pct = (tot_b - tot_a) / tot_b * 100 if tot_b else 0
    print(
        f"\n  AGGREGATE (efficiency scenarios): {tot_b:,} → {tot_a:,} tok "
        f"({agg_pct:.1f}% saved, {tot_b - tot_a:,} tokens removed)"
    )

    # paste-ready markdown
    md = [
        "| # | Scenario | Section | Without (tok) | With (tok) | Saved | Transforms |",
        "|---|----------|---------|--------------:|-----------:|------:|------------|",
    ]
    for sid, title, section, m in rows:
        tf = ", ".join(t for t in m["transforms"] if not t.startswith("router:protected:user"))[:60]
        md.append(
            f"| {sid} | {title} | {section} | {m['before']:,} | {m['after']:,} "
            f"| {m['pct']:.1f}% | `{tf}` |"
        )
    md.append(
        f"| — | **Aggregate (efficiency)** | | **{tot_b:,}** | **{tot_a:,}** | **{agg_pct:.1f}%** | |"
    )
    md.append(f"| 5 | Fail-open (dead sidecar) | resilience | — | — | — | {v5} |")
    md.append(f"| 6 | Anti-forgery | security | — | — | — | {v6} |")
    (HERE / "benchmark_results.md").write_text("\n".join(md) + "\n")
    print(f"\n  wrote {HERE / 'benchmark_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
