"""Roll the per-call timing log up into per-user-question totals.

The probe (src/continuum/llm/timing_probe.py) writes one JSON line per HTTP call
to an LLM provider. One user question is several of those — the agent re-enters
the model after every tool result — so the raw log cannot answer "what did this
question cost". agent.py tags each turn with a shared id; this reads them back.

    CONTINUUM_TIMING_LOG=1 python web.py          # collect
    python timing_report.py                       # report

Reads CONTINUUM_TIMING_LOG_PATH, or continuum-timing.jsonl in the working
directory, or a path given as the first argument.

Phases are NOT hardcoded here. Whatever numeric keys the gateway sent are
summed, so a phase added to the gateway's registry shows up without editing this
file — the same reason the gateway's own harness stopped keeping a hand-written
field list.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_PATH = "continuum-timing.jsonl"

# The gateway's phases are not a flat partition of the request — some contain
# others, and summing them together double-counts. Three groups:
#
#   AGGREGATE   spans covering the whole request. Reported, never summed.
#   NESTED      a strict subset of another phase. Shown indented under the
#               phase that contains it, and excluded from the total.
#   everything else — disjoint, and what the breakdown actually adds up.
#
# Verified against a live payload: classifier_llm_call_ms (3423.2) +
# classifier_extract_ms (1.9) reconstructs classifier_ms (3429.0), and
# provider_attempt_1_ms (1796.4) matches upstream_ms (1796.3) on a single-attempt
# request — upstream_ms being the sum over attempts.
AGGREGATE_PHASES = {"gateway_total_ms", "gateway_pre_upstream_ms"}

NESTED_PHASES = {
    "classifier_llm_call_ms": "classifier_ms",
    "classifier_extract_ms": "classifier_ms",
}


def _parent_of(key: str) -> str | None:
    """The phase that contains `key`, or None if it stands alone."""
    if key in NESTED_PHASES:
        return NESTED_PHASES[key]
    # provider_attempt_N_ms are the per-attempt spans upstream_ms sums.
    if key.startswith("provider_attempt_") and key.endswith("_ms"):
        return "upstream_ms"
    # request_shaping_N_ms are disjoint passes, not subsets — left to sum.
    return None


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"no timing log at {path}. Run with CONTINUUM_TIMING_LOG=1 first.")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("CONTINUUM_TIMING_LOG_PATH") or DEFAULT_PATH
    )
    records = load(path)

    turns: dict[str | None, list[dict]] = defaultdict(list)
    for r in records:
        turns[r.get("turn_id")].append(r)

    untagged = turns.pop(None, [])
    if not turns:
        sys.exit(
            f"{len(records)} records, none carrying a turn_id. "
            "Calls made outside chat()/chat_stream() are not tagged."
        )

    print(f"\n{len(records)} calls across {len(turns)} turns   ({path})")
    if untagged:
        print(f"  ({len(untagged)} untagged calls excluded — made outside a turn)")

    per_turn_client: list[float] = []
    per_turn_phase: dict[str, list[float]] = defaultdict(list)
    per_turn_calls: list[int] = []

    for tid, calls in turns.items():
        calls.sort(key=lambda r: r.get("seq") or 0)
        meta = next((c.get("turn_meta") for c in calls if c.get("turn_meta")), {}) or {}
        client_total = sum(c.get("client_ms") or 0 for c in calls)

        per_turn_client.append(client_total)
        per_turn_calls.append(len(calls))

        print("\n" + "=" * 78)
        print(f"turn {tid}   {len(calls)} gateway calls   {client_total:.0f} ms total")
        if meta.get("question"):
            print(f'  "{meta["question"]}"   [{meta.get("agent_model", "?")}]')
        print("-" * 78)
        print(f"  {'#':>2}  {'client_ms':>10}  {'gateway':>9}  {'upstream':>9}  {'classifier':>11}")

        totals: dict[str, float] = defaultdict(float)
        skipped: list[tuple[Any, dict]] = []
        for c in calls:
            t = c.get("timing") or {}
            for k, v in t.items():
                if isinstance(v, (int, float)):
                    totals[k] += v

            # A call that ran prompt extraction but produced no classifier_ms was
            # not "cheap" — the gateway found no user text and skipped
            # classification entirely, routing with no complexity signal. Read as
            # a saving it flatters the numbers; flag it instead.
            cls = t.get("classifier_ms")
            req = c.get("request") or {}
            if cls is None and "classifier_extract_ms" in t:
                skipped.append((c.get("seq"), req))

            print(
                f"  {c.get('seq', '?'):>2}  {c.get('client_ms') or 0:>10.1f}"
                f"  {t.get('gateway_total_ms', 0):>9.1f}"
                f"  {t.get('upstream_ms', 0):>9.1f}"
                f"  {(f'{cls:.1f}' if cls is not None else 'skipped'):>11}"
                f"   {'/'.join(req.get('roles') or []) or '?'}"
            )

        for seq, req in skipped:
            why = (
                f"declared complexity={req['declared_complexity']!r}"
                if req.get("declared_complexity")
                else f"no user text (last user message: {req.get('last_user_chars', '?')} chars"
                f", roles {req.get('roles')})"
            )
            print(f"\n  !! call {seq} was NOT classified — {why}")
            print("     routed with no complexity signal, and no x-aura-classification header.")

        for k, v in totals.items():
            per_turn_phase[k].append(v)

        disjoint = {
            k: v
            for k, v in totals.items()
            if k not in AGGREGATE_PHASES and _parent_of(k) is None and v >= 0.05
        }
        children: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for k, v in totals.items():
            parent = _parent_of(k)
            if parent and v >= 0.05:
                children[parent].append((k, v))

        if disjoint:
            print(f"\n  where the {client_total:.0f} ms went (summed across the turn):")
            for k, v in sorted(disjoint.items(), key=lambda kv: -kv[1]):
                share = v / client_total * 100 if client_total else 0
                bar = "█" * max(0, min(40, round(share / 2.5)))
                print(f"    {k:<26}{v:>9.1f} ms  {share:>5.1f}%  {bar}")
                for ck, cv in sorted(children.get(k, []), key=lambda kv: -kv[1]):
                    print(f"      └ {ck:<22}{cv:>9.1f} ms")
            named = sum(disjoint.values())
            rest = client_total - named
            print(
                f"    {'(client-side + untimed)':<26}{rest:>9.1f} ms"
                f"  {rest / client_total * 100 if client_total else 0:>5.1f}%"
            )

    if len(turns) > 1:
        print("\n" + "=" * 78)
        print(f"ACROSS {len(turns)} TURNS")
        print("-" * 78)
        print(f"  {'':<26}{'median':>10}{'min':>10}{'max':>10}")
        print(f"  {'gateway calls per turn':<26}{st.median(per_turn_calls):>10.1f}"
              f"{min(per_turn_calls):>10}{max(per_turn_calls):>10}")
        print(f"  {'client_ms per turn':<26}{st.median(per_turn_client):>10.1f}"
              f"{min(per_turn_client):>10.1f}{max(per_turn_client):>10.1f}")
        for k in sorted(per_turn_phase, key=lambda k: -st.median(per_turn_phase[k])):
            v = per_turn_phase[k]
            print(f"  {k:<26}{st.median(v):>10.1f}{min(v):>10.1f}{max(v):>10.1f}")

        med_client = st.median(per_turn_client)
        med_cls = st.median(per_turn_phase.get("classifier_ms", [0]))
        if med_client and med_cls:
            print(f"\n  classifier as a share of one user question: {med_cls / med_client * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()
