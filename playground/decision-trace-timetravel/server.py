"""
Decision-Trace TimeTravel — MCP tools server (financial-close domain).

Reuses local/glassbox's DETERMINISTIC close engine so that, across *every*
multi-agent topology, the verdict is computed by tools (not LLM mood) and the
materiality threshold is the single universal lever: lower it past $2M and the
silently-waived D1 misstatement is caught — CONTROL_ISSUE flips to CLEAN.

  get_close_data        entities + 7 reconciliation discrepancies (D1 = $2M, affects balance)
  get_materiality_policy  the seeded $5M threshold (too high → D1 waived) + escalation floor
  get_intercompany      intercompany transactions to eliminate (never the bug)
  assess_materiality(threshold)     deterministic material/immaterial per discrepancy
  compute_consolidation(threshold)  deterministic suspense + CLEAN/CONTROL_ISSUE verdict

Run standalone:  python server.py   (MCP server on :8896)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("decision-trace-timetravel")

BUGGY_MATERIALITY_THRESHOLD_USD = 5_000_000
ESCALATION_FLOOR_USD = 1_000_000  # SOX-style floor: $1M+ items can't be silently waived

ENTITIES = [
    {"name": "Alpha Corp", "assets": 48_000_000, "liabilities": 30_000_000, "equity": 18_000_000},
    {
        "name": "Beta Holdings",
        "assets": 26_000_000,
        "liabilities": 15_000_000,
        "equity": 11_000_000,
    },
    {"name": "Gamma Labs", "assets": 19_000_000, "liabilities": 12_000_000, "equity": 7_000_000},
]

# Only D1 affects the balance (a real $2M misstatement); the rest are immaterial.
DISCREPANCIES = [
    {
        "id": "D1",
        "entity": "Alpha Corp",
        "account": "1990 — Suspense / unreconciled bank posting",
        "amount_usd": 2_000_000,
        "affects_balance": True,
        "kind": "misstatement",
    },
    {
        "id": "D2",
        "entity": "Beta Holdings",
        "account": "2100 — Vendor accrual",
        "amount_usd": 120_000,
        "affects_balance": False,
        "kind": "timing",
    },
    {
        "id": "D3",
        "entity": "Alpha Corp",
        "account": "7400 — FX revaluation",
        "amount_usd": 40_000,
        "affects_balance": False,
        "kind": "rounding",
    },
    {
        "id": "D4",
        "entity": "Gamma Labs",
        "account": "5100 — Payroll accrual",
        "amount_usd": 85_000,
        "affects_balance": False,
        "kind": "timing",
    },
    {
        "id": "D5",
        "entity": "Beta Holdings",
        "account": "1200 — AR cutoff",
        "amount_usd": 260_000,
        "affects_balance": False,
        "kind": "timing",
    },
    {
        "id": "D6",
        "entity": "Gamma Labs",
        "account": "1400 — Inventory write-down",
        "amount_usd": 300_000,
        "affects_balance": False,
        "kind": "estimate",
    },
    {
        "id": "D7",
        "entity": "Alpha Corp",
        "account": "2400 — Lease reclassification",
        "amount_usd": 75_000,
        "affects_balance": False,
        "kind": "reclassification",
    },
]

INTERCOMPANY = [
    {
        "id": "IC1",
        "from_entity": "Alpha Corp",
        "to_entity": "Beta Holdings",
        "amount_usd": 1_500_000,
        "description": "Intercompany management fee",
    },
    {
        "id": "IC2",
        "from_entity": "Beta Holdings",
        "to_entity": "Gamma Labs",
        "amount_usd": 900_000,
        "description": "Intercompany inventory transfer",
    },
]


@mcp.tool()
def get_close_data() -> dict:
    """Return the month-end close data: entities and the reconciliation
    discrepancies (each with amount, whether it affects the balance, and kind)."""
    return {"period": "April 2026", "entities": ENTITIES, "discrepancies": DISCREPANCIES}


@mcp.tool()
def get_materiality_policy() -> dict:
    """Return the materiality policy. materiality_threshold_usd is the editable
    lever: discrepancies >= it are MATERIAL; below it are waived. escalation_floor_usd
    is a hard regulatory floor."""
    return {
        "materiality_threshold_usd": BUGGY_MATERIALITY_THRESHOLD_USD,
        "escalation_floor_usd": ESCALATION_FLOOR_USD,
    }


@mcp.tool()
def get_intercompany() -> dict:
    """Return the intercompany transactions eliminated on consolidation (these
    always net out — never the cause of the control issue)."""
    return {"intercompany": INTERCOMPANY}


@mcp.tool()
def assess_materiality(threshold_usd: int) -> dict:
    """Deterministically classify each discrepancy MATERIAL (amount_usd >=
    threshold_usd) vs immaterial. Owns the arithmetic so the call is reliable;
    the agent supplies the threshold it was told to use."""
    assessments, material_ids, waived_ids = [], [], []
    for d in DISCREPANCIES:
        material = d["amount_usd"] >= threshold_usd
        (material_ids if material else waived_ids).append(d["id"])
        assessments.append(
            {
                "id": d["id"],
                "amount_usd": d["amount_usd"],
                "affects_balance": d["affects_balance"],
                "material": material,
            }
        )
    return {
        "threshold_usd": threshold_usd,
        "assessments": assessments,
        "material_ids": material_ids,
        "waived_ids": waived_ids,
    }


@mcp.tool()
def compute_consolidation(threshold_usd: int) -> dict:
    """Deterministically assemble the consolidated position for a materiality
    threshold. Classifies materiality (amount >= threshold → booked), leaves
    balance-affecting items below the threshold in suspense, and returns the
    verdict — CLEAN only when suspense nets to zero. The single source of truth
    for the outcome in EVERY topology, so the threshold lever flips the result
    deterministically (D1 = $2M: waived at $5M → CONTROL_ISSUE; caught at <= $2M
    → CLEAN). Narrate the result, do not recompute."""
    booked = {d["id"] for d in DISCREPANCIES if d["amount_usd"] >= threshold_usd}
    required = sum(d["amount_usd"] for d in DISCREPANCIES if d["affects_balance"])
    booked_amt = sum(
        d["amount_usd"] for d in DISCREPANCIES if d["affects_balance"] and d["id"] in booked
    )
    suspense = required - booked_amt
    balanced = suspense == 0
    return {
        "threshold_usd": threshold_usd,
        "material_ids": sorted(booked),
        "combined_assets_usd": sum(e["assets"] for e in ENTITIES),
        "consolidated_equity_usd": sum(e["equity"] for e in ENTITIES) - booked_amt,
        "required_adjustments_usd": required,
        "booked_adjustments_usd": booked_amt,
        "suspense_balance_usd": suspense,
        "intercompany_eliminated_usd": sum(x["amount_usd"] for x in INTERCOMPANY),
        "balanced": balanced,
        "status": "CLEAN" if balanced else "CONTROL_ISSUE",
    }


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    print("Decision-Trace TimeTravel MCP server running at http://localhost:8896/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8896)
