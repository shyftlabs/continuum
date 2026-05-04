"""Golden routing benchmarks (route-only and optional e2e+judge)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from orchestrator.agent.config import RouterConfig
from orchestrator.agent.smart_layer.classifier import classify_product_tier
from orchestrator.agent.smart_layer.runner_facade import run_model_tier_turn
from orchestrator.agent.smart_layer.types import ProductTier, parse_product_tier
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.llm import LLMClient
from orchestrator.llm.config import LLMConfig

from stats_store import HEAVY_TIERS


def load_golden_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def golden_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, int] = {}
    for r in rows:
        et = str(r.get("expected_tier", "")).lower()
        by_tier[et] = by_tier.get(et, 0) + 1
    return {"total_examples": len(rows), "expected_tier_breakdown": by_tier}


async def run_route_only(
    rows: list[dict[str, Any]],
    *,
    router_config: RouterConfig,
    llm: LLMClient,
    instructions: str,
) -> dict[str, Any]:
    correct = 0
    predictions: list[str] = []
    expected_list: list[str] = []
    cls_when_llm: list[float] = []
    agent = RouterAgent(
        name="golden-router",
        instructions=instructions,
        routes=[],
        router_config=router_config,
    )

    per_row: list[dict[str, Any]] = []

    for r in rows:
        q = str(r.get("query", "")).strip()
        exp = str(r.get("expected_tier", "")).lower().strip()
        expected_list.append(exp)

        out = await classify_product_tier(
            user_text=q,
            router_config=agent.router_config,
            llm_client=llm,
            forced_tier=None,
        )
        pred = out.tier.value
        predictions.append(pred)
        ok = pred == exp
        if ok:
            correct += 1
        if not out.skipped_classifier:
            cls_when_llm.append(out.classify_ms)

        per_row.append(
            {
                "query_preview": q[:120] + ("…" if len(q) > 120 else ""),
                "expected_tier": exp,
                "predicted_tier": pred,
                "match": ok,
                "skipped_classifier": out.skipped_classifier,
                "classify_ms": round(out.classify_ms, 3),
            }
        )

    n = len(rows)
    top1 = correct / n if n else 0.0
    escalations = sum(1 for p in predictions if p in HEAVY_TIERS)
    escalation_rate = escalations / n if n else 0.0

    cls_mean = statistics.mean(cls_when_llm) if cls_when_llm else None
    cls_p95 = None
    if len(cls_when_llm) >= 2:
        sorted_c = sorted(cls_when_llm)
        idx = min(len(sorted_c) - 1, int(0.95 * (len(sorted_c) - 1)))
        cls_p95 = round(sorted_c[idx], 3)

    # Static relative cost proxy vs all-frontier (tier weights heuristic)
    WEIGHT = {"nano": 0.15, "fast": 0.25, "balanced": 0.5, "specialist": 0.75, "frontier": 1.0}
    routed_cost = sum(WEIGHT.get(p, 0.5) for p in predictions)
    all_frontier = n * 1.0
    cost_proxy_saved_vs_frontier = round(1.0 - routed_cost / all_frontier, 4) if n else 0.0

    return {
        "mode": "route_only",
        "rows_evaluated": n,
        "top1_accuracy": round(top1, 4),
        "escalation_rate_heavy_tiers": round(escalation_rate, 4),
        "classifier_latency_ms": {
            "mean_when_classifier_ran": round(cls_mean, 3) if cls_mean is not None else None,
            "p95_when_classifier_ran": cls_p95,
            "runs_with_llm_classifier": len(cls_when_llm),
        },
        "cost_proxy": {
            "relative_spend_vs_all_frontier": round(routed_cost / all_frontier, 4) if n else 0.0,
            "estimated_savings_fraction_vs_all_frontier": cost_proxy_saved_vs_frontier,
        },
        "per_row": per_row,
    }


async def judge_score(llm: LLMClient, *, user_query: str, answer: str) -> float:
    prompt = f"""Rate the assistant answer quality for the user query on a scale of 1 to 5 only.
5 = excellent, accurate, complete; 1 = useless or wrong.
Respond with a single digit 1-5 only.

User query:
{user_query}

Assistant answer:
{answer}
"""
    resp = await llm.chat(
        messages=[{"role": "user", "content": prompt}],
        config=LLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=8),
        auto_session=False,
    )
    text = (resp.content or "").strip()
    for ch in text:
        if ch in "12345":
            return float(ch)
    return 3.0


async def run_e2e_judge(
    rows: list[dict[str, Any]],
    *,
    router_config: RouterConfig,
    llm: LLMClient,
    instructions: str,
    max_rows: int = 5,
) -> dict[str, Any]:
    subset = rows[:max_rows]
    agent = RouterAgent(
        name="golden-e2e-router",
        instructions=instructions,
        routes=[],
        router_config=router_config,
    )

    per_row: list[dict[str, Any]] = []
    pgrs: list[float] = []

    for r in subset:
        q = str(r.get("query", "")).strip()
        exp = str(r.get("expected_tier", "")).lower()

        routed = await run_model_tier_turn(agent, llm, user_text=q)
        nano = await run_model_tier_turn(
            agent,
            llm,
            user_text=q,
            forced_tier=ProductTier.nano,
        )
        frontier = await run_model_tier_turn(
            agent,
            llm,
            user_text=q,
            forced_tier=ProductTier.frontier,
        )

        sr = await judge_score(llm, user_query=q, answer=routed.content)
        sn = await judge_score(llm, user_query=q, answer=nano.content)
        sf = await judge_score(llm, user_query=q, answer=frontier.content)

        denom = max(0.01, sf - sn)
        pgr = (sr - sn) / denom
        pgrs.append(pgr)

        per_row.append(
            {
                "query_preview": q[:100] + ("…" if len(q) > 100 else ""),
                "expected_tier": exp,
                "routed_tier": routed.routing.get("tier"),
                "judge_routed": sr,
                "judge_nano": sn,
                "judge_frontier": sf,
                "pgr": round(pgr, 4),
                "timings_ms": routed.routing.get("timings_ms"),
            }
        )

    mean_pgr = statistics.mean(pgrs) if pgrs else 0.0
    mean_judge_routed = statistics.mean(float(x["judge_routed"]) for x in per_row) if per_row else 0.0

    return {
        "mode": "e2e_judge",
        "rows_evaluated": len(subset),
        "max_rows_cap": max_rows,
        "mean_judge_routed": round(mean_judge_routed, 3),
        "mean_pgr": round(mean_pgr, 4),
        "pgr_formula": "(judge_routed - judge_nano) / max(0.01, judge_frontier - judge_nano)",
        "per_row": per_row,
    }
