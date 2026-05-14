#!/usr/bin/env python3
"""
CLI for SDK Feature Test playground.

Run full pipeline:  python -m playground.sdk_feature_test run --scenario full
Health only:        python -m playground.sdk_feature_test run --scenario health
Workflow only:     python -m playground.sdk_feature_test run --scenario workflow-only
No Temporal:       python -m playground.sdk_feature_test run --scenario no-temporal

Interactive:       python -m playground.sdk_feature_test
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root and src on path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from orchestrator import get_logger, setup_logging
from orchestrator.agent.types import generate_run_id

from playground.sdk_feature_test.config import get_config
from playground.sdk_feature_test.pipeline import (
    run_conditional_pipeline,
    run_full_pipeline,
    run_health_only,
    run_hitl_pipeline,
    run_mcp_temporal_pipeline,
    run_no_temporal,
    run_parallel_pipeline,
    run_reasoning_patterns,
    run_workflow_only,
)

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDK Feature Test: multi-agent pipeline that exercises all Orchestrator SDK features.",
    )
    sub = parser.add_subparsers(dest="command", help="Command")
    run_parser = sub.add_parser("run", help="Run a scenario")
    run_parser.add_argument(
        "--scenario",
        choices=["full", "workflow-only", "no-temporal", "health", "hitl", "reasoning-patterns", "parallel", "conditional", "mcp-temporal"],
        default="full",
        help="full = all features; workflow-only = core + router/seq/par/loop; no-temporal = full without Temporal; health = health checks only; hitl = Temporal with human-in-the-loop approval; reasoning-patterns = two-pass reasoning + ReAct + ReflectionAgent; parallel = Temporal ParallelStep with two concurrent agents; conditional = Temporal ConditionalStep with branch selection; mcp-temporal = Temporal workflow with MCP tool calls inside an activity",
    )
    run_parser.add_argument(
        "--user-id",
        default=None,
        help="User ID for session/memory (default: generated)",
    )
    run_parser.add_argument(
        "--resume",
        default=None,
        metavar="WORKFLOW_ID",
        help="Resume an existing paused HITL workflow by ID instead of starting a new one (hitl scenario only)",
    )
    run_parser.add_argument(
        "--cancel",
        default=None,
        metavar="WORKFLOW_ID",
        help="Cancel a running workflow by ID and verify it reaches cancelled status",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


async def _cancel_workflow(workflow_id: str, verbose: bool) -> int:
    if verbose:
        setup_logging(level="DEBUG")
    try:
        from orchestrator.temporal import get_temporal_client
    except ImportError:
        logger.error("Temporal SDK not installed.")
        return 1

    client = get_temporal_client()
    try:
        await client.connect()
    except Exception as e:
        logger.error(f"Cannot connect to Temporal: {e}")
        return 1

    try:
        await client.cancel_workflow(workflow_id)
        print(f"\nCancellation signal sent to workflow: {workflow_id}")

        # Poll for cancelled status
        for _ in range(15):
            await asyncio.sleep(2)
            try:
                status_raw = await client.query_workflow(workflow_id, "get_status")
                status = status_raw.get("status", "") if isinstance(status_raw, dict) else ""
                print(f"  Status: {status}")
                if status in ("cancelled", "completed", "failed"):
                    break
            except Exception:
                print("  Workflow finished (no longer queryable)")
                break

        print(f"\nCancellation test complete for: {workflow_id}")
        return 0
    except Exception as e:
        logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
        return 1
    finally:
        await client.disconnect()


async def _run_scenario(scenario: str, user_id: str | None, verbose: bool, resume: str | None = None) -> int:
    cfg = get_config()  # respects TEMPORAL_ENABLED etc. from env
    if verbose:
        setup_logging(level="DEBUG")
    uid = user_id or f"sdk-test-{generate_run_id()[-8:]}"
    session_id: str | None = None

    if scenario == "health":
        ok = await run_health_only(cfg)
        return 0 if ok else 1
    if scenario == "workflow-only":
        ok = await run_workflow_only(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "no-temporal":
        ok = await run_no_temporal(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "hitl":
        ok = await run_hitl_pipeline(cfg, uid, session_id, resume_workflow_id=resume)
        return 0 if ok else 1
    if scenario == "full":
        ok = await run_full_pipeline(cfg, uid, session_id, include_temporal=cfg.enable_temporal)
        return 0 if ok else 1
    if scenario == "reasoning-patterns":
        ok = await run_reasoning_patterns(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "parallel":
        ok = await run_parallel_pipeline(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "conditional":
        ok = await run_conditional_pipeline(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "mcp-temporal":
        ok = await run_mcp_temporal_pipeline(cfg, uid, session_id)
        return 0 if ok else 1
    return 1


def _interactive_menu() -> int:
    print("\nSDK Feature Test - Research & Report Pipeline")
    print(" 1. Health only")
    print(" 2. Workflow only (router, sequential, parallel, loop)")
    print(" 3. Full (no Temporal)")
    print(" 4. Full (with Temporal if enabled)")
    print(" 5. Human-in-the-Loop (Temporal + approval gate)")
    print(" 6. Reasoning patterns (two-pass + ReAct + ReflectionAgent)")
    print(" 7. Parallel (Temporal ParallelStep, two concurrent agents)")
    print(" 8. Conditional (Temporal ConditionalStep, branch on question vs. statement)")
    print(" 9. MCP + Temporal (researcher agent with MCP tools inside Temporal activity)")
    print(" q. Quit")
    choice = input("Choice [1-9/q]: ").strip().lower()
    if choice == "q":
        return 0
    scenario_map = {
        "1": "health",
        "2": "workflow-only",
        "3": "no-temporal",
        "4": "full",
        "5": "hitl",
        "6": "reasoning-patterns",
        "7": "parallel",
        "8": "conditional",
        "9": "mcp-temporal",
    }
    scenario = scenario_map.get(choice, "health")
    return asyncio.run(_run_scenario(scenario, None, verbose=True))


def main() -> int:
    args = _parse_args()
    if args.command != "run":
        return _interactive_menu()
    if getattr(args, "cancel", None):
        return asyncio.run(_cancel_workflow(args.cancel, getattr(args, "verbose", False)))
    return asyncio.run(_run_scenario(args.scenario, args.user_id, getattr(args, "verbose", False), resume=getattr(args, "resume", None)))


if __name__ == "__main__":
    sys.exit(main())
