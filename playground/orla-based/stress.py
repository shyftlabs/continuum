"""
Verify PriorityDispatcher works end-to-end.

Forces max_concurrent=1 so only one LLM call runs at a time.
Fires 5 requests simultaneously: 3 free (priority=2) + 2 premium (priority=9).

Expected behaviour:
  - One request grabs the single worker immediately (race — any tier)
  - The remaining 4 queue up
  - Dispatcher serves queued premium requests before free ones
  - All premiums complete before all frees (except the first-to-grab)

Usage:
    python stress.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator import LogLevel, setup_logging
from orchestrator.llm.dispatcher import PriorityDispatcher

from agents import OrlaPlayground
from config import AppConfig, build_policy_store
from pipeline import run_direct

MESSAGE = "Reply with exactly one word: hello"


async def main() -> None:
    setup_logging(level=LogLevel.WARNING)

    # max_concurrent=1 forces all but one request into the priority queue
    config = AppConfig(
        policy_store=build_policy_store(),
        dispatcher=PriorityDispatcher(max_concurrent=1),
    )

    app = OrlaPlayground(config)
    await app.initialize()

    print("=" * 60)
    print("  PriorityDispatcher stress test  (max_concurrent=1)")
    print("  free priority=2   premium priority=9")
    print("=" * 60)
    print()

    results: list[tuple[float, str, str]] = []   # (elapsed, label, tier)
    start = time.monotonic()

    async def fire(label: str, tier: str, idx: int) -> None:
        sent_at = time.monotonic() - start
        print(f"  {sent_at:.2f}s  SENT  {label:12s}  tier={tier}")
        await run_direct(
            app,
            message=MESSAGE,
            tier=tier,
            user_id=f"stress-user-{idx}",
            conversation_id=f"stress-conv-{idx}",
        )
        done_at = time.monotonic() - start
        results.append((done_at, label, tier))
        print(f"  {done_at:.2f}s  DONE  {label:12s}  tier={tier}")

    # Fire all 5 simultaneously
    await asyncio.gather(
        fire("free-1",    "free",    1),
        fire("free-2",    "free",    2),
        fire("premium-1", "premium", 3),
        fire("free-3",    "free",    4),
        fire("premium-2", "premium", 5),
    )

    # --- Report ---
    ordered = sorted(results)
    print()
    print("Completion order:")
    for rank, (t, label, tier) in enumerate(ordered, 1):
        tag = "PREMIUM" if tier == "premium" else "free   "
        print(f"  {rank}. {t:.2f}s  [{tag}]  {label}")

    # Validate: among requests 2–5 (queued ones), premiums should come first
    queued = ordered[1:]   # skip position-1 (grabbed worker before any queue existed)
    queued_tiers = [tier for _, _, tier in queued]
    premium_positions = [i for i, t in enumerate(queued_tiers) if t == "premium"]
    free_positions    = [i for i, t in enumerate(queued_tiers) if t == "free"]

    print()
    if premium_positions and free_positions:
        if max(premium_positions) < min(free_positions):
            print("PASS — all queued premium requests completed before queued free ones")
        else:
            print("FAIL — premium did not consistently beat free in queue")
            print("       (check that priority is wired through executor → dispatcher)")
    else:
        print("NOTE — not enough of both tiers in queue to compare ordering")


if __name__ == "__main__":
    asyncio.run(main())
