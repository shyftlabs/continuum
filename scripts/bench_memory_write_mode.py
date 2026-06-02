#!/usr/bin/env python3
"""
Memory write-mode latency benchmark (sync vs. background).

Measures how long ``SessionClient.add_message`` takes to RETURN under each
``memory_write_mode``, using a mocked mem0 write whose latency you control.
No real infrastructure is used (no Redis, no mem0/LLM, no vector store), so it
runs anywhere and isolates the one thing that matters: does 'background' mode
keep the slow long-term-memory write off the response path?

This is a local sanity check for the Phase 2 rollout — it does NOT replace the
staging measurement against real traffic.

Usage:
    python scripts/bench_memory_write_mode.py
    python scripts/bench_memory_write_mode.py --iterations 100 --mem-latency-ms 300

Interpretation:
    'sync' per-call time should track --mem-latency-ms (the write blocks the
    return). 'background' per-call time should be near zero (the write is
    scheduled and drained later). The delta is the latency moved off the
    request path.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path (mirror scripts/health_check.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from orchestrator.core.background_tasks import BackgroundTaskRegistry  # noqa: E402
from orchestrator.llm.types import ChatMessage  # noqa: E402
from orchestrator.session.client import SessionClient  # noqa: E402
from orchestrator.session.config import SessionConfig  # noqa: E402
from orchestrator.session.types import SessionMetadata  # noqa: E402


def _metadata(session_id: str = "bench-session") -> SessionMetadata:
    now = datetime.now(UTC)
    return SessionMetadata(
        session_id=session_id,
        user_id="bench-user",
        agent_id="bench-agent",
        conversation_id="bench-conv",
        created_at=now,
        last_accessed_at=now,
    )


def _build_client(mode: str, mem_latency_s: float, registry: BackgroundTaskRegistry):
    """SessionClient with a mocked provider + a mem0 add() that sleeps."""

    async def _slow_add(*args, **kwargs):
        await asyncio.sleep(mem_latency_s)
        return MagicMock(results=[])

    mem = MagicMock()
    mem.is_enabled = True
    mem.add = AsyncMock(side_effect=_slow_add)
    mem.delete = AsyncMock()

    provider = MagicMock()
    provider.add_message = AsyncMock()
    provider.get_session_metadata = AsyncMock(return_value=_metadata())

    return SessionClient(
        session_config=SessionConfig(enabled=True, memory_write_mode=mode),
        memory_client=mem,
        provider=provider,
        auto_initialize=False,
        background_tasks=registry,
    )


async def _measure(mode: str, iterations: int, mem_latency_s: float) -> list[float]:
    """Return per-call add_message() return-times (ms) for the given mode."""
    registry = BackgroundTaskRegistry(name=f"bench-{mode}")
    client = _build_client(mode, mem_latency_s, registry)
    timings_ms: list[float] = []

    for i in range(iterations):
        msg = ChatMessage(role="user", content=f"benchmark message {i}")
        start = time.perf_counter()
        await client.add_message("bench-session", msg)
        timings_ms.append((time.perf_counter() - start) * 1000)

    # Drain any backgrounded writes so they don't leak across modes.
    await registry.drain(timeout=30.0)
    return timings_ms


def _summarize(label: str, timings_ms: list[float]) -> dict[str, float]:
    timings_sorted = sorted(timings_ms)
    p50 = statistics.median(timings_sorted)
    p95 = timings_sorted[min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.95))]
    mean = statistics.fmean(timings_sorted)
    print(
        f"  {label:<11} mean={mean:8.2f}ms  p50={p50:8.2f}ms  "
        f"p95={p95:8.2f}ms  (n={len(timings_ms)})"
    )
    return {"mean": mean, "p50": p50, "p95": p95}


async def main_async(args: argparse.Namespace) -> None:
    mem_latency_s = args.mem_latency_ms / 1000.0

    print("\n" + "=" * 60)
    print("  ⏱  Memory write-mode latency benchmark")
    print("=" * 60)
    print(f"  iterations={args.iterations}  simulated mem0 write latency={args.mem_latency_ms}ms\n")

    sync = _summarize("sync", await _measure("sync", args.iterations, mem_latency_s))
    bg = _summarize("background", await _measure("background", args.iterations, mem_latency_s))

    delta_p50 = sync["p50"] - bg["p50"]
    delta_p95 = sync["p95"] - bg["p95"]
    print("\n  latency removed from response path:")
    print(f"    p50: {delta_p50:8.2f}ms     p95: {delta_p95:8.2f}ms")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sync vs background memory writes")
    parser.add_argument(
        "--iterations", type=int, default=50, help="add_message calls per mode (default: 50)"
    )
    parser.add_argument(
        "--mem-latency-ms",
        type=float,
        default=250.0,
        help="Simulated mem0 write latency in ms (default: 250)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
