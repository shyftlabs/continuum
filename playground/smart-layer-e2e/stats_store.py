"""Per-session aggregate stats for the routing playground UI."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


LIGHT_TIERS = frozenset({"nano", "fast"})
HEAVY_TIERS = frozenset({"balanced", "specialist", "frontier"})


@dataclass
class SessionStats:
    total_requests: int = 0
    tier_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    light_turns: int = 0
    heavy_turns: int = 0
    # Latency sums for averages (classifier: only when classifier actually ran remotely)
    sum_classify_ms_when_llm: float = 0.0
    count_classify_llm: int = 0
    sum_classify_ms_all: float = 0.0
    sum_llm_ms: float = 0.0
    count_llm: int = 0

    def record_turn(self, routing: dict[str, Any]) -> None:
        self.total_requests += 1
        tier = str(routing.get("tier") or "balanced")
        self.tier_counts[tier] += 1
        if tier in LIGHT_TIERS:
            self.light_turns += 1
        elif tier in HEAVY_TIERS:
            self.heavy_turns += 1

        timings = routing.get("timings_ms") or {}
        cls_ms = float(timings.get("classify") or 0)
        llm_ms = float(timings.get("llm") or 0)

        self.sum_classify_ms_all += cls_ms
        skipped = bool(routing.get("skipped_classifier"))
        if not skipped:
            self.sum_classify_ms_when_llm += cls_ms
            self.count_classify_llm += 1

        self.sum_llm_ms += llm_ms
        self.count_llm += 1

    def to_public_dict(self) -> dict[str, Any]:
        avg_cls_remote = (
            round(self.sum_classify_ms_when_llm / max(1, self.count_classify_llm), 3)
            if self.count_classify_llm
            else None
        )
        avg_llm = round(self.sum_llm_ms / max(1, self.count_llm), 3) if self.count_llm else None
        avg_cls_all = (
            round(self.sum_classify_ms_all / max(1, self.total_requests), 3)
            if self.total_requests
            else None
        )
        return {
            "total_requests": self.total_requests,
            "tier_counts": dict(self.tier_counts),
            "light_turns": self.light_turns,
            "heavy_turns": self.heavy_turns,
            "averages_ms": {
                "classifier_when_llm_ran": avg_cls_remote,
                "classifier_including_shortcuts": avg_cls_all,
                "completion": avg_llm,
            },
            "samples_for_classifier_remote_avg": self.count_classify_llm,
        }


class StatsStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionStats] = {}

    async def record(self, session_id: str, routing: dict[str, Any]) -> None:
        sid = session_id or "default"
        async with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = SessionStats()
            self._sessions[sid].record_turn(routing)

    async def get(self, session_id: str) -> SessionStats:
        sid = session_id or "default"
        async with self._lock:
            return self._sessions.get(sid) or SessionStats()

    async def clear(self, session_id: str) -> None:
        sid = session_id or "default"
        async with self._lock:
            self._sessions.pop(sid, None)
