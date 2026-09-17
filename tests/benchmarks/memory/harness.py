"""
Reproducible performance and quality benchmark harness for Continuum's Memory subsystem.

Measures:
- add/search/update/delete latency percentiles (p50, p95, p99)
- Search throughput (QPS) under varying concurrency
- Latency scaling vs corpus size (1k, 10k, 100k)
- Semantic retrieval quality (Precision@k, Recall@k, MRR)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from continuum.memory.base import BaseMemoryProvider
from continuum.memory.client import MemoryClient
from continuum.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemorySearchResult,
)

# ---------------------------------------------------------------------------
# Metrics & Result Data Models
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """Statistical summary of operation latency in milliseconds."""

    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    throughput_qps: float

    @classmethod
    def from_durations_ns(cls, durations_ns: list[int], total_elapsed_s: float) -> LatencyStats:
        if not durations_ns:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        ms = sorted([d / 1_000_000.0 for d in durations_ns])
        count = len(ms)

        def percentile(data: list[float], pct: float) -> float:
            if not data:
                return 0.0
            idx = int(math.ceil((pct / 100.0) * len(data))) - 1
            idx = max(0, min(len(data) - 1, idx))
            return data[idx]

        p50 = percentile(ms, 50.0)
        p95 = percentile(ms, 95.0)
        p99 = percentile(ms, 99.0)
        min_v = ms[0]
        max_v = ms[-1]
        mean_v = statistics.mean(ms)
        std_v = statistics.stdev(ms) if count > 1 else 0.0
        qps = (count / total_elapsed_s) if total_elapsed_s > 0 else 0.0

        return cls(
            count=count,
            p50_ms=round(p50, 3),
            p95_ms=round(p95, 3),
            p99_ms=round(p99, 3),
            min_ms=round(min_v, 3),
            max_ms=round(max_v, 3),
            mean_ms=round(mean_v, 3),
            stddev_ms=round(std_v, 3),
            throughput_qps=round(qps, 2),
        )


@dataclass
class RetrievalQualityStats:
    """Information retrieval quality metrics."""

    precision_at_k: float
    recall_at_k: float
    mrr: float
    k: int
    total_queries: int


@dataclass
class BenchmarkReport:
    """Full benchmark run results report."""

    backend: str
    corpus_size: int
    concurrency: int
    timestamp_utc: str
    add_latency: LatencyStats
    search_latency: LatencyStats
    update_latency: LatencyStats
    delete_latency: LatencyStats
    concurrent_search_qps: LatencyStats
    retrieval_quality: RetrievalQualityStats
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic Corpus Generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticMemory:
    id: str
    text: str
    scope_type: str
    user_id: str | None
    agent_id: str | None
    conversation_id: str | None
    topic: str
    keywords: list[str]


@dataclass
class LabeledQueryProbe:
    query: str
    expected_memory_id: str
    scope_kwargs: dict[str, Any]


class CorpusGenerator:
    """Deterministic, seedable memory corpus and query probe generator."""

    TOPICS = [
        ("preferences", ["theme", "editor", "language", "shortcuts", "dark mode", "keybindings"]),
        ("projects", ["deployment", "kubernetes", "database", "redis", "temporal", "qdrant"]),
        ("profiles", ["engineer", "designer", "researcher", "architect", "manager", "analyst"]),
        ("security", ["token", "oauth", "jwt", "tls", "encryption", "rbac", "permissions"]),
        ("performance", ["cache", "indexing", "latency", "memory", "profiling", "throughput"]),
    ]

    TEMPLATES = [
        "User prefers {kw1} for all {topic} configurations and sets {kw2} as default.",
        "Agent notes that {topic} task requires handling {kw1} alongside {kw2}.",
        "Conversation context establishes requirement for {kw1} with optimized {kw2}.",
        "System requirement: optimize {topic} by monitoring {kw1} and enforcing {kw2}.",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_corpus(self, size: int) -> list[SyntheticMemory]:
        memories: list[SyntheticMemory] = []
        for i in range(size):
            topic, keywords = self.TOPICS[i % len(self.TOPICS)]
            kw1 = keywords[i % len(keywords)]
            kw2 = keywords[(i + 1) % len(keywords)]
            template = self.TEMPLATES[i % len(self.TEMPLATES)]
            text = template.format(topic=topic, kw1=kw1, kw2=kw2)

            scope_type = ["user", "agent", "shared"][i % 3]
            user_id = f"user-{i % 10}"
            agent_id = f"agent-{i % 5}"
            conversation_id = f"conv-{i % 20}" if i % 2 == 0 else None

            mem = SyntheticMemory(
                id=f"mem-{i:06d}",
                text=text,
                scope_type=scope_type,
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                topic=topic,
                keywords=[kw1, kw2],
            )
            memories.append(mem)
        return memories

    def generate_query_probes(
        self, corpus: list[SyntheticMemory], num_probes: int = 20
    ) -> list[LabeledQueryProbe]:
        probes: list[LabeledQueryProbe] = []
        step = max(1, len(corpus) // max(1, num_probes))
        for idx in range(0, min(len(corpus), num_probes * step), step):
            mem = corpus[idx]
            query = f"What is the preferred {mem.topic} setup regarding {mem.keywords[0]}?"
            scope_kwargs: dict[str, Any] = {}
            if mem.user_id:
                scope_kwargs["user_id"] = mem.user_id
            if mem.agent_id:
                scope_kwargs["agent_id"] = mem.agent_id
            if mem.conversation_id:
                scope_kwargs["conversation_id"] = mem.conversation_id

            probes.append(
                LabeledQueryProbe(
                    query=query,
                    expected_memory_id=mem.id,
                    scope_kwargs=scope_kwargs,
                )
            )
        return probes


# ---------------------------------------------------------------------------
# In-Memory Mock Benchmark Provider
# ---------------------------------------------------------------------------


class InMemoryBenchmarkProvider(BaseMemoryProvider):
    """
    Lightweight, thread-safe in-memory provider for benchmark testing.

    Performs deterministic term-frequency lexical/semantic scoring so search
    recall and ranking can be verified without external Qdrant or Milvus services.
    """

    def __init__(self, simulate_latency_ms: float = 0.0):
        self._storage: dict[str, dict[str, Any]] = {}
        self._simulate_latency_ms = simulate_latency_ms

    @property
    def provider_name(self) -> str:
        return "in_memory_benchmark"

    @property
    def is_initialized(self) -> bool:
        return True

    async def _sleep(self):
        if self._simulate_latency_ms > 0:
            await asyncio.sleep(self._simulate_latency_ms / 1000.0)

    async def add(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        custom_prompt: str | None = None,
        infer: bool = True,
    ) -> MemoryAddResult:
        await self._sleep()
        text = messages if isinstance(messages, str) else json.dumps(messages)
        mem_id = (metadata or {}).get("memory_id") or f"mem-{len(self._storage):06d}"

        entry = MemoryEntry(
            id=mem_id,
            memory=text,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            metadata=dict(metadata or {}),
        )
        self._storage[mem_id] = {
            "entry": entry,
            "tokens": set(text.lower().replace(".", " ").replace(",", " ").split()),
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
        }
        return MemoryAddResult(
            message="Memory added successfully",
            results=[{"id": mem_id, "memory": text}],
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        await self._sleep()
        q_tokens = set(query.lower().replace("?", " ").replace(".", " ").split())
        candidates: list[tuple[float, MemoryEntry]] = []

        for item in self._storage.values():
            if user_id and item["user_id"] != user_id:
                continue
            if agent_id and item["agent_id"] != agent_id:
                continue
            if conversation_id and item["conversation_id"] != conversation_id:
                continue

            overlap = len(q_tokens & item["tokens"])
            score = (overlap / max(1, len(q_tokens)))
            candidates.append((score, item["entry"]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [entry for _, entry in candidates[:limit]]
        return MemorySearchResult(results=top, query=query, limit=limit, total_results=len(top))

    async def get(self, memory_id: str) -> MemoryEntry | None:
        await self._sleep()
        item = self._storage.get(memory_id)
        return item["entry"] if item else None

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        await self._sleep()
        res = [
            item["entry"]
            for item in self._storage.values()
            if (not user_id or item["user_id"] == user_id)
            and (not agent_id or item["agent_id"] == agent_id)
            and (not conversation_id or item["conversation_id"] == conversation_id)
        ]
        return res[:limit] if limit else res

    async def delete(self, memory_id: str) -> bool:
        await self._sleep()
        return bool(self._storage.pop(memory_id, None))

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        await self._sleep()
        keys_to_del = [
            k
            for k, v in self._storage.items()
            if (not user_id or v["user_id"] == user_id)
            and (not agent_id or v["agent_id"] == agent_id)
            and (not conversation_id or v["conversation_id"] == conversation_id)
        ]
        for k in keys_to_del:
            del self._storage[k]
        return True

    async def update(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        await self._sleep()
        item = self._storage.get(memory_id)
        if not item:
            raise KeyError(f"Memory {memory_id} not found")
        item["entry"].memory = data
        item["tokens"] = set(data.lower().replace(".", " ").replace(",", " ").split())
        return item["entry"]

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        await self._sleep()
        return [{"id": memory_id, "version": 1}]

    async def reset(self) -> bool:
        await self._sleep()
        self._storage.clear()
        return True

    def reset_sync(self) -> bool:
        return asyncio.run(self.reset())

    async def close(self) -> None:
        self._storage.clear()

    # Sync implementations
    def add_sync(self, *args, **kwargs) -> MemoryAddResult:
        return asyncio.run(self.add(*args, **kwargs))

    def search_sync(self, *args, **kwargs) -> MemorySearchResult:
        return asyncio.run(self.search(*args, **kwargs))

    def get_sync(self, *args, **kwargs) -> MemoryEntry | None:
        return asyncio.run(self.get(*args, **kwargs))

    def get_all_sync(self, *args, **kwargs) -> list[MemoryEntry]:
        return asyncio.run(self.get_all(*args, **kwargs))

    def delete_sync(self, *args, **kwargs) -> bool:
        return asyncio.run(self.delete(*args, **kwargs))

    def delete_all_sync(self, *args, **kwargs) -> bool:
        return asyncio.run(self.delete_all(*args, **kwargs))

    def update_sync(self, *args, **kwargs) -> MemoryEntry:
        return asyncio.run(self.update(*args, **kwargs))

    def history_sync(self, *args, **kwargs) -> list[dict[str, Any]]:
        return asyncio.run(self.history(*args, **kwargs))


# ---------------------------------------------------------------------------
# Benchmark Harness
# ---------------------------------------------------------------------------


class MemoryBenchmarkHarness:
    """Benchmark harness coordinating execution, timing, and metric export."""

    def __init__(
        self,
        client: MemoryClient,
        backend_name: str = "mock",
        seed: int = 42,
    ):
        self.client = client
        self.backend_name = backend_name
        self.generator = CorpusGenerator(seed=seed)

    async def bench_add(self, corpus: list[SyntheticMemory]) -> LatencyStats:
        """Measure single-entry add latency and throughput."""
        durations_ns: list[int] = []
        t0 = time.perf_counter()

        for mem in corpus:
            start = time.perf_counter_ns()
            await self.client.add(
                mem.text,
                user_id=mem.user_id,
                agent_id=mem.agent_id,
                conversation_id=mem.conversation_id,
                metadata={"memory_id": mem.id, "topic": mem.topic},
            )
            durations_ns.append(time.perf_counter_ns() - start)

        total_elapsed = time.perf_counter() - t0
        return LatencyStats.from_durations_ns(durations_ns, total_elapsed)

    async def bench_search(self, probes: list[LabeledQueryProbe]) -> LatencyStats:
        """Measure single-query search latency."""
        durations_ns: list[int] = []
        t0 = time.perf_counter()

        for probe in probes:
            start = time.perf_counter_ns()
            await self.client.search(
                probe.query,
                limit=5,
                **probe.scope_kwargs,
            )
            durations_ns.append(time.perf_counter_ns() - start)

        total_elapsed = time.perf_counter() - t0
        return LatencyStats.from_durations_ns(durations_ns, total_elapsed)

    async def bench_concurrent_search(
        self, probes: list[LabeledQueryProbe], concurrency: int = 5
    ) -> LatencyStats:
        """Measure sustained search throughput (QPS) under N concurrent workers."""
        durations_ns: list[int] = []
        t0 = time.perf_counter()

        sem = asyncio.Semaphore(concurrency)

        async def worker(probe: LabeledQueryProbe):
            async with sem:
                start = time.perf_counter_ns()
                await self.client.search(
                    probe.query,
                    limit=5,
                    **probe.scope_kwargs,
                )
                durations_ns.append(time.perf_counter_ns() - start)

        tasks = [worker(p) for p in probes]
        await asyncio.gather(*tasks)

        total_elapsed = time.perf_counter() - t0
        return LatencyStats.from_durations_ns(durations_ns, total_elapsed)

    async def bench_update(self, memory_ids: list[str]) -> LatencyStats:
        """Measure update latency."""
        durations_ns: list[int] = []
        t0 = time.perf_counter()

        for mem_id in memory_ids:
            start = time.perf_counter_ns()
            await self.client.update(mem_id, f"Updated configuration value for {mem_id}")
            durations_ns.append(time.perf_counter_ns() - start)

        total_elapsed = time.perf_counter() - t0
        return LatencyStats.from_durations_ns(durations_ns, total_elapsed)

    async def bench_delete(self, memory_ids: list[str]) -> LatencyStats:
        """Measure delete latency."""
        durations_ns: list[int] = []
        t0 = time.perf_counter()

        for mem_id in memory_ids:
            start = time.perf_counter_ns()
            await self.client.delete(mem_id)
            durations_ns.append(time.perf_counter_ns() - start)

        total_elapsed = time.perf_counter() - t0
        return LatencyStats.from_durations_ns(durations_ns, total_elapsed)

    async def bench_recall(
        self, probes: list[LabeledQueryProbe], k: int = 5
    ) -> RetrievalQualityStats:
        """Compute Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)."""
        precisions: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []

        for probe in probes:
            result = await self.client.search(
                probe.query,
                limit=k,
                **probe.scope_kwargs,
            )
            retrieved_ids = [m.id for m in result.results]

            matched = probe.expected_memory_id in retrieved_ids
            p_at_k = 1.0 / k if matched else 0.0
            r_at_k = 1.0 if matched else 0.0

            rr = 0.0
            if matched:
                rank = retrieved_ids.index(probe.expected_memory_id) + 1
                rr = 1.0 / rank

            precisions.append(p_at_k)
            recalls.append(r_at_k)
            reciprocal_ranks.append(rr)

        avg_p = statistics.mean(precisions) if precisions else 0.0
        avg_r = statistics.mean(recalls) if recalls else 0.0
        mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

        return RetrievalQualityStats(
            precision_at_k=round(avg_p, 4),
            recall_at_k=round(avg_r, 4),
            mrr=round(mrr, 4),
            k=k,
            total_queries=len(probes),
        )

    async def run_full_suite(
        self, corpus_size: int = 100, concurrency: int = 5
    ) -> BenchmarkReport:
        """Execute the complete performance and quality benchmark suite."""
        corpus = self.generator.generate_corpus(corpus_size)
        probes = self.generator.generate_query_probes(corpus, num_probes=min(50, corpus_size // 2))

        # 1. Measure Add
        add_stats = await self.bench_add(corpus)

        # 2. Measure Single Search
        search_stats = await self.bench_search(probes)

        # 3. Measure Concurrent Search
        concurrent_probes = probes * max(1, (concurrency * 2) // max(1, len(probes)))
        concurrent_stats = await self.bench_concurrent_search(concurrent_probes, concurrency=concurrency)

        # 4. Measure Quality (Recall@k, Precision@k, MRR)
        quality_stats = await self.bench_recall(probes, k=5)

        # 5. Measure Update on subset
        update_ids = [m.id for m in corpus[: max(5, corpus_size // 10)]]
        update_stats = await self.bench_update(update_ids)

        # 6. Measure Delete on subset
        delete_ids = [m.id for m in corpus[-max(5, corpus_size // 10) :]]
        delete_stats = await self.bench_delete(delete_ids)

        return BenchmarkReport(
            backend=self.backend_name,
            corpus_size=corpus_size,
            concurrency=concurrency,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            add_latency=add_stats,
            search_latency=search_stats,
            update_latency=update_stats,
            delete_latency=delete_stats,
            concurrent_search_qps=concurrent_stats,
            retrieval_quality=quality_stats,
            metadata={
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "os": os.name,
            },
        )


# ---------------------------------------------------------------------------
# Report Serialization Helpers
# ---------------------------------------------------------------------------


def export_reports_to_json_and_csv(
    reports: list[BenchmarkReport], json_path: Path, csv_path: Path
) -> None:
    """Save benchmark reports to structured JSON and flat tabular CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    data = [r.to_dict() for r in reports]
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Write CSV
    rows = []
    for r in reports:
        for op, stats in [
            ("add", r.add_latency),
            ("search", r.search_latency),
            ("update", r.update_latency),
            ("delete", r.delete_latency),
            ("concurrent_search", r.concurrent_search_qps),
        ]:
            rows.append(
                {
                    "Operation": op,
                    "Backend": r.backend,
                    "Corpus size": r.corpus_size,
                    "Concurrency": r.concurrency,
                    "p50 (ms)": stats.p50_ms,
                    "p95 (ms)": stats.p95_ms,
                    "p99 (ms)": stats.p99_ms,
                    "Throughput (QPS)": stats.throughput_qps,
                    "Recall@k": r.retrieval_quality.recall_at_k,
                    "Precision@k": r.retrieval_quality.precision_at_k,
                    "MRR": r.retrieval_quality.mrr,
                    "Notes": f"Samples: {stats.count}, Timestamp: {r.timestamp_utc}",
                }
            )

    fieldnames = [
        "Operation",
        "Backend",
        "Corpus size",
        "Concurrency",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
        "Throughput (QPS)",
        "Recall@k",
        "Precision@k",
        "MRR",
        "Notes",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuum Long-Term Memory Benchmark Harness")
    parser.add_argument(
        "--backend",
        choices=["mock", "qdrant", "milvus"],
        default="mock",
        help="Memory backend to profile (default: mock)",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Corpus sizes to profile (e.g. 100 1000 10000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency level for throughput testing (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic corpus generation (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast verification mode with small scale (size=20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/reports",
        help="Directory to save benchmark reports (default: tests/reports)",
    )

    args = parser.parse_args()

    scales = [20] if args.quick else args.scales

    if args.backend == "mock":
        provider = InMemoryBenchmarkProvider()
        client = MemoryClient(provider=provider)
    elif args.backend == "qdrant":
        from continuum.memory.config import MemoryConfig

        config = MemoryConfig(provider="mem0", vector_db_provider="qdrant", enabled=True)
        client = MemoryClient(config=config)
    elif args.backend == "milvus":
        from continuum.memory.config import MemoryConfig

        config = MemoryConfig(provider="mem0", vector_db_provider="milvus", enabled=True)
        client = MemoryClient(config=config)
    else:
        raise ValueError(f"Unknown backend {args.backend}")

    harness = MemoryBenchmarkHarness(client=client, backend_name=args.backend, seed=args.seed)

    print("=" * 70)
    print(f"CONTINUUM MEMORY SUBSYSTEM BENCHMARK [{args.backend.upper()}]")
    print(f"Scales: {scales} | Concurrency: {args.concurrency} | Seed: {args.seed}")
    print("=" * 70)

    reports: list[BenchmarkReport] = []
    for s in scales:
        print(f"\n[*] Running benchmark suite with corpus size {s}...")
        report = asyncio.run(harness.run_full_suite(corpus_size=s, concurrency=args.concurrency))
        reports.append(report)

        print(f"  + Add Latency     : p50={report.add_latency.p50_ms}ms, p95={report.add_latency.p95_ms}ms, QPS={report.add_latency.throughput_qps}")
        print(f"  + Search Latency  : p50={report.search_latency.p50_ms}ms, p95={report.search_latency.p95_ms}ms, QPS={report.search_latency.throughput_qps}")
        print(f"  + Concurrent QPS  : {report.concurrent_search_qps.throughput_qps} QPS (p95={report.concurrent_search_qps.p95_ms}ms)")
        print(f"  + Retrieval Quality: Recall@5={report.retrieval_quality.recall_at_k}, Precision@5={report.retrieval_quality.precision_at_k}, MRR={report.retrieval_quality.mrr}")

    out_dir = Path(args.output_dir)
    json_path = out_dir / "memory_benchmark_baselines.json"
    csv_path = out_dir / "memory_benchmark_baselines.csv"
    export_reports_to_json_and_csv(reports, json_path, csv_path)
    print(f"\n[+] Results successfully exported to:\n    - {json_path}\n    - {csv_path}")


if __name__ == "__main__":
    main()
