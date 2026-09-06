"""
Unit tests for the long-term memory benchmark harness.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from continuum.memory.client import MemoryClient
from tests.benchmarks.memory.harness import (
    BenchmarkReport,
    CorpusGenerator,
    InMemoryBenchmarkProvider,
    LatencyStats,
    MemoryBenchmarkHarness,
    RetrievalQualityStats,
    export_reports_to_json_and_csv,
)

pytestmark = pytest.mark.unit


class TestCorpusGenerator:
    """Tests for synthetic corpus and query probe generator."""

    def test_generate_corpus_size(self):
        generator = CorpusGenerator(seed=123)
        corpus = generator.generate_corpus(50)
        assert len(corpus) == 50
        assert corpus[0].id == "mem-000000"
        assert corpus[49].id == "mem-000049"

    def test_generate_corpus_determinism(self):
        gen1 = CorpusGenerator(seed=42)
        gen2 = CorpusGenerator(seed=42)
        c1 = gen1.generate_corpus(20)
        c2 = gen2.generate_corpus(20)
        assert [m.text for m in c1] == [m.text for m in c2]
        assert [m.keywords for m in c1] == [m.keywords for m in c2]

    def test_generate_query_probes(self):
        generator = CorpusGenerator(seed=42)
        corpus = generator.generate_corpus(30)
        probes = generator.generate_query_probes(corpus, num_probes=5)
        assert len(probes) == 5
        for p in probes:
            assert p.query.startswith("What is the preferred")
            assert p.expected_memory_id.startswith("mem-")
            assert "user_id" in p.scope_kwargs


class TestLatencyStats:
    """Tests for latency distribution calculations."""

    def test_empty_durations(self):
        stats = LatencyStats.from_durations_ns([], 1.0)
        assert stats.count == 0
        assert stats.p50_ms == 0.0
        assert stats.throughput_qps == 0.0

    def test_known_percentiles(self):
        # 100 samples from 1ms to 100ms
        durations_ns = [i * 1_000_000 for i in range(1, 101)]
        stats = LatencyStats.from_durations_ns(durations_ns, total_elapsed_s=0.5)

        assert stats.count == 100
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert stats.p50_ms == 50.0
        assert stats.p95_ms == 95.0
        assert stats.p99_ms == 99.0
        assert stats.throughput_qps == 200.0


class TestInMemoryBenchmarkProvider:
    """Tests for lightweight benchmark mock provider."""

    @pytest.mark.asyncio
    async def test_crud_lifecycle(self):
        provider = InMemoryBenchmarkProvider()
        client = MemoryClient(provider=provider)

        # Add
        add_res = await client.add("Developer uses Python and uv", user_id="user-1")
        assert add_res.message == "Memory added successfully"

        # Search
        search_res = await client.search("Python", user_id="user-1")
        assert len(search_res.results) >= 1
        mem_id = search_res.results[0].id

        # Update
        updated = await client.update(mem_id, "Developer uses Python 3.13 and uv")
        assert "3.13" in updated.memory

        # Delete
        deleted = await client.delete(mem_id)
        assert deleted is True

        # Search after delete
        empty_res = await client.search("Python", user_id="user-1")
        assert len(empty_res.results) == 0


class TestMemoryBenchmarkHarness:
    """Tests for benchmark harness coordination."""

    @pytest.mark.asyncio
    async def test_full_suite_execution(self):
        provider = InMemoryBenchmarkProvider()
        client = MemoryClient(provider=provider)
        harness = MemoryBenchmarkHarness(client=client, backend_name="mock", seed=99)

        report = await harness.run_full_suite(corpus_size=15, concurrency=2)

        assert isinstance(report, BenchmarkReport)
        assert report.backend == "mock"
        assert report.corpus_size == 15
        assert report.add_latency.count == 15
        assert report.search_latency.count > 0
        assert report.retrieval_quality.recall_at_k >= 0.0
        assert report.retrieval_quality.k == 5

    def test_export_reports_to_json_and_csv(self, tmp_path: Path):
        json_file = tmp_path / "baselines.json"
        csv_file = tmp_path / "baselines.csv"

        dummy_stats = LatencyStats(
            count=10,
            p50_ms=1.2,
            p95_ms=2.5,
            p99_ms=3.0,
            min_ms=0.5,
            max_ms=3.0,
            mean_ms=1.5,
            stddev_ms=0.4,
            throughput_qps=500.0,
        )
        dummy_quality = RetrievalQualityStats(
            precision_at_k=0.2,
            recall_at_k=1.0,
            mrr=0.8,
            k=5,
            total_queries=10,
        )
        dummy_report = BenchmarkReport(
            backend="mock",
            corpus_size=50,
            concurrency=5,
            timestamp_utc="2026-09-06T12:00:00Z",
            add_latency=dummy_stats,
            search_latency=dummy_stats,
            update_latency=dummy_stats,
            delete_latency=dummy_stats,
            concurrent_search_qps=dummy_stats,
            retrieval_quality=dummy_quality,
        )

        export_reports_to_json_and_csv([dummy_report], json_file, csv_file)

        assert json_file.exists()
        assert csv_file.exists()

        # Validate JSON content
        loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(loaded_json) == 1
        assert loaded_json[0]["backend"] == "mock"
        assert loaded_json[0]["add_latency"]["p50_ms"] == 1.2

        # Validate CSV content
        with open(csv_file, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 5  # add, search, update, delete, concurrent_search
            assert reader[0]["Operation"] == "add"
            assert float(reader[0]["p50 (ms)"]) == 1.2
