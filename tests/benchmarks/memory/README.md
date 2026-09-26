# Continuum Long-Term Memory Benchmark Suite

Reproducible performance and quality benchmarking harness for Continuum's memory subsystem (src/continuum/memory/).

This suite measures operation latency distributions, throughput (QPS) under concurrency, scaling behavior across corpus sizes, and semantic search retrieval quality.

---

## 1. What is Measured

| Metric | Description | Target / Significance |
| :--- | :--- | :--- |
| **p50, p95, p99 Latency** | Millisecond response time percentiles for dd, search, update, delete. | Evaluates tail latency under realistic loads. |
| **Throughput (QPS)** | Operations completed per second under single and concurrent workers. | Establishes throughput ceiling. |
| **Corpus Scaling** | Latency and memory consumption across scales (100, 1,000, 10,000, 100,000). | Detects sub-linear degradation or memory leaks. |
| **Recall@k** | Fraction of relevant ground-truth memories retrieved in top-$ results (=5$). | Ensures semantic search precision is preserved. |
| **Precision@k** | Fraction of retrieved memories in top-$ that are relevant. | Evaluates query noise and ranking precision. |
| **MRR** | Mean Reciprocal Rank (/\text{rank}$) of the first relevant memory. | Measures how early the correct memory appears. |

---

## 2. Quick Start & Execution

The harness supports three execution modes:
- **mock**: In-memory lexical/semantic provider. Runs immediately without external services or network dependencies (ideal for CI and local verification).
- **qdrant**: Live Qdrant vector database (port 6333).
- **milvus**: Live Milvus vector database (port 19530).

### Fast Verification Mode (< 2s)
`ash
python -m tests.benchmarks.memory.harness --quick
`

### Full Benchmark Run
`ash
# In-memory baseline
python -m tests.benchmarks.memory.harness --backend mock --scales 100 1000 10000 --concurrency 5

# Against real Qdrant cluster
python -m tests.benchmarks.memory.harness --backend qdrant --scales 1000 10000 --concurrency 10

# Against real Milvus cluster
python -m tests.benchmarks.memory.harness --backend milvus --scales 1000 10000 --concurrency 10
`

### Running via Pytest
The suite is excluded from default unit test runs via the @pytest.mark.benchmark marker. To trigger via pytest:
`ash
pytest tests/benchmarks/memory/ -m benchmark
`

---

## 3. Output Artifacts & Baselines

All benchmark runs export results to:
- **JSON**: 	ests/reports/memory_benchmark_baselines.json (Structured, programmatic diffing)
- **CSV**: 	ests/reports/memory_benchmark_baselines.csv (Spreadsheet import ready)

### CSV Column Specification
Operation | Backend | Corpus size | Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (QPS) | Recall@k | Precision@k | MRR | Notes

---

## 4. Backend Comparison: Qdrant vs Milvus Guide

| Feature | Qdrant | Milvus | Continuum Recommendation |
| :--- | :--- | :--- | :--- |
| **Architecture** | Single-binary Rust engine with on-disk HNSW & payload indexing. | Distributed cloud-native engine (etcd, Pulsar/Kafka, MinIO). | **Qdrant** for single-node / edge / rapid developer deployments. |
| **Filter Ergonomics** | Native payload filtering tightly coupled with HNSW vector traversal. | Boolean expression filtering on scalar fields. | **Qdrant** provides cleaner API ergonomics for multi-tenant scopes (user_id, gent_id). |
| **Scale Ceiling** | Tens of millions of vectors per node; horizontal sharding in cluster mode. | Billions of vectors with distributed segment management. | **Milvus** recommended for enterprise clusters exceeding 100M memories. |
| **Memory Footprint** | Low (supports quantized vectors & mmap storage). | Moderate to High (requires MinIO/etcd supporting services). | **Qdrant** minimizes operational overhead for local test stacks. |

---

## 5. Machine Specification Format

When committing official baseline updates, record the host hardware spec in the report metadata:
`json
{
  "cpu": "Intel/AMD x86_64",
  "ram_gb": 32,
  "os": "Ubuntu 24.04 / Windows 11",
  "python": "3.13.x",
  "vector_db_version": "qdrant:v1.13.0"
}
`
