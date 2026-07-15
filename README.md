<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/shyftlabs/continuum/main/docs/assets/continuum-logo-dark.png" />
  <img src="https://raw.githubusercontent.com/shyftlabs/continuum/main/docs/assets/continuum-logo.png" alt="Continuum" width="460" />
</picture>

##### by **[ShyftLabs](https://shyftlabs.io/)**

### The agent runtime for builders who ship.

Build, run, and deploy reliable AI agents at enterprise scale — multi-LLM routing, persistent memory, MCP-native tools, durable workflows, and full observability, out of the box.

<br />

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-0a0a0a.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-0a0a0a.svg?style=for-the-badge)](https://github.com/shyftlabs/continuum/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/shyftlabs-continuum?style=for-the-badge&color=0a0a0a&label=pypi)](https://pypi.org/project/shyftlabs-continuum/)

[![CI](https://img.shields.io/github/actions/workflow/status/shyftlabs/continuum/ci.yml?branch=main&label=CI&logo=github)](https://github.com/shyftlabs/continuum/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-continuum.shyftlabs.io-blue?logo=readthedocs&logoColor=white)](https://docs.continuum.shyftlabs.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/shyftlabs/continuum/blob/main/CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-v2.1-ff69b4.svg)](https://github.com/shyftlabs/continuum/blob/main/CODE_OF_CONDUCT.md)

[**📖 Documentation**](https://docs.continuum.shyftlabs.io/) · [**⚡ Quick start**](#-quick-start) · [**⚙️ Configuration**](#️-configuring-continuum) · [**🧩 Components**](#-components) · [**🧪 Examples**](#-examples) · [**🤝 Contributing**](https://github.com/shyftlabs/continuum/blob/main/CONTRIBUTING.md)

</div>

---

**Continuum** is a production-grade Python framework for building, orchestrating, and shipping autonomous AI agents at enterprise scale. It unifies a clean, typed agent core with cost-aware multi-model inference, stateful long- and short-term memory, open standards-based tool calling, durable execution, and end-to-end observability — all behind one small, composable, type-safe API.

## ✨ Features

- 🤖 **Agentic core & orchestration** — a strongly-typed agent primitive with full lifecycle hooks, schema-validated structured outputs, and nine composable multi-agent patterns (sequential, parallel, loop, routing, planning, reflection, debate, scatter, supervised).
- 🔀 **Smart Inference** — cost-aware inference routing that classifies every request by complexity and dispatches it to the cheapest capable model, with seamless cross-provider failover and zero lock-in.
- 🧠 **Stateful memory** — persistent semantic long-term recall plus low-latency working memory, with multi-tenant isolation scopes and built-in PII redaction for privacy-by-default agents.
- 🔌 **Open tool calling** — plug into any standards-based tool ecosystem (Model Context Protocol) across multiple transports, with fine-grained capability scoping, context capture/injection, and rich generative-UI artifacts.
- 🔁 **Durable execution** — long-running, crash- and restart-safe agent workflows with human-in-the-loop approval gates and exactly-once guarantees.
- 🔭 **Full observability** — first-class distributed tracing, token/latency/error telemetry, and one-line function instrumentation for complete run transparency.
- 🌐 **Model-agnostic** — target frontier or open-weight models through a single model string; swap providers without touching agent code.
- 🤝 **Multi-agent handoffs** — context-preserving agent-to-agent delegation with history summarization, cycle detection, and depth control.
- 📡 **Real-time streaming** — token-, tool-, handoff-, and memory-level events streamed the moment they happen.
- ✅ **Built-in evaluation** — turn live production traces into golden datasets and regression-test agent quality with standard LLM-evaluation metrics.

## 🚀 Quick start

**Requirements:** Python 3.13+ and Docker (for Redis · Qdrant/Milvus · Langfuse).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install shyftlabs-continuum

continuum up                 # start local infra (Redis + Qdrant); writes ./.env
echo "OPENAI_API_KEY=sk-…" >> .env   # add your provider key(s) — see Configuration below
```

`continuum up` ships with the package — it locates the bundled Docker stack and starts
it for you, so there's no compose file to find or copy. It defaults to the **minimal**
profile (Redis + Qdrant); pick a bigger one with `continuum up standard` / `continuum up full`.

> **Contributors** working from a clone: `git clone https://github.com/shyftlabs/continuum.git && cd continuum`,
> then `python3.13 -m venv .venv && source .venv/bin/activate`, `pip install -e ".[dev]"`,
> `cp .env.template .env`, and `continuum up`.

#### Infrastructure profiles

| Command | Services started | Use it when |
|---|---|---|
| `continuum up` *(minimal)* | Redis + Qdrant (2 containers) | Day-to-day development — a stateful agent with memory, nothing heavy. |
| `continuum up standard` | minimal + Langfuse stack (8) | You want tracing/observability in the [Langfuse UI](http://localhost:3000). |
| `continuum up full` | everything (13), incl. Temporal + Milvus | Durable workflows (Temporal) or the Milvus vector store. |

Each profile also writes a managed block to `./.env` (`VECTOR_STORE_PROVIDER`, `LANGFUSE_ENABLED`, …)
so the SDK only talks to services that are actually running. Other commands:
`continuum down [-v]`, `continuum status`, `continuum logs [service] [-f]`, `continuum config-path`.

**Port conflicts?** Every published host port is overridable via `.env` (defaults shown):
`SESSION_REDIS_PORT=6380`, `QDRANT_PORT=6333`, `QDRANT_GRPC_PORT=6334`, `MILVUS_PORT=19530`,
`LANGFUSE_WEB_PORT=3000`, `LANGFUSE_WORKER_PORT=3030`, `LANGFUSE_POSTGRES_PORT=5433`,
`LANGFUSE_REDIS_PORT=6382`, `CLICKHOUSE_HTTP_PORT=8123`, `CLICKHOUSE_NATIVE_PORT=9000`,
`MINIO_API_PORT=9090`, `MINIO_CONSOLE_PORT=9091`, `TEMPORAL_PORT=7233`, `TEMPORAL_UI_PORT=8233`,
`TEMPORAL_POSTGRES_PORT=5434`. For the stores the SDK connects to (`QDRANT_PORT`, `MILVUS_PORT`,
`SESSION_REDIS_PORT`), the same variable drives both the container and the client, so they stay in sync.

Your first agent:

```python
import asyncio
from continuum.agent import BaseAgent, AgentRunner

async def main():
    agent = BaseAgent(
        name="hello-agent",
        instructions="You are a friendly assistant.",
        model="gpt-4o-mini",
    )
    runner = AgentRunner()
    response = await runner.run(agent, "Hi!")
    print(response.content)

asyncio.run(main())
```

`AgentRunner.run()` returns an `AgentResponse` with `content`, `structured_output`, `usage`, `tool_calls`, `run_artifacts`, `latency_ms`, and the full handoff chain. See the [**docs**](https://docs.continuum.shyftlabs.io/) for streaming, tools/MCP, memory, handoffs, and workflows.

## ⚙️ Configuring Continuum

Continuum is configured through environment variables (copy [`.env.template`](https://github.com/shyftlabs/continuum/blob/main/.env.template) → `.env`). Set keys only for the providers and components you use — everything else has sensible defaults. The most common settings:

#### LLM providers & routing

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Provider API keys — set the one(s) you use | `sk-…` |
| `DEFAULT_LLM_MODEL` | Default model (`provider/model`, or bare name for OpenAI) | `gemini/gemini-2.5-flash` |
| `FALLBACK_LLM_MODEL` | Model used if the default fails | `gpt-4o-mini` |
| `LLM_ENABLE_FALLBACK` | Automatically fall back on provider errors | `true` |
| `SMART_LAYER_ENABLED` | Enable cost-aware tier routing (Smart Inference) | `true` |

#### Memory (long-term) & embeddings

| Variable | Description | Example |
|---|---|---|
| `MEMORY_ENABLED` | Enable mem0-backed long-term memory | `true` |
| `VECTOR_STORE_PROVIDER` | Vector store backend | `qdrant` / `milvus` |
| `EMBEDDER_PROVIDER` / `EMBEDDER_MODEL` | Embedding provider & model | `openai` / `text-embedding-3-small` |
| `MEMORY_ISOLATION` | Scope of memory isolation | `user` / `agent` / `run` / `shared` |

#### Sessions (short-term)

| Variable | Description | Example |
|---|---|---|
| `SESSION_ENABLED` | Enable Redis-backed conversation sessions | `true` |
| `SESSION_REDIS_HOST` / `SESSION_REDIS_PORT` | Redis connection | `localhost` / `6380` |
| `SESSION_TTL_SECONDS` | Session lifetime | `172800` |

#### Observability (Langfuse)

| Variable | Description | Example |
|---|---|---|
| `LANGFUSE_ENABLED` | Enable tracing | `true` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse credentials | `pk-…` / `sk-…` |
| `LANGFUSE_HOST` | Langfuse endpoint | `http://localhost:3000` |

#### Temporal (optional, durable workflows)

| Variable | Description | Example |
|---|---|---|
| `TEMPORAL_ENABLED` | Enable durable workflow orchestration | `false` |
| `TEMPORAL_HOST` | Temporal frontend | `localhost:7233` |

> Optional extras: `pip install -e ".[temporal]"` for Temporal, `".[eval]"` for evaluation, `".[embeddings]"` for local embeddings, `".[headroom-local]"` for in-process Headroom compression (`".[headroom-local-ml]"` adds ML prose compression). See [`.env.template`](https://github.com/shyftlabs/continuum/blob/main/.env.template) for the complete, annotated reference.

## 🧩 Components

| Component | What it does |
|---|---|
| **Agents** | `BaseAgent` + `AgentRunner` — config, hooks, structured outputs, ReAct |
| **Workflows** | Nine multi-agent patterns for chaining, branching, looping, and self-improvement |
| **Smart Inference** | Request classifier + cost-aware model routing with fallback |
| **Memory** | mem0 + Qdrant/Milvus (long-term) · Redis (sessions) · multi-tenant scopes |
| **Tools / MCP** | MCP servers over Stdio/SSE/StreamableHTTP, tool filtering, widget artifacts |
| **Temporal** | Durable, restart-safe workflows with human-in-the-loop gates |
| **Observability** | Langfuse traces, metrics, `@observe` decorators |
| **Evaluation** | Golden datasets + DeepEval / RAGAS metrics |

## 📚 Documentation

Full documentation lives at **[docs.continuum.shyftlabs.io](https://docs.continuum.shyftlabs.io/)** — guides for building & running agents, Smart Inference, memory, tools/MCP, workflows, handoffs, streaming, evaluation, and the research behind it.

Markdown sources are also in [`docs/`](https://github.com/shyftlabs/continuum/tree/main/docs) if you prefer reading on GitHub — e.g. [`agent.md`](https://github.com/shyftlabs/continuum/blob/main/docs/agent.md), [`memory.md`](https://github.com/shyftlabs/continuum/blob/main/docs/memory.md), [`tools.md`](https://github.com/shyftlabs/continuum/blob/main/docs/tools.md), and the integration [`GUIDE.md`](https://github.com/shyftlabs/continuum/blob/main/docs/GUIDE.md).

## 🧪 Examples

Runnable demos live under [`playground/`](https://github.com/shyftlabs/continuum/tree/main/playground):

- **`gateway-local-shop`** — an MCP server + agent + chat UI for a pet-shop assistant (end-to-end: server → agent → UI).
- **`gateway-multi-agent-shop`** — a multi-agent workflow variant with routing and handoffs.
- **`frontend/`** — the demo web UIs (`assortment`, `commerce-chat`).

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](https://github.com/shyftlabs/continuum/blob/main/CONTRIBUTING.md) for the branch model, Conventional Commits, DCO sign-off, and local setup. By participating you agree to our [Code of Conduct](https://github.com/shyftlabs/continuum/blob/main/CODE_OF_CONDUCT.md).

- 🐛 **Bugs & features** — use the [issue templates](https://github.com/shyftlabs/continuum/tree/main/.github/ISSUE_TEMPLATE)
- 💬 **Questions & ideas** — [GitHub Discussions](https://github.com/shyftlabs/continuum/discussions)
- 🔒 **Security** — report privately via [`SECURITY.md`](https://github.com/shyftlabs/continuum/blob/main/SECURITY.md), never a public issue

## 📄 License

Licensed under the [Apache License, Version 2.0](https://github.com/shyftlabs/continuum/blob/main/LICENSE). Copyright © 2025–2026 [ShyftLabs Inc.](https://shyftlabs.io/)

For commercial / enterprise inquiries — SLAs, indemnification, hosted offerings, custom features — contact **[continuum@shyftlabs.io](mailto:continuum@shyftlabs.io)**.

<div align="center">
<br />
<sub>Built with ❤️ by <a href="https://shyftlabs.io/">ShyftLabs</a> · <a href="mailto:continuum@shyftlabs.io">continuum@shyftlabs.io</a></sub>
</div>
