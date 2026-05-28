<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/continuum-logo-dark.png" />
  <img src="docs/assets/continuum-logo.png" alt="Continuum" width="460" />
</picture>

##### by **[ShyftLabs](https://shyftlabs.io/)**

### The agent runtime for builders who ship.

Build, run, and deploy reliable AI agents at enterprise scale — multi-LLM routing, persistent memory, MCP-native tools, durable workflows, and full observability, out of the box.

<br />

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-0a0a0a.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-0a0a0a.svg?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-0a0a0a.svg?style=for-the-badge)](https://github.com/shyftlabs/continuum/releases)

[![CI](https://img.shields.io/github/actions/workflow/status/shyftlabs/continuum/ci.yml?branch=main&label=CI&logo=github)](https://github.com/shyftlabs/continuum/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-continuum.shyftlabs.io-blue?logo=readthedocs&logoColor=white)](https://docs.continuum.shyftlabs.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-v2.1-ff69b4.svg)](CODE_OF_CONDUCT.md)

[**📖 Documentation**](https://docs.continuum.shyftlabs.io/) · [**⚡ Quick start**](#-quick-start) · [**⚙️ Configuration**](#️-configuring-continuum) · [**🧩 Components**](#-components) · [**🧪 Examples**](#-examples) · [**🤝 Contributing**](CONTRIBUTING.md)

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

**Requirements:** Python 3.13+ and Docker (for Redis · Milvus/Qdrant · Langfuse).

```bash
git clone https://github.com/shyftlabs/continuum.git
cd continuum

python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.template .env        # add your provider key(s) — see Configuration below
docker compose up -d         # Redis · Milvus/Qdrant · Langfuse
```

Your first agent:

```python
import asyncio
from orchestrator.agent import BaseAgent, AgentRunner

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

Continuum is configured through environment variables (copy [`.env.template`](.env.template) → `.env`). Set keys only for the providers and components you use — everything else has sensible defaults. The most common settings:

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

> Optional extras: `pip install -e ".[temporal]"` for Temporal, `".[eval]"` for evaluation, `".[embeddings]"` for local embeddings. See [`.env.template`](.env.template) for the complete, annotated reference.

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

Markdown sources are also in [`docs/`](docs/) if you prefer reading on GitHub — e.g. [`agent.md`](docs/agent.md), [`memory.md`](docs/memory.md), [`tools.md`](docs/tools.md), and the integration [`GUIDE.md`](docs/GUIDE.md).

## 🧪 Examples

Runnable demos live under [`playground/`](playground/):

- **`gateway-local-shop`** — an MCP server + agent + chat UI for a pet-shop assistant (end-to-end: server → agent → UI).
- **`gateway-multi-agent-shop`** — a multi-agent workflow variant with routing and handoffs.
- **`frontend/`** — the demo web UIs (`assortment`, `commerce-chat`).

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branch model, Conventional Commits, DCO sign-off, and local setup. By participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

- 🐛 **Bugs & features** — use the [issue templates](.github/ISSUE_TEMPLATE)
- 💬 **Questions & ideas** — [GitHub Discussions](https://github.com/shyftlabs/continuum/discussions)
- 🔒 **Security** — report privately via [`SECURITY.md`](SECURITY.md), never a public issue

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright © 2025–2026 [ShyftLabs Inc.](https://shyftlabs.io/)

For commercial / enterprise inquiries — SLAs, indemnification, hosted offerings, custom features — contact **[continuum@shyftlabs.io](mailto:continuum@shyftlabs.io)**.

<div align="center">
<br />
<sub>Built with ❤️ by <a href="https://shyftlabs.io/">ShyftLabs</a> · <a href="mailto:continuum@shyftlabs.io">continuum@shyftlabs.io</a></sub>
</div>
