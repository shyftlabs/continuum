<div align="center">

```
 ██████╗  ██████╗  ███╗   ██╗ ████████╗ ██╗ ███╗   ██╗ ██╗   ██╗ ██╗   ██╗ ███╗   ███╗
██╔════╝ ██╔═══██╗ ████╗  ██║ ╚══██╔══╝ ██║ ████╗  ██║ ██║   ██║ ██║   ██║ ████╗ ████║
██║      ██║   ██║ ██╔██╗ ██║    ██║    ██║ ██╔██╗ ██║ ██║   ██║ ██║   ██║ ██╔████╔██║
██║      ██║   ██║ ██║╚██╗██║    ██║    ██║ ██║╚██╗██║ ██║   ██║ ██║   ██║ ██║╚██╔╝██║
╚██████╗ ╚██████╔╝ ██║ ╚████║    ██║    ██║ ██║ ╚████║ ╚██████╔╝ ╚██████╔╝ ██║ ╚═╝ ██║
 ╚═════╝  ╚═════╝  ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚═╝  ╚═══╝  ╚═════╝   ╚═════╝  ╚═╝     ╚═╝
```

#### · · ·   by   **S H Y F T L A B S**   · · ·

**The agent runtime for builders who ship.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-0a0a0a.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-0a0a0a.svg?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-0a0a0a.svg?style=for-the-badge)](https://github.com/shyftlabs/continuum)

[**Documentation**](docs/index.html) · [**Smart Inference**](docs/index.html#sec-smart) · [**Examples**](docs/index.html#sec-examples) · [**Research**](docs/index.html#sec-research) · [shyftlabs.io](https://shyftlabs.io)

</div>

---

Continuum is a Python framework for building, running, and deploying reliable AI agents at enterprise scale — multi-LLM routing through **Smart Inference**, persistent two-tier memory, MCP-native tooling, Temporal-durable workflows, and Langfuse-traced observability, out of the box.

## Quick start

```bash
git clone https://github.com/shyftlabs/continuum.git
cd continuum

python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.template .env       # add your provider keys
docker compose up -d        # Redis · Milvus · Langfuse

python playground/gateway-local-shop/web.py   # → http://localhost:8081
```

## Documentation

Open **[`docs/index.html`](docs/index.html)** in your browser for the full doc site — it's a self-contained SPA with:

| Section | What's inside |
|---|---|
| [Home](docs/index.html#sec-home) | Hero, key features, quickstart, architecture |
| [Build Agents](docs/index.html#sec-build) | `BaseAgent`, hooks, prompts, structured outputs, 9 workflow patterns |
| [Run Agents](docs/index.html#sec-run) | `AgentRunner`, streaming, sessions, context management, FastAPI |
| [Smart Inference](docs/index.html#sec-smart) | Cost-aware routing, classifier, modes (strict/modest/quality) |
| [Components](docs/index.html#sec-components) | Tools/MCP, memory (mem0 + Milvus/Qdrant), Temporal |
| [Integrations](docs/index.html#sec-integrations) | OpenAI · Anthropic · Gemini · Azure · Langfuse |
| [Research](docs/index.html#sec-research) | Tool Attention, context compression, tier classifiers, handoffs, eval |
| [Examples](docs/index.html#sec-examples) | End-to-end Pet Shop walkthrough — MCP server → agent → UI |

Markdown sources also live under [`docs/`](docs/) if you prefer reading on GitHub.

## Playgrounds

- [`playground/gateway-local-shop/`](playground/gateway-local-shop/) — single agent, MCP-backed pet shop, ChatGPT-style web UI, Smart Inference routing.
- [`playground/gateway-multi-agent-shop/`](playground/gateway-multi-agent-shop/) — multi-agent triage / specialist pattern over the same shop.

## License

Proprietary. Copyright © 2025–2026 [ShyftLabs Inc.](https://shyftlabs.io) All rights reserved. See [LICENSE](LICENSE).
