# Continuum Agent Framework

**Agentic Framework for Enterprise-Wide Execution**

Built by [Bhavik Ardeshna](mailto:bhavik@shyftlabs.io) at [ShyftLabs](https://shyftlabs.io)

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/shyftlabs/continuum)

---

## Table of Contents

- [What is Continuum?](#what-is-continuum)
- [Why Continuum?](#why-continuum)
- [Architecture](#architecture)
- [Core Concepts](#core-concepts)
- [Features](#features)
  - [Multi-LLM Provider Support](#multi-llm-provider-support)
  - [Two-Tier Memory System](#two-tier-memory-system)
  - [MCP-Native Tool Integration](#mcp-native-tool-integration)
  - [Agent Handoff System](#agent-handoff-system)
  - [Observability & Tracing](#observability--tracing)
  - [Structured Outputs](#structured-outputs)
  - [Prompt Engineering](#prompt-engineering)
- [Workflow Agents](#workflow-agents)
- [Durable Workflows with Temporal](#durable-workflows-with-temporal)
- [Evaluation Framework](#evaluation-framework)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Infrastructure & Deployment](#infrastructure--deployment)
- [Configuration Reference](#configuration-reference)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [License](#license)

---

## What is Continuum?

Continuum is an agentic framework for enterprise-wide execution, built from the ground up in Python. It provides a comprehensive foundation for building, deploying, and managing intelligent AI agents that can reason, use tools, remember context, collaborate with other agents, and operate reliably in enterprise production environments.

Unlike prototype-level agent frameworks, Continuum is designed with enterprise requirements at its core: persistent memory across conversations, durable workflow execution that survives process crashes, full observability with distributed tracing, multi-LLM provider support across 100+ models, and a batteries-included infrastructure stack deployable with a single Docker Compose command.

## Why Continuum?

- **Enterprise Memory Architecture** — Two-tier memory combining Redis-backed short-term sessions with semantic long-term memory (mem0 + Qdrant), featuring four isolation scopes for multi-tenant safety.
- **Multi-LLM Provider Support** — Access 100+ models from OpenAI, Anthropic, Google, Azure, AWS Bedrock, Cohere, Ollama, and more through a unified LiteLLM-powered interface.
- **MCP-Native Tool Integration** — Built on the Model Context Protocol from day one. Any MCP-compatible server works out of the box with zero adapters.
- **Durable Workflow Orchestration** — Optional Temporal integration for crash-proof, long-running multi-agent workflows with human-in-the-loop approval gates.
- **Full Observability** — Built-in Langfuse integration for distributed tracing, metrics collection, and error reporting across the entire agent lifecycle.
- **Self-Hosted & Secure** — Complete Docker Compose stack for self-hosted deployments, keeping all data within your own infrastructure.

### Design Principles

- **Code-First** — Agents are Python dataclasses. No YAML configs, no drag-and-drop builders, no hidden magic.
- **Async-Native** — Built on asyncio from the ground up. Every I/O operation is non-blocking.
- **Modular Architecture** — Each layer is independently configurable and replaceable via protocol-based abstractions.
- **Production-Ready** — Health checks, circuit breakers, graceful shutdown, dependency injection, and comprehensive error handling are built in.

---

## Architecture

Continuum is organized into clearly separated layers:

```
┌─────────────────────────────────────────────────────┐
│                  Application Layer                   │
│   BaseAgent  ·  AgentRunner  ·  Workflow Agents      │
├─────────────────────────────────────────────────────┤
│                  Execution Layer                     │
│   LLMClient (LiteLLM)  ·  ToolExecutor (MCP)        │
│   Handoff Manager  ·  Structured Output Parser       │
├─────────────────────────────────────────────────────┤
│                State Management Layer                │
│   SessionClient (Redis)  ·  MemoryClient (mem0+Qdrant)│
├─────────────────────────────────────────────────────┤
│                Observability Layer                   │
│   TracingManager (Langfuse)  ·  Metrics  ·  Logs     │
├─────────────────────────────────────────────────────┤
│           Optional: Durable Workflow Layer            │
│   Temporal Integration  ·  Approval Gates  ·  Signals │
├─────────────────────────────────────────────────────┤
│                  Infrastructure                      │
│   Docker Compose  ·  DI Container  ·  Health Checks   │
└─────────────────────────────────────────────────────┘
```

### Source Layout

```
src/orchestrator/
├── agent/          # BaseAgent, AgentRunner, Handoffs, Workflow agents
├── llm/            # LLMClient, LLMConfig, context management
├── memory/         # Long-term memory (mem0 + Qdrant vector DB)
├── session/        # Short-term conversation history (Redis)
├── tools/          # MCP server integration, ToolExecutor, filtering
├── observability/  # Langfuse tracing, metrics, error reporting
├── core/           # DI container, lifecycle management, health checks
├── temporal/       # Durable workflow engine (Temporal)
├── evaluation/     # Evaluation framework (DeepEval, RAGAS)
├── config.py       # Global settings (pydantic-settings)
├── protocols.py    # Protocol definitions (ILLMClient, IMemoryClient, etc.)
└── exceptions.py   # Exception hierarchy
```

---

## Core Concepts

### BaseAgent

The `BaseAgent` dataclass is the fundamental abstraction. It encapsulates identity, model configuration, tools, handoffs, memory settings, structured output schemas, lifecycle hooks, and prompt engineering features.

| Attribute | Description |
|---|---|
| `name` | Unique identifier (alphanumeric with hyphens/underscores) |
| `instructions` | System prompt with `{template_var}` support |
| `model` | LLM model via LiteLLM (e.g., `gpt-4o`, `claude-sonnet-4-20250514`, `gemini/gemini-2.5-flash`) |
| `tools` | Tool definitions available to the agent |
| `mcp_servers` | MCP servers for tool discovery and execution |
| `handoffs` | Agent-to-agent transition definitions |
| `memory_config` | Memory search/store behavior and scopes |
| `output_schema` | Pydantic model or JSON schema for validated output |
| `config` | Max turns, reasoning mode, termination conditions |
| `template_vars` | Static variables for `{slot}` placeholders |
| `examples` | Few-shot examples (`[{input, output}]`) |
| `instruction_modifiers` | Dynamic prompt modification callables |
| `on_start/on_end/on_error/on_tool_call/on_handoff` | Lifecycle hooks |

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.config import AgentMemoryConfig
from orchestrator.agent.types import Handoff, MemoryScope

support_agent = BaseAgent(
    name="support-agent",
    instructions="You are a helpful customer support agent for {company}.",
    model="gpt-4o",
    temperature=0.3,
    tools=[search_kb_tool, create_ticket_tool],
    handoffs=[
        Handoff(
            target_agent="billing-agent",
            description="Hand off billing-related inquiries",
        ),
    ],
    memory_config=AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,
    ),
    template_vars={"company": "Acme Corp"},
    examples=[
        {"input": "How do I reset my password?",
         "output": "Navigate to Settings > Security > Reset Password."},
    ],
)
```

### AgentRunner

The execution engine that manages the complete lifecycle: LLM calls, tool execution, handoffs, memory retrieval/storage, and observability tracing. Supports both synchronous (`run`) and streaming (`run_stream`) execution.

```python
runner = AgentRunner()

# Non-streaming
response = await runner.run(
    support_agent,
    "I need help with my billing.",
    user_id="user-123",
    session_id="session-456",
)
print(response.content)

# Streaming
async for event in runner.run_stream(support_agent, "Hello!"):
    if event.type == EventType.CONTENT_DELTA:
        print(event.data["content"], end="")
```

### Execution Flow

1. Initialize context and services via the DI container
2. Retrieve conversation history (Redis) and relevant memories (Qdrant)
3. Build the message array: system prompt + history + memories + user input
4. Call the LLM with tool definitions
5. If tool calls returned → execute via MCP ToolExecutor → feed results back
6. If handoff returned → summarize history → transfer to target agent
7. Store new memories and update session
8. Return `AgentResponse` with optional validated `structured_output`

---

## Features

### Multi-LLM Provider Support

Unified interface to 100+ models via LiteLLM. Switching models requires only changing the model string.

| Provider | Example Model | Auth |
|---|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Azure OpenAI | `azure/gpt-4o` | `AZURE_API_KEY` |
| AWS Bedrock | `bedrock/anthropic.claude-3` | AWS credentials |
| Cohere | `command-r-plus` | `COHERE_API_KEY` |
| Ollama (local) | `ollama/llama3` | None (local) |

**Automatic Compatibility** — Handles provider-specific quirks transparently (e.g., auto-disabling JSON mode with tools on Gemini). Built-in token-bucket rate limiter.

**Context Management** — Automatic compression when approaching token limits (configurable threshold, default 80%). Preserves recent messages while summarizing older ones.

### Two-Tier Memory System

**Short-Term (Redis Sessions)** — Fast conversation history with configurable TTL (default 7 days), max message limits, and namespace prefixes for multi-tenant isolation.

**Long-Term (mem0 + Qdrant)** — Intelligent fact extraction with semantic vector search. Automatically extracts key facts and retrieves relevant memories on subsequent interactions.

**Memory Isolation Scopes:**

| Scope | Description |
|---|---|
| `SHARED` | Accessible by all agents and users. Global knowledge base. |
| `USER` | Scoped to a specific user across all agents. Default scope. |
| `AGENT` | Scoped to a specific agent across all users. |
| `RUN` | Scoped to a single execution run. Ephemeral. |

```python
from orchestrator.agent.config import AgentMemoryConfig
from orchestrator.agent.types import MemoryScope

memory_config = AgentMemoryConfig(
    search_memories=True,
    store_memories=True,
    search_scope=MemoryScope.USER,
    store_scope=MemoryScope.USER,
)
```

### MCP-Native Tool Integration

Built on the Model Context Protocol from day one. Three transport types:

| Transport | Use Case |
|---|---|
| `MCPServerStdio` | Local process-based MCP servers |
| `MCPServerSse` | Legacy SSE-based remote servers |
| `MCPServerStreamableHttp` | Modern HTTP-based remote servers (recommended) |

```python
from orchestrator.tools import MCPServerStdio, MCPServerStreamableHttp

local_tools = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
)

remote_tools = MCPServerStreamableHttp(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer token"},
)

agent = BaseAgent(
    name="tool-agent",
    instructions="You have access to filesystem and API tools.",
    mcp_servers=[local_tools, remote_tools],
)
```

Supports tool filtering, context state sharing, and artifact capture.

### Agent Handoff System

Dynamic agent-to-agent transitions with history transfer modes (full, summary, recent-N, hybrid), automatic cycle detection, depth tracking, and configurable return-to-parent behavior.

```python
triage = BaseAgent(
    name="triage-agent",
    instructions="Route customer inquiries to the right specialist.",
    handoffs=[
        Handoff(target_agent="billing-agent", description="Billing and payment issues"),
        Handoff(target_agent="technical-agent", description="Technical support"),
        Handoff(target_agent="sales-agent", description="Product inquiries"),
    ],
)
```

### Observability & Tracing

Built-in Langfuse integration with automatic tracing across the entire agent lifecycle.

- **TracingManager** — Automatic span creation for all operations with configurable sample rates.
- **MetricsCollector** — Latency, token usage, error rates, and custom metrics.
- **ErrorReporter** — Automatic error reporting with full execution context.
- **`@observe` Decorator** — Easy custom instrumentation for any function.

```python
from orchestrator.observability import observe

@observe(name="data-processing", capture_output=True)
async def process_data(input_data: dict) -> dict:
    result = await transform(input_data)
    return result
```

### Structured Outputs

Validated structured outputs using Pydantic models or raw JSON schemas.

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]
    summary: str

analyst = BaseAgent(
    name="analyst",
    instructions="Analyze the sentiment and topics of the input.",
    output_schema=AnalysisResult,
)

response = await runner.run(analyst, "Customer feedback text...")
result: AnalysisResult = response.structured_output
```

### Prompt Engineering

First-class prompt engineering with three built-in mechanisms:

- **Template Variables** — `{slot}` placeholders resolved from `template_vars`, `RunContext` metadata, and system values (`{user_id}`, `{session_id}`, `{date}`).
- **Few-Shot Examples** — Input/output pairs automatically injected into the system prompt.
- **Instruction Modifiers** — Callables that dynamically modify prompts based on runtime context.

```python
def add_tier_note(prompt: str, ctx: RunContext) -> str:
    tier = ctx.metadata.get("user_tier", "free")
    if tier == "enterprise":
        return prompt + "\nThis is an enterprise user. Prioritise SLA."
    return prompt

agent = BaseAgent(
    name="adaptive-agent",
    instructions="You are helping {user_name}. Today is {date}.",
    template_vars={"user_name": "Alice"},
    instruction_modifiers=[add_tier_note],
)
```

---

## Workflow Agents

Nine specialized workflow patterns for multi-agent orchestration:

| Workflow Agent | Description |
|---|---|
| `RouterAgent` | LLM-based dynamic routing/triage to specialist agents |
| `SequentialAgent` | Pipeline execution — output of each agent feeds the next |
| `ParallelAgent` | Concurrent execution with result merging |
| `LoopAgent` | Iterative execution until termination condition met |
| `ReflectionAgent` | Self-critique with iterative quality improvement |
| `PlannerAgent` | Dynamic multi-step planning with replanning |
| `DebateAgent` | Pro/con/judge synthesis pattern |
| `ScatterAgent` | LLM splits input into slices for parallel processing |
| `SupervisedAgent` | Sequential pipeline with LLM quality gating per step |

```python
from orchestrator.agent.workflow import create_sequential_agent

researcher = BaseAgent(name="researcher", instructions="Research the topic thoroughly.")
writer = BaseAgent(name="writer", instructions="Write a polished article.")
editor = BaseAgent(name="editor", instructions="Edit for clarity and style.")

pipeline = create_sequential_agent(
    name="content-pipeline",
    agents=[researcher, writer, editor],
)

response = await AgentRunner().run(pipeline, "AI in healthcare")
```

---

## Durable Workflows with Temporal

Optional [Temporal](https://temporal.io) integration for crash-proof, long-running agent workflows.

### Install

```bash
pip install -e ".[temporal]"
```

### Workflow Primitives

| Step Type | Description |
|---|---|
| `AgentStep` | Execute a registered BaseAgent |
| `ApprovalStep` | Pause for human approval |
| `ParallelStep` | Execute multiple agents concurrently |
| `ConditionalStep` | Branch based on prior results |
| `WaitStep` | Pause for duration or signal |

### Human-in-the-Loop

Approval gates with configurable timeout, escalation, approval/rejection with comments, and webhook notifications.

### Example

```python
from orchestrator.agent import BaseAgent
from orchestrator.temporal import (
    AgentWorkflow, WorkflowInput,
    get_agent_registry, get_temporal_client, get_worker_manager,
)

registry = get_agent_registry()
registry.register(BaseAgent(name="summarizer", instructions="Summarize the input."))
registry.register(BaseAgent(name="reviewer", instructions="Review for accuracy."))

client = get_temporal_client()
await client.connect()
await get_worker_manager().start()

handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "summarizer"},
            {"type": "approval", "description": "Review before publishing"},
            {"type": "agent", "agent_name": "reviewer"},
        ],
        initial_input="Temporal is a durable execution platform...",
    ),
    id="my-workflow",
    task_queue="orchestrator-agents",
)
result = await handle.result()
```

### Features

- **Agent-agnostic**: any `BaseAgent` works as a workflow step
- **Declarative steps**: sequential, parallel, conditional, loop, wait, approval
- **Human-in-the-loop**: approval gates with notifications, escalation, timeout
- **Fault-tolerant**: automatic retries, durable state, workflow cancellation
- **Docker Compose**: Temporal server, UI, and Postgres included

See the [Temporal docs](docs/temporal/) for the full guide.

---

## Evaluation Framework

Built-in evaluation for systematic agent quality measurement:

- **DeepEval Integration** — Criterion-based evaluation with customizable metrics.
- **RAGAS Framework** — LLM evaluation metrics for RAG pipelines.
- **EvaluatorAgent** — Specialized agent for evaluating other agents' outputs.
- **Golden Datasets** — Build evaluation datasets from Langfuse traces.

---

## Installation

### Prerequisites

- **Python 3.13** (required)
- **Docker & Docker Compose** (for infrastructure services)
- **LLM API Key** (at least one provider, e.g., `OPENAI_API_KEY`)

### Environment Setup

```bash
# Using pyenv (Recommended)
pyenv install 3.13.9
pyenv virtualenv 3.13.9 continuum-sdk
pyenv activate continuum-sdk

# Using venv
python3.13 -m venv continuum-sdk
source continuum-sdk/bin/activate

# Using conda
conda create -n continuum-sdk python=3.13
conda activate continuum-sdk
```

### Install the Framework

```bash
git clone https://github.com/shyftlabs/continuum.git
cd continuum

# Standard install
pip install -e .

# With optional extras
pip install -e ".[temporal]"     # Durable workflows
pip install -e ".[eval]"         # Evaluation framework
pip install -e ".[embeddings]"   # Local embeddings
```

### Verify Installation

```bash
python -c "from orchestrator import __version__; print(f'Continuum version: {__version__}')"
python -c "from orchestrator.agent import BaseAgent, AgentRunner; from orchestrator.llm import LLMClient; print('All imports successful')"
```

### Troubleshooting

```bash
pip install --upgrade "aiohttp>=3.13.2"
pip install -e . --upgrade --force-reinstall
```

---

## Quick Start

```python
import asyncio
from orchestrator.agent import BaseAgent, AgentRunner

async def main():
    agent = BaseAgent(
        name="my-agent",
        instructions="You are a helpful assistant.",
    )

    runner = AgentRunner()
    response = await runner.run(
        agent,
        "Hello! What can you help me with?",
        user_id="user-123",
    )

    print(response.content)

asyncio.run(main())
```

---

## Infrastructure & Deployment

### Docker Compose Stack

A single command brings up the entire infrastructure:

```bash
docker compose up -d
```

| Service | Purpose | Default Port |
|---|---|---|
| Langfuse | Observability & tracing UI | 3000 |
| Qdrant | Vector database (long-term memory) | 6333 |
| Redis | Session storage & state management | 6380 |
| PostgreSQL | Langfuse data store | 5432 |
| ClickHouse | Langfuse analytics engine | 8123 |
| Temporal* | Durable workflow engine | 7233 |
| Temporal UI* | Workflow management dashboard | 8080 |

*\* Temporal services are optional.*

### Health Checks

Built-in health probes for all infrastructure dependencies:

```python
from orchestrator.core.health import check_health

health = await check_health()
# {'redis': {'status': 'healthy'}, 'qdrant': {'status': 'healthy'}, ...}
```

### Dependency Injection & Lifecycle

The DI container lazily initializes services (LLMClient, MemoryClient, SessionClient, TracingManager) and the `OrchestratorLifecycle` class manages graceful startup/shutdown with shared or standalone service modes.

---

## Configuration Reference

All configuration via environment variables (`.env` file via pydantic-settings):

### LLM

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Default LLM model |
| `FALLBACK_LLM_MODEL` | `gemini/gemini-1.5-flash` | Fallback model |
| `DEFAULT_LLM_TEMPERATURE` | `0.7` | Default temperature |
| `DEFAULT_LLM_MAX_TOKENS` | `4096` | Max output tokens |
| `LLM_REQUEST_TIMEOUT` | `60` | Request timeout (sec) |
| `LLM_MAX_RETRIES` | `3` | Max retry attempts |

### Memory

| Variable | Default | Description |
|---|---|---|
| `MEMORY_ENABLED` | `true` | Enable long-term memory |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `EMBEDDER_PROVIDER` | `openai` | Embedding provider |
| `EMBEDDER_MODEL` | `text-embedding-3-small` | Embedding model |
| `MEMORY_ISOLATION` | `user` | Default isolation scope |
| `MEMORY_SEARCH_LIMIT` | `5` | Memories to retrieve |

### Session

| Variable | Default | Description |
|---|---|---|
| `SESSION_ENABLED` | `true` | Enable session management |
| `SESSION_REDIS_HOST` | `localhost` | Redis host |
| `SESSION_REDIS_PORT` | `6380` | Redis port |
| `SESSION_TTL_SECONDS` | `604800` | Session TTL (7 days) |
| `SESSION_MAX_MESSAGES` | `1000` | Max messages per session |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_ENABLED` | `true` | Enable Langfuse tracing |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | Trace sampling rate |

### Context Management

| Variable | Default | Description |
|---|---|---|
| `CONTEXT_MANAGEMENT_ENABLED` | `true` | Enable auto-compression |
| `CONTEXT_COMPRESSION_THRESHOLD` | `0.8` | Compress at 80% capacity |
| `CONTEXT_KEEP_RECENT_MESSAGES` | `10` | Recent messages to keep |

### Temporal (Optional)

| Variable | Default | Description |
|---|---|---|
| `TEMPORAL_ENABLED` | `false` | Enable Temporal workflows |
| `TEMPORAL_HOST` | `localhost:7233` | Temporal server |
| `TEMPORAL_ENABLE_HUMAN_IN_LOOP` | `true` | Enable approval gates |
| `TEMPORAL_APPROVAL_TIMEOUT_SECONDS` | `86400` | Approval timeout (24h) |

---

## Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | >= 3.13 | Runtime (required) |
| LiteLLM | >= 1.71.0 | Multi-LLM provider abstraction |
| Pydantic | >= 2.10.0 | Data validation & structured outputs |
| pydantic-settings | >= 2.6.0 | Environment-based configuration |
| mem0ai | >= 1.0.0 | Long-term memory with fact extraction |
| qdrant-client | >= 1.16.0 | Vector database for semantic memory |
| redis | >= 5.0.0 | Session storage & state |
| langfuse | >= 2.57.0 | Observability & tracing |
| mcp | >= 1.23.0 | Model Context Protocol |
| aiohttp | >= 3.13.2 | Async HTTP client |

### Optional Dependencies

| Package | Install Extra | Purpose |
|---|---|---|
| temporalio >= 1.23.0 | `[temporal]` | Durable workflow engine |
| sentence-transformers >= 2.2.0 | `[embeddings]` | Local embedding models |
| cohere >= 5.0.0 | `[cohere]` | Cohere embeddings |
| deepeval >= 1.0.0 | `[eval]` | Evaluation framework |
| ragas >= 0.2.0 | `[eval]` | RAG evaluation metrics |

---

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Installation Guide](docs/installation.md) — Setup and configuration
- [Agent Module](docs/agent.md) — Agent creation and execution
- [LLM Module](docs/llm.md) — Multi-provider LLM client
- [Memory Module](docs/memory.md) — Long-term memory
- [Session Module](docs/session.md) — Conversation history
- [Observability](docs/observability.md) — Tracing and metrics
- [Tools](docs/tools.md) — MCP integration
- [Core](docs/core.md) — Container and lifecycle
- [Temporal Integration](docs/temporal/) — Durable workflow orchestration

---

## About

**Author:** Bhavik Ardeshna ([bhavik@shyftlabs.io](mailto:bhavik@shyftlabs.io))

**Company:** [ShyftLabs Inc.](https://shyftlabs.io)

**Website:** [continuum.shyftlabs.io](https://continuum.shyftlabs.io)

## License

Proprietary. Copyright (c) 2025-2026 ShyftLabs Inc. All rights reserved.

See [LICENSE](LICENSE) for full terms.
