# Continuum Framework — Deep Dive

## What Is Continuum?

Continuum is a **Python SDK for agentic AI orchestration** built by ShyftLabs. It provides a production-grade foundation for building AI agents with multi-LLM provider support, persistent memory, session management, tool integration via MCP (Model Context Protocol), and full observability. It targets teams that need to go beyond prototyping into reliable, enterprise-ready agent deployments.

---

## Architecture Overview

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
├── agent/          # BaseAgent, AgentRunner, Handoff, workflow agents
├── llm/            # LLMClient, LLMConfig, context management
├── memory/         # Long-term memory (mem0 + Qdrant vector DB)
├── session/        # Short-term conversation history (Redis)
├── tools/          # MCP server integration, ToolExecutor, filtering
├── observability/  # Langfuse tracing and metrics
├── core/           # DI container, lifecycle management, health checks
├── temporal/       # Durable workflow engine (Temporal)
├── evaluation/     # Evaluation and metrics (DeepEval, RAGAS)
└── utils/          # Utility functions
```

---

## Core Concepts

### 1. Agents (`BaseAgent`)

The `BaseAgent` dataclass is the central abstraction. An agent encapsulates:

- **Identity**: name, description, instructions (system prompt)
- **Model config**: model ID (via LiteLLM), temperature, max tokens
- **Tools**: tool definitions + a `ToolExecutor` to run them
- **Handoffs**: routing rules to pass control to other agents
- **Memory config**: what to search, what to store, at which isolation scope
- **Structured output**: optional Pydantic model or JSON schema for validated responses
- **Prompt engineering**: template variables, few-shot examples, instruction modifiers
- **Lifecycle hooks**: `on_start`, `on_end`, `on_error`, `on_tool_call`, `on_handoff`

```python
agent = BaseAgent(
    name="analyst",
    instructions="You are an expert analyst. Focus on {topic}.",
    model="gemini/gemini-2.5-flash",
    temperature=0.3,
    tools=tool_dicts,
    tool_executor=executor,
    memory_config=AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,
    ),
    template_vars={"topic": "market trends"},
    output_schema=AnalysisResult,
)
```

### 2. Agent Runner

`AgentRunner` is the execution engine. Its loop:

1. Initialize context and services via the DI container
2. Retrieve conversation history (session) and relevant memories
3. Build the message array: system prompt + history + memories + user input
4. Call the LLM with tool definitions
5. If LLM returns tool calls → execute via `ToolExecutor` → feed results back
6. If LLM returns a handoff → summarize history → transfer to target agent
7. Store new memories and session messages
8. Return the final `AgentResponse` (with optional `structured_output`)

### 3. Tools (MCP Integration)

Continuum uses the **Model Context Protocol (MCP)** as its universal tool interface. This means any MCP-compatible server — local CLI tools, remote APIs, databases — can be plugged in without writing adapter code.

**Server types supported:**
- `MCPServerStdio` — local process (e.g., `npx @modelcontextprotocol/server-puppeteer`)
- `MCPServerStreamableHttp` — remote HTTP API
- `MCPServerSse` — legacy SSE-based servers

**Tool filtering**: restrict which tools from a server an agent can see using `create_static_tool_filter()`.

**Context state & artifacts**: `ToolExecutor` maintains shared state across calls and captures run artifacts (structured outputs, widgets).

### 4. Handoffs

Agents can hand off control to other agents. A `Handoff` specifies the target agent and how conversation history should be transferred:

- **full** — pass entire history
- **summary** — LLM-generated summary of the conversation so far
- **recent-N** — last N messages only
- **hybrid** — summary + recent messages

This is critical for multi-agent systems where context windows are limited.

### 5. Memory

Continuum has a two-tier memory system:

**Short-term (Session)** — Redis-backed conversation history with configurable TTL. Stores raw messages per `(user_id, session_id)`.

**Long-term (Memory)** — mem0 + Qdrant vector database. The LLM extracts facts from conversations, embeds them, and stores them for semantic retrieval. Four isolation scopes:

| Scope | Isolation | Use Case |
|-------|-----------|----------|
| `SHARED` | Global | Company knowledge base |
| `USER` | Per user | User preferences, history |
| `AGENT` | Per agent | Agent-specific knowledge |
| `RUN` | Per session | Ephemeral, one-time context |

### 6. Workflows

Continuum provides **declarative workflow patterns** for orchestrating multiple agents:

- **Sequential**: agents run one after another, output feeds forward
- **Parallel**: agents run concurrently, results merged via configurable strategy
- **Conditional**: LLM evaluates a condition to pick a branch
- **Loop**: iterative execution with termination condition
- **Approval Gates**: human-in-the-loop checkpoints

Convenience builders:
```python
router = create_router_agent(name="router", agents=[a, b, c])
pipeline = create_sequential_agent(name="pipeline", agents=[a, b, c])
fan_out = create_parallel_agent(name="fan_out", agents=[a, b, c])
refiner = create_loop_agent(name="refiner", agent=a, max_iterations=5)
```

### 7. Durable Workflows (Temporal)

For mission-critical, long-running orchestration, Continuum integrates with **Temporal**:

- Any `BaseAgent` can be used as a workflow activity
- Workflows survive process crashes, network failures, and restarts
- Approval gates with notifications and escalation policies
- Signals and queries for runtime interaction
- Automatic retries with configurable backoff

### 8. Observability

Full **Langfuse** integration provides:

- Distributed tracing across agent runs, LLM calls, and tool executions
- Token usage and cost tracking per model
- Latency metrics
- Per-agent and per-run tracing context
- `@observe` decorator for custom instrumentation

### 9. LLM Client

The `LLMClient` wraps **LiteLLM** for access to 100+ model providers:

- OpenAI, Anthropic, Google Gemini, Azure, AWS Bedrock, local models, etc.
- Streaming and non-streaming responses
- Automatic context compression when approaching token limits
- Model-specific compatibility handling (e.g., auto-disabling JSON mode when tools are present on Gemini)
- Structured output via JSON mode or Pydantic schema validation

### 10. Infrastructure

- **Docker Compose** brings up all backing services: Langfuse, Qdrant, Redis, PostgreSQL, ClickHouse, Temporal
- **DI Container** (`get_container()`) provides singleton access to all clients
- **Lifecycle management** (`initialize_orchestrator()` / `shutdown_orchestrator()`)
- **Health checks** (`check_all_health()`) for Redis, Qdrant, Langfuse, LLM

---

## Differentiating Features

1. **Memory isolation scopes** — built-in multi-tenant memory with 4 isolation levels
2. **MCP-native tool system** — universal tool interface, no custom adapters needed
3. **Automatic model compatibility** — handles provider quirks transparently (e.g., JSON mode + tools conflicts)
4. **Durable workflows via Temporal** — not just orchestration patterns, but crash-proof execution
5. **Prompt engineering primitives** — template vars, few-shot examples, instruction modifiers as first-class features
6. **Lifecycle hooks** — fine-grained control at every stage of agent execution
7. **Full observability stack** — Langfuse integration for tracing, metrics, and cost tracking out of the box
8. **Progressive context compression** — automatic handling of long conversations approaching token limits
9. **Structured output validation** — Pydantic models for type-safe agent responses
10. **Run artifacts** — capture and return structured data from tool executions

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| LLM Access | LiteLLM (100+ providers) |
| Tool Protocol | MCP (Model Context Protocol) |
| Vector DB | Qdrant |
| Memory Engine | mem0 |
| Session Store | Redis |
| Observability | Langfuse |
| Durable Workflows | Temporal |
| Evaluation | DeepEval, RAGAS |
| Infrastructure | Docker Compose |
| Embeddings | OpenAI text-embedding-3-small (configurable) |
