# Continuum Hackathon Quick-Start

Continuum (`shyftlabs-continuum`) is an async Python framework for building AI agents. You define agents with instructions and tools, wire them together into multi-agent workflows, and run them with a single `runner.run()` call. This doc gets you from zero to a working agent and covers everything you need to build something meaningful in a hackathon window.

---

## 1. Setup

### Prerequisites

- **Python 3.13** — required. Use `pyenv install 3.13` or `uv python install 3.13` if needed.
- **An LLM API key** — OpenAI, Anthropic, or Gemini. `OPENAI_API_KEY` is required at startup even if you use Claude or Gemini (the memory system uses it for embeddings).
- **Docker** — only needed if you want session history (Redis) or long-term memory (Milvus). You can skip Docker entirely for a stateless prototype.

### Install

```bash
python3.13 -m venv .venv    
source .venv/bin/activate
pip install shyftlabs-continuum
```

### Configure

Copy `.env.template` to `.env` and fill in your API keys. The full working configuration:

```bash
# ── LLM Provider Keys ────────────────────────────────────────────────────────
OPENAI_API_KEY=your-openai-api-key        # required (also used for embeddings)
GEMINI_API_KEY=your-gemini-api-key        # if using Gemini
# ANTHROPIC_API_KEY=your-anthropic-api-key  # if using Claude

# ── Default LLM ──────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL=gemini/gemini-2.5-flash
FALLBACK_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.7
DEFAULT_LLM_MAX_TOKENS=4096
LLM_REQUEST_TIMEOUT=60
LLM_MAX_RETRIES=3
LLM_ENABLE_FALLBACK=true

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDER_PROVIDER=openai
EMBEDDER_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536

# ── Memory (mem0 + Milvus) ───────────────────────────────────────────────────
MEMORY_ENABLED=true
VECTOR_STORE_PROVIDER=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=
MILVUS_COLLECTION=orchestrator_memories
MEMORY_LLM_MODEL=gemini/gemini-2.5-flash
MEMORY_LLM_TEMPERATURE=0.1
MEMORY_ISOLATION=user
MEMORY_SEARCH_LIMIT=5
MEMORY_HISTORY_DB_PATH=~/.orchestrator/memory_history.db

# ── Session (Redis) ───────────────────────────────────────────────────────────
SESSION_ENABLED=true
SESSION_REDIS_HOST=localhost
SESSION_REDIS_PORT=6380
SESSION_REDIS_PASSWORD=sdk123456789
SESSION_REDIS_DB=0
SESSION_REDIS_SSL=false
SESSION_TTL_SECONDS=172800
SESSION_MAX_MESSAGES=1000
SESSION_KEY_PREFIX=orchestrator:session

# ── Langfuse Observability ────────────────────────────────────────────────────
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_FLUSH_INTERVAL=1
LANGFUSE_FLUSH_AT=15
LANGFUSE_DEBUG=false

# ── Temporal (durable workflows) ─────────────────────────────────────────────
TEMPORAL_ENABLED=true
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
TEMPORAL_ENABLE_HUMAN_IN_LOOP=true
TEMPORAL_APPROVAL_TIMEOUT_SECONDS=86400
TEMPORAL_WORKFLOW_EXECUTION_TIMEOUT=604800
TEMPORAL_ACTIVITY_START_TO_CLOSE_TIMEOUT=300
TEMPORAL_ACTIVITY_RETRY_MAX_ATTEMPTS=3

# ── Misc ──────────────────────────────────────────────────────────────────────
ENVIRONMENT=development
LOG_LEVEL=INFO
SHARED_SERVICES_ENABLED=true
MEM0_TELEMETRY=false
ANONYMIZED_TELEMETRY=false
TOKENIZERS_PARALLELISM=false
```

Start services:

```bash
docker compose up -d
```

### Verify

```python
import asyncio
from orchestrator.agent import BaseAgent, AgentRunner

async def main():
    agent = BaseAgent(name="test", instructions="You are helpful.")
    response = await AgentRunner().run(agent, "Say hello in one sentence.")
    print(response.content)

asyncio.run(main())
```

---

## 2. Example Projects

Two complete runnable examples are in `playground/`:

| Project | What it shows |
| ------- | ------------- |
| [local-shop](../playground/local-shop) | Single agent with MCP tools (HTTP transport) |
| [multi-agent-shop](../playground/multi-agent-shop) | Multi-agent workflow: parallel search, routing, synthesis |

Run them with `python -m playground.local_shop.cli` or `python -m playground.multi_agent_shop.cli`. Read `agent.py` / `agents.py` and `workflows.py` before building your own — they show real patterns you can copy from.

---

## 3. Models & Providers

Provider routing is automatic based on the model name — no configuration needed.


| Model name prefix           | Provider      | Examples                                           |
| --------------------------- | ------------- | -------------------------------------------------- |
| `claude-…` or `anthropic/…` | Anthropic     | `claude-sonnet-4-5`, `claude-opus-4-5`             |
| `gemini/…`                  | Google Gemini | `gemini/gemini-2.0-flash`, `gemini/gemini-1.5-pro` |
| anything else               | OpenAI        | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o3-mini`        |


Set a default in `.env`:

```bash
DEFAULT_LLM_MODEL=gpt-4o-mini
```

Or pass `model=` per agent (overrides the default):

```python
agent = BaseAgent(name="agent", instructions="...", model="claude-sonnet-4-5")
```

---

## 4. Your First Agent

```python
import asyncio
from orchestrator import AgentConfig, AgentMemoryConfig
from orchestrator.agent import BaseAgent, AgentRunner

async def main():
    agent = BaseAgent(
        name="assistant",
        instructions="You are a helpful assistant. Be concise.",
        model="gpt-4o-mini",
        # disable memory and session for a stateless agent
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )
    runner = AgentRunner()
    response = await runner.run(agent, "What are three uses for Python?")
    print(response.content)     # the text reply
    print(response.usage)       # token counts
    print(response.latency_ms)  # timing

asyncio.run(main())
```

### Stateless vs stateful

```python
BaseAgent(
    name="my-agent",
    instructions="...",
    memory_config=AgentMemoryConfig(search_memories=True, store_memories=True),
    config=AgentConfig(log_to_session=True, session_history_turns=None),
)
```

Based on your needs, your agent can be stateless or stateful; if stateless, disable long-term and short-term memories; if stateful, you may only need either long-term memories(Memory) or short-term memories(Session) or both, you can control them by setting the parameters in `AgentConfig` and `AgentMemoryConfig`

**Memory (default):**

- `search_memories=True` — looks up `long-term memories` before responding
- `store_memories=True` — saves `long-term memories` after responding

**Session (default):**

- `log_to_session=True` — saves to session history (`short-term memory`)
- `session_history_turns=None` — loads last 20 turns of history (`short-term memory`)

#### `session_history_turns` behaviour


| Value            | Behavior                                    |
| ---------------- | ------------------------------------------- |
| `None` (default) | load last **20** turns from Redis           |
| `0`              | **skip** Redis call entirely — load nothing |
| `5`              | load last **5** turns from Redis            |


If you want session history to work (load prior turns, save new ones), you must create a session before calling `runner.run()`, because you need create or get session id as a redis key to save or load short-term memory.

```python
# Step 1: create session
session_id = await session_client.get_or_create_session(
    session_id=session_id,
    user_id="user-123",
    conversation_id="conv-456",   # optional — see below
)

# Step 2: run
response = await runner.run(
    agent=agent,
    input="Hello!",
    session_id=session_id,
    user_id="user-123",
)
```

> If you pass a `session_id` that was never created, the runner will not crash — but messages will silently fail to save and history will not load.

#### How `session_id` is computed

`get_or_create_session()` computes a deterministic session ID based on what you pass:


| Arguments                     | Computed `session_id`             |
| ----------------------------- | --------------------------------- |
| explicit `session_id`         | used as-is                        |
| `conversation_id` + `user_id` | `c:{conversation_id}:u:{user_id}` |
| `user_id` only                | `u:{user_id}`                     |
| neither                       | random UUID                       |


#### What is `conversation_id`

Take chat UI projects as an example, if you have multiple chat windows, use `conversation_id` when you want to keep separate chat windows per user:

- Without `conversation_id`: one session per user (`u:{user_id}`) — all conversations share the same history
- With `conversation_id`: one session per conversation (`c:{conversation_id}:u:{user_id}`) — each conversation has its own isolated history

**You need to customize `conversation_id` based on your projects:**

- Chat UI projects (e.g. multiple chat windows per user): Generate `conversation_id` on the backend when the user creates a new conversation (POST /conversations), and return only the `conversation_id` to the frontend. The frontend passes it back with each message. `get_or_create_session` will use `conversation_id` and `user_id` to generate `session_id` at the first time.
- Non-chat projects (task-based, webhook-triggered, background jobs): There is no chat window. Instead, you may use your natural entity ID (e.g. ticket ID, invoice ID, job ID) as `conversation_id`. Generate it on the backend at entity creation time. Each independent task gets its own session ID — never reuse session IDs across unrelated tasks.

### For a complete single-agent pipeline example, see [playground/local-shop](../playground/local-shop).

---

## 5. Multi-Agent Patterns

All workflow agents are in `orchestrator.agent`. For complete multi-agent examples, see [playground/multi-agent-shop](../playground/multi-agent-shop).

### Session saving

Every workflow agent such as Sequential, Parallel calls `runner.run()` one or more times internally — one per sub-agent or iteration. By default, `runner.run()` auto-saves each turn to session history, which would result in noisy intermediate turns the user never saw.

To prevent this, every workflow agent must:

**1. Set** `suppress_session_log = True` **at the start of** `execute()`  

```python
async def execute(self, input_text, runner, context) -> AgentResponse:
    context.suppress_session_log = True  # blocks auto-saving for all sub-agent runs
    ...
    response = await runner.run(
        agent=agent,
        input=current_input,
        context=context,   # same context passed to every runner.run()
    )
```

The same `context` object is passed to every `runner.run()` call. Inside `run_finalizer.py`, each run checks `context.suppress_session_log` and skips saving if `True`.

**2. Call `save_turn()` once at the end**

```python
await runner.save_turn(
    session_id=context.session_id,
    user_message=input_text,           # what user originally sent
    assistant_message=final_output,    # what user actually sees
)
```

This saves exactly one clean turn — the original input and the final output — to session history.

> **If you build a custom workflow agent, you must follow the same pattern.** Forgetting `suppress_session_log = True` will cause every sub-agent turn to be saved to session history.

### Deciding which agent's output is the final response

In a multi-agent workflow, you must explicitly decide which agent's output is the final response during a request — this is what you pass to `save_turn()`.

Two common patterns:

**Pipeline (e.g. Sequential):** agents run one after another, each passing output to the next. The *last* agent produces the final response:

```python
await runner.save_turn(session_id, user_input, last_agent_response.content)
```

**Handoff / Router:** sub-agents do work and their results are injected back into the top-level agent's message list. The top-level agent then synthesizes and generates its own final response — so the *top-level agent's* output is what to save, not the sub-agents' intermediate results:

```python
await runner.save_turn(session_id, user_input, top_level_agent_response.content)
```

> If you save an intermediate agent's output by mistake, session history will contain turns the user never saw — and future turns will load them as prior context.

### Built-in workflow implementations

Built-in workflow agents are provided in `src/orchestrator/agent/workflow/`:

```
sequential.py  parallel.py  loop.py  reflection.py
planner.py     router.py    scatter.py  supervised.py  debate.py
```

You can refer to `playground/multi-agent-shop/workflows.py` and `playground/multi-agent-shop/agents.py` as usage examples.

### Sequential — pipeline

Each agent's output becomes the next agent's input. Good for research → write → review flows.

```python
from orchestrator.agent import BaseAgent, AgentRunner, create_sequential_agent

researcher = BaseAgent(name="researcher", instructions="Research the topic. Output key facts.", model="gpt-4o-mini")
writer     = BaseAgent(name="writer",     instructions="Write a short report from the research.", model="gpt-4o-mini")
editor     = BaseAgent(name="editor",     instructions="Polish the report for clarity.", model="gpt-4o-mini")

pipeline = create_sequential_agent(
    name="research-pipeline",
    agents=[researcher, writer, editor],
)

runner = AgentRunner()
response = await runner.run(pipeline, "The impact of AI on healthcare")
print(response.content)  # editor's final output
```

### Router — dispatch to a specialist

The router reads the input and sends it to the most relevant agent. Strategy can be `"llm"`, `"rule_based"`, or `"hybrid"`.

```python
from orchestrator.agent import BaseAgent, AgentRunner, create_router_agent

billing   = BaseAgent(name="billing-agent",   instructions="Handle billing and payment questions.")
technical = BaseAgent(name="technical-agent", instructions="Handle technical support questions.")
general   = BaseAgent(name="general-agent",   instructions="Handle general questions.")

router = create_router_agent(
    name="triage",
    routes=[
        ("billing-agent",   "billing, invoice, payment, subscription, refund"),
        ("technical-agent", "bug, error, crash, not working, how to"),
    ],
    fallback="general-agent",
    strategy="hybrid",
)

runner = AgentRunner(agent_registry={
    "billing-agent": billing,
    "technical-agent": technical,
    "general-agent": general,
})
response = await runner.run(router, "My payment failed twice this week")
print(response.content)
```

### Handoff — mid-conversation delegation

An agent hands off to another agent when it encounters something outside its scope. The target agent returns control with its result.

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.agent.types import Handoff

specialist = BaseAgent(
    name="specialist",
    instructions="You are a tax specialist. Answer tax questions accurately.",
)

triage = BaseAgent(
    name="triage",
    instructions=(
        "You handle general customer questions. "
        "If the question involves taxes, hand off to the specialist."
    ),
    handoffs=[
        Handoff(target_agent="specialist", description="Tax-related questions")
    ],
)

runner = AgentRunner(agent_registry={"specialist": specialist})
response = await runner.run(triage, "How do I handle capital gains on crypto?")
print(response.content)           # specialist's answer
print(response.agents_involved)   # ["triage", "specialist"]
```

### Loop — iterate until done

Run an agent in a loop until a termination condition is met.

```python
from orchestrator.agent import BaseAgent, AgentRunner, create_loop_agent
from orchestrator.agent.types import TerminationType

refiner = BaseAgent(
    name="refiner",
    instructions=(
        "Improve the given text. Make it clearer and more concise. "
        "If the text is already good, say DONE and output it unchanged."
    ),
)

loop = create_loop_agent(
    name="refinement-loop",
    agent=refiner,
    termination_type=TerminationType.OUTPUT_MATCH,
    termination_pattern=r"\bDONE\b",
    max_iterations=3,
)

response = await AgentRunner().run(loop, "This is a very long and complicated sentence that could benefit from some editing.")
print(response.content)
```

Other termination types: `TerminationType.LLM_DECISION` (ask an LLM "are we done?"), `TerminationType.TOOL_CALL` (stop when agent calls a specific tool), `TerminationType.MAX_ITERATIONS` (always run N times).

### Parallel — same input, all agents at once

```python
from orchestrator.agent import create_parallel_agent
from orchestrator.agent.types import MergeStrategy

parallel = create_parallel_agent(
    name="parallel-analysts",
    agents=[analyst_a, analyst_b, analyst_c],
    merge_strategy=MergeStrategy.CONCATENATE,   # or LLM_SUMMARIZE, STRUCTURED_DICT
)
```

### Build your own workflow instead of using built-ins

We recommend defining and building your own workflows using `BaseAgent` directly, rather than relying solely on the built-in workflow agents, because workflows are closely tied to your project's business logic — a custom agent gives you full control over the flow, session saving, and memory behaviour. You may want some agents to be stateless and the others to be stateful in a workflow, you must control them by yourself based on your actual and specific requirements of projects.

**Example: `ParallelCoordinatorAgent` in `playground/multi-agent-shop/workflows.py`**

Instead of using `ParallelAgent` directly, the pet shop defines a custom `BaseAgent` subclass that orchestrates the parallel search and synthesis steps manually:

```python
class ParallelCoordinatorAgent(BaseAgent):
    synthesiser: BaseAgent | None = None
    parallel: ParallelAgent | None = None

    async def execute(self, input_text, runner, context, llm_client=None) -> AgentResponse:
        context.suppress_session_log = True

        # Step 1: run parallel workers with a fresh stateless context (no history, no save)
        parallel_ctx = create_run_context(user_id=context.user_id, conversation_id=context.conversation_id)
        parallel_result = await self.parallel.execute(input_text, runner, parallel_ctx)

        # Step 2: synthesiser builds the user-facing reply using session history + memory
        # suppress_session_log=True blocks auto-save; context carries session_id so
        # message_builder loads history and memory normally.
        final = await runner.run(agent=self.synthesiser, input=synthesis_input, context=context)

        # Step 3: save one clean turn
        await runner.save_turn(session_id=context.session_id, user_message=input_text, assistant_message=final.content)
```

This gives the parallel workers (Parallel workflow based on several Base Agents) a clean stateless context (no Redis calls) while the synthesiser (based on a Base Agent) still loads session history and memory — something the built-in `ParallelAgent` cannot do out of the box.

---

## 6. Tools (MCP)

Agents use tools via the [Model Context Protocol](https://modelcontextprotocol.io). Connect any MCP server and the agent will discover and call its tools automatically.

### Connect to an HTTP MCP server

```python
from orchestrator.tools import MCPServerStreamableHttp, ToolExecutor, MCPUtil
from orchestrator.agent import BaseAgent, AgentRunner

server = MCPServerStreamableHttp(
    {"url": "https://your-mcp-server.com/mcp"},
    name="my-tools",
)
await server.connect()

executor = ToolExecutor({server: None})    # None = expose all tools
await executor.initialize()

agent = BaseAgent(
    name="tool-agent",
    instructions="Use the available tools to answer questions.",
    mcp_servers=[server],
)
response = await AgentRunner().run(agent, "What is the weather in Toronto?")
```

### Connect to a stdio MCP server

```python
from orchestrator.tools import MCPServerStdio

server = MCPServerStdio(
    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-fetch"]},
    name="fetch",
)
await server.connect()
```

### Pass tools explicitly (if you need the tool definitions)

```python
tool_defs = await MCPUtil.get_function_tools(server)
tools = [t.model_dump() for t in tool_defs]

agent = BaseAgent(
    name="agent",
    instructions="...",
    tools=tools,
    tool_executor=executor,
)
```

---

## 7. Long-term Memory Management (mem0)

### Controlling what gets stored

`**store_memories**` — master on/off switch. If `False`, nothing goes to mem0.

`**extraction_prompt**` — tells mem0 what facts to extract from the conversation. By default mem0 decides. Override it to be specific:

```python
AgentMemoryConfig(
    store_memories=True,
    extraction_prompt=(
        "Only extract long-term facts about the user's pets, animal preferences, "
        "and dietary needs. Do NOT store transient actions like adding to cart or searches."
    ),
)
```

`**pre_store_filter**` — runs after mem0 stores facts. Facts not returned by the filter are **deleted from mem0**. Use it to remove PII or irrelevant facts that slipped through extraction:

```python
def my_filter(facts: list[str]) -> list[str]:
    return [f for f in facts if "credit card" not in f]

AgentMemoryConfig(
    store_memories=True,
    pre_store_filter=my_filter,
)
```

The full flow:

```
conversation → extraction_prompt extracts facts → stored in mem0
             → pre_store_filter runs → blocked facts deleted from mem0
```

### Memory management

Get `memory_client` from the container:

```python
from orchestrator.core.container import get_container
memory_client = get_container().memory_client
```

**View memories:**

```python
# get all memories for a user
memories = await memory_client.get_all(user_id="user-123")
for m in memories:
    print(m.memory, m.id)

# search by query
results = await memory_client.search("pet preferences", user_id="user-123")
```

**Delete memories:**

```python
# delete a specific memory
await memory_client.delete(memory_id="abc-123")

# delete ALL memories for a user
await memory_client.delete_all(user_id="user-123")
```

> Consider exposing memory management to your frontend — let users view and delete what the AI remembers about them. This is important for privacy compliance (GDPR "right to be forgotten") and user trust and experience.

---

## 8. Common Mistakes


| Mistake                                          | Fix                                                                                                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `redis.exceptions.ConnectionError`               | Redis port is **6380**, not 6379. Set `SESSION_REDIS_PORT=6380`.                                                                                  |
| `Failed to initialize mem0: Missing credentials` | `OPENAI_API_KEY` is required even if you use Claude/Gemini. Set it, or add `MEMORY_ENABLED=false`.                                                |
| Agent runs but never calls tools                 | Did you call `await server.connect()` and `await executor.initialize()`? Both are required.                                                       |
| Session history not loading                      | You must call `get_or_create_session()` first. Passing a `session_id` that was never created silently fails.                                      |
| Noisy session history in multi-agent workflows   | Set `context.suppress_session_log = True` in your workflow's `execute()`, then call `runner.save_turn()` once at the end. See `docs/GUIDE.md §3`. |
| `Route(target=...)` import error                 | The old API is removed. Use `create_router_agent(routes=[("agent-name", "description"), ...])` — tuples, not `Route` objects.                     |
| All intermediate agent responses saved to Redis  | In a sequential or custom workflow, all sub-agents save independently by default. Use `suppress_session_log` pattern from `docs/GUIDE.md §3`.     |
| `ModuleNotFoundError: orchestrator`              | Your venv isn't active. Run `source .venv/bin/activate`.                                                                                          |


---

## 9. Full Reference


| What you need                                                                              | Where to look                                                                                                   |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| All `BaseAgent` fields and defaults                                                        | [docs/agent.md](agent.md)                                                                                       |
| Workflow agents (Sequential, Parallel, Loop, Router, Planner, Debate, Scatter, Reflection) | [docs/agent.md §6](agent.md)                                                                                    |
| MCP server setup, tool filters, context injection                                          | [docs/tools.md](tools.md)                                                                                       |
| Session config, Redis setup                                                                | [docs/session.md](session.md)                                                                                   |
| Memory mechanism and issues                                                                | [docs/update-docs/memory-issue-analysis.md](update-docs/memory-issue-analysis.md) & [docs/memory.md](memory.md) |
| Provider routing, structured outputs, context compression                                  | [docs/llm.md](llm.md)                                                                                           |
| Temporal durable workflows (WaitStep, ConditionalStep, etc.)                               | [docs/temporal/](temporal/)                                                                                     |
| Env vars reference                                                                         | [docs/installation.md](installation.md)                                                                         |
| All features in one table                                                                  | [docs/features.md](features.md)                                                                                 |


