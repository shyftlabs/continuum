# Agent Module

The agent module is the heart of Continuum. It defines the `BaseAgent`
abstraction, the `AgentRunner` execution engine, all nine workflow
patterns (Router/Sequential/Parallel/Loop/Reflection/Planner/Debate/
Scatter/SupervisedSequential), agent-to-agent **handoffs** with
history summarization, durable run state, and the lifecycle hooks that
let you observe or extend everything.

---

## 1 · Quick start

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

`AgentRunner.run()` returns an `AgentResponse` containing `content`,
`structured_output` (if you set `output_schema`), `usage`, `tool_calls`,
`run_artifacts`, `latency_ms`, and the full handoff chain.

---

## 2 · `BaseAgent`

`from orchestrator.agent import BaseAgent`

A `@dataclass` describing an agent. Every parameter has a sensible
default — only `name` is required.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Unique id; matches `[A-Za-z0-9_-]+` |
| `instructions` | `str` | `""` | System prompt; supports `{slot}` template variables |
| `description` | `str` | `""` | Short blurb shown when this agent is referenced as a tool/route |
| `model` | `str` | `settings.default_llm_model` | LLM model — provider routing is by prefix (see [`llm.md`](llm.md)) |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Response cap |
| `tools` | `list[ToolDefinition] \| list[dict]` | `[]` | Pre-shaped tool definitions for the LLM |
| `tool_executor` | `ToolExecutor \| None` | `None` | Override the runner's executor for this agent |
| `mcp_servers` | `list[MCPServer]` | `[]` | MCP servers — tools are auto-discovered |
| `handoffs` | `list[Handoff]` | `[]` | Allowed agent-to-agent transitions |
| `memory_config` | `AgentMemoryConfig` | `AgentMemoryConfig()` | Memory search/store behavior |
| `config` | `AgentConfig` | `AgentConfig()` | Run-level config (max_turns, ReAct, scanners, …) |
| `output_schema` | `type[BaseModel] \| None` | `None` | Pydantic model for validated structured output |
| `enable_json_mode` | `bool` | `False` | Force JSON output |
| `json_schema` | `dict \| type[BaseModel] \| None` | `None` | JSON schema for `enable_json_mode` |
| `json_strict` | `bool` | `True` | Strict JSON validation |
| `input_schema` | `type[BaseModel] \| None` | `None` | Validate user input against this schema |
| `template_vars` | `dict[str, Any]` | `{}` | Static slots resolved into `instructions` (e.g. `{company}`) |
| `examples` | `list[dict[str, str]]` | `[]` | Few-shot pairs `[{"input": ..., "output": ...}]` |
| `instruction_modifiers` | `list[Callable[[str, RunContext], str]]` | `[]` | Dynamic prompt rewrites at runtime |
| `on_start` | `Callable[[BaseAgent, dict], None] \| None` | `None` | Hook before run |
| `on_end` | `Callable[[BaseAgent, dict], None] \| None` | `None` | Hook after run |
| `on_error` | `Callable[[BaseAgent, Exception, dict], None] \| None` | `None` | Hook on error |
| `on_tool_call` | `Callable[[BaseAgent, str, dict], None] \| None` | `None` | Hook before each tool call |
| `on_handoff` | `Callable[[BaseAgent, str, dict], None] \| None` | `None` | Hook when handing off |
| `metadata` | `dict[str, Any]` | `{}` | Free-form metadata (forwarded to traces) |
| `tags` | `list[str]` | `[]` | Tags forwarded to traces |

### Methods

| Method | Returns | Description |
|---|---|---|
| `system_prompt` *(property)* | `str` | Resolved prompt without runtime context |
| `resolve_system_prompt(context=None)` | `str` | Resolves with template vars + few-shot + modifiers |
| `get_tools_for_llm()` | `list[dict]` | All tools (incl. handoffs and `think`) shaped for the LLM |
| `get_handoff(target_name)` | `Handoff \| None` | Lookup handoff by target |
| `can_handoff_to(target_name)` | `bool` | True if a matching handoff is configured |
| `is_handoff_tool_call(tool_name)` | `tuple[bool, str \| None]` | Detect if a tool call is actually a handoff |
| `clone(**overrides)` | `BaseAgent` | Deep copy with overrides |
| `to_dict()` | `dict` | Serializable representation |
| `to_tool_definition(description=None)` | `dict` | Wrap this agent so another agent can call it as a tool |

### `create_agent` factory

Sugar for the common case:

```python
from orchestrator.agent import create_agent, MemoryScope

agent = create_agent(
    name="support",
    instructions="Help the customer.",
    model="gpt-4o-mini",
    tools=[...],
    handoffs=[...],
    memory_scope=MemoryScope.USER,
    store_memories=True,
    search_memories=True,
    output_schema=None,
)
```

### `agent_as_tool`

Wrap one agent as a callable tool for another agent (multi-agent
delegation without explicit handoffs):

```python
from orchestrator.agent import agent_as_tool

researcher_tool = agent_as_tool(researcher, description="Research a topic")
writer = BaseAgent(name="writer", tools=[researcher_tool], ...)
```

---

## 3 · `AgentRunner`

`from orchestrator.agent import AgentRunner`

```python
runner = AgentRunner(
    container=None,                  # DI container; defaults to global
    llm_client=None,                 # Override LLM client
    memory_client=None,              # Override memory
    session_client=None,             # Override session
    tool_executor=None,              # Override tool executor
    tracing_manager=None,            # Override tracing
    state_manager=None,              # Override RunStateManager (durable runs)
    config=None,                     # RunnerConfig() — see below
    agent_registry=None,             # {name: BaseAgent} for handoff resolution
)
```

### `run()` (non-streaming)

```python
response: AgentResponse = await runner.run(
    agent,
    input,                           # str | list[dict] | list[ChatMessage]
    *,
    session_id=None,                 # Redis history key
    user_id=None,                    # Memory scoping
    context=None,                    # Pre-built RunContext
    max_turns=None,                  # Override agent.config.max_turns
    trace_id=None,                   # Pre-existing trace
    metadata=None,                   # Forwarded to traces
    tags=None,                       # Forwarded to traces
)
```

### `run_stream()`

```python
from orchestrator.agent import EventType

async for event in runner.run_stream(agent, "Tell me a story"):
    if event.type == EventType.CONTENT_DELTA:
        print(event.data["content"], end="", flush=True)
    elif event.type == EventType.TOOL_CALL_START:
        ...
```

### Runner methods

- `register_agent(agent)` — required for handoff targets
- `get_agent(name) -> BaseAgent | None`
- `llm_client` / `memory_client` / `session_client` / `state_manager` *(properties)*

### Execution flow (what `run()` does)

1. Validate input against `agent.input_schema` (if any)
2. Build `RunContext` and `RunState`; start trace span
3. Load `tool_context_state` from session (carries MCP `session_id` etc.)
4. **`MessageBuilder.prepare_messages`** assembles:
   - System prompt → ReAct scaffold (if enabled)
   - Tool-context injection
   - Long-term memory facts (Qdrant search) as a system message
   - Session history (Redis)
   - Optional `rag_context`
   - Sanitize user input → run `input_scanners`
   - Append user message
   - Proactive context compression if near token limit
5. Optional silent **reasoning pass** if `agent.config.reasoning_mode=True`
6. **`Executor.execute_loop`** repeats up to `max_turns`:
   - Call LLM with tools
   - If tool calls → `ToolHandler` runs them, results fed back as messages
   - If a handoff tool was returned → `HandoffExecutor` summarizes history and recursively runs the target
7. **`RunFinalizer`** persists session, stores memories, cleans up
8. Return `AgentResponse`

A circuit breaker guards the loop (`circuit_breaker_threshold` /
`circuit_breaker_cooldown` on `RunnerConfig`).

---

## 4 · Configuration dataclasses

All importable from `orchestrator.agent`.

### `AgentMemoryConfig`

| Field | Type | Default | Notes |
|---|---|---|---|
| `search_memories` | `bool` | `True` | Retrieve long-term memories before LLM call |
| `search_scope` | `MemoryScope` | `USER` | Where to search |
| `search_limit` | `int` | `5` | Top-K |
| `search_threshold` | `float` | `0.0` | Min similarity |
| `store_memories` | `bool` | `True` | Run mem0 fact extraction after the turn |
| `store_scope` | `MemoryScope` | `USER` | Where to write |
| `store_assistant_messages` | `bool` | `True` | Store LLM outputs |
| `store_user_messages` | `bool` | `True` | Store user inputs |
| `extraction_prompt` | `str \| None` | `None` | Custom prompt for fact extraction |
| `pre_store_filter` | `Callable \| None` | `None` | PII filter `(text) -> text` |
| `on_stored` | `Callable \| None` | `None` | Side-effect callback after store |
| `broadcast_learnings` | `bool` | `False` | Share high-importance memories |
| `broadcast_to` | `list[str] \| None` | `None` | Target agent names |
| `broadcast_threshold` | `float` | `0.8` | Importance threshold for broadcast |

### `AgentConfig`

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `str` | `settings.default_llm_model` | |
| `temperature` | `float` | `0.7` | |
| `max_tokens` | `int \| None` | `None` | |
| `max_turns` | `int` | `25` | LLM/tool turn cap |
| `timeout` | `int` | `300` | Seconds per turn |
| `retry_count` | `int` | `3` | LLM call retries |
| `memory` | `AgentMemoryConfig` | default | |
| `handoff` | `HandoffConfig` | default | |
| `context_management` | `ContextManagementConfig \| None` | `None` | Per-agent compression override |
| `input_sanitization` | `bool` | `True` | Strip control chars from input |
| `injection_detection` | `bool` | `False` | Log suspected prompt-injection patterns |
| `output_type` | `Literal["text","json","structured"]` | `"text"` | |
| `reasoning_mode` | `bool` | `False` | Silent think-first pass before main loop |
| `react_mode` | `bool` | `False` | Inject the `think` tool with ReAct scaffold |
| `session_history_turns` | `int \| None` | `None` | Cap history pulled from Redis (None = 20 turns) |
| `require_context` | `bool` | `False` | Fail if no `rag_context` provided |
| `retrieval_top_k` | `int \| None` | `None` | RAG hook |
| `rerank_enabled` | `bool \| None` | `None` | RAG hook |
| `rag_context` | `str \| None` | `None` | Inject as a "PROVIDED CONTEXT" system message |
| `input_scanners` | `list[Callable]` | `[]` | `(text) -> (text, is_safe, reason)` |
| `output_scanners` | `list[Callable]` | `[]` | Same shape, applied to model output |
| `trace_all_turns` | `bool` | `True` | |
| `log_to_session` | `bool` | `True` | Persist tool summaries into session metadata |

### `HandoffConfig`

| Field | Type | Default |
|---|---|---|
| `transfer_history` | `bool` | `True` |
| `summarize_history` | `bool` | `True` |
| `summarization_mode` | `HistorySummarizationMode` | `HYBRID` |
| `recent_turns` | `int` | `3` |
| `summary_model` | `str \| None` | `None` |
| `return_to_parent` | `bool` | `True` |
| `max_handoff_depth` | `int` | `10` |

### `RunnerConfig`

| Field | Type | Default |
|---|---|---|
| `default_max_turns` | `int` | `25` |
| `default_timeout` | `int` | `300` |
| `persist_state` | `bool` | `True` |
| `state_ttl` | `int` | `86400` |
| `parallel_tool_calls` | `bool` | `True` |
| `max_parallel_tools` | `int` | `5` |
| `tool_timeout` | `int` | `60` |
| `retry_on_error` | `bool` | `True` |
| `max_retries` | `int` | `3` |
| `circuit_breaker_threshold` | `int` | `5` |
| `circuit_breaker_cooldown` | `int` | `60` |
| `trace_enabled` | `bool` | `True` |

---

## 5 · Types

All importable from `orchestrator.agent`.

### Enums

```python
class EventType(str, Enum):
    RUN_START / RUN_END / RUN_ERROR
    AGENT_START / AGENT_END
    CONTENT_DELTA / CONTENT_COMPLETE
    ROUTING
    TOOL_CALL_START / TOOL_CALL_END / TOOL_CALL_ERROR
    HANDOFF_START / HANDOFF_END / HANDOFF_RETURN
    MEMORY_RETRIEVAL / MEMORY_STORAGE
    WORKFLOW_STEP / LOOP_ITERATION

class ResponseStatus(str, Enum):
    SUCCESS / ERROR / HANDOFF / TOOL_CALL / MAX_TURNS_REACHED / CANCELLED

class RunStatus(str, Enum):
    PENDING / RUNNING / WAITING_FOR_INPUT / WAITING_FOR_TOOL
    HANDOFF_PENDING / PAUSED / COMPLETED / FAILED / CANCELLED

class MemoryScope(str, Enum):           # NB: this is the enum re-exported from `orchestrator.agent`
    SHARED / USER / AGENT / CONVERSATION # `orchestrator.memory.scopes.MemoryScope` is a dataclass — different type

class MergeStrategy(str, Enum):
    CONCATENATE / LLM_SUMMARIZE / STRUCTURED / FIRST_SUCCESS

class FailStrategy(str, Enum):
    FAIL_FAST / CONTINUE_ON_ERROR / REQUIRE_ALL

class TerminationType(str, Enum):
    LLM_DECISION / TOOL_CALL / OUTPUT_MATCH / CUSTOM

class HistorySummarizationMode(str, Enum):
    FULL / SUMMARY / RECENT_N / HYBRID
```

### Key dataclasses

#### `Handoff`
```python
Handoff(
    target_agent: str,
    description: str,
    condition: str | Callable | None = None,
    transfer_history: bool = True,
    summarize_history: bool = True,
    summarization_mode: HistorySummarizationMode = HYBRID,
    recent_turns: int = 3,
    return_to_parent: bool = True,
)
```

#### `Route` (used by `RouterAgent`)
```python
Route(agent_name: str, description: str, condition=None, priority: int = 0)
```

#### `RunContext`
Carries `run_id`, `session_id`, `user_id`, `trace_id`, `parent_span_id`,
`agent_stack`, `handoff_chain`, `retrieved_memories`, `max_turns`,
`metadata`, `tags`, `usage`. The runner constructs one for you, but you
can pass a pre-built `RunContext` to `runner.run(context=...)`.

#### `AgentResponse`
What `runner.run()` returns. Notable fields:

- `content: str` — final assistant message
- `structured_output: BaseModel | None` — populated if `output_schema` is set
- `tool_calls: list[ToolCall] | None`
- `tool_results: list[dict] | None`
- `run_artifacts: dict | None` — UI widgets / structured tool data
- `handoff: HandoffData | None` and `handoff_result: HandoffResult | None`
- `messages: list[ChatMessage]` — full message tape
- `usage: TokenUsage` — `prompt_tokens`/`completion_tokens`/`total_tokens`/`model_usage`
- `latency_ms: int`, `turn_count: int`, `agents_used: list[str]`,
  `handoff_chain: list[str]`
- `trace_id: str | None`, `span_id: str | None`
- `error: str | None`, `error_type: str | None`

#### `AgentEvent` (streaming)
`type: EventType`, `agent_name`, `run_id`, `data: dict`, `timestamp`,
`trace_id`, `span_id`.

#### `RunState` (durable; stored by `RunStateManager`)
`run_id`, `session_id`, `user_id`, `current_agent`, `agent_stack`,
`entry_agent`, `messages`, `pending_tool_calls`, `handoff_chain`,
`current_handoff`, `status`, `turn_count`, `max_turns`, `metadata`,
`trace_id`, `parent_span_id`, `usage`.

#### `TokenUsage`
`prompt_tokens`, `completion_tokens`, `total_tokens`,
`model_usage: dict[str, dict[str, int]]`.

### ID generators

```python
generate_run_id()      # "run_<16-hex>"
generate_handoff_id()  # "handoff_<12-hex>"
```

---

## 6 · Workflow agents

All workflow agents are themselves `BaseAgent` subclasses, so they nest
arbitrarily. Each ships with a `create_*` factory plus a config
dataclass.

### `SequentialAgent`

Pipe agents — each one's output feeds the next.

```python
from orchestrator.agent import create_sequential_agent, FailStrategy

pipeline = create_sequential_agent(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    pass_full_history=False,                  # default: only last output forwards
    fail_strategy=FailStrategy.FAIL_FAST,
)
```

### `ParallelAgent`

Run agents concurrently, merge results.

```python
from orchestrator.agent import create_parallel_agent, MergeStrategy, FailStrategy

fanout = create_parallel_agent(
    name="fanout",
    agents=[a, b, c],
    merge_strategy=MergeStrategy.LLM_SUMMARIZE,   # also CONCATENATE / STRUCTURED / FIRST_SUCCESS
    fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
    timeout=300,
)
```

### `LoopAgent`

Run an inner agent in a loop until a termination condition fires.

```python
from orchestrator.agent import create_loop_agent, TerminationType

iterate = create_loop_agent(
    name="iterate-until-done",
    agent=worker,
    termination_type=TerminationType.LLM_DECISION,
    max_iterations=10,
    termination_prompt="Reply COMPLETE if done, CONTINUE otherwise.",
    # …or termination_tool="finish" / termination_pattern="^DONE" / termination_condition=fn
)
```

### `ReflectionAgent`

Run, critique, retry — until a critic agent passes the output.

```python
from orchestrator.agent import create_reflection_agent

self_improving = create_reflection_agent(
    name="self-improving",
    agent=writer,
    critique_prompt=None,                     # uses default
    max_reflections=2,
    reflection_model=None,                    # defaults to writer's model
)
```

`generate_critique_prompt(user_query, llm_client, model=None)` produces
a query-specific critique prompt programmatically.

### `RouterAgent`

LLM-driven routing to one of several specialists.

```python
from orchestrator.agent import create_router_agent

router = create_router_agent(
    name="triage",
    routes=[
        ("billing-agent", "Billing & payment issues"),
        ("technical-agent", "Technical support"),
        ("sales-agent", "Sales / pricing inquiries"),
    ],
    fallback="general-agent",
    strategy="hybrid",                        # "llm" | "rule_based" | "hybrid"
    model=None,
)
```

`router.add_route(Route(...))` / `router.remove_route("billing-agent")`
manage routes at runtime. You can also pass `custom_router=callable` to
override the LLM with deterministic logic.

### `PlannerAgent`

Decompose a goal into steps and execute them. Two modes:

- **Single-agent**: one `agent` plays every step
- **Agent-pool**: an LLM picks a specialist from `agents=[...]` for each step

```python
from orchestrator.agent import create_planner_agent, FailStrategy

planner = create_planner_agent(
    name="planner",
    agents=[researcher, coder, reviewer],
    instructions="You are a planning agent.",
    max_steps=10,
    enable_replanning=False,
    replan_on_failure=True,
    planning_model=None,
    fail_strategy=FailStrategy.FAIL_FAST,
    strict_agent_pool=False,                  # raise if the LLM picks a non-pool agent
)
```

### `DebateAgent`

Pro / con / judge synthesis pattern.

```python
from orchestrator.agent import create_debate_agent

debate = create_debate_agent(
    name="debate",
    topic_description="…",
    pro_stance="Argue in favor.",
    con_stance="Argue against.",
    judge_instructions=None,                  # default judge prompt
    summarise_arguments=False,
    summarise_model=None,
    truncate_chars=2000,
)
```

Pro and con run in parallel; the judge synthesizes a verdict.

### `ScatterAgent`

LLM splits the input into N focused sub-tasks (one per branch),
runs all branches concurrently, merges results.

```python
from orchestrator.agent import create_scatter_agent, MergeStrategy

scatter = create_scatter_agent(
    name="scatter",
    agents=[a, b, c],
    input_slices=None,                        # let the LLM split, or pass pre-cut slices
    merge_strategy=MergeStrategy.LLM_SUMMARIZE,
    fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
    split_model=None,
    timeout=300,
)
```

### `SupervisedSequentialAgent`

Sequential pipeline with an LLM quality gate per step. If a step scores
below `quality_threshold` it retries up to `max_retries`.

```python
from orchestrator.agent import create_supervised_agent

supervised = create_supervised_agent(
    name="supervised",
    agents=[step1, step2, step3],
    quality_threshold=0.7,
    max_retries=2,
    supervisor_model=None,
    pass_full_history=False,
)
```

---

## 7 · Handoffs

Define them on the source agent:

```python
from orchestrator.agent import Handoff, HistorySummarizationMode

triage = BaseAgent(
    name="triage",
    instructions="Route customer requests.",
    handoffs=[
        Handoff(
            target_agent="billing",
            description="Anything about invoices, refunds, payments.",
            transfer_history=True,
            summarize_history=True,
            summarization_mode=HistorySummarizationMode.HYBRID,
            recent_turns=3,
            return_to_parent=True,
        ),
        Handoff(target_agent="technical", description="Bugs, errors, outages."),
    ],
)
```

Then **register both source and target** with the runner:

```python
runner = AgentRunner(agent_registry={"billing": billing, "technical": technical})
runner.register_agent(triage)
await runner.run(triage, "I want a refund for invoice 1234")
```

The framework injects a hidden `handoff_to_<target>` tool per handoff,
detects when the LLM invokes it, summarizes history per the
`summarization_mode`, and recursively runs the target. Cycles, depth
overflow, missing targets, and disallowed handoffs all raise typed
exceptions (see Section 9).

`HandoffManager` (`from orchestrator.agent import HandoffManager`)
exposes `validate_handoff()`, `detect_cycle()`, and `prepare_handoff()`
if you want to drive handoffs manually.

---

## 8 · State persistence

`from orchestrator.agent import RunStateManager,
get_global_state_manager, initialize_global_state_manager`

If `RunnerConfig.persist_state=True` (default), `RunState` is written to
Redis with TTL `state_ttl`. This lets you pause/resume long runs (e.g.
a workflow waiting on tool output) across processes. The default state
manager uses the same Redis instance configured for sessions.

---

## 9 · Exception hierarchy

All inherit from `OrchestratorError`. Importable from
`orchestrator.agent`.

```
AgentError                                    # base for agent errors
├── AgentNotFoundError(agent_name, ...)
├── AgentConfigurationError(message, config_key=None, ...)
├── AgentExecutionError(message, turn=None, ...)
├── AgentTimeoutError(message, timeout=None, ...)
├── MaxTurnsExceededError(max_turns, current_turn, ...)
└── AgentToolError(message, tool_name=None, tool_args=None, ...)

HandoffError(message, from_agent, to_agent, handoff_id, ...)
├── HandoffNotAllowedError
├── HandoffDepthExceededError(current_depth, max_depth, ...)
├── HandoffTargetNotFoundError(from_agent, to_agent, ...)
└── HandoffCycleDetectedError(from_agent, to_agent, agent_stack, ...)
   # NOTE: HandoffCycleDetectedError is not re-exported from
   # orchestrator.agent — import it from orchestrator.agent.exceptions.

WorkflowError(message, workflow_type, step, ...)
├── SequentialWorkflowError(failed_agent=None, ...)
├── ParallelWorkflowError(failed_agents=None, ...)
├── LoopWorkflowError(iteration=None, ...)
├── LoopMaxIterationsError(max_iterations, ...)
├── RouterError(message, input_text=None, ...)
└── NoRouteFoundError(available_routes=None, ...)

RunStateError
├── RunStateNotFoundError(run_id, ...)
└── RunStatePersistenceError
```

---

## 10 · Hooks & lifecycle

```python
def on_start(agent, ctx):       print(f"start {agent.name}")
def on_tool_call(agent, name, args): print(f"tool {name}")
def on_handoff(agent, target, data): print(f"-> {target}")
def on_end(agent, ctx):         print(f"end {agent.name}")
def on_error(agent, exc, ctx):  print(f"err {exc}")

agent = BaseAgent(
    name="my-agent",
    instructions="…",
    on_start=on_start, on_tool_call=on_tool_call,
    on_handoff=on_handoff, on_end=on_end, on_error=on_error,
)
```

Hooks run synchronously inside the runner. Keep them lightweight; for
heavy work use `@observe` or push to a queue.

---

## 11 · Common patterns

**Structured output**

```python
from pydantic import BaseModel
class Plan(BaseModel):
    intent: str
    steps: list[str]

agent = BaseAgent(name="planner", instructions="…", output_schema=Plan)
resp = await runner.run(agent, "…")
plan = resp.structured_output         # Plan instance
```

**RAG injection**

```python
from orchestrator.agent.config import AgentConfig

agent = BaseAgent(
    name="rag-agent",
    instructions="Answer using PROVIDED CONTEXT only.",
    config=AgentConfig(rag_context=retrieved_docs, require_context=True),
)
```

**Per-tier instruction modifier**

```python
def upgrade_for_enterprise(prompt, ctx):
    if ctx.metadata.get("tier") == "enterprise":
        return prompt + "\nThis is an enterprise account. SLA priority."
    return prompt

agent = BaseAgent(
    name="adaptive",
    instructions="You are helping {user_name}.",
    template_vars={"user_name": "Alice"},
    instruction_modifiers=[upgrade_for_enterprise],
)
await runner.run(agent, "...", metadata={"tier": "enterprise"})
```

**Streaming with tool events**

```python
async for ev in runner.run_stream(agent, "..."):
    if ev.type == EventType.CONTENT_DELTA:
        print(ev.data["content"], end="", flush=True)
    elif ev.type == EventType.TOOL_CALL_START:
        print(f"\n[tool: {ev.data['tool_name']}]")
```

---

## 12 · Gotchas

- **Agent registry**: handoff targets must be registered with the runner
  (`runner.register_agent(target)` or via `agent_registry={...}`),
  otherwise you get `HandoffTargetNotFoundError`.
- **`MemoryScope` namespace**: the enum exported from
  `orchestrator.agent` is the one you pass to `AgentMemoryConfig`. The
  *dataclass* `MemoryScope` lives in `orchestrator.memory.scopes` and
  isn't interchangeable.
- **Hook signatures**: hooks are sync-style callables. Async hooks are
  not awaited.
- **Streaming + tools**: tool calls in streaming mode emit
  `TOOL_CALL_START` / `_END` events but the `CONTENT_DELTA` stream
  pauses while a tool runs.
- **`AgentMemoryConfig()` defaults to `search_memories=True` /
  `store_memories=True`** — disable explicitly if you don't have
  Qdrant/mem0 running.
