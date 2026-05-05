# Continuum — Complete Feature Reference

Full inventory of every feature available in Continuum (updated at 2026-05-05 ), compiled from source. Organised by layer so you can find what you need and know where to look in the code. 

---

## LLM & Model Layer


| Feature                       | Description                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Multi-provider support        | OpenAI, Anthropic, Google Gemini via native SDKs — not proxied                                                              |
| Priority dispatcher           | Queues external API calls by request priority (1–10) under load                                                             |
| Two-level dispatcher          | For self-hosted models (vLLM, SGLang): stage priority × request priority                                                    |
| Smart layer / tier classifier | Routes to cheap vs. expensive models by query complexity; supports fixed rules, JSON classifier, remote Qwen, or local Qwen |
| Streaming                     | `run_stream()` yields real-time events: content delta, tool calls, handoffs                                                 |
| Structured output             | JSON mode, Pydantic output schemas, strict mode                                                                             |


---

## Agent Core


| Feature                       | Description                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------- |
| BaseAgent                     | Model, instructions, tools, handoffs, memory, policy — all configurable per agent |
| Template variables            | `{user_id}`, `{date}`, custom `{slots}` resolved in the system prompt at runtime  |
| Few-shot examples             | `examples` list auto-appended to system prompt as an Examples block               |
| Dynamic instruction modifiers | Callables applied to the prompt at runtime (e.g. add tier-specific text)          |
| Input validation              | Pydantic `input_schema` checked before the LLM is called                          |
| Input sanitization            | Prompt injection detection at the system boundary                                 |
| Input/output scanners         | Pluggable callables for PII redaction, content filtering, custom safety checks    |
| Lifecycle hooks               | `on_start`, `on_end`, `on_error`, `on_tool_call`, `on_handoff`                    |
| Reasoning mode                | Silent think-first LLM pass before the main turn loop                             |
| ReAct mode                    | Injects a `think` tool so the LLM expresses Thought/Action/Observation steps      |
| Reflection / self-critique    | Inner agent + critic in a configurable self-critique loop (`ReflectionConfig`)    |
| Agent cloning                 | `agent.clone(**overrides)` — deep copy with field overrides                       |
| Agent as tool                 | Wrap any agent as a callable tool for another agent (`agent_as_tool`)             |
| Run state persistence         | Full state (messages, handoff chain, turn count) stored in Redis per run          |
| Circuit breaker               | Opens after N failures, cools down before retrying                                |
| Retry logic                   | Configurable retry count on transient failures                                    |


---

## Workflow Agents


| Agent                       | Pattern                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `RouterAgent`               | LLM, rule-based, or hybrid routing to one of N target agents                                                        |
| `SequentialAgent`           | Pipeline: each agent's output is the next agent's input                                                             |
| `ParallelAgent`             | Same input to all agents concurrently; merge strategies: concatenate, LLM summarize, structured dict, first-success |
| `LoopAgent`                 | Iterates until a termination condition: LLM decision, tool call, regex match, or custom callable                    |
| `ReflectionAgent`           | Inner agent + critic loop; retries until critic approves or `max_reflections` reached                               |
| `PlannerAgent`              | LLM generates a step-by-step plan, executes each step; re-plans on failure                                          |
| `DAGAgent`                  | Dependency-aware workflow; independent stages run in parallel automatically                                         |
| `DebateAgent`               | Pro agent + con agent run concurrently; judge synthesizes a balanced answer                                         |
| `ScatterAgent`              | LLM splits the task into N slices, one slice per agent, all run in parallel (scatter/gather)                        |
| `SupervisedSequentialAgent` | Sequential pipeline with an LLM quality gate after each step; retries below score threshold                         |


---

## Multi-Agent Coordination


| Feature                | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| Handoffs               | Agent delegates mid-conversation to another agent                             |
| History transfer modes | Full history, LLM summary, last-N messages, or hybrid (summary + recent N)    |
| Handoff depth limit    | Configurable max nested handoff depth                                         |
| Return-to-parent       | Target agent returns control to the caller after completing its task          |
| Agent registry         | Named agents registered in the runner for lookup and handoff routing          |
| Handoff chain tracking | Full audit trail of which agents were involved in a run, stored in `RunState` |


---

## Tool Execution


| Feature                    | Description                                                                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MCP integration            | stdio, SSE, and streamable HTTP transports                                                                                                                  |
| Tool registry              | Maps tool names to MCP server + tool definition                                                                                                             |
| Namespace support          | `server__tool_name` prefixing to avoid cross-server name collisions                                                                                         |
| Parallel tool calls        | Up to N concurrent tools per turn                                                                                                                           |
| Rate limiting              | Token-bucket rate limiter per `ToolExecutor`                                                                                                                |
| Concurrency control        | Semaphore-based cap on concurrent calls                                                                                                                     |
| Per-tool timeout           | Returns a structured error message to the LLM on expiry                                                                                                     |
| Context variable injection | Auto-injects captured session vars (e.g. `session_id`, `auth_token`) into tool arguments                                                                    |
| Context variable capture   | Extracts and persists key values from tool responses across turns                                                                                           |
| Tool filters               | Static allowlists (`ToolFilterStatic`), callable filters, context-aware filters                                                                             |
| Tool attention             | Semantic per-turn routing: sends only the K most relevant tool schemas to the LLM, reducing token cost when an agent has many tools (`ToolAttentionConfig`) |
| Run artifacts              | Full MCP response data (widgets, structured content) collected per run and returned in `AgentResponse.run_artifacts`                                        |


---

## Security & Access Control


| Feature                 | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| PolicyStore             | Deny-overrides ACL engine; glob pattern matching on subject × resource                   |
| AccessPolicy            | Resource prefixes: `tool:*`, `memory:*`, `data:*`                                        |
| ToolAccessDeniedError   | Policy denial surfaced to the LLM with a configurable message                            |
| Data sensitivity labels | Taint labels (e.g. `pii`, `phi`) propagate through handoffs via `RunContext.data_labels` |
| Input sanitization      | Injection detection at the system boundary                                               |
| Secret utilities        | Utilities for safe handling of secrets and credentials                                   |


---

## Memory


| Feature                   | Description                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| Long-term memory          | Semantic vector search backed by mem0 + Qdrant                                                   |
| Memory scopes             | `USER`, `AGENT`, `SHARED`, `CONVERSATION`                                                        |
| Memory isolation levels   | Configurable per deployment for multi-tenant safety                                              |
| Custom extraction prompt  | Override mem0's fact extraction with a domain-specific prompt                                    |
| Pre-store filter          | Callable to drop or rewrite facts before they are stored                                         |
| On-stored callback        | Fires after facts are committed to the vector store                                              |
| Broadcast learnings       | Agent shares important facts to specific other agents                                            |
| Importance scoring        | LLM assigns a 0–1 importance score at store time; blended with semantic similarity at retrieval  |
| Time-weighted decay       | Recent memories get a relevance boost; old, low-importance memories can be pruned                |
| Entity memory             | Named entity extraction and tagging (people, orgs, products) for accurate reference resolution   |
| User knowledge profiles   | Structured per-user profile (preferences, employer, expertise) built and updated across sessions |
| Short-term session memory | Conversation history per session stored in Redis                                                 |
| Session history turns     | Configurable number of turns to load from Redis into each run                                    |
| Tool context state        | Per-MCP-server session variables (e.g. `session_id`) persisted across turns                      |


---

## Context Management


| Feature                           | Description                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| Per-model context window tracking | Knows token limits for all major models                                            |
| Progressive context manager       | Proactive summarization before hitting the context limit                           |
| Compression strategies            | `summarize_old`, `truncate_oldest`, `smart` (try summarize, fall back to truncate) |
| Async summarization               | Non-blocking context compression                                                   |
| RAG context injection             | `rag_context` field injected after conversation history                            |
| require_context                   | Skip the LLM and return a no-knowledge message when no RAG context is found        |
| retrieval_top_k / rerank_enabled  | Hints for downstream RAG retrieval pipelines                                       |


---

## Observability


| Feature                  | Description                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Langfuse tracing         | Traces, spans, generation spans, token usage, latency — all sent to Langfuse            |
| `@trace_tool` decorator  | Automatic span creation around tool calls                                               |
| `@trace_agent` decorator | Automatic span creation around agent runs                                               |
| `@observe` decorator     | General-purpose span wrapper for any function                                           |
| Metrics collector        | Latency, token counts, tool call counts                                                 |
| Error reporter           | Structured error reporting with context (`report_error`, `report_exception`)            |
| Token usage tracking     | Per-model breakdown across multi-agent runs                                             |
| ToolExecutionSummary     | Lightweight per-turn summary (tools used, latencies, errors) stored as session metadata |
| Run artifacts            | Per-run tool call audit trail returned in `AgentResponse`                               |


---

## Evaluation


| Feature                | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| EvaluatorAgent         | LLM-as-judge with configurable criteria (correctness, conciseness, etc.)  |
| EvalCase / EvalResult  | Structured types for test cases and results                               |
| deepeval integration   | `DeepEvalEvaluator` — optional (`pip install deepeval`)                   |
| RAGAS integration      | `RagasEvaluator` for RAG quality metrics — optional (`pip install ragas`) |
| LangfuseDatasetClient  | Manage golden datasets in Langfuse for regression testing                 |
| Golden dataset builder | Tooling to build and export eval datasets from production traces          |


---

## Lifecycle & Infrastructure


| Feature                        | Description                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| OrchestratorLifecycle          | Eager connection verification, graceful shutdown, signal handlers (`SIGTERM`, `SIGINT`) |
| Health checks                  | Per-dependency health status: LLM, memory, session, Langfuse                            |
| Dependency injection container | `Container` wires all clients; every component overridable for testing                  |
| Configuration validation       | Checks required env vars at startup based on which features are enabled                 |
| Circuit breaker                | Opens after N failures, cools down before accepting new calls                           |
| Retry logic                    | Configurable retry count on transient failures                                          |


---

## Durable Execution — Temporal (optional)

Requires `pip install shyftlabs-continuum[temporal]`.


| Feature                                 | Description                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| AgentWorkflow                           | Temporal workflow wrapper for long-running agents that survive process crashes |
| AgentStep / ParallelStep / ApprovalStep | Composable workflow step types                                                 |
| Human-in-the-loop                       | `HumanInLoopManager` for approval gating: approve, reject, escalate            |
| Auto-approve conditions                 | Programmatic bypass for known-safe cases                                       |
| Escalation                              | Configurable escalation handler and timeout when approval is not received      |
| Worker manager                          | Start and stop Temporal workers                                                |
| Agent registry                          | Named agent lookup for workflow activities                                     |


---

## Source Map


| Layer                      | Location                                 |
| -------------------------- | ---------------------------------------- |
| Agent core                 | `src/orchestrator/agent/`                |
| Workflow agents            | `src/orchestrator/agent/workflow/`       |
| Smart layer (tier routing) | `src/orchestrator/agent/smart_layer/`    |
| LLM clients & dispatchers  | `src/orchestrator/llm/`                  |
| Tool execution             | `src/orchestrator/tools/`                |
| Tool attention             | `src/orchestrator/tools/tool_attention/` |
| Security & access control  | `src/orchestrator/security/`             |
| Memory                     | `src/orchestrator/memory/`               |
| Session                    | `src/orchestrator/session/`              |
| Observability              | `src/orchestrator/observability/`        |
| Evaluation                 | `src/orchestrator/evaluation/`           |
| Lifecycle & health         | `src/orchestrator/core/`                 |
| Temporal integration       | `src/orchestrator/temporal/`             |


