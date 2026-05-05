# Tool-Attention Implementation — 2026-05-05

## Problem

Every LLM turn, Continuum sent the full schema of every tool to the model. For agents with many tools, this means:

- **High token cost per turn** — each tool schema (name, description, all parameters) is included in every API call regardless of whether the tool is relevant to the current query.
- **Context dilution** — irrelevant tool schemas consume context window space that could be used for conversation history or RAG content.
- **Slower inference** — larger prompts increase time-to-first-token, especially on external APIs.

The tool-attention paper and reference implementation address this by routing only semantically relevant tool schemas to the LLM on each turn, while still making all tools available through a compact catalogue summary.

---

## How It Was Fixed

### Phase 1 — Tool Catalogue (compact summary)

A compact list of all available tools is injected as a system message before the user message. Each entry contains only the tool name and a one-line description — no parameters. This gives the LLM awareness of every tool without the token cost of full schemas.

Because Phase 1 is a stable system message that does not change between turns (tool names and descriptions are fixed), it is a strong candidate for **prompt caching**. On Anthropic, the Phase 1 message can be marked with `cache_control` so subsequent turns reuse the cached KV state rather than reprocessing the tool catalogue tokens. On OpenAI, prompt caching is automatic for prefixes of at least 1024 tokens. In both cases, Phase 1 sitting at the top of the system prompt — before session history and user messages — maximises the cacheable prefix length and reduces per-turn cost.

Example:
```
[Available tools]
- search_products: Search the product catalogue by keyword or category
- add_to_cart: Add a product to the shopping cart
- view_cart: View current cart contents and totals
- checkout: Complete the purchase
- think: Internal reasoning step

If a tool you need is listed above but its parameters are not available,
output: NEED_TOOL:<tool_name>
```

### Phase 2 — Promoted Tools (full schemas)

A semantic search using Milvus embeddings matches the current user query against tool descriptions. The top-k most relevant tools are "promoted" — their full schemas are sent to the LLM as callable tools. The LLM can only make tool calls for promoted tools.

**Files involved:**
- `src/orchestrator/tools/tool_attention/config.py` — `ToolAttentionConfig` dataclass
- `src/orchestrator/tools/tool_attention/registry.py` — `ToolSummaryRegistry`, manages Milvus collection, upserts tool embeddings, runs semantic search
- `src/orchestrator/tools/tool_attention/router.py` — `ToolAttentionRouter`, orchestrates Phase 1 + Phase 2; `apply_tool_attention()` is the public entry point
- `src/orchestrator/agent/execution/message_builder.py` — calls `apply_tool_attention()`, stores result in `context.metadata["_filtered_tools"]` and Phase 1 summary in `context.metadata["tool_summary_message"]`
- `src/orchestrator/agent/execution/executor.py` — reads `_filtered_tools` from metadata, passes them to the LLM; detects `NEED_TOOL:` signal and expands tool set on fallback
- `src/orchestrator/agent/runner.py` — same as executor, for the streaming path
- `src/orchestrator/agent/services/tool_service.py` — hallucination gate: blocks tool calls for tools not in `promoted_tools` set

### NEED_TOOL Fallback

If the LLM needs a tool whose full schema was not promoted (parameters not available), it outputs `NEED_TOOL:<tool_name>`. The executor detects this signal, adds the requested tool to the promoted set, updates `context.metadata["_filtered_tools"]` and `context.metadata["promoted_tools"]`, and retries the LLM call with the expanded tool set.

---

## How to Enable or Disable

### Enable (per agent)

Add `tool_attention` to `AgentConfig`:

```python
from orchestrator.agent.config import AgentConfig
from orchestrator.tools.tool_attention.config import ToolAttentionConfig

agent = BaseAgent(
    ...
    config=AgentConfig(
        tool_attention=ToolAttentionConfig(
            k=3,          # number of tools to promote per turn
            min_tools=3,  # only activates when agent has >= this many tools
        ),
    ),
)
```

Tool-attention only activates when the agent has at least `min_tools` tools. If the agent has fewer tools than `min_tools`, all tools are sent as normal (no filtering). The default value of `min_tools` is 5, meaning filtering only kicks in when the agent has 5 or more tools. Adjust this based on your agent's tool count — for example, the local-shop agent has 5 tools so `min_tools=3` is used to ensure filtering activates.

### Disable

Set `tool_attention=None` (the default) or do not set it:

```python
config=AgentConfig(
    # tool_attention not set — all tool schemas sent every turn
)
```

### Configuration options (`ToolAttentionConfig`)

| Field | Default | Description |
|---|---|---|
| `k` | 3 | Number of tools promoted per turn |
| `min_tools` | 5 | Minimum tool count to activate filtering |
| `threshold` | None | Optional similarity score threshold |
| `always_promote` | `["think"]` | Tools always included regardless of routing |
| `collection_name` | `"tool_summaries"` | Milvus collection name |
| `embedding_model` | `"text-embedding-3-small"` | Model used for tool description embeddings |
| `embedding_dim` | 1536 | Embedding dimension |

---

## Potential Issue When Enabled

### Routing Miss — Needed Tool Outside Retrieved Set

Each turn, the router retrieves the top-k tools most semantically similar to the user query and promotes only those to the LLM with full schemas. The LLM can only call promoted tools.

The risk: the tool the user actually needs may not be in the top-k results. Semantic similarity is based on the literal query text, and indirect or follow-up queries can point to a different tool than the one that scores highest.

**Example (see image):** After items are added to the cart successfully, the user asks "what's in my cart?". The router may score `add_to_cart` higher than `view_cart` for this query and not promote `view_cart`. The LLM then has no way to call `view_cart` and incorrectly reports the cart as empty — even though the items were added.

**How it is addressed — NEED_TOOL fallback:**

The Phase 1 catalogue message instructs the LLM:

```
If you need to call a tool but its parameters are not available,
output: NEED_TOOL:<tool_name>
```

When the LLM recognises it needs a tool whose schema was not promoted, it outputs `NEED_TOOL:view_cart` instead of hallucinating. The executor detects this signal, adds the requested tool to the promoted set, updates `context.metadata["_filtered_tools"]` and `context.metadata["promoted_tools"]`, and retries the LLM call with the expanded tool set. The correct tool call is then made on the retry.

---

## Summary

| Aspect | Detail |
|---|---|
| Token saving | Only `k` full tool schemas sent per turn instead of all N |
| Activation | Automatic when `len(tools) >= min_tools` and `ToolAttentionConfig` is set |
| Fallback (infrastructure) | Degrades to all-tools if Milvus unavailable or routing returns None |
| Key risk | Top-k miss: needed tool not in retrieved set — mitigated by NEED_TOOL fallback |
| Requires | Milvus running for semantic search; OpenAI embedding model access |
