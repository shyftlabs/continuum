# Branch Comparison: `main` vs `tomli-dev`

> **Generated:** 2026-04-22  
> **Merge base:** `657607a` (Refactor LLM provider integration to remove LiteLLM dependency)  
> **Commits in `tomli-dev` ahead of `main`:** 3  


---

## Summary


| Metric                | Value  |
| --------------------- | ------ |
| Files changed         | 66     |
| Lines added           | +4,595 |
| Lines removed         | −1,420 |
| Net new lines         | +3,175 |
| New test files        | 10     |
| Modified source files | 51     |
| New integration tests | 4      |
| New unit tests        | 8      |


The `tomli-dev` branch represents a **major refactoring iteration** focused on three themes:

1. **Terminology rename:** `run_id` → `conversation_id` across the entire SDK
2. **Pluggable vector store:** Milvus support alongside existing Qdrant
3. **Workflow correctness:** Fixing session logging duplication, race conditions, and memory isolation bugs in multi-agent pipelines

---

## Commits


| #   | Hash      | Date             | Message                  |
| --- | --------- | ---------------- | ------------------------ |
| 1   | `5593ec4` | 2026-04-21 12:52 | First push in the branch |
| 2   | `233377a` | 2026-04-22 10:34 | Update                   |
| 3   | `8f46623` | 2026-04-22 15:47 | Update                   |


---

## 1. Core Terminology Rename: `run_id` → `conversation_id`

### Problem Solved

The original `run_id` was regenerated on every request, making it unsuitable for scoping memories to a logical conversation (a chat window). Users needed a stable identifier that persists across multiple requests in the same chat session.

### Changes

#### `src/orchestrator/agent/types.py`

- `MemoryScope.RUN` → `MemoryScope.CONVERSATION`
- `RunContext` gains a new `conversation_id: str | None` field
- `RunState.to_dict()` now uses `get_agent_stack_snapshot()` instead of raw `agent_stack` (prevents serialization of non-serializable objects)
- `PrepareRunResult.initial_message_count` → `user_message_index` (more precise semantics)
- `RunContext` gains `is_handoff: bool = False` flag to skip redundant Redis history loads during handoffs

#### `src/orchestrator/agent/utils/context_utils.py`

- `create_run_context()` now accepts `conversation_id` parameter and passes it through to `RunContext`

#### `src/orchestrator/memory/types.py`

- `MemoryEntry.run_id` → `MemoryEntry.conversation_id` (maps to mem0's `run_id` internally)
- `MemoryFilter.run_id` → `MemoryFilter.conversation_id` (translates to `run_id` in mem0 API calls)

#### `src/orchestrator/memory/scopes.py`

- All occurrences of `run_id` parameter/field → `conversation_id`
- `MemoryScope.run()` factory method → `MemoryScope.conversation()`
- `_run_id` metadata key → `_conversation_id`
- Scope registry: `"run"` mode → `"conversation"` mode

#### `src/orchestrator/memory/providers/mem0.py`

- `_build_identifiers()` maps `conversation_id` → mem0's `run_id` (backward-compatible with mem0 API)
- All method signatures: `run_id` → `conversation_id` (add, search, get_all, delete_all, etc.)

#### `src/orchestrator/session/types.py`

- `SessionMetadata` gains `conversation_id: str | None` field
- `SessionMessage.from_dict()` now **raises `ValueError`** on missing `timestamp` instead of silently defaulting to `datetime.now()` (prevents subtle ordering bugs)
- Imports `UTC` for timezone-aware datetimes

#### `src/orchestrator/config.py`

- `memory_isolation` literal choices: `"run"` → `"conversation"`

---

## 2. Pluggable Vector Store: Milvus Support

### Problem Solved

The SDK was hard-coded to Qdrant. Production environments may use Milvus/Zilliz Cloud. This change adds Milvus as a first-class alternative without breaking existing Qdrant deployments.

### Changes

#### `src/orchestrator/config.py`

- New setting: `vector_store_provider: str = "qdrant"` (accepts `"qdrant"` | `"milvus"`)
- New settings block for Milvus: `milvus_host`, `milvus_port`, `milvus_token`, `milvus_collection`

#### `.env.template`

- Added 16 lines of Milvus-related environment variable templates

#### `docker-compose.yml`

- Added ~55 lines for Milvus services (etcd, MinIO, Milvus standalone) with proper health checks and volume mounts

#### `pyproject.toml` / `requirements.txt`

- Added `pymilvus` dependency

#### `src/orchestrator/core/health.py`

- New `_check_milvus()` internal health check (mirrors `_check_qdrant()` pattern)
- New `check_milvus()` public method
- New `check_vector_store()` dispatcher method that routes to the configured provider
- `_register_default_checks()` now conditionally registers either `qdrant` or `milvus` based on `vector_store_provider`
- `check_qdrant()` preserved as a backward-compatible alias

#### `scripts/health_check.py`

- Health check CLI now supports `vector_store`, `qdrant`, and `milvus` service names

#### `src/orchestrator/memory/providers/mem0.py`

- New `_patch_milvus_strong_consistency()` method: Patches `MilvusDB.list()` to use `consistency_level="Strong"` so JSON-field filter queries see growing segments immediately after writes
- New `_flush_milvus()` method: Called only in `delete_all()` (rare, expensive operation) to ensure all growing segments are sealed before deletion
- `kwargs["prompts"]` → `kwargs["prompt"]` (fixes mem0 API parameter name bug, affected both `add()` custom prompts and `update()` custom prompts)

#### `src/orchestrator/memory/config.py`

- Config builder now routes to Milvus provider configuration when `vector_store_provider == "milvus"`

#### `src/orchestrator/memory/base.py`

- Method signatures updated: `run_id` → `conversation_id`

#### `src/orchestrator/memory/client.py`

- All public methods updated: `run_id` → `conversation_id`
- Scope building calls updated accordingly

#### `src/orchestrator/memory/intelligence.py`

- Scope references updated for new naming

---

## 3. Workflow Session Logging Fix (Duplicate Message Bug)

### Problem Solved

In multi-agent workflows (Sequential, Parallel, Supervised, Planner, Loop, Scatter, Reflection), **every sub-agent run was independently writing messages to Redis session history**. This caused:

- Duplicate user queries appearing in conversation history
- Intermediate agent outputs polluting the chat log
- Session history growing unboundedly with internal pipeline chatter

### Solution Pattern

All workflow agents now follow the same pattern:

1. **Disable** `log_to_session` on sub-agents before execution
2. **Restore** original config after execution (in a `finally` block)
3. **Call `runner.save_turn()`** once at the end to record only `(user_query, final_response)`

### Changes

#### `src/orchestrator/agent/runner.py`

- New `save_turn()` method: Saves exactly one conversation turn (user + assistant) to the session — the canonical way for workflows to persist visible messages
- `_prepare_run()` and `run()` now accept `conversation_id` parameter
- `threading.Lock` → `asyncio.Lock` for artifact clearing (fixes potential deadlock in async context)
- `initial_message_count` → `user_message_index` throughout

#### `src/orchestrator/agent/workflow/sequential.py` (~266 lines changed)

- Wraps execution in `try/finally` to temporarily disable `log_to_session` on sub-agents
- Calls `runner.save_turn()` at completion
- Injects `pipeline_context` into sub-agent `RunContext.metadata` so later agents can see prior outputs

#### `src/orchestrator/agent/workflow/parallel.py` (~190 lines changed)

- Same pattern: disable sub-agent logging → run all in parallel → save single turn
- Restored original `log_to_session` in `finally` block

#### `src/orchestrator/agent/workflow/supervised.py` (~329 lines changed)

- Same pattern applied to supervised sequential with quality scoring
- Fixed indentation/scope issues in retry loop error handling

#### `src/orchestrator/agent/workflow/planner.py` (~358 lines changed)

- Same pattern for planner with dynamic replanning
- Sub-agents disabled from logging; planner saves final consolidated turn

#### `src/orchestrator/agent/workflow/loop.py` (~178 lines changed)

- Same pattern for iterative agent loops
- Saves `(original_input, final_iteration_output)` as single turn

#### `src/orchestrator/agent/workflow/scatter.py` (~225 lines changed)

- Same pattern for scatter-gather workflows

#### `src/orchestrator/agent/workflow/reflection.py` (~124 lines changed)

- Same pattern for two-pass reasoning workflows

#### `src/orchestrator/agent/workflow/debate.py` (+25 lines)

- Added debate workflow changes consistent with session logging pattern

#### `src/orchestrator/agent/workflow/router.py` (~14 lines)

- Minor updates for consistency

---

## 4. Session & Message Builder Refactoring

### Problem Solved

- Session history loading was always performed, even during handoff turns where the handoff messages already carry summarized context (causing duplication)
- The `initial_message_count` heuristic for determining which messages are new was fragile

### Changes

#### `src/orchestrator/agent/execution/message_builder.py`

- `prepare_messages()` now returns `tuple[list[dict], int]` — the second element is `user_message_index`, the exact position where user input begins
- **Skips Redis history during handoffs** (`context.is_handoff` check) to prevent duplicate context
- `session_history_limit` → `session_history_turns` with default changed from `50` to `20` (complete turns, not raw messages)
- `session_history_turns=0` explicitly skips the Redis call entirely
- Injects `pipeline_context` from `context.metadata` as a system message for sub-agents
- After context compression, re-scans for user message index to maintain accuracy
- Final prompt log now includes agent name for easier debugging

#### `src/orchestrator/agent/services/session_service.py` (~120 lines removed/refactored)

- Dramatically simplified `save_messages()`: removed ~90 lines of verbose logging and redundant condition checks
- Added `agent_id` parameter to `add_message()` calls
- Default history limit: `50` → `20` with updated docstring (turns, not individual messages)

#### `src/orchestrator/agent/execution/run_finalizer.py`

- Updated to use `user_message_index` instead of `initial_message_count`

#### `src/orchestrator/agent/execution/executor.py`

- Minor refactoring for consistency with new types

#### `src/orchestrator/agent/execution/handoff_executor.py`

- Sets `context.is_handoff = True` so MessageBuilder skips redundant history loading

#### `src/orchestrator/agent/config.py`

- `session_history_limit` → `session_history_turns` rename

#### `src/orchestrator/agent/handoff/history.py`

- Updated for `conversation_id` rename

#### `src/orchestrator/agent/handoff/manager.py`

- Updated for `conversation_id` rename

#### `src/orchestrator/agent/services/memory_service.py` (~97 lines changed)

- Refactored scope building to use `conversation_id` instead of `run_id`
- Simplified code flow

---

## 5. Session Provider (Redis) Improvements

### Problem Solved

- Non-atomic Redis operations could lead to inconsistent state under concurrent access
- Timezone-naive datetimes caused ambiguity
- Orphaned user+agent mapping cleanup was unreliable and added complexity

### Changes

#### `src/orchestrator/session/providers/redis.py` (~378 lines changed)

- **Atomic pipeline operations**: `get_messages`, `clear_session`, and `update_session_metadata` now use `redis.pipeline(transaction=True)` to batch metadata updates + TTL refreshes atomically
- **New method**: `update_session_metadata()` — standalone metadata updater with TTL refresh on both keys
- **Timezone-aware datetimes**: `datetime.now()` → `datetime.now(UTC)` throughout
- **Removed**: user+agent mapping cleanup in `delete_session()` (was unreliable; simplifies deletion logic)

#### `src/orchestrator/session/base.py`

- Added `conversation_id` to session base interface
- Updated method signatures

#### `src/orchestrator/session/client.py`

- `conversation_id` support throughout session lifecycle
- Updated create/get session methods

#### `src/orchestrator/session/config.py`

- Minor config updates

---

## 6. Peripheral Fixes

### Temporal Module — Graceful Import Errors

All Temporal integration files now catch `ImportError` and re-raise with a helpful install message instead of crashing with a generic traceback:

- `src/orchestrator/temporal/activities.py`
- `src/orchestrator/temporal/client.py`
- `src/orchestrator/temporal/worker.py`
- `src/orchestrator/temporal/workflows/agent_workflow.py`
- `src/orchestrator/temporal/workflows/loop_workflow.py`
- `src/orchestrator/temporal/workflows/parallel_workflow.py`
- `src/orchestrator/temporal/workflows/sequential_workflow.py`

### Observability — Langfuse Retry Fix

`**src/orchestrator/observability/providers/langfuse_client.py`**

- **Bug fix**: `ImportError` now sets `_initialized = True` (package missing won't change, no point retrying), but **transient failures** (network, bad credentials) leave `_initialized = False` so they can be retried on next call

### MCP Tool Cleanup — Shutdown Error Handling

`**src/orchestrator/tools/mcp.py`**

- Expanded exception handling during MCP server cleanup to catch `"already running"` RuntimeErrors and `WouldBlock`/`Busy` anyio errors (common during FastAPI shutdown)

### LLM Client

`**src/orchestrator/llm/client.py**`

- Minor 2-line change for consistency

### Agent Service Interface

`**src/orchestrator/agent/interfaces/service_interface.py**`

- Updated interface method signature for `conversation_id`

---

## 7. Test Suite Additions

### New Unit Tests (8 files, ~1,463 lines)


| File                               | Lines | What It Tests                                                                                          |
| ---------------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| `test_agent_runner_save_turn.py`   | 136   | `AgentRunner.save_turn()` correctly writes exactly one turn to Redis                                   |
| `test_handoff_context.py`          | 179   | `is_handoff` flag prevents duplicate history loading                                                   |
| `test_memory_service_scoping.py`   | 181   | Memory service correctly resolves scopes with `conversation_id`                                        |
| `test_message_builder_refactor.py` | 188   | `prepare_messages()` returns correct `user_message_index`, pipeline context injection                  |
| `test_run_context_new_fields.py`   | 84    | `RunContext.conversation_id` and `is_handoff` serialization                                            |
| `test_run_finalizer_refactor.py`   | 154   | Finalizer uses `user_message_index` to identify new messages                                           |
| `test_session_service_refactor.py` | 180   | Simplified session service with `agent_id` parameter                                                   |
| `test_workflow_transparency.py`    | 361   | All workflow types (Sequential, Parallel, Loop, etc.) disable sub-agent logging and call `save_turn()` |


### New Integration Tests (4 files, ~1,096 lines)


| File                       | Lines | What It Tests                                                                                             |
| -------------------------- | ----- | --------------------------------------------------------------------------------------------------------- |
| `test_memory_leak.py`      | 263   | Memory cleanup correctness: `delete_all` leaves no orphans, growth stability under repeated writes        |
| `test_memory_scenarios.py` | 505   | End-to-end memory scenarios: multi-user isolation, conversation-scoped memories, semantic search accuracy |
| `test_memory_stress.py`    | 259   | Stress tests: concurrent writes, bulk operations, large payloads                                          |
| `test_milvus_memory.py`    | 69    | Milvus-specific integration: basic CRUD, health check                                                     |


### Modified Tests


| File                        | Change                                                                  |
| --------------------------- | ----------------------------------------------------------------------- |
| `tests/conftest.py`         | Added `real_milvus` fixture for Milvus integration tests                |
| `test_memory_refactored.py` | +132 lines: Updated for `conversation_id` rename, added new scope tests |
| `test_redis_session.py`     | Removed 2 lines (cleaned up obsolete assertions)                        |


---

## 8. Bugs Fixed


| #   | Bug                                                 | Root Cause                                                                                          | Fix Location                                                                    |
| --- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | **Duplicate messages in chat history**              | Every sub-agent in a workflow independently wrote messages to Redis                                 | All workflow files + `runner.save_turn()`                                       |
| 2   | **Memory scoped to request, not conversation**      | `run_id` was regenerated per-request; no stable conversation identifier                             | `conversation_id` introduced across SDK                                         |
| 3   | **Handoff duplicate context loading**               | Handoff already passes summarized context, but `MessageBuilder` reloaded Redis history on top of it | `is_handoff` flag on `RunContext`                                               |
| 4   | **Milvus filter queries miss recent writes**        | mem0's `MilvusDB.list()` uses default consistency level, which can't see growing segments           | `_patch_milvus_strong_consistency()`                                            |
| 5   | **mem0 custom prompt parameter name wrong**         | Code passed `prompts=` but mem0 API expects `prompt=`                                               | `mem0.py`: `kwargs["prompts"]` → `kwargs["prompt"]`                             |
| 6   | **Non-atomic Redis session operations**             | Separate `SET` + `EXPIRE` calls could leave keys in inconsistent state on failure                   | Redis pipeline with `transaction=True`                                          |
| 7   | **Timezone-naive datetimes in session metadata**    | `datetime.now()` created naive datetimes causing comparison issues                                  | `datetime.now(UTC)` throughout session module                                   |
| 8   | `**SessionMessage` silently defaulting to `now()`** | Missing `timestamp` field went unnoticed, causing incorrect message ordering                        | Now raises `ValueError`                                                         |
| 9   | `**threading.Lock` in async context**               | `_artifact_lock` used `threading.Lock` which can deadlock in async code                             | Changed to `asyncio.Lock`                                                       |
| 10  | **Langfuse retries after `ImportError`**            | Provider marked as initialized even on `ImportError`, preventing retries of transient failures      | Split: `ImportError` → `_initialized=True`; other errors → `_initialized=False` |
| 11  | **MCP cleanup crash on `WouldBlock`/`Busy`**        | Unhandled anyio errors during FastAPI shutdown                                                      | Extended exception handling                                                     |
| 12  | **Temporal crashes without helpful message**        | Bare `from temporalio import ...` crashes with generic `ImportError`                                | Wrapped in try/except with install instructions                                 |


---

## 9. Files Changed Summary (Excluding `playground/`)

Click to expand full file list (66 files)

```
 .env.template                                      |  16 +-
 docker-compose.yml                                 |  55 ++-
 pyproject.toml                                     |   1 +
 requirements.txt                                   |   1 +
 scripts/health_check.py                            |   6 +-
 src/orchestrator/agent/config.py                   |  10 +-
 src/orchestrator/agent/execution/executor.py       |   9 +-
 src/orchestrator/agent/execution/handoff_executor.py |  22 +-
 src/orchestrator/agent/execution/message_builder.py |  48 +-
 src/orchestrator/agent/execution/run_finalizer.py  |   8 +-
 src/orchestrator/agent/handoff/history.py          |   4 +-
 src/orchestrator/agent/handoff/manager.py          |  13 +-
 src/orchestrator/agent/interfaces/service_interface.py |   2 +-
 src/orchestrator/agent/runner.py                   |  73 ++-
 src/orchestrator/agent/services/memory_service.py  |  97 ++--
 src/orchestrator/agent/services/session_service.py | 120 +----
 src/orchestrator/agent/types.py                    |  18 +-
 src/orchestrator/agent/utils/context_utils.py      |   3 +
 src/orchestrator/agent/workflow/debate.py          |  25 +
 src/orchestrator/agent/workflow/loop.py            | 178 ++++----
 src/orchestrator/agent/workflow/parallel.py        | 190 ++++----
 src/orchestrator/agent/workflow/planner.py         | 358 ++++++++-------
 src/orchestrator/agent/workflow/reflection.py      | 124 +++--
 src/orchestrator/agent/workflow/router.py          |  14 +-
 src/orchestrator/agent/workflow/scatter.py         | 225 ++++-----
 src/orchestrator/agent/workflow/sequential.py      | 266 ++++++-----
 src/orchestrator/agent/workflow/supervised.py      | 329 ++++++++------
 src/orchestrator/config.py                         |  13 +-
 src/orchestrator/core/health.py                    |  83 +++-
 src/orchestrator/llm/client.py                     |   2 +-
 src/orchestrator/memory/base.py                    |  26 +-
 src/orchestrator/memory/client.py                  |  48 +-
 src/orchestrator/memory/config.py                  |  93 ++--
 src/orchestrator/memory/intelligence.py            |   8 +-
 src/orchestrator/memory/providers/mem0.py          | 120 +++--
 src/orchestrator/memory/scopes.py                  |  70 +--
 src/orchestrator/memory/types.py                   |  19 +-
 src/orchestrator/observability/providers/langfuse_client.py |   4 +-
 src/orchestrator/session/base.py                   |  21 +-
 src/orchestrator/session/client.py                 |  87 ++--
 src/orchestrator/session/config.py                 |   9 +-
 src/orchestrator/session/providers/redis.py        | 378 ++++++++-------
 src/orchestrator/session/types.py                  |  14 +-
 src/orchestrator/temporal/activities.py            |   8 +-
 src/orchestrator/temporal/client.py                |   8 +-
 src/orchestrator/temporal/worker.py                |   8 +-
 src/orchestrator/temporal/workflows/agent_workflow.py |  10 +-
 src/orchestrator/temporal/workflows/loop_workflow.py |  10 +-
 src/orchestrator/temporal/workflows/parallel_workflow.py |  10 +-
 src/orchestrator/temporal/workflows/sequential_workflow.py |  10 +-
 src/orchestrator/tools/mcp.py                      |  22 +-
 tests/conftest.py                                  |  26 ++
 tests/integration/test_memory_leak.py              | 263 +++++++++++
 tests/integration/test_memory_scenarios.py         | 505 +++++++++++++++++++++
 tests/integration/test_memory_stress.py            | 259 +++++++++++
 tests/integration/test_milvus_memory.py            |  69 +++
 tests/integration/test_redis_session.py            |   2 -
 tests/unit/test_agent_runner_save_turn.py          | 136 ++++++
 tests/unit/test_handoff_context.py                 | 179 ++++++++
 tests/unit/test_memory_refactored.py               | 132 +++++-
 tests/unit/test_memory_service_scoping.py          | 181 ++++++++
 tests/unit/test_message_builder_refactor.py        | 188 ++++++++
 tests/unit/test_run_context_new_fields.py          |  84 ++++
 tests/unit/test_run_finalizer_refactor.py          | 154 +++++++
 tests/unit/test_session_service_refactor.py        | 180 ++++++++
 tests/unit/test_workflow_transparency.py           | 361 +++++++++++++++
```



---

## 10. Breaking Changes

> [!WARNING]
> The following are **breaking changes** that require updates in downstream code:


| Change                                                               | Migration                                     |
| -------------------------------------------------------------------- | --------------------------------------------- |
| `MemoryScope.RUN` → `MemoryScope.CONVERSATION`                       | Update enum references                        |
| `run_id` parameter → `conversation_id` in memory APIs                | Update all `run_id=` kwargs in memory calls   |
| `session_history_limit` → `session_history_turns` in `AgentConfig`   | Update config objects                         |
| `initial_message_count` → `user_message_index` in `PrepareRunResult` | Only affects internal SDK usage               |
| `prepare_messages()` return type: `list` → `tuple[list, int]`        | Only affects internal SDK usage               |
| `SessionMessage` now raises on missing `timestamp`                   | Ensure all stored messages include timestamps |
| `memory_isolation="run"` → `"conversation"` in Settings              | Update `.env` / config files                  |

### Migration Guide: Switching from `main` to `tomli-dev`

> [!IMPORTANT]
> Follow this checklist when upgrading a project from `main` to `tomli-dev`. Items marked **REQUIRED** will cause runtime errors if not done. Items marked **RECOMMENDED** prevent silent bugs.

#### Step 1: Install new dependency — **REQUIRED**

```bash
pip install pymilvus   # only if using Milvus; skip if staying with Qdrant
```

#### Step 2: Update `.env` file — **REQUIRED if using `run` isolation**

```diff
- MEMORY_ISOLATION=run
+ MEMORY_ISOLATION=conversation
```

If you were using `shared`, `user`, or `agent` isolation, no `.env` change is needed.

#### Step 3: Pass `conversation_id` from frontend — **RECOMMENDED**

This is the most important integration change. Every API endpoint that calls `runner.run()` should now pass `conversation_id`:

```diff
  response = await runner.run(
      agent=my_agent,
      input=request.message,
-     session_id=request.session_id,
+     session_id=request.session_id,            # can omit — auto-derived from conversation_id + user_id
+     conversation_id=request.conversation_id,  # NEW: from frontend, unique per chat window
      user_id=request.user_id,
  )
```

If you don't pass `conversation_id`:
- **Short-term memory:** all chat windows for the same user share one Redis session (key = `u:{user_id}`)
- **Long-term memory:** facts from all conversations are searched/stored globally for the user

#### Step 4: Rename `run_id` → `conversation_id` in memory calls — **REQUIRED if calling memory APIs directly**

If your product calls `memory_client.search()`, `memory_client.add()`, or `memory_client.get_all()` directly:

```diff
  results = await memory_client.search(
      query="user preferences",
      user_id=user_id,
-     run_id=session_id,
+     conversation_id=conversation_id,
  )
```

```diff
  await memory_client.add(
      messages=[...],
      user_id=user_id,
-     run_id=session_id,
+     conversation_id=conversation_id,
  )
```

```diff
  await memory_client.delete_all(
-     run_id=session_id,
+     conversation_id=conversation_id,
  )
```

If you only use `runner.run()` and never call memory APIs directly, skip this step.

#### Step 5: Rename `MemoryScope.RUN` → `MemoryScope.CONVERSATION` — **REQUIRED if referenced in code**

```diff
  from orchestrator.agent.types import MemoryScope

  config = AgentMemoryConfig(
-     search_scope=MemoryScope.RUN,
-     store_scope=MemoryScope.RUN,
+     search_scope=MemoryScope.CONVERSATION,
+     store_scope=MemoryScope.CONVERSATION,
  )
```

#### Step 6: Rename `session_history_limit` → `session_history_turns` — **REQUIRED if set explicitly**

```diff
  agent_config = AgentConfig(
-     session_history_limit=10,
+     session_history_turns=10,   # now counts complete turns (request+response pairs), not raw messages
  )
```

Note: the semantics changed slightly — `10` now means 10 complete turns (20 raw messages), not 10 individual messages.

#### Step 7: Update `MemoryEntry.run_id` references — **REQUIRED if accessing memory entry fields**

```diff
  for entry in memories:
-     print(entry.run_id)
+     print(entry.conversation_id)
```

#### Step 8: Handle `SessionMessage` timestamp validation — **RECOMMENDED**

`SessionMessage.from_dict()` now raises `ValueError` on missing `timestamp` instead of silently defaulting to `datetime.now()`. If you construct `SessionMessage` manually or have existing Redis data with missing timestamps, ensure the `timestamp` field is always present.

#### Step 9: No code changes needed (internal SDK improvements)

These changes are internal to the SDK and require **no action** from product code:

- ✅ Workflow session logging fix (no more duplicate messages in Redis)
- ✅ `threading.Lock` → `asyncio.Lock` for artifact clearing
- ✅ Redis pipeline atomicity improvements
- ✅ Timezone-aware datetimes (`datetime.now(UTC)`)
- ✅ Langfuse retry fix for transient failures
- ✅ MCP cleanup error handling improvements
- ✅ Temporal graceful import errors
- ✅ Milvus strong consistency patch
- ✅ `is_handoff` flag to prevent duplicate history loading
- ✅ `pipeline_context` injection for sub-agents in workflows

#### Quick Validation

After migration, run:

```bash
# Unit tests (no infrastructure needed)
pytest tests/unit/ -v

# Integration tests (requires Redis + Qdrant/Milvus running)
pytest tests/integration/ -v

# Health check
python scripts/health_check.py
```

> **Tip:** For this update, you can also use `playground/local-shop` to interact with the frontend for end-to-end testing of the new memory controls and conversation ID flows.

---

## 11. Memory Control Quick Reference

The SDK exposes **two memory layers** that are independently controllable per-agent:


| Layer                 | Backend                             | Purpose                                                                    | Persistence                                        |
| --------------------- | ----------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| **Long-term memory**  | mem0 + vector store (Qdrant/Milvus) | Semantic facts extracted from conversations (e.g. "User works in finance") | Permanent (until explicitly deleted)               |
| **Short-term memory** | Redis session store                 | Recent conversation turns (user + assistant messages)                      | TTL-based (default 7 days), scoped to `session_id` |


### Prompt Assembly Order

`MessageBuilder.prepare_messages()` assembles the final LLM prompt in this order:

```
1. System prompt
2. ReAct scaffold (if enabled)
3. Tool context (if exists)
4. ★ Long-term memories (mem0)          ← AgentMemoryConfig.search_memories
5. Pipeline context (for workflows)
6. ★ Short-term history (Redis)         ← AgentConfig.session_history_turns
7. RAG context (if provided)
8. User input (current query)
```

### Controlling Long-term Memory (mem0)

Configured per-agent via `AgentMemoryConfig`:

```python
from orchestrator.agent.config import AgentMemoryConfig, AgentConfig
from orchestrator.agent.types import MemoryScope

agent_config = AgentConfig(
    memory=AgentMemoryConfig(
        # === READ (loading memories into LLM context) ===
        search_memories=True,          # False = skip long-term memory entirely
        search_limit=5,                # How many memories to inject (default: 5)
        search_scope=MemoryScope.USER, # USER | AGENT | CONVERSATION | SHARED
        search_threshold=0.0,          # Minimum similarity score

        # === WRITE (storing new facts after each turn) ===
        store_memories=True,           # False = don't extract/store facts
        store_scope=MemoryScope.USER,
        extraction_prompt=None,        # Custom fact extraction prompt for mem0
    ),
)
```

**Global kill switch** in `.env`:

```env
MEMORY_ENABLED=false          # Disables the entire mem0 system
MEMORY_ISOLATION=user         # "shared" | "user" | "agent" | "conversation"
MEMORY_SEARCH_LIMIT=5         # Default number of memories to retrieve
```

### Controlling Short-term Memory (Redis Session History)

Configured per-agent via `AgentConfig`:

```python
agent_config = AgentConfig(
    session_history_turns=20,   # Number of turns to load (default: 20)
                                 # Set to 0 to disable short-term memory loading
    log_to_session=True,         # Whether to write this agent's output to Redis
)
```

**Global settings** in `.env`:

```env
SESSION_ENABLED=false           # Disables the entire Redis session system
SESSION_TTL_SECONDS=604800      # 7 days (default)
SESSION_MAX_MESSAGES=1000       # Max messages per session
```

### Quick Examples

**Read-only agent** (loads memories but never writes new facts):

```python
AgentMemoryConfig(search_memories=True, store_memories=False)
```

**Fully stateless agent** (no memory of any kind):

```python
AgentConfig(
    memory=AgentMemoryConfig(search_memories=False, store_memories=False),
    session_history_turns=0,
    log_to_session=False,
)
```

**Minimal context window usage** (useful for cheap/small models):

```python
AgentConfig(
    memory=AgentMemoryConfig(search_memories=True, search_limit=2),
    session_history_turns=3,  # Only last 3 turns
)
```

**Conversation-scoped memory** (memories isolated per chat window):

```python
AgentConfig(
    memory=AgentMemoryConfig(
        search_memories=True,
        search_scope=MemoryScope.CONVERSATION,
        store_scope=MemoryScope.CONVERSATION,
    ),
)
# At runtime, pass conversation_id:
response = await runner.run(agent, user_input, conversation_id="chat-window-abc123")
```

**Agent-scoped memory** (each agent has its own memory silo):

```python
AgentConfig(
    memory=AgentMemoryConfig(
        search_scope=MemoryScope.AGENT,
        store_scope=MemoryScope.AGENT,
    ),
)
```

---

## 12. Frontend Integration: `conversation_id` Requirement

> [!IMPORTANT]
> When a product integrates Continuum, the **frontend must pass `conversation_id`** on every request. This is the single most important integration contract.

### What is `conversation_id`?

`conversation_id` is a **stable identifier for a chat window**. It maps 1:1 to what the user sees as a conversation in the UI:

```
┌─────────────────────────────────────────────┐
│  Chat Window (frontend)                     │
│  conversation_id = "conv-a1b2c3d4"          │
│                                             │
│  User: "What tax deductions can I claim?"   │  ← request 1 (run_id = "run-001")
│  Bot:  "Here are the common deductions..."  │
│                                             │
│  User: "What about home office?"            │  ← request 2 (run_id = "run-002")
│  Bot:  "Since you're a freelancer..."       │    (remembers user context from request 1)
│                                             │
│  User: "Thanks, what about vehicle costs?"  │  ← request 3 (run_id = "run-003")
│  Bot:  "Based on your freelance status..."  │    (still has full conversation context)
└─────────────────────────────────────────────┘
```

- **`run_id`** is regenerated on every request — it's ephemeral
- **`conversation_id`** persists across all requests in the same chat window — it's stable
- **`session_id`** is for Redis short-term history (may or may not equal `conversation_id`)

### Why it matters

Without `conversation_id`, the SDK cannot:
1. **Scope long-term memories to a conversation** — memories from unrelated chat windows would leak across
2. **Load the correct session history** — the agent wouldn't know which past messages belong to this conversation
3. **Support conversation-level isolation** — `MEMORY_ISOLATION=conversation` requires this ID

### How to pass it

**Backend API endpoint** (the product's router/controller):

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = await runner.run(
        agent=my_agent,
        input=request.message,
        session_id=request.session_id,            # Redis session key
        conversation_id=request.conversation_id,  # ← REQUIRED from frontend
        user_id=request.user_id,
    )
    return {"response": response.content}
```

**Frontend** (React/Next.js example):

```typescript
// Generate conversation_id when user opens a new chat window
const conversationId = crypto.randomUUID();

// Send on every message in that chat window
const response = await fetch("/api/chat", {
  method: "POST",
  body: JSON.stringify({
    message: userInput,
    conversation_id: conversationId,  // stable for entire chat window lifetime
    session_id: sessionId,            // can be same as conversation_id, or separate
    user_id: currentUser.id,
  }),
});
```

### Lifecycle

| Event | Action |
|-------|--------|
| User opens new chat window | Frontend generates new `conversation_id` (UUID) |
| User sends message in existing chat | Frontend sends the **same** `conversation_id` |
| User opens a different chat window | Frontend generates a **different** `conversation_id` |
| User closes and reopens same chat | Frontend should **restore** the original `conversation_id` (persist in localStorage or backend) |

### What happens if `conversation_id` is not passed

> [!CAUTION]
> Not passing `conversation_id` causes **memory leakage across chat windows** in both memory layers. This is the most common integration mistake.

#### Long-term memory (mem0): facts leak across conversations

The SDK logs a warning and falls back to **unscoped** behavior:

```
WARNING: memory_isolation='conversation' but context.conversation_id is None —
memory search will be unscoped. Pass conversation_id when calling runner.run().
```

- Facts extracted in Chat Window A are visible in Chat Window B
- The agent may reference information from an unrelated conversation, confusing the user
- `delete_all` for a conversation cannot target just that conversation's facts

#### Short-term memory (Redis session): chat history shared across windows

The Redis session provider uses `conversation_id` to derive the Redis key via `_compute_session_id()` in `session/providers/redis.py`:

```python
def _compute_session_id(self, session_id, user_id, conversation_id) -> str:
    if session_id:                        # explicit → use as-is
        return session_id
    if conversation_id and user_id:       # ← "c:{conversation_id}:u:{user_id}"
        return f"c:{conversation_id}:u:{user_id}"
    if user_id:                           # user only → "u:{user_id}"
        return f"u:{user_id}"
    return generate_session_id()          # fallback → random UUID
```

**With `conversation_id`:** each chat window gets its own Redis key → isolated history ✅

```
Chat Window A (conversation_id="conv-aaa", user_id="user-123")
  → Redis key: "c:conv-aaa:u:user-123"

Chat Window B (conversation_id="conv-bbb", user_id="user-123")
  → Redis key: "c:conv-bbb:u:user-123"
```

**Without `conversation_id`:** falls back to `"u:{user_id}"` → all chat windows share one session ❌

```
Chat Window A (user_id="user-123")  ──┐
                                      ├── Redis key: "u:user-123" → SHARED history
Chat Window B (user_id="user-123")  ──┘

User opens Chat Window A → asks about taxes
User opens Chat Window B → asks about recipes

Result: Chat Window B sees "What tax deductions can I claim?" in its context.
The agent gives a confused response mixing tax and recipe topics.
```

#### Summary of dangers

| Layer | Without `conversation_id` | Risk |
|-------|--------------------------|------|
| **Short-term (Redis)** | All chat windows for the same user share Redis key `u:{user_id}` | Agent sees messages from other chat windows, causing confused responses |
| **Long-term (mem0)** | Facts stored/searched globally for the user | Agent recalls facts from unrelated conversations |
| **Privacy** | Sensitive data from one conversation leaks into another | User sees information they shared in a private context |
| **Multi-agent workflows** | Pipeline context from one conversation bleeds into another | Workflow state becomes unpredictable |


