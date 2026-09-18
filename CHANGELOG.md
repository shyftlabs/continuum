# Changelog

All notable changes to Continuum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Continuum adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **Session ownership** — a session id names storage; it is not proof of who is calling. Previously a caller could hand back any session id with a different `user_id`, or with none at all, and `AgentRunner` would log a warning and carry on: history was read into their prompt, MCP tool state was restored, and their turns were written back into the other user's session. Omitting the id was the quietest path of all, because the check short-circuited before it could fire. `SessionClient` now compares the session's stored owner against a principal the application binds from a credential it verified, and refuses when they disagree. The check sits at the client rather than the runner because `LLMClient.achat(session_id=...)` reads and writes history directly and accepts no `user_id`, so a runner-level gate cannot see it.
- **Session ids are no longer derivable** — ids were built in plaintext from the identifiers they scope (`u:{user_id}`), so anyone who knew a user id could *construct* that user's session id and reach their conversation. No leak was required. `SESSION_HASH_IDS=true` derives them as an HMAC under `SESSION_ID_SECRET` instead, preserving determinism (the same identifiers still resolve to the same session) while making the id impossible to compute without the secret. It also keeps user identifiers out of Redis keys, log lines and traces. Off by default; existing plaintext sessions migrate to their new keys on first access.
- **Weak `SESSION_ID_SECRET` values are refused at startup** — routed through the same fail-closed guard as the Redis password, plus a 32-character floor and a rejection of values beginning with `#` (a mis-parsed `.env` inline comment, which is long enough to clear a length check and identical in every deployment that copied the file). The floor applies here and not to the Redis password because the attack differs in kind: this secret is brute-forced offline, since every user holds a matched pair from their own session. `CONTINUUM_ALLOW_INSECURE=1` downgrades the refusal to a warning for local work.

### Added
- **`continuum.session.bind_principal()`** — binds the caller's verified identity for the current execution context, so `AgentRunner`, `LLMClient` and direct `SessionClient` calls all inherit it without an argument being threaded through. The framework never derives a principal itself: from inside `run()` a validated user id and a typed-in string are both just `str`, and only the application knows which came from a credential it checked.
- **`SessionOwnershipError`**, and `SessionConfig.session_ownership` / `require_principal` / `hash_session_ids` / `session_id_secret`, with the matching `SESSION_*` environment variables.
- **`audit` mode** — reports what enforcement would refuse (log line plus a `session_ownership_*` metric) while still allowing the call, so a deployment can measure impact before enforcing.

### Changed
- **BREAKING: session ownership is enforced by default** (`SESSION_OWNERSHIP=enforce`). An application that does not call `bind_principal()` will have every read and write of an *owned* session refused. Sessions created without a `user_id` have no owner and are unaffected, so anonymous and single-user deployments keep working untouched. To upgrade without downtime, set `SESSION_OWNERSHIP=audit`, adopt `bind_principal()` at your auth boundary, watch the metric reach zero, then remove it. The strict setting is the default deliberately: shipping "report but allow" would have shipped a check that in most deployments never refuses anything — the same shape as the warning it replaces.
- `SESSION_REQUIRE_PRINCIPAL` ships **off**, which is a compatibility choice and not a security one: an application written before `bind_principal()` existed names no principal, and requiring one would refuse it on upgrade. `enforce` does not cover that path — it refuses a caller who gives the *wrong* identity, not one who gives *none* — and with `SESSION_HASH_IDS` also off the session id is derived in plaintext from the user id, so anyone who knows a user id can construct it and present it while naming nobody. **A multi-tenant deployment must set either `SESSION_REQUIRE_PRINCIPAL=true` or `SESSION_HASH_IDS=true`**; either one closes it. The combination is pinned in the test suite so it stays a stated property rather than a discovery.
- `AgentRunner` no longer swallows `SessionOwnershipError`. Every other session-store failure keeps its degrade-and-continue behaviour, so a Redis blip still costs one run its history rather than the whole request.

### Fixed
- `docs/session.md` documented `get_or_create_session(..., agent_id=...)`, a parameter that method has never accepted. The documented signatures are now asserted against the real ones in the test suite.

## [1.2.0] — 2026-07-17

### Added
- **Headroom Compression** (optional) — a per-turn engine that shrinks bulky tool output (logs, tables, search/RAG dumps, long prose) and collapses stale/redundant file reads *before* the payload reaches the model. Runs in two modes: **local** (in-process, default — `pip install "shyftlabs-continuum[headroom-local]"`) or **endpoint** (an out-of-process `headroom proxy` sidecar). Includes **CCR reversible retrieval** — compressed content can be pulled back on demand via the `continuum_headroom_retrieve` tool, guarded against forged tickets. Off by default (`HEADROOM_ENABLED=false`); configured via `HEADROOM_*` env vars. Per-role token effects are logged and exposed for observability. See the Run → Headroom Compression docs and the `playground/headroom-compression-incident-desk` rig.
- **Provider-aware default model** — when `DEFAULT_LLM_MODEL` is unset, the chat/meta-operation default is derived from whichever provider key is present, so an Anthropic- or Gemini-only deployment no longer needs an OpenAI key for chat, routing, reflection, or summarization. New `ANTHROPIC_DEFAULT_MODEL` (default `claude-haiku-4-5`) and `GEMINI_DEFAULT_MODEL` (default `gemini/gemini-2.5-flash`). (The mem0 embedder still defaults to OpenAI — see `EMBEDDER_PROVIDER`.)
- **Configurable agent temperature across all workflow types** — `temperature` is now `float | None` on agents and every workflow (planner, reflection, supervised, …); `None` omits it from the request so provider defaults apply.
- **Loud session preflight** — when a `session_id` is passed to `runner.run()` for a session that was never created, the runner now surfaces it explicitly (`SessionNotCreatedError` / `strict_sessions`) instead of failing silently downstream.
- **`MEMORY_MAX_QUERY_CHARS`** (default `8000`) — bounds a memory search query before it reaches the embedder, avoiding hard failures / silently-empty results at the embedder's token cap. `None`/blank disables.

### Changed
- **Run-state persistence is now off by default** — `RunnerConfig.persist_state` defaults to `False`; enable globally with `PERSIST_RUN_STATE=true` or per-runner.
- **`LLM_REQUEST_TIMEOUT` is now enforced** as a bounded, controllable deadline on LLM calls.
- The Headroom retrieval tool was renamed `continuum_retrieve` → **`continuum_headroom_retrieve`**.
- Behind Headroom, the summarizer's trigger is raised via `HEADROOM_CONTEXT_THRESHOLD` (uses `max()` semantics — never lowers an explicitly higher `CONTEXT_COMPRESSION_THRESHOLD`).
- Smart Gateway: honest error attribution, model-fidelity fixes, and a shadow-model warning that recommends gateway provider ids.
- `httpx` is now declared as a direct dependency.

### Fixed
- **Workflow agents are now dispatched to `execute()`** in `runner.run()` — previously a workflow nested in the conversation loop was silently flattened into a single bare, tool-less LLM call. `run_stream()` also falls back to `run()` for workflow agents instead of flattening.
- **Workflow drivers no longer launder an ERROR into SUCCESS** — a returned error (e.g. circuit-breaker-open prose) is treated as a failed step, not chained forward as an answer or persisted to memory.
- **Per-request trace grouping for workflow agents** — a planner/pool/drafter run now nests its steps under one `agent-run-<workflow>` trace, and a failing step attaches an ERROR event to that trace instead of spawning an orphan `error-UNKNOWN_ERROR` trace.
- Redis TLS pool configuration corrected, and session probes now report the real failure cause (#74).
- `Container.set_memory_client` now propagates to an already-initialized `SessionClient`.
- Context compression keeps tool-call/tool-result pairs intact and counts tool-call payloads; the anti-doom-loop restore is keyed by `tool_call_id` rather than list index.
- The error-reporter's exit flush is bounded, fixing a process hang on shutdown.
- The mem0 gateway LLM is pinned to OpenAI rather than an `auto`/cheap tier.

## [1.1.0] — 2026-07-02

### Added
- **Connector module** (`continuum.connectors`) — a uniform, pluggable layer for external-service connections (Redis, vector store, Temporal, Langfuse). Every connector shares one interface (`is_enabled` / `is_configured` / `mode` / `connect` / `aping` / `describe`) and registers in a shared registry, so services are configured and probed consistently via API keys, local Docker, or custom hosts. Connection **mode** (`local_docker` / `cloud` / `custom` / `disabled`) is inferred from config — TLS/API-key → cloud, localhost → local-docker. Adding a service is one file plus one registration line. `health_check_all()` probes every enabled connector (disabled ones cost zero connection attempts). New config: `TEMPORAL_TLS`, `TEMPORAL_API_KEY`. LLM providers are intentionally excluded (a per-request router, not a persistent connection). Documented in [`docs/connectors.md`](docs/connectors.md).
- **In-memory session provider** (`provider="memory"`) — a non-durable, zero-dependency `MemorySessionProvider` mirroring Redis semantics (deterministic ids, sliding-window, metadata); never raises connection errors. Serves both explicit ephemeral/test flows and the automatic fallback target.
- **`SESSION_FALLBACK_MODE`** switch (`degrade` | `fail`, default `degrade`) — `degrade` falls back to the in-memory store and keeps serving when Redis is unreachable (at first use *or* mid-session); `fail` raises `SessionConnectionError` instead of silently degrading. Documented in [`docs/session.md`](docs/session.md) §10.
- **Persistence-degraded observability** — `SessionClient.persistence_degraded` flag, a `session_persistence` health check (flips `healthy → degraded` on fallback, observing without force-creating the client), and a `session_persistence_degraded` gauge metric (`1.0` degraded / `0.0` healthy). Documented in [`docs/observability.md`](docs/observability.md).
- GitHub issue templates (`bug.yml`, `feature.yml`, `question.yml`) and a chooser config that disables blank issues and routes security reports to a private advisory.
- Pull request template with Conventional-Commits typing, DCO checkbox, and lint/type/test gates.
- `CODEOWNERS` for automatic review routing.
- Dependabot vulnerability **alerts** (surfaced in the Security tab); automated dependency PRs are disabled.
- `MAINTAINERS.md` with the current maintainer list, tone rules, and escalation path.
- `SECURITY.md` with the private disclosure channel and severity SLAs.
- Minimal CI workflow: ruff lint + format check and unit tests on `main`/`dev`.
- `continuum` CLI for one-command infra startup — `continuum up [minimal|standard|full]`, plus `down`, `status`, `logs`, and `config-path`. The Docker Compose stack and Temporal dynamic config are now bundled in the wheel, so there's no compose file to locate or copy after a `pip install`. Each profile writes a managed block to `./.env` so the SDK only targets services that are actually running.
- All published Docker host ports are overridable via `.env` (e.g. `QDRANT_PORT`, `SESSION_REDIS_PORT`, `MILVUS_PORT`, `LANGFUSE_WEB_PORT`, `TEMPORAL_PORT`), with defaults preserving prior behavior — avoids collisions on multi-project machines.
- "Releasing (maintainers)" section in `CONTRIBUTING.md` linking the canonical [`docs/versioning.md`](docs/versioning.md) publish guide.
- Open LLM provider registry — `register_provider(prefix, factory)` and `register_default_provider(factory)` let new backends extend model-name routing without editing core (`get_provider` now resolves via longest-prefix match against the registry).

### Changed
- **Session persistence now initializes lazily** — the `SessionClient` no longer connects to Redis at construction; the provider is resolved on first use. Disabled or unconfigured persistence makes **no** connection attempt and logs **no** warnings, and startup no longer blocks on a slow/absent Redis. When Redis is unreachable the client degrades to the in-memory store with a **single** warning instead of an error per request, and the Redis provider's op-level failures dropped from `error` to `debug` on the degrade path.
- README now renders correctly on the PyPI project page: the logo and all repository links use absolute URLs (PyPI does not resolve repo-relative paths), and the version badge is a dynamic `pypi/v` shield instead of a hardcoded number that drifted out of date.

### Fixed
- **Anthropic temperature compatibility** — Claude 4.6+ adaptive-thinking models (e.g. Claude Opus 4.8) reject an explicit `temperature` parameter with a 400, which previously killed any agent pointed at them. The provider now sends `temperature` normally and, if the API rejects it, strips it and retries once, caching the model so later calls omit it up front (one wasted round-trip per model per process, at most). This is error-driven — the API is the authority — so new/unknown models are handled automatically with no model-name hardcoding.
- **Context-window limit for new Claude models** — a Claude id not in the exact limits table (e.g. `claude-opus-4-8`, whose hyphenated minor didn't match the dot-separated table keys) fell back to a 4096-token window — a ~48× underestimate that caused unnecessarily aggressive context truncation. Added provider-family defaults (any `claude` → 200k, any `gemini` → 1M) consulted before the conservative fallback, so new models within a known family get the right window automatically.
- Docker healthchecks for `qdrant` (now probes `/readyz` over bash `/dev/tcp`, since the image ships no `curl`) and `temporal` (`BIND_ON_IP=0.0.0.0` so the localhost healthcheck can reach the frontend) — both previously reported `unhealthy` while serving correctly.
- `continuum down`/`status`/`logs` now activate all compose profiles, so profiled containers from `minimal`/`standard` are no longer orphaned.
- `structured_output` is now populated across all providers in both `run()` and `run_stream()`; previously it was left empty on several provider paths.
- Lifecycle `on_end` hook now fires for agents reached via handoff — previously only the entry agent's hook ran, and a raising hook no longer masks a successful handoff as a failure.
- Output scanners now run in streaming mode, closing a PII-leak path where the streamed final response bypassed redaction applied in non-streaming runs.
- `return_to_parent=True` handoffs are now bounded by a loop guard (`HandoffLoopError`) instead of recursing unbounded between parent and child.
- `RunContext.data_labels` are now enforced **end-to-end** as additional policy subjects across six sinks — model routing (`llm:<model>`), tool calls (`tool:<name>`), long-term memory (`memory:<scope>`), telemetry, session persistence, and the decision trace — not just tool access. A deny `AccessPolicy` on the label subject takes effect at each, in both streaming and non-streaming runs.
- HITL `ApprovalStep` now authorizes decisions: a decision is honored only when `decided_by` is in `approvers`, gating both approvals and rejections; unauthorized decisions are recorded (`unauthorized_attempts`) and ignored, leaving the step pending. An empty `approvers` list preserves the prior open behavior, and `escalated` decisions are exempt (they re-target rather than resolve).
- `@observe` now records async/sync **generator** spans correctly — the span stays open across iteration, so streaming spans (e.g. `llm_chat_stream`) capture duration and exceptions, including the `WARNING` level a data-label model-routing deny raises mid-stream (previously the span closed before the body ran).
- Removed docs references to a non-existent `PIIPolicy` API; clarified that PII redaction is opt-in via `pre_store_filter` (memory) and output scanners (runs).

---

## [0.2.3] — 2026-06-10

### Fixed
- The published wheel now includes the `continuum` console-script entry point. `0.2.2` on PyPI was built before the CLI landed and shipped without the `continuum` command; `pip install shyftlabs-continuum` now provides `continuum up` / `status` / `down` as documented in the README.


## [0.2.2] — 2026-06

### Fixed
- `continuum.__version__` is now derived from the installed package metadata via
  `importlib.metadata` instead of a hardcoded string, so it always matches the
  distribution version. In `0.2.1` the attribute incorrectly reported `0.2.0`
  because the literal was never bumped. See [docs/versioning.md](docs/versioning.md).

---

## [0.2.1] — 2026-06

### Changed
- Renamed the importable package from `orchestrator` to `continuum`. The
  distribution is unchanged (`pip install shyftlabs-continuum`); imports are now
  `import continuum` / `from continuum.… import …`. Runtime config keys
  (Temporal task queue, memory collection defaults, session key prefix,
  Prometheus metric names) and the `initialize_orchestrator`/`shutdown_orchestrator`
  functions are unchanged.

### Added
- `continuum/py.typed` marker so consumers receive the package's type hints (PEP 561).

### Deprecated
- _Nothing yet._

### Removed
- _Nothing yet._

### Fixed
- `memory_agent` parameter now propagates through all workflow agent types (`SequentialAgent`, `ParallelAgent`, `ScatterAgent`, `PlannerAgent`, `SupervisedAgent`, `LoopAgent`, `ReflectionAgent`, `DebateAgent`) so long-term memory writes via `save_turn` work correctly across all workflows.

### Security
- _Nothing yet._

---

## [0.2.0] — 2026-05

Initial public release. See the repository history for details prior to this changelog being introduced.

[Unreleased]: https://github.com/shyftlabs/continuum/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/shyftlabs/continuum/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/shyftlabs/continuum/releases/tag/v0.2.0
