# Data-Label Clinic — testing guide

This was one 1,650-line file. It is now five, because it was three documents
fused together: a runbook, a design rationale, and a findings log, braided
through scenarios covering three unrelated findings. Someone testing F6 read
past 900 lines that did not concern them.

| Guide | Covers | Lines |
|---|---|---|
| **[Labels & policy](labels-and-policy.md)** | the core mechanic — one label, six sinks, all gated. **Start here** | ~345 |
| [F6 — memory poisoning](F6-memory.md) | provenance, fencing, recall modes, review, the write filter, hidden characters | ~375 |
| [F3 — MCP server trust](F3-server-trust.md) | rug pulls, pinning, drift, the pin gate | ~375 |
| [F7 — approval](F7-approval.md) | a human-in-the-loop gate before a declared tool call |  ~210 |
| [Namespacing](namespacing.md) | two servers, one colliding tool name; transports | ~495 |

> Commands in this guide run from `playground/data-label-clinic/`, one level
> up from this file.

This guide explains the project, its use case, how to run it, and exactly how to
use it to test **data-label enforcement end-to-end (memory, model routing,
telemetry)**.

## What it is

A **clinic patient-intake assistant** — a chat agent that answers general clinic
questions and, when asked, looks up patient records. It's deliberately a domain
where some data is **sensitive (PHI — protected health information)** and some
isn't, so you can watch the system behave *differently* depending on whether
sensitive data has entered the run.

It's built as a **glassbox**: next to the chat, a panel shows you the machinery
that's normally invisible — the run's current taint, which model answered, and
every policy gate that fired. That's the difference from `gateway-local-shop`:
that project is for testing *shopping/tool flows*; this one exists purely to make
**data-label enforcement visible and testable**.

**The single idea it demonstrates:** the SDK has **no PII detector**. A run
doesn't become "sensitive" because someone typed "diabetes." It becomes
sensitive because a tool *declared* to return PHI was actually called. That
declared **provenance** taints the run, and the taint then **denies resources**
through policy. The clinic makes that chain concrete.

## Setup

Needs `OPENAI_API_KEY` in your repo-root `.env` (the two model tiers share one
key).

```bash
cd playground/data-label-clinic
python server.py            # terminal 1 — clinic MCP tools on :8911
python pharmacy_server.py   # terminal 2 — pharmacy MCP tools on :8912
python web.py               # terminal 3 — web UI on http://localhost:8910
```

Both servers must be up. `web.py` connects to each at startup and reports which
one it could not reach. Open **[http://localhost:8910](http://localhost:8910)**.

`PHARMACY_TRANSPORT=stdio` drops terminal 2 — the agent launches the pharmacy
itself over pipes. Useful wherever you restart `web.py` repeatedly.

Optional infra, needed by the memory scenarios:

- **Long-term** — `docker compose up -d milvus milvus-etcd`, then run `web.py`
  with `MEMORY_ENABLED=true VECTOR_STORE_PROVIDER=milvus`.
- **Short-term** — `docker compose up -d redis-sdk`, which enables the
  session/Redis panel.

## Switches

All are read once at import, so each change needs a `web.py` restart. Setting
one in the repo-root `.env` also works, but note `config.py` loads it with
`override=True`, so a value in `.env` **beats** a shell prefix — pick one place.

| Variable | Values | What it changes |
|---|---|---|
| `CLINIC_RECALL` | `fence` (default) · `drop` · `block` | what a labelled recalled row does ([F6](F6-memory.md)) |
| `CLINIC_FILTER` | `off` (default) · `pii` · `broken` | the memory-write content filter ([F6](F6-memory.md)) |
| `CLINIC_APPROVAL` | `off` (default) · `auto` · `deny` · `ask` · `queue` | who answers a tool-approval prompt ([F7](F7-approval.md)) |
| `CLINIC_APPROVAL_TIMEOUT` | seconds, default `30` | how long the gate waits for a person ([F7](F7-approval.md)) |
| `CLINIC_POISON` | `1` | serve hostile tool *descriptions* ([F3](F3-server-trust.md)) |
| `WEB_POISON` | `1` | `web_lookup` returns a planted instruction ([F6](F6-memory.md)) |
| `PHARMACY_TRANSPORT` | `streamable-http` (default) · `sse` · `stdio` | how the pharmacy is reached ([namespacing](namespacing.md)) |
| `CLINIC_PIN_GATE` | `1` | drop drifted tools instead of warning ([F3](F3-server-trust.md)) |
| `LOG_FULL_PROMPT` | `true` | lift the 2000-char per-message truncation in the `FINAL PROMPT` log |

## Offline scripts

These need no UI and no model key beyond what each says:

```bash
python review.py                 # print both tool catalogues (F3)
python memory_provenance_test.py # stamp -> re-taint -> fence -> gate (F6)
python hidden_char_test.py       # invisible codepoints do not survive a write (F6)
python -m pytest test_server_trust.py -q    # 63 offline assertions
```

`pytest` from the repo root collects none of these — `testpaths = ["tests"]`.
Run them from inside this directory, and never in the same process as
`playground/gateway-local-shop`: both define a top-level `agent.py` and the
imports collide.

## How the project uses Continuum

The project is a thin **consumer**: it wires MCP tools + a two-tier model + a
`PolicyStore` + label declarations onto a `BaseAgent`, runs it through
`AgentRunner`, and the **data-label feature does the rest automatically** —
provenance taints the run, and the model/tool/telemetry/memory/session gates
fire. The web layer just *reads back* `ctx.data_labels`, the caught deny
exceptions, and `redact_for_telemetry` to make all of it visible. (The `mask_ssn`
output scanner is a separate, pre-existing content filter the demo composes in —
see §1b — not part of the data-label feature.)

### 6a. The data-label feature (what's under test)


| Capability                          | How the project uses it                                                                                                                                                                         | File                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **Policy engine**                   | `build_policy_store()` → `PolicyStore` with 5 `AccessPolicy(effect="deny")` rules: subjects=`["phi"]`, resources `llm:gpt-4o`, `tool:send_referral_email`/`web_lookup`, `memory:*`, `telemetry`, `session` | `config.py`                      |
| **Attach policy to agent**          | `BaseAgent(..., policy_store=...)` — this single wire turns on all five gates (model/tool/memory/telemetry/session read `getattr(agent, "policy_store")` or the ambient run policy)             | `agent.py`                                 |
| **Producer #1 — tool provenance**   | `AgentConfig(tool_data_labels={"lookup_patient": {"phi"}})` — calling the tool taints the run                                                                                                   | `config.py` / `agent.py`                   |
| **Producer #2 — memory read=taint** | available (`AgentMemoryConfig.scope_data_labels`) but intentionally unused here — user memory holds non-sensitive prefs                                                                         | `config.py`                                |
| **Producer #3 — run-level**         | not used here; available via `RunContext(data_labels=…)`                                                                                                                                        | —                                          |
| **Read the taint**                  | after `runner.run(context=ctx)`, read `ctx.data_labels` → taint chips                                                                                                                           | `agent.py` `chat()`                        |
| **Model-routing gate**              | catch `ModelAccessDeniedError` (`continuum.agent.exceptions`) → re-run on the on-prem model                                                                                                     | `agent.py` `chat()`                        |
| **Tool gate**                       | scan `resp.messages` for the `POLICY DENIED` tool message the gate produced                                                                                                                     | `agent.py` `chat()`                        |
| **Telemetry gate**                  | `redact_for_telemetry(..., mask_secrets=False)` → clean-vs-PHI redaction                                                                                                                        | `web.py` `/telemetry/inspect`              |
| **Memory-write gate**               | `memory_client.add(..., policy_store=, subject=, data_labels=)` → catch `MemoryAccessDeniedError`; policy `deny phi → memory:`*                                                                 | `agent.py` `attempt_memory_write()`        |
| **Memory management**               | `get_all` / `delete` / `delete_all` on the USER scope — list/delete/clear stored (non-sensitive) memories                                                                                       | `web.py` `/memory/list`,`/delete`,`/clear` |
| **Short-term (session) gate**       | `SessionService.save_messages` substitutes a placeholder for a tainted run's answer; policy `deny phi → session`. Works in the background (no UI panel — gateway-local-shop style)               | `agent.py` `_ensure_session()` + `RunContext(session_id=…)` |
| **Output scanner (NOT a gate)**     | `output_scanners=[mask_ssn]` — a pattern-based content filter run by the SDK over the final answer; masks SSNs. Composes with, but is independent of, the label gates (§1b)                      | `config.py` `mask_ssn` + `agent.py` (AgentConfig) |


Design point: you **opt in with declarations** (`policy_store` + `tool_data_labels`
+ `scope_data_labels`) and the runtime does the gating — no detector, no
per-call plumbing. The output scanner is a *separate* opt-in hook
(`output_scanners`), not part of that gating path.

### 6b. Other Continuum features it relies on


| Feature                      | Usage                                                                                                                                               | File                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Agent core**               | `BaseAgent` (instructions, model, temperature, tools, memory_config, config)                                                                        | `agent.py`                         |
| **Runner**                   | `AgentRunner(container, tool_executor, RunnerConfig(persist_state=False, …))`; `register_agent`; `run(agent, input, context=, user_id=)`            | `agent.py`                         |
| **MCP tool integration**     | all three session transports — `MCPServerStreamableHttp`/`MCPServerSse({"url": …})`, `MCPServerStdio({"command": …})`; `ToolExecutor({server: None, …}).initialize()` → `get_tool_definitions()` — tools discovered over MCP, two servers at once | `agent.py` `build_mcp_servers()` + `server.py`/`pharmacy_server.py` (FastMCP) |
| **MCP server trust (F3)**    | `ToolTrustConfig(pin_path=…, on_unreviewed=…, on_drift=…)`; `review_server(server)`; pins keyed by server name, not URL or protocol | `agent.py` `build_trust_config()` + `config.py` `tool_pin_path`, `review.py` |
| **Multi-turn tool loop**     | executor runs `lookup_patient`, then the next turn hits the model gate                                                                              | (SDK, implicit)                    |
| **Parallel tool calls**      | default-on; the path that surfaced the same-turn exfil bug we fixed                                                                                 | (SDK)                              |
| **RunContext**               | per-request `run_id`/`user_id`/`conversation_id` + the live `data_labels`                                                                           | `agent.py`                         |
| **DI container + lifecycle** | `get_lifecycle_manager(...)`, `get_container()` wire memory/session                                                                                 | `agent.py`                         |
| **Memory (optional)**        | `container.memory_client` (mem0); gated `add()`; off by default                                                                                     | `agent.py`                         |
| **Observability**            | telemetry redaction rides the SDK `SpanScope` chokepoint; demo calls the same `redact_for_telemetry`                                                | `web.py`                           |
| **Logging**                  | `setup_logging(LogLevel.INFO)`, `get_logger(__name__)`                                                                                              | `web.py` / `agent.py`              |
| **Model flexibility**        | swap `agent.model` between `gpt-4o` and `gpt-4o-mini` per run                                                                                       | `agent.py` `_run_once()`           |
| **Env/config loading**       | `load_dotenv(repo_root/.env, override=True)` + gateway-var guard → direct provider                                                                  | `config.py`                        |
