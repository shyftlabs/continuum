# Data labels & policy control — the core mechanic

**Clinic guides** — [Setup & index](TESTING_GUIDE.md) · [Labels & policy](labels-and-policy.md) · [F6 memory](F6-memory.md) · [F3 server trust](F3-server-trust.md) · [F7 approval](F7-approval.md) · [Namespacing](namespacing.md)

> Commands in this guide run from `playground/data-label-clinic/`, one level
> up from this file.

One label on a run, gates at every sink. This is the clinic's main subject and
the prerequisite for the other three guides: F6's row provenance is a *producer*
of these labels, and F3's tool pinning is a different mechanism that composes
with them.

## The core mechanic — `ctx.data_labels = {phi} → every gate`

Everything below rests on one idea. Here it is in plain English.

`**ctx.data_labels` — a sticky note on the run.** Every request creates a
`RunContext` (`ctx`) that travels with that one run. Its `data_labels` field is a
**set of tags describing what kind of sensitive data this run has touched.** It
starts empty:

```
ctx.data_labels = {}        # nothing sensitive yet
```

`**= {phi}` — the set now holds one tag.** `{phi}` is a Python **set** containing
the string `"phi"` (Protected Health Information). When a declared *source* runs
— here, `lookup_patient` returns a record — the SDK calls `context.taint("phi")`,
which adds the tag:

```
ctx.data_labels = {"phi"}   # this run has now touched PHI
```

It's a flag stuck on the run — *"⚠️ this run has handled PHI"* — and it stays on
for the rest of the run.

**`→ every gate reads it and acts.`** The **five** gates are checkpoints. Before
each lets an action happen, it **reads `ctx.data_labels`** and asks: *"is `phi`
in here? if so, am I allowed to do this with PHI?"* None of them set the flag —
they only check it.


| Gate             | What it reads     | What it does when it sees `{phi}`                                  |
| ---------------- | ----------------- | ------------------------------------------------------------------ |
| **model**        | `ctx.data_labels` | refuses to send the data to `gpt-4o` (cloud) → use on-prem instead |
| **tool**         | `ctx.data_labels` | refuses to run `send_referral_email` / `web_lookup`                |
| **memory-write** | `ctx.data_labels` | refuses to save the data to long-term memory                       |
| **telemetry**    | `ctx.data_labels` | redacts the span payload before it goes into logs/traces           |
| **short-term**   | `ctx.data_labels` | persists a placeholder (not the answer) to session/Redis           |


The picture:

```
        lookup_patient returns PHI
                  │
                  ▼
   ctx.data_labels = {phi}    ←─ one flag, set once, lives on the run
                  │
      ┌───────────┬───────────┼──────────────┬───────────────┐
      ▼           ▼           ▼              ▼               ▼
   model        tool      memory-write   telemetry      short-term     ← five gates,
   reads {phi}  reads {phi}  reads {phi}   reads {phi}    reads {phi}      each just
      │           │           │              │               │           checks the
   deny gpt-4o  deny email  deny memory    redact         session          same flag
                 /web        write          payload        placeholder
```

`ctx.data_labels` is the **single shared piece of state**. Because the set now
contains `phi`, all five gates independently look at it and apply their
restriction — the flag carries "this is sensitive" from the one place that
*produced* the data to every place that might *leak* it.

**Key invariant:** taint flows one direction — **source → context → gates.** A
gate firing is the *end* of the chain, never the start; a gate never produces
taint, only reads it.

## Gates vs. the output scanner

The clinic also wires an **output scanner** (`mask_ssn`), and it is important not
to confuse it with the gates. It is **not** a data-label gate and it is **not**
part of the feature under test — it is a separate, pre-existing SDK content
filter that the demo includes to show the two **compose**.

|              | Data-label gates (the feature)              | Output scanner (`output_scanners` hook)        |
| ------------ | ------------------------------------------- | ---------------------------------------------- |
| Triggered by | **provenance** — a declared source tainted the run | **pattern** in the text (a regex)       |
| Decides via  | `PolicyStore` (policy-as-code), by label    | a callable you supply `(prompt, content) → (sanitized, flagged, reason)` |
| Acts on      | model / tool / memory / telemetry / session | the visible answer string                      |
| When         | at each egress point during the run         | over the final answer (finalizer + streaming)  |
| Failure mode | moving toward fail-**closed**               | fail-**open** (a scanner that raises is skipped) |
| New?         | **yes** — the capability being demonstrated | no — existing SDK hook (`AgentConfig.output_scanners`) |

So in the clinic: the **gates** keep PHI off the cloud model, out of long-term
memory, out of telemetry, and out of the session verbatim; the **scanner**
independently masks an SSN in the answer the clinician actually sees. They run at
different points and neither depends on the other.

## How it's wired (5 files)

```
data-label-clinic/
  config.py            # the PolicyStore (5 deny rules) + which tools are declared PHI + 2 model tiers + mask_ssn scanner
  server.py            # clinic FastMCP server (:8911): clinic_info, lookup_patient, send_referral_email, web_lookup
  pharmacy_server.py   # pharmacy FastMCP server (:8912, BEARER TOKEN): lookup_patient, check_interactions
                       #   PHARMACY_POISON=1 hides its payload in a parameter description, not a docstring
                       #   PHARMACY_TRANSPORT=sse|stdio serves the same tools over another transport
  review.py            # read both catalogues before trusting them (review_server)
  agent.py             # ClinicAgent: connects both servers; wires policy_store + labels; cloud->on-prem fallback
  web.py               # FastAPI backend + glassbox web UI (:8910)
```

**Two MCP servers, and they collide on `lookup_patient`.** That is deliberate.
With one server, tool namespacing is invisible and every name-matched setting
appears to work by accident; with two, `tool:lookup_patient` stops meaning one
thing and each setting has to say which server it means. The clinic's
`lookup_patient` returns a clinical record, the pharmacy's returns dispensing
history — both PHI, both separately declared. Layer D tests this.

One **producer** (where taint comes from), five **gates** (what taint denies),
and one composing **scanner** (independent of taint — see §1b):


|                          | What                                                              | Wired in                                                  |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------- |
| Producer                 | both `lookup_patient` tools declared PHI -> calling either taints the run | `config.py` `tool_data_labels={"clinic__lookup_patient":{"phi"}, "pharmacy__lookup_patient":{"phi"}}` |
| Gate — **model routing** | PHI run denied cloud `gpt-4o`, re-routed to on-prem `gpt-4o-mini` | policy `phi-no-cloud-model`                               |
| Gate — **tool**          | PHI run denied `send_referral_email` / `web_lookup`                            | policy `phi-no-exfiltration-tools`                        |
| Gate — **memory**        | PHI run denied long-term memory write in ANY scope                | policy `phi-never-persisted` (`memory:*`)                 |
| Gate — **telemetry**     | PHI run's span payload redacted                                   | policy `phi-redact-telemetry`                             |
| Gate — **short-term**    | PHI run's answer persisted to session/Redis as a placeholder      | policy `phi-no-short-term` (`session`)                    |
| Scanner (not a gate)     | SSN-shaped strings masked in the visible answer (pattern, not label) | `config.py` `output_scanners=[mask_ssn]`               |


(Memory read=taint, the second provenance producer, is intentionally not wired
here — see `config.py`: the user's memory holds non-sensitive prefs that must
not taint a benign run.)

## Running it

Needs `OPENAI_API_KEY` in your repo-root `.env` (the two model tiers share one
key).

```bash
cd playground/data-label-clinic
python server.py            # terminal 1 — clinic MCP tools on :8911
python pharmacy_server.py   # terminal 2 — pharmacy MCP tools on :8912
python web.py               # terminal 3 — web UI on http://localhost:8910
```

Both servers must be up. `web.py` connects to each at startup and reports which
one it could not reach.

Optional infra (for the memory gates):
- **Long-term** (Test 4): `docker compose up -d milvus milvus-etcd` + `MEMORY_ENABLED=true VECTOR_STORE_PROVIDER=milvus`.
- **Short-term** (Test 5): `docker compose up -d redis-sdk` — enables the session/Redis panel.

Open **[http://localhost:8910](http://localhost:8910)**. Type in the chat (or use the three suggestion
chips). The right-hand panel updates after every turn.

## Layer A — offline policy check (already passing)

The gate *logic* — does a PHI subject get denied each resource — is verifiable
without any LLM:

```
llm:gpt-4o                clean=allow   phi=DENY (phi-no-cloud-model)
llm:gpt-4o-mini           clean=allow   phi=allow          <- on-prem spared
tool:send_referral_email  clean=allow   phi=DENY
tool:web_lookup           clean=allow   phi=DENY
memory:u1  (any scope)    clean=allow   phi=DENY   (phi-never-persisted, memory:*)
telemetry                 clean=allow   phi=DENY
session                   clean=allow   phi=DENY   (phi-no-short-term)
```

(The output scanner has no policy row — it is pattern-based, not label-gated;
see §1b. Its offline check is just `mask_ssn("…123-45-6789…") == "…[SSN REDACTED]…"`.)

This proves the **policy wiring**. It doesn't prove the *runtime* actually
consults it — that's Layer B.

## Layer B — live, in the UI (proves the runtime enforces, end-to-end)

Each step is a controlled experiment: a benign case and a sensitive case that
differ only by whether a PHI tool was hit.

## Test 1 — model routing (the headline test)

1. Send **"What are your clinic hours?"** -> panel: taint = `clean`, model =
  **gpt-4o (cloud)**.
2. Send **"Summarize patient P-123 history"** -> `lookup_patient` fires -> taint
  chip turns `**phi`** -> the next cloud turn is denied -> gate log shows
   *"cloud gpt-4o DENIED for PHI -> re-routing on-prem"* -> model =
   **gpt-4o-mini**.
  - **What it proves:** taint arrived from provenance (not the words), and
  `ModelAccessDeniedError` actually fired mid-run and forced the compliant
  model. Same agent, opposite routing.

## Test 2 — tool gate (exfiltration blocked)
3. Send **"Look up patient P-123 and email a summary to [dr@external.com](mailto:dr@external.com)"** ->
   after the PHI lookup, the email tool comes back `**POLICY DENIED*`* in the
   gate log, and the assistant tells you it can't. The `send_referral_email`
   body never executes.

- **What it proves:** a tainted run is blocked from exfiltration tools, and
the denial is reported to the model (soft-fail by design), not a silent
crash.

> **A second gate sits on the same call.** The policy check here decides *may
> this run use this tool*, by rule, with no human and no sight of the arguments.
> A human-in-the-loop gate then decides *should this call, with these arguments,
> happen* — see [F7 approval](F7-approval.md). It is configured on the agent
> (`tool_approval`) rather than written as a policy resource, because "allow, but
> ask someone first" is not something an allow/deny rule can express.

## Test 3 — telemetry redaction
4. Click **"inspect (clean)"** then **"inspect (PHI)"**. Clean -> full payload
   with `prompt_tokens: 412` intact; PHI -> `{"_redacted": "restricted by    data-label policy ..."}`.

- **What it proves:** the label-deny redaction replaces sensitive payloads
before egress, *and* token/cost fields survive on the clean path (the
masking-regression guard — this is the bug the demo itself caught).

## Test 4 — memory-write gate (sensitive data is never persisted)

Rule: `deny phi -> memory:`* — a PHI-tainted run may not write long-term memory
in ANY scope. Ordinary (non-sensitive) memory still works.

Setup (needs a vector store; this project uses Milvus):

```
docker compose up -d milvus milvus-etcd redis-sdk     # from repo root
# in repo-root .env:  MEMORY_ENABLED=true   VECTOR_STORE_PROVIDER=milvus
```

`ClinicConfig.enable_memory = True` is already set. The agent stores/recalls in
the **USER** scope (`user_id="u1"`), so memories file under `memory:u1`.

Steps (right-hand panel):
5. Click **"save PHI note (blocked)"** -> `denied: true (phi-never-persisted)`;
   it does **not** appear in the Long-term-memory list.
6. Click **"save normal note (allowed)"** -> stored; it **appears** in the list.
7. Use the **Long-term memory (user u1)** card to **refresh / delete / clear**
   the stored (non-sensitive) memories.

- **What it proves:** sensitive data is blocked from persistence in every
scope, while ordinary memory works and is fully manageable — you can *see*
PHI never landed in the store while the normal note did. (With memory off,
the buttons report `skipped: memory not enabled`.)

## Test 5 — short-term memory gate (session/Redis)

Rule: `deny phi -> session`. A PHI run's assistant answer must not be persisted
*verbatim* to the conversation store; the SDK substitutes a fixed placeholder.
This is the short-term complement to Test 4 (long-term) — together a PHI run
persists nowhere.

Short-term memory is scoped per `(user_id, conversation_id)` — the same pattern
as `gateway-local-shop`: the UI generates a fresh `conversation_id` per chat
window (`crypto.randomUUID()`, regenerated by **"new chat"**), and the backend
resolves a deterministic `session_id` via `session_client.get_or_create_session`.

Setup: needs Redis (`docker compose up -d redis-sdk`). `enable_session=True` is
already set. Without Redis the panel shows "session not enabled" and the chat
still works (no persistence) — and you can still use the offline preview buttons.

Short-term memory works **in the background** — there is no UI panel for it
(matching `gateway-local-shop`, which only surfaces long-term memory management).
The conversation is loaded from Redis into each turn's prompt and saved after,
with the gate placeholdering a tainted answer.

Verify it (with Redis up):
8. Ask **"What are your clinic hours?"** then **"Summarize patient P-123
   history"** in the same conversation.
9. Inspect what landed in Redis — either via the server logs or directly:
   ```
   redis-cli KEYS 'session:*'        # find the session key
   redis-cli LRANGE <key> 0 -1       # benign answer verbatim; PHI answer = placeholder
   ```
   The PHI turn's assistant message is stored as
   `[Response omitted: it contained sensitive information …]`, while the
   chat on the left still showed the full answer.
   - **What it proves, end-to-end:** the answer the user saw is **not** what
     landed in Redis — a tainted turn is persisted as a placeholder. Because the
     response *might* contain PHI and we can't verify which parts, the whole
     value is replaced (same conservative approach as telemetry). The placeholder
     is plain-language since the model re-reads it as its own prior turn. Gate
     lives in `SessionService.save_messages`; the expected long-term-write block
     is logged as a quiet `🛡️ … blocked by policy` INFO (no traceback).

## Test 6 — output scanner (SSN masking, NOT a data-label gate)

This is the §1b mechanism — independent of taint. It fires on a **pattern in the
answer**, not on a label, so it works on *any* run (tainted or clean).

10. Ask **"Summarize patient P-123 history"** (P-123's record contains an SSN).
    The answer shown in the chat has the SSN replaced with `[SSN REDACTED]`, and
    the gate log shows *"🧹 OUTPUT SCANNER — SSN masked in the visible answer"*.
    - **What it proves:** a content filter (`output_scanners=[mask_ssn]`) runs
      over the final answer and composes cleanly with the label gates — the gates
      handle routing/persistence/telemetry by *provenance*, while the scanner
      sanitizes the *visible text* by *pattern*. The two are independent: the
      scanner masks the SSN whether or not the run was tainted, and a tainted run
      is still re-routed/redacted whether or not the scanner matched.
    - **Note:** the scanner is **fail-open** (a scanner that raises is logged and
      skipped) — the opposite of the direction the gates are moving (fail-closed).

## What this does and doesn't cover

- **Covers:** all 5 data-label gates + the tool/memory producers + the composing
output scanner, live through the real runtime (not just unit mocks), with a
clean-vs-sensitive contrast for each.
- **Doesn't cover:** the **fork/time-travel taint preservation** (it needs the
decision-trace store configured — left out to keep the project runnable
without that infra). To cover it, add a `fork_check.py` script (the convention
used by `refund-glassbox`) that builds a trace, forks from the post-
`lookup_patient` step, and asserts the resumed context is still `{phi}`.

---

## Where each test hits the SDK

| Test   | SDK code path exercised                                                                                         |
| ------ | --------------------------------------------------------------------------------------------------------------- |
| 1      | `LLMClient._enforce_model_routing_policy` + ambient publish in `runner.run` + tool provenance in `tool_service` |
| 2      | tool gate in `tools/executor.execute_tool_calls` (folds `data_labels`) + POLICY-DENIED message path             |
| 3      | `observability/data_redaction.redact_for_telemetry` (label-deny + `mask_secrets` guard)                         |
| 4      | `MemoryClient.add` write gate via `resolve_active_policy`                                                       |
| 5      | `SessionService.save_messages` short-term gate (`session` resource, explicit `data_labels`) → placeholder       |
| 6      | `agent/utils/validation_utils.apply_output_scanners` (runner finalizer + streaming) — NOT a data-label gate     |
| (fork) | `DecisionStep.data_labels` + `runner.fork` seeding — not wired in this project                                  |
| BM1    | provenance stamp in `MemoryClient.add` (`PROVENANCE_LABELS_KEY: sorted(eff_labels)`) — a sorted list, not a set, to survive the JSON round trip |
| BM2    | `_row_provenance_labels` + `context.taint(*...)` in `MemoryService` — the taint producer that needs no declaration |
| BM3    | `_render_memory_context` in `message_builder` (splits clean/untrusted) + `fence_untrusted(body, MEMORY_TAG)` under `MEMORY_INSTRUCTION` |
| BM4    | same gate as Test 2 — `ToolExecutor.execute_tool_call` folds `data_labels` into the policy subjects; here the labels came from storage, not a tool |
| BM5    | `MemoryClient._enforce_memory_policy` checking `memory:{operation}:{scope}` then the legacy `memory:{scope}` — one gate, both directions |
| BM6    | the `on_labeled_recall` branch in `MemoryService` (fence / drop / `raise MemoryReviewRequiredError`) |
| BM7    | `MemoryClient.mark_reviewed` (pops `PROVENANCE_LABELS_KEY`, writes `REVIEWED_KEY`) + `POST /memory/approve` in `web.py` |
| BM8    | `MemoryService._warn_if_provenance_undeclared` — once per agent, not per turn |
| BM9    | `FilteredMemory._create_memory` gating on the active filter (`memory/providers/filtered_memory.py`), scoped by `Mem0Provider.add`; the delete-and-report block in `SessionClient._store_in_memory` remains as the fallback |
| BM10   | `_strip_hidden_from_messages` + `strip_hidden_chars` (`_HIDDEN_CHARS_RE`) called in `MemoryClient.add` before the provider write |
| C1     | `MCPServer._check_tool_digests` + `_cache_dirty` reset in `connect()` (drift after approval)                    |
| C2     | `PolicyStore.default_deny` tool gate + `_clean_tool` hidden-char stripping + `format_tool_catalog` review output |
| C3     | `ToolTrustConfig(on_unreviewed=, on_drift=)` via `MCPServer._apply_trust_policy` (drops drifted/unapproved tools before the prompt) |
| C4     | `find_identical_catalog` + `_handle_renamed_server` (a moved server is re-filed, not re-approved) |
| D1/D2  | `build_namespaced_tool_name` + duplicate-key check in `ToolExecutor._build_registry`             |
| D3     | `PolicyStore.check` against namespaced `tool:` resources under `default_deny`                    |
| D4     | `resolve_tool_data_labels` + `ToolService._warn_on_unresolvable_data_labels`                     |
| D4b    | `_combined_unreviewed_error` + the collect-then-raise loop in `ToolExecutor._build_registry`     |
| D4c    | `MCPServerStreamableHttp.review_url` (None when headers are set) + `pinning.review_server`        |
| D4d    | `_tool_digest` over inputSchema + `hidden_char_delta` and `format_tool_catalog` across both fields |
| D4e    | every F3 method on `_MCPServerWithClientSession`, shared by Stdio/Sse/StreamableHttp             |
| D4f    | `MCPServerStdio` + `MCPServerStdio.review_unavailable_reason` (no URL exists to review with)     |
| D5     | per-server keys in `tool-pins.json` (`MCPServer._load_approved` reads `pins[self.name]`)         |
