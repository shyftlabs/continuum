# Incident Desk — Headroom end-to-end test rig

An on-call incident-investigation copilot for a fictional `checkout-api`
outage, built to exercise **every path of Continuum's Headroom integration**
(Phase 1 compression + Phase 2 CCR retrieval) against the real sidecar and a
real LLM. Each tool emits exactly the payload shape that triggers a different
Headroom transform; every claim is asserted headlessly in `e2e_test.py` with
ground truth from `data.py`.

The rig only **consumes** shipped SDK API — nothing in `src/` changes. It
lives in `playground/headroom-compression-incident-desk/`, alongside the other
playground rigs.

## Run it

```bash
# Terminal 0 — the sidecar (offline baseline; from repo root)
cd extensions/headroom && HEADROOM_CCR_BACKEND=memory HEADROOM_OFFLINE=1 \
  HF_HUB_OFFLINE=1 uv run headroom proxy --port 8787

# from this directory; needs OPENAI_API_KEY in the repo-root .env
python e2e_test.py              # headless proof, scenarios 1–8, 10, 11 (spawns server.py itself)
KOMPRESS=1 python e2e_test.py   # + scenario 9 (Kompress characterization)

# or interactively:
python server.py                # terminal 1 — MCP tools on :8921
python web.py                   # terminal 2 — glassbox UI on :8920
```

`config.py` forces `HEADROOM_ENABLED=true` for the process and preflights the
sidecar (a dead sidecar is not fatal — that's scenario 5).

## The scenarios (verified live, sidecar v0.29.0 + gpt-4o-mini; 1–9 on 2026-07-08, 10–11 on 2026-07-10)

| # | Payload → Headroom path | Proves | Observed |
|---|---|---|---|
| 1 | 43 uniform JSON rows → SmartCrusher | Correct count + max-refund extraction **from the compressed view** | ~18% saved list-wide |
| 2 | ~900 log lines → lossy crush + **CCR** | Model calls `continuum_headroom_retrieve` with the marker hash; needle answered; **anti-doom-loop** held (compression runs again after the retrieve) | 98.8% saved; 75k-char original restored |
| 3 | 20 grep-style runbook hits → search/mixed | Right runbook cited from compressed results, payload actually compressed | ~63% saved |
| 4 | `read` tool → **exclusion list** | `router:excluded:tool` — payload untouched ("disk is the source of truth") | byte-identical |
| 5 | dead sidecar → **fail-open** | Run still succeeds, zero compression | answers correct |
| 6 | fabricated 24-hex hash → **anti-forgery** | "not issued" rejection; sidecar never contacted | ✓ |
| 7 | scenario 2 via `chat_stream` | The **runner** interception path (retrieve never surfaces as a tool event) | needle answered live |
| 8 | two crushed payloads in one run | Hash accumulation; both needles retrieved and answered | 2 retrieves, both exact |
| 9 | ~6k words of prose → Kompress (opt-in) | Warm-gate: passthrough cold, `router:text:<ratio>` once the ML model is warm | ~22% trim when warm |
| 10 | runbook text via native `rag_context` (position-7 **system** message) | Same bytes that crush ~98% as a tool result (scenario 3) pass through **uncompressed** — `router:protected:system_message` | before/after byte-identical |
| 11 | `read` service.yaml → `write` service.yaml → **`ReadLifecycleManager`** | The earlier read goes **STALE** (file edited after it) and is replaced by a marker + CCR hash — Headroom's **file-read** compression. Deterministic in `e2e_test.py` (crafted read→write list, no LLM); model-driven via the UI button | `read_lifecycle:stale:service.yaml`; read result 680 → 152 chars (tool role −70%) |

## Payload-shape rules (probed directly against `/v1/compress`)

Discovered building this rig — **what Headroom will and won't compress**:

- **JSON**: detection requires a top-level **array** (`[...]`). SmartCrusher
  handles uniform dict rows; keep arrays ≤ the sidecar's
  `max_items_after_crush` (50) if you need every row to survive losslessly.
- **Text-heavy JSON** (RAG hits as `[{title, url, snippet}, …]`) **passes
  through 0%-compressed** — SmartCrusher declines it. Serve search results
  **grep-style** (`path:line:text`, ≥30% of lines) to hit the search path.
- **Logs**: the detector matches `ERROR|FAIL|WARN|INFO…` *anywhere* in a line
  (10% of lines suffices). Corollary: paragraph-per-line *prose about an
  outage* classifies as LOG — wrap lines and mind the vocabulary if you need
  TEXT/Kompress routing (bit us live: `router:log:0.31` on a postmortem).
- **Prose** (TEXT) routes to Kompress only when the ML model is **warm**;
  cold is a silent passthrough (no router entry at all). Warm shows as
  `router:text:<ratio>` — the string "kompress" never appears.
- **Tools named** `read`/`glob`/`grep`/`write`/`edit` (any case) are excluded
  from **content-shape** compression wholesale (`router:excluded:tool`).
  Independently, a separate pre-pass — `ReadLifecycleManager` — still rewrites
  a `read` output once a later `write`/`edit` on the same path makes it **STALE**
  (scenario 11). So a `read` is *content-shape–excluded* yet *lifecycle-compressible*;
  the two mechanisms are orthogonal. Lifecycle keys on the exact names
  `Read`/`Edit`/`Write` (see Headroom `config.py`), so a differently-named
  filesystem tool (`read_file`, …) gets only content-shape handling, not the
  STALE lifecycle.

## Three integration boundaries worth remembering

1. **A retrieved original must fit the context window.** First cut used
   4,000-line logs (~138k tokens); `continuum_headroom_retrieve` worked, but the SDK's
   own context manager then (correctly) truncated the restored original away
   — the model never saw it. Size CCR-retrievable payloads to fit, or accept
   that oversized originals are unrecoverable in one hop.
2. **Long-term memory can pollute unrelated rigs.** With memory on and a
   shared `user_id`, mem0 injected another demo's "User profile" into this
   agent's prompts. The rig runs with
   `AgentMemoryConfig(search_memories=False, store_memories=False)`.
3. **A STALE marker assumes a write→read-consistent tool world.** The
   `read_lifecycle:stale` marker tells the model to "re-read the file for
   current content." If a `write` tool is a fake stub that reports success but
   doesn't change what a subsequent `read` returns, the model writes → re-reads
   the *old* value → thinks the write failed → writes again → **infinite
   re-read loop** (hit MaxTurns live while building scenario 11). Fix: the
   mock filesystem must actually mutate — `server.py` keeps a mutable
   `_service_yaml` so `write` changes what `read` returns, and `write`'s result
   states the new value so the model can answer without re-reading. Real
   filesystems / MCP file servers give this consistency for free; only a
   memoryless mock breaks it.

## Files

| File | Role |
|---|---|
| `data.py` | Seeded generators + `GROUND_TRUTH` (needles, aggregates, shape-tuning notes) + `SERVICE_YAML` fixture |
| `server.py` | FastMCP tool server on :8921 (6 tools, one per payload shape; `read`+`write` share a mutable `_service_yaml` for scenario 11) |
| `config.py` | .env guard, forces `HEADROOM_ENABLED`, sidecar preflight/stats helpers |
| `agent.py` | `IncidentAgent` — chat + chat_stream returning the per-turn Headroom glassbox |
| `e2e_test.py` | Headless scenarios 1–11 with hard asserts (exit 1 on failure); 9 is opt-in via `KOMPRESS=1` |
| `web.py` | FastAPI glassbox UI on :8920 — scenario buttons (incl. 11 read→edit), sidecar badge, kill/forge controls |
