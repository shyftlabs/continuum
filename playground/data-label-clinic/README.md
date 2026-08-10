# Data-Label Clinic — end-to-end enforcement demo

A patient-intake assistant that exercises **data-label enforcement** across all
four gates: **model routing, tools, memory, telemetry**. It's a glassbox: the
web UI shows, per turn, the run's taint, which model answered, and every gate
decision.

## The one idea

The SDK ships **no PII detector**. A run becomes sensitive ("tainted") only
through declared **provenance**, and that taint then **denies resources** via
policy. This demo wires all three producers and all four gates:

| | Mechanism | Where |
|---|---|---|
| **Producer** | both `lookup_patient` tools declared PHI → calling either taints the run | `config.py` `tool_data_labels` |
| **Gate** | PHI run denied the **cloud model** → re-routed on-prem | policy `phi-no-cloud-model` |
| **Gate** | PHI run denied **exfiltration tools** (email / web) | policy `phi-no-exfiltration-tools` |
| **Gate** | PHI run's **long-term memory write denied** (any scope) | policy `phi-never-persisted` (`memory:*`) |
| **Gate** | PHI run's **telemetry redacted** | policy `phi-redact-telemetry` |
| **Gate** | PHI run's answer **not persisted verbatim to short-term memory** (session/Redis) → placeholder | policy `phi-no-short-term` (`session`) |

Taint comes from *calling a tool declared sensitive*, not from reading the words
"diabetes" — that's the whole point.

## Run it

```bash
# from this directory; needs OPENAI_API_KEY in your .env (repo root)
python server.py            # terminal 1 — clinic MCP tools on :8911
python pharmacy_server.py   # terminal 2 — pharmacy MCP tools on :8912 (needs a token)
python web.py               # terminal 3 — web UI on  :8910
```

Optional switches, each documented in TESTING_GUIDE.md:

| | |
|---|---|
| `CLINIC_POISON=1` | clinic serves poisoned tool **descriptions** and an extra tool |
| `PHARMACY_POISON=1` | pharmacy hides its payload in a **parameter description** instead |
| `PHARMACY_TRANSPORT=sse` | pharmacy serves SSE rather than streamable HTTP (set it on `web.py` too) |
| `PHARMACY_TRANSPORT=stdio` | agent launches the pharmacy as a subprocess — **no second terminal**, and no token |
| `CLINIC_PIN_GATE=1` | drop unreviewed/drifted tools instead of reporting them |

Two MCP servers, deliberately overlapping on `lookup_patient` — the clinic
returns a clinical record, the pharmacy a dispensing history. That collision is
what makes tool namespacing (`<server>__<tool>`) observable instead of
theoretical; see TESTING_GUIDE.md Layer D.

Open http://localhost:8910.

### If it refuses to start

With `CLINIC_PIN_GATE=1` (or a fresh `rm -rf tool-trust`) the agent refuses any
server whose tool catalogue nobody has read — that is the feature, not a fault.
Read both catalogues, then accept them:

```bash
python review.py                                                   # read
continuum mcp approve clinic   --pins tool-trust/tool-pins.json --all
continuum mcp approve pharmacy --pins tool-trust/tool-pins.json --all
```

`python review.py --write-pins` does both in one step, for approving before the
agent has ever run; give it a path to write somewhere other than the configured
one.

**Give `review.py` the same `PHARMACY_TRANSPORT` as the running pharmacy.** It
builds the servers through the same `build_mcp_servers()` the agent uses, so a
bare `python review.py` against an SSE pharmacy looks for `:8912/mcp` and finds
nothing. You get one catalogue read, one `Could not review 'pharmacy'` line, and
if you approve both anyway, a pin file that vouches for a server nobody looked
at. See TESTING_GUIDE.md D4e.

`review.py` exists because `continuum mcp inspect` cannot review the pharmacy:
that command sends a bare URL and the pharmacy requires a bearer token, so it
gets a 401 however correct the URL is. `review.py` imports the *same*
`build_mcp_servers()` the agent uses, so what it shows you is the server the
agent runs, header and all — reviewing a separately-specified copy is how you
end up approving something nobody looked at.

The refusal itself explains this per server, but it cannot know this file
exists — it can only tell you to write the two lines `review.py` already
contains.

### The scripted demo (in the UI)

1. **"What are your clinic hours?"** → benign. Answered on **cloud (gpt-4o)**, taint = *clean*.
2. **"Summarize patient P-123 history"** → `lookup_patient` fires → taint = **phi** →
   the next cloud turn is **denied** → the app re-runs on **on-prem (gpt-4o-mini)**.
   *Same agent, opposite routing — driven purely by provenance.*
3. **"Look up P-123 and email a summary to dr@external.com"** → PHI taint →
   `send_referral_email` comes back **POLICY DENIED**; the email never sends.
4. **Telemetry redaction** buttons → compare a span payload `clean` vs `PHI`:
   PHI → `{"_redacted": …}`; note `prompt_tokens` survives the clean path
   (the masking regression guard).
5. **Memory-write gate** button → attempts a shared-scope write carrying PHI →
   `MemoryAccessDeniedError` (only when memory is enabled; see below).

## Two model tiers, one key

`cloud = gpt-4o`, `on-prem = gpt-4o-mini` — both OpenAI so the routing gate is
demonstrable with a single key. In production the on-prem tier would be a local
endpoint. The deny rule targets `llm:gpt-4o` **exactly**, so it does not catch
`gpt-4o-mini`.

## Notes / limits

- **Model, tool, telemetry** gates work with just an LLM key.
- **Memory-write** gate needs memory enabled (Redis + mem0); set
  `ClinicConfig.enable_memory = True`. Otherwise the button reports it was
  skipped (the SDK's write gate runs after `_ensure_enabled()`).
- This project lives entirely in `playground/` and only *consumes* the shipped
  SDK API — it changes nothing in `src/`.

## Files

| File | Role |
|---|---|
| `config.py` | PolicyStore (the 5 deny rules) + agent declarations + model tiers |
| `server.py` | clinic FastMCP server (`clinic_info`, `lookup_patient`, `send_referral_email`, `web_lookup`) |
| `pharmacy_server.py` | pharmacy FastMCP server (`lookup_patient`, `check_interactions`) — bearer token, switchable transport |
| `agent.py` | `ClinicAgent` + `build_mcp_servers()` — wires policy + labels; cloud→on-prem fallback; glassbox output |
| `review.py` | read both catalogues before trusting them (`review_server`) |
| `web.py` | FastAPI backend + glassbox web UI |
