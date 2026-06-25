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
| **Producer** | `lookup_patient` declared PHI → calling it taints the run | `config.py` `tool_data_labels` |
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
python server.py      # terminal 1 — MCP tools on :8911
python web.py         # terminal 2 — web UI on  :8910
```

Open http://localhost:8910.

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
| `config.py` | PolicyStore (the 4 deny rules) + agent declarations + model tiers |
| `server.py` | FastMCP tool server (`clinic_info`, `lookup_patient`, `send_referral_email`, `web_lookup`) |
| `agent.py` | `ClinicAgent` — wires policy + labels; cloud→on-prem fallback; glassbox output |
| `web.py` | FastAPI backend + glassbox web UI |
