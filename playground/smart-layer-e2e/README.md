# Routing playground (smart layer E2E)

Browser UI + API to exercise **model_tier** routing: streaming chat, routing transparency (tier, classifier mode, shortcut vs LLM, intended vs actual completion model, timings), per-session aggregate stats, and optional **golden** benchmarks.

## Setup

```bash
cd playground/smart-layer-e2e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../..   # or: export PYTHONPATH=../../src
cp .env.example .env   # set OPENAI_API_KEY (repo root .env also loaded)
```

Optional:

- **`LLM_ROUTE_*`** — loaded automatically into router defaults (same as SDK helper `apply_llm_route_env_overrides`). See `.env.template`: `LLM_ROUTE_TIER_CLASSIFIER`, `LLM_ROUTE_ROUTER_MODEL`, `LLM_ROUTE_ROUTER_API_BASE`, `LLM_ROUTE_ROUTER_API_KEY`, `LLM_ROUTE_FORCE_COMPLETION_MODEL`. Request JSON / UI fields override env when non-empty.
- `SMART_LAYER_LOCAL_CLASSIFIERS=id_a,id_b` — shows IDs in the UI (registry hook for future server wiring).
- `SMART_LAYER_SYSTEM_PROMPT` — system prompt for completions.

You still need **`OPENAI_API_KEY`** if `DEFAULT_LLM_MODEL` is an OpenAI model (default completions path). If `DEFAULT_LLM_MODEL` is `gemini/...`, configure Gemini / Vertex instead — the UI shows a hint from `GET /info` when the default looks non-OpenAI.

## Run

```bash
cd playground/smart-layer-e2e
PYTHONPATH=../../src uvicorn app:app --reload --port 8765
```

Open **http://127.0.0.1:8765/** for the UI.

## API (still available)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | SPA |
| GET | `/info` | Server defaults, classifier mode catalog, resolved tier models |
| GET | `/classifiers/registry` | Local classifier IDs from env |
| GET | `/routing/stats` | Aggregates (header `X-Routing-Session-Id`) |
| POST | `/routing/stats/clear` | Clear server aggregates for session |
| GET | `/golden/metadata` | Golden JSONL row count + tier breakdown |
| POST | `/golden/benchmark/route-only` | Accuracy / escalation / classifier latency / cost proxy |
| POST | `/golden/benchmark/e2e-judge` | Routed vs nano vs frontier + judge + PGR (small cap) |
| POST | `/chat/stream` | SSE chat |
| POST | `/classify` | Classify only |

### curl SSE example

```bash
curl -sN -X POST http://127.0.0.1:8765/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Routing-Session-Id: demo-session' \
  -d '{"message":"debug this traceback","tier_classifier":"light_only"}'
```

Golden data: [`data/golden_routing_eval.jsonl`](data/golden_routing_eval.jsonl).
