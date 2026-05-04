"""
Routing playground — API + static UI for smart layer (model_tier).

Run from this directory:
  PYTHONPATH=../../src uvicorn app:app --reload --port 8765
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

TierClassifierOption = Literal["light_only", "heavy_only", "gpt_4o_mini", "qwen", "qwen_local"]

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_root = Path(__file__).resolve().parents[2]
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_root / ".env")

# orchestrator.settings is built once from os.environ; refresh after explicit .env loads
# so OPENAI_API_KEY from repo root is visible even if config was imported elsewhere first.
import orchestrator.config as _orch_cfg

_orch_cfg.get_settings.cache_clear()
_orch_cfg.settings = _orch_cfg.get_settings()

from benchmark import golden_metadata, load_golden_rows, run_e2e_judge, run_route_only
from orchestrator.agent.config import RouterConfig, apply_llm_route_env_overrides
from orchestrator.agent.smart_layer.classifier import classify_product_tier
from orchestrator.agent.smart_layer.errors import TierClassifierError
from orchestrator.agent.smart_layer.defaults import DEFAULT_TIER_MODELS
from orchestrator.agent.smart_layer.resolve import resolve_model_for_tier
from orchestrator.agent.smart_layer.runner_facade import stream_model_tier_turn
from orchestrator.agent.smart_layer.types import ProductTier, parse_product_tier
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.config import settings
from orchestrator.llm import LLMClient
from stats_store import StatsStore

logger = logging.getLogger("routing_playground")

GOLDEN_PATH = _HERE / "data" / "golden_routing_eval.jsonl"
STATIC_DIR = _HERE / "static"

stats_store = StatsStore()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is missing after loading dotenv from %s and %s. "
            "POST /chat/stream and benchmarks return 503 until it is set (see GET /info).",
            _HERE / ".env",
            _root / ".env",
        )
    yield


def _demo_router(rc: RouterConfig) -> RouterAgent:
    return RouterAgent(
        name="routing-playground-router",
        instructions=os.getenv("SMART_LAYER_SYSTEM_PROMPT", ""),
        routes=[],
        router_config=rc,
    )


def _router_config_from_body(
    *,
    tier_classifier: TierClassifierOption | None,
    tier_router_api_base: str | None,
    tier_router_api_key: str | None,
    tier_local_router_api_base: str | None,
    tier_local_router_api_key: str | None,
    tier_classifier_llm_model: str | None,
    tier_force_completion_model: str | None,
    tier_classifier_heuristic_shortcut: bool | None = None,
    routing_strategy: Literal["model_tier"] = "model_tier",
) -> RouterConfig:
    """Merge ``LLM_ROUTE_*`` from Settings first, then non-empty request fields."""
    rc = RouterConfig(routing_strategy=routing_strategy)
    apply_llm_route_env_overrides(rc)
    if tier_classifier:
        rc.tier_classifier = tier_classifier
    if tier_router_api_base and tier_router_api_base.strip():
        rc.tier_router_api_base = tier_router_api_base.strip()
    if tier_router_api_key and tier_router_api_key.strip():
        rc.tier_router_api_key = tier_router_api_key.strip()
    if tier_local_router_api_base and tier_local_router_api_base.strip():
        rc.tier_local_router_api_base = tier_local_router_api_base.strip()
    if tier_local_router_api_key and tier_local_router_api_key.strip():
        rc.tier_local_router_api_key = tier_local_router_api_key.strip()
    if tier_classifier_llm_model and tier_classifier_llm_model.strip():
        rc.tier_classifier_llm_model = tier_classifier_llm_model.strip()
    if tier_force_completion_model and tier_force_completion_model.strip():
        rc.tier_force_completion_model = tier_force_completion_model.strip()
    if tier_classifier_heuristic_shortcut is not None:
        rc.tier_classifier_heuristic_shortcut = tier_classifier_heuristic_shortcut
    return rc


app = FastAPI(
    title="Routing playground",
    version="1.0.0",
    description=(
        "Try tiered LLM routing: see chosen tier, classifier mode, models, timings, "
        "session stats, and optional golden benchmarks."
    ),
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(TierClassifierError)
async def _tier_classifier_error_handler(_request: Request, exc: TierClassifierError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


class ChatStreamBody(BaseModel):
    message: str = Field(..., min_length=1)
    tier_classifier: TierClassifierOption | None = None
    forced_product_tier: Literal["nano", "fast", "balanced", "specialist", "frontier"] | None = None
    tier_force_completion_model: str | None = None
    tier_router_api_base: str | None = None
    tier_router_api_key: str | None = None
    tier_local_router_api_base: str | None = None
    tier_local_router_api_key: str | None = None
    tier_classifier_llm_model: str | None = None
    tier_classifier_heuristic_shortcut: bool | None = None


class ClassifyBody(BaseModel):
    message: str = Field(..., min_length=1)
    tier_classifier: TierClassifierOption | None = None
    forced_product_tier: Literal["nano", "fast", "balanced", "specialist", "frontier"] | None = None
    tier_router_api_base: str | None = None
    tier_router_api_key: str | None = None
    tier_local_router_api_base: str | None = None
    tier_local_router_api_key: str | None = None
    tier_classifier_llm_model: str | None = None
    tier_classifier_heuristic_shortcut: bool | None = None


class BenchmarkRouteBody(BaseModel):
    tier_classifier: TierClassifierOption | None = None
    tier_router_api_base: str | None = None
    tier_router_api_key: str | None = None
    tier_local_router_api_base: str | None = None
    tier_local_router_api_key: str | None = None
    tier_classifier_llm_model: str | None = None
    tier_force_completion_model: str | None = None
    tier_classifier_heuristic_shortcut: bool | None = None
    max_rows: int | None = Field(default=None, ge=1, le=500)


class BenchmarkE2EBody(BenchmarkRouteBody):
    max_rows: int = Field(default=5, ge=1, le=12)


@app.get("/info")
async def info() -> dict[str, Any]:
    rc_plain = RouterConfig()
    rc_env = RouterConfig()
    apply_llm_route_env_overrides(rc_env)

    tiers_out = {}
    for t in ProductTier:
        tiers_out[t.value] = resolve_model_for_tier(t, rc_env, settings.default_llm_model)

    local_raw = os.getenv("SMART_LAYER_LOCAL_CLASSIFIERS", "").strip()
    local_ids = [x.strip() for x in local_raw.split(",") if x.strip()] if local_raw else []

    dm = (settings.default_llm_model or "").lower()
    provider_hint = None
    if dm.startswith("gemini/") or dm.startswith("vertex_ai/"):
        provider_hint = (
            "default_llm_model points at Gemini/Vertex; ensure GEMINI_API_KEY or Vertex "
            "credentials are set, or set DEFAULT_LLM_MODEL to an OpenAI model (e.g. gpt-4o-mini)."
        )

    return {
        "product_name": "Routing playground",
        "tagline": (
            "Single-turn tier routing: see tier choice, classifier path, intended vs actual "
            "completion model, and timings — not a full multi-agent graph."
        ),
        "smart_layer_enabled": settings.smart_layer_enabled,
        "default_llm_model": settings.default_llm_model,
        "openai_api_key_configured": bool(settings.openai_api_key),
        "provider_hint": provider_hint,
        "server_default_classifier": rc_plain.tier_classifier,
        "effective_from_env": {
            "tier_classifier": rc_env.tier_classifier,
            "tier_classifier_llm_model": rc_env.tier_classifier_llm_model,
            "tier_classifier_heuristic_shortcut": rc_env.tier_classifier_heuristic_shortcut,
            "tier_router_api_base": rc_env.tier_router_api_base,
            "tier_router_api_key_configured": bool(
                rc_env.tier_router_api_key or settings.llm_route_router_api_key or settings.hf_api_key
            ),
            "hf_api_key_configured": bool(settings.hf_api_key),
            "tier_local_router_api_base": rc_env.tier_local_router_api_base,
            "tier_local_router_api_key_configured": bool(rc_env.tier_local_router_api_key),
            "tier_force_completion_model": rc_env.tier_force_completion_model,
        },
        "resolved_tier_models": tiers_out,
        "defaults_table": {k.value: v for k, v in DEFAULT_TIER_MODELS.items()},
        "classifier_modes": [
            {
                "id": "light_only",
                "label": "Fixed fast slot",
                "detail": "Always route to the fast tier; classifier skipped.",
            },
            {
                "id": "heavy_only",
                "label": "Fixed balanced slot",
                "detail": "Always route to the balanced tier; classifier skipped.",
            },
            {
                "id": "gpt_4o_mini",
                "label": "gpt-4o-mini (API call)",
                "detail": (
                    "Tier classifier calls OpenAI’s Chat Completions API on your host credentials "
                    "(OPENAI_API_KEY). Uses tier_classifier_llm_model or gpt-4o-mini. "
                    "Does not use the remote Qwen router URL."
                ),
            },
            {
                "id": "qwen",
                "label": "Qwen (API call)",
                "detail": (
                    "HF router classifier: set HF_API_KEY (or LLM_ROUTE_ROUTER_API_KEY). "
                    "Defaults to https://router.huggingface.co/v1 and a Qwen3 router model unless you "
                    "override via tier_router_* or LLM_ROUTE_ROUTER_* . Completion uses OPENAI_API_KEY. "
                    "Keyword shortcuts are off."
                ),
            },
            {
                "id": "qwen_local",
                "label": "Qwen (locally hosted)",
                "detail": (
                    "Requires tier_local_router_api_base or LLM_ROUTE_LOCAL_ROUTER_API_BASE, and a "
                    "model id: tier_classifier_llm_model, LLM_ROUTE_LOCAL_ROUTER_MODEL (MLX repo id), "
                    "or LLM_ROUTE_ROUTER_MODEL. Remote HF URL field is ignored. Optional local API key."
                ),
            },
        ],
        "optional_env_hints": {
            "llm_route_vars": (
                "Remote Qwen classifier: LLM_ROUTE_TIER_CLASSIFIER=qwen plus HF_API_KEY "
                "(optional overrides: LLM_ROUTE_ROUTER_MODEL, LLM_ROUTE_ROUTER_API_BASE, LLM_ROUTE_ROUTER_API_KEY). "
                "Local Qwen: qwen_local plus LLM_ROUTE_LOCAL_ROUTER_API_BASE and "
                "LLM_ROUTE_LOCAL_ROUTER_MODEL (MLX model id, e.g. mlx-community/Qwen2.5-3B-Instruct-4bit), "
                "optional LLM_ROUTE_LOCAL_ROUTER_API_KEY. "
                "gpt-4o-mini classifier uses OPENAI_API_KEY only (ignores router URL). "
                "Also LLM_ROUTE_FORCE_COMPLETION_MODEL. "
                "LLM_ROUTE_TIER_CLASSIFIER_HEURISTIC_SHORTCUT=false disables keyword/length shortcuts "
                "(always run classifier LLM when mode allows)."
            ),
            "mlx_or_local_url": (
                "Remote Qwen: tier_router_api_base field / LLM_ROUTE_ROUTER_API_BASE. "
                "Local qwen_local: tier_local_router_api_base / LLM_ROUTE_LOCAL_ROUTER_API_BASE."
            ),
            "local_classifier_registry": (
                "Comma IDs in SMART_LAYER_LOCAL_CLASSIFIERS for display only; "
                "wire-up is server-specific."
            ),
        },
        "local_classifier_ids": local_ids,
    }


@app.get("/classifiers/registry")
async def classifiers_registry() -> dict[str, Any]:
    raw = os.getenv("SMART_LAYER_LOCAL_CLASSIFIERS", "").strip()
    ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
    return {"items": [{"id": i, "label": i} for i in ids]}


@app.get("/routing/stats")
async def routing_stats(
    x_routing_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    st = await stats_store.get(x_routing_session_id or "default")
    return st.to_public_dict()


@app.post("/routing/stats/clear")
async def routing_stats_clear(
    x_routing_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    await stats_store.clear(x_routing_session_id or "default")
    return {"status": "cleared", "session": x_routing_session_id or "default"}


@app.get("/golden/metadata")
async def golden_meta() -> dict[str, Any]:
    if not GOLDEN_PATH.is_file():
        raise HTTPException(404, "golden_routing_eval.jsonl not found")
    rows = load_golden_rows(GOLDEN_PATH)
    meta = golden_metadata(rows)
    meta["path"] = str(GOLDEN_PATH.name)
    return meta


@app.post("/golden/benchmark/route-only")
async def golden_benchmark_route_only(
    body: BenchmarkRouteBody,
    x_routing_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise HTTPException(
            503,
            "OPENAI_API_KEY not set — required for tier completions and judge (see GET /info provider_hint).",
        )
    if not GOLDEN_PATH.is_file():
        raise HTTPException(404, "golden file missing")

    rows = load_golden_rows(GOLDEN_PATH)
    if body.max_rows is not None:
        rows = rows[: body.max_rows]

    rc = _router_config_from_body(
        tier_classifier=body.tier_classifier,
        tier_router_api_base=body.tier_router_api_base,
        tier_router_api_key=body.tier_router_api_key,
        tier_local_router_api_base=body.tier_local_router_api_base,
        tier_local_router_api_key=body.tier_local_router_api_key,
        tier_classifier_llm_model=body.tier_classifier_llm_model,
        tier_force_completion_model=body.tier_force_completion_model,
        tier_classifier_heuristic_shortcut=body.tier_classifier_heuristic_shortcut,
    )
    llm = LLMClient(enable_langfuse=False)
    result = await run_route_only(
        rows,
        router_config=rc,
        llm=llm,
        instructions=os.getenv("SMART_LAYER_SYSTEM_PROMPT", ""),
    )
    result["session"] = x_routing_session_id or "default"
    return result


@app.post("/golden/benchmark/e2e-judge")
async def golden_benchmark_e2e(
    body: BenchmarkE2EBody,
    x_routing_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise HTTPException(
            503,
            "OPENAI_API_KEY not set — required for tier completions and judge (see GET /info provider_hint).",
        )
    if not GOLDEN_PATH.is_file():
        raise HTTPException(404, "golden file missing")

    rows = load_golden_rows(GOLDEN_PATH)
    rc = _router_config_from_body(
        tier_classifier=body.tier_classifier,
        tier_router_api_base=body.tier_router_api_base,
        tier_router_api_key=body.tier_router_api_key,
        tier_local_router_api_base=body.tier_local_router_api_base,
        tier_local_router_api_key=body.tier_local_router_api_key,
        tier_classifier_llm_model=body.tier_classifier_llm_model,
        tier_force_completion_model=body.tier_force_completion_model,
        tier_classifier_heuristic_shortcut=body.tier_classifier_heuristic_shortcut,
    )
    llm = LLMClient(enable_langfuse=False)
    try:
        result = await run_e2e_judge(
            rows,
            router_config=rc,
            llm=llm,
            instructions=os.getenv("SMART_LAYER_SYSTEM_PROMPT", ""),
            max_rows=body.max_rows,
        )
    except Exception as e:
        raise HTTPException(502, f"Benchmark failed: {e}") from e
    result["session"] = x_routing_session_id or "default"
    return result


@app.post("/classify")
async def classify_only(body: ClassifyBody) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise HTTPException(
            503,
            "OPENAI_API_KEY not set — required unless completions use another configured provider.",
        )
    rc = _router_config_from_body(
        tier_classifier=body.tier_classifier,
        tier_router_api_base=body.tier_router_api_base,
        tier_router_api_key=body.tier_router_api_key,
        tier_local_router_api_base=body.tier_local_router_api_base,
        tier_local_router_api_key=body.tier_local_router_api_key,
        tier_classifier_llm_model=body.tier_classifier_llm_model,
        tier_force_completion_model=None,
        tier_classifier_heuristic_shortcut=body.tier_classifier_heuristic_shortcut,
    )
    agent = _demo_router(rc)
    llm = LLMClient(enable_langfuse=False)
    forced = parse_product_tier(body.forced_product_tier) if body.forced_product_tier else None

    out = await classify_product_tier(
        user_text=body.message,
        router_config=agent.router_config,
        llm_client=llm,
        forced_tier=forced,
    )
    return {
        "tier": out.tier.value,
        "skipped_classifier": out.skipped_classifier,
        "classifier_skip_reason": out.skip_reason,
        "classify_ms": round(out.classify_ms, 3),
        "tier_classifier": agent.router_config.tier_classifier,
    }


@app.post("/chat/stream")
async def chat_stream(
    body: ChatStreamBody,
    x_routing_session_id: Annotated[str | None, Header()] = None,
):
    if not settings.openai_api_key:
        raise HTTPException(
            503,
            "OPENAI_API_KEY not set — required for completions with OpenAI models (see GET /info).",
        )

    rc = _router_config_from_body(
        tier_classifier=body.tier_classifier,
        tier_router_api_base=body.tier_router_api_base,
        tier_router_api_key=body.tier_router_api_key,
        tier_local_router_api_base=body.tier_local_router_api_base,
        tier_local_router_api_key=body.tier_local_router_api_key,
        tier_classifier_llm_model=body.tier_classifier_llm_model,
        tier_force_completion_model=body.tier_force_completion_model,
        tier_classifier_heuristic_shortcut=body.tier_classifier_heuristic_shortcut,
    )

    agent = _demo_router(rc)
    llm = LLMClient(enable_langfuse=False)
    forced = parse_product_tier(body.forced_product_tier) if body.forced_product_tier else None
    sid = x_routing_session_id or "default"

    async def gen() -> AsyncGenerator[str, None]:
        last_routing: dict[str, Any] = {}
        try:
            async for ev in stream_model_tier_turn(
                agent,
                llm,
                user_text=body.message.strip(),
                forced_tier=forced,
            ):
                if ev.kind == "routing" and ev.routing:
                    last_routing = ev.routing
                    payload = {"type": "routing", **ev.routing}
                    yield f"data: {json.dumps(payload)}\n\n"
                elif ev.kind == "content_delta" and ev.text:
                    yield f"data: {json.dumps({'type': 'content', 'text': ev.text})}\n\n"
                elif ev.kind == "done" and ev.routing:
                    last_routing = ev.routing
                    await stats_store.record(sid, ev.routing)
                    yield f"data: {json.dumps({'type': 'done', 'routing': ev.routing})}\n\n"
        except Exception as e:
            err_payload: dict[str, Any] = {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            if last_routing:
                err_payload["routing"] = last_routing
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def spa_root() -> FileResponse:
    """SPA last so API paths like /info are not shadowed by routing order."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
