#!/usr/bin/env python3
"""TL-65 Web UI — interactively verify Continuum's provider-independence.

A tiny FastAPI page to SEE the fix:
  - which model each provider scenario resolves to (offline), and
  - a live routing call per scenario, including "Anthropic-only (OpenAI key removed)"
    which proves meta-operations need no OpenAI credential.

Run:  python playground/tl65_web.py   (serves on http://localhost:8095)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import continuum.config as config_mod  # noqa: E402
from continuum.config import Settings  # noqa: E402
from continuum.llm import LLMClient  # noqa: E402
from continuum.llm.config import LLMConfig  # noqa: E402
from continuum.llm.providers import get_provider  # noqa: E402

app = FastAPI(title="TL-65 provider independence")

ROUTING_PROMPT = (
    "Available agents:\n- billing: payments, invoices, refunds\n- technical: bugs, outages\n\n"
    "User request: my invoice is wrong\nRespond with ONLY the agent name."
)

SCENARIOS = {
    "openai": {"label": "OpenAI only", "keys": {"openai_api_key": "x"}},
    "anthropic": {"label": "Anthropic only", "keys": {"anthropic_api_key": "x"}},
    "gemini": {"label": "Gemini only", "keys": {"gemini_api_key": "x"}},
}


def _resolve(keys: dict) -> str:
    base = {"openai_api_key": None, "anthropic_api_key": None, "gemini_api_key": None}
    base.update(keys)
    return Settings(**base).default_llm_model


def _provider_name(model: str) -> str:
    return type(get_provider(LLMConfig(model=model))).__name__


@app.get("/api/resolve")
def resolve() -> JSONResponse:
    rows = []
    for key, sc in SCENARIOS.items():
        model = _resolve(sc["keys"])
        rows.append(
            {
                "id": key,
                "label": sc["label"],
                "model": model,
                "provider": _provider_name(model),
                "openai": "gpt" in model.lower(),
            }
        )
    # env-derived (what THIS machine would use with its real keys)
    env_model = config_mod.settings.default_llm_model
    rows.append(
        {
            "id": "env",
            "label": "Your .env (as configured now)",
            "model": env_model,
            "provider": _provider_name(env_model),
            "openai": "gpt" in env_model.lower(),
        }
    )
    return JSONResponse({"rows": rows})


class CallReq(BaseModel):
    scenario: str  # "openai" | "anthropic" | "gemini" | "anthropic_no_openai"


@app.post("/api/call")
async def call(req: CallReq) -> JSONResponse:
    disable_openai = req.scenario == "anthropic_no_openai"
    sc_key = "anthropic" if disable_openai else req.scenario
    sc = SCENARIOS.get(sc_key)
    if not sc:
        return JSONResponse(
            {"ok": False, "error": f"unknown scenario {req.scenario}"}, status_code=400
        )

    model = _resolve(sc["keys"])
    saved_env = None
    saved_setting = None
    if disable_openai:
        saved_env = os.environ.pop("OPENAI_API_KEY", None)
        saved_setting = config_mod.settings.openai_api_key
        config_mod.settings.openai_api_key = None
    try:
        client = LLMClient(config=LLMConfig(model=model), enable_langfuse=False)
        resp = await client.chat(
            messages=[{"role": "user", "content": ROUTING_PROMPT}],
            config=LLMConfig(model=model, temperature=0.1, max_tokens=16),
            auto_session=False,
        )
        return JSONResponse(
            {
                "ok": True,
                "model": model,
                "provider": _provider_name(model),
                "openai_disabled": disable_openai,
                "response": (resp.content or "").strip(),
            }
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "model": model, "openai_disabled": disable_openai, "error": str(e)[:300]}
        )
    finally:
        if disable_openai:
            if saved_env is not None:
                os.environ["OPENAI_API_KEY"] = saved_env
            config_mod.settings.openai_api_key = saved_setting


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>TL-65 - Provider Independence</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:32px auto;padding:0 16px;color:#1a2a4a}
 h1{font-size:22px} h2{font-size:16px;margin-top:28px;color:#2563eb}
 table{border-collapse:collapse;width:100%;margin:10px 0}
 th,td{border:1px solid #dfe3ea;padding:8px 10px;text-align:left;font-size:14px}
 th{background:#1a2a4a;color:#fff}
 code{background:#f4f6fa;padding:1px 5px;border-radius:4px}
 .ok{color:#167a3c;font-weight:600}.bad{color:#aa2828;font-weight:600}
 button{background:#2563eb;color:#fff;border:0;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13px}
 button:hover{background:#1e50c0}
 .card{border:1px solid #dfe3ea;border-radius:8px;padding:14px;margin:10px 0}
 .muted{color:#6a7180;font-size:13px}
 pre{background:#f4f6fa;padding:10px;border-radius:6px;white-space:pre-wrap}
</style></head><body>
<h1>TL-65 - Continuum provider independence</h1>
<p class="muted">Before the fix, background operations (routing, reflection, memory, summarization)
always defaulted to <code>gpt-4o-mini</code>, so an Anthropic-only shop hit "Missing OpenAI credentials".
This page shows the default is now provider-aware, and lets you fire a real routing call per provider.</p>

<h2>1. Which model does each setup resolve to? (offline)</h2>
<table id="resolveTbl"><thead><tr><th>Scenario</th><th>Resolved default model</th><th>Provider</th><th>Uses OpenAI?</th></tr></thead><tbody></tbody></table>

<h2>2. Live routing call</h2>
<div class="card">
 <button onclick="callScenario('openai')">Call as OpenAI</button>
 <button onclick="callScenario('anthropic')">Call as Anthropic</button>
 <button onclick="callScenario('anthropic_no_openai')">Anthropic-only (remove OpenAI key)</button>
 <p class="muted">The last one temporarily removes the OpenAI key before the call - if it still
 answers, the routing path needed no OpenAI credential.</p>
 <pre id="out">(no call yet)</pre>
</div>

<script>
async function load(){
  const r = await fetch('/api/resolve'); const d = await r.json();
  const tb = document.querySelector('#resolveTbl tbody'); tb.innerHTML='';
  for(const row of d.rows){
    const uses = row.openai ? '<span class="bad">yes</span>' : '<span class="ok">no</span>';
    tb.innerHTML += `<tr><td>${row.label}</td><td><code>${row.model}</code></td><td>${row.provider}</td><td>${uses}</td></tr>`;
  }
}
async function callScenario(s){
  const out = document.getElementById('out'); out.textContent = 'calling ('+s+')...';
  try{
    const r = await fetch('/api/call',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:s})});
    const d = await r.json();
    if(d.ok){
      out.innerHTML = `<b class="ok">OK</b>  model=<code>${d.model}</code>  provider=${d.provider}`
        + (d.openai_disabled ? '  (OpenAI key was removed)' : '')
        + `\nrouting answer: <b>${d.response}</b>`;
    } else {
      out.innerHTML = `<b class="bad">FAIL</b>  model=<code>${d.model||''}</code>\n${d.error}`;
    }
  }catch(e){ out.textContent = 'error: '+e; }
}
load();
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


if __name__ == "__main__":
    print("TL-65 web UI -> http://localhost:8095")
    uvicorn.run(app, host="127.0.0.1", port=8095, log_level="warning")
