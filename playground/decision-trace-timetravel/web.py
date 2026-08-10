#!/usr/bin/env python3
"""
Decision-Trace TimeTravel — Web UI + REST backend.

Exercises the multi-agent Decision Trace work (handoff fork + Sequential/Router
workflow fork) end-to-end against the real runner, MCP tools, and Redis.

  POST /run    {mode, message}          run a topology; persist + return its trace
  GET  /runs                             in-process run index → tree
  GET  /trace/{id}                       reload a trace from Redis (full detail)
  POST /fork   {run_id, from_step, ...}  runner.fork — handoff resume OR Forkable
  GET  /diff   ?before&after             diff_traces

Usage:
  Terminal 1: python server.py   (MCP tools on :8896)
  Terminal 2: python web.py      (Web UI on :8087)
"""

import os
import sys

import config  # noqa: F401  (.env + DECISION_TRACE_* env, before orchestrator import)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
import itertools
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from agents import BUILDERS
from config import default_config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from continuum import (
    AgentRunner,
    LogLevel,
    MCPServerStreamableHttp,
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    get_logger,
    setup_logging,
)
from continuum.agent.trace import diff_traces
from continuum.agent.trace.types import TraceDetail
from continuum.agent.utils.context_utils import create_run_context
from continuum.core.container import get_container
from continuum.core.lifecycle import get_lifecycle_manager

setup_logging(level=LogLevel.INFO)
logger = get_logger(__name__)

# Workflow orchestrators: run via entry.execute(); fork via agent=entry. Everything
# except "handoff" (which runs via runner.run and resumes the step's agent).
WORKFLOW_MODES = {
    "sequential",
    "router",
    "loop",
    "reflection",
    "supervised",
    "planning",
    "parallel",
    "scatter",
    "debate",
}


class _State:
    def __init__(self) -> None:
        self.lifecycle = None
        self.container = None
        self.mcp = None
        self.tool_executor = None
        self.runner: AgentRunner | None = None
        self.entries: dict[str, Any] = {}  # mode -> entry agent/orchestrator
        self.runs: dict[str, dict[str, Any]] = {}
        self.seq = itertools.count()
        self.bg_tasks: set[asyncio.Task] = set()
        self.ready = False
        self.init_error: str | None = None


state = _State()


def _enable_trace_settings() -> None:
    from continuum.agent.trace import config as trace_config
    from continuum.config import settings

    settings.decision_trace_enabled = True
    settings.decision_trace_detail = default_config.decision_trace_detail
    settings.decision_trace_store = default_config.decision_trace_store
    settings.decision_trace_checkpoint = default_config.decision_trace_checkpoint
    trace_config.get_trace_store.cache_clear()


async def _startup() -> None:
    try:
        _enable_trace_settings()
        state.lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False, verify_connections=True, enable_signal_handlers=False
        )
        await state.lifecycle.initialize()
        state.container = get_container()

        state.mcp = MCPServerStreamableHttp(
            params={"url": default_config.mcp_url},
            name=default_config.mcp_server_name,
            client_session_timeout_seconds=default_config.mcp_timeout,
        )
        await state.mcp.connect()
        tool_defs = await MCPUtil.get_function_tools(state.mcp)
        tools = [t if isinstance(t, dict) else t.model_dump() for t in tool_defs]
        state.tool_executor = ToolExecutor({state.mcp: None})
        await state.tool_executor.initialize()

        state.runner = AgentRunner(
            container=state.container,
            tool_executor=state.tool_executor,
            config=RunnerConfig(default_max_turns=default_config.max_turns),
        )
        # Build all three topologies; register every concrete agent on the runner.
        m = default_config.model
        for mode, builder in BUILDERS.items():
            entry, agents = builder(tools, state.tool_executor, m)
            state.entries[mode] = entry
            for a in agents:
                state.runner.register_agent(a)
        state.ready = True
        logger.info("✓ Decision-Trace TimeTravel ready (modes: %s)", ", ".join(state.entries))
    except Exception as e:
        state.init_error = str(e)
        logger.error(f"Startup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield
    if state.mcp:
        try:
            await state.mcp.cleanup()
        except Exception:
            pass
    if state.lifecycle:
        try:
            await state.lifecycle.shutdown()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


import re as _re

_VERDICT_RE = _re.compile(r"(?:DECISION:\s*|STATUS\s*=\s*)(CLEAN|CONTROL_ISSUE)", _re.IGNORECASE)


def _outcome(text: str) -> str | None:
    """Read the close verdict from the authoritative DECISION:/STATUS= line (not
    the prose — a CLEAN summary often says 'no control issues'). Returns 'clean'
    or 'issue'."""
    m = _VERDICT_RE.findall(text or "")
    if m:
        return "issue" if m[-1].upper() == "CONTROL_ISSUE" else "clean"
    return None


# The deterministic source of truth: compute_consolidation returns
# {"status": "CLEAN"|"CONTROL_ISSUE", ...}. Read THAT from the trace rather than
# the report LLM's free-form DECISION: line, which can disagree (the report stage
# narrates and may call a booked-but-material $2M misstatement a "control issue"
# even when consolidation nets to zero). Tolerates escaped quotes (\"status\")
# from tool results serialized as JSON strings. Step status fields use
# completed/running/error — never CLEAN/CONTROL_ISSUE — so this only matches the
# consolidation result.
_STATUS_RE = _re.compile(r'\\?"status\\?"\s*:\s*\\?"(CLEAN|CONTROL_ISSUE)', _re.IGNORECASE)


def _outcome_from_trace(trace) -> str | None:
    """Deterministic verdict from compute_consolidation's status in the trace.
    Returns 'clean'/'issue', or None if no consolidation result is present (the
    caller then falls back to the DECISION:/STATUS= text)."""
    if not trace:
        return None
    import json as _json

    matches = _STATUS_RE.findall(_json.dumps(trace))
    if matches:
        return "issue" if matches[-1].upper() == "CONTROL_ISSUE" else "clean"
    return None


def _register_run(run_id, mode, label, query, final, *, parent=None, from_step=None, trace=None):
    entry = {
        "run_id": run_id,
        "mode": mode,
        "label": label,
        "query": query,
        "parent_run_id": parent,
        "forked_from_step": from_step,
        # Prefer the deterministic tool verdict; fall back to the LLM DECISION: line.
        "outcome": _outcome_from_trace(trace) or _outcome(final),
        "seq": next(state.seq),
    }
    state.runs[run_id] = entry
    return entry


async def _trace_dict(run_id: str):
    from continuum.agent.trace.config import get_trace_store

    trace = await get_trace_store().get(run_id)
    return trace.to_dict(TraceDetail.FULL) if trace else None


def _not_ready():
    if state.ready:
        return None
    return JSONResponse(status_code=503, content={"error": state.init_error or "starting up"})


class RunRequest(BaseModel):
    mode: str = "handoff"
    message: str


class ForkRequest(BaseModel):
    run_id: str
    from_step: str
    override: dict[str, Any] | None = None
    label: str | None = None


class BranchRequest(BaseModel):
    run_id: str
    from_step: str
    branches: list[dict[str, Any]]  # [{label, override}, ...] — forked concurrently


def _fork_agent_arg(run_id: str):
    """(mode, agent=) for forking a run — workflows resume via the orchestrator
    (Forkable); handoffs resume the step's agent (agent=None)."""
    parent = state.runs.get(run_id)
    mode = parent["mode"] if parent else "handoff"
    return mode, (state.entries.get(mode) if mode in WORKFLOW_MODES else None)


@app.get("/status")
async def status():
    return {
        "ready": state.ready,
        "error": state.init_error,
        "modes": list(state.entries),
        "model": default_config.model,
        "run_count": len(state.runs),
    }


@app.get("/runs")
async def runs():
    return {"runs": sorted(state.runs.values(), key=lambda r: r["seq"])}


@app.get("/trace/{run_id}")
async def trace(run_id: str):
    if nr := _not_ready():
        return nr
    t = await _trace_dict(run_id)
    if t is None:
        return JSONResponse(status_code=404, content={"error": f"No trace for {run_id}"})
    return {"trace": t}


@app.post("/run")
async def run(req: RunRequest):
    if nr := _not_ready():
        return nr
    if req.mode not in state.entries:
        return JSONResponse(status_code=400, content={"error": f"unknown mode {req.mode}"})
    entry = state.entries[req.mode]
    ctx = create_run_context(max_turns=default_config.max_turns)
    try:
        if req.mode in WORKFLOW_MODES:
            resp = await entry.execute(req.message, state.runner, ctx)
        else:
            resp = await state.runner.run(entry, req.message, context=ctx)
    except Exception as e:
        logger.error(f"run failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    run_id = resp.run_id or ctx.run_id
    label = f"[{req.mode}] " + (req.message[:38] + ("…" if len(req.message) > 38 else ""))
    trace = await _trace_dict(run_id)
    _register_run(run_id, req.mode, label, req.message, resp.content or "", trace=trace)
    return {"run_id": run_id, "response": resp.content, "trace": trace}


@app.post("/fork")
async def fork(req: ForkRequest):
    if nr := _not_ready():
        return nr
    parent = state.runs.get(req.run_id)
    mode = parent["mode"] if parent else "handoff"
    # Workflows resume via the orchestrator (Forkable); handoffs resume the step's agent.
    agent_arg = state.entries.get(mode) if mode in WORKFLOW_MODES else None
    try:
        resp = await state.runner.fork(
            req.run_id, req.from_step, override=req.override, agent=agent_arg, label=req.label
        )
    except Exception as e:
        logger.error(f"fork failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    run_id = resp.run_id
    trace = await _trace_dict(run_id)
    _register_run(
        run_id,
        mode,
        req.label or f"fork @ {req.from_step}",
        parent["query"] if parent else "",
        resp.content or "",
        parent=req.run_id,
        from_step=req.from_step,
        trace=trace,
    )
    return {"run_id": run_id, "response": resp.content, "trace": trace}


async def _branch_fork(run_id, from_step, override, label, placeholder_id, mode, agent_arg):
    """One branch, run as a fire-and-forget task; replaces its pending placeholder
    in the run index with the real forked run when it completes."""
    try:
        resp = await state.runner.fork(
            run_id, from_step, override=override, agent=agent_arg, label=label
        )
        parent = state.runs.get(run_id)
        trace = await _trace_dict(resp.run_id)
        _register_run(
            resp.run_id,
            mode,
            label,
            parent["query"] if parent else "",
            resp.content or "",
            parent=run_id,
            from_step=from_step,
            trace=trace,
        )
        state.runs.pop(placeholder_id, None)
    except Exception as e:
        logger.error(f"branch fork failed: {e}")
        if placeholder_id in state.runs:
            state.runs[placeholder_id]["outcome"] = "error"
            state.runs[placeholder_id]["label"] = f"{label} (failed)"


@app.post("/branch")
async def branch(req: BranchRequest):
    """Branch & compare: fork the same step several ways CONCURRENTLY (fire-and-
    forget); the UI polls /runs and fills the comparison in as each finishes."""
    if nr := _not_ready():
        return nr
    mode, agent_arg = _fork_agent_arg(req.run_id)
    started = []
    for b in req.branches:
        label = b.get("label", "branch")
        pid = f"pending-{next(state.seq)}"
        state.runs[pid] = {
            "run_id": pid,
            "mode": mode,
            "label": label,
            "query": "",
            "parent_run_id": req.run_id,
            "forked_from_step": req.from_step,
            "outcome": "pending",
            "seq": next(state.seq),
        }
        task = asyncio.create_task(
            _branch_fork(req.run_id, req.from_step, b.get("override"), label, pid, mode, agent_arg)
        )
        state.bg_tasks.add(task)
        task.add_done_callback(state.bg_tasks.discard)
        started.append(pid)
    return {"started": started, "count": len(started)}


@app.get("/diff")
async def diff(before: str, after: str):
    if nr := _not_ready():
        return nr
    from continuum.agent.trace.config import get_trace_store

    store = get_trace_store()
    a = await store.get(before)
    b = await store.get(after)
    if a is None or b is None:
        return JSONResponse(status_code=404, content={"error": "run(s) not found"})
    d = diff_traces(a, b)
    # Time-travel savings: a fork re-executes only steps from the fork point on;
    # earlier steps were replayed from the checkpoint (no LLM/tool calls).
    d["savings"] = {
        "parent_total_steps": len(a.steps),
        "reexecuted_steps": len(b.steps),
        "replayed_from_checkpoint": max(0, len(a.steps) - len(b.steps)),
    }
    return {"diff": d}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Decision-Trace TimeTravel</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0e1116;color:#e6edf3;height:100vh;display:flex;flex-direction:column}
  .topbar{background:#161b22;border-bottom:1px solid #30363d;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .brand{font-weight:700}.brand b{color:#58a6ff}.tag{color:#8b949e;font-size:12px}.spacer{flex:1}
  select,#q{padding:8px 10px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:13px}
  #q{flex:1;min-width:220px}
  button{cursor:pointer;border-radius:8px;border:1px solid #30363d;background:#21262d;color:#e6edf3;padding:8px 14px;font-size:13px}
  button.primary{background:#238636;border-color:#2ea043;font-weight:600}button:disabled{opacity:.5}
  .chips{padding:6px 16px;display:flex;gap:6px;flex-wrap:wrap}
  .chip{font-size:12px;padding:4px 10px;border-radius:14px;background:#161b22;border:1px solid #30363d;color:#8b949e}
  .chip:hover{color:#58a6ff;border-color:#58a6ff}
  .layout{flex:1;display:grid;grid-template-columns:240px 1fr 430px;overflow:hidden}
  .col{overflow-y:auto;padding:12px}.col-runs{border-right:1px solid #30363d}.col-center{border-right:1px solid #30363d}
  .col-head{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b949e;margin-bottom:10px}
  .hint{color:#6e7681;font-size:12px;line-height:1.5}
  .run-item{padding:8px 10px;border-radius:8px;border:1px solid #30363d;margin-bottom:6px;background:#161b22}
  .run-item:hover{border-color:#58a6ff}.run-item.active{border-color:#58a6ff;background:#1c2433}.run-item.child{margin-left:14px}
  .run-item .label{font-size:12px;margin-bottom:4px}.run-item .fork-tag{font-size:11px;color:#d29922;margin-bottom:3px}
  .pill{font-size:11px;padding:1px 8px;border-radius:10px;font-weight:600}
  .pill.clean{background:#18351f;color:#3fb950}.pill.issue{background:#3a1a1c;color:#f85149}.pill.none{background:#21262d;color:#8b949e}
  .pill.pending{background:#2a2410;color:#d29922;animation:pulse 1.1s ease-in-out infinite}.pill.error{background:#3a1a1c;color:#f85149}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
  .savings{background:#102f2a;border:1px solid #1a5;border-radius:8px;padding:8px;font-size:12px;color:#56d4bc;margin-top:6px}
  table.cmp{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}.cmp td,.cmp th{border:1px solid #30363d;padding:5px 7px;text-align:left}
  .step{border:1px solid #30363d;border-left-width:3px;border-radius:8px;padding:9px 11px;margin-bottom:8px;background:#161b22}
  .step:hover{border-color:#58a6ff}.step.active{background:#1c2433}
  .step-head{display:flex;align-items:center;gap:8px;margin-bottom:3px}
  .agent{font-size:11px;font-weight:700;padding:1px 7px;border-radius:6px;background:#21262d}
  .kind{font-size:10px;color:#8b949e;text-transform:uppercase}
  .step .snippet{font-size:12px;color:#adbac7;margin-top:4px;white-space:pre-wrap;max-height:70px;overflow:hidden}
  .step .meta{font-size:11px;color:#6e7681;margin-top:4px}
  .card{border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;background:#161b22}
  .card h4{font-size:12px;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.04em}
  pre.code{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px;font-size:11px;overflow-x:auto;white-space:pre-wrap;color:#adbac7}
  .answer{font-size:13px;line-height:1.5;white-space:pre-wrap}
  label.fld{display:block;font-size:11px;color:#8b949e;margin:8px 0 3px}
  textarea,input.txt{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:12px;padding:6px;font-family:inherit}
  textarea{min-height:50px}.fork-btn{margin-top:6px;width:100%}
  .lever{background:#1c2433;border:1px solid #2a4a7f;border-radius:8px;padding:10px;margin:8px 0}
  input.num{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:13px;padding:7px}
  .btn-row{display:flex;gap:6px;margin-top:6px}.btn-row button{flex:1}
  .step.bt{box-shadow:inset 0 0 0 1px #d29922}
  .bt-badge{margin-left:6px;font-size:10px;color:#d29922;border:1px solid #d29922;border-radius:8px;padding:0 6px}
  .route-btns{display:flex;gap:6px;flex-wrap:wrap}.route-btns button{flex:1}
  .flip{background:#0d1117;border-radius:6px;padding:8px;font-size:12px}.flip .before{color:#f85149}.flip .after{color:#3fb950;margin-top:4px}
  .agentchain{font-size:11px;color:#8b949e;margin-top:6px}
  .metric{display:inline-block;font-size:11px;color:#8b949e;margin-right:12px}
  .banner{background:#3a1a1c;color:#f85149;padding:8px 16px;font-size:13px}.empty{color:#6e7681;font-size:13px;padding:20px 4px}
</style></head><body>
<div class="topbar">
  <div class="brand">◆ <b>Decision-Trace TimeTravel</b></div>
  <div class="tag">trace + fork across handoffs & workflows</div>
  <span class="spacer"></span>
  <select id="mode">
    <option value="sequential">sequential (ingest→…→variance→…→report)</option>
    <option value="supervised">supervised (close, supervisor-scored)</option>
    <option value="planning">planning (plan → execute the close)</option>
    <option value="loop">loop (close-officer ↻ until verdict)</option>
    <option value="reflection">reflection (draft → critique balance → revise)</option>
    <option value="parallel">parallel (materiality ‖ intercompany → relay)</option>
    <option value="scatter">scatter (close facets ‖ → gather)</option>
    <option value="router">router (strict $1M vs lenient $5M close)</option>
    <option value="debate">debate (materially-fine vs control-issue → judge)</option>
    <option value="handoff">handoff (triage→officer→controller)</option>
  </select>
  <input id="q" placeholder="e.g. Refund my order ORD-1004"/>
  <button class="primary" id="run-btn" onclick="startRun()">▶ Run</button>
</div>
<div id="banner" class="banner" style="display:none"></div>
<div class="chips" id="chips"></div>
<div class="layout">
  <div class="col col-runs"><div class="col-head">Run tree</div><div id="runs"></div></div>
  <div class="col col-center"><div class="col-head">Decision steps (by agent)</div>
    <div id="steps"><div class="empty">Pick a mode, run a query, then click a step to fork it.</div></div></div>
  <div class="col col-right"><div id="inspector"><div class="empty">Select a run.</div></div></div>
</div>
<script>
const _CLOSE=["Run the April 2026 month-end close and give the verdict.","Close the April 2026 books and report the control status."];
const SAMPLES={sequential:_CLOSE,supervised:_CLOSE,planning:_CLOSE,loop:_CLOSE,reflection:_CLOSE,
  parallel:_CLOSE,scatter:_CLOSE,
  // Router conditions are the literal words "strict"/"lenient" — a generic close
  // query carries no signal and always falls through to the lenient default, so
  // these samples exercise BOTH routes: "strict" → $1M → CLEAN, "lenient" → $5M → CONTROL_ISSUE.
  router:["Run a STRICT SOX close for April 2026 and give the verdict.","Run the standard LENIENT April 2026 close and give the verdict."],
  debate:["Is the April 2026 close materially fine, or a control issue?"],handoff:_CLOSE};
const COLORS={};const PALETTE=["#1f2a44","#2d2440","#102f2a","#3a2a12","#3a1a2a","#13314a","#2a2a13"];
function agentColor(n){if(!(n in COLORS)){COLORS[n]=PALETTE[Object.keys(COLORS).length%PALETTE.length];}return COLORS[n];}
let RUNS=[],activeRunId=null,activeTrace=null,selStep=null,ROUTES=["lenient-close","strict-close"];
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function pillText(o){const t=o==='pending'?'running…':o==='error'?'failed':(o||'—');return `<span class="pill ${o||'none'}">${t}</span>`;}
function shortJson(v,n=200){if(v==null)return '';let s=typeof v==='string'?v:JSON.stringify(v);if(s==null||s==='null')return '';return s.length>n?s.slice(0,n)+'…':s;}
async function api(p,o){const r=await fetch(p,o);const d=await r.json().catch(()=>({error:'bad'}));if(!r.ok)throw new Error(d.error||r.status);return d;}
function setBanner(m){const b=document.getElementById('banner');if(m){b.textContent=m;b.style.display='block';}else b.style.display='none';}
function curMode(){return document.getElementById('mode').value;}
function renderChips(){const m=curMode();document.getElementById('chips').innerHTML=(SAMPLES[m]||[]).map(s=>`<span class="chip" onclick='pick(${JSON.stringify(s)})'>${esc(s)}</span>`).join('');}
function pick(s){document.getElementById('q').value=s;startRun();}
async function refreshRuns(){try{RUNS=(await api('/runs')).runs;}catch(e){RUNS=[];}renderRuns();}
function renderRuns(){const el=document.getElementById('runs');if(!RUNS.length){el.innerHTML='<div class="hint">No runs yet.</div>';return;}
  el.innerHTML=RUNS.map(r=>{const cls=(r.run_id===activeRunId?'active ':'')+(r.parent_run_id?'child':'');
    const pill=pillText(r.outcome);
    const tag=r.parent_run_id?`<div class="fork-tag">↳ fork @ ${esc(r.forked_from_step)}</div>`:'';
    return `<div class="run-item ${cls}" onclick='selectRun(${JSON.stringify(r.run_id)})'>${tag}<div class="label">${esc(r.label)}</div>${pill}</div>`;}).join('');}
async function startRun(){const q=document.getElementById('q').value.trim();if(!q)return;setBanner(null);
  document.getElementById('run-btn').disabled=true;document.getElementById('steps').innerHTML='<div class="empty">Running…</div>';
  try{const d=await api('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:curMode(),message:q})});
    await refreshRuns();selectRunData(d.run_id,d.trace);}catch(e){setBanner('Run failed: '+e.message);document.getElementById('steps').innerHTML='';}
  document.getElementById('run-btn').disabled=false;}
async function selectRun(id){try{const d=await api('/trace/'+id);selectRunData(id,d.trace);}catch(e){setBanner(e.message);}}
function selectRunData(id,t){activeRunId=id;activeTrace=t;selStep=null;renderRuns();renderSteps();renderInspector();}
function renderSteps(){const el=document.getElementById('steps');
  if(!activeTrace||!activeTrace.steps||!activeTrace.steps.length){el.innerHTML='<div class="empty">No steps.</div>';return;}
  const bt=branchTarget();const btid=bt?bt.step_id:null;  // the step "branch & compare" rewinds from
  el.innerHTML=activeTrace.steps.map((s,i)=>{const active=(selStep===i?'active':'');
    const snip=s.rationale||shortJson(s.output)||shortJson(s.decision);
    const fork=(s.kind==='llm_call'&&s.messages_snapshot)?' · ⑃':(s.kind==='routing'?' · ⑃ route':'');
    const isBT=s.step_id===btid;
    const btBadge=isBT?`<span class="bt-badge" title="the branch & compare button rewinds from here; earlier steps replay from cache">↻ rewind point</span>`:'';
    return `<div class="step ${active}${isBT?' bt':''}" style="border-left-color:${agentColor(s.agent_name)}" onclick="selStepF(${i})">
      <div class="step-head"><span class="agent" style="background:${agentColor(s.agent_name)}">${esc(s.agent_name)}</span><span class="kind">${esc(s.kind)}</span>${btBadge}</div>
      ${snip?`<div class="snippet">${esc(snip)}</div>`:''}
      <div class="meta">${esc(s.step_id)} · stack: ${esc((s.agent_stack||[]).join(' › ')||'—')}${fork}</div></div>`;}).join('');}
function selStepF(i){selStep=i;renderSteps();renderInspector();}
function renderInspector(){const el=document.getElementById('inspector');if(!activeTrace){el.innerHTML='<div class="empty">Select a run.</div>';return;}
  const m=activeTrace.metrics||{};let h='';
  h+=`<div class="card"><h4>Run · ${esc(activeTrace.root_agent)}</h4>
    <div class="answer">${esc(activeTrace.final_response)||'<i>(none)</i>'}</div>
    <div style="margin-top:8px"><span class="metric">steps ${m.step_count||0}</span><span class="metric">agents ${(m.agents||[]).length}</span><span class="metric">tokens ${m.total_tokens||0}</span></div>
    <div class="agentchain">agents: ${esc((m.agents||[]).join(' , '))}</div>
    ${(()=>{const bt=(activeTrace.steps&&activeTrace.steps.length)?branchTarget():null;if(!bt)return '';
      const _m=(RUNS.find(r=>r.run_id===activeRunId)||{}).mode;
      const _replay=['parallel','scatter'].includes(_m)?'sibling branches replay from cache':(_m==='router'?'re-routes — little to replay':'earlier steps replay from cache');
      const _blabel=_m==='router'?'branch &amp; compare routes (strict vs lenient)':'branch &amp; compare materiality threshold';
      return `<div class="btn-row" style="margin-top:8px"><button onclick="branchThresholds()">⑃⑃⑃ ${_blabel}</button></div><div class="hint" style="margin-top:4px">↻ rewinds from <b>${esc(bt.step_id)}</b> · ${esc(bt.agent_name)} — ${_replay}</div>`;})()}
    ${activeTrace.parent_run_id?`<div class="hint" style="margin-top:6px">forked from <b>${esc(activeTrace.parent_run_id)}</b> @ ${esc(activeTrace.forked_from_step)} · edit ${esc(shortJson(activeTrace.edit))}</div>`:''}</div>`;
  h+=`<div id="diffcard"></div>`;
  if(selStep!==null&&activeTrace.steps[selStep]){const s=activeTrace.steps[selStep];
    h+=`<div class="card"><h4>${esc(s.agent_name)} · ${esc(s.step_id)} · ${esc(s.kind)}</h4>
      ${s.rationale?`<div class="answer" style="margin-bottom:8px">${esc(s.rationale)}</div>`:''}
      <pre class="code">${esc(JSON.stringify({input:s.input,decision:s.decision,output:s.output},null,2))}</pre></div>`;
    h+=forkPanel(s);}
  else h+=`<div class="card"><div class="hint">Click a step to fork it.</div></div>`;
  el.innerHTML=h;
  const entry=RUNS.find(r=>r.run_id===activeRunId);if(entry&&entry.parent_run_id)loadDiff(entry.parent_run_id,activeRunId);}
function _THR(t){const m=/MATERIALITY_THRESHOLD_USD\s*=\s*(\d+)/.exec(t||'');return m?parseInt(m[1],10):null;}
function forkLever(step){const s=activeTrace.steps.find(x=>x.step_id===step);const v=parseInt(document.getElementById('lever').value,10);
  doFork(step,{replace_last_user:thrLine(lastUserMsg(s),v)},'threshold $'+v.toLocaleString());}
function forkPanel(s){const mode=(RUNS.find(r=>r.run_id===activeRunId)||{}).mode||'handoff';
  let h=`<div class="card"><h4>⑃ Rewind from ${esc(s.step_id)} · ${esc(s.agent_name)} — what-if</h4>`;
  // Materiality threshold lever — shown when this step's input carries the line
  // (the Variance stage). Mirrors local/glassbox: edit the number, rewind, branch.
  const _thr=_THR(lastUserMsg(s));
  // Decider-fetches-threshold modes: the agent calls get_materiality_policy, so
  // its input has NO threshold line — but workflow resume re-runs fresh, so a
  // replace_last_user that INJECTS the line works (the agent uses it instead of
  // calling the tool, per _GET_THRESHOLD). set_tool_result would NOT work here.
  const _inject=['loop','reflection','parallel','scatter','debate'];
  const _bt=branchTarget();const _isBT=_bt&&_bt.step_id===s.step_id;
  if(_thr!=null||(_isBT&&_inject.includes(mode))){const _lv=_thr!=null?_thr:5000000;
    h+=`<div class="lever"><div class="hint">Materiality threshold lever — discrepancies ≥ this are MATERIAL. The $2M D1 misstatement is waived at $5M; lower it to catch it. Re-runs from here; earlier stages replay from cache.</div>
      <label class="fld">materiality_threshold_usd</label>
      <input class="num" id="lever" type="number" value="${_lv}"/>
      <div class="btn-row"><button class="primary" onclick='forkLever(${JSON.stringify(s.step_id)})'>⑃ rewind &amp; re-run</button></div>
      <div class="btn-row"><button onclick='branchThresholds()'>⑃⑃⑃ branch &amp; compare 5M / 2.5M / 1M</button></div></div>`;}
  if(mode==='router'&&s.kind==='routing'){
    h+=`<div class="hint">Re-route this request to a different specialist.</div><div class="route-btns">`;
    h+=ROUTES.map(r=>`<button onclick='doFork(${JSON.stringify(s.step_id)},{route:${JSON.stringify(r)}},"re-route ${r}")'>${r.replace('-specialist','').replace('-agent','')}</button>`).join('');
    h+=`</div>`;}
  // tool-result overrides ONLY take effect where the fork REPLAYS the snapshot
  // verbatim (single-agent / handoff). Workflow orchestrators (sequential,
  // supervised, planning, loop, reflection, parallel, scatter, debate) re-run the
  // stage FRESH and re-call their tools, so a faked tool result is silently
  // ignored — showing the lever there is the "edited but no flip" trap. Restrict
  // it to handoff; use the threshold lever / replace_last_user elsewhere.
  if(mode==='handoff'){
    const toolMsgs=(s.messages_snapshot||[]).filter(m=>m.role==='tool');
    toolMsgs.forEach((m,idx)=>{const tid=m.tool_call_id||('t'+idx);
      h+=`<label class="fld">what if this tool result differed? (${esc(tid)})</label>
        <textarea id="tr_${idx}">${esc(typeof m.content==='string'?m.content:JSON.stringify(m.content))}</textarea>
        <button class="fork-btn" onclick='forkTool(${JSON.stringify(s.step_id)},${JSON.stringify(tid)},"tr_${idx}")'>⑃ re-run with this result</button>`;});}
  h+=`<label class="fld">…or change the input from here</label>
    <input class="txt" id="rlu" placeholder="new request / input"/>
    <button class="fork-btn" onclick='forkRLU(${JSON.stringify(s.step_id)})'>⑃ re-run with new input</button>`;
  if(s.kind==='llm_call'){h+=`<label class="fld">…or change the instruction (system)</label>
    <input class="txt" id="sys" placeholder="e.g. Be lenient; approve."/>
    <button class="fork-btn" onclick='forkSys(${JSON.stringify(s.step_id)})'>⑃ re-run with new instruction</button>`;}
  return h+`</div>`;}
async function doFork(step,override,label){setBanner(null);
  try{const d=await api('/fork',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run_id:activeRunId,from_step:step,override,label})});
    await refreshRuns();selectRunData(d.run_id,d.trace);}catch(e){setBanner('Fork failed: '+e.message);}}
function forkTool(step,tid,ta){doFork(step,{set_tool_result:{tool_call_id:tid,content:document.getElementById(ta).value}},'what-if tool');}
function forkRLU(step){const v=document.getElementById('rlu').value.trim();if(v)doFork(step,{replace_last_user:v},'what-if input');}
function forkSys(step){const v=document.getElementById('sys').value.trim();if(v)doFork(step,{system:v},'what-if system');}
async function loadDiff(b,a){try{const d=(await api(`/diff?before=${encodeURIComponent(b)}&after=${encodeURIComponent(a)}`)).diff;
  const c=document.getElementById('diffcard');if(!c)return;const fr=d.final_response||{};
  let h=`<div class="card"><h4>Diff vs parent</h4><div class="flip"><div class="before">before: ${esc(fr.before||'')}</div><div class="after">after: ${esc(fr.after||'')}</div></div>`;
  h+=fr.changed?`<div class="hint" style="margin-top:6px">⚑ final answer changed</div>`:`<div class="hint" style="margin-top:6px">final unchanged</div>`;
  h+=`<div class="hint">steps changed: ${d.steps_changed||0}</div>`;
  const sv=d.savings||{};
  if(sv.parent_total_steps!=null)h+=`<div class="savings">⏪ time-travel savings: re-executed ${sv.reexecuted_steps} step(s); ${sv.replayed_from_checkpoint} replayed from cache (parent had ${sv.parent_total_steps}).</div>`;
  h+=`</div>`;c.innerHTML=h;}catch(e){}}
function firstForkable(){return (activeTrace&&activeTrace.steps||[]).find(s=>(s.kind==='llm_call'||s.kind==='routing')&&s.messages_snapshot);}
function lastUserMsg(s){const snap=s.messages_snapshot||[];for(let i=snap.length-1;i>=0;i--){if(snap[i].role==='user')return String(snap[i].content||'');}return '';}
function thrLine(t,v){const ln='MATERIALITY_THRESHOLD_USD='+v;return /MATERIALITY_THRESHOLD_USD\s*=\s*\d+/.test(t)?t.replace(/MATERIALITY_THRESHOLD_USD\s*=\s*\d+/,ln):((t?t+'\n':'')+ln);}
// The universal lever is the materiality threshold. Pick the fork point that
// flips the verdict and (where possible) leaves upstream to replay from cache:
//  pipeline (sequential/supervised/planning) → the VARIANCE stage (mid) → flip + savings
//  parallel/scatter → the materiality decider branch → flip + sibling cache
//  router → the routing decision (re-route strict/lenient)
//  debate → the judge; loop/reflection → the decider; handoff → the policy step
function branchTarget(){const steps=(activeTrace&&activeTrace.steps)||[];
  const mode=(RUNS.find(r=>r.run_id===activeRunId)||{}).mode;
  const ll=steps.filter(s=>s.kind==='llm_call'&&s.messages_snapshot);
  const byAgent=re=>steps.find(s=>s.kind==='llm_call'&&s.messages_snapshot&&re.test(s.agent_name||''));
  if(['sequential','supervised','planning'].includes(mode))return byAgent(/variance/)||ll[0];
  if(['parallel','scatter'].includes(mode))return byAgent(/materiality/)||ll[0];
  if(mode==='debate')return byAgent(/judge/)||ll[ll.length-1];
  if(mode==='router'){return steps.find(s=>s.kind==='routing')||ll[0];}  // routing step has no snapshot but resume_from re-routes
  if(mode==='handoff'){const pol=steps.find(s=>s.kind==='llm_call'&&(s.messages_snapshot||[]).some(m=>m.role==='tool'&&typeof m.content==='string'&&m.content.includes('materiality_threshold_usd')));return pol||ll[ll.length-1];}
  return ll[ll.length-1]||firstForkable();  // loop / reflection: the deciding step
}
// Build a [{label, override}] for thresholds 5M/2.5M/1M at the chosen step.
function thresholdBranches(s,mode){const THR=[5000000,2500000,1000000];
  if(mode==='router'){return [["lenient $5M","lenient-close"],["strict $1M","strict-close"]]
      .map(([label,route])=>({label,override:{route}}));}
  if(mode==='handoff'){const snap=s.messages_snapshot||[];const tool=snap.find(m=>m.role==='tool'&&typeof m.content==='string'&&m.content.includes('materiality_threshold_usd'));
    const tid=tool?tool.tool_call_id:null;
    return THR.map(v=>({label:'$'+v.toLocaleString(),override:{set_tool_result:{tool_call_id:tid,content:JSON.stringify({materiality_threshold_usd:v,escalation_floor_usd:1000000})}}}));}
  const base=lastUserMsg(s);
  return THR.map(v=>({label:'$'+v.toLocaleString(),override:{replace_last_user:thrLine(base,v)}}));}
async function branchThresholds(){const s=branchTarget();if(!s){setBanner('No forkable step to branch.');return;}
  const parent=activeRunId;const mode=(RUNS.find(r=>r.run_id===parent)||{}).mode;
  const branches=thresholdBranches(s,mode);
  try{await api('/branch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run_id:parent,from_step:s.step_id,branches})});}
  catch(e){setBanner('Branch failed: '+e.message);return;}
  function render(){const kids=RUNS.filter(r=>r.parent_run_id===parent);
    const rows=branches.map(b=>{const r=kids.find(k=>k.label===b.label||k.label===b.label+' (failed)');
      const o=r?r.outcome:'pending';
      const open=(r&&!String(r.run_id).startsWith('pending'))?`<a href="#" onclick='selectRun(${JSON.stringify(r?r.run_id:'')});return false;'>open</a>`:'';
      return `<tr><td>${esc(b.label)}</td><td>${pillText(o)}</td><td>${open}</td></tr>`;}).join('');
    document.getElementById('inspector').innerHTML=`<div class="card"><h4>⑃⑃⑃ Branch &amp; compare · materiality threshold</h4>
      <div class="hint">Same fork point, run concurrently. The verdict is computed deterministically by compute_consolidation, so the $2M D1 misstatement is waived at $5M (CONTROL ISSUE) but caught at ≤$2M (CLEAN). For pipeline modes this forks the <b>Variance</b> stage, so earlier stages replay from cache — open a branch to see the savings.</div>
      <table class="cmp"><tr><th>${mode==='router'?'route':'threshold'}</th><th>verdict</th><th></th></tr>${rows}</table></div>`;}
  let ticks=0;await refreshRuns();render();
  const iv=setInterval(async()=>{ticks++;await refreshRuns();render();
    const kids=RUNS.filter(r=>r.parent_run_id===parent);
    const pending=kids.filter(r=>r.outcome==='pending'||String(r.run_id).startsWith('pending')).length;
    if((kids.length>=branches.length&&pending===0)||ticks>40)clearInterval(iv);
  },1500);}
document.getElementById('mode').addEventListener('change',renderChips);
renderChips();refreshRuns();
fetch('/status').then(r=>r.json()).then(s=>{if(!s.ready)setBanner(s.error||'Backend starting… is server.py running on :8896?');});
</script></body></html>
"""

if __name__ == "__main__":
    print(f"Decision-Trace TimeTravel UI at http://localhost:{default_config.web_port}")
    print(f"Make sure the MCP server is running:  python server.py  (:{default_config.mcp_port})")
    uvicorn.run(app, host="0.0.0.0", port=default_config.web_port)
