#!/usr/bin/env python3
"""
LeadFlow — AI Lead Discovery & Outreach
Browser UI on :8085

Usage:
  python web.py
  python web.py --host 0.0.0.0 --port 8085

Steps exercised:
  1. Parallel scraping   (create_parallel_agent via Temporal ParallelStep)
  2. Scoring & ranking   (BaseAgent + output_schema=RankedLeadList)
  3. Human review gate   (Temporal ApprovalStep + HumanInLoopManager)
  4. Voice outreach      (BaseAgent + FakeTwilioMCP + Handoff return_to_parent)
  5. Campaign durability (Temporal AgentWorkflow — survives restarts)
  7. Observability       (Langfuse traces tagged with campaign_id)
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

import argparse
import json
import re
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_root = Path(__file__).resolve().parents[1]
_here = Path(__file__).resolve().parent
for p in (_root / "src", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import LeadFlowConfig, default_config
from schemas import RankedLeadList
from temporal.campaign import build_workflow_input
from temporal.worker import setup_registry, start_worker, stop_worker

from orchestrator import LogLevel, setup_logging
from orchestrator.temporal.client import TemporalClient
from orchestrator.temporal.human_in_loop import HumanInLoopManager
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow

setup_logging(level=LogLevel.INFO)

# ── App state ────────────────────────────────────────────────────────────────

_temporal_client: TemporalClient | None = None
_hitl_manager: HumanInLoopManager | None = None
_app_config: LeadFlowConfig = default_config

# campaign_id → {niche, location, started_at, workflow_id}
_campaigns: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _temporal_client, _hitl_manager

    cfg = _app_config
    client = TemporalClient()
    try:
        from orchestrator.temporal.config import TemporalConfig

        tc = TemporalConfig(
            host=cfg.temporal_host,
            namespace=cfg.temporal_namespace,
            task_queue=cfg.task_queue,
        )
        client = TemporalClient(config=tc)
        await client.connect()
        _temporal_client = client
        _hitl_manager = HumanInLoopManager(client=client)

        registry = await setup_registry(cfg)
        await start_worker(client, registry, cfg)
    except Exception as e:
        print(f"[LeadFlow] Temporal unavailable: {e}. Campaign routes will return 503.")

    yield

    await stop_worker()
    if _temporal_client:
        await _temporal_client.disconnect()


app = FastAPI(title="LeadFlow", lifespan=lifespan)


# ── Request / response models ─────────────────────────────────────────────────


class StartCampaignRequest(BaseModel):
    niche: str
    location: str
    model: str = "gemini/gemini-2.5-flash"
    leads_per_source: int = 5


class ApproveRequest(BaseModel):
    decided_by: str = "user"
    reason: str = ""


# ── API routes ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"ok": True, "temporal": _temporal_client is not None}


@app.post("/campaign")
async def start_campaign(req: StartCampaignRequest):
    if not _temporal_client:
        raise HTTPException(503, "Temporal not connected")

    campaign_id = f"lf-{uuid.uuid4().hex[:12]}"

    cfg = LeadFlowConfig(
        model=req.model,
        leads_per_source=req.leads_per_source,
        temporal_host=_app_config.temporal_host,
        task_queue=_app_config.task_queue,
    )
    wf_input = build_workflow_input(req.niche, req.location, campaign_id, cfg)

    handle = await _temporal_client.run_agent_workflow(
        input=wf_input,
        id=campaign_id,
        task_queue=cfg.task_queue,
    )

    _campaigns[campaign_id] = {
        "niche": req.niche,
        "location": req.location,
        "workflow_id": handle.id,
        "started_at": datetime.utcnow().isoformat(),
    }

    return {"campaign_id": campaign_id, "workflow_id": handle.id}


@app.get("/campaign/{campaign_id}/status")
async def campaign_status(campaign_id: str):
    if campaign_id not in _campaigns:
        raise HTTPException(404, "Campaign not found")
    if not _temporal_client:
        raise HTTPException(503, "Temporal not connected")

    meta = _campaigns[campaign_id]
    try:
        handle = await _temporal_client.get_workflow_handle(campaign_id)
        status: dict = await handle.query(AgentWorkflow.get_status)
        pending: list = await handle.query(AgentWorkflow.get_pending_approvals)
    except Exception as e:
        return {"campaign_id": campaign_id, "error": str(e), **meta}

    leads_data = None
    if pending:
        raw_context = pending[0].get("context", "")
        leads_data = _try_parse_leads(raw_context)

    return {
        "campaign_id": campaign_id,
        "niche": meta["niche"],
        "location": meta["location"],
        "started_at": meta["started_at"],
        **status,
        "pending_approvals": pending,
        "leads": leads_data,
    }


@app.post("/campaign/{campaign_id}/approve")
async def approve_campaign(campaign_id: str, req: ApproveRequest):
    if not _hitl_manager:
        raise HTTPException(503, "Temporal not connected")

    pending = await _hitl_manager.get_pending_approvals(campaign_id)
    if not pending:
        raise HTTPException(400, "No pending approval for this campaign")

    request_id = pending[0]["request_id"]
    await _hitl_manager.approve(
        workflow_id=campaign_id,
        request_id=request_id,
        decided_by=req.decided_by,
        reason=req.reason,
    )
    return {"ok": True, "request_id": request_id}


@app.post("/campaign/{campaign_id}/reject")
async def reject_campaign(campaign_id: str, req: ApproveRequest):
    if not _hitl_manager:
        raise HTTPException(503, "Temporal not connected")

    pending = await _hitl_manager.get_pending_approvals(campaign_id)
    if not pending:
        raise HTTPException(400, "No pending approval for this campaign")

    request_id = pending[0]["request_id"]
    await _hitl_manager.reject(
        workflow_id=campaign_id,
        request_id=request_id,
        decided_by=req.decided_by,
        reason=req.reason,
    )
    return {"ok": True, "request_id": request_id}


@app.get("/campaigns")
async def list_campaigns():
    return {"campaigns": [{"campaign_id": cid, **meta} for cid, meta in _campaigns.items()]}


def _try_parse_leads(text: str) -> dict | None:
    """Try to parse the scoring agent's JSON output into RankedLeadList."""
    if not text:
        return None
    # Strip markdown code fences if the model returned ```json ... ```
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("`").rstrip()
    try:
        data = json.loads(cleaned)
        leads = RankedLeadList.model_validate(data)
        return leads.model_dump()
    except Exception:
        return {"raw": text}


# ── Browser UI ────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeadFlow — AI Lead Discovery & Outreach</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }

  header { background: #1e293b; border-bottom: 1px solid #334155;
           padding: 14px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 700; color: #f8fafc; }
  header .subtitle { font-size: 12px; color: #64748b; }

  .layout { display: grid; grid-template-columns: 340px 1fr; height: calc(100vh - 53px); }

  /* Left panel */
  .sidebar { background: #1e293b; border-right: 1px solid #334155;
             display: flex; flex-direction: column; overflow: hidden; }
  .sidebar-section { padding: 16px; border-bottom: 1px solid #334155; }
  .sidebar-section h2 { font-size: 13px; font-weight: 600; color: #94a3b8;
                        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }

  .form-group { margin-bottom: 10px; }
  .form-group label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
  .form-group input, .form-group select {
    width: 100%; padding: 8px 10px; background: #0f172a; border: 1px solid #334155;
    border-radius: 6px; color: #e2e8f0; font-size: 13px; outline: none;
  }
  .form-group input:focus, .form-group select:focus { border-color: #6366f1; }

  .btn { width: 100%; padding: 10px; border: none; border-radius: 8px;
         font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .btn-primary { background: #6366f1; color: white; }
  .btn-primary:hover:not(:disabled) { background: #4f46e5; }
  .btn-approve { background: #10b981; color: white; }
  .btn-approve:hover:not(:disabled) { background: #059669; }
  .btn-reject  { background: #ef4444; color: white; }
  .btn-reject:hover:not(:disabled)  { background: #dc2626; }

  .campaign-list { flex: 1; overflow-y: auto; }
  .campaign-item { padding: 12px 16px; border-bottom: 1px solid #1e293b; cursor: pointer; }
  .campaign-item:hover { background: #334155; }
  .campaign-item.active { background: #334155; border-left: 3px solid #6366f1; }
  .campaign-item .c-title { font-size: 13px; font-weight: 600; }
  .campaign-item .c-meta  { font-size: 11px; color: #64748b; margin-top: 2px; }
  .campaign-item .c-badge { display: inline-block; padding: 2px 6px; border-radius: 4px;
                            font-size: 10px; font-weight: 600; margin-top: 4px; }
  .badge-running   { background: #1d4ed8; color: #bfdbfe; }
  .badge-waiting   { background: #92400e; color: #fde68a; }
  .badge-completed { background: #065f46; color: #a7f3d0; }
  .badge-rejected  { background: #7f1d1d; color: #fecaca; }
  .badge-failed    { background: #7f1d1d; color: #fecaca; }

  /* Right panel */
  .main { overflow-y: auto; padding: 24px; }
  .empty-state { display: flex; flex-direction: column; align-items: center;
                 justify-content: center; height: 100%; color: #475569; text-align: center; }
  .empty-state h2 { font-size: 20px; margin-bottom: 8px; }

  /* Step progress */
  .steps { display: flex; gap: 0; margin-bottom: 24px; }
  .step { flex: 1; padding: 10px 8px; background: #1e293b; text-align: center;
          font-size: 11px; color: #64748b; border-right: 1px solid #0f172a;
          position: relative; }
  .step:last-child { border-right: none; border-radius: 0 8px 8px 0; }
  .step:first-child { border-radius: 8px 0 0 8px; }
  .step.done    { background: #065f46; color: #a7f3d0; }
  .step.active  { background: #1d4ed8; color: #bfdbfe; font-weight: 600; }
  .step.waiting { background: #92400e; color: #fde68a; font-weight: 600; }
  .step-num { display: block; font-size: 16px; margin-bottom: 2px; }

  /* Lead cards */
  .section-title { font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 12px; }
  .leads-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .lead-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }
  .lead-card .lead-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
  .lead-card .lead-name { font-size: 14px; font-weight: 600; color: #f1f5f9; }
  .lead-card .lead-rank { font-size: 11px; color: #64748b; }
  .score-bar { height: 4px; background: #334155; border-radius: 2px; margin-bottom: 8px; }
  .score-fill { height: 100%; border-radius: 2px; }
  .score-high { background: #10b981; }
  .score-mid  { background: #f59e0b; }
  .score-low  { background: #ef4444; }
  .lead-field { font-size: 11px; color: #94a3b8; margin-bottom: 3px; }
  .lead-field strong { color: #cbd5e1; }
  .lead-hook { font-size: 11px; color: #818cf8; font-style: italic; margin-top: 6px; }
  .sources-pill { display: inline-block; background: #1d4ed8; color: #bfdbfe;
                  padding: 1px 6px; border-radius: 4px; font-size: 10px; margin: 2px 2px 0 0; }

  /* Approval panel */
  .approval-panel { background: #451a03; border: 1px solid #92400e; border-radius: 10px;
                    padding: 16px; margin-bottom: 20px; }
  .approval-panel h3 { color: #fde68a; font-size: 14px; margin-bottom: 8px; }
  .approval-panel p  { color: #fcd34d; font-size: 12px; margin-bottom: 12px; }
  .approval-btns { display: flex; gap: 8px; }
  .approval-btns .btn { width: auto; flex: 1; }

  /* Call results */
  .call-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px;
               padding: 14px; margin-bottom: 12px; }
  .call-card .call-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .call-card .call-name { font-size: 14px; font-weight: 600; }
  .call-outcome { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .outcome-meeting  { background: #065f46; color: #a7f3d0; }
  .outcome-voicemail { background: #1e3a5f; color: #93c5fd; }
  .outcome-not-interested { background: #7f1d1d; color: #fecaca; }
  .outcome-no-answer { background: #334155; color: #94a3b8; }
  .outcome-callback  { background: #4c1d95; color: #ddd6fe; }
  .transcript { font-size: 11px; color: #94a3b8; font-family: monospace; white-space: pre-wrap;
                background: #0f172a; padding: 10px; border-radius: 6px; margin-top: 8px;
                max-height: 200px; overflow-y: auto; }

  .raw-output-wrap { position: relative; }
  .raw-output { font-size: 11px; color: #94a3b8; font-family: monospace; white-space: pre-wrap;
                background: #0f172a; padding: 10px; border-radius: 6px; max-height: 300px; overflow-y: auto; }
  .copy-btn { position: absolute; top: 8px; right: 8px; padding: 3px 10px;
              background: #334155; color: #94a3b8; border: 1px solid #475569;
              border-radius: 5px; font-size: 11px; cursor: pointer; }
  .copy-btn:hover { background: #475569; color: #e2e8f0; }
  .copy-btn.copied { background: #065f46; color: #a7f3d0; border-color: #10b981; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #334155;
             border-top-color: #6366f1; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
  .tag-step1 { background: #1d4ed8; color: #bfdbfe; }
</style>
</head>
<body>

<header>
  <div>
    <h1>⚡ LeadFlow</h1>
    <div class="subtitle">AI Lead Discovery &amp; Outreach — powered by Continuum</div>
  </div>
</header>

<div class="layout">
  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-section">
      <h2>New Campaign</h2>
      <div class="form-group">
        <label>Niche</label>
        <input id="inp-niche" type="text" placeholder="e.g. coffee shops" value="coffee shops" />
      </div>
      <div class="form-group">
        <label>Location</label>
        <input id="inp-location" type="text" placeholder="e.g. Austin, TX" value="Austin, TX" />
      </div>
      <div class="form-group">
        <label>Model</label>
        <select id="inp-model">
          <option value="openai/gpt-4o" selected>gpt-4o</option>
          <option value="openai/gpt-4o-mini">gpt-4o-mini</option>
          <option value="gemini/gemini-2.5-flash">gemini-2.5-flash</option>
          <option value="anthropic/claude-3-5-haiku-20241022">claude-3-5-haiku</option>
        </select>
      </div>
      <div class="form-group">
        <label>Leads per source</label>
        <select id="inp-leads">
          <option value="3">3</option>
          <option value="5" selected>5</option>
          <option value="8">8</option>
        </select>
      </div>
      <button class="btn btn-primary" id="btn-start" onclick="startCampaign()">
        Discover Leads
      </button>
    </div>

    <h2 style="padding:12px 16px 4px; font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.05em;">Campaigns</h2>
    <div class="campaign-list" id="campaign-list"></div>
  </div>

  <!-- Main -->
  <div class="main" id="main-panel">
    <div class="empty-state">
      <h2>No campaign selected</h2>
      <p>Fill in the form and click <strong>Discover Leads</strong> to start.</p>
    </div>
  </div>
</div>

<script>
const STEPS = [
  { label: "1\nScraping",   key: "scraping"  },
  { label: "2\nScoring",    key: "scoring"   },
  { label: "3\nReview",     key: "review"    },
  { label: "4\nOutreach",   key: "outreach"  },
  { label: "✓\nDone",       key: "done"      },
];

let _campaigns = {};       // id → meta
let _activeCampaign = null;
let _pollTimer = null;

async function startCampaign() {
  const niche    = document.getElementById('inp-niche').value.trim();
  const location = document.getElementById('inp-location').value.trim();
  const model    = document.getElementById('inp-model').value;
  const leads    = parseInt(document.getElementById('inp-leads').value);
  if (!niche || !location) { alert('Niche and location are required.'); return; }

  document.getElementById('btn-start').disabled = true;
  try {
    const res = await fetch('/campaign', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({niche, location, model, leads_per_source: leads}),
    });
    if (!res.ok) {
      const err = await res.json();
      alert('Error: ' + (err.detail || 'unknown'));
      return;
    }
    const data = await res.json();
    _campaigns[data.campaign_id] = {
      niche, location,
      campaign_id: data.campaign_id,
      status: 'running',
      current_step_index: 0,
    };
    renderSidebar();
    selectCampaign(data.campaign_id);
  } finally {
    document.getElementById('btn-start').disabled = false;
  }
}

function selectCampaign(id) {
  _activeCampaign = id;
  renderSidebar();
  renderMain();
  startPolling(id);
}

function startPolling(id) {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(() => pollStatus(id), 2500);
  pollStatus(id);
}

async function pollStatus(id) {
  try {
    const res = await fetch(`/campaign/${id}/status`);
    if (!res.ok) return;
    const data = await res.json();
    _campaigns[id] = {..._campaigns[id], ...data};
    if (_activeCampaign === id) renderMain();
    renderSidebar();
    const s = data.status;
    if (s === 'completed' || s === 'rejected' || s === 'failed' || s === 'timed_out') {
      clearInterval(_pollTimer);
    }
  } catch(e) {}
}

async function approveCampaign() {
  const id = _activeCampaign;
  if (!id) return;
  document.getElementById('btn-approve').disabled = true;
  document.getElementById('btn-reject').disabled = true;
  try {
    const res = await fetch(`/campaign/${id}/approve`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({decided_by: 'user', reason: 'Leads look good'}),
    });
    if (!res.ok) { const e = await res.json(); alert('Error: ' + e.detail); }
    else pollStatus(id);
  } finally {
    document.getElementById('btn-approve').disabled = false;
    document.getElementById('btn-reject').disabled = false;
  }
}

async function rejectCampaign() {
  const id = _activeCampaign;
  if (!id) return;
  if (!confirm('Reject this campaign? Outreach will be cancelled.')) return;
  document.getElementById('btn-approve').disabled = true;
  document.getElementById('btn-reject').disabled = true;
  try {
    const res = await fetch(`/campaign/${id}/reject`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({decided_by: 'user', reason: 'Leads not suitable'}),
    });
    if (!res.ok) { const e = await res.json(); alert('Error: ' + e.detail); }
    else pollStatus(id);
  } finally {
    document.getElementById('btn-approve').disabled = false;
    document.getElementById('btn-reject').disabled = false;
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function statusBadgeClass(status) {
  if (!status || status === 'running') return 'badge-running';
  if (status === 'waiting_for_approval') return 'badge-waiting';
  if (status === 'completed') return 'badge-completed';
  if (status === 'rejected' || status === 'failed') return 'badge-rejected';
  return 'badge-running';
}

function statusLabel(status) {
  const map = {
    running: 'Running',
    waiting_for_approval: 'Needs Review',
    completed: 'Completed',
    rejected: 'Rejected',
    failed: 'Failed',
    timed_out: 'Timed Out',
  };
  return map[status] || status || 'Running';
}

function renderSidebar() {
  const el = document.getElementById('campaign-list');
  const ids = Object.keys(_campaigns).reverse();
  if (!ids.length) { el.innerHTML = '<div style="padding:16px;color:#475569;font-size:13px;">No campaigns yet.</div>'; return; }
  el.innerHTML = ids.map(id => {
    const c = _campaigns[id];
    const active = id === _activeCampaign ? 'active' : '';
    return `<div class="campaign-item ${active}" onclick="selectCampaign('${id}')">
      <div class="c-title">${esc(c.niche)} — ${esc(c.location)}</div>
      <div class="c-meta">${id.slice(0,16)}</div>
      <span class="c-badge ${statusBadgeClass(c.status)}">${statusLabel(c.status)}</span>
    </div>`;
  }).join('');
}

function getStepIndex(c) {
  if (!c) return 0;
  const s = c.status;
  if (s === 'completed') return 4;
  if (s === 'rejected' || s === 'failed') return -1;
  if (s === 'waiting_for_approval') return 2;
  const idx = c.current_step_index || 0;
  if (idx === 0) return 0;  // scraping
  if (idx === 1) return 1;  // scoring
  if (idx === 2) return 2;  // approval
  if (idx === 3) return 3;  // voice
  return 0;
}

function renderSteps(c) {
  const active = getStepIndex(c);
  const isWaiting = c && c.status === 'waiting_for_approval';
  return `<div class="steps">${STEPS.map((s, i) => {
    let cls = '';
    if (i < active) cls = 'done';
    else if (i === active) cls = isWaiting && i === 2 ? 'waiting' : 'active';
    const label = s.label.replace('\n', '<br>');
    return `<div class="step ${cls}"><span class="step-num">${label}</span></div>`;
  }).join('')}</div>`;
}

function scoreClass(score) {
  if (score >= 7) return 'score-high';
  if (score >= 4) return 'score-mid';
  return 'score-low';
}

function renderLeadCards(leads) {
  if (!leads || !leads.leads || !leads.leads.length) return '';
  return `
    <div class="section-title">Ranked Leads (${leads.total || leads.leads.length})</div>
    <div class="leads-grid">
      ${leads.leads.map(l => `
        <div class="lead-card">
          <div class="lead-header">
            <div class="lead-name">${esc(l.name)}</div>
            <div class="lead-rank">#${l.rank} · ${l.score}/10</div>
          </div>
          <div class="score-bar"><div class="score-fill ${scoreClass(l.score)}" style="width:${l.score*10}%"></div></div>
          ${l.address  ? `<div class="lead-field"><strong>📍</strong> ${esc(l.address)}</div>` : ''}
          ${l.phone    ? `<div class="lead-field"><strong>📞</strong> ${esc(l.phone)}</div>` : ''}
          ${l.website  ? `<div class="lead-field"><strong>🌐</strong> ${esc(l.website)}</div>` : ''}
          ${l.description ? `<div class="lead-field">${esc(l.description)}</div>` : ''}
          ${l.outreach_hook ? `<div class="lead-hook">"${esc(l.outreach_hook)}"</div>` : ''}
          <div style="margin-top:6px;">${(l.sources||[]).map(s=>`<span class="sources-pill">${esc(s)}</span>`).join('')}</div>
          ${l.score_reason ? `<div class="lead-field" style="margin-top:6px;font-size:10px;color:#64748b;">${esc(l.score_reason)}</div>` : ''}
        </div>
      `).join('')}
    </div>`;
}

function renderApprovalPanel(c) {
  if (!c || c.status !== 'waiting_for_approval') return '';
  return `
    <div class="approval-panel">
      <h3>⏸ Human Review Gate</h3>
      <p>The campaign has scored and ranked the leads. Review the list below, then approve to begin voice outreach.</p>
      <div class="approval-btns">
        <button id="btn-approve" class="btn btn-approve" onclick="approveCampaign()">✓ Approve & Call</button>
        <button id="btn-reject"  class="btn btn-reject"  onclick="rejectCampaign()">✕ Reject</button>
      </div>
    </div>`;
}

function renderCallResults(c) {
  if (!c || !c.step_results) return '';
  // Voice step is step index 3
  const voiceResults = (c.step_results || []).filter((_, i) => {
    // Parallel step produces 3 results; scoring = 1; voice = last
    return false; // step_results not exposed via query — show raw last_output instead
  });
  return '';
}

function renderMain() {
  const el = document.getElementById('main-panel');
  if (!_activeCampaign) {
    el.innerHTML = '<div class="empty-state"><h2>No campaign selected</h2></div>';
    return;
  }
  const c = _campaigns[_activeCampaign];
  if (!c) return;

  const isTerminal = ['completed','rejected','failed','timed_out'].includes(c.status);
  const isWaiting = c.status === 'waiting_for_approval';

  let html = `
    <div style="margin-bottom:20px;">
      <div style="font-size:20px;font-weight:700;color:#f1f5f9;">${esc(c.niche)} · ${esc(c.location)}</div>
      <div style="font-size:12px;color:#64748b;margin-top:2px;">
        Campaign ID: ${_activeCampaign}
        ${c.started_at ? ' · Started ' + new Date(c.started_at+'Z').toLocaleTimeString() : ''}
      </div>
    </div>
    ${renderSteps(c)}
  `;

  if (!isTerminal && !isWaiting) {
    html += `<div style="color:#94a3b8;font-size:13px;margin-bottom:16px;">
      <span class="spinner"></span> &nbsp;${statusLabel(c.status)}…
    </div>`;
  }

  html += renderApprovalPanel(c);

  // Lead cards (shown during approval or after completion)
  if (c.leads) {
    if (c.leads.leads) {
      html += renderLeadCards(c.leads);
    } else if (c.leads.raw) {
      const rawId = 'raw-' + _activeCampaign;
      html += `<div class="section-title">Scoring Output</div>
        <div class="raw-output-wrap">
          <button class="copy-btn" onclick="copyRaw('${rawId}', this)">Copy</button>
          <div class="raw-output" id="${rawId}">${esc(c.leads.raw)}</div>
        </div>`;
    }
  }

  if (c.status === 'completed') {
    html += `<div style="background:#065f46;border:1px solid #10b981;border-radius:10px;padding:14px;margin-top:16px;">
      <div style="font-size:14px;font-weight:600;color:#a7f3d0;">✓ Campaign Completed</div>
      <div style="font-size:12px;color:#6ee7b7;margin-top:4px;">Voice outreach finished. Check Langfuse for full traces.</div>
    </div>`;
  }

  if (c.status === 'rejected') {
    html += `<div style="background:#7f1d1d;border:1px solid #ef4444;border-radius:10px;padding:14px;margin-top:16px;">
      <div style="font-size:14px;font-weight:600;color:#fecaca;">✕ Campaign Rejected</div>
    </div>`;
  }

  if (c.error) {
    html += `<div style="background:#1e1b4b;border:1px solid #ef4444;border-radius:10px;padding:12px;margin-top:12px;">
      <div style="font-size:12px;color:#ef4444;">Error: ${esc(c.error)}</div>
    </div>`;
  }

  el.innerHTML = html;
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function copyRaw(id, btn) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// Init
renderSidebar();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--temporal-host", default="localhost:7233")
    args = parser.parse_args()

    _app_config = LeadFlowConfig(
        model=args.model,
        temporal_host=args.temporal_host,
    )
    print(f"LeadFlow UI → http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
