#!/usr/bin/env python3
"""
Incident Desk — web UI + backend (the interactive glassbox).

Usage:
  Terminal 0: sidecar     (cd extensions/headroom && HEADROOM_CCR_BACKEND=memory
                           HEADROOM_OFFLINE=1 HF_HUB_OFFLINE=1
                           uv run headroom proxy --port 8787)
  Terminal 1: python server.py   (MCP tools on :8921)
  Terminal 2: python web.py      (Web UI on :8920)

Alongside the chat, the UI shows per turn what Headroom did: tokens
before/after, % saved, the transforms applied, CCR hashes issued, and every
continuum_headroom_retrieve the model made. Extra controls kill/restore the sidecar
binding (fail-open, scenario 5) and fire a forged retrieve (scenario 6).
"""

import json

import config as rig_config  # noqa: F401 — MUST be first (env guard + HEADROOM_ENABLED)
import uvicorn
from agent import IncidentAgent
from config import sidecar_health
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from continuum import LogLevel, setup_logging

setup_logging(level=LogLevel.INFO)

# Show Headroom request/response payloads in the terminal. TWO gates must open:
#   1. the client LOGGER must allow DEBUG (setLevel below), and
#   2. the console HANDLER must allow DEBUG — setup_logging() created it at INFO,
#      so a DEBUG record reaches the handler and is dropped. We lower the handler
#      too. Because only this one logger is at DEBUG (everything else inherits
#      INFO from the `continuum` root), only the headroom client's request/
#      response payloads print at DEBUG — the rest of the app stays at INFO.
import logging as _logging

_logging.getLogger("continuum.llm.headroom.client").setLevel(_logging.DEBUG)
for _h in _logging.getLogger("continuum").handlers:
    _h.setLevel(_logging.DEBUG)

_agent: IncidentAgent | None = None
_init_error: str | None = None
_sidecar_dead = False


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _init_error
    _agent = IncidentAgent()
    try:
        await _agent.initialize()
        print(f"✓ Agent ready — {len(_agent.tools)} tools loaded")
    except Exception as e:
        _init_error = str(e)
        print(f"✗ Agent init failed: {e}\nIs the MCP server running?  python server.py")
    yield
    if _agent and _agent._initialized:
        try:
            await _agent.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


class ForgeRequest(BaseModel):
    hash: str = "f" * 24


class SidecarToggle(BaseModel):
    dead: bool


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/sidecar")
async def sidecar():
    h = sidecar_health()
    h["pointed_dead"] = _sidecar_dead
    return h


@app.post("/sidecar-toggle")
async def sidecar_toggle(req: SidecarToggle):
    global _sidecar_dead
    if not _agent:
        return {"error": "agent not ready"}
    _sidecar_dead = req.dead
    base = _agent.point_sidecar("http://127.0.0.1:9" if req.dead else None)
    return {
        "pointed_at": base,
        "dead": req.dead,
        "note": "compressor rebuilt — previously issued CCR hashes were forgotten",
    }


@app.post("/forge")
async def forge(req: ForgeRequest):
    if not _agent:
        return {"error": "agent not ready"}
    return {"result": await _agent.forge_retrieve(req.hash)}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _agent or not _agent._initialized:
        return {
            "response": f"Agent not connected. {_init_error or 'Start server.py first.'}",
            "tools_called": [],
            "retrieve_calls": [],
            "headroom": {},
        }
    return await _agent.chat(req.message)


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        if not _agent or not _agent._initialized:
            yield (
                json.dumps(
                    {
                        "type": "done",
                        "response": "Agent not connected.",
                        "tools_called": [],
                        "retrieve_calls": [],
                        "headroom": {},
                    }
                )
                + "\n"
            )
            return
        async for ev in _agent.chat_stream(req.message):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat-rag")
async def chat_rag():
    """Scenario 10: inject the runbook text through Continuum's NATIVE
    rag_context slot (a position-7 SYSTEM message) instead of a tool result.
    Headroom protects system messages, so the glassbox should show this payload
    passing through UNCOMPRESSED — unlike scenario 3, where the same bytes crush
    ~98% as a tool result."""
    if not _agent or not _agent._initialized:
        return {
            "response": f"Agent not connected. {_init_error or 'Start server.py first.'}",
            "tools_called": [],
            "retrieve_calls": [],
            "headroom": {},
        }
    from data import format_runbook_results

    payload = format_runbook_results("database connection pool exhaustion")
    return await _agent.chat_with_rag_context(
        "Using ONLY the provided context, which runbook covers database "
        "connection pool exhaustion? Give its ID.",
        rag_context=payload,
    )


@app.get("/messages-dump")
async def messages_dump():
    """Raw before/after message payloads from the last Headroom compression.
    Use: curl localhost:8920/messages-dump | python -m json.tool"""
    from continuum.config import settings

    # Disabled → don't construct the compressor just to read stats.
    if not settings.headroom_enabled:
        return {"enabled": False}

    from continuum.llm.headroom.compressor import get_headroom_compressor

    comp = get_headroom_compressor()
    return {
        "messages_before": comp.last_messages_before,
        "messages_after": comp.last_messages_after,
        "stats": {
            "tokens_before": comp.last_stats.tokens_before if comp.last_stats else None,
            "tokens_after": comp.last_stats.tokens_after if comp.last_stats else None,
            "transforms": list(comp.last_stats.transforms_applied) if comp.last_stats else [],
        },
    }


SCENARIOS = [
    (
        "1 · DB rows (SmartCrusher)",
        "Query the orders database for failed orders. How many orders failed, and "
        "which single order has the largest refund? Give its order id and the exact refund amount.",
    ),
    (
        "2 · Logs + CCR needle",
        "Fetch the checkout-api logs and tell me the exact incident reference token "
        "recorded in the audit trail.",
    ),
    (
        "3 · Runbook search (RAG)",
        "Search the runbooks for guidance on database connection pool exhaustion. "
        "Which runbook applies? Give its ID.",
    ),
    (
        "4 · read tool (excluded)",
        "Use the read tool to read service.yaml, then report the exact values of "
        "database.pool_max_size and workers.refund_worker_concurrency.",
    ),
    (
        "11 · read→edit (file STALE)",
        "Use the read tool to read the file at path 'service.yaml'. The DB "
        "connection pool is undersized for this incident — raise "
        "database.pool_max_size to 50 and use the write tool to save the updated "
        "config back to path 'service.yaml'. Then confirm the new pool size.",
    ),
    (
        "8 · Two needles, one run",
        "Fetch the logs for BOTH checkout-api and payments-svc. Then report the exact "
        "incident reference token recorded in each service's audit trail.",
    ),
    (
        "9 · Postmortem prose (Kompress)",
        "Fetch the postmortem for INC-2417 and state the exact total number of checkout "
        "attempts that failed.",
    ),
]

_SCENARIO_BTNS = "".join(
    f'<button class="scn" data-msg="{msg}">{label}</button>' for label, msg in SCENARIOS
)

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Incident Desk — Headroom glassbox</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
          --dim:#8b949e; --accent:#58a6ff; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,'Segoe UI',sans-serif; }
  header { padding:12px 20px; border-bottom:1px solid var(--border);
           display:flex; justify-content:space-between; align-items:center; }
  header h1 { font-size:16px; margin:0; }
  #badge { font-size:12px; padding:3px 10px; border-radius:12px; border:1px solid var(--border); }
  #badge.up { color:var(--good); } #badge.down { color:var(--bad); }
  main { display:grid; grid-template-columns: 1fr 540px; gap:0;
         height:calc(100vh - 49px); }
  #left { display:flex; flex-direction:column; border-right:1px solid var(--border); }
  #chat { flex:1; overflow-y:auto; padding:16px 20px; }
  .msg { margin:0 0 12px; max-width:85%; padding:10px 14px; border-radius:10px;
         white-space:pre-wrap; word-break:break-word; }
  .user { background:#1f6feb33; margin-left:auto; }
  .bot  { background:var(--panel); border:1px solid var(--border); }
  .sys  { color:var(--dim); font-size:12px; text-align:center; max-width:100%; }
  #controls { padding:10px 20px; border-top:1px solid var(--border); }
  #scenarios { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
  button { background:var(--panel); color:var(--fg); border:1px solid var(--border);
           border-radius:6px; padding:5px 10px; cursor:pointer; font-size:12px; }
  button:hover { border-color:var(--accent); }
  button.danger { color:var(--bad); }
  #inputrow { display:flex; gap:8px; }
  #inp { flex:1; background:var(--panel); border:1px solid var(--border); color:var(--fg);
         border-radius:8px; padding:9px 12px; font-size:14px; }
  label.tgl { font-size:12px; color:var(--dim); display:flex; align-items:center; gap:4px; }
  #right { overflow-y:auto; padding:14px 16px; background:#010409; }
  #right h2 { font-size:13px; color:var(--dim); text-transform:uppercase;
              letter-spacing:.08em; margin:0 0 10px; }
  .turn { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:10px 12px; margin-bottom:10px; font-size:12px; }
  .turn .big { font-size:18px; font-weight:600; }
  .turn .big.good { color:var(--good); } .turn .big.zero { color:var(--warn); }
  .kv { color:var(--dim); } .kv b { color:var(--fg); font-weight:500; }
  .chip { display:inline-block; background:#0d1117; border:1px solid var(--border);
          border-radius:10px; padding:1px 8px; margin:2px 3px 0 0; font-size:11px;
          color:var(--dim); }
  .chip.retrieve { color:var(--accent); border-color:var(--accent); }
  .chip.excluded { color:var(--warn); }
  .hash { font-family:ui-monospace,monospace; font-size:11px; color:var(--accent); }
  /* message dump panels */
  .msg-dump { margin-top:8px; }
  .msg-dump summary { cursor:pointer; color:var(--accent); font-size:11px;
    font-weight:600; text-transform:uppercase; letter-spacing:.04em;
    padding:4px 0; user-select:none; }
  .msg-dump summary:hover { text-decoration:underline; }
  .msg-dump-list { max-height:320px; overflow-y:auto; margin:4px 0 0;
    padding:0; list-style:none; }
  .msg-dump-item { background:#0d1117; border:1px solid var(--border); border-radius:6px;
    padding:6px 8px; margin-bottom:4px; font-size:11px; font-family:ui-monospace,monospace; }
  .msg-dump-item .role { font-weight:700; text-transform:uppercase; font-size:10px; }
  .msg-dump-item .role-system { color:var(--warn); }
  .msg-dump-item .role-user { color:var(--accent); }
  .msg-dump-item .role-assistant { color:var(--good); }
  .msg-dump-item .role-tool { color:#bc8cff; }
  .msg-dump-item .content-preview { color:var(--dim); white-space:pre-wrap;
    word-break:break-all; margin-top:3px; max-height:120px; overflow-y:auto; }
</style></head>
<body>
<header>
  <h1>🗜️ Incident Desk — Headroom glassbox</h1>
  <span id="badge">sidecar: …</span>
</header>
<main>
  <div id="left">
    <div id="chat"><div class="msg sys">Pick a scenario or ask about the checkout-api incident.</div></div>
    <div id="controls">
      <div id="scenarios">__SCENARIO_BTNS__
        <button id="ragBtn">🧩 10 · RAG-context (pos 7, protected)</button>
        <button class="danger" id="killBtn">☠ kill sidecar (fail-open)</button>
        <button id="forgeBtn">🔐 forge a hash</button>
      </div>
      <div id="inputrow">
        <input id="inp" placeholder="Ask Incident Desk…" />
        <label class="tgl"><input type="checkbox" id="streamTgl" checked> stream</label>
        <button id="sendBtn">Send</button>
      </div>
    </div>
  </div>
  <div id="right"><h2>What Headroom did</h2><div id="turns"></div></div>
</main>
<script>
const chat = document.getElementById('chat'), turns = document.getElementById('turns');
const inp = document.getElementById('inp');
let dead = false;

function add(cls, text) {
  const d = document.createElement('div'); d.className = 'msg ' + cls;
  d.textContent = text; chat.appendChild(d); chat.scrollTop = chat.scrollHeight; return d;
}
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

function renderMsgDump(label, msgs) {
  if (!msgs || !msgs.length) return '';
  let items = msgs.map(m => {
    const role = esc(m.role || '?');
    const roleCls = 'role-' + (m.role || 'unknown');
    let detail = '';
    if (m.content !== undefined && m.content !== null) {
      detail += `<span class="kv">${(m.content_len||0).toLocaleString()} chars</span>`;
      if (m.tool_calls) detail += ` · <span class="kv">${m.tool_calls} tool_call(s)</span>`;
      if (m.tool_call_id) detail += ` · <span class="kv">id=${esc(m.tool_call_id)}</span>`;
      detail += `<div class="content-preview">${esc(m.content)}</div>`;
    } else {
      if (m.tool_calls) detail += `<span class="kv">${m.tool_calls} tool_call(s)</span>`;
      if (m.tool_call_id) detail += `<span class="kv">id=${esc(m.tool_call_id)}</span>`;
    }
    return `<li class="msg-dump-item"><span class="role ${roleCls}">${role}</span> ${detail}</li>`;
  }).join('');
  return `<details class="msg-dump"><summary>${esc(label)} (${msgs.length} messages)</summary><ul class="msg-dump-list">${items}</ul></details>`;
}

function renderTurn(box, retrieves, tools) {
  const lc = box.last_call;
  const pct = lc ? lc.pct_saved : null;
  const el = document.createElement('div'); el.className = 'turn';
  let h = '';
  if (lc) {
    h += `<div class="big ${pct > 0 ? 'good' : 'zero'}">${pct}% saved</div>`;
    h += `<div class="kv"><b>${lc.tokens_before.toLocaleString()}</b> → <b>${lc.tokens_after.toLocaleString()}</b> tokens (last LLM call)</div>`;
    h += (lc.transforms||[]).map(t =>
      `<span class="chip ${t.includes('exclude')?'excluded':''}">${esc(t)}</span>`).join('');
  } else {
    h += `<div class="big zero">no compression</div><div class="kv">sidecar unreachable → fail-open (uncompressed)</div>`;
  }
  if ((box.new_hashes||[]).length)
    h += `<div style="margin-top:6px" class="kv">CCR issued: ` +
         box.new_hashes.map(x=>`<span class="hash">${x}</span>`).join(' ') + `</div>`;
  (retrieves||[]).forEach(r => {
    h += `<div style="margin-top:4px"><span class="chip retrieve">continuum_headroom_retrieve</span>` +
         `<span class="kv"> <span class="hash">${r.hash}</span> → <b>${r.chars.toLocaleString()}</b> chars restored</b></span></div>`;
  });
  if ((tools||[]).length)
    h += `<div style="margin-top:6px" class="kv">tools: ${tools.map(esc).join(', ')}</div>`;
  const d = box.sidecar_delta || {};
  h += `<div style="margin-top:6px" class="kv">run total: <b>${(d.tokens_removed||0).toLocaleString()}</b> tokens removed across <b>${d.llm_calls_compressed||0}</b> compressed calls</div>`;
  // Message dump panels
  h += renderMsgDump('Messages BEFORE compression', box.messages_before);
  h += renderMsgDump('Messages AFTER compression', box.messages_after);
  el.innerHTML = h; turns.prepend(el);
}

async function send(msg) {
  if (!msg.trim()) return;
  add('user', msg); inp.value = '';
  const busy = add('sys', '…thinking');
  try {
    if (document.getElementById('streamTgl').checked) {
      const resp = await fetch('/chat-stream', {method:'POST',
        headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
      const reader = resp.body.getReader(); const dec = new TextDecoder();
      let buf = '', bot = null, content = '';
      while (true) {
        const {done, value} = await reader.read(); if (done) break;
        buf += dec.decode(value, {stream:true});
        let nl; while ((nl = buf.indexOf('\\n')) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl+1);
          if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.type === 'token') { if (!bot) bot = add('bot',''); content += ev.text; bot.textContent = content; }
          else if (ev.type === 'message') { if (!bot) bot = add('bot','');
            content = ev.text; bot.textContent = content; }
          else if (ev.type === 'tool') add('sys', '🔧 ' + ev.name);
          else if (ev.type === 'done') {
            if (!bot) add('bot', ev.response);
            renderTurn(ev.headroom || {}, ev.retrieve_calls, ev.tools_called);
          }
          chat.scrollTop = chat.scrollHeight;
        }
      }
    } else {
      const r = await fetch('/chat', {method:'POST',
        headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
      const j = await r.json();
      add('bot', j.response);
      renderTurn(j.headroom || {}, j.retrieve_calls, j.tools_called);
    }
  } catch (e) { add('sys', '⚠️ ' + e); }
  busy.remove();
}

document.getElementById('sendBtn').onclick = () => send(inp.value);
inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(inp.value); });
document.querySelectorAll('.scn').forEach(b => b.onclick = () => send(b.dataset.msg));

document.getElementById('killBtn').onclick = async () => {
  dead = !dead;
  const r = await fetch('/sidecar-toggle', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({dead})});
  const j = await r.json();
  document.getElementById('killBtn').textContent = dead ? '💚 restore sidecar' : '☠ kill sidecar (fail-open)';
  add('sys', (dead ? '☠ SDK now points at a dead port — runs continue UNCOMPRESSED (fail-open). '
                   : '💚 sidecar binding restored. ') + (j.note || ''));
  refreshBadge();
};
document.getElementById('ragBtn').onclick = async () => {
  add('user', '[scenario 10] inject the runbook text via Continuum rag_context (position-7 system message) instead of a tool result');
  const busy = add('sys', '…thinking');
  try {
    const r = await fetch('/chat-rag', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'});
    const j = await r.json();
    add('bot', j.response);
    add('sys', 'Compare the AFTER dump: the PROVIDED CONTEXT system block is byte-identical to BEFORE → Headroom protects system messages (router:protected:system_message). The same bytes crush ~98% as a tool result in scenario 3.');
    renderTurn(j.headroom || {}, j.retrieve_calls, j.tools_called);
  } catch(e){ add('sys','⚠️ '+e); }
  busy.remove();
};
document.getElementById('forgeBtn').onclick = async () => {
  const r = await fetch('/forge', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({})});
  const j = await r.json();
  add('sys', '🔐 forged hash ffff… → ' + j.result);
};

async function refreshBadge() {
  const b = document.getElementById('badge');
  try {
    const j = await (await fetch('/sidecar')).json();
    if (j.pointed_dead) { b.className='down'; b.textContent = 'sidecar: BYPASSED (fail-open demo)'; }
    else if (j.up) { b.className='up'; b.textContent = `sidecar: healthy v${j.version}`; }
    else { b.className='down'; b.textContent = 'sidecar: DOWN (fail-open)'; }
  } catch { b.className='down'; b.textContent = 'sidecar: ?'; }
}
refreshBadge(); setInterval(refreshBadge, 10000);
</script>
</body></html>"""

HTML_PAGE = HTML_PAGE.replace("__SCENARIO_BTNS__", _SCENARIO_BTNS)


if __name__ == "__main__":
    h = sidecar_health()
    if not h["up"]:
        print(f"⚠️  Headroom sidecar DOWN — UI will demo fail-open. Restart: {h['restart']}")
    print("Incident Desk UI on http://localhost:8920")
    uvicorn.run(app, host="0.0.0.0", port=8920, log_level="warning")
