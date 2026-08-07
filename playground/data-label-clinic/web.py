#!/usr/bin/env python3
"""
Data-label clinic — web UI + backend.

Usage:
  Terminal 1: python server.py            (clinic MCP tools on :8911)
  Terminal 2: python pharmacy_server.py   (pharmacy MCP tools on :8912)
  Terminal 3: python web.py               (Web UI on :8910)

Two MCP servers, because they overlap on `lookup_patient` -- which is what makes
tool namespacing observable rather than theoretical.

The UI is a glassbox: alongside the chat it shows, per turn, the run's data-label
taint, which model answered (and whether the cloud model was denied), and every
gate decision. Extra buttons demonstrate the telemetry-redaction and
memory-write gates directly.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from contextlib import asynccontextmanager

import uvicorn
from agent import ClinicAgent
from config import PHI, default_config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from continuum import LogLevel, get_logger, setup_logging
from continuum.observability.data_redaction import redact_for_telemetry
from continuum.tools.exceptions import MCPServerUnreviewedError

setup_logging(level=LogLevel.INFO)
logger = get_logger(__name__)

_agent: ClinicAgent | None = None
_init_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _init_error
    _agent = ClinicAgent(config=default_config)
    try:
        await _agent.initialize()
        print(f"✓ Agent ready — {len(_agent.tools)} tools loaded")
    except MCPServerUnreviewedError as e:
        # Handled separately from the catch-all below, which assumes any init
        # failure is the server being down and says so. Here the server is up
        # and answering -- that is how its catalogue got counted -- so that
        # hint would send the reader to check something that is fine, right
        # after a message that already told them exactly what to run.
        # e.message, not str(e): the latter appends "| Context: server_name=..."
        # which is exception plumbing, and the message already names the server.
        _init_error = e.message
        print(f"✗ Agent init failed: {e.message}")
    except Exception as e:
        _init_error = str(e)
        print(f"✗ Agent init failed: {e}")
        print("Are BOTH MCP servers running?  python server.py  /  python pharmacy_server.py")
    yield
    if _agent and _agent._initialized:
        try:
            await _agent.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "u1"
    conversation_id: str = "c1"
    # Output scanner: ON = sanitized, per-turn (no token typing); OFF = live
    # token deltas (visible typing, unredacted). The two are mutually exclusive.
    scanner_on: bool = True


class MemWriteRequest(BaseModel):
    text: str = "Patient P-123 summary: Type 2 diabetes."
    labels: list[str] = [PHI]
    user_id: str = "u1"


class MemDeleteRequest(BaseModel):
    memory_id: str


class MemClearRequest(BaseModel):
    user_id: str = "u1"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _agent or not _agent._initialized:
        return {
            "response": f"Agent unavailable. {_init_error or 'Start both MCP servers: python server.py and python pharmacy_server.py'}",
            "taint": [],
            "model_used": None,
            "gate_events": [],
            "tools_called": [],
        }
    try:
        return await _agent.chat(
            req.message,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            scanner_on=req.scanner_on,
        )
    except Exception as e:
        # Answer in the shape the UI parses. Letting this escape gives FastAPI's
        # plain-text "Internal Server Error", which the browser then feeds to
        # response.json() -- so a run that hit its turn limit is reported as
        # `SyntaxError: Unexpected token 'I'`, and the reader debugs the wrong
        # layer. Seen with MaxTurnsExceededError after the model looped on the
        # one tool it had left.
        logger.exception("chat failed")
        # `failed` so the glassbox can say "unavailable" rather than render the
        # empty lists as clean/none/no-gates. The run died mid-flight; work had
        # already happened (25 send_referral_email calls, in the case that
        # exposed this) and claiming otherwise is the same false all-clear the
        # panels exist to prevent.
        return {
            "response": f"{type(e).__name__}: {e}",
            "failed": True,
            "taint": [],
            "model_used": None,
            "gate_events": [],
            "tools_called": [],
        }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events twin of /chat: the same run, streamed. Each line is
    `data: {json}` — token/message (chat bubble), reroute (cloud→on-prem), and a
    final `done` carrying the full glassbox payload that fills the side panels."""
    if not _agent or not _agent._initialized:

        async def _err():
            payload = {
                "type": "done",
                "response": f"Agent unavailable. {_init_error or 'Start both MCP servers: python server.py and python pharmacy_server.py'}",
                "taint": [],
                "model_used": None,
                "gate_events": [],
                "tools_called": [],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_err(), media_type="text/event-stream")

    async def _gen():
        try:
            async for ev in _agent.chat_stream(
                req.message,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                scanner_on=req.scanner_on,
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            # A raise mid-stream just severs the connection: the browser sees a
            # truncated event-stream and waits for a `done` that never comes.
            # Emit one so the failure reaches the chat bubble.
            logger.exception("chat_stream failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "response": f"{type(e).__name__}: {e}",
                        "failed": True,
                        "taint": [],
                        "model_used": None,
                        "gate_events": [],
                        "tools_called": [],
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/policies")
async def policies():
    if not _agent:
        return {"policies": []}
    return {
        "policies": [
            {
                "name": p.name,
                "subjects": p.subjects,
                "resources": p.resources,
                "effect": p.effect,
            }
            for p in _agent.policy_store.list_policies()
        ]
    }


@app.get("/telemetry/inspect")
async def telemetry_inspect(tainted: bool = True):
    """Show what a span payload becomes after telemetry redaction, with vs
    without the PHI taint — exercising redact_for_telemetry against the policy."""
    if not _agent:
        return {"error": "agent not ready"}
    sample = {
        "patient": "Jane Doe",
        "diagnosis": "Type 2 diabetes",
        "prompt_tokens": 412,
        "total_tokens": 530,
    }
    labels = {PHI} if tainted else set()
    # mask_secrets=False mirrors how SpanScope redacts real span payloads: the
    # label-deny gate still applies, but legitimate fields like prompt_tokens
    # (substring "token") are NOT masked — the token/cost observability guard.
    redacted = redact_for_telemetry(
        sample,
        policy_store=_agent.policy_store,
        subject=default_config.agent_name,
        labels=labels,
        mask_secrets=False,
    )
    return {"tainted": tainted, "raw": sample, "sent_to_telemetry": redacted}


@app.post("/memory/write")
async def memory_write(req: MemWriteRequest):
    """Attempt a long-term memory write. With labels=['phi'] the write-gate
    denies it (nothing persisted); with no labels it is stored under the user."""
    if not _agent:
        return {"ok": False, "reason": "agent not ready"}
    return await _agent.attempt_memory_write(req.text, req.labels, user_id=req.user_id)


@app.get("/memory/list")
async def memory_list(user_id: str = "u1"):
    """List what is actually persisted for this user — proves PHI never landed
    while ordinary notes did."""
    client = _agent.memory_client() if _agent else None
    if client is None:
        return {"ok": False, "skipped": True, "reason": "memory not enabled", "memories": []}
    try:
        entries = await client.get_all(user_id=user_id)
        return {"ok": True, "memories": [{"id": e.id, "text": e.memory} for e in entries]}
    except Exception as e:
        return {"ok": False, "error": str(e), "memories": []}


@app.post("/memory/delete")
async def memory_delete(req: MemDeleteRequest):
    client = _agent.memory_client() if _agent else None
    if client is None:
        return {"ok": False, "reason": "memory not enabled"}
    try:
        await client.delete(req.memory_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/memory/clear")
async def memory_clear(req: MemClearRequest):
    client = _agent.memory_client() if _agent else None
    if client is None:
        return {"ok": False, "reason": "memory not enabled"}
    try:
        await client.delete_all(user_id=req.user_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/status")
async def status():
    return {
        "ready": bool(_agent and _agent._initialized),
        "error": _init_error,
        "tools": [t.function.name for t in (_agent.tools if _agent else [])],
        "cloud_model": default_config.cloud_model,
        "onprem_model": default_config.onprem_model,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data-Label Clinic — glassbox</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1419; color: #e6e6e6; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a2332; padding: 14px 22px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #2a3548; }
  header h1 { font-size: 16px; font-weight: 600; flex: 1; }
  header .pill { font-size: 12px; padding: 3px 9px; border-radius: 10px; background: #243044; color: #9fb3c8; }
  #wrap { flex: 1; display: flex; overflow: hidden; }
  #left { flex: 1.3; display: flex; flex-direction: column; border-right: 1px solid #2a3548; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 10px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #2563eb; color: white; }
  .assistant { align-self: flex-start; background: #1e293b; }
  .thinking { align-self: flex-start; color: #7c8aa0; font-style: italic; font-size: 13px; }
  #input-row { padding: 14px 18px; background: #141b26; border-top: 1px solid #2a3548; display: flex; gap: 8px; align-items: center; }
  #stream-label, #scanner-label { font-size: 12px; color: #9fb3c8; display: flex; align-items: center; gap: 4px; cursor: pointer; white-space: nowrap; }
  #input { flex: 1; padding: 10px 12px; background: #0f1419; border: 1px solid #2a3548; color: #e6e6e6; border-radius: 8px; font-size: 14px; outline: none; }
  #send { padding: 10px 18px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; }
  #send:disabled { background: #555; cursor: not-allowed; }
  .suggestions { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 18px 12px; }
  .suggestions button { font-size: 12px; padding: 6px 10px; background: #1e293b; color: #9fb3c8; border: 1px solid #2a3548; border-radius: 14px; cursor: pointer; }
  #panel { flex: 1; overflow-y: auto; padding: 18px; background: #0c1118; }
  .card { background: #141b26; border: 1px solid #2a3548; border-radius: 10px; padding: 14px; margin-bottom: 14px; }
  .card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #7c8aa0; margin-bottom: 10px; }
  .chip { display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 12px; margin: 2px; }
  .chip.phi { background: #7f1d1d; color: #fecaca; }
  .chip.clean { background: #14532d; color: #bbf7d0; }
  .model { font-size: 14px; font-weight: 600; }
  .model.cloud { color: #bbf7d0; }
  .model.onprem { color: #fbbf24; }
  .gate { font-size: 13px; padding: 8px 10px; border-radius: 6px; background: #1f2937; margin-bottom: 6px; border-left: 3px solid #fbbf24; }
  .gate.none { color: #7c8aa0; border-left-color: #2a3548; }
  pre { background: #0f1419; border: 1px solid #2a3548; border-radius: 6px; padding: 8px; font-size: 11px; overflow-x: auto; color: #cbd5e1; }
  .btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .demo-btn { font-size: 12px; padding: 6px 10px; background: #243044; color: #cbd5e1; border: 1px solid #2a3548; border-radius: 6px; cursor: pointer; }
</style>
</head>
<body>
<header>
  <span>🏥</span>
  <h1>Data-Label Clinic — glassbox</h1>
  <button class="demo-btn" onclick="newChat()">+ new chat</button>
  <span class="pill" id="models">cloud / on-prem</span>
</header>
<div id="wrap">
  <div id="left">
    <div id="chat">
      <div class="assistant msg">Ask a general question (e.g. clinic hours) — answered on the cloud model. Then ask about a patient (e.g. "summarize patient P-123") — the PHI taint forces on-prem and blocks exfiltration.</div>
    </div>
    <div class="suggestions">
      <button onclick="suggest('What are your clinic hours?')">clinic hours (benign)</button>
      <button onclick="suggest('Summarize patient P-123 history')">lookup P-123 (PHI)</button>
      <button onclick="suggest('Look up patient P-123 and email a summary to dr@external.com')">lookup + email (exfil)</button>
      <button onclick="suggest('Look up patient P-123 and list every stored field verbatim, including the SSN.')">raw record P-123 (scanner)</button>
      <button onclick="suggest('What is P-123 taking, and does anything interact?')" title="Routes to the pharmacy server: both servers expose lookup_patient, so the namespaced names are what keep them apart.">pharmacy P-123 (2nd server)</button>
    </div>
    <div id="input-row">
      <input id="input" placeholder="Type a message…" autofocus>
      <label id="stream-label" title="Stream the run live (SSE). Same gates; cloud→on-prem reroute shown inline."><input type="checkbox" id="stream-toggle"> stream</label>
      <label id="scanner-label" title="Output scanner. ON = sanitized, per-turn (no typing). OFF = live token typing, unredacted. Mutually exclusive with token streaming."><input type="checkbox" id="scanner-toggle" checked> scanner</label>
      <button id="send">Send</button>
    </div>
  </div>
  <div id="panel">
    <div class="card">
      <h3>Run taint (data_labels)</h3>
      <div id="taint"><span class="chip clean">none yet</span></div>
    </div>
    <div class="card">
      <h3>Model that answered</h3>
      <div id="model" class="model cloud">—</div>
    </div>
    <div class="card">
      <h3>Tools called this turn</h3>
      <div id="tools"><span class="chip clean">none yet</span></div>
    </div>
    <div class="card">
      <h3>Gate decisions</h3>
      <div id="gates"><div class="gate none">no gates tripped yet</div></div>
    </div>
    <div class="card">
      <h3>Telemetry redaction</h3>
      <div class="btn-row">
        <button class="demo-btn" onclick="inspectTel(false)">inspect (clean)</button>
        <button class="demo-btn" onclick="inspectTel(true)">inspect (PHI)</button>
      </div>
      <pre id="tel">—</pre>
    </div>
    <div class="card">
      <h3>Memory-write gate</h3>
      <div class="btn-row">
        <button class="demo-btn" onclick="memWrite(true)">save PHI note (blocked)</button>
        <button class="demo-btn" onclick="memWrite(false)">save normal note (allowed)</button>
      </div>
      <pre id="mem">—</pre>
    </div>
    <div class="card">
      <h3>Long-term memory (user u1)</h3>
      <div class="btn-row">
        <button class="demo-btn" onclick="listMem()">refresh</button>
        <button class="demo-btn" onclick="clearMem()">clear all</button>
      </div>
      <div id="memlist"><span class="chip clean">—</span></div>
    </div>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');

// Short-term memory is scoped per (user_id, conversation_id) — same pattern as
// gateway-local-shop. A stable user, a fresh conversation id per chat window.
const USER_ID = 'u1';
function uuid(){
  return (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2,9);
}
let currentConversationId = uuid();

fetch('/status').then(r=>r.json()).then(s=>{
  document.getElementById('models').textContent = `cloud: ${s.cloud_model}  •  on-prem: ${s.onprem_model}`;
});

function scannerOn(){ return document.getElementById('scanner-toggle').checked; }
function add(cls, text){ const d=document.createElement('div'); d.className=cls+' msg'; d.textContent=text; chat.appendChild(d); chat.scrollTop=chat.scrollHeight; return d; }
function suggest(t){ input.value=t; sendMsg(); }

function renderTaint(taint){
  const el=document.getElementById('taint');
  if(!taint || !taint.length){ el.innerHTML='<span class="chip clean">clean</span>'; return; }
  el.innerHTML = taint.map(t=>`<span class="chip ${t==='phi'?'phi':'clean'}">${t}</span>`).join('');
}
function renderModel(m){
  const el=document.getElementById('model');
  el.textContent=m||'—';
  el.className='model '+(m && m.includes('mini')?'onprem':'cloud');
}
function renderUnknown(){
  // A failed run tells us nothing about what did or did not happen before it
  // died. "clean"/"none"/"no gates tripped" would be an assertion we cannot
  // make -- and the panels are here precisely so nobody has to guess.
  document.getElementById('taint').innerHTML='<span class="chip">unavailable — run failed</span>';
  document.getElementById('tools').innerHTML='<span class="chip">unavailable — run failed</span>';
  document.getElementById('gates').innerHTML='<div class="gate">unavailable — run failed</div>';
  const m=document.getElementById('model'); m.textContent='—'; m.className='model';
}
function renderTools(tools){
  const el=document.getElementById('tools');
  if(!tools || !tools.length){ el.innerHTML='<span class="chip clean">none</span>'; return; }
  // Match the trailing segment, not the whole string. This list holds the
  // LLM-facing names, which are namespaced (clinic__send_referral_email), so
  // comparing against the bare name matched nothing and every tool -- including
  // the exfiltration attempt -- rendered green. The panel's job is to show what
  // happened; a green chip over a blocked exfil call is the one output worse
  // than no panel. Splitting on '__' keeps it right whether or not
  // namespace_tools is on.
  const egress = ['send_referral_email', 'web_lookup'];
  el.innerHTML = tools.map(t=>`<span class="chip ${egress.includes(t.split('__').pop())?'phi':'clean'}">${t}</span>`).join('');
}
function renderGates(events){
  const el=document.getElementById('gates');
  if(!events || !events.length){ el.innerHTML='<div class="gate none">no gates tripped</div>'; return; }
  el.innerHTML = events.map(e=>`<div class="gate">${e}</div>`).join('');
}

async function sendMsg(){
  const text=input.value.trim(); if(!text) return;
  if(document.getElementById('stream-toggle').checked){ return sendMsgStream(text); }
  add('user', text); input.value=''; send.disabled=true;
  const thinking=add('thinking','…');
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,user_id:USER_ID,conversation_id:currentConversationId,scanner_on:scannerOn()})});
    const d=await r.json();
    thinking.remove();
    add('assistant', d.response);
    if(d.failed){ renderUnknown(); } else { renderTaint(d.taint); renderModel(d.model_used); renderTools(d.tools_called); renderGates(d.gate_events); }
    listMem();      // long-term memory may have changed (stored, or blocked)
  }catch(e){ thinking.textContent='Error: '+e; }
  send.disabled=false; input.focus();
}

async function sendMsgStream(text){
  // SSE twin of sendMsg: the same run, streamed. token/message fill the chat
  // bubble live; reroute clears it and notes the cloud→on-prem switch; the
  // final `done` event fills the side panels exactly like the non-streaming path.
  add('user', text); input.value=''; send.disabled=true;
  let bubble=add('assistant',''); let acc='';
  try{
    const r=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,user_id:USER_ID,conversation_id:currentConversationId,scanner_on:scannerOn()})});
    const reader=r.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){
      const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\\n\\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+2);
        if(!line.startsWith('data:')) continue;
        const ev=JSON.parse(line.slice(5).trim());
        if(ev.type==='token'){ acc+=ev.text; bubble.textContent=acc; }
        else if(ev.type==='message'){ acc=ev.text; bubble.textContent=acc; }
        else if(ev.type==='tool'){ bubble.textContent=acc||('🔧 calling '+ev.name+'…'); }
        else if(ev.type==='reroute'){ acc=''; bubble.remove(); add('thinking', ev.text); bubble=add('assistant',''); }
        else if(ev.type==='done'){
          bubble.textContent=ev.response||acc||'(no content)';
          if(ev.failed){ renderUnknown(); } else { renderTaint(ev.taint); renderModel(ev.model_used); renderTools(ev.tools_called); renderGates(ev.gate_events); }
          listMem();
        }
        chat.scrollTop=chat.scrollHeight;
      }
    }
  }catch(e){ bubble.textContent='Error: '+e; }
  send.disabled=false; input.focus();
}
send.onclick=sendMsg;
input.addEventListener('keydown',e=>{ if(e.key==='Enter') sendMsg(); });

async function inspectTel(tainted){
  const r=await fetch('/telemetry/inspect?tainted='+tainted); const d=await r.json();
  document.getElementById('tel').textContent =
    'raw:\\n'+JSON.stringify(d.raw,null,1)+'\\n\\nsent_to_telemetry:\\n'+JSON.stringify(d.sent_to_telemetry,null,1);
}
function newChat(){
  // Fresh conversation_id → new short-term (session) scope, gateway-local-shop style.
  currentConversationId = uuid();
  chat.innerHTML = '<div class="assistant msg">New conversation started.</div>';
  input.focus();
}
async function memWrite(phi){
  const body = phi
    ? {text:'Patient P-123, Jane Doe: Type 2 diabetes; SSN 123-45-6789', labels:['phi'], user_id:'u1'}
    : {text:'I prefer morning appointments and email reminders.', labels:[], user_id:'u1'};
  const r=await fetch('/memory/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const d=await r.json();
  document.getElementById('mem').textContent = JSON.stringify(d,null,1);
  listMem();
}
async function listMem(){
  const r=await fetch('/memory/list?user_id=u1'); const d=await r.json();
  const el=document.getElementById('memlist');
  if(d.skipped){ el.innerHTML='<span class="chip clean">memory not enabled</span>'; return; }
  if(d.ok===false){ el.innerHTML='<span class="chip phi">error: '+(d.error||'unknown')+'</span>'; return; }
  if(!d.memories || !d.memories.length){ el.innerHTML='<span class="chip clean">empty</span>'; return; }
  el.innerHTML = d.memories.map(m=>
    `<div class="gate"><span>${m.text}</span> <button class="demo-btn" onclick="delMem('${m.id}')">delete</button></div>`
  ).join('');
}
async function delMem(id){
  await fetch('/memory/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({memory_id:id})});
  listMem();
}
async function clearMem(){
  await fetch('/memory/clear',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({user_id:'u1'})});
  listMem();
}
listMem();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Data-label clinic web UI:  http://localhost:8910")
    uvicorn.run(app, host="0.0.0.0", port=8910)
