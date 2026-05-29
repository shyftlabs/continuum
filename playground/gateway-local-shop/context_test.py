#!/usr/bin/env python3
"""
Context Management Test — Local Shop.

Demonstrates ContextManagementConfig with a low compression_threshold
so compression fires quickly during testing.

Usage:
  Terminal 1: python server.py          (MCP server on :8888)
  Terminal 2: python context_test.py    (Web UI on :8081)
"""

import dataclasses
import logging
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import uvicorn
from agent import LocalShopAgent
from config import default_config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from orchestrator import (
    CompressionStrategy,
    ContextManagementConfig,
    LogLevel,
    setup_logging,
)

setup_logging(level=LogLevel.INFO)

# ── Capture compression log events ─────────────────────────────────────────────

_compression_events: deque = deque(maxlen=100)


class _CompressionCapture(logging.Handler):
    """Intercepts context_management log lines and stores them for the UI."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "compress" in msg.lower() or "threshold" in msg.lower():
            _compression_events.append({
                "ts": time.strftime("%H:%M:%S"),
                "level": record.levelname,
                "msg": msg,
            })


logging.getLogger("orchestrator.llm.context_management").addHandler(_CompressionCapture())

# ── Context management config ───────────────────────────────────────────────────

_CONTEXT_CFG = ContextManagementConfig(
    compression_strategy=CompressionStrategy.SMART,
    compression_threshold=0.001,
    keep_recent_messages=1,
)


# ── Agent: subclass LocalShopAgent, inject context_management ──────────────────

class ContextTestAgent(LocalShopAgent):
    """LocalShopAgent with ContextManagementConfig injected into AgentConfig."""

    def _create_agent(self) -> None:
        super()._create_agent()
        self._agent.config = dataclasses.replace(self._agent.config, context_management=_CONTEXT_CFG)


# ── FastAPI ─────────────────────────────────────────────────────────────────────

_agent: ContextTestAgent | None = None
_init_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _init_error
    _agent = ContextTestAgent(config=default_config)
    try:
        await _agent.initialize()
        print(f"✓ Agent ready — {len(_agent.tools)} tools loaded")
        print(f"  Context: threshold=1%, strategy=SMART, keep_recent=1")
    except Exception as e:
        _init_error = str(e)
        print(f"✗ Agent init failed: {e}")
        print("Is the MCP server running?  python server.py")
    yield
    if _agent and _agent._initialized:
        try:
            await _agent.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _agent or not _agent._initialized:
        msg = f"Agent not connected. {_init_error or 'Start the MCP server: python server.py'}"
        return {"response": msg}
    response = await _agent.chat(req.message, user_id=req.user_id, conversation_id=req.conversation_id)
    return {"response": response}


@app.get("/context-events")
async def context_events():
    return {"events": list(_compression_events)}


@app.get("/status")
async def status():
    return {
        "ready": bool(_agent and _agent._initialized),
        "error": _init_error,
        "tools": [t.get("function", {}).get("name") for t in (_agent.tools if _agent else [])],
        "config": _CONTEXT_CFG.to_dict(),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Context Management Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a1a2e; color: white; padding: 14px 24px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 17px; font-weight: 600; flex: 1; }
  header .badge { padding: 4px 10px; background: #e74c3c; border-radius: 20px;
                  font-size: 12px; font-weight: 600; }
  .main { flex: 1; display: flex; overflow: hidden; }

  /* ── Chat panel ── */
  .chat-panel { flex: 1; display: flex; flex-direction: column; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
          flex-direction: column; gap: 10px; }
  .msg { max-width: 72%; padding: 11px 15px; border-radius: 12px;
         line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #1a1a2e; color: white;
          border-bottom-right-radius: 4px; }
  .assistant { align-self: flex-start; background: white; color: #333;
               border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .thinking { align-self: flex-start; color: #aaa; font-style: italic; font-size: 13px; }
  #input-row { padding: 14px 20px; background: white; border-top: 1px solid #e0e0e0;
               display: flex; gap: 8px; }
  #input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px;
           font-size: 14px; outline: none; }
  #input:focus { border-color: #1a1a2e; }
  #send { padding: 10px 20px; background: #1a1a2e; color: white; border: none;
          border-radius: 8px; cursor: pointer; font-size: 14px; }
  #send:disabled { background: #aaa; cursor: not-allowed; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; padding: 8px 20px; }
  .chip { padding: 5px 12px; background: white; border: 1px solid #ddd;
          border-radius: 14px; font-size: 12px; cursor: pointer; color: #555; }
  .chip:hover { border-color: #1a1a2e; color: #1a1a2e; }

  /* ── Context panel ── */
  .ctx-panel { width: 320px; border-left: 1px solid #ddd; background: #fff;
               display: flex; flex-direction: column; overflow: hidden; }
  .ctx-panel h2 { padding: 14px 16px; font-size: 14px; font-weight: 700;
                  border-bottom: 1px solid #eee; background: #fafafa; color: #333; }
  .ctx-section { padding: 12px 16px; border-bottom: 1px solid #f0f0f0; }
  .ctx-section h3 { font-size: 11px; font-weight: 700; text-transform: uppercase;
                    color: #888; margin-bottom: 8px; letter-spacing: .5px; }
  .cfg-row { display: flex; justify-content: space-between; align-items: center;
             margin-bottom: 6px; }
  .cfg-label { font-size: 12px; color: #555; }
  .cfg-val { font-size: 12px; font-weight: 600; font-family: monospace;
             background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
  .cfg-val.highlight { background: #fde8e8; color: #c0392b; }
  #events-list { flex: 1; overflow-y: auto; padding: 8px 16px; }
  .event { padding: 7px 10px; border-radius: 6px; margin-bottom: 6px;
           font-size: 12px; line-height: 1.4; }
  .event.info { background: #e8f4fd; border-left: 3px solid #3498db; }
  .event.warning { background: #fef9e7; border-left: 3px solid #f39c12; }
  .event.error { background: #fde8e8; border-left: 3px solid #e74c3c; }
  .event .evt-ts { font-size: 11px; color: #999; margin-bottom: 2px; }
  .empty-events { color: #bbb; font-size: 13px; text-align: center; padding: 24px 0; }
  #clear-events { margin: 8px 16px; padding: 6px; width: calc(100% - 32px);
                  background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px;
                  font-size: 12px; cursor: pointer; color: #555; }
  #clear-events:hover { background: #eee; }

  /* Login */
  #login-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.75);
                   display: flex; align-items: center; justify-content: center; z-index: 999; }
  .login-box { background: white; padding: 32px; border-radius: 12px; width: 340px;
               display: flex; flex-direction: column; gap: 14px; text-align: center; }
  .login-box h2 { font-size: 20px; }
  .login-box p { font-size: 13px; color: #666; line-height: 1.5; }
  .login-box input { padding: 11px; border: 1px solid #ddd; border-radius: 8px; font-size: 15px; }
  .login-box button { padding: 11px; background: #1a1a2e; color: white; border: none;
                      border-radius: 8px; cursor: pointer; font-size: 15px; font-weight: 600; }
</style>
</head>
<body>

<div id="login-overlay">
  <div class="login-box">
    <h2>Context Management Test</h2>
    <p>Keep chatting — compression fires at 10% of the token limit. Watch the right panel for events.</p>
    <input id="login-input" type="text" placeholder="e.g. alice, user1" autofocus
           onkeydown="if(event.key==='Enter') doLogin()" />
    <button onclick="doLogin()">Start</button>
  </div>
</div>

<header>
  <h1>Context Management Test — <span id="display-user">Not logged in</span></h1>
  <div class="badge">threshold = 10%</div>
</header>

<div class="main">
  <!-- Chat -->
  <div class="chat-panel">
    <div class="chips">
      <div class="chip" onclick="send('show me dog toys')">Dog toys</div>
      <div class="chip" onclick="send('show me cat food')">Cat food</div>
      <div class="chip" onclick="send('show me bird accessories')">Bird accessories</div>
      <div class="chip" onclick="send('what products do you have?')">All products</div>
      <div class="chip" onclick="send(&quot;what's in my cart?&quot;)">View cart</div>
    </div>
    <div id="chat">
      <div class="msg assistant">Hi! Keep chatting — compression fires at 10% of the token limit. Watch the panel on the right for events.</div>
    </div>
    <div id="input-row">
      <input id="input" type="text" placeholder="Ask about products, cart, or anything..."
             onkeydown="if(event.key==='Enter') sendClick()" />
      <button id="send" onclick="sendClick()">Send</button>
    </div>
  </div>

  <!-- Context panel -->
  <div class="ctx-panel">
    <h2>Context Management</h2>

    <div class="ctx-section">
      <h3>Config</h3>
      <div class="cfg-row">
        <span class="cfg-label">Strategy</span>
        <span class="cfg-val">SMART</span>
      </div>
      <div class="cfg-row">
        <span class="cfg-label">Compression threshold</span>
        <span class="cfg-val highlight">10%</span>
      </div>
      <div class="cfg-row">
        <span class="cfg-label">Keep recent messages</span>
        <span class="cfg-val">2</span>
      </div>
    </div>

    <div class="ctx-section" style="flex:0 0 auto;">
      <h3>Compression Events</h3>
    </div>
    <div id="events-list">
      <div class="empty-events" id="empty-msg">No compression events yet.<br>Keep chatting to trigger it.</div>
    </div>
    <button id="clear-events" onclick="clearEvents()">Clear events</button>
  </div>
</div>

<script>
let userId = null;
let conversationId = null;

function doLogin() {
  const v = document.getElementById('login-input').value.trim();
  if (!v) return;
  userId = v;
  conversationId = crypto.randomUUID ? crypto.randomUUID() : 'conv_' + Date.now();
  document.getElementById('display-user').textContent = userId;
  document.getElementById('login-overlay').style.display = 'none';
  document.getElementById('input').focus();
  setInterval(fetchEvents, 2000);
}

async function fetchEvents() {
  try {
    const res = await fetch('/context-events');
    const data = await res.json();
    renderEvents(data.events || []);
  } catch (_) {}
}

function renderEvents(events) {
  const list = document.getElementById('events-list');
  if (events.length === 0) return;
  document.getElementById('empty-msg')?.remove();
  list.innerHTML = events.slice().reverse().map(e => {
    const cls = e.level === 'WARNING' ? 'warning' : e.level === 'ERROR' ? 'error' : 'info';
    return `<div class="event ${cls}">
      <div class="evt-ts">${e.ts} &mdash; ${e.level}</div>
      <div>${e.msg}</div>
    </div>`;
  }).join('');
}

function clearEvents() {
  document.getElementById('events-list').innerHTML =
    '<div class="empty-events" id="empty-msg">No compression events yet.<br>Keep chatting to trigger it.</div>';
}

function send(text) {
  document.getElementById('input').value = text;
  sendClick();
}

async function sendClick() {
  if (!userId) { alert('Please log in first!'); return; }
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  document.getElementById('send').disabled = true;
  appendMsg('user', msg);
  const thinking = appendMsg('thinking', 'Thinking...');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: msg, user_id: userId, conversation_id: conversationId }),
    });
    const data = await res.json();
    thinking.remove();
    appendMsg('assistant', data.response || '(no response)');
  } catch (e) {
    thinking.remove();
    appendMsg('assistant', 'Error: ' + e.message);
  }
  document.getElementById('send').disabled = false;
  input.focus();
}

function appendMsg(role, text) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 55)
    print("  Context Management Test  —  http://localhost:8081")
    print("=" * 55)
    print()
    print("Config: strategy=SMART, threshold=10%, keep_recent=2")
    print()
    print("Make sure MCP server is running first:")
    print("  python server.py")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8081)
