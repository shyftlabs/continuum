#!/usr/bin/env python3
"""
No-MCP Demo Web UI — all 10 workflow modes, no external tools.

Usage:
  python web.py      (Web UI on :8083)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from contextlib import asynccontextmanager

import uvicorn
from config import default_config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from workflows import MODES, _BaseWorkflow, create_workflow

from orchestrator import LogLevel, setup_logging

setup_logging(level=LogLevel.INFO)

_workflows: dict[str, _BaseWorkflow] = {}
_init_errors: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for wf in _workflows.values():
        if wf._initialized:
            try:
                await asyncio.wait_for(wf.close(), timeout=1.0)
            except Exception:
                pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str
    mode: str = "sequential"


async def get_workflow(mode: str) -> tuple[_BaseWorkflow | None, str | None]:
    if mode in _init_errors:
        return None, _init_errors[mode]
    if mode not in _workflows:
        try:
            wf = create_workflow(mode)
            await wf.initialize()
            _workflows[mode] = wf
        except Exception as e:
            _init_errors[mode] = str(e)
            return None, str(e)
    return _workflows[mode], None


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.mode not in MODES:
        return {"response": f"Unknown mode '{req.mode}'. Choose from: {', '.join(MODES)}"}
    wf, error = await get_workflow(req.mode)
    if error:
        return {"response": f"Failed to initialize '{req.mode}' mode: {error}"}
    response = await wf.chat(req.message, user_id=req.user_id, conversation_id=req.conversation_id)
    return {"response": response}


@app.get("/status")
async def status():
    return {
        "modes": list(MODES.keys()),
        "initialized": list(_workflows.keys()),
        "errors": _init_errors,
    }


MODE_DESCRIPTIONS = default_config.mode_descriptions

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>No-MCP Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f4f8; height: 100vh; display: flex; flex-direction: column; }

  header { background: #1a365d; color: white; padding: 12px 20px;
           display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 16px; font-weight: 600; flex: 1; min-width: 160px; }
  #user-display { font-size: 13px; opacity: 0.8; }

  #mode-bar { background: #2a4a7f; padding: 8px 20px; display: flex; align-items: center; gap: 10px; }
  #mode-bar label { color: #cdd; font-size: 13px; white-space: nowrap; }
  #mode-select { padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px;
                 background: white; cursor: pointer; }
  #mode-desc { color: #aac; font-size: 12px; flex: 1; font-style: italic; }

  .hdr-btn { padding: 6px 14px; background: transparent; border: 1px solid rgba(255,255,255,0.5);
             color: white; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .hdr-btn:hover { background: rgba(255,255,255,0.1); }
  .hdr-btn.hidden { display: none; }

  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
          flex-direction: column; gap: 10px; }
  .msg { max-width: 74%; padding: 10px 14px; border-radius: 12px;
         line-height: 1.5; font-size: 14px; }
  .user { align-self: flex-end; background: #1a365d; color: white; border-bottom-right-radius: 4px; }
  .assistant { align-self: flex-start; background: white; color: #333;
               border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .thinking { align-self: flex-start; color: #999; font-style: italic; font-size: 13px; }
  .mode-tag { font-size: 11px; color: #888; margin-bottom: 3px; }

  .suggestions { display: flex; gap: 6px; flex-wrap: wrap; padding: 6px 20px 0; }
  .chip { padding: 5px 11px; background: white; border: 1px solid #ddd;
          border-radius: 14px; font-size: 12px; cursor: pointer; color: #555; white-space: nowrap; }
  .chip:hover { border-color: #1a365d; color: #1a365d; }

  #input-row { padding: 12px 20px; background: white; border-top: 1px solid #e0e0e0;
               display: flex; gap: 8px; }
  #input { flex: 1; padding: 9px 13px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }
  #input:focus { border-color: #1a365d; }
  #send { padding: 9px 18px; background: #1a365d; color: white; border: none;
          border-radius: 8px; cursor: pointer; font-size: 14px; }
  #send:hover { background: #2a4a7f; }
  #send:disabled { background: #aaa; cursor: not-allowed; }

  #login-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.75); display: flex;
                   align-items: center; justify-content: center; z-index: 100; }
  .login-box { background: white; padding: 28px; border-radius: 12px; width: 360px;
               display: flex; flex-direction: column; gap: 14px; text-align: center; }
  .login-box h2 { color: #1a365d; }
  .login-box p { font-size: 13px; color: #666; }
  .login-box input { padding: 11px; border: 1px solid #ddd; border-radius: 8px; font-size: 15px; }
  .login-box button { padding: 11px; background: #1a365d; color: white; border: none;
                      border-radius: 8px; cursor: pointer; font-size: 15px; font-weight: 600; }
  .login-box button:hover { background: #2a4a7f; }
</style>
</head>
<body>

<div id="login-overlay">
  <div class="login-box">
    <h2>Research Assistant</h2>
    <p>Enter a User ID to start. Reuse the same ID and the assistant will remember you.</p>
    <input id="login-input" type="text" placeholder="e.g. alice, bob, user_123"
           onkeydown="if(event.key==='Enter') doLogin()" autofocus />
    <button onclick="doLogin()">Start</button>
  </div>
</div>

<header>
  <span>🔬</span>
  <h1>No-MCP Demo — Research &amp; Writing Assistant</h1>
  <span id="user-display"></span>
  <button id="change-user-btn" class="hdr-btn hidden" onclick="changeUser()">Switch User</button>
  <button id="new-chat-btn" class="hdr-btn hidden" onclick="startNewChat()">+ New Chat</button>
</header>

<div id="mode-bar">
  <label>Mode:</label>
  <select id="mode-select" onchange="onModeChange()">
    <option value="sequential">sequential</option>
    <option value="parallel">parallel</option>
    <option value="loop">loop</option>
    <option value="scatter">scatter</option>
    <option value="supervised">supervised</option>
    <option value="planner">planner</option>
    <option value="debate">debate</option>
    <option value="reflection">reflection</option>
    <option value="router">router</option>
    <option value="handoff">handoff</option>
  </select>
  <span id="mode-desc"></span>
</div>

<div class="suggestions" id="suggestions"></div>

<div id="chat"></div>

<div id="input-row">
  <input id="input" type="text" placeholder="Ask anything — research, writing, analysis..."
         onkeydown="if(event.key==='Enter') sendClick()" />
  <button id="send" onclick="sendClick()">Send</button>
</div>

<script>
const MODE_DESCRIPTIONS = """ + str({k: v for k, v in MODE_DESCRIPTIONS.items()}).replace("'", '"') + """;

const MODE_SUGGESTIONS = {
  sequential:  ["Explain quantum computing", "How does the internet work?", "What is blockchain?"],
  parallel:    ["Explain climate change", "Analyse the French Revolution", "Explain artificial intelligence"],
  loop:        ["Explain recursion until I understand", "Explain photosynthesis simply", "How does GPS work?"],
  scatter:     ["Compare Python, JavaScript, and Rust", "Compare solar, wind, and nuclear energy"],
  supervised:  ["Write an essay about the Renaissance", "Write about the impact of the printing press"],
  planner:     ["Help me understand machine learning from scratch", "Explain the history of computing"],
  debate:      ["Should AI replace human jobs?", "Is remote work better than office work?", "Nuclear energy: good or bad?"],
  reflection:  ["Write a cover letter for a software engineer role", "Write a summary of World War II"],
  router:      ["How does photosynthesis work?", "Write a haiku about autumn", "Is the Great Wall visible from space?"],
  handoff:     ["Research the causes of World War I", "Explain the theory of relativity", "What is quantum entanglement?"],
};

let currentUserId = null;
let currentConversationId = null;
let currentMode = "sequential";

function doLogin() {
  const v = document.getElementById('login-input').value.trim();
  if (!v) return;
  currentUserId = v;
  document.getElementById('user-display').textContent = 'User: ' + v;
  document.getElementById('login-overlay').style.display = 'none';
  document.getElementById('new-chat-btn').classList.remove('hidden');
  document.getElementById('change-user-btn').classList.remove('hidden');
  startNewChat();
}

function changeUser() {
  currentUserId = null;
  document.getElementById('user-display').textContent = '';
  document.getElementById('login-overlay').style.display = 'flex';
  document.getElementById('login-input').value = '';
  document.getElementById('login-input').focus();
}

function startNewChat() {
  currentConversationId = (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : 'chat_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
  const chat = document.getElementById('chat');
  chat.innerHTML = '';
  appendMsg('assistant', getWelcome(currentMode));
  document.getElementById('input').focus();
}

function getWelcome(mode) {
  return `Hi! Running in ${mode.toUpperCase()} mode.\\n${MODE_DESCRIPTIONS[mode] || ''}\\n\\nAsk me anything!`;
}

function onModeChange() {
  currentMode = document.getElementById('mode-select').value;
  document.getElementById('mode-desc').textContent = MODE_DESCRIPTIONS[currentMode] || '';
  renderSuggestions(currentMode);
  startNewChat();
}

function renderSuggestions(mode) {
  const chips = (MODE_SUGGESTIONS[mode] || []).map(q =>
    `<div class="chip" onclick='send(${JSON.stringify(q)})'>${q}</div>`
  ).join('');
  document.getElementById('suggestions').innerHTML = chips;
}

function send(text) {
  document.getElementById('input').value = text;
  sendClick();
}

async function sendClick() {
  if (!currentUserId) { alert('Please log in first!'); return; }
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  document.getElementById('send').disabled = true;
  appendMsg('user', msg);
  const thinking = appendMsg('thinking', `[${currentMode}] thinking...`);
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        user_id: currentUserId,
        conversation_id: currentConversationId,
        mode: currentMode,
      })
    });
    const data = await res.json();
    thinking.remove();
    const div = appendMsg('assistant', '');
    div.innerHTML = `<div class="mode-tag">[${currentMode}]</div>` +
      escapeHtml(data.response || data.error || 'No response').replace(/\\n/g, '<br>');
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

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('mode-desc').textContent = MODE_DESCRIPTIONS['sequential'] || '';
  renderSuggestions('sequential');
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("No-MCP Demo Web UI at http://localhost:8083")
    uvicorn.run(app, host="0.0.0.0", port=8083)
