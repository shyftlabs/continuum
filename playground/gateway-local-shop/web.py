#!/usr/bin/env python3
"""
Gateway Shop Web UI.

Usage:
  Terminal 1: python server.py   (MCP server on :8888)
  Terminal 2: python web.py      (Web UI on :8081)

Requires Smart Gateway running at http://localhost:8787/v1
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from contextlib import asynccontextmanager

import uvicorn
from agent import LocalShopAgent
from config import default_config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator import LogLevel, setup_logging

setup_logging(level=LogLevel.INFO)

_agent: LocalShopAgent | None = None
_init_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _init_error
    _agent = LocalShopAgent(config=default_config)
    try:
        await _agent.initialize()
        print(f"✓ Agent ready — {len(_agent.tools)} tools loaded")
    except Exception as e:
        _init_error = str(e)
        print(f"✗ Agent init failed: {e}")
        print("Is the MCP server running?  python server.py")
    yield
    if _agent and _agent._initialized:
        try:
            await asyncio.wait_for(_agent.close(), timeout=1.0)
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str


class ClearMemoryRequest(BaseModel):
    user_id: str


class DeleteMemoryRequest(BaseModel):
    memory_id: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _agent or not _agent._initialized:
        msg = f"Agent not connected to MCP server. {_init_error or 'Start the MCP server with: python server.py'}"
        return {"response": msg}
    response = await _agent.chat(
        req.message,
        user_id=req.user_id,
        conversation_id=req.conversation_id
    )
    return {"response": response}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not _agent or not _agent._initialized:
        msg = f"Agent not connected to MCP server. {_init_error or 'Start the MCP server with: python server.py'}"
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    return StreamingResponse(
        _agent.chat_stream(req.message, user_id=req.user_id, conversation_id=req.conversation_id),
        media_type="text/event-stream",
    )


def _get_memory_client():
    if not _agent or not _agent._initialized:
        return None
    client = _agent._container.memory_client if _agent._container else None
    return client if client and client.is_enabled else None


@app.get("/memory/list")
async def list_memories(user_id: str):
    client = _get_memory_client()
    if not client:
        return {"success": False, "error": "Memory not available"}
    try:
        entries = await client.get_all(user_id=user_id)
        return {"success": True, "memories": [{"id": e.id, "text": e.memory} for e in entries]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/memory/delete")
async def delete_memory(req: DeleteMemoryRequest):
    client = _get_memory_client()
    if not client:
        return {"success": False, "error": "Memory not available"}
    try:
        await client.delete(req.memory_id)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/memory/clear")
async def clear_memory(req: ClearMemoryRequest):
    client = _get_memory_client()
    if not client:
        return {"success": False, "error": "Memory not available"}
    try:
        await client.delete_all(user_id=req.user_id)
        return {"success": True, "message": f"All memories cleared for user '{req.user_id}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/status")
async def status():
    tools = []
    if _agent and _agent.tools:
        for t in _agent.tools:
            try:
                tools.append(t.function.name if hasattr(t, "function") else t.get("function", {}).get("name"))
            except Exception:
                pass
    return {
        "ready": bool(_agent and _agent._initialized),
        "error": _init_error,
        "tools": tools,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gateway Shop Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #2c3e50; color: white; padding: 16px 24px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; flex: 1; }
  #new-chat-btn { padding: 8px 16px; background: transparent; border: 1px solid white; color: white; border-radius: 6px; cursor: pointer; font-size: 14px; display: none; }
  #new-chat-btn:hover { background: rgba(255,255,255,0.1); }
  #stream-toggle { padding: 8px 16px; border: 1px solid white; border-radius: 6px; cursor: pointer; font-size: 14px; margin-right: 8px; background: transparent; color: white; }
  #stream-toggle.on { background: #27ae60; border-color: #27ae60; }
  #stream-toggle:hover { opacity: 0.85; }
  header span { font-size: 24px; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex;
          flex-direction: column; gap: 12px; }
  .msg { max-width: 70%; padding: 12px 16px; border-radius: 12px;
         line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #2c3e50; color: white;
          border-bottom-right-radius: 4px; }
  .assistant { align-self: flex-start; background: white; color: #333;
               border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .thinking { align-self: flex-start; color: #999; font-style: italic; font-size: 13px; }
  .streaming-cursor { display: inline-block; color: #999; font-size: 13px; margin-left: 4px; animation: blink 1s step-start infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
  #input-row { padding: 16px 24px; background: white; border-top: 1px solid #e0e0e0;
               display: flex; gap: 10px; }
  #input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px;
           font-size: 14px; outline: none; }
  #input:focus { border-color: #2c3e50; }
  #send { padding: 10px 20px; background: #2c3e50; color: white; border: none;
          border-radius: 8px; cursor: pointer; font-size: 14px; }
  #send:hover { background: #34495e; }
  #send:disabled { background: #aaa; cursor: not-allowed; }
  .suggestions { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 24px 12px; }
  .chip { padding: 6px 12px; background: white; border: 1px solid #ddd;
          border-radius: 16px; font-size: 12px; cursor: pointer; color: #555; }
  .chip:hover { border-color: #2c3e50; color: #2c3e50; }
  /* Login Screen */
  #login-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: flex; 
                   align-items: center; justify-content: center; z-index: 1000; }
  .login-box { background: white; padding: 32px; border-radius: 12px; width: 350px;
               display: flex; flex-direction: column; gap: 16px; text-align: center; }
  .login-box input { padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
  .login-box button { padding: 12px; background: #2c3e50; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; }
</style>
</head>
<body>
<div id="memory-overlay" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:999; align-items:center; justify-content:center;">
  <div style="background:white; border-radius:12px; width:480px; max-height:80vh; display:flex; flex-direction:column; overflow:hidden;">
    <div style="padding:16px 20px; border-bottom:1px solid #eee; display:flex; justify-content:space-between; align-items:center;">
      <h3 style="margin:0;">Long-term Memories</h3>
      <button onclick="closeMemoryPanel()" style="background:none; border:none; font-size:20px; cursor:pointer;">✕</button>
    </div>
    <div id="memory-list" style="flex:1; overflow-y:auto; padding:12px 20px;"></div>
  </div>
</div>
<div id="login-overlay">
  <div class="login-box">
    <h2>Login</h2>
    <p>Enter a User ID to start chatting. If you reuse the same ID, the assistant will remember you!</p>
    <input id="login-input" type="text" placeholder="e.g. alice, bob, user_123" onkeydown="if(event.key==='Enter') doLogin()" autofocus />
    <button onclick="doLogin()">Start Chatting</button>
  </div>
</div>
<header>
  <span>🐾</span>
  <h1>Gateway Shop Assistant - <span id="display-user">Not Logged In</span></h1>
  <div style="display:flex;">
    <button id="change-user-btn" style="display:none; background:#7f8c8d; margin-right:8px;" onclick="changeUser()">Switch User</button>
    <button id="manage-memory-btn" style="display:none; background:#8e44ad; margin-right:8px;" onclick="openMemoryPanel()">Memories</button>
  <button id="clear-memory-btn" style="display:none; background:#c0392b; margin-right:8px;" onclick="clearMemory()">Clear All</button>
    <button id="stream-toggle" onclick="toggleStream()">○ Stream</button>
    <button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
  </div>
</header>
<div class="suggestions">
  <div class="chip" onclick="send('show me dog toys')">Dog toys</div>
  <div class="chip" onclick="send('show me cat food')">Cat food</div>
  <div class="chip" onclick="send(&quot;what&apos;s in my cart?&quot;)">View cart</div>
  <div class="chip" onclick="send('checkout')">Checkout</div>
</div>
<div id="chat">
  <div class="msg assistant">Hi! I'm your pet shop assistant. Ask me to search for products, add them to your cart, or checkout. 🐶🐱</div>
</div>
<div id="input-row">
  <input id="input" type="text" placeholder="Ask about products, cart, checkout..." autofocus
         onkeydown="if(event.key==='Enter') sendClick()" />
  <button id="send" onclick="sendClick()">Send</button>
</div>
<script>
let currentUserId = null;
let currentConversationId = null;
let streamMode = false;

function toggleStream() {
  streamMode = !streamMode;
  const btn = document.getElementById('stream-toggle');
  btn.textContent = streamMode ? '✓ Stream' : '○ Stream';
  btn.classList.toggle('on', streamMode);
}

function doLogin() {
  const v = document.getElementById('login-input').value.trim();
  if(!v) return;
  currentUserId = v;
  document.getElementById('display-user').textContent = currentUserId;
  document.getElementById('login-overlay').style.display = 'none';
  document.getElementById('new-chat-btn').style.display = 'block';
  document.getElementById('change-user-btn').style.display = 'block';
  document.getElementById('clear-memory-btn').style.display = 'block';
  document.getElementById('manage-memory-btn').style.display = 'block';
  startNewChat();
}

async function openMemoryPanel() {
  if (!currentUserId) return;
  document.getElementById('memory-overlay').style.display = 'flex';
  const list = document.getElementById('memory-list');
  list.innerHTML = '<p style="color:#999;">Loading...</p>';
  const res = await fetch(`/memory/list?user_id=${encodeURIComponent(currentUserId)}`);
  const data = await res.json();
  if (!data.success) {
    list.innerHTML = `<p style="color:red;">${data.error}</p>`;
    return;
  }
  if (data.memories.length === 0) {
    list.innerHTML = '<p style="color:#999;">No memories found.</p>';
    return;
  }
  list.innerHTML = data.memories.map(m => `
    <div style="display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #f0f0f0;">
      <span style="flex:1; font-size:14px;">${m.text}</span>
      <button onclick="deleteMemory('${m.id}', this)" style="padding:4px 10px; background:#c0392b; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">Delete</button>
    </div>
  `).join('');
}

function closeMemoryPanel() {
  document.getElementById('memory-overlay').style.display = 'none';
}

async function deleteMemory(memoryId, btn) {
  btn.disabled = true;
  btn.textContent = '...';
  const res = await fetch('/memory/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({memory_id: memoryId})
  });
  const data = await res.json();
  if (data.success) {
    btn.closest('div').remove();
    const list = document.getElementById('memory-list');
    if (!list.children.length) list.innerHTML = '<p style="color:#999;">No memories found.</p>';
  } else {
    btn.disabled = false;
    btn.textContent = 'Delete';
    alert('Failed: ' + data.error);
  }
}

async function clearMemory() {
  if (!currentUserId) return;
  if (!confirm(`Clear all long-term memories for user '${currentUserId}'?`)) return;
  const res = await fetch('/memory/clear', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: currentUserId})
  });
  const data = await res.json();
  if (data.success) {
    appendMsg('assistant', `Memory cleared. I no longer remember anything about you.`);
  } else {
    appendMsg('assistant', `Failed to clear memory: ${data.error}`);
  }
}

function changeUser() {
  currentUserId = null;
  document.getElementById('display-user').textContent = "Not Logged In";
  document.getElementById('login-overlay').style.display = 'flex';
  document.getElementById('login-input').value = '';
  document.getElementById('login-input').focus();
}

function startNewChat() {
  currentConversationId = (window.crypto && window.crypto.randomUUID) 
    ? window.crypto.randomUUID() 
    : 'chat_' + Date.now().toString() + '_' + Math.random().toString(36).substring(2, 9);
  const chat = document.getElementById('chat');
  chat.innerHTML = '<div class="msg assistant">Hi! I\\'m your pet shop assistant. Ask me to search for products, add them to your cart, or checkout. 🐶🐱</div>';
  document.getElementById('input').focus();
}

function send(text) {
  document.getElementById('input').value = text;
  sendClick();
}
async function sendClick() {
  if (!currentUserId) {
    alert("Please log in first!");
    return;
  }
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  document.getElementById('send').disabled = true;
  appendMsg('user', msg);
  const thinking = appendMsg('thinking', 'Thinking...');
  let assistantDiv = null;
  let cursorSpan = null;
  try {
    if (streamMode) {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, user_id: currentUserId, conversation_id: currentConversationId})
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === 'content') {
            if (!assistantDiv) {
              thinking.remove();
              assistantDiv = appendMsg('assistant', '');
              cursorSpan = document.createElement('span');
              cursorSpan.className = 'streaming-cursor';
              cursorSpan.textContent = '...';
              assistantDiv.appendChild(cursorSpan);
            }
            assistantDiv.insertBefore(document.createTextNode(event.content), cursorSpan);
            document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
          } else if (event.type === 'tool_call' && event.status === 'start') {
            if (!assistantDiv) thinking.textContent = `Using ${event.tool_name}...`;
          } else if (event.type === 'error') {
            thinking.remove();
            if (cursorSpan) cursorSpan.remove();
            appendMsg('assistant', 'Error: ' + event.error);
          } else if (event.type === 'done') {
            thinking.remove();
            if (cursorSpan) cursorSpan.remove();
            if (!assistantDiv) appendMsg('assistant', '(no response)');
          }
        }
      }
    } else {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, user_id: currentUserId, conversation_id: currentConversationId})
      });
      const data = await res.json();
      thinking.remove();
      appendMsg('assistant', data.response || data.error || 'No response');
    }
  } catch(e) {
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
    print("Gateway Shop Web UI at http://localhost:8081")
    print("Make sure MCP server is running:  python server.py")
    print("Make sure Smart Gateway is running: http://localhost:8787/v1")
    uvicorn.run(app, host="0.0.0.0", port=8081)
