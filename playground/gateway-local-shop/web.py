#!/usr/bin/env python3
"""
Gateway Shop Web UI.

Usage:
  Terminal 1: python server.py   (MCP server on :8888)
  Terminal 2: python web.py      (Web UI on :8081)

Requires Smart Gateway running at http://localhost:8787/v1
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from contextlib import asynccontextmanager

import uvicorn
from agent import LocalShopAgent
from config import default_config
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.jpeg")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/assets/logo.jpeg")
async def logo():
    return FileResponse(_LOGO_PATH, media_type="image/jpeg")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(_LOGO_PATH, media_type="image/jpeg")


@app.post("/chat")
async def chat(req: ChatRequest):
    import time as _t
    if not _agent or not _agent._initialized:
        msg = f"Agent not connected to MCP server. {_init_error or 'Start the MCP server with: python server.py'}"
        return {"response": msg, "duration_ms": 0}
    t0 = _t.perf_counter()
    response = await _agent.chat(
        req.message,
        user_id=req.user_id,
        conversation_id=req.conversation_id
    )
    return {"response": response, "duration_ms": int((_t.perf_counter() - t0) * 1000)}


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
<title>Continuum · Pet Shop</title>
<link rel="icon" type="image/jpeg" href="/assets/logo.jpeg">
<link rel="shortcut icon" type="image/jpeg" href="/assets/logo.jpeg">
<link rel="apple-touch-icon" href="/assets/logo.jpeg">
<meta name="theme-color" content="#0d0d0d">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.11/dist/purify.min.js"></script>
<style>
  :root {
    --bg: #ffffff;
    --bg-soft: #f9f9fb;
    --bg-sidebar: #fafafa;
    --bg-hover: #ececf1;
    --bg-user: #f3f3f5;
    --bg-input: #ffffff;
    --text: #0d0d0d;
    --text-soft: #525261;
    --text-mute: #8e8ea0;
    --border: #e8e8ec;
    --border-strong: #d0d0d8;
    --accent: #10a37f;
    --accent-soft: #e6f7f1;
    --accent-violet: #8b5cf6;
    --accent-violet-soft: #f0eaff;
    --accent-amber: #f59e0b;
    --shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.05);
    --shadow-strong: 0 4px 24px rgba(15,23,42,.08), 0 12px 40px rgba(15,23,42,.08);
    --radius: 14px;
    --radius-sm: 10px;
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", system-ui, sans-serif;
  }
  html[data-theme="dark"] {
    --bg: #131316;
    --bg-soft: #1b1b1f;
    --bg-sidebar: #0f0f12;
    --bg-hover: #26262c;
    --bg-user: #2a2a31;
    --bg-input: #1f1f24;
    --text: #ececec;
    --text-soft: #b4b4b8;
    --text-mute: #80808a;
    --border: #2a2a30;
    --border-strong: #3a3a42;
    --accent: #1ec694;
    --accent-soft: #14352c;
    --accent-violet: #a78bfa;
    --accent-violet-soft: #2a1f4a;
    --accent-amber: #fbbf24;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 20px rgba(0,0,0,.3);
    --shadow-strong: 0 4px 24px rgba(0,0,0,.5), 0 14px 40px rgba(0,0,0,.4);
  }
  @media (prefers-color-scheme: dark) {
    html:not([data-theme]) { /* default to dark when system says dark */
      --bg: #131316; --bg-soft: #1b1b1f; --bg-sidebar: #0f0f12;
      --bg-hover: #26262c; --bg-user: #2a2a31; --bg-input: #1f1f24;
      --text: #ececec; --text-soft: #b4b4b8; --text-mute: #80808a;
      --border: #2a2a30; --border-strong: #3a3a42;
      --accent: #1ec694; --accent-soft: #14352c;
      --accent-violet: #a78bfa; --accent-violet-soft: #2a1f4a;
      --accent-amber: #fbbf24;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 20px rgba(0,0,0,.3);
      --shadow-strong: 0 4px 24px rgba(0,0,0,.5), 0 14px 40px rgba(0,0,0,.4);
    }
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
  body {
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.55;
    color: var(--text);
    background: var(--bg);
    overflow: hidden;
    background-image:
      radial-gradient(800px circle at 0% 0%, var(--accent-soft) 0%, transparent 45%),
      radial-gradient(700px circle at 100% 100%, var(--accent-violet-soft) 0%, transparent 45%);
    background-attachment: fixed;
  }
  button { font-family: inherit; cursor: pointer; border: none; background: none; color: inherit; }
  input, textarea { font-family: inherit; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }

  /* Layout */
  .app { display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
  @media (max-width: 768px) { .app { grid-template-columns: 1fr; } .sidebar { display: none; } }

  /* Sidebar */
  .sidebar {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 12px;
    gap: 6px;
    overflow-y: auto;
  }
  .sidebar::-webkit-scrollbar { width: 4px; }
  .brand {
    display: flex; align-items: center; gap: 11px;
    padding: 10px 12px 18px;
    line-height: 1;
  }
  .brand .mark {
    width: 34px; height: 34px;
    border-radius: 8px;
    object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,.12);
    flex-shrink: 0;
    display: block;
  }
  .brand .text { display: flex; flex-direction: column; gap: 5px; line-height: 1; min-width: 0; }
  .brand .wordmark {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
    font-weight: 600;
    font-size: 20px;
    letter-spacing: -.025em;
    color: var(--text);
  }
  .brand .by {
    font-size: 10.5px;
    letter-spacing: .03em;
    color: var(--text-mute);
    font-weight: 500;
    font-family: 'Geist', -apple-system, sans-serif;
  }
  .brand .by em {
    font-style: normal;
    color: var(--text-soft);
    font-weight: 600;
  }
  .side-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 10px;
    width: 100%; text-align: left;
    font-size: 14px; color: var(--text);
    transition: background .15s ease;
  }
  .side-btn:hover { background: var(--bg-hover); }
  .side-btn .icon { width: 18px; height: 18px; flex-shrink: 0; color: var(--text-soft); }
  .side-btn.primary {
    background: var(--bg);
    border: 1px solid var(--border);
  }
  .side-btn.primary:hover { background: var(--bg-hover); }
  .side-divider { height: 1px; background: var(--border); margin: 8px 4px; }
  .side-section-label {
    padding: 8px 12px 4px;
    font-size: 11px; font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; color: var(--text-mute);
  }
  .side-footer { margin-top: auto; padding-top: 8px; }
  .user-chip {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 10px;
    transition: background .15s ease;
  }
  .user-chip:hover { background: var(--bg-hover); cursor: pointer; }
  .user-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #6e6efb 0%, #b95cf4 100%);
    color: white; font-weight: 600; font-size: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .user-name { font-size: 13px; font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Mode card */
  .mode-card {
    margin: 8px 4px;
    padding: 12px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent-soft) 0%, var(--accent-violet-soft) 100%);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }
  .mode-card::before {
    content: ''; position: absolute; inset: -1px;
    border-radius: 12px;
    padding: 1px;
    background: linear-gradient(135deg, var(--accent), var(--accent-violet));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    opacity: .35;
    pointer-events: none;
  }
  .mode-card .label {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; color: var(--text-soft);
  }
  .mode-card .label .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .55; transform: scale(.85); }
  }
  .mode-card .title { font-size: 14px; font-weight: 600; margin-top: 6px; color: var(--text); }
  .mode-card .sub { font-size: 12px; color: var(--text-soft); margin-top: 2px; }

  /* Stats panel */
  .stats {
    margin: 6px 4px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
  }
  .stat-box {
    padding: 10px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .stat-box .k { font-size: 10.5px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--text-mute); }
  .stat-box .v { font-size: 16px; font-weight: 600; color: var(--text); margin-top: 2px; font-variant-numeric: tabular-nums; }
  .stat-box.full { grid-column: span 2; }

  /* Recent chats list */
  .recents {
    display: flex; flex-direction: column; gap: 2px;
    margin-top: 4px;
  }
  .recent-item {
    display: flex; align-items: center; gap: 9px;
    padding: 8px 10px;
    border-radius: 9px;
    font-size: 13px;
    color: var(--text);
    transition: background .15s ease;
    text-align: left;
    width: 100%;
  }
  .recent-item:hover { background: var(--bg-hover); }
  .recent-item.active { background: var(--bg-hover); }
  .recent-item .ttl { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .recent-item .delx {
    width: 22px; height: 22px; border-radius: 6px;
    display: none; align-items: center; justify-content: center;
    color: var(--text-mute);
  }
  .recent-item .delx:hover { background: var(--bg); color: #e5484d; }
  .recent-item:hover .delx { display: inline-flex; }

  /* Top bar pills */
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    background: var(--bg);
    border: 1px solid var(--border);
    font-size: 12px; font-weight: 500;
    color: var(--text-soft);
    font-variant-numeric: tabular-nums;
  }
  .pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
  .pill strong { color: var(--text); font-weight: 600; }
  .pill.violet .dot { background: var(--accent-violet); }
  .pill.amber .dot { background: var(--accent-amber); }

  /* Assistant attribution */
  .asst-attribution {
    margin-top: 10px;
    margin-bottom: 2px;
    line-height: 1;
  }
  .asst-attribution .brand-mini {
    font-family: 'Instrument Serif', Georgia, "Times New Roman", serif;
    font-size: 17px;
    color: var(--text-soft);
    font-weight: 400;
    letter-spacing: -.005em;
  }
  .asst-attribution .by-line {
    display: block;
    margin-top: 3px;
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-mute);
    font-weight: 500;
  }

  /* Message meta + actions */
  .msg-meta {
    display: flex; align-items: center; gap: 8px;
    font-size: 11.5px; color: var(--text-mute);
    margin-top: 6px;
  }
  .msg-meta .meta-dot { width: 3px; height: 3px; border-radius: 50%; background: var(--text-mute); opacity: .6; }
  .msg-row.assistant { position: relative; }
  .copy-btn {
    position: absolute;
    top: 2px; right: 0;
    padding: 5px;
    border-radius: 6px;
    color: var(--text-mute);
    opacity: 0;
    transition: opacity .15s ease, background .15s ease;
  }
  .msg-row.assistant:hover .copy-btn { opacity: 1; }
  .copy-btn:hover { background: var(--bg-hover); color: var(--text); }
  .copy-btn .icon { width: 14px; height: 14px; }

  /* Feature row in empty state */
  .feature-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
    margin-top: 18px; max-width: 560px; width: 100%;
  }
  @media (max-width: 540px) { .feature-row { grid-template-columns: 1fr; } }
  .feature {
    padding: 14px 14px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    text-align: left;
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
  }
  .feature:hover { transform: translateY(-2px); border-color: var(--border-strong); box-shadow: var(--shadow); }
  .feature .ficon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 8px; font-size: 16px;
  }
  .feature.smart .ficon { background: var(--accent-soft); color: var(--accent); }
  .feature.mem .ficon { background: var(--accent-violet-soft); color: var(--accent-violet); }
  .feature.tools .ficon { background: #fef3c7; color: var(--accent-amber); }
  html[data-theme="dark"] .feature.tools .ficon { background: #3a2e1a; }
  .feature h4 { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
  .feature p { font-size: 12px; color: var(--text-soft); line-height: 1.45; }

  /* Main column */
  .main { display: flex; flex-direction: column; min-height: 0; background: var(--bg); }
  .top-bar {
    height: 52px;
    padding: 0 24px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid transparent;
  }
  .top-title { font-weight: 600; font-size: 14px; color: var(--text-soft); }
  .top-actions { display: flex; gap: 6px; align-items: center; }
  .icon-btn {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-soft);
    transition: background .15s ease, color .15s ease;
  }
  .icon-btn:hover { background: var(--bg-hover); color: var(--text); }
  .icon-btn .icon { width: 18px; height: 18px; }

  /* Chat */
  .chat-scroll {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .chat-inner {
    max-width: 760px;
    margin: 0 auto;
    padding: 24px 24px 200px;
    display: flex; flex-direction: column; gap: 12px;
  }

  /* Empty state */
  .empty {
    flex: 1;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center;
    padding: 24px;
    min-height: calc(100vh - 240px);
  }
  .empty-logo {
    width: 64px; height: 64px; border-radius: 18px;
    background: linear-gradient(135deg, #10a37f 0%, #1ec694 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(16,163,127,.25);
  }
  .empty-title { font-size: 24px; font-weight: 600; margin-bottom: 8px; }
  .empty-sub { color: var(--text-soft); font-size: 14px; margin-bottom: 28px; }
  .suggest-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(200px, 280px));
    gap: 10px;
    width: 100%;
    max-width: 600px;
  }
  @media (max-width: 540px) { .suggest-grid { grid-template-columns: 1fr; } }
  .suggest-card {
    padding: 14px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    text-align: left;
    transition: background .15s ease, border-color .15s ease, transform .12s ease;
  }
  .suggest-card:hover {
    background: var(--bg-hover);
    border-color: var(--border-strong);
    transform: translateY(-1px);
  }
  .suggest-title { font-size: 13.5px; font-weight: 600; margin-bottom: 2px; }
  .suggest-sub { font-size: 12.5px; color: var(--text-soft); }

  /* Messages */
  .msg-row { display: flex; gap: 12px; padding: 4px 0; animation: fade-in .25s ease; }
  @keyframes fade-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  .msg-row.user { justify-content: flex-end; }
  .avatar {
    width: 30px; height: 30px; border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600; color: white;
  }
  .avatar.assistant { background: linear-gradient(135deg, #10a37f 0%, #1ec694 100%); }
  .avatar.user { background: linear-gradient(135deg, #6e6efb 0%, #b95cf4 100%); font-size: 13px; }
  .msg {
    padding: 10px 14px;
    border-radius: 18px;
    line-height: 1.55;
    font-size: 15px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }
  .msg.assistant { background: transparent; padding-left: 0; padding-right: 0; }
  .msg.user { background: var(--bg-user); color: var(--text); border-bottom-right-radius: 6px; width: fit-content; max-width: 100%; }
  .user-stack {
    display: inline-flex; flex-direction: column; align-items: flex-end; gap: 3px;
    max-width: calc(100% - 50px);
    min-width: 0;
  }
  .asst-stack {
    flex: 1; min-width: 0;
    max-width: calc(100% - 42px);
    position: relative;
  }
  .msg.assistant > :first-child { margin-top: 0; }
  .msg.assistant > :last-child { margin-bottom: 0; }
  .msg p { margin: 0 0 10px; }
  .msg p:last-child { margin-bottom: 0; }
  .msg h1, .msg h2, .msg h3, .msg h4 { margin: 16px 0 8px; font-weight: 600; line-height: 1.3; }
  .msg h1 { font-size: 20px; } .msg h2 { font-size: 18px; } .msg h3 { font-size: 16px; }
  .msg ul, .msg ol { margin: 8px 0 10px; padding-left: 22px; }
  .msg li { margin: 3px 0; }
  .msg code {
    background: var(--bg-hover);
    padding: 2px 6px; border-radius: 5px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }
  .msg pre {
    background: var(--bg-hover);
    padding: 12px 14px; border-radius: 10px;
    overflow-x: auto;
    margin: 10px 0;
  }
  .msg pre code { background: transparent; padding: 0; font-size: 13px; }
  .msg table {
    border-collapse: collapse; margin: 10px 0;
    font-size: 14px; width: 100%;
  }
  .msg th, .msg td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  .msg th { font-weight: 600; background: var(--bg-soft); }
  .msg a { color: var(--accent); text-decoration: none; }
  .msg a:hover { text-decoration: underline; }
  .msg strong { font-weight: 600; }
  .msg hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }

  /* Typing dots */
  .typing {
    display: inline-flex; gap: 4px; padding: 8px 0;
  }
  .typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--text-mute);
    animation: typing-bounce 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes typing-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: .4; }
    40% { transform: translateY(-5px); opacity: 1; }
  }

  /* Composer */
  .composer-wrap {
    position: absolute;
    left: 260px; right: 0; bottom: 0;
    padding: 14px 24px 20px;
    background: var(--bg);
    pointer-events: none;
  }
  .composer-wrap::before {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: 100%;
    height: 28px;
    background: linear-gradient(to bottom, transparent, var(--bg));
    pointer-events: none;
  }
  @media (max-width: 768px) { .composer-wrap { left: 0; } }
  .composer {
    max-width: 760px; margin: 0 auto;
    pointer-events: auto;
  }
  .quick-row {
    display: flex; gap: 8px;
    margin: 0 4px 10px;
    overflow-x: auto;
    scrollbar-width: none;
    padding-bottom: 2px;
  }
  .quick-row::-webkit-scrollbar { display: none; }
  .quick-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 13px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 13px; font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    transition: background .15s ease, border-color .15s ease, transform .12s ease, box-shadow .15s ease;
  }
  .quick-chip:hover {
    background: var(--bg-hover);
    border-color: var(--border-strong);
    transform: translateY(-1px);
    box-shadow: var(--shadow);
  }
  .quick-chip:active { transform: translateY(0); }
  .quick-chip .emoji { font-size: 14px; line-height: 1; }
  .composer-shell {
    display: flex; align-items: flex-end; gap: 8px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 8px 8px 8px 18px;
    box-shadow: var(--shadow);
    transition: border-color .15s ease, box-shadow .15s ease;
  }
  .composer-shell:focus-within {
    border-color: var(--border-strong);
    box-shadow: var(--shadow-strong);
  }
  .composer textarea {
    flex: 1;
    border: none; outline: none; resize: none;
    background: transparent;
    color: var(--text);
    font-size: 15px; line-height: 1.5;
    padding: 8px 0;
    max-height: 200px;
    min-height: 24px;
  }
  .composer textarea::placeholder { color: var(--text-mute); }
  .send-btn {
    width: 34px; height: 34px; border-radius: 50%;
    background: var(--text);
    color: var(--bg);
    display: flex; align-items: center; justify-content: center;
    transition: opacity .15s ease, transform .15s ease;
  }
  .send-btn:hover:not(:disabled) { opacity: .85; transform: translateY(-1px); }
  .send-btn:disabled { background: var(--border-strong); color: var(--bg); cursor: not-allowed; opacity: 1; }
  .send-btn .icon { width: 16px; height: 16px; }
  .composer-hint { text-align: center; color: var(--text-mute); font-size: 11.5px; margin-top: 8px; }

  /* Overlays */
  .overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,.4);
    display: none; align-items: center; justify-content: center;
    z-index: 1000;
    animation: overlay-in .2s ease;
  }
  .overlay.active { display: flex; }
  @keyframes overlay-in { from { opacity: 0; } to { opacity: 1; } }
  .modal {
    background: var(--bg);
    border-radius: 18px;
    width: 90vw; max-width: 420px;
    max-height: 80vh;
    box-shadow: var(--shadow-strong);
    display: flex; flex-direction: column;
    overflow: hidden;
    animation: modal-in .25s cubic-bezier(.2,.8,.3,1);
  }
  @keyframes modal-in { from { opacity: 0; transform: translateY(8px) scale(.98); } to { opacity: 1; transform: none; } }
  .modal-head {
    padding: 18px 22px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .modal-title { font-size: 16px; font-weight: 600; }
  .modal-body { padding: 18px 22px; overflow-y: auto; flex: 1; }
  .modal-foot { padding: 14px 22px; border-top: 1px solid var(--border); display: flex; gap: 8px; justify-content: flex-end; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-size: 13px; font-weight: 500; color: var(--text-soft); }
  .field input {
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 15px;
    background: var(--bg);
    color: var(--text);
    outline: none;
    transition: border-color .15s ease;
  }
  .field input:focus { border-color: var(--text); }
  .btn {
    padding: 9px 18px;
    border-radius: 10px;
    font-size: 14px; font-weight: 500;
    transition: background .15s ease, opacity .15s ease;
  }
  .btn-primary { background: var(--text); color: var(--bg); }
  .btn-primary:hover { opacity: .85; }
  .btn-ghost { color: var(--text-soft); }
  .btn-ghost:hover { background: var(--bg-hover); color: var(--text); }
  .btn-danger { color: #e5484d; }
  .btn-danger:hover { background: rgba(229,72,77,.08); }

  .mem-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
  }
  .mem-item:last-child { border-bottom: none; }
  .mem-item span { flex: 1; line-height: 1.5; }
  .mem-empty { color: var(--text-mute); text-align: center; padding: 24px 0; font-size: 14px; }

  /* Tool indicator chip on first assistant render */
  .meta-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11.5px; color: var(--text-mute);
    background: var(--bg-soft);
    border: 1px solid var(--border);
    padding: 2px 8px; border-radius: 8px;
    margin-bottom: 6px;
  }
</style>
</head>
<body>

<div class="app" id="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="brand">
      <img class="mark" src="/assets/logo.jpeg" alt="ShyftLabs" />
      <div class="text">
        <div class="wordmark">Continuum</div>
        <div class="by">by <em>ShyftLabs</em></div>
      </div>
    </div>

    <button class="side-btn primary" onclick="startNewChat()">
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
      <span>New chat</span>
    </button>

    <div class="mode-card">
      <div class="label"><span class="dot"></span> Continuum</div>
      <div class="title">Auto routing · Modest</div>
      <div class="sub">Continuum picks the best model per prompt.</div>
    </div>

    <div class="stats">
      <div class="stat-box"><div class="k">Messages</div><div class="v" id="stat-msgs">0</div></div>
      <div class="stat-box"><div class="k">Avg · ms</div><div class="v" id="stat-avg">—</div></div>
      <div class="stat-box full"><div class="k">Last model</div><div class="v" id="stat-model" style="font-size:13px;font-weight:600;">—</div></div>
    </div>

    <div class="side-section-label">Recent</div>
    <div class="recents" id="recents"><div style="font-size:12.5px;color:var(--text-mute);padding:6px 10px;">No conversations yet</div></div>

    <div class="side-footer">
      <button class="side-btn" onclick="openMemoryPanel()" id="mem-btn" style="display:none">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
        <span>Memories</span>
      </button>
      <button class="side-btn" onclick="clearMemory()" id="clr-btn" style="display:none">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"></path></svg>
        <span>Clear memories</span>
      </button>
      <div class="user-chip" id="user-chip" onclick="changeUser()" style="display:none">
        <div class="user-avatar" id="user-avatar">A</div>
        <div class="user-name" id="user-name">Guest</div>
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-mute)"><polyline points="9 18 15 12 9 6"></polyline></svg>
      </div>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <div class="top-bar">
      <div class="top-title" id="top-title">Pet Shop · powered by Continuum</div>
      <div class="top-actions">
        <div class="pill" id="pill-status" title="Connection status"><span class="dot"></span><strong>Live</strong></div>
        <div class="pill violet" id="pill-latency" title="Latency of the last response" style="display:none"><span class="dot"></span><strong id="pill-latency-v">—</strong> ms</div>
        <button class="icon-btn" title="Toggle theme" onclick="cycleTheme()">
          <svg class="icon" id="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        </button>
        <button class="icon-btn" title="New chat" onclick="startNewChat()">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
        </button>
      </div>
    </div>

    <div class="chat-scroll" id="chat-scroll">
      <div class="chat-inner" id="chat"></div>
    </div>

    <div class="composer-wrap">
      <div class="composer">
        <div class="quick-row" id="quick-row">
          <button class="quick-chip" onclick="send('Show me dog toys')"><span class="emoji">🦴</span> Dog toys</button>
          <button class="quick-chip" onclick="send('Recommend something for my kitten')"><span class="emoji">🐱</span> For my kitten</button>
          <button class="quick-chip" onclick="send(&quot;What's in my cart?&quot;)"><span class="emoji">🛒</span> View cart</button>
          <button class="quick-chip" onclick="send('Checkout please')"><span class="emoji">✅</span> Checkout</button>
          <button class="quick-chip" onclick="send('What dog leashes do you have?')"><span class="emoji">🐕</span> Leashes</button>
          <button class="quick-chip" onclick="send('I need shampoo for my dog')"><span class="emoji">🧴</span> Grooming</button>
        </div>
        <div class="composer-shell">
          <textarea id="input" rows="1" placeholder="Message Continuum…" autofocus></textarea>
          <button id="send" class="send-btn" onclick="sendClick()" disabled aria-label="Send">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
          </button>
        </div>
        <div class="composer-hint">Powered by Continuum</div>
      </div>
    </div>
  </main>
</div>

<!-- Login overlay -->
<div class="overlay active" id="login-overlay">
  <div class="modal" style="max-width: 380px;">
    <div class="modal-head"><div class="modal-title">Welcome to Continuum</div></div>
    <div class="modal-body">
      <p style="color:var(--text-soft); font-size:14px; margin-bottom:18px;">
        Enter a user ID to start. Reuse the same ID later — the assistant will remember your pets and preferences.
      </p>
      <div class="field">
        <label for="login-input">User ID</label>
        <input id="login-input" type="text" placeholder="e.g. alice, bob, user_123" onkeydown="if(event.key==='Enter') doLogin()" autofocus />
      </div>
    </div>
    <div class="modal-foot">
      <button class="btn btn-primary" onclick="doLogin()">Start chatting</button>
    </div>
  </div>
</div>

<!-- Memory overlay -->
<div class="overlay" id="memory-overlay">
  <div class="modal" style="max-width: 520px;">
    <div class="modal-head">
      <div class="modal-title">Long-term memories</div>
      <button class="icon-btn" onclick="closeMemoryPanel()" aria-label="Close">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
    </div>
    <div class="modal-body" id="memory-list"></div>
  </div>
</div>

<script>
let currentUserId = null;
let currentConversationId = null;
let session = { msgs: 0, totalMs: 0, lastModel: null };

// Markdown
marked.setOptions({ breaks: true, gfm: true });
function renderMarkdown(text) {
  const html = marked.parse(text || '');
  return DOMPurify.sanitize(html);
}

function initials(s) { return (s || 'U').trim().charAt(0).toUpperCase(); }

function timeNow() {
  return new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}

function fmtMs(ms) {
  if (ms < 1000) return ms + ' ms';
  return (ms / 1000).toFixed(1) + ' s';
}

/* ----- Theme ----- */
function applyTheme(mode) {
  if (mode === 'system') { document.documentElement.removeAttribute('data-theme'); }
  else document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('paws-theme', mode);
  const t = document.getElementById('theme-icon');
  if (!t) return;
  // sun / moon / monitor
  if (mode === 'light') t.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
  else if (mode === 'dark') t.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
  else t.innerHTML = '<rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line>';
}
function cycleTheme() {
  const cur = localStorage.getItem('paws-theme') || 'system';
  const next = cur === 'system' ? 'light' : cur === 'light' ? 'dark' : 'system';
  applyTheme(next);
}
applyTheme(localStorage.getItem('paws-theme') || 'system');

/* ----- Session stats ----- */
function updateStats(durationMs, model) {
  session.msgs++;
  session.totalMs += durationMs;
  if (model) session.lastModel = model;
  document.getElementById('stat-msgs').textContent = session.msgs;
  document.getElementById('stat-avg').textContent = session.msgs ? Math.round(session.totalMs / session.msgs) : '—';
  document.getElementById('stat-model').textContent = session.lastModel || 'auto · modest';
  const pl = document.getElementById('pill-latency');
  pl.style.display = '';
  document.getElementById('pill-latency-v').textContent = durationMs;
}
function resetStats() {
  session = { msgs: 0, totalMs: 0, lastModel: null };
  document.getElementById('stat-msgs').textContent = '0';
  document.getElementById('stat-avg').textContent = '—';
  document.getElementById('stat-model').textContent = '—';
  document.getElementById('pill-latency').style.display = 'none';
}

/* ----- Recents + chat persistence (localStorage only, no backend) ----- */
function recentsKey() { return currentUserId ? `paws-recents-${currentUserId}` : null; }
function chatKey(convId) {
  if (!currentUserId || !convId) return null;
  return `paws-chat-${currentUserId}-${convId}`;
}
function loadRecents() {
  const k = recentsKey();
  if (!k) return [];
  try { return JSON.parse(localStorage.getItem(k) || '[]'); } catch (e) { return []; }
}
function saveRecents(list) {
  const k = recentsKey();
  if (!k) return;
  localStorage.setItem(k, JSON.stringify(list.slice(0, 20)));
}
function loadChat(convId) {
  const k = chatKey(convId);
  if (!k) return [];
  try { return JSON.parse(localStorage.getItem(k) || '[]'); } catch (e) { return []; }
}
function persistMessage(role, text, extra) {
  const k = chatKey(currentConversationId);
  if (!k) return;
  const msgs = loadChat(currentConversationId);
  msgs.push({ role, text, ts: Date.now(), ...(extra || {}) });
  localStorage.setItem(k, JSON.stringify(msgs));
}
function ensureConvId() {
  if (!currentConversationId) {
    currentConversationId = (window.crypto && window.crypto.randomUUID)
      ? window.crypto.randomUUID()
      : 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2,9);
  }
}
function upsertRecent(title) {
  if (!currentConversationId || !currentUserId) return;
  const list = loadRecents();
  const idx = list.findIndex(r => r.id === currentConversationId);
  if (idx >= 0) {
    // Keep original title — only bump timestamp + move to top.
    const entry = list.splice(idx, 1)[0];
    entry.ts = Date.now();
    list.unshift(entry);
  } else {
    list.unshift({ id: currentConversationId, title, ts: Date.now() });
  }
  saveRecents(list);
  renderRecents();
}
function renderRecents() {
  const root = document.getElementById('recents');
  if (!root) return;
  const list = loadRecents();
  if (!list.length) { root.innerHTML = '<div style="font-size:12.5px;color:var(--text-mute);padding:6px 10px;">No conversations yet</div>'; return; }
  root.innerHTML = list.map(r => `
    <button class="recent-item ${r.id === currentConversationId ? 'active' : ''}" onclick="openRecent('${r.id}', event)">
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-mute)"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
      <span class="ttl">${escapeHtml(r.title)}</span>
      <span class="delx" onclick="deleteRecent('${r.id}', event)" title="Delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:12px;height:12px"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </span>
    </button>
  `).join('');
}
function openRecent(id, ev) {
  if (ev) ev.stopPropagation();
  currentConversationId = id;
  document.getElementById('chat').innerHTML = '';
  resetStats();
  const msgs = loadChat(id);
  if (!msgs.length) {
    renderEmpty();
  } else {
    for (const m of msgs) {
      if (m.role === 'user') {
        appendUser(m.text, { time: m.ts });
      } else {
        appendAssistant(m.text, { durationMs: m.durationMs, model: m.model, time: m.ts });
        if (m.durationMs) { session.msgs++; session.totalMs += m.durationMs; if (m.model) session.lastModel = m.model; }
      }
    }
    // Refresh sidebar stats to reflect restored history
    document.getElementById('stat-msgs').textContent = session.msgs;
    document.getElementById('stat-avg').textContent = session.msgs ? Math.round(session.totalMs / session.msgs) : '—';
    document.getElementById('stat-model').textContent = session.lastModel || 'auto · modest';
  }
  renderRecents();
}
function deleteRecent(id, ev) {
  if (ev) { ev.stopPropagation(); ev.preventDefault(); }
  const list = loadRecents().filter(r => r.id !== id);
  saveRecents(list);
  const k = chatKey(id);
  if (k) localStorage.removeItem(k);
  if (id === currentConversationId) startNewChat();
  else renderRecents();
}

function doLogin() {
  const v = document.getElementById('login-input').value.trim();
  if (!v) return;
  currentUserId = v;
  document.getElementById('user-name').textContent = currentUserId;
  document.getElementById('user-avatar').textContent = initials(currentUserId);
  document.getElementById('user-chip').style.display = '';
  document.getElementById('mem-btn').style.display = '';
  document.getElementById('clr-btn').style.display = '';
  document.getElementById('login-overlay').classList.remove('active');
  renderRecents();
  startNewChat();
  setTimeout(() => document.getElementById('input').focus(), 100);
}

function changeUser() {
  currentUserId = null;
  document.getElementById('user-chip').style.display = 'none';
  document.getElementById('mem-btn').style.display = 'none';
  document.getElementById('clr-btn').style.display = 'none';
  document.getElementById('login-overlay').classList.add('active');
  document.getElementById('login-input').value = '';
  document.getElementById('login-input').focus();
}

async function openMemoryPanel() {
  if (!currentUserId) return;
  document.getElementById('memory-overlay').classList.add('active');
  const list = document.getElementById('memory-list');
  list.innerHTML = '<div class="mem-empty">Loading…</div>';
  const res = await fetch(`/memory/list?user_id=${encodeURIComponent(currentUserId)}`);
  const data = await res.json();
  if (!data.success) { list.innerHTML = `<div class="mem-empty" style="color:#e5484d">${data.error}</div>`; return; }
  if (!data.memories.length) { list.innerHTML = '<div class="mem-empty">No memories yet. Keep chatting and I\\'ll remember the important stuff.</div>'; return; }
  list.innerHTML = data.memories.map(m => `
    <div class="mem-item">
      <span>${escapeHtml(m.text)}</span>
      <button class="btn btn-ghost btn-danger" style="padding:4px 10px;font-size:13px" onclick="deleteMemory('${m.id}', this)">Delete</button>
    </div>
  `).join('');
}
function closeMemoryPanel() { document.getElementById('memory-overlay').classList.remove('active'); }

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function deleteMemory(memoryId, btn) {
  btn.disabled = true; btn.textContent = '…';
  const res = await fetch('/memory/delete', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({memory_id: memoryId}) });
  const data = await res.json();
  if (data.success) {
    btn.closest('.mem-item').remove();
    const list = document.getElementById('memory-list');
    if (!list.children.length) list.innerHTML = '<div class="mem-empty">No memories yet.</div>';
  } else {
    btn.disabled = false; btn.textContent = 'Delete';
    alert('Failed: ' + data.error);
  }
}

async function clearMemory() {
  if (!currentUserId) return;
  if (!confirm(`Clear all long-term memories for '${currentUserId}'?`)) return;
  const res = await fetch('/memory/clear', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_id: currentUserId}) });
  const data = await res.json();
  if (data.success) appendAssistant('Memory cleared. I no longer remember anything about you.');
  else appendAssistant('Failed to clear memory: ' + data.error);
}

function startNewChat() {
  // Lazy: don't allocate a conv_id until the user actually sends a message.
  currentConversationId = null;
  document.getElementById('chat').innerHTML = '';
  resetStats();
  renderEmpty();
  renderRecents();
}

function renderEmpty() {
  const chat = document.getElementById('chat');
  chat.innerHTML = `
    <div class="empty">
      <div class="empty-logo">🐾</div>
      <div class="empty-title">How can I help you today?</div>
      <div class="empty-sub">Pick a quick action below or ask anything about our pet shop.</div>
      <div class="feature-row">
        <div class="feature smart">
          <div class="ficon">⚡</div>
          <h4>Continuum</h4>
          <p>Every prompt is classified and routed to the optimal model.</p>
        </div>
        <div class="feature mem">
          <div class="ficon">🧠</div>
          <h4>Long-term memory</h4>
          <p>I remember your pets and preferences across sessions.</p>
        </div>
        <div class="feature tools">
          <div class="ficon">🛠️</div>
          <h4>Live tools</h4>
          <p>Search, cart, and checkout through the MCP shop server.</p>
        </div>
      </div>
    </div>
  `;
}

function clearEmpty() {
  const empty = document.querySelector('#chat .empty');
  if (empty) empty.remove();
}

function send(text) {
  document.getElementById('input').value = text;
  autoGrow();
  document.getElementById('send').disabled = !text.trim();
  sendClick();
}

function fmtTimeFromTs(ts) {
  return new Date(ts || Date.now()).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}

function appendUser(text, opts) {
  opts = opts || {};
  clearEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="user-stack">
      <div class="msg user">${escapeHtml(text)}</div>
      <div class="msg-meta"><span>${fmtTimeFromTs(opts.time)}</span></div>
    </div>
    <div class="avatar user">${initials(currentUserId)}</div>
  `;
  document.getElementById('chat').appendChild(row);
  scrollToEnd();
  return row;
}

function appendAssistant(text, opts) {
  opts = opts || {};
  clearEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  const metaBits = [];
  metaBits.push(`<span>${fmtTimeFromTs(opts.time)}</span>`);
  if (opts.durationMs) metaBits.push(`<span class="meta-dot"></span><span>${fmtMs(opts.durationMs)}</span>`);
  const metaHTML = `<div class="msg-meta">${metaBits.join('')}</div>`;
  row.innerHTML = `
    <div class="avatar assistant">🐾</div>
    <div class="asst-stack">
      <div class="msg assistant">${renderMarkdown(text)}</div>
      ${metaHTML}
      <button class="copy-btn" title="Copy" onclick="copyMessage(this)">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      </button>
    </div>
  `;
  row.dataset.raw = text;
  document.getElementById('chat').appendChild(row);
  scrollToEnd();
  return row;
}

function copyMessage(btn) {
  const row = btn.closest('.msg-row');
  if (!row) return;
  const raw = row.dataset.raw || row.innerText;
  navigator.clipboard.writeText(raw).then(() => {
    const old = btn.innerHTML;
    btn.innerHTML = '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    setTimeout(() => { btn.innerHTML = old; }, 1200);
  });
}

function appendTyping() {
  clearEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = `
    <div class="avatar assistant">🐾</div>
    <div class="msg assistant"><div class="typing"><span></span><span></span><span></span></div></div>
  `;
  document.getElementById('chat').appendChild(row);
  scrollToEnd();
  return row;
}

function scrollToEnd() {
  const scroll = document.getElementById('chat-scroll');
  scroll.scrollTop = scroll.scrollHeight;
}

async function sendClick() {
  if (!currentUserId) { changeUser(); return; }
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  autoGrow();
  const send = document.getElementById('send');
  send.disabled = true;

  ensureConvId();  // first message materialises the chat
  appendUser(msg);
  persistMessage('user', msg);
  upsertRecent(msg.length > 36 ? msg.slice(0, 36).trim() + '…' : msg);

  const typing = appendTyping();
  const t0 = performance.now();
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, user_id: currentUserId, conversation_id: currentConversationId})
    });
    const data = await res.json();
    const clientMs = Math.round(performance.now() - t0);
    const serverMs = data.duration_ms || clientMs;
    typing.remove();
    const reply = data.response || data.error || 'No response';
    appendAssistant(reply, { durationMs: serverMs });
    persistMessage('assistant', reply, { durationMs: serverMs });
    updateStats(serverMs, data.model || null);
  } catch(e) {
    typing.remove();
    appendAssistant('**Error:** ' + e.message);
  }
  input.focus();
}

// Auto-grow textarea + Enter to send
const input = document.getElementById('input');
function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  document.getElementById('send').disabled = !input.value.trim();
}
input.addEventListener('input', autoGrow);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (input.value.trim()) sendClick();
  }
});

// Initial empty state shown after login
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Gateway Shop Web UI at http://localhost:8081")
    print("Make sure MCP server is running:  python server.py")
    print("Make sure Smart Gateway is running: http://localhost:8787/v1")
    uvicorn.run(app, host="0.0.0.0", port=8081)
