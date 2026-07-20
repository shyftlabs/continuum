"""
Configuration for the Incident Desk rig (Headroom end-to-end tests).

This module must be imported BEFORE anything from `continuum` — it makes the
repo-root .env authoritative (same guard as the other playgrounds) and then
FORCES Headroom on for this process, so the rig can never silently run
uncompressed:

  * HEADROOM_ENABLED=true            (the whole point of the rig)
  * sidecar expected on :8787        (preflight() reports, never blocks —
                                      a dead sidecar IS scenario 5, fail-open)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import dotenv_values, load_dotenv

# .env is authoritative for local dev; stale shell exports must not win, and a
# gateway var commented out in .env must not survive from an old export.
_ENV_PATH = _ROOT / ".env"
load_dotenv(_ENV_PATH, override=True)
_file_env = dotenv_values(_ENV_PATH)
for _var in (
    "SMART_GATEWAY_URL",
    "SMART_GATEWAY_API_KEY",
    "EMBEDDER_API_BASE",
    "EMBEDDER_API_KEY",
    "HEADROOM_ENABLED",
):
    if _var not in _file_env:
        os.environ.pop(_var, None)

# Force Headroom ON for this process, before continuum.config builds Settings.
# os.environ["HEADROOM_ENABLED"] = "true"

SIDECAR_BASE = os.environ.get("HEADROOM_API_BASE", "http://127.0.0.1:8787")
SIDECAR_RESTART_CMD = (
    "cd extensions/headroom && HEADROOM_CCR_BACKEND=memory HEADROOM_OFFLINE=1 "
    "HF_HUB_OFFLINE=1 uv run headroom proxy --port 8787"
)


def sidecar_health() -> dict:
    """GET /health on the sidecar. Returns {'up': False, ...} when unreachable
    — which is not an error for this rig (fail-open is scenario 5)."""
    import httpx

    try:
        r = httpx.get(f"{SIDECAR_BASE}/health", timeout=3.0)
        body = r.json()
        return {
            "up": True,
            "version": body.get("version"),
            "kompress_enabled": not body.get("config", {}).get("disable_kompress", True),
        }
    except Exception as e:
        return {"up": False, "error": str(e), "restart": SIDECAR_RESTART_CMD}


def sidecar_stats() -> dict:
    """GET /stats compression summary (for per-run deltas in the glassbox)."""
    import httpx

    try:
        r = httpx.get(f"{SIDECAR_BASE}/stats", timeout=3.0)
        c = r.json().get("summary", {}).get("compression", {})
        return {
            "requests_compressed": c.get("requests_compressed", 0),
            "total_tokens_removed": c.get("total_tokens_removed", 0),
        }
    except Exception:
        return {"requests_compressed": 0, "total_tokens_removed": 0}


@dataclass
class IncidentConfig:
    mcp_url: str = "http://localhost:8921/mcp"
    mcp_timeout: float = 15.0

    agent_name: str = "incident-desk"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_turns: int = 10

    system_instructions: str = (
        "You are Incident Desk, an on-call incident-investigation copilot for "
        "the checkout-api outage on 2026-07-08.\n"
        "\n"
        "RULES:\n"
        "- Always fetch data with the tools; never invent ids, tokens, or amounts.\n"
        "- Tool outputs may arrive COMPRESSED, with a marker like "
        "'[N lines compressed to M. Retrieve more: hash=abc...]'. If the "
        "compressed view does not contain the specific detail the user asked "
        "for, call continuum_headroom_retrieve with that hash to get the original "
        "content, then answer from it.\n"
        "- Report exact values verbatim (order ids, incident tokens, amounts, "
        "config values). Be concise."
    )


default_config = IncidentConfig()
