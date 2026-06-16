"""
Decision-Trace GlassBox configuration (financial-close domain).

Recreated on top of local/glassbox: the same deterministic month-end close,
expressed across all 9 multi-agent patterns (+ handoff), with the materiality
threshold as the universal fork lever. Loads the project-root .env and turns the
Decision Trace feature ON. Import before `continuum.config` is first used.
"""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=True)

_file_env = dotenv_values(_ENV_PATH)
for _var in (
    "SMART_GATEWAY_URL",
    "SMART_GATEWAY_API_KEY",
    "EMBEDDER_API_BASE",
    "EMBEDDER_API_KEY",
):
    if _var not in _file_env:
        os.environ.pop(_var, None)

os.environ.setdefault("DECISION_TRACE_ENABLED", "true")
os.environ.setdefault("DECISION_TRACE_DETAIL", "full")
os.environ.setdefault("DECISION_TRACE_STORE", "redis")
os.environ.setdefault("DECISION_TRACE_CHECKPOINT", "true")

from dataclasses import dataclass


@dataclass
class GlassboxConfig:
    mcp_url: str = "http://localhost:8896/mcp"
    mcp_timeout: float = 15.0
    # gpt-4o follows the multi-step tool instructions reliably across topologies.
    model: str = "openai/gpt-4o"
    max_turns: int = 12

    decision_trace_detail: str = "full"
    decision_trace_store: str = "redis"
    decision_trace_checkpoint: bool = True

    web_port: int = 8087
    mcp_port: int = 8896


default_config = GlassboxConfig()
