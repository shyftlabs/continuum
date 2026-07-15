"""
Incident Desk MCP server (FastMCP, streamable-HTTP transport).

Run standalone:  python server.py   (serves http://localhost:8921/mcp)

Each tool exists to emit ONE payload shape that exercises a specific Headroom
path — see data.py's header for the size/threshold tuning:

  * fetch_logs(service)        — ~4,000 plain-text log lines → LogCompressor
                                 (lossy) + a CCR retrieve marker. The incident
                                 token needle is buried mid-file.
  * query_orders_db(status)    — 43 uniform JSON rows → SmartCrusher lossless
                                 schema+CSV reformat.
  * search_runbooks(query)     — 20 vector-search-shaped results →
                                 SearchCompressor (lossy) + CCR.
  * read(path)                 — service config file. The name `read` is on
                                 Headroom's DEFAULT_EXCLUDE_TOOLS ("disk is the
                                 source of truth") → passes through UNTOUCHED.
  * write(path, content)       — edits the same config file. Named `write` so
                                 Headroom's ReadLifecycleManager recognizes it
                                 as a mutation: a prior read(path) of the SAME
                                 file becomes STALE and is replaced with a
                                 marker + CCR hash (scenario 11, file-read
                                 lifecycle compression).
  * fetch_postmortem(id)       — ~6,000 words of prose → Kompress candidate
                                 (scenario 9; passes through unless the ML
                                 model is installed AND warm).

Nothing here imports Continuum or Headroom — a tool server is just tools.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from data import (
    SERVICE_YAML,
    format_runbook_results,
    generate_failed_orders,
    generate_logs,
    generate_postmortem,
)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("incident-desk")

# Mutable copy of the config "file" on disk. read() returns the current state;
# write() edits it. This must be a real read→write→read-consistent world: the
# STALE marker tells the model to "re-read for current content", so if a write
# did NOT change what a re-read returns, the model loops forever trying to
# reconcile the value it wrote against the value it keeps reading back.
_service_yaml = SERVICE_YAML


def _apply_config_edit(current: str, content: str) -> str:
    """Best-effort edit of the in-memory config so re-reads are consistent.

    The model may pass the full updated YAML, a fragment, or just
    'pool_max_size: 50'. If we can find a pool_max_size in the incoming
    content, patch that single line in the current file (keeps it full-size,
    so the STALE compression still has a >512-byte read to crush). Otherwise,
    if the content looks like a whole config, replace wholesale.
    """
    m = re.search(r"pool_max_size\s*:\s*(\d+)", content)
    if m:
        return re.sub(r"(pool_max_size\s*:\s*)\d+", rf"\g<1>{m.group(1)}", current)
    if "service:" in content and "database:" in content:
        return content
    return current  # nothing recognizable — leave the file as-is


@mcp.tool()
def fetch_logs(service: str = "checkout-api") -> str:
    """Fetch the application logs for a service during the incident window.

    Valid services: 'checkout-api', 'payments-svc'. Returns the raw log text
    (~4,000 lines).
    """
    service = service.strip().lower()
    if service not in ("checkout-api", "payments-svc"):
        return f"Unknown service {service!r}. Valid: checkout-api, payments-svc."
    return generate_logs(service)


@mcp.tool()
def query_orders_db(status: str = "failed") -> dict:
    """Query the orders database for orders in the incident window, filtered
    by status. Currently only status='failed' returns rows."""
    if status.strip().lower() != "failed":
        return {"count": 0, "orders": [], "note": "only status='failed' has rows in this window"}
    orders = generate_failed_orders()
    return {"count": len(orders), "status_filter": "failed", "orders": orders}


@mcp.tool()
def search_runbooks(query: str) -> str:
    """Semantic search over the internal runbook library. Returns the top 20
    results with relevance scores and snippets (text format)."""
    # Text, not JSON — Headroom's search path compresses grep-style text;
    # text-heavy JSON arrays pass through uncompressed (see data.py).
    return format_runbook_results(query)


@mcp.tool()
def read(path: str = "service.yaml") -> str:
    """Read a configuration file from the service repo. Available: service.yaml."""
    if path.strip().lstrip("./") != "service.yaml":
        return f"File not found: {path!r}. Available: service.yaml"
    return _service_yaml


@mcp.tool()
def write(path: str = "service.yaml", content: str = "") -> str:
    """Write an updated configuration file back to the service repo. Available:
    service.yaml. Use after read() to apply a config change (e.g. raise the DB
    connection-pool size to remediate the incident)."""
    global _service_yaml
    if path.strip().lstrip("./") != "service.yaml":
        return f"File not found: {path!r}. Available: service.yaml"
    _service_yaml = _apply_config_edit(_service_yaml, content)
    m = re.search(r"pool_max_size\s*:\s*(\d+)", _service_yaml)
    now = f" pool_max_size is now {m.group(1)}." if m else ""
    return f"Wrote {len(content)} bytes to {path}. Change applied.{now}"


@mcp.tool()
def fetch_postmortem(incident_id: str = "INC-2417") -> str:
    """Fetch the full postmortem document for an incident (prose, ~6k words).
    Available: INC-2417."""
    if incident_id.strip().upper() != "INC-2417":
        return f"No postmortem found for {incident_id!r}. Available: INC-2417"
    return generate_postmortem()


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    print("Incident Desk MCP server running at http://localhost:8921/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8921, log_level="warning")
