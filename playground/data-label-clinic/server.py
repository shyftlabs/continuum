"""
Clinic MCP server (FastMCP, streamable-HTTP transport).

Run standalone:  python server.py   (serves http://localhost:8911/mcp)

Tools, by role in the data-label demo:
  * clinic_info(topic)            — BENIGN. Hours/address/departments. No taint.
  * lookup_patient(patient_id)    — PROVENANCE. Returns a record; the agent
                                     declares this tool as PHI (config.py
                                     tool_data_labels), so calling it taints the
                                     run. The tool itself knows nothing about
                                     labels — provenance is declared agent-side.
  * send_referral_email(to, body) — EXFILTRATION. Denied once the run is tainted.
  * web_lookup(query)             — EXFILTRATION. Denied once the run is tainted.

Nothing here imports Continuum security — a tool server is just tools. The
labeling/gating lives entirely in the agent + policy (config.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("data-label-clinic")

# Fake patient records — the "sensitive" data whose provenance taints a run.
_PATIENTS = {
    "P-123": {
        "patient_id": "P-123",
        "name": "Jane Doe",
        "dob": "1986-04-12",
        "ssn": "123-45-6789",
        "diagnosis": "Type 2 diabetes; hypertension",
        "medications": ["metformin", "lisinopril"],
        "last_visit": "2026-05-02",
    },
    "P-456": {
        "patient_id": "P-456",
        "name": "John Roe",
        "dob": "1972-11-30",
        "ssn": "987-65-4321",
        "diagnosis": "Post-op follow-up (knee arthroscopy)",
        "medications": ["ibuprofen"],
        "last_visit": "2026-06-10",
    },
}

_CLINIC_INFO = {
    "hours": "Mon–Fri 8am–6pm, Sat 9am–1pm, closed Sunday.",
    "address": "200 Wellness Way, Springfield.",
    "departments": "General practice, cardiology, endocrinology, physiotherapy.",
    "parking": "Free patient parking in the rear lot off Elm Street.",
}


@mcp.tool()
def clinic_info(topic: str = "hours") -> dict:
    """Answer a general, non-sensitive clinic question (hours, address,
    departments, parking). Carries no patient data."""
    key = topic.strip().lower()
    if key in _CLINIC_INFO:
        return {"topic": key, "answer": _CLINIC_INFO[key]}
    return {
        "topic": key,
        "answer": "I can share hours, address, departments, or parking.",
        "available": sorted(_CLINIC_INFO),
    }


@mcp.tool()
def lookup_patient(patient_id: str) -> dict:
    """Look up a patient's clinical record by ID (e.g. 'P-123').

    Returns protected health information (PHI). The agent declares this tool as
    PHI-bearing, so a successful call taints the run and trips the PHI policies.
    """
    record = _PATIENTS.get(patient_id.strip().upper())
    if not record:
        return {"error": f"No patient found with id {patient_id!r}"}
    return record


@mcp.tool()
def send_referral_email(to: str, body: str) -> dict:
    """Email a referral/summary to an external address. (Exfiltration path —
    a PHI-tainted run is denied this tool by policy before it ever runs.)"""
    # If this body executes, the email "went out". The policy gate prevents the
    # call from reaching here when the run is tainted.
    return {"sent": True, "to": to, "chars": len(body)}


@mcp.tool()
def web_lookup(query: str) -> dict:
    """Look something up on the public web. (Exfiltration path — denied for a
    PHI-tainted run.)"""
    return {"query": query, "result": f"(pretend web result for {query!r})"}


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    print("Clinic MCP server running at http://localhost:8911/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8911)
