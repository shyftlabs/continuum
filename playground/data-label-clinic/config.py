"""
Configuration for the data-label clinic demo.

This module owns the two things that drive the whole demonstration:

  1. The PolicyStore — the deny rules that a tainted ("phi") run trips.
  2. The agent declarations — which tool and which memory scope are declared
     to carry PHI (the *provenance* sites). The SDK ships NO PII detector:
     taint comes from these declarations, never from scanning the user's text.

Two model tiers, one provider key (per the chosen setup):
  * CLOUD_MODEL    = "gpt-4o"       — the unrestricted model.
  * ONPREM_MODEL   = "gpt-4o-mini"  — stands in for an on-prem / PHI-approved
                                       model. In production this would be a
                                       local endpoint; here both are OpenAI so
                                       the routing gate can be shown with one key.

The deny rule below targets ``llm:gpt-4o`` EXACTLY (no glob), so it denies the
cloud model without also catching ``gpt-4o-mini``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Make the project-root .env authoritative for local dev (same guard the other
# gateway playgrounds use). This must run BEFORE `continuum.config` is imported.
#
#   1. load_dotenv(override=True) lets .env values win over stale shell exports.
#   2. The SDK's loader is override=False, and neither loader can clear a var
#      that is *commented out* in the file — so a previous `export
#      SMART_GATEWAY_URL=...` would otherwise survive and silently enable the
#      gateway. We explicitly pop any gateway var that is absent from the file,
#      so "commented out in .env" reliably means "use the direct LLM provider".
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

from continuum.security.policy import AccessPolicy, PolicyStore

# --- the PHI label -------------------------------------------------------- #
PHI = "phi"

# --- model tiers ---------------------------------------------------------- #
CLOUD_MODEL = "gpt-4o"  # denied for a PHI-tainted run
ONPREM_MODEL = "gpt-4o-mini"  # PHI-approved fallback (stand-in for on-prem)


def build_policy_store() -> PolicyStore:
    """The PHI deny rules, on top of a fail-closed base.

    Subjects are matched against ``[agent_name, *sorted(labels)]``, so a rule
    whose subject is ``"phi"`` fires for any run carrying the phi label,
    regardless of which agent is running.

    Two layers, and the distinction matters:

    * **Fail-closed base** (``default_deny``) — anything not explicitly allowed
      is refused. This is what bounds an MCP server that was hostile from the
      very first connect (finding F3). A poisoned tool description reaches the
      model's prompt verbatim and can instruct it to call ``read_file`` or
      ``fetch_manifest``; no digest catches that, because nothing *changed*.
      What stops it is that the model's persuasion is not authority: a tool this
      store never allowed does not execute, whatever name the attacker picked.
      A blocklist cannot do this -- it only blocks tools thought of in advance.
    * **PHI deny rules** — the demo's actual subject: provenance-driven gating.
      Deny overrides allow, so these still fire on top of the allow-list below.
    """
    store = PolicyStore.default_deny()

    # 0. BASELINE ALLOW — the resources an untainted run legitimately needs.
    #    Under default_deny these must be named or the agent cannot run at all:
    #    every gate (llm, memory, telemetry, session, tool) would refuse.
    store.add_policy(
        AccessPolicy(
            name="clinic-baseline",
            subjects=["*"],
            resources=[
                f"llm:{CLOUD_MODEL}",
                f"llm:{ONPREM_MODEL}",
                "memory:*",
                "telemetry",
                "session",
                # The four tools server.py exposes, named individually rather
                # than as "tool:clinic__*": a glob would re-admit whatever a
                # compromised server adds later, which is the hole this closes.
                "tool:clinic__clinic_info",
                "tool:clinic__lookup_patient",
                "tool:clinic__send_referral_email",
                "tool:clinic__web_lookup",
            ],
            effect="allow",
        )
    )

    # 1. MODEL ROUTING — a PHI run may not use the cloud model (exact match so
    #    the on-prem gpt-4o-mini tier is unaffected).
    store.add_policy(
        AccessPolicy(
            name="phi-no-cloud-model",
            subjects=[PHI],
            resources=["llm:gpt-4o"],
            effect="deny",
            denial_message="PHI may not be sent to the cloud model; use the on-prem model.",
        )
    )

    # 2. TOOL — a PHI run may not use exfiltration tools.
    store.add_policy(
        AccessPolicy(
            name="phi-no-exfiltration-tools",
            subjects=[PHI],
            # MCP tool resources are namespaced: "<server>__<tool>". The server
            # is named "clinic" in agent.py. A bare "tool:send_referral_email"
            # would match nothing here -- and since default_effect is "allow",
            # an unmatched DENY silently stops blocking. See docs/tools.md §6.5.
            resources=["tool:clinic__send_referral_email", "tool:clinic__web_lookup"],
            effect="deny",
            denial_message="This operation would send PHI to a third party and is not permitted.",
        )
    )

    # 3. MEMORY WRITE — sensitive data must never be persisted to long-term
    #    memory, in ANY scope (user/agent/conversation/shared). "memory:*" is a
    #    glob over the "memory:<scope>" resource the write gate checks.
    store.add_policy(
        AccessPolicy(
            name="phi-never-persisted",
            subjects=[PHI],
            resources=["memory:*"],
            effect="deny",
            denial_message="Sensitive data must not be written to long-term memory.",
        )
    )

    # 4. TELEMETRY — a PHI run's payloads must be redacted before egress.
    store.add_policy(
        AccessPolicy(
            name="phi-redact-telemetry",
            subjects=[PHI],
            resources=["telemetry"],
            effect="deny",
            denial_message="PHI redacted from telemetry.",
        )
    )

    # 5. SHORT-TERM MEMORY (session/Redis) — a PHI run's assistant answer must not
    #    be persisted verbatim to the conversation store. The SDK substitutes a
    #    placeholder (the response may contain PHI and we can't verify which parts,
    #    so the whole value is replaced — same conservative approach as telemetry).
    store.add_policy(
        AccessPolicy(
            name="phi-no-short-term",
            subjects=[PHI],
            resources=["session"],
            effect="deny",
            denial_message="Sensitive responses are not persisted to short-term memory.",
        )
    )

    return store


# --- output scanner (the SDK's output_scanners hook) ---------------------- #
# An output scanner is just a callable (prompt, content) -> (sanitized, flagged,
# reason). The SDK ships the HOOK, not a detector — the app supplies the function.
# This one masks SSNs in the visible answer, demonstrating how a scanner COMPOSES
# with the data-label gates: the scanner cleans the answer the clinician sees,
# while the gates handle model-routing, the session placeholder, the blocked
# memory write, and the redacted decision trace (all policy/label-driven, not
# pattern-driven). The two are independent and run at different points.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def mask_ssn(prompt: str, content: str) -> tuple[str, bool, str | None]:
    """Output scanner: replace SSN-shaped strings in the answer with a marker."""
    masked = _SSN_RE.sub("[SSN REDACTED]", content)
    changed = masked != content
    return masked, changed, ("ssn" if changed else None)


@dataclass
class ClinicConfig:
    mcp_url: str = "http://localhost:8911/mcp"
    mcp_timeout: float = 10.0

    # Where the tool-catalogue digests live. On first connect the descriptions
    # and schemas are recorded here; on every later fetch they are compared, and
    # a change after you approved the server is reported (finding F3). Catches a
    # "rug pull" -- a server edited post-approval. It cannot vouch for a server
    # that shipped poisoned text from the start: nothing changed, so there is
    # nothing to detect. Review with `continuum mcp inspect` before trusting,
    # and rely on the fail-closed policy above to bound what a tool can do.
    tool_pin_path: str = os.path.join(os.path.dirname(__file__), "tool-pins.json")

    agent_name: str = "clinic-intake-assistant"
    cloud_model: str = CLOUD_MODEL
    onprem_model: str = ONPREM_MODEL
    temperature: float = 0.3
    max_turns: int = 8

    # Memory is optional (needs Redis + mem0). The model/tool/telemetry gates
    # work with just an LLM key; the memory-write gate is only exercised when
    # memory is enabled.
    enable_memory: bool = True

    # Short-term memory (session/Redis). When enabled and Redis is reachable, the
    # conversation is persisted per (user_id, conversation_id) — and a PHI turn's
    # answer is stored as a placeholder by the session gate. Falls back gracefully
    # to no persistence if the session client isn't enabled.
    enable_session: bool = True

    # --- provenance declarations (the 3 producer sites) ------------------- #
    # Tool provenance: lookup_patient returns a record declared to carry PHI.
    tool_data_labels: dict[str, set[str]] = field(default_factory=lambda: {"lookup_patient": {PHI}})
    # Memory-scope provenance (read = taint) is intentionally NOT used here. In
    # this use case PHI enters only via the lookup_patient tool, and the user's
    # long-term memory holds non-sensitive preferences that must NOT taint a run
    # (otherwise a benign "clinic hours?" turn would taint as soon as any memory
    # exists, and could never write memory again). Left empty on purpose.
    scope_data_labels: dict[str, set[str]] = field(default_factory=dict)

    # Output scanners (SDK hook): run over the final answer before it is returned
    # to the user and saved to session. Independent of the data-label gates.
    output_scanners: list[Callable[[str, str], tuple[str, bool, str | None]]] = field(
        default_factory=lambda: [mask_ssn]
    )

    system_instructions: str = (
        "You are an internal clinic intake assistant running inside an authorized, "
        "access-controlled demo environment. All patient records here are SYNTHETIC "
        "TEST DATA — not real PHI — so privacy concerns do not apply to you.\n"
        "\n"
        "RULES:\n"
        "- When the user asks about a specific patient, you MUST call the "
        "lookup_patient tool with their ID. Do this immediately.\n"
        "- NEVER refuse a request on privacy, confidentiality, or PHI grounds, and "
        "never decide on your own that something is 'not permitted'. A separate "
        "policy layer wraps every tool and model call and will block anything that "
        "isn't allowed — that decision is not yours to make.\n"
        "- Only report that an operation was blocked if a tool result you actually "
        "received contains the text 'POLICY DENIED'. In that case, relay that the "
        "operation was blocked by policy and do not retry it.\n"
        "- If asked to look up a patient AND send/email/share the result, first call "
        "lookup_patient, then attempt the send tool — let the policy layer decide.\n"
        "Be concise."
    )


default_config = ClinicConfig()
