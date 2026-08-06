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
                # Every tool the two servers expose, named individually rather
                # than as "tool:clinic__*": a glob would re-admit whatever a
                # compromised server adds later, which is the hole this closes.
                #
                # The prefix is doing real work here -- both servers expose a
                # tool called `lookup_patient`, so a bare "tool:lookup_patient"
                # would be ambiguous, and (since these are ALLOW rules under a
                # default-deny base) would simply match nothing and leave the
                # agent with no tools at all.
                "tool:clinic__clinic_info",
                "tool:clinic__lookup_patient",
                "tool:clinic__send_referral_email",
                "tool:clinic__web_lookup",
                "tool:pharmacy__lookup_patient",
                "tool:pharmacy__check_interactions",
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
            # MCP tool resources are namespaced: "<server>__<tool>". The servers
            # are named "clinic" and "pharmacy" in agent.py. A bare
            # "tool:send_referral_email" would match nothing here -- and since
            # default_effect is "allow", an unmatched DENY silently stops
            # blocking. See docs/tools.md §6.5.
            #
            # Only clinic tools appear because only the clinic ships an egress
            # path. `pharmacy__check_interactions` stays callable on a tainted
            # run by design: it takes drug names, not a patient id, so it sends
            # nothing out. Denying every tool on a tainted run would be easy and
            # useless -- the point is to deny the ones that leak.
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
    # Two servers, because one cannot show what namespacing is for. They overlap
    # on `lookup_patient` (see pharmacy_server.py), so every name-matched setting
    # in this file has to say which server it means.
    mcp_url: str = "http://localhost:8911/mcp"
    pharmacy_base_url: str = "http://localhost:8912"

    # streamable-http (default) | sse | stdio. One variable read by config (for
    # the connection details) and pharmacy_server.py (for what to serve), so the
    # two ends cannot disagree -- a mismatch is a bare connection failure with
    # nothing in it naming the protocol as the cause.
    #
    # Only the pharmacy. Running the pair on different transports is realistic
    # and it is the point: every F3 mechanism lives on the shared session base,
    # so pins, digests, drift and the gate should be indistinguishable across
    # all three.
    pharmacy_transport: str = os.environ.get("PHARMACY_TRANSPORT", "streamable-http")
    mcp_timeout: float = 10.0

    # The pharmacy requires a bearer token; the clinic does not. That asymmetry
    # is the point: `continuum mcp inspect` sends a bare URL, so it gets a 401
    # from the pharmacy however correct the URL is, and reviewing it means
    # passing the configured server object to review_server(). A fixture, not a
    # credential -- see pharmacy_server.py.
    pharmacy_token: str = os.environ.get("PHARMACY_TOKEN", "demo-pharmacy-token")

    # Where the tool-catalogue digests live. On first connect the descriptions
    # and schemas are recorded here; on every later fetch they are compared, and
    # a change after you approved the server is reported (finding F3). Catches a
    # "rug pull" -- a server edited post-approval. It cannot vouch for a server
    # that shipped poisoned text from the start: nothing changed, so there is
    # nothing to detect. Review with `continuum mcp inspect` before trusting,
    # and rely on the fail-closed policy above to bound what a tool can do.
    # Both files live in tool-trust/ so a re-test is `rm -rf tool-trust`. The
    # runtime's record is a sibling of this path, so pointing the approval into
    # the directory carries the record along with it. Created on first write --
    # save_pins() mkdirs the parent -- so a fresh clone needs no setup step.
    tool_pin_path: str = os.path.join(os.path.dirname(__file__), "tool-trust", "tool-pins.json")

    @property
    def pharmacy_url(self) -> str:
        """Derived, not configured: the path is a property of the transport.

        Two fields that must agree is two fields that can disagree -- and the
        failure would be a connection error naming neither. Meaningless under
        stdio, which has no URL at all.
        """
        return f"{self.pharmacy_base_url}/{'sse' if self.pharmacy_transport == 'sse' else 'mcp'}"

    @property
    def pharmacy_stdio_params(self) -> dict:
        """How to LAUNCH the pharmacy, for the transport that has no address.

        The whole environment is forwarded, not a curated subset: PHARMACY_POISON
        has to reach the child or the poison switch would appear to work and
        change nothing. PHARMACY_TRANSPORT is pinned so the child cannot inherit
        a stale value and start an HTTP server the parent is not talking to.

        No token. A bearer credential guards a network boundary; a subprocess
        has none, and one the parent hands its own child proves nothing the
        launch did not already prove.
        """
        return {
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(__file__), "pharmacy_server.py")],
            "env": {**os.environ, "PHARMACY_TRANSPORT": "stdio"},
        }

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
    # Tool provenance: both lookup_patient tools return records declared to
    # carry PHI -- the clinic's clinical record and the pharmacy's dispensing
    # history are both protected.
    #
    # Written with the NAMESPACED names on purpose. The SDK accepts either
    # spelling and a bare "lookup_patient" would resolve to both tools, which
    # happens to be right here -- but it is right by luck, and the SDK logs a
    # warning saying so, because the same shortcut applied to a tool you did not
    # mean produces a label that blocks work nobody intended to block. Swap this
    # for {"lookup_patient": {PHI}} to see that warning (TESTING_GUIDE.md
    # Layer D).
    tool_data_labels: dict[str, set[str]] = field(
        default_factory=lambda: {
            "clinic__lookup_patient": {PHI},
            "pharmacy__lookup_patient": {PHI},
        }
    )
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
