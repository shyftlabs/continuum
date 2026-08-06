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

from dataclasses import dataclass, field


@dataclass
class LeadFlowConfig:
    model: str = "gpt-4o-mini"
    max_turns: int = 8
    leads_per_source: int = 5

    # How many ranked leads reach voice outreach. Three scrapers x
    # leads_per_source is more than a demo needs to call, and every extra lead
    # costs the voice agent turns -- see voice_max_turns.
    outreach_leads: int = 5

    @property
    def voice_max_turns(self) -> int:
        """Turn budget for the voice agent, derived from the lead count.

        Per lead the agent spends: check_availability, handoff to crm_lookup,
        call_lead, and -- for the no_answer/voicemail outcomes -- leave_voicemail.
        That is 4, plus one turn for the closing per-lead summary and one of slack.

        Derived rather than hardcoded because the two were previously independent:
        10 leads met a max_turns of 20 and the campaign died mid-lead with
        MaxTurnsExceededError, losing the summary for the calls it had completed.
        """
        return 4 * self.outreach_leads + 2

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "leadflow"
    enable_tracing: bool = True
    approval_timeout: int = 86400  # 24h
    # Who is allowed to approve/reject the lead-review gate. A non-empty list
    # activates approver authorization in AgentWorkflow; only these decided_by
    # values are honored, everyone else is ignored + audited.
    approvers: list[str] = field(default_factory=lambda: ["user"])


default_config = LeadFlowConfig()
