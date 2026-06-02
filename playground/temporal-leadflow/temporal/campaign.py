"""
Build the LeadFlow WorkflowInput — the declarative step list fed to AgentWorkflow.

Steps:
  1. ParallelStep  — 3 scrapers run concurrently
  2. AgentStep     — scoring-agent (output_schema=RankedLeadList)
  3. ApprovalStep  — blocks until UI sends /approve or /reject
  4. AgentStep     — voice-agent (mock Twilio + crm-lookup handoff)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LeadFlowConfig, default_config

from orchestrator.temporal.types import (
    AgentStep,
    ApprovalStep,
    ParallelStep,
    WorkflowInput,
)


def build_workflow_input(
    niche: str,
    location: str,
    campaign_id: str,
    config: LeadFlowConfig | None = None,
) -> WorkflowInput:
    cfg = config or default_config
    initial = (
        f"Find {cfg.leads_per_source} leads per source for: {niche} in {location}. "
        "Return realistic fictional business data as instructed."
    )

    scrape_step = ParallelStep(
        agents=[
            AgentStep(agent_name="google-maps-agent", timeout=120, retries=2),
            AgentStep(agent_name="linkedin-agent", timeout=120, retries=2),
            AgentStep(agent_name="web-agent", timeout=120, retries=2),
        ],
        merge_strategy="concatenate",
    )

    score_step = AgentStep(
        agent_name="scoring-agent",
        timeout=180,
        retries=2,
        metadata={"campaign_id": campaign_id, "niche": niche, "location": location},
    )

    approval_step = ApprovalStep(
        description=(
            f"Review the scored lead list for '{niche}' in '{location}'. "
            "Approve to begin voice outreach, or reject to cancel the campaign."
        ),
        timeout=cfg.approval_timeout,
    )

    voice_step = AgentStep(
        agent_name="voice-agent",
        timeout=300,
        retries=1,
        metadata={"campaign_id": campaign_id},
    )

    return WorkflowInput(
        steps=[
            scrape_step.model_dump(),
            score_step.model_dump(),
            approval_step.model_dump(),
            voice_step.model_dump(),
        ],
        initial_input=initial,
        session_id=campaign_id,
        user_id="leadflow",
        metadata={"niche": niche, "location": location, "campaign_id": campaign_id},
    )
