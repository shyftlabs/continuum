"""
Scoring agent — BaseAgent with output_schema=RankedLeadList.

The LLM returns valid JSON matching the schema; the runner parses it
into a RankedLeadList instance stored in AgentResponse.structured_output.
The UI reads from the schema directly (no text parsing needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import RankedLeadList

from orchestrator import AgentConfig, AgentMemoryConfig, BaseAgent

_NO_MEMORY = AgentMemoryConfig(search_memories=False, store_memories=False)


def make_scoring_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="scoring-agent",
        instructions=(
            "You are a lead scoring specialist. "
            "You receive raw lead data scraped from multiple sources. "
            "Deduplicate businesses that appear under similar names across sources. "
            "Score each unique lead 1-10 based on: "
            "contact info completeness (3pts), business maturity (3pts), "
            "outreach accessibility (2pts), description clarity (2pts). "
            "Rank leads from highest score to lowest. "
            "The 'sources' field lists which scrapers found the business "
            "(google-maps-agent, linkedin-agent, web-agent). "
            "Write a one-sentence outreach_hook for each lead.\n\n"
            "Return ONLY a raw JSON object with NO markdown fences, NO explanation. "
            "Use this exact structure:\n"
            '{{"niche": "<business niche>", "location": "<city>", "total": <count>, '
            '"leads": [{{"rank": 1, "name": "...", "address": "...", "phone": "...", '
            '"website": "...", "description": "...", "score": 8, "score_reason": "...", '
            '"outreach_hook": "...", "sources": ["google-maps-agent"]}}]}}'
        ),
        model=model,
        output_schema=RankedLeadList,
        memory_config=_NO_MEMORY,
        config=AgentConfig(log_to_session=False, session_history_turns=0, max_turns=3),
        tags=["scoring", "leadflow"],
    )
