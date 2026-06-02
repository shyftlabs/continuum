"""
Voice outreach agents.

voice-agent — multi-turn BaseAgent with mock Twilio tools.
              Mid-call CRM queries hand off to crm_lookup with return_to_parent=True.

crm_lookup  — lightweight agent that interprets CRM tool output and returns a
              structured note back to the voice agent.

Note: requires OpenAI-compatible models. Gemini rejects function_response messages
in handoff history when the target agent has no tools defined.
"""

from __future__ import annotations

from orchestrator import AgentConfig, AgentMemoryConfig, BaseAgent
from orchestrator.agent.types import Handoff

_NO_MEMORY = AgentMemoryConfig(search_memories=False, store_memories=False)


def make_crm_lookup_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="crm_lookup",
        instructions=(
            "You are a CRM assistant. "
            "You receive the raw output of a check_availability tool call. "
            "Parse it and return a one-sentence summary: "
            "the lead's preferred contact window and any prior interaction notes."
        ),
        model=model,
        memory_config=_NO_MEMORY,
        config=AgentConfig(log_to_session=False, session_history_turns=0, max_turns=2),
    )


def make_voice_agent(model: str, tool_executor, tools: list) -> BaseAgent:
    return BaseAgent(
        name="voice-agent",
        instructions=(
            "You are a voice outreach specialist. "
            "You receive a ranked list of business leads to call.\n\n"
            "Process leads ONE AT A TIME. Do NOT batch tool calls across leads. "
            "Complete the full cycle for lead N before starting lead N+1:\n"
            "  1. call check_availability for THIS lead only\n"
            "  2. handoff to crm_lookup (one handoff per lead)\n"
            "  3. call call_lead for THIS lead with goal='book a 15-minute demo call'\n"
            "  4. if the transcript shows NO ANSWER or VOICEMAIL, call leave_voicemail\n"
            "  5. only then move to the next lead\n\n"
            "After ALL leads are processed, output a final summary with one paragraph per lead: "
            "lead name, call outcome, and any booked meeting time."
        ),
        model=model,
        tools=tools,
        tool_executor=tool_executor,
        handoffs=[
            Handoff(
                target_agent="crm_lookup",
                description="Hand off CRM availability data to crm_lookup for interpretation. "
                "Use this after calling check_availability to get a clean summary.",
                return_to_parent=True,
            )
        ],
        memory_config=_NO_MEMORY,
        config=AgentConfig(log_to_session=False, session_history_turns=0, max_turns=20),
        tags=["voice", "outreach", "leadflow"],
    )
