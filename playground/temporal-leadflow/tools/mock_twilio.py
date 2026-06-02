"""
FakeTwilioMCP — in-process mock of a Twilio voice MCP server.

Provides three tools:
  call_lead          — simulate placing a call; returns a brief transcript
  leave_voicemail    — simulate leaving a voicemail; returns confirmation
  check_availability — look up a lead's preferred contact window from mock CRM

Real Twilio: replace this server with a real MCP server URL in temporal/worker.py.
"""

from __future__ import annotations

import json

from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool


class FakeTwilioMCP:
    name = "twilio-mcp"
    context_config = None  # required by ToolExecutor._inject_context_variables

    def __init__(self) -> None:
        self._connected = False
        self._tools = [
            Tool(
                name="call_lead",
                description="Place a simulated outbound call to a lead. Returns a call transcript.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Lead business name"},
                        "phone": {"type": "string", "description": "Phone number to call"},
                        "goal": {
                            "type": "string",
                            "description": "Goal of the call (e.g. book a meeting)",
                        },
                    },
                    "required": ["name", "phone", "goal"],
                },
            ),
            Tool(
                name="leave_voicemail",
                description="Leave a voicemail for a lead who did not answer.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Lead business name"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "message": {"type": "string", "description": "Voicemail message script"},
                    },
                    "required": ["name", "phone", "message"],
                },
            ),
            Tool(
                name="check_availability",
                description="Query the CRM for a lead's preferred contact window or previous interaction notes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Lead business name"},
                    },
                    "required": ["name"],
                },
            ),
        ]

    async def connect(self) -> None:
        self._connected = True

    async def cleanup(self) -> None:
        self._connected = False

    async def list_tools(self, metadata: dict | None = None) -> list[Tool]:
        if not self._connected:
            raise RuntimeError("FakeTwilioMCP not connected")
        return self._tools

    # Deterministic outcome rotation based on lead name hash
    _OUTCOMES = [
        "meeting_booked",
        "no_answer",
        "not_interested",
        "callback_requested",
        "voicemail",
    ]
    _CRM_WINDOWS = [
        ("weekday mornings 9am–11am", "Prefers early calls. Owner is very responsive."),
        (
            "Tuesday/Thursday afternoons 2pm–4pm",
            "Spoke briefly last quarter, showed mild interest.",
        ),
        ("Friday mornings only", "Very busy — keep calls under 3 minutes."),
        ("any weekday", "New lead. No prior contact. Decision maker is the manager."),
        ("Monday mornings 8am–10am", "Previously requested a callback but never followed up."),
    ]

    def _outcome_for(self, name: str) -> str:
        return self._OUTCOMES[hash(name) % len(self._OUTCOMES)]

    def _crm_for(self, name: str) -> tuple[str, str]:
        return self._CRM_WINDOWS[hash(name) % len(self._CRM_WINDOWS)]

    async def call_tool(self, tool_name: str, arguments: dict | None) -> CallToolResult:
        if not self._connected:
            raise RuntimeError("FakeTwilioMCP not connected")
        args = arguments or {}

        if tool_name == "call_lead":
            name = args.get("name", "the business")
            phone = args.get("phone", "unknown")
            goal = args.get("goal", "discuss services")
            outcome = self._outcome_for(name)

            if outcome == "meeting_booked":
                transcript = (
                    f"[CALL LOG] Calling {name} at {phone}\n"
                    f"Ring... Ring...\n"
                    f"Owner: Hello, {name}.\n"
                    f"Agent: Hi! I'm reaching out to {goal}. Do you have 15 minutes this week for a quick demo?\n"
                    f"Owner: Sure, how about Thursday at 2pm?\n"
                    f"Agent: Perfect! I'll send a calendar invite right now.\n"
                    f"[OUTCOME: MEETING BOOKED — Thursday 2pm]"
                )
            elif outcome == "no_answer":
                transcript = (
                    f"[CALL LOG] Calling {name} at {phone}\n"
                    f"Ring... Ring... Ring... Ring...\n"
                    f"[OUTCOME: NO ANSWER — went to voicemail system]"
                )
            elif outcome == "not_interested":
                transcript = (
                    f"[CALL LOG] Calling {name} at {phone}\n"
                    f"Ring...\n"
                    f"Owner: Hello?\n"
                    f"Agent: Hi! I'm reaching out to {goal}.\n"
                    f"Owner: We're not interested in any sales calls, please remove us from your list.\n"
                    f"Agent: Absolutely, sorry to bother you. Have a great day!\n"
                    f"[OUTCOME: NOT INTERESTED — do not contact]"
                )
            elif outcome == "callback_requested":
                transcript = (
                    f"[CALL LOG] Calling {name} at {phone}\n"
                    f"Ring... Ring...\n"
                    f"Receptionist: {name}, how can I help?\n"
                    f"Agent: Hi! I'm reaching out to {goal}. Is the owner available?\n"
                    f"Receptionist: They're in a meeting right now. Can I take a message?\n"
                    f"Agent: Sure — could they call back at their convenience?\n"
                    f"Receptionist: I'll pass that along. Try again Friday morning.\n"
                    f"[OUTCOME: CALLBACK REQUESTED — try Friday morning]"
                )
            else:  # voicemail
                transcript = (
                    f"[CALL LOG] Calling {name} at {phone}\n"
                    f"Ring... Ring... Ring...\n"
                    f"Voicemail: You've reached {name}. Please leave a message.\n"
                    f"Agent: Hi, this is a message for the owner. I'd love to show you how our AI platform "
                    f"can help with {goal}. I'll try again or feel free to reach back.\n"
                    f"[OUTCOME: VOICEMAIL LEFT]"
                )

            return CallToolResult(
                content=[TextContent(type="text", text=transcript)],
                isError=False,
            )

        if tool_name == "leave_voicemail":
            name = args.get("name", "the business")
            message = args.get("message", "Please call us back.")
            result = (
                f"[VOICEMAIL LOG] No answer at {name}.\n"
                f'Voicemail left: "{message}"\n'
                f"[OUTCOME: VOICEMAIL]"
            )
            return CallToolResult(
                content=[TextContent(type="text", text=result)],
                isError=False,
            )

        if tool_name == "check_availability":
            name = args.get("name", "unknown")
            window, notes = self._crm_for(name)
            crm_note = json.dumps(
                {
                    "lead": name,
                    "last_contact": "never" if "New lead" in notes else "3 months ago",
                    "preferred_window": window,
                    "notes": notes,
                }
            )
            return CallToolResult(
                content=[TextContent(type="text", text=crm_note)],
                isError=False,
            )

        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name: str, arguments: dict | None = None) -> GetPromptResult:
        return GetPromptResult(messages=[])
