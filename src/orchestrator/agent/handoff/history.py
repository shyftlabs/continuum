"""
History summarization for handoffs.

Provides utilities for summarizing conversation history when
transferring control between agents (inspired by OpenAI SDK).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from orchestrator.agent.types import HistorySummarizationMode
from orchestrator.logging import get_logger

logger = get_logger(__name__)


# Conversation history markers (OpenAI SDK style)
_DEFAULT_HISTORY_START = "<CONVERSATION HISTORY>"
_DEFAULT_HISTORY_END = "</CONVERSATION HISTORY>"
_history_start = _DEFAULT_HISTORY_START
_history_end = _DEFAULT_HISTORY_END


def set_history_markers(start: str | None = None, end: str | None = None) -> None:
    """
    Set custom markers for conversation history in handoff summaries.

    Args:
        start: Start marker (None to keep current)
        end: End marker (None to keep current)
    """
    global _history_start, _history_end
    if start is not None:
        _history_start = start
    if end is not None:
        _history_end = end


def reset_history_markers() -> None:
    """Reset history markers to defaults."""
    global _history_start, _history_end
    _history_start = _DEFAULT_HISTORY_START
    _history_end = _DEFAULT_HISTORY_END


def get_history_markers() -> tuple[str, str]:
    """Get current history markers."""
    return (_history_start, _history_end)


def _find_turn_boundary(messages: list[dict[str, Any]], n_turns: int) -> int:
    """Return the start index of the last n_turns in messages.

    A turn starts at each user message. All messages within a turn
    (tool calls, tool results, assistant responses) are included.
    """
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= n_turns:
        return 0
    return user_indices[-n_turns]


@dataclass
class HistorySummarizer:
    """
    Configuration for history summarization.

    Attributes:
        mode: Summarization mode
        recent_turns: Number of recent conversation turns for RECENT_N or HYBRID modes
        max_length: Maximum length for summaries
        include_tool_calls: Whether to include tool call details
        include_metadata: Whether to include message metadata
    """

    mode: HistorySummarizationMode = HistorySummarizationMode.HYBRID
    recent_turns: int = 3
    max_length: int = 4000
    include_tool_calls: bool = True
    include_metadata: bool = False

    async def summarize(
        self,
        messages: list[dict[str, Any]],
        llm_client: Any | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Summarize conversation history based on configured mode.

        Args:
            messages: List of messages to summarize
            llm_client: Optional LLM client for LLM-based summarization
            model: Model to use for summarization

        Returns:
            Summarized messages
        """
        if not messages:
            return []

        if self.mode == HistorySummarizationMode.FULL:
            return deepcopy(messages)

        elif self.mode == HistorySummarizationMode.RECENT_N:
            boundary = _find_turn_boundary(messages, self.recent_turns)
            return deepcopy(messages[boundary:])

        elif self.mode == HistorySummarizationMode.SUMMARY:
            return await self._create_summary(messages, llm_client, model)

        elif self.mode == HistorySummarizationMode.HYBRID:
            boundary = _find_turn_boundary(messages, self.recent_turns)
            if boundary == 0:
                return deepcopy(messages)

            older_messages = messages[:boundary]
            recent_messages = messages[boundary:]

            summary = await self._create_summary(older_messages, llm_client, model)
            return summary + deepcopy(recent_messages)

        return deepcopy(messages)

    async def _create_summary(
        self,
        messages: list[dict[str, Any]],
        llm_client: Any | None,
        model: str | None,
    ) -> list[dict[str, Any]]:
        """Create a summary of messages."""
        if not messages:
            return []

        # If no LLM client, use text-based summary
        if llm_client is None:
            return [self._text_summary(messages)]

        try:
            # Use LLM for summarization
            summary_prompt = self._build_summary_prompt(messages)

            from orchestrator.llm.config import LLMConfig
            from orchestrator.llm.types import ChatMessage

            llm_config = LLMConfig(model=model) if model else None
            response = await llm_client.chat(
                messages=[ChatMessage(role="user", content=summary_prompt)],
                config=llm_config,
                auto_session=False,
            )

            return [
                {
                    "role": "assistant",
                    "content": f"{_history_start}\n{response.content}\n{_history_end}",
                }
            ]

        except Exception as e:
            logger.warning(f"LLM summarization failed, using text summary: {e}")
            return [self._text_summary(messages)]

    def _text_summary(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a text-based summary without LLM."""
        summary_lines = ["For context, here is the conversation so far:"]
        summary_lines.append(_history_start)

        for i, msg in enumerate(messages, 1):
            line = format_message_for_summary(msg, self.include_tool_calls)
            summary_lines.append(f"{i}. {line}")

        summary_lines.append(_history_end)

        content = "\n".join(summary_lines)

        # Truncate if too long
        if len(content) > self.max_length:
            content = content[: self.max_length - 3] + "..."

        return {
            "role": "assistant",
            "content": content,
        }

    def _build_summary_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Build prompt for LLM summarization."""
        formatted = []
        for msg in messages:
            formatted.append(format_message_for_summary(msg, self.include_tool_calls))

        conversation = "\n".join(formatted)

        return f"""Summarize the following conversation concisely, preserving key information, decisions, and any important context for continuing the conversation.

CONVERSATION:
{conversation}

SUMMARY:"""


def format_message_for_summary(
    message: dict[str, Any],
    include_tool_calls: bool = True,
) -> str:
    """
    Format a single message for inclusion in a summary.

    Args:
        message: Message dict
        include_tool_calls: Whether to include tool call details

    Returns:
        Formatted string
    """
    role = message.get("role", "unknown")
    content = message.get("content", "")
    name = message.get("name")

    # Build prefix
    prefix = role
    if name:
        prefix = f"{role} ({name})"

    # Handle different message types
    if role == "tool":
        tool_call_id = message.get("tool_call_id", "")
        if include_tool_calls:
            return f"{prefix} [tool_call_id={tool_call_id}]: {_stringify_content(content)}"
        return f"{prefix}: [tool result]"

    # Handle tool calls in assistant messages
    tool_calls = message.get("tool_calls", [])
    if tool_calls and include_tool_calls:
        tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        tools_str = ", ".join(tool_names)
        content_str = _stringify_content(content)
        if content_str:
            return f"{prefix}: {content_str} [called tools: {tools_str}]"
        return f"{prefix}: [called tools: {tools_str}]"

    return f"{prefix}: {_stringify_content(content)}"


def _stringify_content(content: Any) -> str:
    """Convert content to string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def default_history_mapper(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Default mapper that creates a summary message from transcript.

    This is the OpenAI SDK style - creates a single assistant message
    containing the conversation summary.

    Args:
        messages: List of messages to summarize

    Returns:
        List containing single summary message
    """
    summarizer = HistorySummarizer(mode=HistorySummarizationMode.SUMMARY)
    return [summarizer._text_summary(messages)]


def summarize_conversation(
    messages: list[dict[str, Any]],
    mode: HistorySummarizationMode = HistorySummarizationMode.HYBRID,
    recent_turns: int = 3,
) -> list[dict[str, Any]]:
    """
    Convenience function to summarize a conversation.

    Args:
        messages: Messages to summarize
        mode: Summarization mode
        recent_turns: Number of recent conversation turns for hybrid/recent_n mode

    Returns:
        Summarized messages
    """
    summarizer = HistorySummarizer(mode=mode, recent_turns=recent_turns)

    # Use sync version (no LLM)
    if mode == HistorySummarizationMode.FULL:
        return deepcopy(messages)
    elif mode == HistorySummarizationMode.RECENT_N:
        boundary = _find_turn_boundary(messages, recent_turns)
        return deepcopy(messages[boundary:])
    elif mode == HistorySummarizationMode.SUMMARY:
        return [summarizer._text_summary(messages)]
    elif mode == HistorySummarizationMode.HYBRID:
        boundary = _find_turn_boundary(messages, recent_turns)
        if boundary == 0:
            return deepcopy(messages)
        older = messages[:boundary]
        recent = messages[boundary:]
        return [summarizer._text_summary(older)] + deepcopy(recent)

    return deepcopy(messages)


def extract_nested_history(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Extract nested history from a summary message.

    Parses the OpenAI SDK style history markers to extract
    individual messages from a summary.

    Args:
        message: Message that may contain nested history

    Returns:
        List of extracted messages or None if not a summary
    """
    content = message.get("content")
    if not isinstance(content, str):
        return None

    start_idx = content.find(_history_start)
    end_idx = content.find(_history_end)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    # Extract body between markers
    start_idx += len(_history_start)
    body = content[start_idx:end_idx]

    # Parse lines
    lines = [line.strip() for line in body.splitlines() if line.strip()]

    parsed = []
    for line in lines:
        msg = _parse_summary_line(line)
        if msg:
            parsed.append(msg)

    return parsed if parsed else None


def _parse_summary_line(line: str) -> dict[str, Any] | None:
    """Parse a single line from a summary back into a message."""
    stripped = line.strip()
    if not stripped:
        return None

    # Remove line number prefix (e.g., "1. ")
    dot_idx = stripped.find(".")
    if dot_idx != -1 and stripped[:dot_idx].isdigit():
        stripped = stripped[dot_idx + 1 :].lstrip()

    # Split role and content
    role_part, sep, content = stripped.partition(":")
    if not sep:
        return None

    role_text = role_part.strip()
    if not role_text:
        return None

    # Parse role and optional name
    role, name = _split_role_and_name(role_text)

    message: dict[str, Any] = {"role": role}
    if name:
        message["name"] = name

    content_str = content.strip()
    if content_str:
        message["content"] = content_str

    return message


def _split_role_and_name(role_text: str) -> tuple[str, str | None]:
    """Split role text into role and optional name."""
    if role_text.endswith(")") and "(" in role_text:
        open_idx = role_text.rfind("(")
        name = role_text[open_idx + 1 : -1].strip()
        role = role_text[:open_idx].strip()
        if name:
            return (role or "assistant", name)
    return (role_text or "assistant", None)


def flatten_nested_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten nested history in messages.

    Expands any summary messages that contain nested history
    back into individual messages.

    Args:
        messages: Messages that may contain nested histories

    Returns:
        Flattened messages
    """
    flattened = []

    for msg in messages:
        nested = extract_nested_history(msg)
        if nested is not None:
            flattened.extend(nested)
        else:
            flattened.append(deepcopy(msg))

    return flattened
