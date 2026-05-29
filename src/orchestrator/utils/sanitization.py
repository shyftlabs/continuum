"""Input sanitization utilities for prompt injection prevention."""

from __future__ import annotations

import re
from typing import Any

# Zero-width and invisible unicode characters used in prompt injection attacks
_INVISIBLE_UNICODE_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f'   # zero-width spaces/joiners/marks
    r'\u202a-\u202e'                       # directional formatting characters
    r'\u2060-\u2064'                       # word joiner, invisible separators
    r'\ufeff'                              # BOM / zero-width no-break space
    r'\u00ad]'                             # soft hyphen
)

INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+(a|an)\s+"),
    re.compile(r"(?i)system:\s*"),
    re.compile(r"(?i)<<\s*SYS\s*>>"),
    re.compile(r"(?i)\[INST\]"),
    re.compile(r"(?i)###\s*(system|instruction|prompt)"),
]


def detect_injection_patterns(text: str) -> list[str]:
    """Detect potential prompt injection patterns. Returns list of matched patterns."""
    matches = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_user_input(
    text: str,
    max_length: int = 50000,
    strip_control_chars: bool = True,
) -> str:
    """Sanitize user input for safe inclusion in prompts."""
    if not text:
        return text

    if len(text) > max_length:
        text = text[:max_length]

    if strip_control_chars:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = _INVISIBLE_UNICODE_RE.sub("", text)

    return text


def sanitize_message_content(message: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a message dict's content field (only user messages)."""
    if message.get("role") == "user" and message.get("content"):
        message = message.copy()
        message["content"] = sanitize_user_input(message["content"])
    return message
