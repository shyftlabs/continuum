"""Input sanitization utilities for prompt injection prevention."""

from __future__ import annotations

import re
from typing import Any

# Zero-width and invisible unicode characters used in prompt injection attacks
_INVISIBLE_UNICODE_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"  # zero-width spaces/joiners/marks
    r"\u202a-\u202e"  # directional formatting characters
    r"\u2060-\u2064"  # word joiner, invisible separators
    r"\ufeff"  # BOM / zero-width no-break space
    r"\u00ad]"  # soft hyphen
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


# Allowlist for user_id and conversation_id: alphanumeric, hyphen, underscore, dot, @, pipe.
# Colons are explicitly excluded — they are key delimiters in Redis session keys
# ("c:{conversation_id}:u:{user_id}") and cart_session_id (f"{user_id}:{conversation_id}").
# A colon in either value would silently collapse into another user's scoping key.
# Pipe ("|") IS allowed: Auth0-style JWT 'sub' claims look like "auth0|64abc..." and
# the pipe is not a key delimiter here, so it is safe (PR #55 review follow-up).
_ID_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9\-_.@|]")
_ID_MAX_LENGTH = 128


class InvalidIdentifierError(ValueError):
    """Raised when a user_id or conversation_id is unsafe for use as a
    session/memory scope key.

    These IDs become Redis key fragments ("u:{user_id}",
    "c:{conversation_id}:u:{user_id}") and memory bucket scopes. We REJECT
    malformed values rather than silently coercing them: coercion can collapse
    two distinct identities into one key (e.g. "user:1" and "user_1" would both
    become "user_1", and a 200-char id would truncate into a shared 128-char
    prefix), which leaks data across tenants. A hard error forces the caller to
    supply a clean id (normally a JWT sub / OAuth id).
    """

    def __init__(self, field: str, value: str, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid {field}: {reason}")


def _validate_id(value: str | None, field: str) -> str | None:
    """Validate a user_id/conversation_id for safe use as a scope/session key.

    Whitespace and invisible unicode are stripped (never meaningful in an ID).
    Anything that remains outside the allowlist — or that exceeds the length
    limit — raises InvalidIdentifierError instead of being silently rewritten.
    An empty/None value (or one that is only whitespace/invisible) returns None,
    which downstream treats as "anonymous" (random session id, unscoped memory).
    """
    if not value:
        return None
    cleaned = _INVISIBLE_UNICODE_RE.sub("", value).strip()
    if not cleaned:
        return None
    if len(cleaned) > _ID_MAX_LENGTH:
        raise InvalidIdentifierError(
            field, value, f"length {len(cleaned)} exceeds maximum of {_ID_MAX_LENGTH}"
        )
    bad = _ID_ALLOWED_RE.search(cleaned)
    if bad:
        raise InvalidIdentifierError(
            field,
            value,
            f"contains disallowed character {bad.group()!r}; "
            "allowed characters are letters, digits, and - _ . @ |",
        )
    return cleaned


def validate_user_id(user_id: str | None) -> str | None:
    """Validate user_id for safe use as a Redis key fragment and memory scope key.

    Returns the normalized id, None for empty/anonymous input, or raises
    InvalidIdentifierError for malformed input.
    """
    return _validate_id(user_id, "user_id")


def validate_conversation_id(conversation_id: str | None) -> str | None:
    """Validate conversation_id for safe use as a Redis key fragment.

    Returns the normalized id, None for empty/anonymous input, or raises
    InvalidIdentifierError for malformed input.
    """
    return _validate_id(conversation_id, "conversation_id")
