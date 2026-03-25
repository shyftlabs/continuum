"""Utility functions for masking and redacting sensitive data."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|credential|bearer)"),
    re.compile(r"(?i)(access[_-]?key|private[_-]?key|session[_-]?secret)"),
]

SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"pk-lf-[a-zA-Z0-9]+"),
    re.compile(r"sk-lf-[a-zA-Z0-9]+"),
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+"),
]


def mask_value(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive string, showing only last N characters."""
    if not value or len(value) <= visible_chars:
        return "****"
    return f"****{value[-visible_chars:]}"


def is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key likely contains sensitive data."""
    return any(pattern.search(key) for pattern in SENSITIVE_KEY_PATTERNS)


def redact_dict(
    data: dict[str, Any],
    depth: int = 0,
    max_depth: int = 5,
    _seen: set[int] | None = None,
) -> dict[str, Any]:
    """Recursively redact sensitive values from a dictionary.

    Handles circular references via identity tracking and returns
    "[REDACTED - max depth]" at depth limit instead of raw data.
    """
    # Track seen object ids to prevent infinite recursion on circular refs
    if _seen is None:
        _seen = set()

    obj_id = id(data)
    if obj_id in _seen:
        return {"_redacted": "[CIRCULAR REFERENCE]"}
    _seen.add(obj_id)

    if depth > max_depth:
        # Return redacted placeholder instead of raw unredacted data
        return {"_redacted": "[REDACTED - max depth exceeded]"}

    result = {}
    for key, value in data.items():
        if is_sensitive_key(key):
            if isinstance(value, str):
                result[key] = mask_value(value)
            else:
                result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict(value, depth + 1, max_depth, _seen)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item, depth + 1, max_depth, _seen) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = redact_sensitive_values(value)
        else:
            result[key] = value

    _seen.discard(obj_id)
    return result


def redact_sensitive_values(text: str) -> str:
    """Redact known sensitive patterns from a text string."""
    result = text
    for pattern in SENSITIVE_VALUE_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result
