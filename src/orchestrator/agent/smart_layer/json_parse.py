"""Parse classifier output: strict JSON, legacy complexity, regex salvage."""

from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.agent.smart_layer.errors import TierClassifierError
from orchestrator.agent.smart_layer.types import PRODUCT_TIERS, ProductTier, parse_product_tier

_TIER_RE = re.compile(r'"tier"\s*:\s*"([^"]+)"', re.I)
_COMPLEXITY_RE = re.compile(r'"complexity"\s*:\s*"([^"]+)"', re.I)


def _coerce_classifier_tier_from_text(text: str) -> ProductTier | None:
    """Parse tier from non-empty stripped classifier text; None if nothing matched."""
    # Try strict JSON first
    try:
        data: Any = json.loads(text)
        if isinstance(data, dict):
            tier = parse_product_tier(data.get("tier"))
            if tier:
                return tier
            cx = data.get("complexity")
            if isinstance(cx, str):
                cxl = cx.lower().strip()
                if cxl == "simple":
                    return ProductTier.fast
                if cxl == "complex":
                    return ProductTier.balanced
    except json.JSONDecodeError:
        pass

    m = _TIER_RE.search(text)
    if m:
        tier = parse_product_tier(m.group(1))
        if tier:
            return tier

    m = _COMPLEXITY_RE.search(text)
    if m:
        cxl = m.group(1).lower().strip()
        if cxl == "simple":
            return ProductTier.fast
        if cxl == "complex":
            return ProductTier.balanced

    loose = text.lower()
    for name in PRODUCT_TIERS:
        if re.search(rf"\b{re.escape(name)}\b", loose):
            t = parse_product_tier(name)
            if t:
                return t

    return None


def parse_classifier_json(raw: str | None) -> ProductTier:
    """
    Parse tier from classifier output.

    Accepts strict JSON with "tier", legacy "complexity": simple|complex,
    or regex salvage. Falls back to balanced.
    """
    if not raw or not raw.strip():
        return ProductTier.balanced

    t = _coerce_classifier_tier_from_text(raw.strip())
    return ProductTier.balanced if t is None else t


def parse_classifier_tier_strict(raw: str | None) -> ProductTier:
    """
    Parse tier from classifier output; raise TierClassifierError if empty or unparseable.

    Used after a classifier LLM call so silent balanced fallback cannot mask failures.
    """
    if not raw or not raw.strip():
        raise TierClassifierError("Tier classifier returned empty content")

    s = raw.strip()
    t = _coerce_classifier_tier_from_text(s)
    if t is None:
        raise TierClassifierError(
            f"Tier classifier output did not contain a valid tier (snippet): {s[:400]!r}"
        )
    return t
