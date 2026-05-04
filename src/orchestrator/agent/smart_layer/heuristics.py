"""Deterministic pre-classifier before optional LLM tier classification."""

from __future__ import annotations

from orchestrator.agent.smart_layer.types import ProductTier

# Frontier-style cues (hardest reasoning / proofs / deep theory)
_FRONTIER_KEYS = (
    "formal proof",
    "prove that",
    "np-complete",
    "undecidable",
    "measure theory",
    "research paper",
    "novel algorithm",
    "optimization proof",
)

# Specialist / code-repo cues
_SPECIALIST_KEYS = (
    "implement",
    "debug",
    "stack trace",
    "codebase",
    "repository",
    "refactor",
    "unit test",
    "pull request",
    "typescript",
    "python code",
    "stacktrace",
    "compiler error",
)

_DEEP_WORDS = (
    "prove",
    "theorem",
    "architecture",
    "design a system",
    "optimize asymptotically",
    "security audit",
    "threat model",
)


def heuristic_tier(user_text: str) -> ProductTier | None:
    """
    Return a tier if heuristic rules fire; otherwise None (run LLM classifier).

    Rules (lowercased text):
    - Keyword sets → frontier or specialist
    - Very short + no deep wording → nano or fast with length thresholds
    """
    s = user_text.strip().lower()
    if not s:
        return ProductTier.nano

    for kw in _FRONTIER_KEYS:
        if kw in s:
            return ProductTier.frontier

    for kw in _SPECIALIST_KEYS:
        if kw in s:
            return ProductTier.specialist

    has_deep = any(w in s for w in _DEEP_WORDS)
    n = len(s.split())

    if n <= 6 and len(s) < 80 and not has_deep:
        if n <= 3 and len(s) < 40:
            return ProductTier.nano
        return ProductTier.fast

    return None
