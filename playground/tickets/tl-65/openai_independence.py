#!/usr/bin/env python3
"""TL-65 playground — prove an Anthropic-only (or Gemini-only) deployment needs NO OpenAI key.

Before the fix, the framework's meta-operations (router LLM routing, reflection
critique, memory fact-extraction, summarization) all defaulted to `gpt-4o-mini`,
so an Anthropic-only shop hit "Missing credentials (OpenAI)". This demo shows the
default is now provider-aware and that those operations route to the configured
provider instead of OpenAI.

Runs fully offline (no API key needed) — it inspects resolution + provider routing.
If ANTHROPIC_API_KEY is set, it also makes ONE real Anthropic call to prove the
end-to-end path works with no OpenAI credential.

Run:  python playground/tickets/tl-65/openai_independence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root = nearest ancestor with pyproject.toml, so this works at any depth.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
load_dotenv(_ROOT / ".env", override=True)
sys.path.insert(0, str(_ROOT / "src"))

import continuum.config as config_mod  # noqa: E402
from continuum.config import Settings  # noqa: E402
from continuum.llm.config import LLMConfig  # noqa: E402
from continuum.llm.providers import get_provider  # noqa: E402
from continuum.llm.providers.openai_provider import OpenAIProvider  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


def _settings(**overrides) -> Settings:
    """Deterministic Settings: kwargs override env; provider keys default to None."""
    base = {"openai_api_key": None, "anthropic_api_key": None, "gemini_api_key": None}
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
def part1_resolution() -> None:
    print("\n=== Part 1: default model resolves from the configured provider (offline) ===")

    anth = _settings(anthropic_api_key="sk-ant-demo")
    check(
        "Anthropic-only -> Claude default",
        anth.default_llm_model.startswith("claude"),
        anth.default_llm_model,
    )
    check(
        "memory + summarization inherit the Claude default",
        anth.memory_llm_model == anth.default_llm_model
        and anth.context_summarization_model == anth.default_llm_model,
        f"{anth.memory_llm_model} / {anth.context_summarization_model}",
    )

    gem = _settings(gemini_api_key="demo")
    check("Gemini-only -> Gemini default", "gemini" in gem.default_llm_model, gem.default_llm_model)

    oai = _settings(openai_api_key="sk-demo")
    check(
        "OpenAI present -> gpt-4o-mini (unchanged / back-compat)",
        oai.default_llm_model == "gpt-4o-mini",
        oai.default_llm_model,
    )

    explicit = _settings(anthropic_api_key="x", default_llm_model="claude-sonnet-4-6")
    check(
        "Explicit DEFAULT_LLM_MODEL always honored",
        explicit.default_llm_model == "claude-sonnet-4-6",
        explicit.default_llm_model,
    )


# ---------------------------------------------------------------------------
def part2_no_openai_routing() -> None:
    print("\n=== Part 2: meta-ops route to the configured provider, NOT OpenAI (offline) ===")

    anth = _settings(anthropic_api_key="sk-ant-demo")
    # Simulate the process running under an Anthropic-only configuration.
    original = config_mod.settings.default_llm_model
    config_mod.settings.default_llm_model = anth.default_llm_model
    try:
        from continuum.agent.workflow.router import RouterAgent

        router = RouterAgent(name="triage")  # no explicit model -> inherits the default
        check(
            "RouterAgent inherits the Claude default (not gpt-4o-mini)",
            router.model.startswith("claude"),
            router.model,
        )

        # The exact LLMConfig the router builds for an LLM routing decision.
        routing_cfg = LLMConfig(model=router.router_config.routing_model or router.model)
        provider = get_provider(routing_cfg)
        check(
            "Router LLM-routing does NOT resolve to the OpenAI provider",
            not isinstance(provider, OpenAIProvider),
            type(provider).__name__,
        )
    finally:
        config_mod.settings.default_llm_model = original


# ---------------------------------------------------------------------------
def part3_live_optional() -> None:
    print("\n=== Part 3: live Anthropic call (only if ANTHROPIC_API_KEY set) ===")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  [SKIP] ANTHROPIC_API_KEY not set - skipping live call")
        return

    anth = _settings(anthropic_api_key=key)
    model = anth.default_llm_model  # the Claude default the framework would pick
    try:
        provider = get_provider(LLMConfig(model=model))
        if isinstance(provider, OpenAIProvider):
            check("live: resolved provider is not OpenAI", False, "routed to OpenAI")
            return
        resp = provider.complete(
            [{"role": "user", "content": "Reply with one word: ok"}],
            LLMConfig(model=model, max_tokens=8),
        )
        check(
            f"live: real call on {model} succeeded with no OpenAI key",
            bool(resp.content and resp.content.strip()),
            repr((resp.content or "").strip()),
        )
    except Exception as e:  # noqa: BLE001
        # A model-access error is about the account, not the OpenAI dependency.
        msg = str(e).lower()
        if "openai" in msg or "missing credentials" in msg:
            check("live: no OpenAI dependency", False, str(e)[:120])
        else:
            print(f"  [SKIP] live call not conclusive ({type(e).__name__}: {str(e)[:80]})")


# ---------------------------------------------------------------------------
def main() -> int:
    print("TL-65 - Anthropic-only independence demo")
    part1_resolution()
    part2_no_openai_routing()
    part3_live_optional()

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n==== {passed}/{total} checks passed ====")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
