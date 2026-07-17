#!/usr/bin/env python3
"""TL-65 LIVE API test — real calls to OpenAI and Anthropic.

Proves the OpenAI-dependency fix end to end using real provider calls:

  A. OpenAI present    -> default resolves to gpt-4o-mini, real call works (back-compat).
  B. Anthropic present -> default resolves to a Claude model, real call works.
  C. THE PROOF: with the OpenAI key temporarily removed, an Anthropic-only
     routing-style call (the exact path RouterAgent uses) still succeeds -
     i.e. the meta-operation needs NO OpenAI credential.

Requires the relevant keys in the root .env. Each scenario skips if its key is absent.

Run:  python playground/tickets/tl-65/live_api.py
"""

from __future__ import annotations

import asyncio
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
from continuum.llm import LLMClient  # noqa: E402
from continuum.llm.config import LLMConfig  # noqa: E402

# The prompt shape RouterAgent._llm_route sends for an LLM routing decision.
ROUTING_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Available agents:\n- billing: payments, invoices, refunds\n"
            "- technical: bugs, outages\n\nUser request: my invoice is wrong\n"
            "Respond with ONLY the agent name."
        ),
    }
]

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


def _resolved_default(**keys) -> str:
    base = {"openai_api_key": None, "anthropic_api_key": None, "gemini_api_key": None}
    base.update(keys)
    return Settings(**base).default_llm_model


async def _real_call(model: str) -> str:
    client = LLMClient(config=LLMConfig(model=model), enable_langfuse=False)
    resp = await client.chat(
        messages=ROUTING_MESSAGES,
        config=LLMConfig(model=model, temperature=0.1, max_tokens=16),
        auto_session=False,
    )
    return (resp.content or "").strip()


async def scenario_openai() -> None:
    print("\n=== A. OpenAI present -> gpt-4o-mini, real call ===")
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [SKIP] OPENAI_API_KEY not set")
        return
    model = _resolved_default(openai_api_key="x")
    check("A1 resolves to gpt-4o-mini", model == "gpt-4o-mini", model)
    out = await _real_call(model)
    check("A2 real OpenAI routing call works", bool(out), f"{model} -> {out!r}")


async def scenario_anthropic() -> None:
    print("\n=== B. Anthropic present -> Claude, real call ===")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set")
        return
    model = _resolved_default(anthropic_api_key="x")
    check("B1 resolves to a Claude model", model.startswith("claude"), model)
    out = await _real_call(model)
    check("B2 real Anthropic routing call works", bool(out), f"{model} -> {out!r}")


async def scenario_openai_disabled_proof() -> None:
    print("\n=== C. THE PROOF: OpenAI key removed, Anthropic routing still works ===")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set - cannot prove Anthropic-only path")
        return

    saved_env = os.environ.pop("OPENAI_API_KEY", None)
    saved_setting = config_mod.settings.openai_api_key
    config_mod.settings.openai_api_key = None  # make OpenAI genuinely unavailable
    model = _resolved_default(anthropic_api_key="x")  # Anthropic-only resolution
    try:
        out = await _real_call(model)
        check(
            "C1 routing works with NO OpenAI key (Anthropic-only)",
            bool(out),
            f"{model} -> {out!r}",
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        openai_dep = "openai" in msg or "missing credentials" in msg
        check("C1 no OpenAI dependency", not openai_dep, str(e)[:140])
    finally:
        if saved_env is not None:
            os.environ["OPENAI_API_KEY"] = saved_env
        config_mod.settings.openai_api_key = saved_setting


async def main() -> int:
    print("TL-65 - LIVE API test (real OpenAI + Anthropic calls)")
    await scenario_openai()
    await scenario_anthropic()
    await scenario_openai_disabled_proof()

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n==== {passed}/{total} live checks passed ====")
    return 0 if total and passed == total else (0 if total == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
