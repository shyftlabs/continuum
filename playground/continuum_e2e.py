#!/usr/bin/env python3
"""Comprehensive Continuum feature E2E (docs/agent.md + skills).

Covers, live against Smart Inference + Milvus + Redis:
  T1  Structured output         — Pydantic output_schema
  T2  Instruction enrichment    — template_vars + examples + instruction_modifiers
  T3  Agent handoffs            — triage -> specialist transition
  T4  Memory via Smart Inference— fact extraction (auto/cheap) + recall

Run:  python continuum_e2e.py
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel  # noqa: E402

from orchestrator.agent import AgentRunner, BaseAgent, Handoff  # noqa: E402
from orchestrator.agent.config import AgentMemoryConfig  # noqa: E402

NO_MEM = AgentMemoryConfig(search_memories=False, store_memories=False)
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}")


# --------------------------------------------------------------------------
async def t1_structured_output() -> None:
    print("\n=== T1: Structured output (Pydantic output_schema) ===")

    class Plan(BaseModel):
        intent: str
        steps: list[str]

    agent = BaseAgent(
        name="planner",
        instructions=(
            "Produce a short plan. Respond ONLY with a JSON object with keys: "
            "'intent' (a one-phrase string) and 'steps' (an array of 2-4 short imperative strings)."
        ),
        output_schema=Plan,
        # output_schema validates the result, but response_format is only sent
        # when enable_json_mode is True (see LLMConfig.from_agent_config).
        enable_json_mode=True,
        # JSON mode needs a json-capable gateway route: the default 'modest'/mid
        # tier resolves to a thinking model that doesn't enforce response_format;
        # 'strict' (cheap tier → gemini-2.0-flash) supports json_object.
        gateway_mode="strict",
        memory_config=NO_MEM,
    )
    resp = await AgentRunner().run(agent, "Plan a birthday party.", user_id="t1")
    plan = resp.structured_output
    print(f"  structured_output={plan}")
    check("structured output is a validated Plan", isinstance(plan, Plan) and len(plan.steps) >= 2)


# --------------------------------------------------------------------------
async def t2_instruction_enrichment() -> None:
    print("\n=== T2: Instruction enrichment (template_vars + examples + modifier) ===")

    def upgrade_for_enterprise(prompt: str, ctx) -> str:
        if ctx and getattr(ctx, "metadata", {}).get("tier") == "enterprise":
            return prompt + "\nENTERPRISE_SLA_NOTE: prioritise this account."
        return prompt

    agent = BaseAgent(
        name="adaptive",
        instructions="You are helping {user_name}.",
        template_vars={"user_name": "Alice"},
        examples=[{"input": "hi", "output": "Hello Alice!"}],
        instruction_modifiers=[upgrade_for_enterprise],
        memory_config=NO_MEM,
    )

    # Deterministic assertion on the assembled system prompt.
    ctx = SimpleNamespace(
        metadata={"tier": "enterprise"}, user_id="u1", session_id=None, run_id=None
    )
    prompt = agent.resolve_system_prompt(ctx)
    print("  resolved prompt:\n    " + prompt.replace("\n", "\n    "))
    rendered = "Alice" in prompt and "{user_name}" not in prompt
    examples_block = "Examples:" in prompt and "Hello Alice!" in prompt
    modifier_fired = "ENTERPRISE_SLA_NOTE" in prompt

    # And confirm the modifier is tier-gated (not applied for non-enterprise).
    ctx_free = SimpleNamespace(
        metadata={"tier": "free"}, user_id="u1", session_id=None, run_id=None
    )
    not_for_free = "ENTERPRISE_SLA_NOTE" not in agent.resolve_system_prompt(ctx_free)

    check("template_vars rendered ({user_name}->Alice)", rendered)
    check("few-shot examples block injected", examples_block)
    check("instruction_modifier fired for enterprise", modifier_fired)
    check("instruction_modifier gated off for non-enterprise", not_for_free)


# --------------------------------------------------------------------------
async def t3_handoffs() -> None:
    print("\n=== T3: Agent handoffs (triage -> specialist) ===")
    billing = BaseAgent(
        name="billing",
        instructions="Help with invoices and refunds. Answer briefly.",
        memory_config=NO_MEM,
    )
    technical = BaseAgent(
        name="technical",
        instructions="Help with bugs and outages. Answer briefly.",
        memory_config=NO_MEM,
    )
    triage = BaseAgent(
        name="triage",
        instructions="Route the customer to the right specialist via the handoff tools. Do not answer billing/technical questions yourself.",
        handoffs=[
            Handoff(target_agent="billing", description="Billing, payments, refunds, invoices"),
            Handoff(
                target_agent="technical", description="Bugs, errors, outages, integration issues"
            ),
        ],
        memory_config=NO_MEM,
    )
    runner = AgentRunner(
        agent_registry={"triage": triage, "billing": billing, "technical": technical}
    )
    resp = await runner.run(
        triage, "I want a refund for invoice 1234.", user_id="t3", session_id="t3s"
    )

    def _to_name(h):
        if isinstance(h, dict):
            return h.get("to_agent") or h.get("target_agent")
        return h

    chain = [_to_name(h) for h in resp.handoff_chain]
    print(f"  agents_used={resp.agents_used}  handoff_chain={chain}")
    print(f"  final content={resp.content[:120]!r}")
    routed = "billing" in (resp.agents_used or []) or "billing" in str(chain)
    check("triage handed off to billing", routed and len(resp.handoff_chain) >= 1)


# --------------------------------------------------------------------------
async def t4_memory_via_gateway() -> None:
    print("\n=== T4: Memory fact extraction via Smart Inference (auto/cheap) ===")
    from orchestrator.memory.config import MemoryConfig
    from orchestrator.memory.providers.mem0 import Mem0Provider

    llm = MemoryConfig().to_mem0_config()["llm"]["config"]
    print(f"  memory LLM model={llm['model']!r} base_url={llm.get('openai_base_url')!r}")
    p = Mem0Provider(MemoryConfig())
    uid = "t4-user"
    await p.delete_all(user_id=uid)
    add = await p.add(
        [
            {
                "role": "user",
                "content": "I run a golden retriever rescue and only feed grain-free food.",
            }
        ],
        user_id=uid,
    )
    facts = [r.get("memory") for r in add.results]
    print(f"  extracted facts={facts}")
    s = await p.search("what does the user do and feed?", user_id=uid, limit=5)
    print(f"  recall hits={s.total_results}: {[r.memory for r in s.results]}")
    await p.delete_all(user_id=uid)
    check("fact extraction produced memories (gateway LLM)", len(add.results) > 0)
    check("semantic recall returned the stored facts", s.total_results > 0)


# --------------------------------------------------------------------------
async def main() -> int:
    from orchestrator import LogLevel, setup_logging

    setup_logging(level=LogLevel.WARNING)  # quiet — we assert on returned values
    await t1_structured_output()
    await t2_instruction_enrichment()
    await t3_handoffs()
    await t4_memory_via_gateway()

    print("\n" + "=" * 60 + "\n  FEATURE E2E SUMMARY\n" + "=" * 60)
    for n, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    allok = all(results.values())
    print("\n" + ("✅ ALL FEATURE TESTS PASSED" if allok else "❌ SOME TESTS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
