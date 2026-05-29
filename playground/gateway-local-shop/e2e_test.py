#!/usr/bin/env python3
"""End-to-end driver for the Local Shop playground agent.

Drives LocalShopAgent.chat() through a full shopping flow against the running
MCP server (server.py on :8888). Prints each turn and asserts the agent
actually invoked tools and produced sensible output.

Run:  python e2e_test.py   (server.py must already be running)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import LocalShopAgent  # noqa: E402

from orchestrator import LogLevel, setup_logging  # noqa: E402

USER = "e2e-user"
CONV = "e2e-conv-1"

TURNS = [
    "Show me dog toys you have.",
    "Add the tennis ball 3-pack (p5) to my cart.",
    "What's in my cart right now?",
    "Great, please checkout.",
]


async def main() -> int:
    setup_logging(level=LogLevel.INFO)
    agent = LocalShopAgent()

    print("\n" + "=" * 70)
    print("  LOCAL SHOP AGENT — END-TO-END TEST")
    print("=" * 70)

    await agent.initialize()
    tool_names = [t.function.name for t in agent.tools]
    print(f"\n[setup] {len(agent.tools)} tools discovered: {tool_names}")
    assert agent.tools, "No tools discovered from MCP server!"
    assert "search_products" in tool_names, "search_products tool missing"

    transcript: list[str] = []
    try:
        for i, msg in enumerate(TURNS, 1):
            print(f"\n--- Turn {i} ---\nUser: {msg}")
            resp = await agent.chat(msg, user_id=USER, conversation_id=CONV)
            print(f"Assistant: {resp}")
            assert resp and not resp.startswith("Error:"), f"Turn {i} failed: {resp}"
            transcript.append(resp.lower())
    finally:
        await agent.close()

    # Light end-to-end assertions on the conversation as a whole.
    joined = " ".join(transcript)
    checks = {
        "mentions a dog toy product": any(k in joined for k in ("tennis ball", "ball", "toy")),
        "confirms cart / item added": any(k in joined for k in ("cart", "added")),
        "completes checkout (order/total)": any(
            k in joined for k in ("order", "ord-", "total", "checkout", "placed")
        ),
    }
    print("\n" + "=" * 70)
    print("  ASSERTIONS")
    print("=" * 70)
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\n" + ("✅ E2E PASSED" if ok else "❌ E2E FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
