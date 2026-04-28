#!/usr/bin/env python3
"""
Orla-based playground CLI.

Demonstrates all session implementations:
  - @function_tool in-process MCP tools (no external server)
  - PolicyStore deny-overrides (free tier blocked from checkout)
  - data_labels taint tracking
  - RouterAgent priority stamping
  - PriorityDispatcher
  - DAGAgent parallel pipeline

Usage:
    python cli.py

Commands:
    /tier free|premium   Switch user tier
    /dag                 Run DAG parallel pipeline on next message
    /labels add <label>  Add a data label to context
    /labels clear        Clear all data labels
    /labels              Show current labels
    /policy              Show active policies
    /tools               List registered in-process tools
    /summarize           Summarize conversation so far
    /cart                Show current cart
    /reset               Clear cart
    /help                Show this help
    /quit                Exit
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator import LogLevel, setup_logging
from orchestrator.agent.types import generate_run_id
from orchestrator.logging import get_logger

from agents import OrlaPlayground
from config import default_config
from pipeline import run_dag, run_direct, run_summarize
from tools import reset_cart

logger = get_logger(__name__)


def print_help():
    print("""
Commands:
  /tier free|premium   — switch user tier (affects routing + policy)
  /dag                 — run DAG parallel pipeline on next message
  /labels              — show current data labels on context
  /labels add <label>  — add a data label (e.g. /labels add pii)
  /labels clear        — clear all data labels
  /policy              — show active policies in the store
  /tools               — list all registered in-process tools
  /summarize           — summarize conversation so far
  /cart                — show current cart contents
  /reset               — clear cart
  /help                — show this help
  /quit                — exit

Tier effects:
  free    → dispatch_priority=2, checkout BLOCKED by policy
  premium → dispatch_priority=9, full access

Try:
  "show me dog toys"
  "add p5 to cart"
  "checkout"        ← blocked for free tier
  /tier premium
  "checkout"        ← allowed for premium
  /dag "find dog accessories"
""")


async def main():
    setup_logging(level=LogLevel.WARNING)

    print("=" * 60)
    print("  Orla-based Playground — in-process MCP + policies + DAG")
    print("=" * 60)
    user_id_input = input("Enter user ID (or Enter for auto): ").strip()
    user_id = user_id_input or f"orla-user-{generate_run_id()[-8:]}"
    conversation_id = f"orla-conv-{generate_run_id()[-8:]}"
    print(f"  User: {user_id}  |  Conversation: {conversation_id}")

    print("\nInitializing...")

    app = OrlaPlayground(default_config)
    try:
        await app.initialize()
    except Exception as e:
        print(f"\nInit failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"✓ Ready! {len(app.tools)} in-process tools loaded.")
    print("Type /help for commands or start chatting!\n")

    tier = default_config.default_tier
    data_labels: set[str] = set()
    conversation_history: list[str] = []
    use_dag_next = False

    def prompt_prefix():
        labels_str = f" [{','.join(sorted(data_labels))}]" if data_labels else ""
        return f"[{tier}{labels_str}] You: "

    try:
        while True:
            try:
                user_input = input(prompt_prefix()).strip()
                if not user_input:
                    continue

                # --- Commands ---
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()

                    if cmd in ("/quit", "/exit"):
                        print("Goodbye!")
                        break

                    elif cmd == "/help":
                        print_help()

                    elif cmd == "/tier":
                        if len(parts) < 2 or parts[1] not in ("free", "premium"):
                            print("Usage: /tier free|premium")
                        else:
                            tier = parts[1]
                            conversation_id = f"orla-conv-{generate_run_id()[-8:]}"
                            conversation_history.clear()
                            print(f"Switched to {tier} tier.")

                    elif cmd == "/dag":
                        use_dag_next = True
                        print("Next message will run through the DAG parallel pipeline.")

                    elif cmd == "/labels":
                        if len(parts) == 1:
                            print(f"Current labels: {sorted(data_labels) or '(none)'}")
                        elif parts[1] == "add" and len(parts) >= 3:
                            label = parts[2]
                            data_labels.add(label)
                            conversation_id = f"orla-conv-{generate_run_id()[-8:]}"
                            conversation_history.clear()
                            print(f"Added label '{label}'. Current: {sorted(data_labels)}")
                        elif parts[1] == "clear":
                            data_labels.clear()
                            conversation_id = f"orla-conv-{generate_run_id()[-8:]}"
                            conversation_history.clear()
                            print("Labels cleared.")
                        else:
                            print("Usage: /labels | /labels add <label> | /labels clear")

                    elif cmd == "/policy":
                        policies = app.config.policy_store.list_policies()
                        print(f"\nActive policies ({len(policies)}):")
                        for p in policies:
                            print(f"  [{p.effect.upper():5s}] {p.name}")
                            print(f"           subjects:   {p.subjects}")
                            print(f"           resources:  {p.resources}")
                        print()

                    elif cmd == "/tools":
                        tools = app.tools
                        print(f"\nIn-process tools ({len(tools)}):")
                        for t in tools:
                            f = t.get("function", {})
                            print(f"  • {f.get('name','?'):20s} — {f.get('description','')[:60]}")
                        print()

                    elif cmd == "/summarize":
                        if not conversation_history:
                            print("No conversation yet.")
                        else:
                            conv = "\n".join(conversation_history[-6:])
                            print("\nSummarizing (low stage_priority=2)...")
                            summary = await run_summarize(app, conv)
                            print(f"\nSummary: {summary}\n")

                    elif cmd == "/cart":
                        from tools import _cart
                        if not _cart:
                            print("Cart is empty.")
                        else:
                            total = sum(i["price"] * i["quantity"] for i in _cart)
                            print(f"\nCart ({len(_cart)} items, total ${total:.2f}):")
                            for item in _cart:
                                print(f"  {item['product_id']}: {item['name']} x{item['quantity']} @ ${item['price']}")
                            print()

                    elif cmd == "/reset":
                        reset_cart()
                        print("Cart cleared.")

                    else:
                        print(f"Unknown command '{cmd}'. Type /help.")

                    continue

                # --- Chat ---
                conversation_history.append(f"User: {user_input}")

                if use_dag_next:
                    use_dag_next = False
                    print("\nRunning DAG pipeline (fetch + recommend in parallel → synthesize → reply)...")
                    response = await run_dag(app, user_input, data_labels=set(data_labels), user_id=user_id)
                    mode = "DAG"
                else:
                    from config import TIER_PRIORITY
                    route = app.router.get_route(f"{tier}-agent")
                    dispatch_priority = route.dispatch_priority if route else TIER_PRIORITY.get(tier, 5)
                    print(f"\nRouting to {tier}-agent (dispatch_priority={dispatch_priority})...")
                    response = await run_direct(app, user_input, tier=tier, data_labels=set(data_labels), user_id=user_id, conversation_id=conversation_id)
                    mode = tier

                conversation_history.append(f"Assistant: {response}")
                print(f"\nAssistant [{mode}]: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                import traceback
                traceback.print_exc()

    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
