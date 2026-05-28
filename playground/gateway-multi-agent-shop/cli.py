#!/usr/bin/env python3
"""
Gateway Multi-Agent Shop CLI — all 10 workflow modes via Smart Gateway.

Usage:
  Terminal 1:  python server.py                  (start MCP server on :8890)
  Terminal 2:  python cli.py --mode sequential   (pick a mode)
"""

import argparse
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from config import default_config
from workflows import MODES, create_workflow

from orchestrator import LogLevel, setup_logging


def print_help(mode: str) -> None:
    examples = {
        "sequential": ["buy dog food", "get me a cat toy", "I need a dog leash"],
        "parallel": ["what's available for dogs and cats?", "show me all pet products"],
        "loop": ["find me something under $10", "find a dog toy under $15"],
        "scatter": ["compare p1 p2 and p5", "which of these is best value?"],
        "supervised": ["write a buying guide for a new puppy", "create a pet care guide"],
        "planner": ["set up for a new puppy", "I just got a cat, what do I need?"],
        "debate": ["should I buy premium or budget dog food?", "premium vs budget cat food"],
        "reflection": ["write a recommendation email for my friend", "draft a product review"],
        "router": ["show me dog toys", "add p5 to my cart", "how often should I feed my cat?"],
        "handoff": ["show me dog toys", "add p3 to my cart", "what's in my cart?", "checkout"],
    }
    print(f"""
Commands:
  /mode     - Show current workflow mode
  /tools    - List available MCP tools
  /help     - Show this help
  /quit     - Exit

Example queries for '{mode}' mode:""")
    for q in examples.get(mode, []):
        print(f'  "{q}"')
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Gateway Multi-Agent Shop — 10 workflow modes")
    parser.add_argument(
        "--mode",
        "-m",
        choices=list(MODES.keys()),
        default="sequential",
        help="Workflow mode to use (default: sequential)",
    )
    parser.add_argument("--user", "-u", default=None, help="User ID")
    args = parser.parse_args()

    setup_logging(level=LogLevel.DEBUG)

    mode = args.mode
    description = default_config.mode_descriptions.get(mode, "")

    print("=" * 64)
    print(f"  Gateway Multi-Agent Shop  —  mode: {mode.upper()}")
    print(f"  {description}")
    print(f"  gateway: {os.environ.get('SMART_GATEWAY_URL', 'NOT SET')}")
    print("=" * 64)
    print()
    print("Make sure the MCP server is running:")
    print("  python server.py   (in another terminal)")
    print()

    user_id = (
        args.user
        or input("Enter user ID (or Enter for auto): ").strip()
        or f"user-{uuid.uuid4().hex[:8]}"
    )
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"

    print(f"\nUser: {user_id}  |  Conversation: {conversation_id}")
    print("Connecting...\n")

    workflow = create_workflow(mode)
    try:
        await workflow.initialize()
        print(f"✓ Ready! {len(workflow.tools)} tools loaded.")
        print("Type /help for commands or start chatting!\n")
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        print("Is the MCP server running?  python server.py")
        return

    try:
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd in ("/quit", "/exit"):
                        print("Goodbye!")
                        break
                    elif cmd == "/help":
                        print_help(mode)
                    elif cmd == "/mode":
                        print(f"\n  Mode: {mode}  —  {description}\n")
                    elif cmd == "/tools":
                        print(f"\nAvailable Tools ({len(workflow.tools)}):")
                        for t in workflow.tools:
                            f = t.get("function", {})
                            print(f"  • {f.get('name', '?')} — {f.get('description', '')[:80]}")
                        print()
                    else:
                        print("Unknown command. Type /help.")
                    continue

                print("\nThinking...\n")
                response = await workflow.chat(user_input, user_id, conversation_id)
                print(f"Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        await workflow.close()


if __name__ == "__main__":
    asyncio.run(main())
