#!/usr/bin/env python3
"""
Local Shop CLI.

Tests MCPServerStreamableHttp (HTTP transport) + session + memory — same SDK
patterns as commerce-chat but against a local MCP server.

Usage:
  Terminal 1:  python server.py          (start MCP server)
  Terminal 2:  python cli.py             (start agent)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import create_shop_agent
from orchestrator import LogLevel, setup_logging


def print_help():
    print("""
Commands:
  /tools    - List MCP tools discovered from server
  /session  - Show session info
  /clear    - Clear screen
  /help     - Show this help
  /quit     - Exit

Example queries:
  "show me dog toys"
  "add p5 to my cart"          (use product ID from search results)
  "what's in my cart?"
  "checkout"
""")


async def main():
    setup_logging(level=LogLevel.INFO)

    print("=" * 60)
    print("  Local Shop Agent — MCPServerStreamableHttp test")
    print("=" * 60)
    print()
    print("Make sure the MCP server is running:")
    print("  python server.py   (in another terminal)")
    print()

    user_id = input("Enter user ID (or Enter for auto): ").strip() or None

    print("\nConnecting to local MCP server...")
    try:
        agent = await create_shop_agent(user_id=user_id)
        print(f"✓ Ready! {len(agent.tools)} tools loaded.")
        print(f"  Session: {agent.session_id or 'N/A'}")
        print(f"  User:    {agent.user_id}")
        print("\nType /help for commands or start chatting!\n")
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
                        print_help()
                    elif cmd == "/tools":
                        print(f"\nAvailable Tools ({len(agent.tools)}):")
                        for t in agent.tools:
                            f = t.get("function", {})
                            print(f"  • {f.get('name','?')} — {f.get('description','')[:80]}")
                        print()
                    elif cmd == "/session":
                        print(f"\n  User ID:    {agent.user_id}")
                        print(f"  Session ID: {agent.session_id or 'N/A'}\n")
                    elif cmd == "/clear":
                        os.system("clear")
                    else:
                        print("Unknown command. Type /help.")
                    continue

                print("\nThinking...")
                # Pass session_id so the agent can use it for cart operations
                response = await agent.chat(
                    f"[session_id={agent.session_id}] {user_input}"
                )
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
