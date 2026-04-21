#!/usr/bin/env python3
"""
Fetch Agent - CLI Interface.

Interactive command-line interface for the Fetch web assistant.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import FetchAgent, create_fetch_agent

from orchestrator import LogLevel, setup_logging


def print_banner():
    print("""
======================================================================
  Fetch Agent - Web Page Reader
======================================================================
  Ask me to fetch any URL and I'll read and summarize it for you!
======================================================================
""")


def print_help():
    print("""
Commands:
  /tools    - List available MCP tools
  /session  - Show session info
  /memories - Show stored memories
  /clear    - Clear screen
  /help     - Show this help
  /quit     - Exit

Example queries:
  "Fetch https://example.com and summarize it"
  "What is on the page https://httpbin.org/json"
""")


async def main():
    setup_logging(level=LogLevel.INFO)
    print_banner()

    print("Initializing agent...")

    try:
        agent = await create_fetch_agent()
        print(f"✓ Agent ready with {len(agent.tools)} tools!")
        print(f"Type /help for commands or start chatting!\n")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        print("\nMake sure:")
        print("  1. uv is installed: brew install uv")
        print("  2. Required environment variables are set (GEMINI_API_KEY or OPENAI_API_KEY)")
        print("  3. Docker services are running")
        return

    try:
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command in ("/quit", "/exit"):
                        print("\nGoodbye!\n")
                        break
                    elif command == "/help":
                        print_help()
                    elif command == "/tools":
                        print(f"\nAvailable Tools ({len(agent.tools)}):")
                        for tool in agent.tools:
                            func = tool.get("function", {})
                            print(f"  • {func.get('name', '?')} - {func.get('description', '')[:80]}")
                        print()
                    elif command == "/session":
                        print(f"\nUser ID:    {agent.user_id}")
                        print(f"Session ID: {agent.session_id or 'N/A'}\n")
                    elif command == "/memories":
                        memories = await agent.get_memories()
                        if not memories:
                            print("\nNo memories stored (memory may be disabled or empty).\n")
                        else:
                            print(f"\nStored Memories ({len(memories)}):")
                            for i, m in enumerate(memories, 1):
                                print(f"  {i}. {m.get('memory', m)}")
                            print()
                    elif command == "/clear":
                        os.system("clear")
                        print_banner()
                    else:
                        print("Unknown command. Type /help for available commands.")
                    continue

                print("\nThinking...")
                response = await agent.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
