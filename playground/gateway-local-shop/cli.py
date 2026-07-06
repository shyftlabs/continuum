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
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import create_shop_agent

from continuum import LogLevel, setup_logging


def print_help():
    print("""
Commands:
  /session    - Show session info
  /connectors - Show external-service connectors (mode + health)
  /health     - Show session-persistence health check + degraded metric
  /clear      - Clear screen
  /help       - Show this help
  /quit       - Exit

Example queries:
  "show me dog toys"
  "add p5 to my cart"          (use product ID from search results)
  "what's in my cart?"
  "checkout"
""")


async def print_connectors():
    """Show the SDK's external-service connectors: each one's connection mode
    (local_docker / cloud / custom / disabled) plus a live health probe.

    This exercises the connector module end to end — get_container().connectors,
    each connector's describe(), and health_check_all() — from inside the shop.
    """
    from continuum.connectors import health_check_all
    from continuum.core.container import get_container

    connectors = get_container().connectors
    print("\nConnectors (configured connection per service):")
    for name in sorted(connectors):
        d = connectors[name].describe()
        host = d.get("host") or d.get("endpoint") or "-"
        print(
            f"  • {name:<13} mode={d['mode']:<12} "
            f"enabled={d['enabled']!s:<5} configured={d['configured']!s:<5} host={host}"
        )

    print("\nLive health probe (health_check_all):")
    report = await health_check_all()
    for name in sorted(report):
        r = report[name]
        print(f"  • {name:<13} {r['status']}")
    print()


async def print_health():
    """Show the session-persistence health check (#1) and the degraded gauge (#2),
    so the observability wiring is visible from inside the shop.
    """
    from continuum.core.health import HealthCheck
    from continuum.observability.metrics import get_metrics_collector

    result = await HealthCheck()._check_session_persistence()
    print(f"\n  session_persistence: {result.status.value} — {result.message}")
    print(f"  details: {result.details}")

    samples = get_metrics_collector()._custom_metrics.get("session_persistence_degraded", [])
    latest = samples[-1] if samples else "(not emitted yet)"
    print(f"  metric session_persistence_degraded = {latest}\n")


async def main():
    setup_logging(level=LogLevel.INFO)

    print("=" * 60)
    print("  Local Shop Agent — MCPServerStreamableHttp test")
    print("=" * 60)
    print()
    print("Make sure the MCP server is running:")
    print("  python server.py   (in another terminal)")
    print()

    user_id = input("Enter user ID (or Enter for auto): ").strip() or f"cli-{uuid.uuid4().hex[:8]}"
    # The agent takes user_id / conversation_id per chat() call, so we hold them
    # here for the lifetime of this CLI session.
    conversation_id = f"cli-conv-{uuid.uuid4().hex[:8]}"

    print("\nConnecting to local MCP server...")
    try:
        agent = await create_shop_agent()
        print("✓ Ready!")
        print(f"  User:         {user_id}")
        print(f"  Conversation: {conversation_id}")
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
                    elif cmd == "/session":
                        from continuum.core.container import get_container

                        print(f"\n  User ID:         {user_id}")
                        print(f"  Conversation ID: {conversation_id}")
                        scl = get_container().session_client
                        if scl is not None:
                            print(f"  Fallback mode:   {scl.config.fallback_mode}")
                            print(f"  Degraded:        {scl.persistence_degraded}")
                        else:
                            print("  Sessions:        disabled")
                        print()
                    elif cmd == "/connectors":
                        await print_connectors()
                    elif cmd == "/health":
                        await print_health()
                    elif cmd == "/clear":
                        os.system("clear")
                    else:
                        print("Unknown command. Type /help.")
                    continue

                print("\nThinking...")
                response = await agent.chat(
                    user_input, user_id=user_id, conversation_id=conversation_id
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
