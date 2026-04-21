"""
CLI interface for Memory Modes Demo.

Demonstrates the new provider-based memory architecture with all 4 isolation modes.

Features:
- Interactive chat with memory
- Direct memory operations (add, search, list, delete)
- User/agent/session switching to test isolation
- Memory info and status
"""

import asyncio
import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import MemoryModesDemoAgent
from config import default_config

from orchestrator import LogLevel, get_logger, setup_logging

logger = get_logger(__name__)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def print_info(text: str) -> None:
    """Print info message."""
    print(f"ℹ️  {text}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✅ {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"⚠️  {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"❌ {text}")


def print_memory_info(info: dict[str, Any]) -> None:
    """Print memory configuration info."""
    print_section("Memory Configuration")
    print(f"  Isolation Mode: {info.get('isolation_mode', 'unknown')}")
    print(f"  Enabled: {info.get('is_enabled', False)}")
    print(f"  Provider: {info.get('provider', 'unknown')}")
    print(f"  Vector Store: {info.get('vector_store_provider', 'unknown')}")
    print(
        f"  Embedder: {info.get('embedder_provider', 'unknown')}/{info.get('embedder_model', 'unknown')}"
    )
    print(f"  Embedding Dims: {info.get('embedding_dims', 'unknown')}")
    print(f"  Search Limit: {info.get('search_limit', 5)}")
    print_section("Current Scope")
    print(f"  User ID: {info.get('user_id', 'none')}")
    print(f"  Agent ID: {info.get('agent_id', 'none')}")
    session_id = info.get("session_id")
    print(f"  Session ID: {session_id[:12] if session_id else 'none'}...")
    print()


def print_memories(memories: list[dict[str, Any]]) -> None:
    """Print a list of memories."""
    if not memories:
        print_info("No memories found")
        return

    print_section(f"Memories ({len(memories)})")
    for i, mem in enumerate(memories, 1):
        memory_text = mem.get("memory", "N/A")
        score = mem.get("score", "N/A")
        mem_id = mem.get("id", "N/A")[:8]
        print(f"  {i}. [{mem_id}...] {memory_text}")
        if score and score != "N/A":
            print(f"     Score: {score:.3f}")
    print()


def print_help() -> None:
    """Print help message."""
    print_header("Available Commands")

    print("Chat Commands:")
    print("  Just type your message to chat with the agent")
    print()

    print("Memory Commands:")
    print("  /add <text>         - Add a memory directly")
    print("  /search <query>     - Search memories")
    print("  /list               - List all memories for current scope")
    print("  /delete-all         - Delete all memories for current scope")
    print("  /history            - Show session chat history (Redis)")
    print()

    print("Context Commands:")
    print("  /info or /status    - Show memory configuration")
    print("  /switchuser <id>    - Switch to different user")
    print("  /switchagent <id>   - Switch to different agent")
    print("  /newsession         - Start new chat session")
    print()

    print("Other Commands:")
    print("  /clear              - Clear screen")
    print("  /help               - Show this help")
    print("  /quit or /exit      - Exit")
    print()

    print_section("Memory Isolation Modes")
    print("  shared  - All memories accessible to everyone")
    print("  user    - Memories isolated per user (default)")
    print("  agent   - Memories isolated per agent")
    print("  run     - Memories isolated per session")
    print()
    print("  Set via: export MEMORY_ISOLATION=<mode>")
    print()


async def main():
    """Main CLI loop."""
    setup_logging(level=LogLevel.WARNING)

    print_header("🧠 Memory Modes Demo - Provider-Based Architecture")

    # Show current mode
    memory_isolation = os.environ.get("MEMORY_ISOLATION", "user")
    print_info(f"Memory Isolation Mode: {memory_isolation}")
    print_info("Change with: export MEMORY_ISOLATION=<shared|user|agent|run>")
    print()

    # Get initial IDs
    user_id = input("Enter user ID (or Enter for auto): ").strip() or None
    agent_id = input("Enter agent ID (or Enter for default): ").strip() or None

    # Initialize
    print_info("Initializing...")
    agent = MemoryModesDemoAgent(config=default_config)

    try:
        await agent.initialize(user_id=user_id, agent_id=agent_id)
        print_success("Agent initialized!")

        # Show info
        memory_info = await agent.get_memory_info()
        print_memory_info(memory_info)

        print_info("Type /help for commands or just chat!")
        print()

        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                # Exit commands
                if cmd in ("/quit", "/exit"):
                    print_info("Goodbye!")
                    break

                # Clear screen
                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")

                # Help
                elif cmd == "/help":
                    print_help()

                # Info/Status
                elif cmd in ("/info", "/status"):
                    info = await agent.get_memory_info()
                    print_memory_info(info)

                # Add memory
                elif cmd.startswith("/add "):
                    content = user_input[5:].strip()
                    if not content:
                        print_error("Usage: /add <memory content>")
                        continue
                    result = await agent.add_memory(content)
                    if result:
                        print_success(f"Memory added: {result.message}")
                        if result.results:
                            print_info(f"Created {len(result.results)} memory entries")
                    else:
                        print_error("Failed to add memory")

                # Search memories
                elif cmd.startswith("/search "):
                    query = user_input[8:].strip()
                    if not query:
                        print_error("Usage: /search <query>")
                        continue
                    result = await agent.search_memories(query)
                    if result:
                        print_memories([m.to_dict() for m in result.results])
                    else:
                        print_error("Search failed")

                # List memories
                elif cmd == "/list":
                    memories = await agent.get_all_memories()
                    print_memories(memories)

                # Session history
                elif cmd == "/history":
                    messages = await agent.get_session_history()
                    if not messages:
                        print_info("No session history (session may be disabled or empty).")
                    else:
                        print_section(f"Session History ({len(messages)} messages)")
                        for m in messages:
                            role = m["role"].upper()
                            content = m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"]
                            print(f"  [{role}] {content}")
                        print()

                # Delete all memories
                elif cmd == "/delete-all":
                    confirm = input("Are you sure? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        if await agent.delete_all_memories():
                            print_success("All memories deleted for current scope")
                        else:
                            print_error("Failed to delete memories")
                    else:
                        print_info("Cancelled")

                # Switch user
                elif cmd.startswith("/switchuser "):
                    new_user = user_input[12:].strip()
                    if not new_user:
                        print_error("Usage: /switchuser <user_id>")
                        continue
                    await agent.switch_user(new_user)
                    info = await agent.get_memory_info()
                    print_success(f"Switched to user: {new_user}")
                    print_memory_info(info)

                # Switch agent
                elif cmd.startswith("/switchagent "):
                    new_agent = user_input[13:].strip()
                    if not new_agent:
                        print_error("Usage: /switchagent <agent_id>")
                        continue
                    await agent.switch_agent(new_agent)
                    info = await agent.get_memory_info()
                    print_success(f"Switched to agent: {new_agent}")
                    print_memory_info(info)

                # New session
                elif cmd in ("/newsession", "/newchat"):
                    await agent.new_session()
                    info = await agent.get_memory_info()
                    print_success("New session started!")
                    print_memory_info(info)

                # Unknown command
                elif cmd.startswith("/"):
                    print_error(f"Unknown command: {cmd.split()[0]}")
                    print_info("Type /help for available commands")

                # Regular chat
                else:
                    print("Agent: ", end="", flush=True)
                    response = await agent.chat(user_input)
                    print(response)
                    print()

            except KeyboardInterrupt:
                print("\n")
                print_info("Interrupted")
                break
            except Exception as e:
                print_error(f"Error: {e}")
                logger.exception("CLI error")

    finally:
        print_info("Cleaning up...")
        await agent.cleanup()
        print_success("Done!")


if __name__ == "__main__":
    asyncio.run(main())
