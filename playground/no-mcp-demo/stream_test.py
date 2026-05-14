#!/usr/bin/env python3
"""
Option A — CLI streaming test.

Runs a single agent with run_stream() and prints tokens as they arrive.

Usage:
  python stream_test.py
  python stream_test.py "How does the internet work?"
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import default_config
from orchestrator import AgentConfig, AgentMemoryConfig, AgentRunner, BaseAgent, RunnerConfig, get_logger, setup_logging, LogLevel
from orchestrator.agent.types import EventType
from orchestrator.core.container import get_container
from orchestrator.core.lifecycle import get_lifecycle_manager

setup_logging(level=LogLevel.WARNING)
logger = get_logger(__name__)


async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain how the internet works."

    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=False,
        verify_connections=True,
        enable_signal_handlers=False,
    )
    await lifecycle.initialize()
    container = get_container()

    agent = BaseAgent(
        name="stream-agent",
        instructions=(
            "You are a knowledgeable assistant. "
            "Answer questions clearly and concisely."
        ),
        model=default_config.model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )

    runner = AgentRunner(
        container=container,
        config=RunnerConfig(persist_state=False, default_max_turns=5),
    )

    print(f"\nPrompt: {prompt}\n")
    print("-" * 60)

    async for event in runner.run_stream(agent=agent, input=prompt, user_id="stream-test"):
        if event.type == EventType.CONTENT_DELTA:
            print(event.data.get("content", ""), end="", flush=True)
        elif event.type == EventType.RUN_ERROR:
            print(f"\n[ERROR] {event.data.get('error')}")

    print("\n" + "-" * 60)

    await lifecycle.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
