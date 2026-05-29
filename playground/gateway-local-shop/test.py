"""
Gateway integration test.

Runs a simple agent through the Smart Gateway without needing the MCP server.
Tests that:
  1. Requests route through the gateway (SMART_GATEWAY_URL)
  2. session_id / run_id are injected into body.metadata
  3. gateway_mode flows into x-portkey-router-mode header
  4. Stateless agent (no session_id) uses run_id as gateway session key
  5. Stateful agent (with session_id) uses session_id as gateway session key

Usage:
    python test.py

Requires:
    - Smart Gateway running:  docker compose --profile langfuse --profile owui up -d
      (in continuum-backend-smart-inference/)
"""

import asyncio
import os
import sys

# config.py sets SMART_GATEWAY_URL and SMART_GATEWAY_API_KEY before orchestrator imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import config  # noqa: F401 — side effect: sets os.environ gateway vars

from orchestrator import AgentRunner, BaseAgent, LogLevel, RunnerConfig, get_logger, setup_logging
from orchestrator.config import settings
from orchestrator.core.container import get_container
from orchestrator.core.lifecycle import get_lifecycle_manager

logger = get_logger(__name__)

DIVIDER = "─" * 60


def print_gateway_status() -> None:
    print(f"\n{DIVIDER}")
    print("  Smart Gateway Configuration")
    print(DIVIDER)
    print(
        f"  SMART_GATEWAY_URL  : {settings.smart_gateway_url or 'NOT SET — direct provider calls'}"
    )
    print(f"  SMART_GATEWAY_KEY  : {settings.smart_gateway_api_key or 'NOT SET'}")
    print(f"  DEFAULT_MODE       : {settings.smart_gateway_default_mode}")
    print(DIVIDER + "\n")


async def run_test(
    label: str,
    agent: BaseAgent,
    runner: AgentRunner,
    message: str,
    session_id: str | None = None,
    user_id: str = "test-user",
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  TEST: {label}")
    print(f"  model       : {agent.model}")
    print(
        f"  gateway_mode: {agent.gateway_mode or '(default: ' + settings.smart_gateway_default_mode + ')'}"
    )
    print(f"  session_id  : {session_id or '(none — will use run_id)'}")
    print(f"  message     : {message}")
    print("=" * 60)

    response = await runner.run(
        agent=agent,
        input=message,
        session_id=session_id,
        user_id=user_id,
    )

    print(f"\nResponse: {response.content}")
    print(f"Status  : {response.status}")


async def main() -> None:
    setup_logging(level=LogLevel.INFO)
    print_gateway_status()

    if not settings.smart_gateway_url:
        print("ERROR: SMART_GATEWAY_URL is not set. Cannot run gateway test.")
        return

    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=False,
        verify_connections=True,
        enable_signal_handlers=False,
    )
    await lifecycle.initialize()
    container = get_container()

    runner = AgentRunner(
        container=container,
        config=RunnerConfig(persist_state=False, default_max_turns=5),
    )

    # ------------------------------------------------------------------
    # Agent A: stateless, modest mode (default)
    # gateway metadata: session_id = run_id (auto-generated per run)
    # ------------------------------------------------------------------
    stateless_agent = BaseAgent(
        name="stateless-assistant",
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4o-mini",
        # gateway_mode=None → uses SMART_GATEWAY_DEFAULT_MODE ("modest")
    )

    # ------------------------------------------------------------------
    # Agent B: stateless, quality mode override
    # gateway metadata: session_id = run_id, x-portkey-router-mode = quality
    # ------------------------------------------------------------------
    quality_agent = BaseAgent(
        name="quality-assistant",
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4o-mini",
        gateway_mode="quality",
    )

    # ------------------------------------------------------------------
    # Agent C: stateful — session_id comes from Redis
    # gateway metadata: session_id = Redis session_id (stable across turns)
    # ------------------------------------------------------------------
    stateful_agent = BaseAgent(
        name="stateful-assistant",
        instructions="You are a helpful assistant. Answer concisely.",
        model="gpt-4o-mini",
        gateway_mode="modest",
    )

    session_id = None
    session_client = container.session_client if container else None
    if session_client and session_client.is_enabled:
        session_id = await session_client.get_or_create_session(
            user_id="test-user",
            conversation_id="gateway-test-conv",
        )
        logger.info(f"Created Redis session: {session_id}")

    runner.register_agent(stateless_agent)
    runner.register_agent(quality_agent)
    runner.register_agent(stateful_agent)

    # Run tests
    await run_test(
        label="Stateless agent, default (modest) mode",
        agent=stateless_agent,
        runner=runner,
        message="What is 7 times 8? One short sentence.",
    )

    await run_test(
        label="Stateless agent, quality mode override",
        agent=quality_agent,
        runner=runner,
        message="What is the capital of France? One word.",
    )

    if session_id:
        await run_test(
            label="Stateful agent, session_id from Redis",
            agent=stateful_agent,
            runner=runner,
            message="What is 3 + 3? One short sentence.",
            session_id=session_id,
        )
        # Second turn — gateway sees the same session_id, handover can track model continuity
        await run_test(
            label="Stateful agent, second turn (same session_id)",
            agent=stateful_agent,
            runner=runner,
            message="Now double the result you just gave me.",
            session_id=session_id,
        )
    else:
        print("\n(Skipping stateful tests — Redis session not available)")

    print(f"\n{DIVIDER}")
    print("  All gateway tests complete.")
    print(DIVIDER)

    await lifecycle.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
