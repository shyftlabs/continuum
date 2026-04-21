"""
Research & Report pipeline: orchestrates full flow and exercises all SDK features.

- Core: initialize_orchestrator, get_container, check_all_health, validate_configuration
- Session: get_or_create_session, runner uses log_to_session
- Memory: add, search, MemoryScope
- MCP: ToolExecutor + fake or external server
- Agents: Router, Sequential, Parallel, Loop, Handoff
- Structured output: ReportSummary from writer
- Streaming: run_stream once
- Optional: Temporal workflow + HITL
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure src and playground are on path
_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from typing import Any

from orchestrator import (
    AgentMemoryConfig,
    AgentRunner,
    RunnerConfig,
    check_all_health,
    get_container,
    get_lifecycle_manager,
    get_logger,
    initialize_orchestrator,
    shutdown_orchestrator,
    validate_configuration,
)
from orchestrator.agent.types import EventType, generate_run_id
from orchestrator.core import Container
from orchestrator.core.lifecycle import OrchestratorLifecycle

from playground.sdk_feature_test.agents.build import (
    build_fact_checker_agent,
    build_loop_agent,
    build_normal_agent,
    build_parallel_agents,
    build_react_agent,
    build_reflection_agent,
    build_research_route_agent,
    build_router_agent,
    build_sequential_agent,
    build_summarize_route_agent,
    build_two_pass_agent,
    build_qa_route_agent,
)
from playground.sdk_feature_test.config import SDKFeatureTestConfig
from playground.sdk_feature_test.mcp_fake import FakeMCPServer

logger = get_logger(__name__)

# Optional MCP tools import (ToolExecutor may be optional in minimal installs)
try:
    from orchestrator import MCPUtil, ToolExecutor
except ImportError:
    ToolExecutor = None  # type: ignore
    MCPUtil = None  # type: ignore


async def run_health_only(cfg: SDKFeatureTestConfig) -> bool:
    """Run only Core: lifecycle, container, health checks, validate_configuration. Returns True if all pass."""
    logger.info("Running health-only scenario...")
    validate_configuration()
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=cfg.fail_on_unhealthy,
        verify_connections=cfg.verify_connections,
    )
    init_result = await lifecycle.initialize()
    if not init_result.success:
        logger.warning(f"Lifecycle init issues: {init_result.errors}")
    container = get_container()
    health_result = await check_all_health()
    # Success when healthy or degraded (degraded = core services OK, optional e.g. Langfuse missing)
    ok = health_result.status.value in ("healthy", "degraded")
    if ok:
        logger.info(
            f"Health checks passed (status={health_result.status.value})"
        )
    else:
        logger.warning(f"Health checks: {health_result.status.value}")
    await lifecycle.shutdown()
    return ok


async def _connect_fake_mcp() -> tuple[FakeMCPServer, Any, Any]:
    """Connect to in-process fake MCP and return (server, tools_list, tool_executor)."""
    server = FakeMCPServer(name="sdk-feature-test-fake")
    await server.connect()
    tool_defs = await MCPUtil.get_function_tools(server)
    tools_list = [
        t.model_dump() if hasattr(t, "model_dump") else t
        for t in tool_defs
    ]
    executor = ToolExecutor({server: None})
    await executor.initialize()
    return server, tools_list, executor


async def run_workflow_only(cfg: SDKFeatureTestConfig, user_id: str, session_id: str | None) -> bool:
    """
    Run Core + Router + Sequential + Parallel + Loop. No MCP, no memory, no session, no handoff, no structured output, no stream.
    """
    from dataclasses import replace

    logger.info("Running workflow-only scenario...")
    validate_configuration()
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=cfg.fail_on_unhealthy,
        verify_connections=cfg.verify_connections,
    )
    await lifecycle.initialize()
    container = get_container()

    # Workflow-only: no memory, no session (agents use enable_memory=False, enable_session=False)
    cfg_wo = replace(cfg, enable_memory=False, enable_session=False)
    no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)

    # Build agents without MCP, memory, handoff, structured output
    fact_checker = build_fact_checker_agent(cfg_wo)
    sequential = build_sequential_agent(
        cfg_wo,
        tools=[],
        tool_executor=None,
        use_structured_output=False,
        with_handoff=False,
        fact_checker_agent=None,
    )
    parallel = build_parallel_agents(cfg_wo)
    loop = build_loop_agent(cfg_wo)
    router = build_router_agent(cfg_wo)
    research_agent = build_research_route_agent(cfg_wo)
    summarize_agent = build_summarize_route_agent(cfg_wo)
    qa_agent = build_qa_route_agent(cfg_wo)

    # SDK workflow factories don't accept memory_config; disable on composite agents so runner doesn't use memory
    router.memory_config = no_memory
    sequential.memory_config = no_memory
    parallel.memory_config = no_memory
    loop.memory_config = no_memory

    runner = AgentRunner(
        container=container,
        config=RunnerConfig(default_max_turns=cfg.max_turns, persist_state=False),
    )
    runner.register_agent(router)
    runner.register_agent(research_agent)
    runner.register_agent(summarize_agent)
    runner.register_agent(qa_agent)
    runner.register_agent(sequential)
    runner.register_agent(parallel)
    runner.register_agent(loop)

    query = "What are three key benefits of renewable energy?"
    try:
        r = await runner.run(router, query, user_id=user_id, session_id=session_id)
        logger.info(f"Router response length: {len(r.content or '')}")
        r2 = await runner.run(sequential, query, user_id=user_id, session_id=session_id)
        logger.info(f"Sequential response length: {len(r2.content or '')}")
        r3 = await runner.run(parallel, r2.content or query, user_id=user_id, session_id=session_id)
        logger.info(f"Parallel response length: {len(r3.content or '')}")
        r4 = await runner.run(loop, r3.content or query, user_id=user_id, session_id=session_id)
        logger.info(f"Loop response length: {len(r4.content or '')}")
    except Exception as e:
        logger.exception("Workflow-only run failed")
        await lifecycle.shutdown()
        return False
    await lifecycle.shutdown()
    return True


async def run_full_pipeline(
    cfg: SDKFeatureTestConfig,
    user_id: str,
    session_id: str | None,
    include_temporal: bool = False,
) -> bool:
    """
    Run full scenario: Core, Session, Memory, MCP, Router, Sequential, Parallel, Loop, Handoff, Structured output, Streaming.
    Optionally run Temporal workflow + HITL if include_temporal and SDK has Temporal.
    """
    logger.info("Running full pipeline...")
    validate_configuration()
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=cfg.fail_on_unhealthy,
        verify_connections=cfg.verify_connections,
    )
    init_result = await lifecycle.initialize()
    if not init_result.success:
        logger.warning(f"Lifecycle init: {init_result.errors}")

    container = get_container()
    memory_client = container.memory_client if container else None
    session_client = container.session_client if container else None

    # Session
    effective_session_id = session_id
    if session_client and session_client.is_enabled and cfg.enable_session:
        try:
            effective_session_id = await session_client.get_or_create_session(
                user_id=user_id,
                conversation_id="sdk-feature-test-pipeline",
            )
            logger.info(f"Session: {effective_session_id[:12]}...")
        except Exception as e:
            logger.warning(f"Session creation: {e}")

    # Memory: seed and search
    if memory_client and memory_client.is_enabled and cfg.enable_memory:
        try:
            from orchestrator.memory import MemoryScope
            await memory_client.add(
                "User is testing the SDK feature test pipeline; prefer concise answers.",
                user_id=user_id,
                metadata={"source": "sdk_feature_test"},
            )
            search_results = await memory_client.search(
                "What does the user prefer?",
                user_id=user_id,
                limit=cfg.memory_search_limit,
            )
            count = (
                search_results.total_results
                if hasattr(search_results, "total_results") and search_results.total_results is not None
                else len(getattr(search_results, "results", []))
            )
            logger.info(f"Memory search returned {count} result(s)")
        except Exception as e:
            logger.warning(f"Memory seed/search: {e}")

    # MCP
    fake_server = None
    tools_list = []
    tool_executor = None
    if cfg.use_fake_mcp and MCPUtil and ToolExecutor:
        try:
            fake_server, tools_list, tool_executor = await _connect_fake_mcp()
            logger.info(f"MCP fake connected, {len(tools_list)} tools")
        except Exception as e:
            logger.warning(f"MCP fake connect: {e}")

    # Build agents with MCP, memory, handoff, structured output
    fact_checker = build_fact_checker_agent(cfg)
    sequential = build_sequential_agent(
        cfg,
        tools=tools_list,
        tool_executor=tool_executor,
        use_structured_output=True,
        with_handoff=True,
        fact_checker_agent=fact_checker,
    )
    parallel = build_parallel_agents(cfg)
    loop = build_loop_agent(cfg)
    router = build_router_agent(cfg)
    research_agent = build_research_route_agent(cfg)
    summarize_agent = build_summarize_route_agent(cfg)
    qa_agent = build_qa_route_agent(cfg)

    runner = AgentRunner(
        container=container,
        tool_executor=tool_executor,
        config=RunnerConfig(default_max_turns=cfg.max_turns, persist_state=False),
    )
    runner.register_agent(router)
    runner.register_agent(research_agent)
    runner.register_agent(summarize_agent)
    runner.register_agent(qa_agent)
    runner.register_agent(sequential)
    runner.register_agent(parallel)
    runner.register_agent(loop)
    runner.register_agent(fact_checker)

    query = "What are three key benefits of renewable energy? Summarize briefly."
    try:
        # Router
        r1 = await runner.run(
            router, query,
            user_id=user_id,
            session_id=effective_session_id,
        )
        logger.info(f"Router done: {len(r1.content or '')} chars")
        if r1.structured_output:
            logger.info(f"Structured output (router): {type(r1.structured_output)}")

        # Sequential (may use MCP tools and structured ReportSummary)
        r2 = await runner.run(
            sequential, query,
            user_id=user_id,
            session_id=effective_session_id,
        )
        logger.info(f"Sequential done: {len(r2.content or '')} chars")
        if r2.structured_output:
            logger.info(f"Structured output (writer): {r2.structured_output}")

        # Parallel
        r3 = await runner.run(
            parallel, r2.content or query,
            user_id=user_id,
            session_id=effective_session_id,
        )
        logger.info(f"Parallel done: {len(r3.content or '')} chars")

        # Loop
        r4 = await runner.run(
            loop, r3.content or query,
            user_id=user_id,
            session_id=effective_session_id,
        )
        logger.info(f"Loop done: {len(r4.content or '')} chars")

        # Streaming: one run_stream
        streamed_chars = 0
        async for event in runner.run_stream(
            research_agent,
            "Say hello in one short sentence.",
            user_id=user_id,
            session_id=effective_session_id,
        ):
            if event.type == EventType.CONTENT_DELTA and event.data:
                content = event.data.get("content", "")
                if content:
                    streamed_chars += len(content)
        logger.info(f"Streaming received {streamed_chars} content chars")

    except Exception as e:
        logger.exception("Full pipeline run failed")
        if fake_server:
            try:
                await fake_server.cleanup()
            except Exception:
                pass
        await lifecycle.shutdown()
        return False

    if fake_server:
        try:
            await fake_server.cleanup()
        except Exception:
            pass

    # Optional Temporal
    if include_temporal and cfg.enable_temporal:
        try:
            from orchestrator.temporal import (
                AgentWorkflow,
                get_agent_registry,
                get_temporal_client,
                WorkflowInput,
            )
            registry = get_agent_registry()
            registry.register(research_agent)
            registry.register(summarize_agent)
            client = get_temporal_client()
            wf_id = f"sdk-feature-test-{generate_run_id()[-8:]}"
            handle = await client.start_workflow(
                AgentWorkflow.run,
                WorkflowInput(
                    steps=[
                        {"type": "agent", "agent_name": "research"},
                        {"type": "agent", "agent_name": "summarize"},
                    ],
                    initial_input="Briefly: what is machine learning?",
                ),
                id=wf_id,
                task_queue="orchestrator-agents",
            )
            logger.info(f"Temporal workflow started: {wf_id}")
            # Wait for result with timeout so we don't hang if no worker is running
            temporal_wait_seconds = 90
            try:
                result = await asyncio.wait_for(handle.result(), timeout=temporal_wait_seconds)
                logger.info(f"Temporal workflow result: {getattr(result, 'status', 'ok')}")
            except asyncio.TimeoutError:
                logger.info(
                    f"Temporal workflow did not complete within {temporal_wait_seconds}s "
                    "(worker may not be running). Workflow is running; view at http://localhost:8233"
                )
        except ImportError:
            logger.info("Temporal not installed, skipping Temporal step")
        except Exception as e:
            logger.warning(f"Temporal workflow: {e}")

    await lifecycle.shutdown()
    return True


async def run_no_temporal(cfg: SDKFeatureTestConfig, user_id: str, session_id: str | None) -> bool:
    """Same as full pipeline but skip Temporal."""
    return await run_full_pipeline(cfg, user_id, session_id, include_temporal=False)


async def run_reasoning_patterns(
    cfg: SDKFeatureTestConfig, user_id: str, session_id: str | None
) -> bool:
    """
    Smoke-test the three new reasoning pattern features:
    1. Two-pass reasoning (reasoning_mode=True)
    2. ReAct mode (react_mode=True)
    3. ReflectionAgent (self-critique + retry)
    """
    from dataclasses import replace

    logger.info("Running reasoning-patterns scenario...")
    validate_configuration()
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=cfg.fail_on_unhealthy,
        verify_connections=cfg.verify_connections,
    )
    await lifecycle.initialize()
    container = get_container()

    cfg_noreg = replace(cfg, enable_memory=False, enable_session=False)

    # Connect fake MCP so react/normal agents have real tools to compare
    tools_list = []
    tool_executor = None
    fake_server = None
    if cfg.use_fake_mcp and MCPUtil and ToolExecutor:
        try:
            fake_server, tools_list, tool_executor = await _connect_fake_mcp()
            logger.info(f"MCP fake connected for reasoning patterns: {len(tools_list)} tools")
        except Exception as e:
            logger.warning(f"MCP fake connect failed, agents will have no tools: {e}")

    two_pass = build_two_pass_agent(cfg_noreg)
    react = build_react_agent(cfg_noreg, tools=tools_list, tool_executor=tool_executor)
    normal = build_normal_agent(cfg_noreg, tools=tools_list, tool_executor=tool_executor)
    reflection = build_reflection_agent(cfg_noreg)

    runner = AgentRunner(
        container=container,
        tool_executor=tool_executor,
        config=RunnerConfig(default_max_turns=cfg.max_turns, persist_state=False),
    )

    query = "What are three key benefits of renewable energy? Be concise."

    try:
        # 1. Two-pass reasoning
        r1 = await runner.run(two_pass, query, user_id=user_id)
        logger.info(f"[two-pass reasoning] response length: {len(r1.content or '')} | tokens: {r1.usage.total_tokens}")

        # 2. Normal mode (no react) — baseline
        r2 = await runner.run(normal, query, user_id=user_id)
        logger.info(f"[normal mode]        response length: {len(r2.content or '')} | tokens: {r2.usage.total_tokens}")
        logger.info(f"[normal mode]        answer: {(r2.content or '')[:300]}")

        # 3. ReAct mode — same query, same tools, with think() reasoning
        r3 = await runner.run(react, query, user_id=user_id)
        logger.info(f"[react mode]         response length: {len(r3.content or '')} | tokens: {r3.usage.total_tokens}")
        logger.info(f"[react mode]         answer: {(r3.content or '')[:300]}")

        # Token difference shows the cost of ReAct reasoning
        token_diff = r3.usage.total_tokens - r2.usage.total_tokens
        logger.info(f"[comparison]         ReAct used {token_diff:+d} tokens vs normal mode")

        # 4. ReflectionAgent — uses runner.run via execute() internally
        r4 = await reflection.execute(
            input_text=query,
            runner=runner,
            context=None,  # type: ignore[arg-type]
        )
        logger.info(f"[reflection agent]   response length: {len(r4.content or '')} | tokens: {r4.usage.total_tokens}")

    except Exception:
        logger.exception("Reasoning-patterns scenario failed")
        if fake_server:
            try:
                await fake_server.cleanup()
            except Exception:
                pass
        await lifecycle.shutdown()
        return False

    if fake_server:
        try:
            await fake_server.cleanup()
        except Exception:
            pass

    await lifecycle.shutdown()
    return True


async def run_hitl_pipeline(
    cfg: SDKFeatureTestConfig,
    user_id: str,
    session_id: str | None,
) -> bool:
    """
    Run a Temporal workflow with a Human-in-the-Loop approval gate.

    Flow:
      1. User picks which agent to run first (research / summarize / qa)
      2. That agent runs inside a Temporal workflow
      3. An approval step pauses the workflow and prompts the CLI user
      4. If approved the next agent runs; if rejected the workflow aborts

    Exercises: Temporal AgentWorkflow, approval step, HumanInLoopManager,
    signal/query, worker lifecycle, and all underlying SDK features.
    """
    logger.info("Running Human-in-the-Loop pipeline...")
    validate_configuration()
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=cfg.fail_on_unhealthy,
        verify_connections=cfg.verify_connections,
    )
    init_result = await lifecycle.initialize()
    if not init_result.success:
        logger.warning(f"Lifecycle init: {init_result.errors}")

    container = get_container()

    # ------------------------------------------------------------------
    # Build agents
    # ------------------------------------------------------------------
    research_agent = build_research_route_agent(cfg)
    summarize_agent = build_summarize_route_agent(cfg)
    qa_agent = build_qa_route_agent(cfg)
    agents_by_key = {
        "1": ("research", research_agent),
        "2": ("summarize", summarize_agent),
        "3": ("qa", qa_agent),
    }

    # ------------------------------------------------------------------
    # CLI: let the user choose the first agent and follow-up agent
    # ------------------------------------------------------------------
    print("\n--- Human-in-the-Loop: Temporal Workflow with Approval Gate ---")
    print("Choose the FIRST agent to run:")
    print("  1. Research   – deep research on a topic")
    print("  2. Summarize  – summarize content")
    print("  3. Q&A        – answer questions about content")
    first_choice = input("First agent [1-3, default=1]: ").strip() or "1"
    if first_choice not in agents_by_key:
        first_choice = "1"
    first_name, first_agent = agents_by_key[first_choice]

    remaining = {k: v for k, v in agents_by_key.items() if k != first_choice}
    print("\nChoose the FOLLOW-UP agent (runs after your approval):")
    for k, (name, _) in sorted(remaining.items()):
        print(f"  {k}. {name.capitalize()}")
    second_choice = input(f"Follow-up agent [{'/'.join(sorted(remaining))}]: ").strip()
    if second_choice not in remaining:
        second_choice = sorted(remaining)[0]
    second_name, second_agent = remaining[second_choice]

    print("\nEnter your query (paste multi-line text, then press Enter on an empty line to finish):")
    print("(Or just press Enter for the default query)")
    query_lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and query_lines:
            break
        if line == "" and not query_lines:
            break
        query_lines.append(line)
    user_query = "\n".join(query_lines).strip()
    if not user_query:
        user_query = "What are three key benefits of renewable energy? Summarize briefly."

    print(f"\nWorkflow: {first_name} → [APPROVAL GATE] → {second_name}")
    print(f"Query: {user_query}\n")

    # ------------------------------------------------------------------
    # Temporal setup: registry, client, worker
    # ------------------------------------------------------------------
    try:
        from orchestrator.temporal import (
            AgentWorkflow,
            HumanInLoopManager,
            WorkflowInput,
            get_agent_registry,
            get_temporal_client,
            get_worker_manager,
        )
    except ImportError:
        logger.error("Temporal SDK not installed. Install with: pip install shyftlabs-continuum[temporal]")
        await lifecycle.shutdown()
        return False

    registry = get_agent_registry()
    registry.register(first_agent)
    registry.register(second_agent)

    client = get_temporal_client()
    if not client.is_connected:
        try:
            await client.connect()
        except Exception as e:
            logger.error(f"Cannot connect to Temporal: {e}")
            await lifecycle.shutdown()
            return False

    worker_mgr = get_worker_manager(client=client, registry=registry)
    await worker_mgr.start()
    logger.info("Temporal worker started")

    # ------------------------------------------------------------------
    # Start workflow: agent → approval → agent
    # ------------------------------------------------------------------
    wf_id = f"hitl-{generate_run_id()[-8:]}"
    try:
        handle = await client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                steps=[
                    {"type": "agent", "agent_name": first_name},
                    {
                        "type": "approval",
                        "description": (
                            f"Review the output of '{first_name}' before running '{second_name}'. "
                            "Approve to continue, reject to abort."
                        ),
                        "approvers": [user_id],
                        "timeout": 300,
                    },
                    {"type": "agent", "agent_name": second_name},
                ],
                initial_input=user_query,
                user_id=user_id,
                session_id=session_id,
            ),
            id=wf_id,
            task_queue="orchestrator-agents",
        )
        logger.info(f"HITL workflow started: {wf_id}")
    except Exception as e:
        logger.error(f"Failed to start HITL workflow: {e}")
        await worker_mgr.stop()
        await lifecycle.shutdown()
        return False

    # ------------------------------------------------------------------
    # Poll for the approval gate and present it to the user
    # ------------------------------------------------------------------
    hitl = HumanInLoopManager(client)
    approval_handled = False
    poll_interval = 2
    max_polls = 150  # up to 300 seconds

    for _ in range(max_polls):
        await asyncio.sleep(poll_interval)

        try:
            status = await hitl.get_workflow_status(wf_id)
        except Exception:
            continue

        wf_status = status.get("status", "")
        if wf_status in ("completed", "failed", "cancelled", "rejected", "timed_out"):
            break

        if wf_status == "waiting_for_approval" and not approval_handled:
            pending = await hitl.get_pending_approvals(wf_id)
            if not pending:
                continue

            req = pending[0]
            print("\n" + "=" * 60)
            print("  APPROVAL REQUIRED")
            print("=" * 60)
            print(f"  Workflow : {wf_id}")
            print(f"  Request  : {req.get('request_id', 'N/A')}")
            print(f"  Review   : {req.get('description', 'N/A')}")
            context = req.get("context", "")
            if context:
                preview = context[:500] + ("..." if len(context) > 500 else "")
                print(f"\n  --- Agent Output Preview ---\n{preview}\n  ---")
            print("\nOptions:")
            print("  a  = Approve (continue to next agent)")
            print("  r  = Reject  (abort workflow)")
            decision = input("Your decision [a/r, default=a]: ").strip().lower() or "a"

            request_id = req["request_id"]
            if decision == "r":
                reason = input("Rejection reason (optional): ").strip() or "Rejected by user"
                await hitl.reject(
                    workflow_id=wf_id,
                    request_id=request_id,
                    decided_by=user_id,
                    reason=reason,
                )
                print(f"\n  Rejected. Workflow will abort.")
            else:
                reason = input("Approval note (optional): ").strip() or "Approved by user"
                await hitl.approve(
                    workflow_id=wf_id,
                    request_id=request_id,
                    decided_by=user_id,
                    reason=reason,
                )
                print(f"\n  Approved! Workflow continuing to '{second_name}'...")

            approval_handled = True

    # ------------------------------------------------------------------
    # Collect final result
    # ------------------------------------------------------------------
    try:
        result = await asyncio.wait_for(handle.result(), timeout=120)
        print("\n" + "=" * 60)
        print("  WORKFLOW RESULT")
        print("=" * 60)
        print(f"  Status : {result.status}")
        if result.content:
            preview = result.content[:800] + ("..." if len(result.content) > 800 else "")
            print(f"  Output :\n{preview}")
        if result.approval_decisions:
            for ad in result.approval_decisions:
                print(f"  Decision: {ad.decision} by {ad.decided_by} — {ad.reason or ''}")
        print("=" * 60)
        logger.info(f"HITL workflow completed: status={result.status}")
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for HITL workflow result")
    except Exception as e:
        logger.warning(f"Error getting HITL workflow result: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    await worker_mgr.stop()
    logger.info("Temporal worker stopped")
    await lifecycle.shutdown()
    return True
