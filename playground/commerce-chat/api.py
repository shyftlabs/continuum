"""
FastAPI endpoint for Petco Retail Agent.

Exposes the Petco agent as a REST API with streaming support.
Uses a shared MCP connection to avoid connection lifecycle issues.
Session is managed by the backend (like CLI), not frontend.
"""

import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents import create_executor_agent
from config import default_config
from multi_agent import PetcoMultiAgent

from orchestrator import LogLevel, get_logger, setup_logging
from orchestrator.agent.types import EventType, generate_run_id

# Setup logging
setup_logging(level=LogLevel.INFO)
logger = get_logger(__name__)


# =============================================================================
# Global Multi-Agent Resources
# =============================================================================

# Global multi-agent system (initialized once at startup)
_multi_agent_resources: PetcoMultiAgent | None = None


# =============================================================================
# User Session Storage (like CLI - one session per user)
# =============================================================================


async def get_or_create_user_session(user_id: str) -> str:
    """
    Get or create a REAL session for a user (like CLI does).

    This uses SessionClient.get_or_create_session() which automatically handles
    session reuse - it will return the existing session if one exists for the
    user_id + agent_id combination, or create a new one if needed.

    The session is persisted in Redis and will be reused across requests.
    """
    global _multi_agent_resources

    # Get session client from multi-agent's container
    session_client = None
    if _multi_agent_resources and _multi_agent_resources._container:
        session_client = _multi_agent_resources._container.session_client

    # Always use SessionClient.get_or_create_session() - it handles reuse automatically
    # This matches how memory-modes-demo does it
    if session_client and session_client.is_enabled:
        session_id = await session_client.get_or_create_session(
            user_id=user_id,
            conversation_id="petco-multi-agent",
        )
        logger.info(f"✓ Session: {session_id[:8]}... for user {user_id}")
        return session_id
    else:
        # Fallback: generate a session ID (won't be persisted)
        session_id = f"session_{generate_run_id()[-8:]}"
        logger.warning(f"⚠️ Session client not available, using fallback: {session_id}")
        return session_id


# =============================================================================
# FastAPI App with Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - initialize multi-agent resources on startup."""
    global _multi_agent_resources

    # Initialize multi-agent resources
    _multi_agent_resources = PetcoMultiAgent(default_config)
    await _multi_agent_resources.initialize()
    logger.info("✓ Multi-agent resources initialized")

    yield

    # Cleanup
    if _multi_agent_resources:
        await _multi_agent_resources.close()
        logger.info("✓ Multi-agent resources cleaned up")


app = FastAPI(
    title="Petco Retail Agent API",
    description="AI-powered shopping assistant for Petco",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    content: str
    run_artifacts: dict[str, Any] | None = None
    session_id: str | None = None
    user_id: str | None = None


class AgentInfo(BaseModel):
    """Agent info for registration."""

    name: str
    description: str
    tools_count: int
    model: str


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Health check."""
    global _multi_agent_resources
    return {
        "status": "ok",
        "agent": "petco-multi-agent",
        "initialized": _multi_agent_resources is not None and _multi_agent_resources._initialized,
    }


@app.get("/info", response_model=AgentInfo)
async def get_agent_info():
    """Get agent information."""
    global _multi_agent_resources
    tools_count = len(_multi_agent_resources.tools) if _multi_agent_resources else 0
    return AgentInfo(
        name="petco-multi-agent",
        description="AI-powered shopping assistant for Petco retail (multi-agent system)",
        tools_count=tools_count,
        model=default_config.agent_model,
    )


@app.get("/memory/info")
async def get_memory_info():
    """Get memory configuration including isolation mode."""
    global _multi_agent_resources

    if not _multi_agent_resources or not _multi_agent_resources._container:
        raise HTTPException(status_code=503, detail="System not initialized")

    memory_client = _multi_agent_resources._container.memory_client

    if not memory_client or not memory_client.is_enabled:
        return {"enabled": False, "isolation_mode": None, "mode_display_name": "Disabled"}

    isolation_mode = memory_client.config.memory_isolation

    # Map isolation modes to display names
    mode_names = {
        "shared": "Shared Memory",
        "user": "User Memory",
        "agent": "Agent Memory",
        "run": "Session Memory",
    }

    return {
        "enabled": True,
        "isolation_mode": isolation_mode,
        "mode_display_name": mode_names.get(isolation_mode, isolation_mode),
        "provider": memory_client.config.provider,
        "search_limit": memory_client.config.search_limit,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the multi-agent system and get a response.

    Uses Plan-and-Execute pattern:
    - Orchestrator creates execution plan
    - Executor executes the plan

    Session is managed by the backend (like CLI).
    Frontend does NOT need to send session_id.
    """
    global _multi_agent_resources

    if not _multi_agent_resources or not _multi_agent_resources._initialized:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")

    try:
        # Get or create user_id - generate random if not provided
        user_id = request.user_id or f"user_{generate_run_id()[-8:]}"

        logger.info(f"📨 Chat request from {user_id}")

        # Run multi-agent chat with user_id
        response_content = await _multi_agent_resources.chat(request.message, user_id=user_id)

        # Get MCP session_id from tool executor context (captured from create_session tool)
        # This is captured AFTER tools are executed, so it should be available now
        mcp_session_id = None
        if _multi_agent_resources._tool_executor and hasattr(
            _multi_agent_resources._tool_executor, "context_state"
        ):
            context_state = _multi_agent_resources._tool_executor.context_state
            for namespace in context_state.get_all_namespaces():
                captured_id = context_state.get(namespace, "session_id")
                if captured_id:
                    mcp_session_id = captured_id
                    logger.info(
                        f"✅ Found MCP session_id: {mcp_session_id[:8]}... (namespace: {namespace})"
                    )
                    break

        # Use MCP session_id for widgets, fallback to SDK session
        session_id = _multi_agent_resources.session_id
        widget_session_id = mcp_session_id if mcp_session_id else session_id
        logger.info(
            f"📦 Using widget session_id: {widget_session_id[:8] if widget_session_id else 'N/A'}... (MCP: {mcp_session_id is not None})"
        )

        # Get run artifacts from last executor run
        run_artifacts = _multi_agent_resources.last_run_artifacts

        # Inject MCP session_id into widgets if artifacts exist
        if run_artifacts and "tool_artifacts" in run_artifacts:
            widget_count = 0
            for artifact in run_artifacts["tool_artifacts"]:
                has_widget = (
                    artifact.get("meta")
                    and isinstance(artifact["meta"], dict)
                    and artifact["meta"].get("openai/outputTemplate")
                )

                if (
                    has_widget
                    and "structured_content" in artifact
                    and isinstance(artifact["structured_content"], dict)
                ):
                    artifact["structured_content"]["session_id"] = widget_session_id
                    artifact["structured_content"]["sessionId"] = widget_session_id
                    artifact["structured_content"]["mcp_session_id"] = widget_session_id
                    widget_count += 1

            if widget_count > 0:
                logger.info(f"📦 Injected session_id into {widget_count} widget(s)")

        return ChatResponse(
            content=response_content or "",
            run_artifacts=run_artifacts,
            session_id=widget_session_id,
            user_id=user_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message to the multi-agent system and stream the response.

    Uses Plan-and-Execute pattern with streaming support.
    """
    global _multi_agent_resources

    if not _multi_agent_resources or not _multi_agent_resources._initialized:
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'type': 'error', 'error': 'Multi-agent system not initialized'})}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )

    async def generate() -> AsyncGenerator[str]:
        try:
            user_id = request.user_id or f"user_{generate_run_id()[-8:]}"
            session_id = _multi_agent_resources.session_id

            # Track MCP session_id and content
            mcp_session_id = None
            accumulated_content = []
            run_artifacts = None

            # For multi-agent, we need to handle orchestrator and executor separately
            # Since we can't easily stream through both agents, we'll use a hybrid approach:
            # 1. Run orchestrator (quick, no streaming needed)
            # 2. Stream executor execution

            # Handle per-request user_id - get or create session for this user
            effective_session_id = session_id
            session_client = (
                _multi_agent_resources._container.session_client
                if _multi_agent_resources._container
                else None
            )
            if user_id and session_client and session_client.is_enabled:
                try:
                    effective_session_id = await session_client.get_or_create_session(
                        user_id=user_id,
                        conversation_id="petco-multi-agent",
                    )
                    logger.info(
                        f"✓ Using session for user {user_id}: {effective_session_id[:8]}..."
                    )
                except Exception as e:
                    logger.warning(f"Failed to get session for user {user_id}: {e}")

            # Step 1: Run orchestrator (non-streaming, fast)
            logger.info("📋 Orchestrator analyzing request (streaming mode)...")
            orchestrator_response = await _multi_agent_resources._runner.run(
                agent=_multi_agent_resources._orchestrator,
                input=request.message,
                session_id=effective_session_id,
                user_id=user_id,
            )

            # Parse plan
            plan = orchestrator_response.structured_output
            if plan is None:
                # Check if content exists and is not empty
                content = orchestrator_response.content if orchestrator_response.content else ""
                if not content or not content.strip():
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Orchestrator returned empty response. Please try again.'})}\n\n"
                    return
                plan = _multi_agent_resources._parse_plan_from_content(content)

            if plan is None:
                # If parsing failed, check if content is an error message
                content = orchestrator_response.content if orchestrator_response.content else ""
                if content and content.strip():
                    error_msg = content.strip()[:200]  # Limit error message length
                    yield f"data: {json.dumps({'type': 'error', 'error': f'Failed to create execution plan: {error_msg}'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to create execution plan. Please try again.'})}\n\n"
                return

            # Check if direct response
            if plan.respond_directly and plan.direct_response:
                yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'user_id': user_id})}\n\n"
                yield f"data: {json.dumps({'type': 'content', 'content': plan.direct_response})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Step 2: Stream executor execution
            if not plan.steps:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No execution steps in plan'})}\n\n"
                return

            executor_input = _multi_agent_resources._build_executor_input(plan)

            # Filter tools for executor (only plan tools)
            plan_tools = _multi_agent_resources._get_tools_for_plan(plan)
            temp_executor = create_executor_agent(
                tools=plan_tools,
                tool_executor=_multi_agent_resources._tool_executor,
                model=_multi_agent_resources.config.executor_model
                or _multi_agent_resources.config.agent_model,
                temperature=_multi_agent_resources.config.executor_temperature,
            )
            _multi_agent_resources._runner.register_agent(temp_executor)

            # Start streaming - we'll update session_id after tools are called
            # Send initial start event with SDK session_id, will be updated if MCP session_id is captured
            yield f"data: {json.dumps({'type': 'start', 'session_id': effective_session_id, 'user_id': user_id})}\n\n"

            # Stream executor execution
            async for event in _multi_agent_resources._runner.run_stream(
                agent=temp_executor,
                input=executor_input,
                session_id=effective_session_id,
                user_id=user_id,
            ):
                if event.type == EventType.CONTENT_DELTA:
                    content_chunk = event.data.get("content", "")
                    if content_chunk:
                        accumulated_content.append(content_chunk)
                        yield f"data: {json.dumps({'type': 'content', 'content': content_chunk})}\n\n"

                elif event.type == EventType.TOOL_CALL_START:
                    tool_name = event.data.get("tool_name", "unknown")
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': tool_name, 'status': 'start'})}\n\n"

                elif event.type == EventType.TOOL_CALL_END:
                    tool_name = event.data.get("tool_name", "unknown")

                    # After tool call ends, check if MCP session_id was captured (especially from create_session)
                    if tool_name == "create_session" or not mcp_session_id:
                        if _multi_agent_resources._tool_executor and hasattr(
                            _multi_agent_resources._tool_executor, "context_state"
                        ):
                            context_state = _multi_agent_resources._tool_executor.context_state
                            for namespace in context_state.get_all_namespaces():
                                captured_id = context_state.get(namespace, "session_id")
                                if captured_id:
                                    mcp_session_id = captured_id
                                    widget_session_id = mcp_session_id
                                    logger.info(
                                        f"✅ Captured MCP session_id after {tool_name}: {mcp_session_id[:8]}... (namespace: {namespace})"
                                    )
                                    # Send updated session_id event for UI
                                    yield f"data: {json.dumps({'type': 'session_update', 'session_id': widget_session_id, 'mcp_session_id': mcp_session_id})}\n\n"
                                    break

                    yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': tool_name, 'status': 'end'})}\n\n"

                elif event.type == EventType.RUN_END:
                    # Final check for MCP session_id from tool executor context (after all tools have run)
                    if (
                        not mcp_session_id
                        and _multi_agent_resources._tool_executor
                        and hasattr(_multi_agent_resources._tool_executor, "context_state")
                    ):
                        context_state = _multi_agent_resources._tool_executor.context_state
                        for namespace in context_state.get_all_namespaces():
                            captured_id = context_state.get(namespace, "session_id")
                            if captured_id:
                                mcp_session_id = captured_id
                                logger.info(
                                    f"✅ Final MCP session_id capture: {mcp_session_id[:8]}... (namespace: {namespace})"
                                )
                                break

                    # Use MCP session_id for widgets, fallback to SDK session
                    widget_session_id = mcp_session_id if mcp_session_id else effective_session_id

                    # Get artifacts from multiple sources (executor response, tool executor, or stored)
                    run_artifacts = None

                    # Try executor response first (if available in event data)
                    if event.data and "run_artifacts" in event.data:
                        run_artifacts = event.data["run_artifacts"]

                    # Try tool executor directly
                    if (
                        not run_artifacts
                        and _multi_agent_resources._tool_executor
                        and hasattr(_multi_agent_resources._tool_executor, "run_artifacts")
                    ):
                        run_artifacts_obj = _multi_agent_resources._tool_executor.run_artifacts
                        if (
                            run_artifacts_obj
                            and hasattr(run_artifacts_obj, "is_empty")
                            and not run_artifacts_obj.is_empty()
                        ):
                            run_artifacts = (
                                run_artifacts_obj.to_dict()
                                if hasattr(run_artifacts_obj, "to_dict")
                                else None
                            )

                    # Fallback: try multi-agent's stored artifacts
                    if not run_artifacts:
                        run_artifacts = _multi_agent_resources.last_run_artifacts

                    # Inject MCP session_id into widgets if artifacts exist
                    if run_artifacts and "tool_artifacts" in run_artifacts:
                        widget_count = 0
                        for artifact in run_artifacts["tool_artifacts"]:
                            has_widget = (
                                artifact.get("meta")
                                and isinstance(artifact["meta"], dict)
                                and artifact["meta"].get("openai/outputTemplate")
                            )

                            if (
                                has_widget
                                and "structured_content" in artifact
                                and isinstance(artifact["structured_content"], dict)
                            ):
                                # Always inject MCP session_id (or fallback to SDK session_id)
                                artifact["structured_content"]["session_id"] = widget_session_id
                                artifact["structured_content"]["sessionId"] = widget_session_id
                                artifact["structured_content"]["mcp_session_id"] = widget_session_id
                                widget_count += 1
                                logger.debug(
                                    f"💉 Injected session_id {widget_session_id[:8]}... into widget "
                                    f"(tool: {artifact.get('tool_name', 'unknown')})"
                                )

                        if widget_count > 0:
                            logger.info(
                                f"📦 Injected session_id into {widget_count} widget(s) in stream (MCP: {mcp_session_id is not None})"
                            )

                    # Send artifacts
                    if run_artifacts:
                        yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': run_artifacts})}\n\n"

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                elif event.type == EventType.RUN_ERROR:
                    error_msg = event.data.get("error", "Unknown error")
                    yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                    return

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.delete("/session")
async def clear_session(user_id: str | None = None):
    """
    Clear session for a user and reset tool context.

    This will cause a new session to be created on the next request for this user.
    """
    global _multi_agent_resources

    # Clear tool executor's context state to force new MCP session
    if (
        _multi_agent_resources
        and _multi_agent_resources._tool_executor
        and hasattr(_multi_agent_resources._tool_executor, "context_state")
    ):
        context_state = _multi_agent_resources._tool_executor.context_state
        # Clear all namespaces
        for namespace in context_state.get_all_namespaces():
            context_state.clear_namespace(namespace)
            logger.info(f"🗑️ Cleared context namespace: {namespace}")

    # Note: We don't need to manually delete from _user_sessions anymore
    # because we always call get_or_create_session() which handles it.
    # If you want to force a new session, you could delete it from Redis here,
    # but for now we'll just clear the tool context.

    return {
        "status": "ok",
        "message": "Session context cleared. New session will be created on next request.",
    }


@app.post("/session/new")
async def create_new_session(user_id: str | None = None):
    """Create a new session for the current user (keeps user_id)."""
    global _multi_agent_resources

    if not _multi_agent_resources or not _multi_agent_resources._container:
        raise HTTPException(status_code=503, detail="System not initialized")

    session_client = _multi_agent_resources._container.session_client

    if not session_client or not session_client.is_enabled:
        return {"status": "ok", "message": "Session client not available"}

    # Force new session by clearing context
    if _multi_agent_resources._tool_executor and hasattr(
        _multi_agent_resources._tool_executor, "context_state"
    ):
        context_state = _multi_agent_resources._tool_executor.context_state
        for namespace in context_state.get_all_namespaces():
            context_state.clear_namespace(namespace)
            logger.info(f"🗑️ Cleared context namespace: {namespace}")

    logger.info(f"✓ New session created for user: {user_id}")
    return {"status": "ok", "message": "New session created"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
