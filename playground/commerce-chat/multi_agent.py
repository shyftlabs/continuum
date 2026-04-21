"""
Petco Multi-Agent System.

Plan-and-Execute pattern with Orchestrator (plans) and Executor (executes).
"""

import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents import build_tool_catalog, create_executor_agent, create_orchestrator_agent
from config import PetcoConfig, default_config

# Use absolute imports (same directory)
from schemas import ExecutionPlan, Intent

from orchestrator import (
    AgentRunner,
    BaseAgent,
    MCPServerStreamableHttp,
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager

logger = get_logger(__name__)


class PetcoMultiAgent:
    """
    Multi-Agent Petco Shopping Assistant.

    Uses Plan-and-Execute pattern:
    1. Orchestrator analyzes intent and creates execution plan
    2. Executor executes the plan using MCP tools
    """

    def __init__(self, config: PetcoConfig | None = None):
        """
        Initialize the multi-agent system.

        Args:
            config: Configuration for the agents
        """
        self.config = config or default_config

        # Container and Lifecycle (for SDK services)
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None

        # MCP-specific resources (not in Container)
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None

        # Agents
        self._orchestrator: BaseAgent | None = None
        self._executor: BaseAgent | None = None
        self._runner: AgentRunner | None = None

        # State
        self._tools: list[dict[str, Any]] = []
        self._initialized = False
        self._current_session_id: str | None = None
        self._current_user_id: str | None = None
        self._last_run_artifacts: dict[str, Any] | None = None

    async def initialize(self, user_id: str | None = None) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing Petco Multi-Agent System...")

        self._current_user_id = user_id or f"user_{generate_run_id()[-8:]}"

        # Initialize OrchestratorLifecycle for SDK services
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=True,
        )
        init_result = await self._lifecycle.initialize()

        if not init_result.success:
            logger.warning(f"Lifecycle initialization had issues: {init_result.errors}")
        else:
            logger.info("✓ OrchestratorLifecycle initialized")
            if init_result.warnings:
                logger.info(f"Warnings: {init_result.warnings}")

        # Get Container (DI) for client management
        self._container = get_container()
        logger.info("✓ Container (DI) initialized")

        # Access clients from container
        llm_client = self._container.llm_client
        memory_client = self._container.memory_client
        session_client = self._container.session_client

        logger.info(f"✓ LLM client: {'available' if llm_client else 'not available'}")
        logger.info(
            f"✓ Memory client: {'available' if memory_client and memory_client.is_enabled else 'not available'}"
        )
        logger.info(
            f"✓ Session client: {'available' if session_client and session_client.is_enabled else 'not available'}"
        )

        # Create session if session client is available
        if session_client and session_client.is_enabled:
            try:
                self._current_session_id = await session_client.get_or_create_session(
                    user_id=self._current_user_id,
                    conversation_id="petco-multi-agent",
                )
                logger.info(f"✓ Session initialized: {self._current_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")

        # Connect to MCP server and discover tools
        await self._connect_mcp()

        # Build tool catalog for orchestrator
        tool_catalog = build_tool_catalog(self._tools)
        logger.info(f"✓ Tool catalog built with {len(self._tools)} tools")

        # Create agents
        self._orchestrator = create_orchestrator_agent(
            tool_catalog=tool_catalog,
            model=self.config.orchestrator_model or self.config.agent_model,
            temperature=self.config.orchestrator_temperature,
            memory_client=memory_client,
            enable_memory=self.config.enable_memory,
            memory_search_limit=self.config.memory_search_limit,
        )

        self._executor = create_executor_agent(
            tools=self._tools,
            tool_executor=self._tool_executor,
            model=self.config.executor_model or self.config.agent_model,
            temperature=self.config.executor_temperature,
        )

        # Create runner using Container (DI)
        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(
                persist_state=False,
                default_max_turns=self.config.max_turns,
            ),
        )

        # Register both agents
        self._runner.register_agent(self._orchestrator)
        self._runner.register_agent(self._executor)

        logger.info("✓ Runner initialized with Container (DI)")

        # Tracing is handled by Container and Lifecycle
        if self._container.has_langfuse_client():
            logger.info("✓ Langfuse tracing enabled via Container")

        self._initialized = True
        logger.info("✓ Petco Multi-Agent System ready!")

    async def _connect_mcp(self) -> None:
        """Connect to MCP server and discover tools."""
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")

        try:
            # Create MCP StreamableHttp connection
            self._mcp_server = MCPServerStreamableHttp(
                {
                    "url": self.config.mcp_url,
                    "timeout": self.config.mcp_timeout,
                    "sse_read_timeout": self.config.mcp_sse_timeout,
                }
            )

            # Connect
            await self._mcp_server.connect()
            logger.info("✓ Connected to MCP server")

            # Discover tools - get as ToolDefinition dicts for LLM
            tool_definitions = await MCPUtil.get_function_tools(self._mcp_server)

            # Convert to dict format if needed
            self._tools = []
            for tool in tool_definitions:
                if isinstance(tool, dict):
                    self._tools.append(tool)
                elif hasattr(tool, "model_dump"):
                    # Pydantic model
                    self._tools.append(tool.model_dump())
                elif hasattr(tool, "to_dict"):
                    self._tools.append(tool.to_dict())
                else:
                    # Try to access as dict-like
                    self._tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": getattr(tool, "name", str(tool)),
                                "description": getattr(tool, "description", ""),
                                "parameters": getattr(tool, "parameters", {}),
                            },
                        }
                    )

            logger.info(f"✓ Discovered {len(self._tools)} tools from MCP")

            # Log tool names
            tool_names = []
            for t in self._tools:
                if isinstance(t, dict):
                    tool_names.append(t.get("function", {}).get("name", "?"))
                else:
                    tool_names.append(getattr(t, "name", "?"))
            logger.info(
                f"  Tools: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}"
            )

            # Create tool executor
            self._tool_executor = ToolExecutor({self._mcp_server: None})
            await self._tool_executor.initialize()

        except Exception as e:
            logger.error(f"Failed to connect to MCP: {e}")
            raise

    async def chat(self, message: str, user_id: str | None = None) -> str:
        """
        Process user message through multi-agent pipeline.

        Flow:
        1. Orchestrator analyzes and creates ExecutionPlan
        2. If respond_directly=True, return direct response
        3. Otherwise, Executor executes the plan

        Args:
            message: User's message
            user_id: Optional user ID (uses initialized user_id if not provided)

        Returns:
            Agent's response
        """
        if not self._initialized:
            await self.initialize(user_id=user_id)

        # Handle per-request user_id - get or create session for this user
        effective_user_id = user_id or self._current_user_id
        effective_session_id = self._current_session_id

        # If user_id is different, get/create session for that user
        if user_id and user_id != self._current_user_id:
            session_client = self._container.session_client if self._container else None
            if session_client and session_client.is_enabled:
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
                    effective_session_id = self._current_session_id

        try:
            # Step 1: Orchestrator creates plan
            logger.info("📋 Orchestrator analyzing request...")

            orchestrator_response = await self._runner.run(
                agent=self._orchestrator,
                input=message,
                session_id=effective_session_id,
                user_id=effective_user_id,
            )

            # Parse execution plan
            plan: ExecutionPlan | None = orchestrator_response.structured_output

            if plan is None:
                # Fallback: Try to parse JSON from content if structured output failed
                logger.warning(
                    "Structured output not available, attempting to parse JSON from content"
                )

                # Check if content exists and is not empty
                content = orchestrator_response.content if orchestrator_response.content else ""
                if not content or not content.strip():
                    logger.error(
                        "Orchestrator returned empty content - cannot create execution plan"
                    )
                    return "I'm sorry, I couldn't process your request. Please try rephrasing your message."

                logger.debug(f"Content to parse (length: {len(content)}): {content[:500]}")
                plan = self._parse_plan_from_content(content)

                if plan is None:
                    # If we can't parse, check if content looks like an error message
                    if (
                        content.strip()
                        .lower()
                        .startswith(("error", "sorry", "i'm sorry", "i cannot"))
                    ):
                        logger.warning("Orchestrator returned error message instead of plan")
                        return content
                    else:
                        logger.error(
                            "Failed to parse execution plan from content - returning error message"
                        )
                        return "I encountered an issue processing your request. Please try again or rephrase your message."
                else:
                    logger.info("✓ Successfully parsed ExecutionPlan from content JSON")

            logger.info(
                f"📋 Plan created: intent={plan.intent.value}, "
                f"steps={len(plan.steps)}, direct={plan.respond_directly}"
            )

            # Step 2: Check if direct response
            if plan.respond_directly and plan.direct_response:
                logger.info("✓ Orchestrator responding directly (no tools needed)")
                # Clear artifacts since no tools were executed
                self._last_run_artifacts = None
                return plan.direct_response

            # Clear artifacts at start of execution
            self._last_run_artifacts = None

            # Step 3: Executor executes the plan
            if not plan.steps:
                return (
                    plan.direct_response
                    or "I'm not sure how to help with that. Could you rephrase?"
                )

            logger.info(f"⚙️ Executor executing {len(plan.steps)} steps...")

            # Filter tools to only those in the plan (security: prevent unauthorized tool calls)
            plan_tools = self._get_tools_for_plan(plan)

            # Create temporary executor with ONLY plan tools
            # This ensures executor cannot call tools outside the plan
            temp_executor = create_executor_agent(
                tools=plan_tools,
                tool_executor=self._tool_executor,
                model=self.config.executor_model or self.config.agent_model,
                temperature=self.config.executor_temperature,
            )

            # Register temp executor for this run
            self._runner.register_agent(temp_executor)

            # Build executor input with plan details
            executor_input = self._build_executor_input(plan)

            executor_response = await self._runner.run(
                agent=temp_executor,
                input=executor_input,
                session_id=effective_session_id,
                user_id=effective_user_id,
            )

            logger.info("✓ Execution complete")

            # Get MCP session_id from tool executor context (captured from create_session tool)
            # This needs to be done AFTER execution so we can inject it into artifacts
            mcp_session_id = None
            if self._tool_executor and hasattr(self._tool_executor, "context_state"):
                context_state = self._tool_executor.context_state
                for namespace in context_state.get_all_namespaces():
                    captured_id = context_state.get(namespace, "session_id")
                    if captured_id:
                        mcp_session_id = captured_id
                        logger.info(
                            f"✅ Captured MCP session_id: {mcp_session_id[:8]}... (namespace: {namespace})"
                        )
                        break

            # Use MCP session_id for widgets, fallback to SDK session
            widget_session_id = mcp_session_id if mcp_session_id else effective_session_id

            # Store executor artifacts for later retrieval
            # run_artifacts are automatically attached to AgentResponse by runner
            # We'll store them in the class for API access
            if executor_response.run_artifacts:
                self._last_run_artifacts = executor_response.run_artifacts
                artifact_count = len(executor_response.run_artifacts.get("tool_artifacts", []))
                logger.info(f"📦 Stored {artifact_count} artifacts from executor")
            else:
                # Also check tool executor directly if not in response
                if self._tool_executor and hasattr(self._tool_executor, "run_artifacts"):
                    run_artifacts_obj = self._tool_executor.run_artifacts
                    if (
                        run_artifacts_obj
                        and hasattr(run_artifacts_obj, "is_empty")
                        and not run_artifacts_obj.is_empty()
                    ):
                        self._last_run_artifacts = (
                            run_artifacts_obj.to_dict()
                            if hasattr(run_artifacts_obj, "to_dict")
                            else None
                        )
                        artifact_count = (
                            len(self._last_run_artifacts.get("tool_artifacts", []))
                            if self._last_run_artifacts
                            else 0
                        )
                        logger.info(f"📦 Stored {artifact_count} artifacts from tool executor")

            # Inject MCP session_id into stored artifacts if available
            if (
                self._last_run_artifacts
                and mcp_session_id
                and "tool_artifacts" in self._last_run_artifacts
            ):
                widget_count = 0
                for artifact in self._last_run_artifacts["tool_artifacts"]:
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
                    logger.info(
                        f"💉 Injected MCP session_id into {widget_count} widget(s) in stored artifacts"
                    )

            return executor_response.content

        except Exception as e:
            logger.error(f"Error in multi-agent chat: {e}")
            return f"I encountered an error: {str(e)}. Please try again."

    def _parse_plan_from_content(self, content: str) -> ExecutionPlan | None:
        """
        Parse ExecutionPlan from JSON content when structured output fails.

        Handles cases where LLM returns JSON in code blocks or plain JSON.

        Args:
            content: Response content that may contain JSON

        Returns:
            Parsed ExecutionPlan or None if parsing fails
        """
        import json
        import re

        try:
            # Check if content is empty or whitespace only
            if not content or not content.strip():
                logger.warning("Cannot parse plan from empty content")
                return None

            # Try multiple strategies to extract JSON
            json_str = None

            # Strategy 1: Extract from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

            # Strategy 2: Find first JSON object (balanced braces)
            # This handles cases where there's text before/after JSON
            if not json_str:
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(content):
                    if char == "{":
                        if start_idx == -1:
                            start_idx = i
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            json_str = content[start_idx : i + 1]
                            break

            # Strategy 3: Try simple regex (less reliable but catches some cases)
            if not json_str:
                json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)

            # Strategy 4: Use entire content as JSON (last resort)
            if not json_str:
                json_str = content.strip()

            # Validate that we have something to parse
            if not json_str or not json_str.strip():
                logger.warning("No JSON content found to parse")
                return None

            # Clean up the JSON string (remove leading/trailing whitespace)
            json_str = json_str.strip()

            # Try to parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as parse_error:
                # If parsing fails, try to fix common issues
                # Remove any trailing commas before closing braces/brackets
                json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # If still fails, log and return None
                    logger.warning(f"JSON parsing failed even after cleanup: {parse_error}")
                    logger.debug(f"JSON string (first 500 chars): {json_str[:500]}")
                    raise

            # Handle Intent enum - convert string to Intent if needed
            if "intent" in data and isinstance(data["intent"], str):
                try:
                    data["intent"] = Intent(data["intent"])
                except ValueError:
                    # Try to find matching intent (case-insensitive)
                    intent_str = data["intent"].lower()
                    for intent in Intent:
                        if intent.value.lower() == intent_str:
                            data["intent"] = intent
                            break
                    else:
                        # Default to OTHER if no match
                        data["intent"] = Intent.OTHER

            # Handle ToolStep depends_on - convert to list if it's a string
            if "steps" in data and isinstance(data["steps"], list):
                for step in data["steps"]:
                    if "depends_on" in step:
                        if step["depends_on"] is None:
                            continue
                        elif isinstance(step["depends_on"], str):
                            step["depends_on"] = [step["depends_on"]]
                        elif not isinstance(step["depends_on"], list):
                            step["depends_on"] = None

            # Create ExecutionPlan from dict
            plan = ExecutionPlan(**data)
            logger.info("✓ Successfully parsed ExecutionPlan from content")
            return plan

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from content: {e}")
            logger.debug(
                f"Content length: {len(content)}, Content preview: {content[:200] if content else 'EMPTY'}"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to create ExecutionPlan from content: {e}", exc_info=True)
            logger.debug(
                f"Content length: {len(content) if content else 0}, Content preview: {content[:200] if content else 'EMPTY'}"
            )
            return None

    def _get_tools_for_plan(self, plan: ExecutionPlan) -> list[dict[str, Any]]:
        """
        Get only the tools specified in the execution plan.

        This ensures the executor can only call tools that are explicitly
        in the plan, preventing unauthorized tool calls.

        Args:
            plan: Execution plan with tool steps

        Returns:
            Filtered list of tool definitions
        """
        if not plan.steps:
            return []

        # Extract unique tool names from plan steps
        plan_tool_names = {step.tool_name for step in plan.steps}

        # Filter tools to only include those in the plan
        filtered_tools = []
        for tool in self._tools:
            tool_name = None
            if isinstance(tool, dict):
                tool_name = tool.get("function", {}).get("name")
            elif hasattr(tool, "name"):
                tool_name = tool.name

            if tool_name and tool_name in plan_tool_names:
                filtered_tools.append(tool)

        logger.info(
            f"🔒 Filtered tools for executor: {len(filtered_tools)} tools "
            f"(from {len(self._tools)} total): {', '.join(plan_tool_names)}"
        )

        return filtered_tools

    def _build_executor_input(self, plan: ExecutionPlan) -> str:
        """Build the input prompt for Executor from ExecutionPlan."""
        steps_text = ""
        for i, step in enumerate(plan.steps, 1):
            # Handle depends_on - can be string, list, or None
            deps_str = ""
            if step.depends_on:
                if isinstance(step.depends_on, str):
                    deps_str = f" (after: {step.depends_on})"
                elif isinstance(step.depends_on, list):
                    deps_str = f" (after: {', '.join(step.depends_on)})"

            params_str = ", ".join(f"{k}={v!r}" for k, v in step.parameters.items())
            steps_text += f"""
Step {i} [{step.step_id}]{deps_str}:
  Tool: {step.tool_name}
  Parameters: {{{params_str}}}
  Instruction: {step.instruction}
"""

        # Build tool list for prompt
        tool_names = [step.tool_name for step in plan.steps]
        tool_list = ", ".join(tool_names) if tool_names else "None"

        return f"""Execute the following plan:

## INTENT
{plan.intent.value}

## STEPS TO EXECUTE
{steps_text}

## USER CONTEXT (for personalization)
{plan.user_context or "No specific context"}

## RESPONSE INSTRUCTIONS
{plan.response_instructions}

## CRITICAL: TOOL RESTRICTION
You can ONLY call the tools that are EXPLICITLY mentioned in the plan steps above.
Available tools for this execution (ONLY these): {tool_list}
Calling ANY other tool is FORBIDDEN and will cause errors.

Execute each step in order, respecting dependencies. Use results from previous steps for dynamic values as described in instructions.
"""

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up Petco Multi-Agent System...")

        # Shutdown lifecycle (handles SDK services)
        if self._lifecycle:
            await self._lifecycle.shutdown()
            logger.info("✓ Cleanup complete")

    async def close(self) -> None:
        """Clean up resources (alias for cleanup for backward compatibility)."""
        await self.cleanup()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get list of available tools."""
        return self._tools

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._current_session_id

    @property
    def user_id(self) -> str | None:
        """Get current user ID."""
        return self._current_user_id

    @property
    def last_run_artifacts(self) -> dict[str, Any] | None:
        """Get artifacts from the last executor run (widgets, structured content, etc.)."""
        return self._last_run_artifacts


async def create_petco_multi_agent(
    user_id: str | None = None,
    config: PetcoConfig | None = None,
) -> PetcoMultiAgent:
    """
    Factory function to create and initialize a Petco multi-agent system.

    Args:
        user_id: Optional user ID for personalization
        config: Optional configuration

    Returns:
        Initialized PetcoMultiAgent
    """
    agent = PetcoMultiAgent(config)
    await agent.initialize(user_id)
    return agent
