"""
Tool executor for handling tool calls from LLM responses.

Provides utilities for executing MCP tools when the LLM requests them.
Includes rate limiting and concurrency control to prevent overwhelming
external services.

Also handles context variable capture and injection for session management
across tool calls (e.g., capturing session_id from create_session and
injecting into cart operations).
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from orchestrator.llm.types import ChatMessage, ToolCall
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import trace_tool
from orchestrator.tools.exceptions import MCPToolError
from orchestrator.tools.types import (
    MCPToolArtifact,
    RunArtifacts,
    ToolContextState,
)
from orchestrator.tools.util import MCPUtil

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from orchestrator.security.policy import PolicyStore
    from orchestrator.tools.mcp import MCPServer

logger = get_logger(__name__)


# Common variable names that should be auto-captured/injected
COMMON_CONTEXT_VARIABLES = {
    "session_id",
    "sessionId",
    "session",
    "auth_token",
    "token",
    "access_token",
    "authToken",
    "user_id",
    "userId",
    "merchant_id",
    "merchantId",
    "store_id",
    "storeId",
}

SENSITIVE_CONTEXT_VARIABLES = {
    "auth_token",
    "token",
    "access_token",
    "authToken",
    "bearer",
}


class ToolExecutorConfig:
    """Configuration for the ToolExecutor."""

    def __init__(
        self,
        max_concurrent_calls: int = 5,
        rate_limit_per_second: float = 10.0,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize tool executor configuration.

        Args:
            max_concurrent_calls: Maximum number of concurrent tool calls.
            rate_limit_per_second: Maximum tool calls per second (0 to disable).
            timeout_seconds: Timeout for individual tool calls.
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.rate_limit_per_second = rate_limit_per_second
        self.timeout_seconds = timeout_seconds


class RateLimiter:
    """Token bucket rate limiter for controlling tool call frequency."""

    def __init__(self, rate_per_second: float):
        """
        Initialize rate limiter.

        Args:
            rate_per_second: Maximum allowed calls per second.
        """
        self.rate_per_second = rate_per_second
        self.tokens = rate_per_second
        self.last_update: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire a token, waiting if necessary.

        This implements a token bucket algorithm where tokens are
        replenished at rate_per_second.
        """
        if self.rate_per_second <= 0:
            return

        # Calculate wait time inside lock, sleep outside to avoid lock contention
        wait_time = 0.0
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()

            if self.last_update is None:
                self.last_update = now
                self.tokens -= 1
                return

            elapsed = now - self.last_update
            self.last_update = now

            self.tokens = min(
                self.rate_per_second,
                self.tokens + elapsed * self.rate_per_second,
            )

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate_per_second
                self.tokens = 0
            else:
                self.tokens -= 1

        if wait_time > 0:
            await asyncio.sleep(wait_time)


class ToolExecutor:
    """
    Executes tool calls from LLM responses.

    Maps tool names to MCP servers and executes the tools, returning
    results as ChatMessage objects that can be added to the conversation.

    Also handles automatic context variable capture and injection for
    session management across tool calls.

    Example:
        ```python
        from orchestrator.tools import ToolExecutor, MCPServerStdio
        from orchestrator.llm import LLMClient, ChatMessage

        # Setup MCP server with context config
        server = MCPServerStdio({
            "command": "python",
            "args": ["mcp_server.py"]
        })
        await server.connect()

        # Create executor and initialize
        executor = ToolExecutor({server: None})  # None = all tools from server
        await executor.initialize()

        # Get LLM response with tool calls (use Container)
        from orchestrator.core.container import get_container
        llm = get_container().llm_client
        response = await llm.chat(messages, tools=tools)

        # Execute tool calls (context variables auto-captured/injected)
        if response.tool_calls:
            tool_messages = await executor.execute_tool_calls(
                response.tool_calls,
                trace_id=trace_id,
                span_id=span_id
            )
            # Add tool results to conversation
            messages.extend(tool_messages)
            # Continue conversation
            response = await llm.chat(messages)
        ```
    """

    def __init__(
        self,
        tool_registry: dict["MCPServer", list[str] | None] | None = None,
        config: ToolExecutorConfig | None = None,
        context_state: ToolContextState | None = None,
        namespace_tools: bool = False,
    ):
        """
        Initialize the tool executor.

        Args:
            tool_registry: Dictionary mapping MCP servers to lists of tool names.
                If a server maps to None, all tools from that server are available.
                If a server maps to a list, only those tool names are available.
                If None, registry will be empty and must be built via refresh_registry().
            config: Configuration for rate limiting and concurrency control.
            context_state: Shared context state for variable capture/injection.
                If None, a new empty state is created.
            namespace_tools: When True, registry keys are prefixed with the server name
                (e.g. "my-server__search"). Must match the namespace_tools setting used
                in MCPUtil.get_all_function_tools so the LLM-facing names align.
        """
        self.tool_registry: dict[str, tuple[MCPServer, MCPTool]] = {}
        self._tool_registry_config = tool_registry
        self._config = config or ToolExecutorConfig()
        self._namespace_tools = namespace_tools
        self._initialized = False

        # Context state for session management
        self._context_state = context_state or ToolContextState()

        # Run artifacts - cleared per run
        self._run_artifacts = RunArtifacts()

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_calls)
        self._rate_limiter = RateLimiter(self._config.rate_limit_per_second)

    @property
    def context_state(self) -> ToolContextState:
        """Get the context state for external access."""
        return self._context_state

    @context_state.setter
    def context_state(self, state: ToolContextState) -> None:
        """Set the context state."""
        self._context_state = state

    @property
    def run_artifacts(self) -> RunArtifacts:
        """Get run artifacts (cleared per run)."""
        return self._run_artifacts

    def clear_run_artifacts(self, run_id: str | None = None) -> None:
        """
        Clear all run artifacts.

        Call this at the start of each run to reset artifact collection.

        Args:
            run_id: Optional run ID to associate with new artifacts.
        """
        self._run_artifacts.clear()
        self._run_artifacts.run_id = run_id

    @staticmethod
    def _resolve_json_path(data: dict, path: str) -> Any:
        """Walk a dot-notation path through a nested dict. Returns None if any step is missing."""
        node: Any = data
        for part in path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node

    def _get_namespace(self, server: "MCPServer") -> str:
        """Get the namespace for a server (for context variable isolation)."""
        if server.context_config and server.context_config.namespace:
            return server.context_config.namespace
        return server.name

    @staticmethod
    def _validate_injection_type(value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected JSON schema type."""
        type_map: dict[str, tuple[type, ...]] = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "object": (dict,),
        }
        allowed_types = type_map.get(expected_type)
        if allowed_types is None:
            # Unknown schema type — allow injection (don't block on exotic types)
            return True
        return isinstance(value, allowed_types)

    def _inject_context_variables(
        self,
        server: "MCPServer",
        tool: "MCPTool",
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Inject context variables into tool arguments.

        Checks the tool's input schema for parameters that match stored
        context variables and injects them if not already provided.

        Args:
            server: The MCP server for namespace lookup
            tool: The MCP tool definition (has input schema)
            arguments: Current tool arguments from LLM

        Returns:
            Updated arguments with injected context variables
        """
        config = server.context_config
        namespace = self._get_namespace(server)

        # Get tool's expected parameters from schema
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})

        # Track what we inject for logging
        injected = []

        for param_name in properties.keys():
            # Check if we should inject this parameter
            should_inject = False

            # Check explicit config
            if config.should_inject(tool.name, param_name):
                should_inject = True
            # Check common variables
            elif config.auto_capture_common and param_name in COMMON_CONTEXT_VARIABLES:
                should_inject = True

            if not should_inject:
                continue

            # Check if we have a stored value
            stored_value = self._context_state.get(namespace, param_name)
            if stored_value is None:
                continue

            # Validate stored value type against schema before injection
            param_schema = properties.get(param_name, {})
            expected_type = param_schema.get("type")
            if expected_type and not self._validate_injection_type(stored_value, expected_type):
                logger.warning(
                    f"Skipping injection of {param_name}: stored value type "
                    f"{type(stored_value).__name__} does not match schema type '{expected_type}'"
                )
                continue

            # Check if LLM already provided a value
            if param_name in arguments:
                # Get variable config to check override behavior
                var_config = config.get_variable_config(param_name)
                override = var_config.override_llm_value if var_config else True

                if override:
                    # Override LLM value with stored value
                    if arguments[param_name] != stored_value:
                        logger.debug(
                            f"Overriding LLM-provided {param_name}='{arguments[param_name]}' "
                            f"with stored value '{stored_value}'"
                        )
                        arguments[param_name] = stored_value
                        injected.append(f"{param_name} (override)")
                else:
                    # Keep LLM value
                    continue
            else:
                # Inject stored value
                arguments[param_name] = stored_value
                injected.append(param_name)

        if injected:
            logger.info(f"💉 Injected context variables into {tool.name}: {', '.join(injected)}")

        return arguments

    def _capture_context_variables(
        self,
        server: "MCPServer",
        tool_name: str,
        result: str,
    ) -> None:
        """
        Capture context variables from tool result.

        Parses the tool result JSON and extracts any variables that
        should be captured based on the server's context config.

        Args:
            server: The MCP server for namespace lookup
            tool_name: Name of the tool that produced the result
            result: JSON string result from the tool
        """
        config = server.context_config
        namespace = self._get_namespace(server)

        # Parse result
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError) as e:
            # Log JSON parsing errors for debugging
            logger.warning(
                f"⚠️ Failed to parse tool result JSON for context capture ({tool_name}): {e}. "
                f"Result preview: {str(result)[:200]}"
            )
            return

        if not isinstance(data, dict):
            return

        # Track what we capture for logging
        captured = []

        # Handle variables with json_path first (explicit nested extraction)
        for var in config.variables:
            if var.json_path is None:
                continue
            if not config.should_capture(tool_name, var.name):
                continue
            value = self._resolve_json_path(data, var.json_path)
            if value is None:
                continue
            scope = config.get_scope(var.name)
            is_sensitive = var.name in SENSITIVE_CONTEXT_VARIABLES or var.sensitive
            self._context_state.set(namespace, var.name, value, scope=scope, sensitive=is_sensitive)
            display_value = "****" if is_sensitive else str(value)[:30]
            captured.append(f"{var.name}={display_value}")

        # Check each top-level field in the result
        json_path_names = {var.name for var in config.variables if var.json_path is not None}
        for key, value in data.items():
            if value is None:
                continue
            # Skip variables already captured via json_path
            if key in json_path_names:
                continue

            # Check if we should capture this variable
            should_capture = False
            scope = "session"

            # Check explicit config
            if config.should_capture(tool_name, key):
                should_capture = True
                scope = config.get_scope(key)
            # Check common variables
            elif config.auto_capture_common and key in COMMON_CONTEXT_VARIABLES:
                should_capture = True

            if not should_capture:
                continue

            is_sensitive = key in SENSITIVE_CONTEXT_VARIABLES
            self._context_state.set(namespace, key, value, scope=scope, sensitive=is_sensitive)
            display_value = "****" if is_sensitive else str(value)[:30]
            captured.append(f"{key}={display_value}")

        if captured:
            logger.info(f"📥 Captured context variables from {tool_name}: {', '.join(captured)}")

    async def initialize(self) -> None:
        """Initialize the tool registry from MCP servers.

        This must be called after creating the executor if tool_registry was provided.
        """
        if self._tool_registry_config:
            await self._build_registry(self._tool_registry_config)
        self._initialized = True

    def get_tool_definitions(
        self,
        normalize_schemas: bool = True,
        strict_mode: bool = False,
    ) -> list["ToolDefinition"]:
        """Return LLM-facing tool definitions derived from the already-built registry.

        Use this instead of MCPUtil.get_all_function_tools() to avoid a second
        list_tools() round-trip to the MCP server.  Must be called after initialize().

        Args:
            normalize_schemas: Whether to normalize schemas for LLM provider compatibility.
            strict_mode: Whether to apply strict mode (all properties required,
                no additional properties).

        Returns:
            List of ToolDefinition objects ready to pass to BaseAgent(tools=...).
        """
        from orchestrator.tools.util import MCPUtil

        if not self._initialized and self._tool_registry_config:
            raise RuntimeError(
                "get_tool_definitions() called before initialize(). "
                "Call await executor.initialize() first."
            )

        definitions = []
        for registry_key, (server, tool) in self.tool_registry.items():
            tool_def = MCPUtil.to_function_tool(tool, server, normalize_schemas, strict_mode)
            tool_def.function.name = registry_key
            definitions.append(tool_def)
        return definitions

    async def _build_registry(
        self,
        tool_registry: dict["MCPServer", list[str] | None],
        target: dict[str, tuple["MCPServer", "MCPTool"]] | None = None,
    ) -> None:
        """Build the internal tool name to (server, tool) mapping.

        Args:
            tool_registry: Mapping of servers to allowed tool lists.
            target: Dict to write into. Defaults to self.tool_registry.
        """
        dest = target if target is not None else self.tool_registry
        for server, allowed_tools in tool_registry.items():
            try:
                mcp_tools = await server.list_tools()
                for tool in mcp_tools:
                    # If allowed_tools is None, include all tools
                    # Otherwise, only include tools in the allowed list
                    if allowed_tools is None or tool.name in allowed_tools:
                        registry_key = (
                            f"{server.name}__{tool.name}"
                            if self._namespace_tools
                            else tool.name
                        )
                        if registry_key in dest:
                            logger.warning(f"Tool '{registry_key}' already registered, overwriting")
                        dest[registry_key] = (server, tool)
            except Exception as e:
                logger.error(f"Error building tool registry for server {server.name}: {e}")
                raise

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        trace_id: str | None = None,
        span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        policy_store: "PolicyStore | None" = None,
        subject: str | None = None,
    ) -> ChatMessage:
        """
        Execute a single tool call and return the result as a ChatMessage.

        Applies rate limiting and concurrency control to prevent overwhelming
        external services.

        Also handles context variable injection (before call) and capture (after call)
        for automatic session management.

        Args:
            tool_call: The tool call to execute.
            trace_id: Optional trace ID for Langfuse correlation.
            span_id: Optional parent span ID for nesting.
            metadata: Optional additional metadata.
            policy_store: Optional access control policy store. When provided,
                a "tool:<tool_name>" resource check is performed before execution.
            subject: Caller identity for policy evaluation (typically agent name).

        Returns:
            ChatMessage with role="tool" containing the tool result.

        Raises:
            MCPToolError: If the tool is not found or execution fails.
            ToolAccessDeniedError: If a policy denies the tool call.
        """
        tool_name = tool_call.function.name

        if tool_name not in self.tool_registry:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(error_msg)
            raise MCPToolError(
                error_msg,
                tool_name=tool_name,
            )

        server, tool = self.tool_registry[tool_name]

        # Access control check (deny-overrides policy engine)
        if policy_store is not None and subject is not None:
            from orchestrator.agent.exceptions import ToolAccessDeniedError

            decision = policy_store.check(subject, f"tool:{tool_name}")
            if not decision.allowed:
                raise ToolAccessDeniedError(
                    tool_name=tool_name,
                    policy_name=decision.policy_name,
                    subject=subject,
                    denial_message=decision.denial_message,
                )

        # Parse arguments
        try:
            arguments = (
                json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            )
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        # INJECT: Context variables into arguments before execution
        arguments = self._inject_context_variables(server, tool, arguments)

        # Re-serialize arguments for execution
        arguments_str = json.dumps(arguments)

        # Apply rate limiting and concurrency control
        await self._rate_limiter.acquire()

        async with self._semaphore:
            # Execute the tool with timeout
            try:
                result, artifact = await asyncio.wait_for(
                    MCPUtil.invoke_mcp_tool_with_artifact(
                        server,
                        tool,
                        arguments_str,
                        trace_id=trace_id,
                        span_id=span_id,
                        metadata=metadata,
                    ),
                    timeout=self._config.timeout_seconds,
                )

                # Store artifact (per-run)
                self._run_artifacts.add_artifact(artifact)

                # Extension point for subclasses (e.g. debug logging in playground projects)
                self._on_tool_result(tool_name, result, artifact)

                # CAPTURE: Context variables from result after execution
                # This may fail if result is not valid JSON, but that's okay - we continue
                try:
                    self._capture_context_variables(server, tool_name, result)
                except Exception as capture_error:
                    # Context capture failures should not break tool execution
                    # Log but continue - the tool result is still valid
                    logger.debug(
                        f"Context variable capture failed for tool '{tool_name}': {capture_error}",
                        extra={"tool_name": tool_name, "server_name": server.name},
                    )

                # When the MCP server reports isError=True the tool ran but produced
                # a failure. Wrap it in the same error envelope used by the exception
                # paths so the LLM receives a consistent structure it can act on.
                if artifact.is_error:
                    content = json.dumps(
                        {"error": result, "error_type": "MCPToolError", "is_error": True}
                    )
                else:
                    content = result

                # Return as ChatMessage with tool role
                return ChatMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call.id,
                )
            except TimeoutError:
                error_result = json.dumps(
                    {
                        "error": f"Tool '{tool_name}' timed out after {self._config.timeout_seconds}s",
                        "error_type": "TimeoutError",
                    }
                )
                logger.error(f"Tool execution timed out: {tool_name}")

                # Store error artifact
                error_artifact = MCPToolArtifact(
                    tool_name=tool_name,
                    server_name=server.name,
                    text_content=error_result,
                    is_error=True,
                    latency_ms=self._config.timeout_seconds * 1000,
                )
                self._run_artifacts.add_artifact(error_artifact)

                return ChatMessage(
                    role="tool",
                    content=error_result,
                    tool_call_id=tool_call.id,
                )
            except Exception as e:
                # Return error as tool message
                error_msg = str(e)
                error_type = type(e).__name__

                # Check if it's a JSON parsing error
                is_json_error = (
                    "JSON" in error_msg
                    or "json" in error_msg.lower()
                    or "Expecting value" in error_msg
                    or "Unterminated string" in error_msg
                    or "Expecting property name" in error_msg
                    or "JSONDecodeError" in error_type
                )

                if is_json_error:
                    logger.error(
                        f"❌ Tool execution failed due to invalid JSON response for '{tool_name}': {error_msg}",
                        extra={
                            "tool_name": tool_name,
                            "server_name": server.name,
                            "error_type": error_type,
                            "is_json_error": True,
                        },
                    )
                else:
                    logger.error(
                        f"Tool execution failed for '{tool_name}': {error_msg}",
                        extra={
                            "tool_name": tool_name,
                            "server_name": server.name,
                            "error_type": error_type,
                        },
                    )

                error_result = json.dumps({"error": error_msg, "error_type": error_type})

                # Store error artifact
                error_artifact = MCPToolArtifact(
                    tool_name=tool_name,
                    server_name=server.name,
                    text_content=error_result,
                    is_error=True,
                )
                self._run_artifacts.add_artifact(error_artifact)

                return ChatMessage(
                    role="tool",
                    content=error_result,
                    tool_call_id=tool_call.id,
                )

    @trace_tool(name="execute_tool_calls", tool_type="mcp", capture_output=True)
    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        trace_id: str | None = None,
        span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        policy_store: "PolicyStore | None" = None,
        subject: str | None = None,
    ) -> list[ChatMessage]:
        """
        Execute multiple tool calls and return results as ChatMessages.

        Args:
            tool_calls: List of tool calls to execute.
            trace_id: Optional trace ID for Langfuse correlation.
            span_id: Optional parent span ID for nesting.
            metadata: Optional additional metadata.
            policy_store: Optional access control policy store passed to each call.
            subject: Caller identity for policy evaluation (typically agent name).

        Returns:
            List of ChatMessage objects with role="tool" containing tool results.
        """
        # Execute all tool calls concurrently with return_exceptions=True
        # to prevent one tool failure from cancelling all other in-flight calls
        tasks = [
            self.execute_tool_call(
                tc,
                trace_id=trace_id,
                span_id=span_id,
                metadata=metadata,
                policy_store=policy_store,
                subject=subject,
            )
            for tc in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error ChatMessages so they can be reported to LLM
        from orchestrator.agent.exceptions import ToolAccessDeniedError

        processed: list[ChatMessage] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                tc = tool_calls[i]
                if isinstance(result, ToolAccessDeniedError):
                    # Policy denial is expected — log at INFO without traceback
                    logger.info(f"Tool '{tc.function.name}' denied by policy: {result}")
                    denial_message = result.context.get("denial_message", "")
                    if denial_message:
                        content = f"POLICY DENIED: {denial_message}"
                    else:
                        policy_name = result.context.get("policy_name", "unknown")
                        content = (
                            f"POLICY DENIED: '{tc.function.name}' was blocked by policy '{policy_name}'. "
                            "Inform the user this operation is not permitted."
                        )
                else:
                    logger.error(
                        f"Tool call '{tc.function.name}' failed: {result}",
                        exc_info=result,
                    )
                    content = f"Error executing tool '{tc.function.name}': {result}"
                processed.append(
                    ChatMessage(
                        role="tool",
                        content=content,
                        tool_call_id=tc.id,
                    )
                )
            else:
                processed.append(result)
        return processed

    def _on_tool_result(self, tool_name: str, result: str, artifact: Any) -> None:
        """Called after every successful tool execution. Override in subclasses for custom logging."""
        pass

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tool_registry.keys())

    async def refresh_registry(self, tool_registry: dict["MCPServer", list[str] | None]) -> None:
        """Refresh the tool registry from MCP servers.

        Builds into a temporary dict first, then swaps it in so concurrent
        tool calls keep hitting the old registry until the new one is ready.
        """
        temp: dict[str, tuple["MCPServer", "MCPTool"]] = {}
        try:
            await self._build_registry(tool_registry, target=temp)
        except Exception:
            # Build failed — old registry is untouched, tools remain available
            raise
        # Swap only after the temp dict is fully populated
        self.tool_registry = temp
