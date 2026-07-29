"""
Utilities for MCP tool integration.
"""

import hashlib
import json
import re
import weakref
from typing import TYPE_CHECKING, Any

from continuum.llm.types import ToolDefinition
from continuum.logging import get_logger
from continuum.tools.exceptions import MCPError, MCPToolError
from continuum.tools.schema import normalize_schema_for_llm

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from .mcp import MCPServer
    from .types import MCPToolArtifact

logger = get_logger(__name__)

# Servers already reported for an auto-derived namespace prefix. Weak so a
# discarded server does not keep itself alive, and kept out of the server object
# so we never write attributes onto a user-supplied instance.
_WARNED_DERIVED_NAME_SERVERS: "weakref.WeakSet[Any]" = weakref.WeakSet()

# OpenAI and Anthropic both constrain function names to ^[a-zA-Z0-9_-]{1,64}$.
# Server names are auto-derived from the transport when not supplied explicitly
# (e.g. "sse: https://api.example.com/mcp", see MCPServerSse.__init__), so a
# naive f"{server.name}__{tool.name}" prefix produces an invalid name and the
# provider rejects the whole request. Everything below exists to guarantee the
# LLM-facing key satisfies that constraint.
MAX_TOOL_NAME_LENGTH = 64
NAMESPACE_SEPARATOR = "__"
_INVALID_NAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]+")
# Length of the disambiguating hash appended when a prefix must be truncated.
_TRUNCATION_HASH_LENGTH = 6


def sanitize_name_component(name: str) -> str:
    """Collapse runs of provider-invalid characters into a single underscore.

    Collapsing (rather than replacing per-character) also shortens the result --
    ``": https://"`` becomes ``"_"`` -- which keeps most auto-derived server
    names inside the length budget without truncation.
    """
    return _INVALID_NAME_CHARS_RE.sub("_", name).strip("_")


def build_namespaced_tool_name(server_name: str, tool_name: str) -> str:
    """Build a provider-valid ``<server>__<tool>`` name.

    The tool name is preserved verbatim -- it carries the semantics the model
    reasons about -- so the server prefix absorbs the length budget. When the
    prefix must be truncated, a short digest of the *full* server name is
    appended: two long URLs differing only in a late path segment (``/v2/`` vs
    ``/v3/``) would otherwise truncate to the same prefix and silently collide,
    reintroducing the very shadowing this namespacing exists to prevent.

    Args:
        server_name: The MCP server's name (may contain arbitrary characters).
        tool_name: The tool's name as reported by the server.

    Returns:
        A name matching ``^[a-zA-Z0-9_-]{1,64}$`` whenever ``tool_name`` itself
        already fits; an over-long ``tool_name`` is returned prefixed but
        unshortened, since truncating it would break dispatch.
    """
    prefix = sanitize_name_component(server_name)
    budget = MAX_TOOL_NAME_LENGTH - len(tool_name) - len(NAMESPACE_SEPARATOR)

    if budget <= 0:
        # The tool name alone exhausts the budget. Truncating it would change
        # what the model calls, so emit the prefixed name and let the provider
        # complain -- a loud 400 beats a silently mangled tool name.
        logger.warning(
            f"Tool name '{tool_name}' leaves no room for a server namespace "
            f"within {MAX_TOOL_NAME_LENGTH} characters; the provider may reject it."
        )
        return f"{prefix}{NAMESPACE_SEPARATOR}{tool_name}"

    if len(prefix) > budget:
        digest = hashlib.sha256(server_name.encode()).hexdigest()[:_TRUNCATION_HASH_LENGTH]
        keep = budget - _TRUNCATION_HASH_LENGTH - 1  # room for the "_" before the digest
        prefix = prefix[:keep].rstrip("_") + "_" + digest

    return f"{prefix}{NAMESPACE_SEPARATOR}{tool_name}"


class MCPUtil:
    """Set of utilities for interop between MCP and Orchestrator SDK tools."""

    @classmethod
    async def get_all_function_tools(
        cls,
        servers: list["MCPServer"],
        normalize_schemas: bool = True,
        strict_mode: bool = False,
        metadata: dict[str, Any] | None = None,
        namespace_tools: bool = True,
    ) -> list[ToolDefinition]:
        """Get all function tools from a list of MCP servers.

        Args:
            servers: List of MCP servers to get tools from.
            normalize_schemas: Whether to normalize schemas for LLM provider compatibility.
                Enabled by default. This ensures MCP tools work with any LLM provider
                (OpenAI, Gemini, Anthropic, etc.) by fixing common schema issues.
            strict_mode: Whether to apply strict mode (all properties required,
                no additional properties). Only applies if normalize_schemas=True.
            metadata: Optional metadata for tool filtering context.
            namespace_tools: When True (the default), prefix each tool name with its
                server name (e.g. "my-server__search") so tools from different servers
                cannot collide. This matches the MCP specification's guidance for
                clients that aggregate multiple servers, and the behaviour of Claude
                Code and Cursor. Set False for bare tool names -- then duplicate names
                across servers raise MCPError. Must match the namespace_tools setting
                used in ToolExecutor.

        Returns:
            List of ToolDefinition objects that can be used with LLMClient.

        Raises:
            MCPError: If duplicate tool names are found across servers and
                namespace_tools is False.
        """
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            # Fetch unprefixed and apply the namespace here, so the prefix is added
            # exactly once regardless of get_function_tools' own default.
            server_tools = await cls.get_function_tools(
                server, normalize_schemas, strict_mode, metadata, namespace_tools=False
            )
            if namespace_tools:
                cls._warn_if_server_name_is_derived(server)
                for tool_def in server_tools:
                    tool_def.function.name = build_namespaced_tool_name(
                        server.name, tool_def.function.name
                    )
            else:
                server_tool_names = {tool.function.name for tool in server_tools}
                if len(server_tool_names & tool_names) > 0:
                    raise MCPError(
                        f"Duplicate tool names found across MCP servers: "
                        f"{server_tool_names & tool_names}",
                        server_name=server.name,
                    )
                tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools

    @staticmethod
    def _warn_if_server_name_is_derived(server: "MCPServer") -> None:
        """Report a server whose namespace prefix was invented by the transport.

        Without name=, the transports fall back to a display label built from the
        URL or command ("streamable_http: http://localhost:8890/mcp"). Harmless
        while it only appeared in error messages; under namespacing it becomes the
        prefix on every LLM-facing tool name, which means:

        - tool identity carries the host and port, so moving the server to another
          port silently renames every tool and quietly breaks policy resources,
          digest pins, always_promote and capture/inject -- all exact-string
          matches that fail by doing nothing;
        - the prefix eats ~39 of the 64 characters providers allow, so longer tool
          names get hash-truncated into unreadable ids;
        - the model sees the transport and URL instead of a meaningful namespace.

        Warned once per server: registry rebuilds re-enter this path and the
        advice does not change.
        """
        if not getattr(server, "name_is_derived", False):
            return
        if server in _WARNED_DERIVED_NAME_SERVERS:
            return
        try:
            _WARNED_DERIVED_NAME_SERVERS.add(server)
        except TypeError:  # pragma: no cover - server not weak-referenceable
            pass
        logger.warning(
            "MCP server was created without name=, so its tools are namespaced with "
            "the auto-derived prefix %r. That prefix encodes the transport and URL, "
            "so tool names change when the server moves and are truncated to fit the "
            "64-character provider limit. Pass name='<short-stable-name>' to the "
            "server constructor.",
            sanitize_name_component(server.name),
        )

    @classmethod
    async def get_function_tools(
        cls,
        server: "MCPServer",
        normalize_schemas: bool = True,
        strict_mode: bool = False,
        metadata: dict[str, Any] | None = None,
        namespace_tools: bool = True,
    ) -> list[ToolDefinition]:
        """Get all function tools from a single MCP server.

        Args:
            server: The MCP server to get tools from.
            normalize_schemas: Whether to normalize schemas for LLM provider compatibility.
                Enabled by default. This ensures MCP tools work with any LLM provider.
            strict_mode: Whether to apply strict mode (all properties required,
                no additional properties). Only applies if normalize_schemas=True.
            metadata: Optional metadata for tool filtering context.
            namespace_tools: When True (the default), prefix each tool name with the
                server name. Defaults to True so this matches ToolExecutor -- pairing
                a non-namespaced tool list with a namespacing executor would leave the
                model calling names the registry cannot resolve.

        Returns:
            List of ToolDefinition objects.
        """
        from continuum.observability.provider_manager import get_provider_manager

        # Create a span for listing tools
        manager = get_provider_manager()
        span = None
        if manager.is_enabled:
            span = manager.span(
                name="mcp.list_tools",
                input={"server": server.name},
                metadata={
                    "server": server.name,
                    "normalize_schemas": normalize_schemas,
                    "strict_mode": strict_mode,
                },
                level="DEFAULT",
            )

        try:
            mcp_tools = await server.list_tools(metadata=metadata)
            tool_names = [tool.name for tool in mcp_tools]
            tool_definitions = [
                cls.to_function_tool(tool, server, normalize_schemas, strict_mode)
                for tool in mcp_tools
            ]
            if namespace_tools:
                cls._warn_if_server_name_is_derived(server)
                for tool_def in tool_definitions:
                    tool_def.function.name = build_namespaced_tool_name(
                        server.name, tool_def.function.name
                    )

            if span:
                # End span with output
                span.end(
                    output={"tool_count": len(tool_names), "tool_names": tool_names},
                    metadata={"server": server.name, "tool_count": len(tool_names)},
                )

            return tool_definitions
        except Exception as e:
            if span:
                span.end(
                    metadata={"error": str(e), "error_type": type(e).__name__},
                    level="ERROR",
                )
            raise

    @classmethod
    def to_function_tool(
        cls,
        tool: "MCPTool",
        server: "MCPServer",
        normalize_schemas: bool = True,
        strict_mode: bool = False,
    ) -> ToolDefinition:
        """Convert an MCP tool to a ToolDefinition.

        This method transforms MCP tool schemas into a format compatible with
        all major LLM providers (OpenAI, Gemini, Anthropic, etc.).

        Args:
            tool: The MCP tool to convert.
            server: The MCP server this tool belongs to.
            normalize_schemas: Whether to normalize schema for LLM compatibility.
                Default True. Fixes common issues like arrays without 'items',
                objects without 'properties', etc.
            strict_mode: Whether to apply strict mode transformations.
                When True, all properties are marked required and
                additionalProperties is set to false.

        Returns:
            A ToolDefinition that can be used with LLMClient.
        """
        from continuum.llm.types import FunctionDefinition

        # Note: We don't attach invoke_func to ToolDefinition because LLM providers
        # expect tools in OpenAI format (schema only). Tool execution is handled
        # separately via ToolExecutor.
        schema = tool.inputSchema or {}

        # Normalize schema for LLM provider compatibility
        if normalize_schemas:
            try:
                schema = normalize_schema_for_llm(schema, strict=strict_mode)
                logger.debug(f"Normalized schema for tool '{tool.name}' (strict={strict_mode})")
            except Exception as e:
                logger.warning(
                    f"Error normalizing schema for tool '{tool.name}': {e}. Using original schema."
                )
                # Fall back to minimal fixes
                if "properties" not in schema:
                    schema["properties"] = {}
        else:
            # Minimal fix even without normalization - OpenAI requires properties
            if "properties" not in schema:
                schema["properties"] = {}

        function_def = FunctionDefinition(
            name=tool.name,
            description=tool.description or "",
            parameters=schema,
        )

        return ToolDefinition(
            type="function",
            function=function_def,
        )

    @classmethod
    def _extract_mcp_content(cls, content_items: list[Any]) -> str:
        """
        Extract useful content from MCP tool response items.

        MCP tools often return nested structures with both widgets (for UI)
        and text content. This method extracts the actual data that's useful
        for LLM consumption.

        Args:
            content_items: List of MCP content items from CallToolResult.content

        Returns:
            JSON string with extracted useful content
        """
        if not content_items:
            return "[]"

        extracted_texts: list[str] = []

        for item in content_items:
            try:
                # Get item as dict
                item_dict = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                item_type = item_dict.get("type", "")

                # Handle text content
                if item_type == "text":
                    text_content = item_dict.get("text", "")

                    # Try to parse nested JSON in text field
                    # MCP servers often return nested JSON strings
                    if text_content:
                        try:
                            parsed = json.loads(text_content)
                            # If it's a list, extract text from each item
                            if isinstance(parsed, list):
                                for inner_item in parsed:
                                    if isinstance(inner_item, dict):
                                        inner_type = inner_item.get("type", "")
                                        inner_text = inner_item.get("text", "")

                                        # Skip widget placeholders, get actual data
                                        if inner_type == "text" and inner_text:
                                            # Check if it's actual data (JSON) or just a message
                                            try:
                                                # Try to parse as JSON to get actual data
                                                actual_data = json.loads(inner_text)
                                                extracted_texts.append(json.dumps(actual_data))
                                            except (json.JSONDecodeError, TypeError):
                                                # It's plain text
                                                extracted_texts.append(inner_text)
                                        elif inner_type not in ("widget", "ui"):
                                            # Other content types - include as is
                                            extracted_texts.append(json.dumps(inner_item))
                                    else:
                                        extracted_texts.append(str(inner_item))
                            elif isinstance(parsed, dict):
                                # Single dict - might be actual data
                                extracted_texts.append(json.dumps(parsed))
                            else:
                                extracted_texts.append(str(parsed))
                        except (json.JSONDecodeError, TypeError):
                            # Not JSON, use as plain text
                            extracted_texts.append(text_content)

                # Handle resource/embedded content (might have actual data)
                elif item_type in ("resource", "embedded"):
                    resource_data = item_dict.get("resource", {}) or item_dict.get("data", {})
                    if resource_data:
                        extracted_texts.append(json.dumps(resource_data))

                # Skip widgets - they're for UI only
                elif item_type in ("widget", "ui"):
                    logger.debug(f"Skipping widget/UI content type: {item_type}")
                    continue

                # Handle other content types - include as is
                else:
                    extracted_texts.append(json.dumps(item_dict))

            except Exception as e:
                logger.warning(f"Error extracting MCP content item: {e}")
                # Fall back to raw dump
                try:
                    if hasattr(item, "model_dump_json"):
                        extracted_texts.append(item.model_dump_json())
                    else:
                        extracted_texts.append(str(item))
                except Exception:
                    pass

        # Return as single string or JSON array
        if len(extracted_texts) == 0:
            return "[]"
        elif len(extracted_texts) == 1:
            return extracted_texts[0]
        else:
            # Multiple items - return as array if all are JSON
            try:
                # Try to parse each as JSON and return proper array
                parsed_items = []
                for text in extracted_texts:
                    try:
                        parsed_items.append(json.loads(text))
                    except (json.JSONDecodeError, TypeError):
                        parsed_items.append(text)
                return json.dumps(parsed_items)
            except Exception:
                return json.dumps(extracted_texts)

    @classmethod
    async def invoke_mcp_tool_with_artifact(
        cls,
        server: "MCPServer",
        tool: "MCPTool",
        input_json: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, "MCPToolArtifact"]:
        """Invoke an MCP tool and return both text result and full artifact.

        This method captures EVERYTHING the MCP returns - not just text for LLM:
        - meta: Widget templates, accessibility info, invocation messages
        - structuredContent: Actual data for rendering (items, counts, etc.)
        - content: Text content for LLM consumption

        Args:
            server: The MCP server to invoke the tool on.
            tool: The MCP tool to invoke.
            input_json: JSON string of tool arguments.
            trace_id: Optional trace ID for Langfuse correlation.
            span_id: Optional parent span ID for nesting.
            metadata: Optional additional metadata.

        Returns:
            Tuple of (text_output, MCPToolArtifact).

        Raises:
            MCPToolError: If tool invocation fails.
        """
        import time

        from continuum.observability.provider_manager import get_provider_manager
        from continuum.tools.types import MCPToolArtifact

        # Parse input JSON
        try:
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception as e:
            logger.debug(f"Invalid JSON input for tool {tool.name}: {input_json}")
            raise MCPToolError(
                f"Invalid JSON input for tool {tool.name}: {input_json}",
                server_name=server.name,
                tool_name=tool.name,
                original_error=e,
            ) from e

        logger.debug(f"Invoking MCP tool {tool.name} on server {server.name}")

        # Create span for tool call via provider manager
        manager = get_provider_manager()
        span = None
        start_time = time.time()

        if manager.is_enabled:
            span = manager.span(
                trace_id=trace_id,
                parent_observation_id=span_id,
                name=f"mcp.tool.{tool.name}",
                input={"tool": tool.name, "arguments": json_data, "server": server.name},
                metadata={
                    "tool_name": tool.name,
                    "server_name": server.name,
                    "tool_type": "mcp",
                    **(metadata or {}),
                },
                level="DEFAULT",
            )

        try:
            # Invoke the tool
            try:
                result = await server.call_tool(tool.name, json_data)
            except Exception as e:
                # Catch any errors from MCP server (including JSON parsing errors)
                error_msg = str(e)
                is_json_error = (
                    "JSON" in error_msg
                    or "json" in error_msg.lower()
                    or "Expecting value" in error_msg
                    or "Unterminated string" in error_msg
                    or "Expecting property name" in error_msg
                    or "JSONDecodeError" in type(e).__name__
                )

                if is_json_error:
                    logger.error(
                        f"❌ MCP server returned invalid JSON response for tool '{tool.name}': {error_msg[:300]}",
                        extra={
                            "tool_name": tool.name,
                            "server_name": server.name,
                            "error_type": type(e).__name__,
                            "is_json_error": True,
                        },
                    )
                # Re-raise to be handled by caller
                raise
            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"🔍 MCP tool {tool.name} raw response: "
                f"has_structuredContent={result.structuredContent is not None}, "
                f"has_meta={result.meta is not None}, "
                f"content_items={len(result.content) if result.content else 0}"
            )

            # Process text output for LLM
            if result.structuredContent and server.use_structured_content:
                sc = result.structuredContent
                if hasattr(sc, "model_dump"):
                    sc = sc.model_dump()
                elif hasattr(sc, "dict"):
                    sc = sc.dict()
                tool_output = json.dumps(sc)
            else:
                tool_output = cls._extract_mcp_content(result.content)
                logger.debug(
                    f"MCP tool {tool.name} using extracted content for LLM text output "
                    f"(no structuredContent available)"
                )

            # Capture EVERYTHING the MCP returned
            raw_content = None
            if result.content:
                try:
                    raw_content = [
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                        for item in result.content
                    ]
                except Exception:
                    pass

            sc_raw = result.structuredContent
            if sc_raw is not None:
                if hasattr(sc_raw, "model_dump"):
                    structured_content_dict = sc_raw.model_dump()
                elif hasattr(sc_raw, "dict"):
                    structured_content_dict = sc_raw.dict()
                else:
                    structured_content_dict = dict(sc_raw)
            else:
                structured_content_dict = None

            artifact = MCPToolArtifact(
                tool_name=tool.name,
                server_name=server.name,
                meta=dict(result.meta) if result.meta else None,
                structured_content=structured_content_dict,
                text_content=tool_output,
                raw_content=raw_content,
                is_error=result.isError if hasattr(result, "isError") else False,
                latency_ms=duration_ms,
            )

            if span:
                span.end(
                    output=tool_output,
                    metadata={
                        "tool_name": tool.name,
                        "server_name": server.name,
                        "duration_ms": duration_ms,
                        "success": True,
                        "has_meta": artifact.meta is not None,
                        "has_structured_content": artifact.structured_content is not None,
                        "has_widget": artifact.has_widget(),
                    },
                )

            logger.debug(
                f"MCP tool {tool.name} completed in {duration_ms:.2f}ms",
                extra={"server": server.name, "tool": tool.name, "duration_ms": duration_ms},
            )

            return tool_output, artifact

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error invoking MCP tool {tool.name} on server {server.name}: {e}",
                extra={"server": server.name, "tool": tool.name, "error": str(e)},
            )

            # End span with error
            if span:
                span.end(
                    metadata={
                        "tool_name": tool.name,
                        "server_name": server.name,
                        "duration_ms": duration_ms,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    level="ERROR",
                )

            raise MCPToolError(
                f"Error invoking MCP tool {tool.name}: {e}",
                server_name=server.name,
                tool_name=tool.name,
                original_error=e,
            ) from e

    @classmethod
    async def invoke_mcp_tool(
        cls,
        server: "MCPServer",
        tool: "MCPTool",
        input_json: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Invoke an MCP tool and return the result as a string.

        This method creates a Langfuse span for the tool invocation,
        including input, output, duration, and error tracking.

        For full artifact capture (meta, structuredContent, etc.),
        use invoke_mcp_tool_with_artifact() instead.

        Args:
            server: The MCP server to invoke the tool on.
            tool: The MCP tool to invoke.
            input_json: JSON string of tool arguments.
            trace_id: Optional trace ID for Langfuse correlation.
            span_id: Optional parent span ID for nesting.
            metadata: Optional additional metadata.

        Returns:
            JSON string of the tool result.

        Raises:
            MCPToolError: If tool invocation fails.
        """
        text_output, _ = await cls.invoke_mcp_tool_with_artifact(
            server, tool, input_json, trace_id, span_id, metadata
        )
        return text_output
