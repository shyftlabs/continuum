"""
Utilities for MCP tool integration.
"""

import json
from typing import TYPE_CHECKING, Any

from orchestrator.llm.types import ToolDefinition
from orchestrator.logging import get_logger
from orchestrator.tools.exceptions import MCPError, MCPToolError
from orchestrator.tools.schema import normalize_schema_for_llm

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from .mcp import MCPServer
    from .types import MCPToolArtifact

logger = get_logger(__name__)


class MCPUtil:
    """Set of utilities for interop between MCP and Orchestrator SDK tools."""

    @classmethod
    async def get_all_function_tools(
        cls,
        servers: list["MCPServer"],
        normalize_schemas: bool = True,
        strict_mode: bool = False,
        metadata: dict[str, Any] | None = None,
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

        Returns:
            List of ToolDefinition objects that can be used with LLMClient.

        Raises:
            MCPError: If duplicate tool names are found across servers.
        """
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(
                server, normalize_schemas, strict_mode, metadata
            )
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

    @classmethod
    async def get_function_tools(
        cls,
        server: "MCPServer",
        normalize_schemas: bool = True,
        strict_mode: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> list[ToolDefinition]:
        """Get all function tools from a single MCP server.

        Args:
            server: The MCP server to get tools from.
            normalize_schemas: Whether to normalize schemas for LLM provider compatibility.
                Enabled by default. This ensures MCP tools work with any LLM provider.
            strict_mode: Whether to apply strict mode (all properties required,
                no additional properties). Only applies if normalize_schemas=True.
            metadata: Optional metadata for tool filtering context.

        Returns:
            List of ToolDefinition objects.
        """
        from orchestrator.observability.provider_manager import get_provider_manager

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
        from orchestrator.llm.types import FunctionDefinition

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

        from orchestrator.observability.provider_manager import get_provider_manager
        from orchestrator.tools.types import MCPToolArtifact

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

            # LOG: Raw MCP response BEFORE processing (for debugging total/subtotal issues)
            logger.info(
                f"🔍 MCP tool {tool.name} raw response (BEFORE processing): "
                f"has_structuredContent={result.structuredContent is not None}, "
                f"has_meta={result.meta is not None}, "
                f"content_items={len(result.content) if result.content else 0}"
            )

            # Log structuredContent if it exists (especially for cart tools)
            if result.structuredContent:
                structured_dict = dict(result.structuredContent)
                # Log key fields that might contain totals
                total_fields = {}
                for key in [
                    "total",
                    "subtotal",
                    "total_cents",
                    "subtotal_cents",
                    "taxes",
                    "tax_cents",
                ]:
                    if key in structured_dict:
                        total_fields[key] = structured_dict[key]

                if total_fields:
                    logger.info(f"💰 MCP tool {tool.name} structuredContent totals: {total_fields}")

                # Log items count and sample item prices if it's a cart/list tool
                if "items" in structured_dict or "cart_items" in structured_dict:
                    items = structured_dict.get("items") or structured_dict.get("cart_items") or []
                    if isinstance(items, list) and len(items) > 0:
                        sample_item = items[0] if items else {}
                        item_price_fields = {
                            k: v
                            for k, v in sample_item.items()
                            if "price" in k.lower() or "total" in k.lower()
                        }
                        logger.info(
                            f"🛒 MCP tool {tool.name} cart data: "
                            f"items_count={len(items)}, "
                            f"sample_item_price_fields={item_price_fields}, "
                            f"full_structuredContent_keys={list(structured_dict.keys())}"
                        )

            # Process text output for LLM
            # IMPORTANT: If structuredContent exists, prefer it for LLM (has actual data like totals)
            # Otherwise, extract from content (which may be a text summary)
            if result.structuredContent:
                # Use structuredContent directly - it has the actual data
                try:
                    tool_output = json.dumps(result.structuredContent)
                except (TypeError, ValueError) as e:
                    # Handle case where structuredContent can't be serialized to JSON
                    logger.warning(
                        f"⚠️ Failed to serialize structuredContent to JSON for tool '{tool.name}': {e}. "
                        f"Falling back to content extraction.",
                        extra={
                            "tool_name": tool.name,
                            "server_name": server.name,
                            "error_type": type(e).__name__,
                        },
                    )
                    tool_output = cls._extract_mcp_content(result.content)

                # Log what we're sending to LLM (especially for cart tools with totals)
                if tool.name in ["get_cart", "cart", "get_cart_items"]:
                    structured_dict = dict(result.structuredContent)
                    llm_totals = {
                        k: v
                        for k, v in structured_dict.items()
                        if "total" in k.lower() or "subtotal" in k.lower() or "tax" in k.lower()
                    }
                    if llm_totals:
                        logger.info(
                            f"📤 MCP tool {tool.name} sending to LLM (structuredContent): totals={llm_totals}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ MCP tool {tool.name} structuredContent has NO totals for LLM! "
                            f"Keys: {list(structured_dict.keys())}"
                        )
            else:
                # Fall back to extracting from content
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

            # Create artifact with structured content
            structured_content_dict = (
                dict(result.structuredContent) if result.structuredContent else None
            )

            # LOG: What we're storing in artifact (for debugging)
            if structured_content_dict and tool.name in ["get_cart", "cart", "get_cart_items"]:
                artifact_totals = {
                    k: v
                    for k, v in structured_content_dict.items()
                    if "total" in k.lower() or "subtotal" in k.lower() or "tax" in k.lower()
                }
                if artifact_totals:
                    logger.info(
                        f"💾 MCP tool {tool.name} artifact structured_content totals: {artifact_totals}"
                    )

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

            # End span with success
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

            # LOG: Final artifact before returning (verify totals are preserved)
            if artifact.structured_content and tool.name in ["get_cart", "cart", "get_cart_items"]:
                final_totals = {
                    k: v
                    for k, v in artifact.structured_content.items()
                    if "total" in k.lower() or "subtotal" in k.lower() or "tax" in k.lower()
                }
                if final_totals:
                    logger.info(
                        f"✅ MCP tool {tool.name} FINAL artifact structured_content totals (BEFORE passing to LLM): {final_totals}"
                    )
                else:
                    logger.warning(
                        f"⚠️ MCP tool {tool.name} FINAL artifact structured_content has NO total/subtotal fields! "
                        f"Available keys: {list(artifact.structured_content.keys())}"
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
