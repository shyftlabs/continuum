"""
Tools module for the Orchestrator SDK.

Provides MCP (Model Context Protocol) tool integration for extending
agent capabilities with external tools and resources.
"""

try:
    from .executor import ToolExecutor
    from .mcp import (
        FunctionTool,
        MCPServer,
        MCPServerFunction,
        MCPServerSse,
        MCPServerSseParams,
        MCPServerStdio,
        MCPServerStdioParams,
        MCPServerStreamableHttp,
        MCPServerStreamableHttpParams,
        function_tool,
    )
    from .schema import (
        ensure_strict_json_schema,
        normalize_schema_for_llm,
    )
    from .types import (
        MCPToolArtifact,
        RunArtifacts,
        ToolContextConfig,
        ToolContextState,
        ToolContextVariable,
        ToolFilter,
        ToolFilterCallable,
        ToolFilterContext,
        ToolFilterStatic,
        create_static_tool_filter,
    )
    from .util import MCPUtil
except ImportError:
    pass

__all__ = [
    # MCP Server classes
    "MCPServer",
    "MCPServerFunction",
    "MCPServerSse",
    "MCPServerSseParams",
    "MCPServerStdio",
    "MCPServerStdioParams",
    "MCPServerStreamableHttp",
    "MCPServerStreamableHttpParams",
    # In-process function tools
    "FunctionTool",
    "function_tool",
    # Utilities
    "MCPUtil",
    "ToolExecutor",
    # Schema normalization
    "normalize_schema_for_llm",
    "ensure_strict_json_schema",
    # Tool filtering
    "ToolFilter",
    "ToolFilterCallable",
    "ToolFilterContext",
    "ToolFilterStatic",
    "create_static_tool_filter",
    # Tool context (session/state management)
    "ToolContextConfig",
    "ToolContextState",
    "ToolContextVariable",
    # MCP artifacts (per-run)
    "MCPToolArtifact",
    "RunArtifacts",
]
