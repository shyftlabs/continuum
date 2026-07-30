"""
Tools module for the Orchestrator SDK.

Provides MCP (Model Context Protocol) tool integration for extending
agent capabilities with external tools and resources.
"""

try:
    from .exceptions import MCPServerUnreviewedError
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
    from .pinning import (
        PIN_FORMAT_VERSION,
        ToolDiff,
        approve_tools,
        diff_catalogs,
        format_catalog_diff,
        format_tool_catalog,
        load_pins,
        save_pins,
        snapshot_tool_digests,
    )
    from .schema import (
        ensure_strict_json_schema,
        normalize_schema_for_llm,
    )
    from .types import (
        MCPToolArtifact,
        RunArtifacts,
        ToolChangeEvent,
        ToolContextConfig,
        ToolContextState,
        ToolContextVariable,
        ToolFilter,
        ToolFilterCallable,
        ToolFilterContext,
        ToolFilterStatic,
        ToolTrustConfig,
        TrustAction,
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
    # Tool trust: review, pinning, drift (F3)
    "MCPServerUnreviewedError",
    "PIN_FORMAT_VERSION",
    "ToolChangeEvent",
    "ToolTrustConfig",
    "TrustAction",
    "ToolDiff",
    "approve_tools",
    "diff_catalogs",
    "format_catalog_diff",
    "format_tool_catalog",
    "load_pins",
    "save_pins",
    "snapshot_tool_digests",
    # Tool context (session/state management)
    "ToolContextConfig",
    "ToolContextState",
    "ToolContextVariable",
    # MCP artifacts (per-run)
    "MCPToolArtifact",
    "RunArtifacts",
]
