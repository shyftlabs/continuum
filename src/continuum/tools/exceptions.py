"""
Exceptions for the tools module.
"""

from typing import Any

from continuum.exceptions import OrchestratorError


class ToolError(OrchestratorError):
    """Base exception for tool-related errors."""

    default_message = "Tool error"
    default_error_code = "TOOL_ERROR"


class MCPError(ToolError):
    """Raised when MCP operations fail."""

    default_message = "MCP error"
    default_error_code = "MCP_ERROR"

    def __init__(
        self,
        message: str | None = None,
        *,
        server_name: str | None = None,
        tool_name: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if server_name:
            context["server_name"] = server_name
        if tool_name:
            context["tool_name"] = tool_name
        super().__init__(message, context=context, **kwargs)


class MCPConnectionError(MCPError):
    """Raised when MCP connection fails."""

    default_message = "MCP connection error"
    default_error_code = "MCP_CONNECTION_ERROR"


class MCPToolError(MCPError):
    """Raised when MCP tool invocation fails."""

    default_message = "MCP tool error"
    default_error_code = "MCP_TOOL_ERROR"


class MCPServerUnreviewedError(MCPError):
    """Raised when a server has no approved tool catalogue (security finding F3).

    Pinning cannot catch a server that was hostile from first contact -- pin the
    poison and you have pinned the poison. The only defence is a person reading
    the descriptions before they reach a prompt, so leaving that step optional
    would mean the one case with no automated defence is also the one case with
    no forced human step.

    Deliberately *not* an ``MCPConnectionError``: nothing is wrong with the
    connection, and reporting it as one sends the reader to the network.
    """

    default_message = "MCP server tool catalogue has not been reviewed"
    default_error_code = "MCP_SERVER_UNREVIEWED"
