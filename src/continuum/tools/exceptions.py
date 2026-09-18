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

    def __init__(self, message: str | None = None, *, commands: str | None = None, **kwargs: Any):
        """
        Args:
            commands: the shell commands that resolve *this* server, already
                shell-quoted. Carried so a caller holding several of these --
                ``ToolExecutor`` building a registry over multiple servers --
                can compose one refusal listing each server's own commands
                instead of repeating the shared preamble per server.

                A plain attribute, deliberately not ``context``: ``__str__``
                renders every context entry inline as ``k=v``, so a multi-line
                command block there reappears mangled after "| Context:" --
                duplicating the commands in a form nobody can paste.
        """
        super().__init__(message, **kwargs)
        self.commands = commands


class ToolArgumentError(ToolError):
    """Raised when tool arguments do not satisfy the tool's declared schema.

    A tool's ``inputSchema`` reaches the model as a promise about what the tool
    accepts. For in-process tools that promise used to end there: the schema was
    advertised and then discarded, so a parameter declared ``{"type": "string"}``
    could receive a dict and hand it straight to the Python body (security
    finding F5). The model's arguments are attacker-influenced the moment any
    upstream tool result is, which makes an unenforced schema a description of
    the happy path rather than a boundary.

    Deliberately *not* an ``MCPError``: nothing is wrong with the server or the
    transport, and reporting it as an MCP fault sends the reader to the wrong
    layer. The argument is wrong, and the model is the one that can fix it --
    which is why callers envelope this as ``isError=True`` for the model to read
    and correct, rather than propagating it as a run-ending failure.
    """

    default_message = "Tool argument does not match the declared schema"
    default_error_code = "TOOL_ARGUMENT_ERROR"
