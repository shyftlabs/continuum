"""
Type definitions for MCP tools.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol

import httpx
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


# =============================================================================
# MCP Tool Artifacts (full response capture - cleared per run)
# =============================================================================


@dataclass
class MCPToolArtifact:
    """
    Full artifact from an MCP tool call.

    Captures EVERYTHING the MCP returns - not just text for LLM:
    - meta: Widget templates, accessibility info, invocation messages
    - structured_content: Actual data for rendering (items, counts, etc.)
    - content: Text content for LLM consumption

    This is per-run data - cleared at the start of each agent run.

    Example:
        ```python
        artifact = MCPToolArtifact(
            tool_name="get_cart",
            server_name="petco-mcp",
            meta={
                "openai/outputTemplate": "ui://widget/cart.html",
                "openai/widgetAccessible": True,
            },
            structured_content={
                "items": [...],
                "subtotal": 29.99,
                "session_id": "abc-123"
            },
        )
        ```
    """

    tool_name: str
    """Name of the tool that produced this artifact."""

    server_name: str
    """Name of the MCP server."""

    meta: dict[str, Any] | None = None
    """
    MCP response meta - contains widget templates, accessibility info, etc.
    Keys vary by MCP implementation but commonly include:
    - openai/outputTemplate: Widget template URL
    - openai/widgetAccessible: Whether widget is accessible
    - openai/resultCanProduceWidget: Whether result can produce a widget
    - openai/toolInvocation/invoked: Status message
    """

    structured_content: dict[str, Any] | None = None
    """
    MCP response structuredContent - the actual data for rendering.
    This is the raw data that widgets/UI use. Examples:
    - Categories: {items: [...], count: N}
    - Cart: {items: [...], subtotal, session_id}
    - Products: {items: [...], meta: {...}}
    """

    text_content: str | None = None
    """
    Text content extracted from MCP response for LLM consumption.
    This is what gets sent to the LLM as the tool result.
    """

    raw_content: list[dict[str, Any]] | None = None
    """
    Raw content items from CallToolResult.content (serialized).
    Kept for debugging/inspection if needed.
    """

    is_error: bool = False
    """Whether the tool call resulted in an error."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the artifact was captured."""

    latency_ms: float = 0.0
    """Tool execution latency in milliseconds."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "meta": self.meta,
            "structured_content": self.structured_content,
            "text_content": self.text_content,
            "raw_content": self.raw_content,
            "is_error": self.is_error,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolArtifact":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            server_name=data["server_name"],
            meta=data.get("meta"),
            structured_content=data.get("structured_content"),
            text_content=data.get("text_content"),
            raw_content=data.get("raw_content"),
            is_error=data.get("is_error", False),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(UTC),
            latency_ms=data.get("latency_ms", 0.0),
        )

    def has_widget(self) -> bool:
        """Check if this artifact has widget information."""
        if not self.meta:
            return False
        return any(
            key.startswith("openai/") and ("widget" in key.lower() or "template" in key.lower())
            for key in self.meta.keys()
        )

    def get_widget_template(self) -> str | None:
        """Get the widget template URL if available."""
        if not self.meta:
            return None
        return self.meta.get("openai/outputTemplate")


@dataclass
class RunArtifacts:
    """
    Collection of artifacts for a single agent run.

    This is CLEARED at the start of each run - it's run-scoped, not session-scoped.
    Use this to access full MCP responses including widgets, structured data, etc.

    Example:
        ```python
        # In runner
        self._run_artifacts = RunArtifacts()  # Clear at run start

        # After tools execute
        for artifact in self._run_artifacts.tool_artifacts:
            if artifact.has_widget():
                # Send widget to frontend
                send_widget(artifact.get_widget_template(), artifact.structured_content)
        ```
    """

    tool_artifacts: list[MCPToolArtifact] = field(default_factory=list)
    """All tool artifacts captured during this run."""

    run_id: str | None = None
    """ID of the run these artifacts belong to."""

    def add_artifact(self, artifact: MCPToolArtifact) -> None:
        """Add a tool artifact."""
        self.tool_artifacts.append(artifact)

    def clear(self) -> None:
        """Clear all artifacts (called at run start)."""
        self.tool_artifacts.clear()

    def get_by_tool(self, tool_name: str) -> list[MCPToolArtifact]:
        """Get all artifacts for a specific tool."""
        return [a for a in self.tool_artifacts if a.tool_name == tool_name]

    def get_latest_by_tool(self, tool_name: str) -> MCPToolArtifact | None:
        """Get the most recent artifact for a specific tool."""
        artifacts = self.get_by_tool(tool_name)
        return artifacts[-1] if artifacts else None

    def get_widgets(self) -> list[MCPToolArtifact]:
        """Get all artifacts that have widget information."""
        return [a for a in self.tool_artifacts if a.has_widget()]

    def get_structured_data(self) -> dict[str, Any]:
        """
        Get merged structured content from all artifacts.
        Later artifacts override earlier ones for same keys.
        """
        merged: dict[str, Any] = {}
        for artifact in self.tool_artifacts:
            if artifact.structured_content:
                merged.update(artifact.structured_content)
        return merged

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "tool_artifacts": [a.to_dict() for a in self.tool_artifacts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunArtifacts":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id"),
            tool_artifacts=[MCPToolArtifact.from_dict(a) for a in data.get("tool_artifacts", [])],
        )

    def is_empty(self) -> bool:
        """Check if no artifacts were collected."""
        return len(self.tool_artifacts) == 0


# =============================================================================
# Tool Context Variables (for session/state management across tool calls)
# =============================================================================


@dataclass
class ToolContextVariable:
    """
    Configuration for a single context variable to capture/inject.

    Context variables allow automatic extraction of values from tool responses
    (like session_id) and injection into subsequent tool calls.

    Example:
        ```python
        # Capture session_id from create_session, inject into all tools
        ToolContextVariable(
            name="session_id",
            capture_from=["create_session"],  # Or None for all tools
            inject_into=None,                  # None = all tools with this param
        )
        ```
    """

    name: str
    """Variable name to capture/inject (e.g., 'session_id', 'auth_token')."""

    capture_from: list[str] | None = None
    """Tool names to capture this variable from. None = capture from any tool that returns it."""

    inject_into: list[str] | None = None
    """Tool names to inject this variable into. None = inject into any tool with matching parameter."""

    json_path: str | None = None
    """JSONPath to extract value from tool response. None = use variable name as key."""

    scope: Literal["session", "run"] = "session"
    """
    Variable scope:
    - 'session': Persisted in Redis session, survives across runs
    - 'run': Only valid for current run, cleared after completion
    """

    override_llm_value: bool = True
    """If True, override LLM-provided value with stored value. Safer for session management."""

    required: bool = False
    """If True, tool call fails if variable is not available and not provided by LLM."""

    sensitive: bool = False
    """If True, the variable value is masked in serialized output (to_dict, logs)."""


@dataclass
class ToolContextConfig:
    """
    Configuration for tool context variable capture and injection.

    Attach this to an MCPServer to enable automatic session/state management.

    Example:
        ```python
        config = ToolContextConfig(
            variables=[
                ToolContextVariable(name="session_id"),
                ToolContextVariable(name="auth_token", scope="run"),
            ],
            auto_capture_common=True,  # Auto-capture session_id, token, etc.
        )

        server = MCPServerStreamableHttp(
            params={...},
            context_config=config,
        )
        ```
    """

    variables: list[ToolContextVariable] = field(default_factory=list)
    """Explicit variable configurations."""

    auto_capture_common: bool = True
    """
    Automatically capture common variables without explicit configuration:
    - session_id, sessionId, session
    - auth_token, token, access_token
    - user_id, userId

    These are captured from any tool that returns them.
    """

    namespace: str | None = None
    """
    Namespace for variable isolation. Defaults to MCP server name.
    Use to share context across multiple MCP servers if needed.
    """

    inject_into_system_prompt: bool = True
    """If True, inject captured variables into system prompt for LLM awareness."""

    def get_variable_config(self, name: str) -> ToolContextVariable | None:
        """Get configuration for a specific variable."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def should_capture(self, tool_name: str, var_name: str) -> bool:
        """Check if a variable should be captured from a tool."""
        var_config = self.get_variable_config(var_name)
        if var_config:
            if var_config.capture_from is None:
                return True  # Capture from all tools
            return tool_name in var_config.capture_from

        # Auto-capture common variables
        if self.auto_capture_common:
            common_vars = {
                "session_id",
                "sessionId",
                "session",
                "auth_token",
                "token",
                "access_token",
                "user_id",
                "userId",
            }
            return var_name in common_vars

        return False

    def should_inject(self, tool_name: str, var_name: str) -> bool:
        """Check if a variable should be injected into a tool."""
        var_config = self.get_variable_config(var_name)
        if var_config:
            if var_config.inject_into is None:
                return True  # Inject into all tools
            return tool_name in var_config.inject_into

        # Auto-inject common variables
        if self.auto_capture_common:
            return True  # If we're auto-capturing, also auto-inject

        return False

    def get_scope(self, var_name: str) -> Literal["session", "run"]:
        """Get the scope for a variable."""
        var_config = self.get_variable_config(var_name)
        if var_config:
            return var_config.scope
        return "session"  # Default to session scope


TrustAction = Literal["block", "warn", "allow"]
"""What to do about a tool catalogue that was never approved, or that changed."""

_FROM_SETTINGS: Any = object()
"""Sentinel: this field was not passed, so resolve it from global Settings.

Distinguishing "the user asked for this" from "this is what the default
happened to be" matters for exactly one warning -- requesting ``block`` while
leaving ``pin_path`` unset is a mistake worth reporting, but the same
combination reached by inheriting both defaults is just "pinning not in use",
and scolding every user who never opted in would be noise they cannot act on.
"""


@dataclass
class ToolChangeEvent:
    """A change to a server's tool catalogue, as observed on one fetch.

    Exists so an application can *react* -- page oncall, surface a banner, fail
    a CI run. Before this the only report was a ``logger.warning``, which means
    the sole possible response was a human watching a terminal at the moment the
    line scrolled past.
    """

    server_name: str
    changed: list[str] = field(default_factory=list)
    """Tools that kept their name but altered their description or schema. The
    rug-pull signal."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unreviewed: list[str] = field(default_factory=list)
    """Tools with no entry in the approved catalogue."""

    def __bool__(self) -> bool:
        return bool(self.changed or self.added or self.removed or self.unreviewed)


@dataclass
class ToolTrustConfig:
    """How much to trust an MCP server's tool catalogue (security finding F3).

    Grouped into a dataclass rather than added as constructor keywords because
    ``_MCPServerWithClientSession.__init__`` already takes ten, and because
    these four options are only meaningful together.

    Example:
        ```python
        server = MCPServerStreamableHttp(
            params={...},
            name="clinic",
            trust_config=ToolTrustConfig(pin_path="tool-pins.json"),
        )
        ```
    """

    pin_path: str | Path | None = None
    """JSON file holding the *approved* catalogue -- what a human read and
    accepted, written only by ``continuum mcp approve``.

    Defaults to None: digests are kept in memory, which still catches drift
    across a reconnect within one process and avoids a library creating files
    nobody asked for.

    The runtime never writes this file. It writes :attr:`last_seen_path`
    instead. Sharing one file between the two is what let a tripwire rewrite
    promote an attacker's catalogue to "approved" and silently disarm the gate
    on the next restart.
    """

    on_unreviewed: TrustAction = _FROM_SETTINGS  # type: ignore[assignment]
    """What to do about a server, or a tool, with no approved entry.

    Defaults to ``"block"`` (see ``Settings.mcp_on_unreviewed``): pinning cannot
    detect a catalogue that was poisoned before you ever saw it, so human review
    is the only defence that case has, and an optional defence is no defence.

    Only applies when :attr:`pin_path` is set -- without one there is nowhere
    for an approval to live, so there is nothing to enforce.
    """

    on_drift: TrustAction = _FROM_SETTINGS  # type: ignore[assignment]
    """What to do about an approved tool whose description or schema changed.

    Defaults to ``"warn"``: benign drift (a typo fix, a clarified parameter)
    vastly outnumbers attacks, and blocking by default means most users first
    meet this feature as "my agent broke and I changed nothing" -- after which
    they disable it, losing the protection for the rare real case too.
    """

    on_change: "Callable[[ToolChangeEvent], None] | None" = None
    """Optional in-process hook, called whenever a fetch observes a change."""

    def __post_init__(self) -> None:
        self.requested_enforcement = self.on_unreviewed is not _FROM_SETTINGS
        """True if the caller passed on_unreviewed explicitly, rather than
        inheriting it. Lets the server tell a misconfiguration from a default."""
        if self.on_unreviewed is _FROM_SETTINGS:
            self.on_unreviewed = _settings().mcp_on_unreviewed
        if self.on_drift is _FROM_SETTINGS:
            self.on_drift = _settings().mcp_on_drift

    @property
    def last_seen_path(self) -> Path | None:
        """Sibling file recording what the server last served.

        Derived from :attr:`pin_path` rather than configured separately: two
        paths to set is two paths to get inconsistent, and the pairing is not
        something a user should have to maintain.

        Hidden and distinctly named so it is obvious which file is the approval
        and which is scratch -- the runtime rewrites this one constantly, and it
        is the one that should be left out of version control.
        """
        if self.pin_path is None:
            return None
        path = Path(self.pin_path)
        return path.with_name(f".{path.stem}-last-seen{path.suffix}")


def _settings() -> Any:
    """Read global settings lazily.

    Imported inside the function so this module stays importable without
    pulling in pydantic settings resolution at import time, and so tests can
    clear the settings cache between cases.
    """
    from continuum.config import get_settings

    return get_settings()


@dataclass
class ToolContextState:
    """
    Runtime state for tool context variables.

    Stores captured values organized by namespace (MCP server name).
    Can be serialized to/from dict for Redis storage.

    Example:
        ```python
        state = ToolContextState()
        state.set("petco-mcp", "session_id", "fb83832c-...")

        session_id = state.get("petco-mcp", "session_id")
        ```
    """

    # namespace -> variable_name -> value
    _variables: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Track variable metadata (scope, etc.)
    _metadata: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    # Lock for atomic updates to _variables and _metadata together
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def get(self, namespace: str, name: str, default: Any = None) -> Any:
        """Get a variable value."""
        with self._lock:
            return self._variables.get(namespace, {}).get(name, default)

    def set(
        self,
        namespace: str,
        name: str,
        value: Any,
        scope: Literal["session", "run"] = "session",
        sensitive: bool = False,
    ) -> None:
        """Set a variable value atomically with its metadata."""
        with self._lock:
            if namespace not in self._variables:
                self._variables[namespace] = {}
            self._variables[namespace][name] = value

            # Store metadata atomically with the value
            if namespace not in self._metadata:
                self._metadata[namespace] = {}
            self._metadata[namespace][name] = {"scope": scope, "sensitive": sensitive}

    def get_all(self, namespace: str) -> dict[str, Any]:
        """Get all variables for a namespace."""
        with self._lock:
            return self._variables.get(namespace, {}).copy()

    def get_all_namespaces(self) -> list[str]:
        """Get all namespaces with stored variables."""
        with self._lock:
            return list(self._variables.keys())

    def has(self, namespace: str, name: str) -> bool:
        """Check if a variable exists."""
        with self._lock:
            return name in self._variables.get(namespace, {})

    def clear_namespace(self, namespace: str) -> None:
        """Clear all variables for a namespace."""
        self._variables.pop(namespace, None)
        self._metadata.pop(namespace, None)

    def clear_run_scoped(self) -> None:
        """Clear all run-scoped variables (keep session-scoped)."""
        for namespace in list(self._variables.keys()):
            meta = self._metadata.get(namespace, {})
            vars_to_remove = [name for name, m in meta.items() if m.get("scope") == "run"]
            for name in vars_to_remove:
                self._variables[namespace].pop(name, None)
                self._metadata[namespace].pop(name, None)

    def merge_from(self, other: "ToolContextState") -> None:
        """Merge variables from another state (other takes precedence)."""
        for namespace, vars in other._variables.items():
            if namespace not in self._variables:
                self._variables[namespace] = {}
            self._variables[namespace].update(vars)

        for namespace, meta in other._metadata.items():
            if namespace not in self._metadata:
                self._metadata[namespace] = {}
            self._metadata[namespace].update(meta)

    def _is_sensitive(self, namespace: str, name: str) -> bool:
        """Check if a variable is marked as sensitive."""
        return self._metadata.get(namespace, {}).get(name, {}).get("sensitive", False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage. Sensitive values are masked."""
        from continuum.utils.secrets import mask_value

        masked_vars: dict[str, dict[str, Any]] = {}
        for ns, variables in self._variables.items():
            masked_vars[ns] = {}
            for name, value in variables.items():
                if self._is_sensitive(ns, name) and isinstance(value, str):
                    masked_vars[ns][name] = mask_value(value)
                else:
                    masked_vars[ns][name] = value
        return {
            "variables": masked_vars,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolContextState":
        """Deserialize from dictionary."""
        state = cls()
        state._variables = data.get("variables", {})
        state._metadata = data.get("metadata", {})
        return state

    def to_prompt_context(self) -> str | None:
        """
        Generate a context string for system prompt injection.

        Sensitive variables are excluded from the prompt context.

        Returns:
            String describing available context, or None if empty.
        """
        if not self._variables:
            return None

        lines = ["Current tool context (use these values for tool calls):"]
        for namespace, vars in self._variables.items():
            if vars:
                ns_lines: list[str] = []
                for name, value in vars.items():
                    if self._is_sensitive(namespace, name):
                        continue
                    str_value = str(value)
                    if len(str_value) > 50:
                        str_value = str_value[:47] + "..."
                    ns_lines.append(f"    {name}: {str_value}")
                if ns_lines:
                    lines.append(f"  [{namespace}]")
                    lines.extend(ns_lines)

        return "\n".join(lines) if len(lines) > 1 else None

    def is_empty(self) -> bool:
        """Check if state has any variables."""
        return not any(self._variables.values())


class HttpClientFactory(Protocol):
    """Protocol for HTTP client factory functions.

    This interface matches the MCP SDK's McpHttpClientFactory but is defined locally
    to avoid accessing internal MCP SDK modules.
    """

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient: ...


@dataclass
class ToolFilterContext:
    """Context information available to tool filter functions."""

    server_name: str
    """The name of the MCP server."""

    # Optional context that can be passed for dynamic filtering
    metadata: dict[str, Any] | None = None
    """Optional metadata for filtering decisions."""


ToolFilterCallable = Callable[["ToolFilterContext", "MCPTool"], bool | Any]
"""A function that determines whether a tool should be available.

Args:
    context: The context information including server name and metadata.
    tool: The MCP tool to filter.

Returns:
    Whether the tool should be available (True) or filtered out (False).
    Can be async (returns Awaitable[bool]) or sync (returns bool).
"""


class ToolFilterStatic(TypedDict):
    """Static tool filter configuration using allowlists and blocklists."""

    allowed_tool_names: NotRequired[list[str]]
    """Optional list of tool names to allow (whitelist).
    If set, only these tools will be available."""

    blocked_tool_names: NotRequired[list[str]]
    """Optional list of tool names to exclude (blacklist).
    If set, these tools will be filtered out."""


ToolFilter = ToolFilterCallable | ToolFilterStatic | None
"""A tool filter that can be either a function, static configuration, or None (no filtering)."""


def create_static_tool_filter(
    allowed_tool_names: list[str] | None = None,
    blocked_tool_names: list[str] | None = None,
) -> ToolFilterStatic | None:
    """Create a static tool filter from allowlist and blocklist parameters.

    This is a convenience function for creating a ToolFilterStatic.

    Args:
        allowed_tool_names: Optional list of tool names to allow (whitelist).
        blocked_tool_names: Optional list of tool names to exclude (blacklist).

    Returns:
        A ToolFilterStatic if any filtering is specified, None otherwise.
    """
    if allowed_tool_names is None and blocked_tool_names is None:
        return None

    filter_dict: ToolFilterStatic = {}
    if allowed_tool_names is not None:
        filter_dict["allowed_tool_names"] = allowed_tool_names
    if blocked_tool_names is not None:
        filter_dict["blocked_tool_names"] = blocked_tool_names

    return filter_dict
