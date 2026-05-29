"""
MCP (Model Context Protocol) server implementations.

Provides support for connecting to MCP servers via stdio, SSE, and streamable HTTP transports.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import json
import typing
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeVar, Union

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import Tool as MCPTool
from mcp.client.session import MessageHandlerFnT
from mcp.client.sse import sse_client
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    InitializeResult,
    ListPromptsResult,
    Resource,
    TextContent,
)
from typing_extensions import TypedDict

from orchestrator.exceptions import ValidationError
from orchestrator.logging import get_logger
from orchestrator.tools.exceptions import MCPConnectionError, MCPError
from orchestrator.tools.types import (
    HttpClientFactory,
    ToolContextConfig,
    ToolFilter,
    ToolFilterContext,
    ToolFilterStatic,
)

T = TypeVar("T")

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

logger = get_logger(__name__)


class MCPServer(abc.ABC):
    """Base class for Model Context Protocol servers."""

    def __init__(
        self,
        use_structured_content: bool = False,
        context_config: ToolContextConfig | None = None,
    ):
        """
        Args:
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            context_config: Configuration for automatic context variable capture and injection.
                Enables session management across tool calls (e.g., capturing session_id from
                create_session and injecting into subsequent cart operations).
        """
        self.use_structured_content = use_structured_content
        self.context_config = context_config or ToolContextConfig()

    @abc.abstractmethod
    async def connect(self):
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable name for the server."""
        pass

    @abc.abstractmethod
    async def cleanup(self):
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """
        pass

    @abc.abstractmethod
    async def list_tools(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> list[MCPTool]:
        """List the tools available on the server.

        Args:
            metadata: Optional metadata for tool filtering context.
        """
        pass

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        """Invoke a tool on the server."""
        pass

    @abc.abstractmethod
    async def list_prompts(self) -> ListPromptsResult:
        """List the prompts available on the server."""
        pass

    @abc.abstractmethod
    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Get a specific prompt from the server."""
        pass

    @abc.abstractmethod
    async def list_resources(self) -> list[Resource]:
        """List the resources available on the server."""
        pass

    @abc.abstractmethod
    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI. Returns the text content."""
        pass

    async def __aenter__(self) -> MCPServer:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.cleanup()


class _MCPServerWithClientSession(MCPServer, abc.ABC):
    """Base class for MCP servers that use a `ClientSession` to communicate with the server."""

    def __init__(
        self,
        cache_tools_list: bool,
        client_session_timeout_seconds: float | None,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
        context_config: ToolContextConfig | None = None,
        validate_on_connect: bool = False,
    ):
        """
        Args:
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
            cached and only fetched from the server once. If `False`, the tools list will be
            fetched from the server on each call to `list_tools()`. The cache can be invalidated
            by calling `invalidate_tools_cache()`. You should set this to `True` if you know the
            server will not change its tools list, because it can drastically improve latency
            (by avoiding a round-trip to the server every time).

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, used for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
            context_config: Configuration for automatic context variable capture and injection.
                Enables session management across tool calls.
            validate_on_connect: If True, call list_tools() once after connect to fail fast on
                misconfiguration. Default False so slow servers are not penalized.
        """
        super().__init__(
            use_structured_content=use_structured_content,
            context_config=context_config,
        )
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._cleanup_called: bool = False

        # In-flight call tracking for graceful drain before cleanup.
        # _active_calls counts calls currently inside call_tool().
        # _no_active_calls is set when the counter reaches zero so cleanup()
        # can await it without polling.
        self._active_calls: int = 0
        self._no_active_calls: asyncio.Event = asyncio.Event()
        self._no_active_calls.set()  # initially no active calls

        self.cache_tools_list = cache_tools_list
        self.server_initialize_result: InitializeResult | None = None
        self.validate_on_connect = validate_on_connect

        self.client_session_timeout_seconds = client_session_timeout_seconds
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_seconds_base = retry_backoff_seconds_base
        self.message_handler = message_handler

        # The cache is always dirty at startup, so that we fetch tools at least once
        self._cache_dirty = True
        self._tools_list: list[MCPTool] | None = None

        self.tool_filter = tool_filter

    async def _apply_tool_filter(
        self,
        tools: list[MCPTool],
        metadata: dict[str, Any] | None = None,
    ) -> list[MCPTool]:
        """Apply the tool filter to the list of tools."""
        if self.tool_filter is None:
            return tools

        # Handle static tool filter
        if isinstance(self.tool_filter, dict):
            return self._apply_static_tool_filter(tools, self.tool_filter)

        # Handle callable tool filter (dynamic filter)
        else:
            return await self._apply_dynamic_tool_filter(tools, metadata)

    def _apply_static_tool_filter(
        self, tools: list[MCPTool], static_filter: ToolFilterStatic
    ) -> list[MCPTool]:
        """Apply static tool filtering based on allowlist and blocklist."""
        filtered_tools = tools

        # Apply allowed_tool_names filter (whitelist)
        if "allowed_tool_names" in static_filter:
            allowed_names = static_filter["allowed_tool_names"]
            filtered_tools = [t for t in filtered_tools if t.name in allowed_names]

        # Apply blocked_tool_names filter (blacklist)
        if "blocked_tool_names" in static_filter:
            blocked_names = static_filter["blocked_tool_names"]
            filtered_tools = [t for t in filtered_tools if t.name not in blocked_names]

        return filtered_tools

    async def _apply_dynamic_tool_filter(
        self,
        tools: list[MCPTool],
        metadata: dict[str, Any] | None = None,
    ) -> list[MCPTool]:
        """Apply dynamic tool filtering using a callable filter function."""

        # Ensure we have a callable filter
        if not callable(self.tool_filter):
            raise ValidationError("Tool filter must be callable for dynamic filtering")
        tool_filter_func = self.tool_filter

        # Create filter context
        filter_context = ToolFilterContext(
            server_name=self.name,
            metadata=metadata,
        )

        filtered_tools = []
        for tool in tools:
            try:
                # Call the filter function with context
                result = tool_filter_func(filter_context, tool)

                if inspect.isawaitable(result):
                    should_include = await result
                else:
                    should_include = result

                if should_include:
                    filtered_tools.append(tool)
            except Exception as e:
                logger.error(
                    f"Error applying tool filter to tool '{tool.name}' on server '{self.name}': {e}"
                )
                # On error, exclude the tool for safety
                continue

        return filtered_tools

    @abc.abstractmethod
    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the streams for the server."""
        pass

    def invalidate_tools_cache(self):
        """Invalidate the tools cache."""
        self._cache_dirty = True

    async def _run_with_retries(self, func: Callable[[], Awaitable[T]]) -> T:
        attempts = 0
        while True:
            try:
                return await func()
            except (ConnectionError, TimeoutError, OSError, MCPConnectionError):
                attempts += 1
                if self.max_retry_attempts != -1 and attempts > self.max_retry_attempts:
                    raise
                backoff = self.retry_backoff_seconds_base * (2 ** (attempts - 1))
                await asyncio.sleep(backoff)
            except Exception:
                raise  # permanent error — don't retry

    async def connect(self):
        """Connect to the server."""
        try:
            transport = await self.exit_stack.enter_async_context(self.create_streams())
            # streamablehttp_client returns (read, write, get_session_id)
            # sse_client returns (read, write)

            read, write, *_ = transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    timedelta(seconds=self.client_session_timeout_seconds)
                    if self.client_session_timeout_seconds
                    else None,
                    message_handler=self.message_handler,
                )
            )
            server_result = await session.initialize()
            self.server_initialize_result = server_result
            self.session = session

            if self.validate_on_connect:
                await self.list_tools()
        except (Exception, asyncio.CancelledError) as e:
            logger.error(f"Error initializing MCP server: {e}")
            await self.cleanup()
            raise MCPConnectionError(
                f"Failed to connect to MCP server: {e}",
                server_name=self.name,
                original_error=e,
            ) from e

    async def list_tools(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> list[MCPTool]:
        """List the tools available on the server."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )
        session = self.session
        assert session is not None

        # Return from cache if caching is enabled, we have tools, and the cache is not dirty
        if self.cache_tools_list and not self._cache_dirty and self._tools_list:
            tools = self._tools_list
        else:
            # Fetch the tools from the server
            result = await self._run_with_retries(lambda: session.list_tools())
            self._tools_list = result.tools
            self._cache_dirty = False
            tools = self._tools_list

        # Filter tools based on tool_filter
        filtered_tools = tools
        if self.tool_filter is not None:
            filtered_tools = await self._apply_tool_filter(filtered_tools, metadata)
        return filtered_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        """Invoke a tool on the server."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )
        session = self.session
        assert session is not None

        # Register this call so cleanup() can drain before tearing down the connection.
        # Increment is synchronous (no await between it and the clear), so no race.
        self._active_calls += 1
        self._no_active_calls.clear()
        try:
            return await self._run_with_retries(lambda: session.call_tool(tool_name, arguments))
        except Exception as e:
            # Detect JSON parsing errors via exception type first, then string fallback
            is_json_error = isinstance(e, (json.JSONDecodeError,))
            if not is_json_error:
                # Fallback: check exception class name and message for JSON errors
                # from third-party libraries that may wrap json errors
                error_msg = str(e)
                error_type_name = type(e).__name__
                is_json_error = (
                    "JSONDecodeError" in error_type_name
                    or "JsonDecodeError" in error_type_name
                    or isinstance(e, ValueError)
                    and (
                        "Expecting value" in error_msg
                        or "Unterminated string" in error_msg
                        or "Expecting property name" in error_msg
                    )
                )

            if is_json_error:
                error_msg = str(e)
                logger.error(
                    f"❌ MCP server '{self.name}' returned invalid JSON response for tool '{tool_name}': {error_msg}",
                    extra={
                        "tool_name": tool_name,
                        "server_name": self.name,
                        "error_type": type(e).__name__,
                        "is_json_error": True,
                    },
                )
                # Wrap in MCPError for consistent error handling
                raise MCPError(
                    f"Invalid JSON response from MCP server '{self.name}' for tool '{tool_name}': {error_msg}",
                    server_name=self.name,
                    tool_name=tool_name,
                    original_error=e,
                ) from e
            # Re-raise other errors as-is
            raise
        finally:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._no_active_calls.set()

    async def list_prompts(self) -> ListPromptsResult:
        """List the prompts available on the server."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )

        return await self.session.list_prompts()

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Get a specific prompt from the server."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )

        return await self.session.get_prompt(name, arguments)

    async def list_resources(self) -> list[Resource]:
        """List the resources available on the server."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )
        result = await self.session.list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI. Returns the text content."""
        if not self.session:
            raise MCPError(
                "Server not initialized. Make sure you call `connect()` first.",
                server_name=self.name,
            )
        from mcp.types import TextResourceContents
        from pydantic import AnyUrl

        result = await self.session.read_resource(AnyUrl(uri))
        for item in result.contents:
            if isinstance(item, TextResourceContents):
                return item.text
        return ""

    async def cleanup(self):
        """Cleanup the server."""
        # Prevent multiple cleanup calls
        if self._cleanup_called:
            return

        async with self._cleanup_lock:
            # Double-check after acquiring lock
            if self._cleanup_called:
                return

            self._cleanup_called = True

            # Drain in-flight tool calls before tearing down the connection.
            # Orla-style: drain → close → signal done.
            if self._active_calls > 0:
                logger.debug(
                    f"MCP cleanup: waiting for {self._active_calls} in-flight "
                    f"call(s) to complete on server '{self.name}'"
                )
                try:
                    await asyncio.wait_for(self._no_active_calls.wait(), timeout=30.0)
                except TimeoutError:
                    logger.warning(
                        f"MCP cleanup: timed out waiting for in-flight calls to drain "
                        f"on server '{self.name}' ({self._active_calls} still active)"
                    )

            try:
                # Check if current task is cancelled before attempting cleanup
                # If cancelled, skip exit_stack cleanup (resources will be cleaned up on process exit)
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    logger.debug("Task cancelled, skipping exit_stack cleanup")
                    self.session = None
                    return

                # Try to close exit_stack, but handle the cancel scope error gracefully
                await self.exit_stack.aclose()
            except asyncio.CancelledError:
                logger.debug("MCP cleanup cancelled during shutdown, continuing...")
            except RuntimeError as e:
                msg = str(e).lower()
                if "cancel scope" in msg or "different task" in msg or "already running" in msg:
                    logger.debug(f"MCP cleanup cross-task error (expected during shutdown): {e}")
                else:
                    logger.error(f"Error cleaning up server: {e}")
            except Exception as e:
                # anyio.WouldBlock and similar errors during cross-task shutdown
                type_name = type(e).__name__
                if "WouldBlock" in type_name or "Busy" in type_name:
                    logger.debug(f"MCP cleanup blocked (expected during shutdown): {e}")
                else:
                    logger.error(f"Error cleaning up server: {e}")
            finally:
                # Always clear session reference
                self.session = None


class MCPServerStdioParams(TypedDict):
    """Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another
    import.
    """

    command: str
    """The executable to run to start the server. For example, `python` or `node`."""

    args: NotRequired[list[str]]
    """Command line args to pass to the `command` executable. For example, `['foo.py']` or
    `['server.js', '--port', '8080']`."""

    env: NotRequired[dict[str, str]]
    """The environment variables to set for the server."""

    cwd: NotRequired[str | Path]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`."""

    encoding_error_handler: NotRequired[Literal["strict", "ignore", "replace"]]
    """The text encoding error handler. Defaults to `strict`.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """


class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server implementation that uses the stdio transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for
    details.
    """

    def __init__(
        self,
        params: MCPServerStdioParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
        context_config: ToolContextConfig | None = None,
        validate_on_connect: bool = False,
    ):
        """Create a new MCP server based on the stdio transport.

        Args:
            params: The params that configure the server. This includes the command to run to
                start the server, the args to pass to the command, the environment variables to
                set for the server, the working directory to use when spawning the process, and
                the text encoding used when sending/receiving messages to the server.
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).
            name: A readable name for the server. If not provided, we'll create one from the
                command.
            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, used for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
            context_config: Configuration for automatic context variable capture and injection.
                Enables session management across tool calls.
            validate_on_connect: If True, call list_tools() once after connect to fail fast.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
            context_config=context_config,
            validate_on_connect=validate_on_connect,
        )

        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get("encoding_error_handler", "strict"),
        )

        self._name = name or f"stdio: {self.params.command}"

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the streams for the server."""
        return stdio_client(self.params)

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name


class MCPServerSseParams(TypedDict):
    """Mirrors the params in`mcp.client.sse.sse_client`."""

    url: str
    """The URL of the server."""

    headers: NotRequired[dict[str, str]]
    """The headers to send to the server."""

    timeout: NotRequired[float]
    """The timeout for the HTTP request. Defaults to 5 seconds."""

    sse_read_timeout: NotRequired[float]
    """The timeout for the SSE connection, in seconds. Defaults to 5 minutes."""


class MCPServerSse(_MCPServerWithClientSession):
    """MCP server implementation that uses the HTTP with SSE transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse)
    for details.
    """

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
        context_config: ToolContextConfig | None = None,
        validate_on_connect: bool = False,
    ):
        """Create a new MCP server based on the HTTP with SSE transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, and the
                timeout for the SSE connection.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
            context_config: Configuration for automatic context variable capture and injection.
                Enables session management across tool calls.
            validate_on_connect: If True, call list_tools() once after connect to fail fast.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
            context_config=context_config,
            validate_on_connect=validate_on_connect,
        )

        self.params = params
        self._name = name or f"sse: {self.params['url']}"

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the streams for the server."""
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers", None),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
        )

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name


class MCPServerStreamableHttpParams(TypedDict):
    """Mirrors the params in`mcp.client.streamable_http.streamablehttp_client`."""

    url: str
    """The URL of the server."""

    headers: NotRequired[dict[str, str]]
    """The headers to send to the server."""

    timeout: NotRequired[timedelta | float]
    """The timeout for the HTTP request. Defaults to 5 seconds."""

    sse_read_timeout: NotRequired[timedelta | float]
    """The timeout for the SSE connection, in seconds. Defaults to 5 minutes."""

    terminate_on_close: NotRequired[bool]
    """Terminate on close"""

    httpx_client_factory: NotRequired[HttpClientFactory]
    """Custom HTTP client factory for configuring httpx.AsyncClient behavior."""


class MCPServerStreamableHttp(_MCPServerWithClientSession):
    """MCP server implementation that uses the Streamable HTTP transport. See the [spec]
    (https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http)
    for details.
    """

    def __init__(
        self,
        params: MCPServerStreamableHttpParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
        context_config: ToolContextConfig | None = None,
        validate_on_connect: bool = False,
    ):
        """Create a new MCP server based on the Streamable HTTP transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, the
                timeout for the Streamable HTTP connection, whether we need to
                terminate on close, and an optional custom HTTP client factory.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
            context_config: Configuration for automatic context variable capture and injection.
                Enables session management across tool calls (e.g., capturing session_id).
            validate_on_connect: If True, call list_tools() once after connect to fail fast.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
            context_config=context_config,
            validate_on_connect=validate_on_connect,
        )

        self.params = params
        self._name = name or f"streamable_http: {self.params['url']}"

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the streams for the server."""
        # Only pass httpx_client_factory if it's provided
        if "httpx_client_factory" in self.params:
            return streamablehttp_client(
                url=self.params["url"],
                headers=self.params.get("headers", None),
                timeout=self.params.get("timeout", 5),
                sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
                terminate_on_close=self.params.get("terminate_on_close", True),
                httpx_client_factory=self.params["httpx_client_factory"],
            )
        else:
            return streamablehttp_client(
                url=self.params["url"],
                headers=self.params.get("headers", None),
                timeout=self.params.get("timeout", 5),
                sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
                terminate_on_close=self.params.get("terminate_on_close", True),
            )

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name


# =============================================================================
# In-process function tool server (no subprocess / no network)
# =============================================================================

# ---------------------------------------------------------------------------
# Schema generation helpers
# ---------------------------------------------------------------------------


def _type_to_schema(hint: Any) -> dict[str, Any]:
    """Convert a single Python type hint to a JSON Schema fragment.

    Handles: str, int, float, bool, list, dict, Optional[X].
    Falls back to {} (open schema) for anything more complex.
    """
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None)

    # Optional[X]  →  Union[X, None]
    if origin is Union and args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {}

    if hint is str:
        return {"type": "string"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is bool:
        return {"type": "boolean"}
    if hint is list or origin is list:
        return {"type": "array"}
    if hint is dict or origin is dict:
        return {"type": "object"}

    return {}  # unknown type — open schema, LLM sees no constraint


def _schema_from_function(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema ``{"type": "object", ...}`` from a function's signature.

    Parameters without type hints get an open ``{}`` schema.
    Parameters without defaults (and not Optional) are added to ``required``.
    """
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        hint = hints.get(name)
        prop = _type_to_schema(hint) if hint is not None else {}
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            # Optional[X] parameters are not required even without a default
            origin = getattr(hint, "__origin__", None)
            args = getattr(hint, "__args__", None)
            is_optional = origin is Union and args and type(None) in args
            if not is_optional:
                required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def function_tool(fn: Callable[..., Any]) -> FunctionTool:
    """Decorator that converts a Python function to a ``FunctionTool``.

    The tool name is taken from ``fn.__name__``.
    The description is the first line of the docstring (or empty string).
    The input schema is generated from type hints — supports ``str``, ``int``,
    ``float``, ``bool``, ``list``, ``dict``, and ``Optional[X]``.

    The function may use natural Python parameter signatures; arguments are
    unpacked from the dict that ``call_tool`` receives::

        @function_tool
        def format_currency(amount: float, currency: str = "USD") -> str:
            \"\"\"Format a number as a currency string.\"\"\"
            return f"{amount:,.2f} {currency}"

        server = MCPServerFunction("utils", [format_currency])
    """
    name = fn.__name__
    doc = (fn.__doc__ or "").strip()
    description = doc.split("\n")[0].strip() if doc else ""
    input_schema = _schema_from_function(fn)

    # Wrap so call_tool can pass a dict while fn uses natural kwargs signatures.
    if inspect.iscoroutinefunction(fn):

        async def _wrapped(args: dict[str, Any]) -> Any:
            return await fn(**args)
    else:

        def _wrapped(args: dict[str, Any]) -> Any:  # type: ignore[misc]
            return fn(**args)

    return FunctionTool(name=name, fn=_wrapped, description=description, input_schema=input_schema)


def _coerce_to_function_tool(
    item: FunctionTool | Callable[..., Any] | dict[str, Any],
) -> FunctionTool:
    """Normalise the three accepted tool formats into a ``FunctionTool``.

    Accepted formats:
    - ``FunctionTool`` dataclass — passed through unchanged.
    - Callable — schema generated from type hints via ``function_tool()``.
    - dict — must have ``"name"`` and ``"fn"`` keys; ``"description"`` and
      ``"input_schema"`` are optional.
    """
    if isinstance(item, FunctionTool):
        return item
    if callable(item):
        return function_tool(item)
    if isinstance(item, dict):
        fn = item.get("fn")
        if fn is None or not callable(fn):
            raise ValueError(
                f"Tool dict must contain a callable 'fn' key. Got keys: {list(item.keys())}"
            )
        name = item.get("name") or getattr(fn, "__name__", "unknown")
        description = item.get("description") or (fn.__doc__ or "").strip().split("\n")[0]
        input_schema = item.get("input_schema") or _schema_from_function(fn)
        return FunctionTool(name=name, fn=fn, description=description, input_schema=input_schema)
    raise TypeError(f"Expected FunctionTool, callable, or dict — got {type(item).__name__}")


@dataclass
class FunctionTool:
    """A single Python function exposed as an MCP-compatible tool.

    Args:
        name: Tool name (used as the MCP tool identifier).
        fn: Sync or async callable that receives ``dict[str, Any]`` arguments
            and returns any JSON-serialisable value (or raises on failure).
        description: Human-readable description passed to the LLM.
        input_schema: JSON Schema for the tool's arguments.  Defaults to an
            open ``{"type": "object"}`` if omitted.
    """

    name: str
    fn: Callable[[dict[str, Any]], Any]
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})


class MCPServerFunction(MCPServer):
    """Wraps plain Python callables as an in-process MCPServer.

    Tool execution always happens in the calling Python process — never routed
    through a subprocess or network connection. This is an explicit design
    contract: in-process tools have direct access to your application's state,
    databases, and local APIs without any serialization overhead.

    Accepts three tool formats:

    1. ``FunctionTool`` dataclass — full control over name, description, schema.
    2. Decorated function via ``@function_tool`` — schema auto-generated from
       type hints, description from docstring.
    3. Plain callable — same auto-generation as ``@function_tool``.
    4. Dict with ``"fn"`` key — ``{"name": ..., "description": ...,
       "input_schema": ..., "fn": ...}``.

    Example::

        # FunctionTool (explicit schema)
        server = MCPServerFunction("utils", [
            FunctionTool(
                name="format_currency",
                description="Format a number as USD",
                input_schema={
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                },
                fn=lambda args: f"${args['amount']:,.2f}",
            ),
        ])

        # @function_tool decorator (auto-schema)
        @function_tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        server = MCPServerFunction("math", [add])

        # Plain callable (auto-schema from type hints)
        def multiply(a: int, b: int) -> int:
            \"\"\"Multiply two integers.\"\"\"
            return a * b

        server = MCPServerFunction("math", [multiply])

        executor = ToolExecutor({server: None})
        await executor.initialize()
    """

    def __init__(
        self,
        name: str,
        tools: list[FunctionTool | Callable[..., Any] | dict[str, Any]],
        context_config: ToolContextConfig | None = None,
    ) -> None:
        super().__init__(context_config=context_config)
        self._name = name
        self._registry: dict[str, tuple[MCPTool, Callable[[dict[str, Any]], Any]]] = {}
        for item in tools:
            ft = _coerce_to_function_tool(item)
            if ft.name in self._registry:
                raise ValueError(f"MCPServerFunction '{name}': duplicate tool name '{ft.name}'")
            mcp_tool = MCPTool(
                name=ft.name,
                description=ft.description,
                inputSchema=ft.input_schema,
            )
            self._registry[ft.name] = (mcp_tool, ft.fn)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass  # No external connection needed.

    async def cleanup(self) -> None:
        pass  # No resources to release.

    async def list_tools(self, metadata: dict[str, Any] | None = None) -> list[MCPTool]:
        return [mcp_tool for mcp_tool, _ in self._registry.values()]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if tool_name not in self._registry:
            raise MCPError(
                f"Tool '{tool_name}' not registered on server '{self._name}'",
                server_name=self._name,
                tool_name=tool_name,
            )
        _, fn = self._registry[tool_name]
        args = arguments or {}
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(args)
            else:
                result = fn(args)
            text = result if isinstance(result, str) else json.dumps(result)
            return CallToolResult(content=[TextContent(type="text", text=text)])
        except Exception as e:
            error_text = json.dumps({"error": str(e), "error_type": type(e).__name__})
            return CallToolResult(
                content=[TextContent(type="text", text=error_text)],
                isError=True,
            )

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        raise MCPError(
            f"Server '{self._name}' has no prompts",
            server_name=self._name,
        )

    async def list_resources(self) -> list[Resource]:
        return []

    async def read_resource(self, uri: str) -> str:
        return ""
