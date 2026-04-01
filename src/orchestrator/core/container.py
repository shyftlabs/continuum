"""
Dependency Injection Container - Centralized client management.

Provides a container for all SDK clients that can be:
- Configured at startup
- Replaced for testing
- Accessed throughout the application

This makes the SDK more testable and allows for better resource management.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from orchestrator.config import settings
from orchestrator.logging import get_logger

from orchestrator.protocols import ILLMClient, IMemoryClient, ISessionClient

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient
    from orchestrator.memory import MemoryClient
    from orchestrator.observability import TracingManager
    from orchestrator.session import SessionClient
    from orchestrator.session.base import BaseSessionProvider
    from orchestrator.tools import ToolExecutor

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class ContainerConfig:
    """Configuration for the dependency container."""

    # Auto-initialize clients on first access
    auto_initialize: bool = True

    # Enable/disable specific clients
    enable_memory: bool = field(default_factory=lambda: settings.memory_enabled)
    enable_session: bool = field(default_factory=lambda: settings.session_enabled)
    enable_langfuse: bool = field(default_factory=lambda: settings.langfuse_enabled)

    # Custom client configurations (optional)
    llm_config: dict[str, Any] | None = None
    memory_config: dict[str, Any] | None = None
    session_config: dict[str, Any] | None = None
    langfuse_config: dict[str, Any] | None = None


class Container:
    """
    Dependency injection container for all SDK clients.

    Provides centralized management of all client instances, making
    the SDK more testable and allowing for easy configuration.

    Features:
        - Lazy initialization of clients
        - Thread-safe singleton pattern
        - Easy replacement for testing
        - Centralized configuration

    Example:
        ```python
        from orchestrator.core.container import Container, get_container

        # Get the global container
        container = get_container()

        # Access clients
        llm = container.llm_client
        memory = container.memory_client
        session = container.session_client

        # For testing, you can inject mock clients
        container.set_llm_client(mock_llm_client)
        ```

    Testing Example:
        ```python
        from orchestrator.core.container import Container
        from unittest.mock import Mock

        # Create a test container
        container = Container(auto_initialize=False)

        # Inject mock clients
        mock_llm = Mock()
        container.set_llm_client(mock_llm)

        # Use in tests
        assert container.llm_client == mock_llm
        ```
    """

    def __init__(self, config: ContainerConfig | None = None):
        """
        Initialize the container.

        Args:
            config: Optional container configuration
        """
        self._config = config or ContainerConfig()
        self._lock = threading.Lock()

        # Client instances (lazily initialized)
        self._llm_client: LLMClient | None = None
        self._memory_client: MemoryClient | None = None
        self._session_client: SessionClient | None = None
        self._langfuse_client: Any | None = None
        self._tracing_manager: TracingManager | None = None
        self._tool_executor: ToolExecutor | None = None

        # Initialization flags
        self._llm_initialized = False
        self._memory_initialized = False
        self._session_initialized = False
        self._langfuse_initialized = False
        self._tracing_initialized = False
        self._tool_initialized = False

    # =========================================================================
    # LLM Client
    # =========================================================================

    @property
    def llm_client(self) -> LLMClient:
        """
        Get the LLM client instance.

        Lazily initializes the client on first access.
        """
        if not self._llm_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._llm_initialized:
                    self._initialize_llm_client()

        if self._llm_client is None:
            raise RuntimeError("LLM client not initialized. Call set_llm_client() first.")

        return self._llm_client

    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client."""
        from orchestrator.llm import LLMClient

        config = self._config.llm_config or {}
        self._llm_client = LLMClient(**config)
        self._llm_initialized = True
        logger.debug("LLM client initialized via container")

    def set_llm_client(self, client: LLMClient | ILLMClient) -> None:
        """
        Set a custom LLM client (for testing or custom configuration).

        Accepts any object implementing ILLMClient protocol.
        """
        with self._lock:
            self._llm_client = client
            self._llm_initialized = True

    def has_llm_client(self) -> bool:
        """Check if LLM client is available."""
        return self._llm_initialized and self._llm_client is not None

    # =========================================================================
    # Memory Client
    # =========================================================================

    @property
    def memory_client(self) -> MemoryClient | None:
        """
        Get the memory client instance.

        Returns None if memory is disabled.
        """
        if not self._config.enable_memory:
            return None

        if not self._memory_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._memory_initialized:
                    self._initialize_memory_client()

        return self._memory_client

    def _initialize_memory_client(self) -> None:
        """Initialize the memory client."""
        from orchestrator.memory import MemoryClient, MemoryConfig

        config_dict = self._config.memory_config or {}
        config = MemoryConfig(**config_dict)
        self._memory_client = MemoryClient(config=config)
        self._memory_initialized = True
        logger.debug("Memory client initialized via container")

    def set_memory_client(self, client: MemoryClient | IMemoryClient | None) -> None:
        """Set a custom memory client. Accepts any object implementing IMemoryClient protocol."""
        with self._lock:
            self._memory_client = client
            self._memory_initialized = True

    def has_memory_client(self) -> bool:
        """Check if memory client is available and enabled."""
        return (
            self._config.enable_memory
            and self._memory_initialized
            and self._memory_client is not None
            and self._memory_client.is_enabled
        )

    # =========================================================================
    # Session Client
    # =========================================================================

    @property
    def session_client(self) -> SessionClient | None:
        """
        Get the session client instance.

        Returns None if sessions are disabled.
        """
        if not self._config.enable_session:
            return None

        if not self._session_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._session_initialized:
                    self._initialize_session_client()

        return self._session_client

    @property
    def session_provider(self) -> BaseSessionProvider | None:
        """
        Get the session provider instance.

        Returns None if sessions are disabled.
        """
        if not self._config.enable_session:
            return None

        if not self._session_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._session_initialized:
                    self._initialize_session_client()

        if self._session_client:
            return self._session_client.provider
        return None

    def _initialize_session_client(self) -> None:
        """Initialize the session client and provider."""
        from orchestrator.session import SessionClient, SessionConfig
        from orchestrator.session.providers import create_provider

        config_dict = self._config.session_config or {}
        config = SessionConfig(**config_dict)

        # Create provider from registry
        provider = create_provider(config.provider, config)

        # Pass the provider to SessionClient
        self._session_client = SessionClient(
            session_config=config,
            provider=provider,
        )
        self._session_initialized = True
        logger.debug("Session client initialized via container")

    def set_session_client(
        self,
        client: SessionClient | ISessionClient | None,
        provider: BaseSessionProvider | None = None,
    ) -> None:
        """Set a custom session client. Accepts any object implementing ISessionClient protocol."""

        with self._lock:
            self._session_client = client
            if client and provider:
                client.set_provider(provider)
            self._session_initialized = True

    def has_session_client(self) -> bool:
        """Check if session client is available and enabled."""
        return (
            self._config.enable_session
            and self._session_initialized
            and self._session_client is not None
        )

    # =========================================================================
    # Langfuse Client
    # =========================================================================

    @property
    def langfuse_client(self) -> Any | None:
        """
        Get the Langfuse client instance.

        Returns None if Langfuse is disabled.
        """
        if not self._config.enable_langfuse:
            return None

        if not self._langfuse_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._langfuse_initialized:
                    self._initialize_langfuse_client()

        return self._langfuse_client

    def _initialize_langfuse_client(self) -> None:
        """Initialize observability providers."""
        from orchestrator.observability import ObservabilityConfig, initialize_observability
        from orchestrator.observability.providers.registry import get_provider

        config_dict = self._config.langfuse_config or {}
        config = ObservabilityConfig(**config_dict)

        # Initialize providers
        initialize_observability(config)

        # Get Langfuse provider if available
        langfuse_provider = get_provider("langfuse")
        if langfuse_provider and langfuse_provider.is_enabled:
            self._langfuse_client = langfuse_provider.client
        else:
            self._langfuse_client = None
        self._langfuse_initialized = True
        logger.debug("Observability providers initialized via container")

    def set_langfuse_client(self, client: Any | None) -> None:
        """Set a custom Langfuse client (for backward compatibility with existing code)."""
        with self._lock:
            self._langfuse_client = client
            self._langfuse_initialized = True

    def has_langfuse_client(self) -> bool:
        """Check if Langfuse client is available and enabled."""
        return (
            self._config.enable_langfuse
            and self._langfuse_initialized
            and self._langfuse_client is not None
        )

    # =========================================================================
    # Tracing Manager
    # =========================================================================

    @property
    def tracing_manager(self) -> TracingManager | None:
        """Get the tracing manager instance."""
        if not self._config.enable_langfuse:
            return None

        if not self._tracing_initialized and self._config.auto_initialize:
            with self._lock:
                if not self._tracing_initialized:
                    self._initialize_tracing_manager()

        return self._tracing_manager

    def _initialize_tracing_manager(self) -> None:
        """Initialize the tracing manager."""
        from orchestrator.observability import TracingManager

        self._tracing_manager = TracingManager()
        self._tracing_initialized = True
        logger.debug("Tracing manager initialized via container")

    def set_tracing_manager(self, manager: TracingManager | None) -> None:
        """Set a custom tracing manager."""
        with self._lock:
            self._tracing_manager = manager
            self._tracing_initialized = True

    # =========================================================================
    # Tool Executor
    # =========================================================================

    @property
    def tool_executor(self) -> ToolExecutor | None:
        """Get the global tool executor instance."""
        return self._tool_executor

    def set_tool_executor(self, executor: ToolExecutor | None) -> None:
        """Set a custom tool executor."""
        with self._lock:
            self._tool_executor = executor
            self._tool_initialized = True

    def has_tool_executor(self) -> bool:
        """Check if tool executor is available."""
        return self._tool_executor is not None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def reset(self) -> None:
        """
        Reset all clients (useful for testing).

        This clears all client instances and resets initialization flags.
        Note: For proper async cleanup, use shutdown() instead.
        """
        with self._lock:
            self._llm_client = None
            self._memory_client = None
            self._session_client = None
            self._langfuse_client = None
            self._tracing_manager = None
            self._tool_executor = None

            self._llm_initialized = False
            self._memory_initialized = False
            self._session_initialized = False
            self._langfuse_initialized = False
            self._tracing_initialized = False
            self._tool_initialized = False

        logger.debug("Container reset - all clients cleared")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all clients with proper async cleanup.

        Respects shared_services_enabled setting - if True, only flushes
        Langfuse traces and doesn't close Redis connections (they persist).

        This method should be called when the application is shutting down
        to ensure all resources are properly released and data is flushed.

        Example:
            ```python
            container = get_container()
            await container.shutdown()
            ```
        """
        from orchestrator.config import settings

        logger.info("Shutting down container clients...")

        shared_services = settings.shared_services_enabled

        # Langfuse: only shutdown if not a shared service
        # If shared service, do nothing - let it handle its own flushing
        if self._langfuse_initialized and self._langfuse_client is not None:
            if not shared_services:
                try:
                    # Flush traces before shutdown to ensure they're sent
                    self._langfuse_client.flush()
                    self._langfuse_client.shutdown()
                    logger.debug("Langfuse client shutdown complete")
                except Exception as e:
                    logger.warning(f"Error shutting down Langfuse client: {e}")
            else:
                logger.debug(
                    "Langfuse is a shared service, skipping all operations (no flush, no shutdown)"
                )

        # Shutdown tracing manager (only if not shared service)
        if self._tracing_initialized and self._tracing_manager is not None:
            if not shared_services:
                try:
                    self._tracing_manager.shutdown()
                    logger.debug("Tracing manager shutdown complete")
                except Exception as e:
                    logger.warning(f"Error shutting down tracing manager: {e}")
            else:
                logger.debug("Tracing manager is part of shared service, skipping shutdown")

        # Close memory client (with timeout to prevent hanging)
        # Memory client (Qdrant) is typically not shared, so we close it
        if self._memory_initialized and self._memory_client is not None:
            try:
                await asyncio.wait_for(self._memory_client.close(), timeout=5.0)
                logger.debug("Memory client closed")
            except TimeoutError:
                logger.warning("Memory client close timed out after 5s — force-releasing reference")
                self._memory_client = None
            except Exception as e:
                logger.warning(f"Error closing memory client: {e}")

        # Close session provider (Redis connections) only if not a shared service
        provider = self.session_provider
        if self._session_initialized and provider is not None:
            if shared_services:
                logger.debug("Redis is a shared service, skipping connection close")
            else:
                try:
                    if hasattr(provider, "close"):
                        await asyncio.wait_for(provider.close(), timeout=5.0)
                        logger.debug("Session provider closed")
                    else:
                        logger.debug("Session provider doesn't have close() method")
                except TimeoutError:
                    logger.warning(
                        "Session provider close timed out after 5s — force-releasing reference"
                    )
                except Exception as e:
                    logger.warning(f"Error closing session manager: {e}")

        # Close LLM client async resources
        if self._llm_initialized and self._llm_client is not None:
            try:
                if hasattr(self._llm_client, "cleanup"):
                    await asyncio.wait_for(self._llm_client.cleanup(), timeout=5.0)
                    logger.debug("LLM client cleanup complete")
            except TimeoutError:
                logger.warning("LLM client cleanup timed out after 5s — force-releasing reference")
                self._llm_client = None
            except Exception as e:
                logger.warning(f"Error during LLM client cleanup: {e}")

        # Reset all state
        self.reset()
        logger.info("Container shutdown complete")

    async def __aenter__(self) -> Container:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager with cleanup."""
        await self.shutdown()

    def initialize_all(self) -> dict[str, bool]:
        """
        Initialize all enabled clients.

        Returns:
            Dictionary of client names and their initialization status.
        """
        results = {}

        # LLM (always enabled)
        try:
            _ = self.llm_client
            results["llm"] = True
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            results["llm"] = False

        # Memory
        if self._config.enable_memory:
            try:
                _ = self.memory_client
                results["memory"] = (
                    self._memory_client is not None and self._memory_client.is_enabled
                )
            except Exception as e:
                logger.error(f"Failed to initialize Memory client: {e}")
                results["memory"] = False

        # Session
        if self._config.enable_session:
            try:
                _ = self.session_client
                results["session"] = self._session_client is not None
            except Exception as e:
                logger.error(f"Failed to initialize Session client: {e}")
                results["session"] = False

        # Langfuse
        if self._config.enable_langfuse:
            try:
                _ = self.langfuse_client
                results["langfuse"] = self._langfuse_client is not None
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {e}")
                results["langfuse"] = False

        return results


# =============================================================================
# Global Container
# =============================================================================

_global_container: Container | None = None
_global_lock = threading.Lock()


def get_container(config: ContainerConfig | None = None) -> Container:
    """
    Get the global dependency injection container.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Global Container instance

    Example:
        ```python
        from orchestrator.core.container import get_container

        container = get_container()
        llm = container.llm_client
        ```
    """
    global _global_container

    if _global_container is None:
        with _global_lock:
            if _global_container is None:
                _global_container = Container(config=config)

    return _global_container


def reset_container() -> None:
    """
    Reset the global container and all associated global client state.

    Useful for testing to ensure a clean state.
    """
    global _global_container

    with _global_lock:
        if _global_container is not None:
            _global_container.reset()
            _global_container = None

    # Also reset the module-level globals in memory and session clients
    try:
        from orchestrator.memory.client import reset_global_memory

        reset_global_memory()
    except ImportError:
        pass

    try:
        from orchestrator.session.client import reset_global_session

        reset_global_session()
    except ImportError:
        pass
