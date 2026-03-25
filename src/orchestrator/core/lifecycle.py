"""
Lifecycle Manager - Resource initialization and graceful shutdown.

Provides:
- Eager connection verification at startup
- Graceful shutdown with configurable timeout
- Async context manager support
- Framework-agnostic design (works with FastAPI, Flask, etc.)
"""

from __future__ import annotations

import asyncio
import atexit
import signal
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from orchestrator.config import settings
from orchestrator.core.health import (
    HealthStatus,
    OverallHealthResult,
    get_health_checker,
)
from orchestrator.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigurationError:
    """A configuration validation error."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


def validate_configuration() -> tuple[list[ConfigurationError], list[ConfigurationError]]:
    """
    Validate all SDK configuration at startup.

    Checks that required configurations are properly set based on
    which features are enabled.

    Returns:
        Tuple of (errors, warnings)
    """
    errors: list[ConfigurationError] = []
    warnings: list[ConfigurationError] = []

    # Memory configuration
    if settings.memory_enabled:
        if not settings.qdrant_host:
            errors.append(
                ConfigurationError(
                    field="QDRANT_HOST",
                    message="Required when MEMORY_ENABLED=true",
                )
            )
        if not settings.memory_llm_model:
            warnings.append(
                ConfigurationError(
                    field="MEMORY_LLM_MODEL",
                    message="Not set, will use default model for memory extraction",
                    severity="warning",
                )
            )
        if not settings.embedder_model:
            warnings.append(
                ConfigurationError(
                    field="EMBEDDER_MODEL",
                    message="Not set, will use default embedder",
                    severity="warning",
                )
            )

    # Session configuration
    if settings.session_enabled:
        if not settings.session_redis_host:
            errors.append(
                ConfigurationError(
                    field="SESSION_REDIS_HOST",
                    message="Required when SESSION_ENABLED=true",
                )
            )

    # Langfuse configuration
    if settings.langfuse_enabled:
        if not settings.langfuse_public_key:
            errors.append(
                ConfigurationError(
                    field="LANGFUSE_PUBLIC_KEY",
                    message="Required when LANGFUSE_ENABLED=true",
                )
            )
        if not settings.langfuse_secret_key:
            errors.append(
                ConfigurationError(
                    field="LANGFUSE_SECRET_KEY",
                    message="Required when LANGFUSE_ENABLED=true",
                )
            )

    # LLM configuration - at least one API key should be set
    has_llm_key = any(
        [
            settings.openai_api_key,
            settings.anthropic_api_key,
            settings.gemini_api_key,
            settings.azure_api_key,
        ]
    )
    if not has_llm_key:
        warnings.append(
            ConfigurationError(
                field="LLM_API_KEY",
                message="No LLM API key configured (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)",
                severity="warning",
            )
        )

    # Default model validation
    if not settings.default_llm_model:
        warnings.append(
            ConfigurationError(
                field="DEFAULT_LLM_MODEL",
                message="Not set, will need to specify model in each request",
                severity="warning",
            )
        )

    return errors, warnings


class LifecycleState(str, Enum):
    """Lifecycle state of the SDK."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


@dataclass
class InitializationResult:
    """Result of SDK initialization."""

    success: bool
    state: LifecycleState
    health: OverallHealthResult | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    initialized_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "state": self.state.value,
            "health": self.health.to_dict() if self.health else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "initialized_at": self.initialized_at.isoformat(),
        }


class OrchestratorLifecycle:
    """
    Manages SDK lifecycle with eager initialization and graceful shutdown.

    Features:
        - Eager connection verification at startup
        - Graceful shutdown with timeout
        - Async context manager support
        - Framework integration ready (FastAPI, Flask, etc.)
        - Signal handling for graceful termination

    Example:
        ```python
        from orchestrator.core.lifecycle import OrchestratorLifecycle

        # Manual lifecycle management
        lifecycle = OrchestratorLifecycle()
        result = await lifecycle.initialize()
        if not result.success:
            print(f"Initialization failed: {result.errors}")

        # ... use SDK ...

        await lifecycle.shutdown()

        # Or use as context manager
        async with OrchestratorLifecycle() as lifecycle:
            # SDK is initialized and ready
            pass  # SDK shuts down automatically
        ```

    FastAPI Integration:
        ```python
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from orchestrator.core.lifecycle import get_lifecycle_manager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            lifecycle = get_lifecycle_manager()
            await lifecycle.initialize()
            yield
            await lifecycle.shutdown()

        app = FastAPI(lifespan=lifespan)
        ```
    """

    def __init__(
        self,
        shutdown_timeout: float = 10.0,
        fail_on_unhealthy: bool = False,
        verify_connections: bool = True,
        enable_signal_handlers: bool = True,
    ):
        """
        Initialize lifecycle manager.

        Args:
            shutdown_timeout: Timeout in seconds for graceful shutdown
            fail_on_unhealthy: If True, initialization fails if any dependency is unhealthy
            verify_connections: If True, verify all connections at startup (eager mode)
            enable_signal_handlers: If False, don't register signal handlers (useful when
                using with FastAPI lifespan which handles signals itself)
        """
        self._shutdown_timeout = shutdown_timeout
        self._fail_on_unhealthy = fail_on_unhealthy
        self._verify_connections = verify_connections
        self._enable_signal_handlers = enable_signal_handlers

        self._state = LifecycleState.NOT_INITIALIZED
        self._health_checker = get_health_checker()
        self._lock = threading.Lock()

        # Use try/except for asyncio event - get_running_loop() is the modern approach
        # but only works inside async context. For initialization outside async context,
        # we defer the event creation.
        self._shutdown_event: asyncio.Event | None = None
        try:
            # This will only work if we're already in an async context
            asyncio.get_running_loop()
            self._shutdown_event = asyncio.Event()
        except RuntimeError:
            # Not in async context, event will be created lazily when needed
            pass

        # Callbacks for shutdown
        self._shutdown_callbacks: list[Callable[[], Coroutine[Any, Any, None]]] = []

        # Track initialized components for cleanup
        self._initialized_components: list[str] = []

    @property
    def state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if SDK is running."""
        return self._state == LifecycleState.RUNNING

    def register_shutdown_callback(
        self,
        callback: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register a callback to be called during shutdown.

        Args:
            callback: Async function to call during shutdown
        """
        self._shutdown_callbacks.append(callback)

    async def initialize(self) -> InitializationResult:
        """
        Initialize the SDK with eager connection verification.

        Performs:
        1. Health check all dependencies
        2. Initialize global clients (Memory, Session, Langfuse)
        3. Verify connections are working

        Returns:
            InitializationResult with status and health info
        """
        with self._lock:
            if self._state == LifecycleState.RUNNING:
                logger.warning("SDK already initialized")
                return InitializationResult(
                    success=True,
                    state=self._state,
                    warnings=["SDK already initialized"],
                )

            if self._state == LifecycleState.INITIALIZING:
                logger.warning("SDK initialization already in progress")
                return InitializationResult(
                    success=False,
                    state=self._state,
                    errors=["Initialization already in progress"],
                )

            self._state = LifecycleState.INITIALIZING

        errors: list[str] = []
        warnings: list[str] = []
        health_result: OverallHealthResult | None = None

        logger.info("Initializing Orchestrator SDK...")

        try:
            # Step 0: Validate configuration
            logger.info("Validating configuration...")
            config_errors, config_warnings = validate_configuration()

            for err in config_errors:
                errors.append(str(err))
                logger.error(f"❌ Config: {err}")

            for warn in config_warnings:
                warnings.append(str(warn))
                logger.warning(f"⚠️ Config: {warn}")

            if config_errors and self._fail_on_unhealthy:
                self._state = LifecycleState.FAILED
                logger.error(f"Configuration validation failed: {len(config_errors)} error(s)")
                return InitializationResult(
                    success=False,
                    state=self._state,
                    health=health_result,
                    errors=errors,
                    warnings=warnings,
                )
            elif not config_errors:
                logger.info("✓ Configuration validated")

            # Step 1: Verify connections if enabled (eager mode)
            if self._verify_connections:
                logger.info("Verifying dependency connections...")
                health_result = await self._health_checker.check_all(timeout=10.0)

                for check in health_result.checks:
                    if check.status == HealthStatus.UNHEALTHY:
                        msg = f"{check.name}: {check.message}"
                        if self._fail_on_unhealthy:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                    elif check.status == HealthStatus.DEGRADED:
                        warnings.append(f"{check.name}: {check.message}")
                    else:
                        logger.info(f"✓ {check.name}: {check.message}")

                if errors and self._fail_on_unhealthy:
                    self._state = LifecycleState.FAILED
                    logger.error(f"Initialization failed: {errors}")
                    return InitializationResult(
                        success=False,
                        state=self._state,
                        health=health_result,
                        errors=errors,
                        warnings=warnings,
                    )

            # Step 2: Initialize global clients
            await self._initialize_clients()

            # Step 3: Log service configurations
            self._log_service_configurations()

            # Step 4: Setup signal handlers for graceful shutdown (if enabled)
            if self._enable_signal_handlers:
                self._setup_signal_handlers()
            else:
                logger.debug("Signal handlers disabled (likely using FastAPI lifespan)")

            self._state = LifecycleState.RUNNING
            logger.info("Orchestrator SDK initialized successfully")

            return InitializationResult(
                success=True,
                state=self._state,
                health=health_result,
                errors=errors,
                warnings=warnings,
            )

        except (KeyboardInterrupt, SystemExit):
            # Let these propagate — they should not be caught as init failures
            raise
        except Exception as e:
            self._state = LifecycleState.FAILED
            errors.append(f"Initialization error: {str(e)}")
            logger.error(f"SDK initialization failed: {e}", exc_info=True)
            return InitializationResult(
                success=False,
                state=self._state,
                health=health_result,
                errors=errors,
                warnings=warnings,
            )

    async def _initialize_clients(self) -> None:
        """Initialize all global clients."""
        # Initialize observability providers
        try:
            from orchestrator.observability import ObservabilityConfig, initialize_observability

            config = ObservabilityConfig()
            if config.is_configured():
                manager = initialize_observability(config)
                if manager.is_enabled:
                    self._initialized_components.append("observability")
                    logger.debug("Observability providers initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize observability: {e}")

        # Initialize Memory client (long-term memory)
        if settings.memory_enabled:
            try:
                from orchestrator.memory import initialize_global_memory

                if initialize_global_memory():
                    self._initialized_components.append("memory")
                    logger.debug("Memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Memory client: {e}")

        # Initialize Session client (short-term memory)
        if settings.session_enabled:
            try:
                from orchestrator.session import initialize_global_session_client

                if initialize_global_session_client():
                    self._initialized_components.append("session")
                    logger.debug("Session client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Session client: {e}")

        # Initialize Temporal client (optional)
        if settings.temporal_enabled:
            try:
                from orchestrator.temporal import get_temporal_client

                client = get_temporal_client()
                await client.connect()
                self._initialized_components.append("temporal")
                logger.debug("Temporal client connected")
            except ImportError:
                logger.debug("temporalio not installed, skipping Temporal init")
            except Exception as e:
                logger.warning(f"Failed to connect to Temporal: {e}")

    def _log_service_configurations(self) -> None:
        """Log service configurations and modes."""
        logger.info("📊 Service Configurations:")

        # Memory configuration
        if settings.memory_enabled:
            try:
                from orchestrator.memory import get_global_memory_client

                memory_client = get_global_memory_client()
                if memory_client and memory_client.is_enabled:
                    memory_isolation = memory_client.config.memory_isolation
                    embedder_provider = memory_client.config.embedder_provider
                    embedder_model = memory_client.config.embedder_model
                    logger.info(
                        f"  💾 Memory: enabled | isolation={memory_isolation} | "
                        f"embedder={embedder_provider}/{embedder_model}"
                    )
                else:
                    logger.info("  💾 Memory: disabled or not initialized")
            except Exception as e:
                logger.debug(f"Could not get memory configuration: {e}")
                logger.info("  💾 Memory: enabled (configuration unavailable)")
        else:
            logger.info("  💾 Memory: disabled")

        # Session configuration
        if settings.session_enabled:
            try:
                from orchestrator.session import get_global_session_client

                session_client = get_global_session_client()
                if session_client and session_client.is_enabled:
                    redis_host = session_client.config.redis_host
                    logger.info(f"  💬 Session: enabled | redis={redis_host}")
                else:
                    logger.info("  💬 Session: disabled or not initialized")
            except Exception as e:
                logger.debug(f"Could not get session configuration: {e}")
                logger.info("  💬 Session: enabled (configuration unavailable)")
        else:
            logger.info("  💬 Session: disabled")

        # Observability configuration
        try:
            from orchestrator.observability import get_provider_manager

            manager = get_provider_manager()
            if manager and manager.is_enabled:
                providers = list(manager._registry.get_enabled().keys())
                logger.info(f"  📈 Observability: enabled | providers={providers}")
            else:
                logger.info("  📈 Observability: disabled or not configured")
        except Exception as e:
            logger.debug(f"Could not get observability configuration: {e}")
            logger.info("  📈 Observability: configuration unavailable")

        # LLM configuration
        try:
            from orchestrator.core.container import get_container

            container = get_container()
            if container.llm_client:
                default_model = settings.default_llm_model or "not set"
                logger.info(f"  🤖 LLM: enabled | default_model={default_model}")
            else:
                logger.info("  🤖 LLM: not available")
        except Exception as e:
            logger.debug(f"Could not get LLM configuration: {e}")
            logger.info("  🤖 LLM: configuration unavailable")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._signal_handler(s)),
                )
            logger.debug("Signal handlers registered")
        except (RuntimeError, NotImplementedError):
            # Not running in async context or platform doesn't support signals
            # Register atexit handler instead
            atexit.register(self._sync_shutdown)
            logger.debug("Atexit handler registered")

    async def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        # Prevent multiple simultaneous shutdown calls
        if self._state in (LifecycleState.SHUTTING_DOWN, LifecycleState.SHUTDOWN):
            logger.debug(f"Ignoring {sig.name} signal - shutdown already in progress")
            return

        logger.info(f"Received signal {sig.name}, initiating shutdown...")
        await self.shutdown()

    def _sync_shutdown(self) -> None:
        """Synchronous shutdown for atexit."""
        if self._state != LifecycleState.RUNNING:
            return

        try:
            # Try to get existing running loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, schedule the shutdown
                loop.create_task(self.shutdown())
            except RuntimeError:
                # No running loop, create a new one for shutdown
                # This is fine for atexit cleanup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.shutdown())
                finally:
                    loop.close()
        except Exception as e:
            logger.warning(f"Error during sync shutdown: {e}")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the SDK.

        Performs:
        1. Set state to SHUTTING_DOWN
        2. Run registered shutdown callbacks with timeout
        3. Flush and close all clients
        4. Set state to SHUTDOWN
        """
        with self._lock:
            if self._state in (LifecycleState.SHUTTING_DOWN, LifecycleState.SHUTDOWN):
                logger.debug("Shutdown already in progress or completed")
                return

            if self._state != LifecycleState.RUNNING:
                logger.debug(f"Cannot shutdown from state: {self._state}")
                return

            self._state = LifecycleState.SHUTTING_DOWN

        logger.info("Initiating graceful shutdown...")

        try:
            # Run shutdown callbacks with timeout
            if self._shutdown_callbacks:
                logger.debug(f"Running {len(self._shutdown_callbacks)} shutdown callbacks...")
                callback_tasks = [callback() for callback in self._shutdown_callbacks]

                try:
                    await asyncio.wait_for(
                        asyncio.gather(*callback_tasks, return_exceptions=True),
                        timeout=self._shutdown_timeout,
                    )
                except TimeoutError:
                    logger.warning(f"Shutdown callbacks timed out after {self._shutdown_timeout}s")

            # Shutdown initialized components in reverse order
            await self._shutdown_clients()

            self._state = LifecycleState.SHUTDOWN
            logger.info("Orchestrator SDK shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._state = LifecycleState.SHUTDOWN

    async def _shutdown_clients(self) -> None:
        """
        Shutdown all initialized clients.

        Respects shared_services_enabled setting - if True, only flushes
        Langfuse traces and doesn't shutdown clients (they persist).

        Note: Container.shutdown() handles detailed client cleanup.
        This method handles SDK-level global client cleanup.
        """
        # Observability: only shutdown if not a shared service
        # If shared service, do nothing - let it handle its own flushing
        if "observability" in self._initialized_components:
            if not settings.shared_services_enabled:
                try:
                    from orchestrator.observability import get_provider_manager

                    # Flush traces before shutdown to ensure they're sent
                    manager = get_provider_manager()
                    manager.flush()
                    manager.shutdown()
                    logger.debug("Langfuse client shutdown")
                except Exception as e:
                    logger.warning(f"Error shutting down Langfuse: {e}")
            else:
                logger.debug(
                    "Langfuse is a shared service, skipping all operations (no flush, no shutdown)"
                )

        # Temporal: stop worker then disconnect client (closes aiohttp/gRPC connections)
        if "temporal" in self._initialized_components:
            try:
                from orchestrator.temporal import get_worker_manager

                manager = get_worker_manager()
                if manager.is_running:
                    await manager.stop()
                    logger.debug("Temporal worker stopped")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error stopping Temporal worker: {e}")
            try:
                from orchestrator.temporal import get_temporal_client

                client = get_temporal_client()
                if client.is_connected:
                    await client.disconnect()
                    logger.debug("Temporal client disconnected")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error disconnecting Temporal client: {e}")

        # Container shutdown: memory, session, LLM (LiteLLM aiohttp), etc.
        try:
            from orchestrator.core.container import get_container

            container = get_container()
            await container.shutdown()
        except Exception as e:
            logger.warning(f"Error during container shutdown: {e}")

        # The lifecycle initialises a *global* memory client via
        # initialize_global_memory() which is independent from the
        # container's lazily-created memory client.  Close it explicitly
        # so we don't leak the Mem0Provider resources.
        if "memory" in self._initialized_components:
            try:
                from orchestrator.memory.client import (
                    get_global_memory_client,
                    reset_global_memory,
                )

                global_mem = get_global_memory_client()
                if global_mem and global_mem.is_enabled:
                    await global_mem.close()
                reset_global_memory()
            except Exception as e:
                logger.debug(f"Non-critical: global memory client cleanup: {e}")

    async def get_health(self) -> OverallHealthResult:
        """
        Get current health status.

        Returns:
            OverallHealthResult with all dependency statuses
        """
        return await self._health_checker.check_all()

    # Async context manager support
    async def __aenter__(self) -> OrchestratorLifecycle:
        """Enter async context."""
        result = await self.initialize()
        if not result.success and self._fail_on_unhealthy:
            raise RuntimeError(f"SDK initialization failed: {result.errors}")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.shutdown()


# Global lifecycle manager
_global_lifecycle: OrchestratorLifecycle | None = None
_global_lock = threading.Lock()


def get_lifecycle_manager(
    shutdown_timeout: float = 10.0,
    fail_on_unhealthy: bool = False,
    verify_connections: bool = True,
    enable_signal_handlers: bool = True,
) -> OrchestratorLifecycle:
    """
    Get the global lifecycle manager.

    Args:
        shutdown_timeout: Timeout for graceful shutdown
        fail_on_unhealthy: If True, fail initialization if dependencies unhealthy
        verify_connections: If True, verify connections at startup
        enable_signal_handlers: If False, don't register signal handlers (useful when
            using with FastAPI lifespan which handles signals itself)

    Returns:
        OrchestratorLifecycle instance
    """
    global _global_lifecycle

    if _global_lifecycle is None:
        with _global_lock:
            if _global_lifecycle is None:
                _global_lifecycle = OrchestratorLifecycle(
                    shutdown_timeout=shutdown_timeout,
                    fail_on_unhealthy=fail_on_unhealthy,
                    verify_connections=verify_connections,
                    enable_signal_handlers=enable_signal_handlers,
                )

    return _global_lifecycle


async def initialize_orchestrator(
    fail_on_unhealthy: bool = False,
    verify_connections: bool = True,
) -> InitializationResult:
    """
    Convenience function to initialize the SDK.

    Args:
        fail_on_unhealthy: If True, fail if any dependency is unhealthy
        verify_connections: If True, verify all connections at startup

    Returns:
        InitializationResult
    """
    lifecycle = get_lifecycle_manager(
        fail_on_unhealthy=fail_on_unhealthy,
        verify_connections=verify_connections,
    )
    return await lifecycle.initialize()


async def shutdown_orchestrator() -> None:
    """
    Convenience function to shutdown the SDK.
    """
    global _global_lifecycle

    if _global_lifecycle is not None:
        await _global_lifecycle.shutdown()
