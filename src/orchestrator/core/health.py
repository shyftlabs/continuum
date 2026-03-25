"""
Health Check Module - Comprehensive dependency health monitoring.

Provides health checks for all SDK dependencies:
- Redis (Session storage)
- Qdrant (Vector storage for memory)
- Langfuse (Observability)
- LLM Providers (via LiteLLM)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from orchestrator.config import settings
from orchestrator.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class OverallHealthResult:
    """Overall health status combining all checks."""

    status: HealthStatus
    checks: list[HealthCheckResult]
    total_latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "checked_at": self.checked_at.isoformat(),
            "checks": {check.name: check.to_dict() for check in self.checks},
        }


class HealthCheck:
    """
    Health checker for all SDK dependencies.

    Performs startup verification and provides health endpoint data.

    Example:
        ```python
        from orchestrator.core.health import HealthCheck

        health = HealthCheck()

        # Check all dependencies at startup
        result = await health.check_all()
        if result.status != HealthStatus.HEALTHY:
            print(f"Unhealthy dependencies: {result}")

        # Get individual check
        redis_health = await health.check_redis()
        ```
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: dict[str, Callable[[], Coroutine[Any, Any, HealthCheckResult]]] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self._checks["redis"] = self._check_redis
        self._checks["qdrant"] = self._check_qdrant
        self._checks["langfuse"] = self._check_langfuse
        self._checks["llm"] = self._check_llm
        self._checks["temporal"] = self._check_temporal

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, HealthCheckResult]],
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Name of the health check
            check_fn: Async function that returns HealthCheckResult
        """
        self._checks[name] = check_fn

    async def check_all(self, timeout: float = 10.0) -> OverallHealthResult:
        """
        Check all registered dependencies.

        Args:
            timeout: Timeout for all checks in seconds

        Returns:
            OverallHealthResult with status of all checks
        """
        start_time = time.time()
        checks: list[HealthCheckResult] = []

        # Run all checks concurrently with timeout
        tasks = []
        for name, check_fn in self._checks.items():
            task = asyncio.create_task(self._run_check_with_timeout(name, check_fn, timeout))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, HealthCheckResult):
                checks.append(result)
            elif isinstance(result, Exception):
                # Handle unexpected errors
                checks.append(
                    HealthCheckResult(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(result)}",
                    )
                )

        # Determine overall status
        statuses = [check.status for check in checks]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        total_latency = (time.time() - start_time) * 1000

        return OverallHealthResult(
            status=overall_status,
            checks=checks,
            total_latency_ms=total_latency,
        )

    async def _run_check_with_timeout(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, HealthCheckResult]],
        timeout: float,
    ) -> HealthCheckResult:
        """Run a single check with timeout."""
        try:
            return await asyncio.wait_for(check_fn(), timeout=timeout)
        except TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        return await self._check_redis()

    async def _check_redis(self) -> HealthCheckResult:
        """Internal Redis health check."""
        start_time = time.time()

        if not settings.session_enabled:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis disabled (SESSION_ENABLED=false)",
                details={"enabled": False},
            )

        try:
            import redis.asyncio as redis

            client = redis.Redis(
                host=settings.session_redis_host,
                port=settings.session_redis_port,
                password=settings.session_redis_password or None,
                db=settings.session_redis_db,
                ssl=settings.session_redis_ssl,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection with PING
            await client.ping()

            # Get server info for details
            info = await client.info(section="server")

            await client.aclose()

            latency = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                latency_ms=latency,
                details={
                    "enabled": True,
                    "host": settings.session_redis_host,
                    "port": settings.session_redis_port,
                    "redis_version": info.get("redis_version", "unknown"),
                },
            )

        except ImportError:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="redis package not installed",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                latency_ms=latency,
                details={
                    "host": settings.session_redis_host,
                    "port": settings.session_redis_port,
                    "error": str(e),
                },
            )

    async def check_qdrant(self) -> HealthCheckResult:
        """Check Qdrant connectivity."""
        return await self._check_qdrant()

    async def _check_qdrant(self) -> HealthCheckResult:
        """Internal Qdrant health check."""
        start_time = time.time()

        if not settings.memory_enabled:
            return HealthCheckResult(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                message="Qdrant disabled (MEMORY_ENABLED=false)",
                details={"enabled": False},
            )

        try:
            from qdrant_client import QdrantClient

            # Run sync client in thread to not block
            def _check():
                client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key or None,
                    timeout=5,
                )
                # Get collections to verify connection
                collections = client.get_collections()
                return len(collections.collections)

            collection_count = await asyncio.to_thread(_check)

            latency = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                message="Qdrant connection successful",
                latency_ms=latency,
                details={
                    "enabled": True,
                    "host": settings.qdrant_host,
                    "port": settings.qdrant_port,
                    "collection_count": collection_count,
                },
            )

        except ImportError:
            return HealthCheckResult(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                message="qdrant-client package not installed",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant connection failed: {str(e)}",
                latency_ms=latency,
                details={
                    "host": settings.qdrant_host,
                    "port": settings.qdrant_port,
                    "error": str(e),
                },
            )

    async def check_langfuse(self) -> HealthCheckResult:
        """Check Langfuse connectivity."""
        return await self._check_langfuse()

    async def _check_langfuse(self) -> HealthCheckResult:
        """Internal Langfuse health check."""
        start_time = time.time()

        if not settings.langfuse_enabled:
            return HealthCheckResult(
                name="langfuse",
                status=HealthStatus.HEALTHY,
                message="Langfuse disabled (LANGFUSE_ENABLED=false)",
                details={"enabled": False},
            )

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return HealthCheckResult(
                name="langfuse",
                status=HealthStatus.DEGRADED,
                message="Langfuse credentials not configured",
                details={"enabled": True, "configured": False},
            )

        try:
            from langfuse import Langfuse

            # Run sync auth check in thread
            def _check():
                client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                result = client.auth_check()
                client.shutdown()
                return result

            auth_result = await asyncio.to_thread(_check)

            latency = (time.time() - start_time) * 1000

            if auth_result:
                return HealthCheckResult(
                    name="langfuse",
                    status=HealthStatus.HEALTHY,
                    message="Langfuse authentication successful",
                    latency_ms=latency,
                    details={
                        "enabled": True,
                        "configured": True,
                        "host": settings.langfuse_host,
                    },
                )
            else:
                return HealthCheckResult(
                    name="langfuse",
                    status=HealthStatus.UNHEALTHY,
                    message="Langfuse authentication failed",
                    latency_ms=latency,
                    details={
                        "enabled": True,
                        "host": settings.langfuse_host,
                    },
                )

        except ImportError:
            return HealthCheckResult(
                name="langfuse",
                status=HealthStatus.UNHEALTHY,
                message="langfuse package not installed",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="langfuse",
                status=HealthStatus.UNHEALTHY,
                message=f"Langfuse connection failed: {str(e)}",
                latency_ms=latency,
                details={
                    "host": settings.langfuse_host,
                    "error": str(e),
                },
            )

    async def check_llm(self) -> HealthCheckResult:
        """Check LLM provider connectivity."""
        return await self._check_llm()

    async def _check_llm(self) -> HealthCheckResult:
        """Internal LLM health check."""
        start_time = time.time()

        # Check if any LLM API key is configured
        has_openai = bool(settings.openai_api_key)
        has_anthropic = bool(settings.anthropic_api_key)
        has_gemini = bool(settings.gemini_api_key)
        has_azure = bool(settings.azure_api_key)

        if not any([has_openai, has_anthropic, has_gemini, has_azure]):
            return HealthCheckResult(
                name="llm",
                status=HealthStatus.DEGRADED,
                message="No LLM API keys configured",
                details={
                    "openai": False,
                    "anthropic": False,
                    "gemini": False,
                    "azure": False,
                },
            )

        try:
            from orchestrator.llm.context_window import get_context_window_manager

            model = settings.default_llm_model
            limits = get_context_window_manager().get_model_limits(model)
            latency = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="llm",
                status=HealthStatus.HEALTHY,
                message=f"LLM configured with model: {model}",
                latency_ms=latency,
                details={
                    "default_model": model,
                    "openai": has_openai,
                    "anthropic": has_anthropic,
                    "gemini": has_gemini,
                    "azure": has_azure,
                    "max_tokens": limits.max_tokens,
                },
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="llm",
                status=HealthStatus.DEGRADED,
                message=f"LLM check warning: {str(e)}",
                latency_ms=latency,
                details={
                    "default_model": settings.default_llm_model,
                    "openai": has_openai,
                    "anthropic": has_anthropic,
                    "gemini": has_gemini,
                    "azure": has_azure,
                    "warning": str(e),
                },
            )


    async def check_temporal(self) -> HealthCheckResult:
        """Check Temporal connectivity."""
        return await self._check_temporal()

    async def _check_temporal(self) -> HealthCheckResult:
        """Internal Temporal health check."""
        start_time = time.time()

        if not settings.temporal_enabled:
            return HealthCheckResult(
                name="temporal",
                status=HealthStatus.HEALTHY,
                message="Temporal disabled (TEMPORAL_ENABLED=false)",
                details={"enabled": False},
            )

        try:
            from temporalio.client import Client

            client = await Client.connect(
                settings.temporal_host,
                namespace=settings.temporal_namespace,
            )
            latency = (time.time() - start_time) * 1000

            # Close the throwaway client so the underlying gRPC channel is released.
            service_client = getattr(client, "service_client", None)
            if service_client is not None:
                bridge = getattr(service_client, "_bridge_client", None)
                if bridge is not None:
                    # Drop the Rust reference; the gRPC channel will be
                    # reclaimed once the reference count reaches zero.
                    service_client._bridge_client = None

            return HealthCheckResult(
                name="temporal",
                status=HealthStatus.HEALTHY,
                message="Temporal connection successful",
                latency_ms=latency,
                details={
                    "enabled": True,
                    "host": settings.temporal_host,
                    "namespace": settings.temporal_namespace,
                },
            )

        except ImportError:
            return HealthCheckResult(
                name="temporal",
                status=HealthStatus.DEGRADED,
                message="temporalio package not installed (pip install shyftlabs-continuum[temporal])",
                details={"enabled": True, "installed": False},
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="temporal",
                status=HealthStatus.UNHEALTHY,
                message=f"Temporal connection failed: {str(e)}",
                latency_ms=latency,
                details={
                    "host": settings.temporal_host,
                    "namespace": settings.temporal_namespace,
                    "error": str(e),
                },
            )


# Global health checker instance
_global_health_checker: HealthCheck | None = None
_global_lock = threading.Lock()


def get_health_checker() -> HealthCheck:
    """
    Get the global health checker instance.

    Returns:
        HealthCheck instance
    """
    global _global_health_checker

    if _global_health_checker is None:
        with _global_lock:
            if _global_health_checker is None:
                _global_health_checker = HealthCheck()

    return _global_health_checker


async def check_all_health(timeout: float = 10.0) -> OverallHealthResult:
    """
    Convenience function to check all health.

    Args:
        timeout: Timeout for all checks

    Returns:
        OverallHealthResult
    """
    checker = get_health_checker()
    return await checker.check_all(timeout=timeout)
