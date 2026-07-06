"""
Session Providers - Provider implementations for the session module.

This module provides a registry for session providers and exports
available provider implementations.

Available Providers:
    - RedisSessionProvider: Uses Redis for session storage

Adding New Providers:
    1. Create a new file in this directory (e.g., dynamodb.py)
    2. Implement the BaseSessionProvider interface
    3. Register the provider in PROVIDER_REGISTRY
    4. Export from this __init__.py
"""

from continuum.session.base import BaseSessionProvider
from continuum.session.config import SessionConfig

# In-memory provider has no external dependencies — always available. Used as
# the graceful-degradation fallback when Redis is disabled/unconfigured/unreachable.
from continuum.session.providers.memory import MemorySessionProvider

# Provider registry maps provider names to their classes
PROVIDER_REGISTRY: dict[str, type[BaseSessionProvider]] = {}

# Try to import RedisSessionProvider - may fail if redis not installed
try:
    from continuum.session.providers.redis import RedisSessionProvider

    _REDIS_AVAILABLE = True
except ImportError:
    RedisSessionProvider = None  # type: ignore
    _REDIS_AVAILABLE = False


def register_provider(name: str, provider_class: type[BaseSessionProvider]) -> None:
    """
    Register a session provider.

    Args:
        name: Provider name (e.g., "redis", "dynamodb")
        provider_class: Provider class implementing BaseSessionProvider
    """
    PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> type[BaseSessionProvider]:
    """
    Get a provider class by name.

    Args:
        name: Provider name

    Returns:
        Provider class

    Raises:
        ValueError: If provider is not found
    """
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys()) if PROVIDER_REGISTRY else "none"
        raise ValueError(f"Unknown session provider: {name}. Available providers: {available}")
    return PROVIDER_REGISTRY[name]


def create_provider(name: str, config: SessionConfig) -> BaseSessionProvider:
    """
    Create a provider instance.

    Args:
        name: Provider name
        config: Session configuration

    Returns:
        Provider instance
    """
    provider_class = get_provider_class(name)
    return provider_class(config)


def list_providers() -> list[str]:
    """
    List available provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_REGISTRY.keys())


def is_redis_available() -> bool:
    """Check if Redis provider is available."""
    return _REDIS_AVAILABLE


# Register default providers at module load
register_provider("memory", MemorySessionProvider)
if _REDIS_AVAILABLE and RedisSessionProvider is not None:
    register_provider("redis", RedisSessionProvider)


__all__ = [
    # Base class
    "BaseSessionProvider",
    # Registry
    "PROVIDER_REGISTRY",
    # Functions
    "register_provider",
    "get_provider_class",
    "create_provider",
    "list_providers",
    "is_redis_available",
    # Providers
    "MemorySessionProvider",
    # Providers (conditional)
    "RedisSessionProvider",
]
