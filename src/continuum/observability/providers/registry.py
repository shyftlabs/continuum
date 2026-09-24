"""
Provider registry for managing multiple observability providers.
"""

from __future__ import annotations

import threading

from continuum.logging import get_logger
from continuum.observability.providers.base import ObservabilityProvider

logger = get_logger(__name__)

# Global registry instance
_global_registry: ProviderRegistry | None = None
_registry_lock = threading.Lock()


class ProviderRegistry:
    """
    Thread-safe registry for managing observability providers.

    Example:
        ```python
        registry = ProviderRegistry()
        registry.register("langfuse", langfuse_provider)
        provider = registry.get("langfuse")
        ```
    """

    def __init__(self):
        """Initialize the registry."""
        self._providers: dict[str, ObservabilityProvider] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        provider: ObservabilityProvider,
        overwrite: bool = False,
    ) -> None:
        """
        Register a provider.

        Args:
            name: Unique name for the provider
            provider: Provider instance
            overwrite: If True, overwrite existing provider with same name

        Raises:
            ValueError: If provider with name already exists and overwrite=False
        """
        with self._lock:
            if name in self._providers and not overwrite:
                raise ValueError(
                    f"Provider '{name}' already registered. Use overwrite=True to replace."
                )
            self._providers[name] = provider
            logger.debug("Registered observability provider: %s", name)

    def unregister(self, name: str) -> ObservabilityProvider | None:
        """
        Unregister a provider.

        Args:
            name: Name of the provider to unregister

        Returns:
            The unregistered provider or None if not found
        """
        with self._lock:
            provider = self._providers.pop(name, None)
            if provider:
                logger.debug("Unregistered observability provider: %s", name)
            return provider

    def get(self, name: str) -> ObservabilityProvider | None:
        """
        Get a provider by name.

        Args:
            name: Name of the provider

        Returns:
            Provider instance or None if not found
        """
        with self._lock:
            return self._providers.get(name)

    def get_all(self) -> dict[str, ObservabilityProvider]:
        """
        Get all registered providers.

        Returns:
            Dictionary mapping provider names to instances
        """
        with self._lock:
            return dict(self._providers)

    def get_enabled(self) -> dict[str, ObservabilityProvider]:
        """
        Get all enabled providers.

        Returns:
            Dictionary mapping provider names to enabled instances
        """
        with self._lock:
            return {
                name: provider for name, provider in self._providers.items() if provider.is_enabled
            }

    def clear(self) -> None:
        """Clear all registered providers."""
        with self._lock:
            self._providers.clear()
            logger.debug("Cleared all observability providers")

    def shutdown_all(self) -> None:
        """Shutdown all registered providers."""
        with self._lock:
            for name, provider in self._providers.items():
                try:
                    provider.shutdown()
                    logger.debug("Shutdown provider: %s", name)
                except Exception as e:
                    logger.warning("Error shutting down provider %s: %s", name, e)

    def flush_all(self) -> None:
        """Flush all registered providers."""
        with self._lock:
            for name, provider in self._providers.items():
                try:
                    provider.flush()
                except Exception as e:
                    logger.warning("Error flushing provider %s: %s", name, e)


def get_provider_registry() -> ProviderRegistry:
    """
    Get the global provider registry.

    Returns:
        The global ProviderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = ProviderRegistry()
    return _global_registry


def register_provider(
    name: str,
    provider: ObservabilityProvider,
    overwrite: bool = False,
) -> None:
    """
    Register a provider in the global registry.

    Args:
        name: Unique name for the provider
        provider: Provider instance
        overwrite: If True, overwrite existing provider with same name
    """
    registry = get_provider_registry()
    registry.register(name, provider, overwrite=overwrite)


def get_provider(name: str) -> ObservabilityProvider | None:
    """
    Get a provider from the global registry.

    Args:
        name: Name of the provider

    Returns:
        Provider instance or None if not found
    """
    registry = get_provider_registry()
    return registry.get(name)
