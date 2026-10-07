"""
Observability initialization.

Loads and initializes observability providers based on configuration.
"""

from __future__ import annotations

from continuum.logging import get_logger
from continuum.observability.config import ObservabilityConfig
from continuum.observability.provider_manager import ProviderManager, get_provider_manager
from continuum.observability.providers.base import ObservabilityProvider
from continuum.observability.providers.langfuse import LangfuseProvider
from continuum.observability.providers.registry import get_provider_registry

logger = get_logger(__name__)

# Global initialization state
_initialized = False


def initialize_observability(
    config: ObservabilityConfig | None = None,
) -> ProviderManager:
    """
    Initialize observability providers based on configuration.

    This function:
    1. Determines which providers to enable (from explicit config or auto-detection)
    2. Creates and initializes each provider
    3. Registers them in the global registry
    4. Returns a ProviderManager instance

    Args:
        config: Optional ObservabilityConfig. If not provided, creates one from settings.

    Returns:
        ProviderManager instance that routes calls to all registered providers

    Example:
        ```python
        from continuum.observability import ObservabilityConfig, initialize_observability

        config = ObservabilityConfig(
            providers=["langfuse", "vertex"],
            provider_configs={
                "langfuse": {"public_key": "...", "secret_key": "..."},
                "vertex": {"project": "...", "location": "..."},
            }
        )
        manager = initialize_observability(config)
        ```
    """
    global _initialized

    if config is None:
        config = ObservabilityConfig()

    registry = get_provider_registry()
    manager = get_provider_manager()

    # Determine which providers to initialize
    providers_to_init = _determine_providers(config)

    # Initialize each provider
    for provider_name in providers_to_init:
        try:
            provider = _create_provider(provider_name, config)
            if provider and provider.is_enabled:
                registry.register(provider_name, provider, overwrite=True)
                logger.info("Initialized observability provider: %s", provider_name)
            else:
                logger.warning("Provider %s was created but is not enabled", provider_name)
        except Exception as e:
            logger.error("Failed to initialize provider %s: %s", provider_name, e, exc_info=True)

    _initialized = True

    # Wire up the lazy error reporter hook in exceptions module
    try:
        from continuum.exceptions import set_error_reporter
        from continuum.observability.error_reporter import report_error

        set_error_reporter(report_error)
    except Exception:
        pass

    return manager


def _determine_providers(config: ObservabilityConfig) -> list[str]:
    """
    Determine which providers to initialize based on configuration.

    Args:
        config: ObservabilityConfig instance

    Returns:
        List of provider names to initialize
    """
    # If providers are explicitly configured, use them
    if config.providers:
        return config.providers

    # Auto-detect: if Langfuse is configured (via legacy fields), initialize it
    if config.is_configured():
        # Check if Langfuse-specific fields are set
        if config.public_key and config.secret_key:
            return ["langfuse"]

    return []


def _create_provider(
    provider_name: str,
    config: ObservabilityConfig,
) -> ObservabilityProvider | None:
    """
    Create a provider instance.

    Args:
        provider_name: Name of the provider to create
        config: ObservabilityConfig instance

    Returns:
        Provider instance or None if creation failed
    """
    provider_config = config.get_provider_config(provider_name)

    if provider_config is None:
        logger.warning("No configuration found for provider %s, skipping", provider_name)
        return None

    # Map provider names to provider classes
    provider_classes: dict[str, type[ObservabilityProvider]] = {
        "langfuse": LangfuseProvider,
        # Future providers can be added here:
        # "vertex": VertexProvider,
        # "datadog": DatadogProvider,
    }

    provider_class = provider_classes.get(provider_name)
    if provider_class is None:
        logger.error(
            "Unknown provider: %s. Available providers: %s",
            provider_name,
            list(provider_classes.keys()),
        )
        return None

    try:
        # Create provider instance
        # For LangfuseProvider, we can pass the ObservabilityConfig directly
        if provider_name == "langfuse":
            # LangfuseProvider can accept ObservabilityConfig or dict
            return provider_class(name=provider_name, config=config)
        else:
            # Other providers expect dict config
            return provider_class(name=provider_name, config=provider_config)
    except Exception as e:
        logger.error("Failed to create provider %s: %s", provider_name, e, exc_info=True)
        return None


def is_initialized() -> bool:
    """Check if observability has been initialized."""
    return _initialized


def reset_initialization() -> None:
    """Reset initialization state (mainly for testing)."""
    global _initialized
    _initialized = False
    registry = get_provider_registry()
    registry.clear()
