"""
Provider manager for coordinating multiple observability providers.

Routes observability calls to all registered providers and handles errors gracefully.
"""

from __future__ import annotations

from typing import Any

from orchestrator.logging import get_logger
from orchestrator.observability.providers.base import (
    ProviderCapabilities,
)
from orchestrator.observability.providers.registry import get_provider_registry

logger = get_logger(__name__)

# Global provider manager instance
_global_manager: ProviderManager | None = None


class ProviderManager:
    """
    Manages multiple observability providers and routes calls to all of them.

    This class implements the same interface as a single provider but
    coordinates multiple providers. If one provider fails, others continue
    to work.

    Example:
        ```python
        manager = ProviderManager()
        manager.trace("my-trace", user_id="user-123")
        # Routes to all registered providers
        ```
    """

    def __init__(self, registry: Any | None = None):
        """
        Initialize the provider manager.

        Args:
            registry: Optional provider registry (uses global if not provided)
        """
        self._registry = registry or get_provider_registry()

    @property
    def is_enabled(self) -> bool:
        """Check if any provider is enabled."""
        enabled = self._registry.get_enabled()
        return len(enabled) > 0

    def supports_feature(self, feature: ProviderCapabilities) -> bool:
        """
        Check if any provider supports a feature.

        Args:
            feature: The capability to check

        Returns:
            True if at least one provider supports the feature
        """
        providers = self._registry.get_enabled()
        return any(provider.supports_feature(feature) for provider in providers.values())

    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        version: str | None = None,
        public: bool = False,
    ) -> Any:
        """
        Create a trace in all registered providers.

        Returns the result from the first provider that succeeds, or None.
        """
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.TRACE):
                continue

            try:
                result = provider.trace(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input,
                    output=output,
                    metadata=metadata,
                    tags=tags,
                    version=version,
                    public=public,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create trace: {e}",
                    exc_info=True,
                )

        # Return first successful result (typically from first provider in registry)
        return results[0] if results else None

    def span(
        self,
        *,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ) -> Any:
        """Create a span in all registered providers."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.SPAN):
                continue

            try:
                result = provider.span(
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create span: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def generation(
        self,
        *,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
        name: str,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
        usage: dict[str, int] | None = None,
    ) -> Any:
        """Create a generation in all registered providers."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.GENERATION):
                continue

            try:
                result = provider.generation(
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    name=name,
                    model=model,
                    model_parameters=model_parameters,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level,
                    usage=usage,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create generation: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def event(
        self,
        *,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ) -> Any:
        """Create an event in all registered providers."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.EVENT):
                continue

            try:
                result = provider.event(
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create event: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def score(
        self,
        *,
        trace_id: str,
        observation_id: str | None = None,
        name: str,
        value: float,
        comment: str | None = None,
        data_type: str | None = None,
    ) -> Any:
        """Add a score in all registered providers."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.SCORE):
                continue

            try:
                result = provider.score(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    name=name,
                    value=value,
                    comment=comment,
                    data_type=data_type,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create score: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def flush(self) -> None:
        """Flush all registered providers."""
        providers = self._registry.get_enabled()
        for provider_name, provider in providers.items():
            try:
                provider.flush()
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to flush: {e}",
                    exc_info=True,
                )

    def shutdown(self) -> None:
        """Shutdown all registered providers."""
        providers = self._registry.get_all()
        for provider_name, provider in providers.items():
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to shutdown: {e}",
                    exc_info=True,
                )

    def create_prompt(
        self,
        name: str,
        prompt: str | list[dict[str, str]],
        *,
        config: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        is_active: bool = True,
    ) -> Any:
        """Create a prompt in providers that support it."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.PROMPT_MANAGEMENT):
                continue

            try:
                result = provider.create_prompt(
                    name=name,
                    prompt=prompt,
                    config=config,
                    labels=labels,
                    is_active=is_active,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create prompt: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        cache_ttl_seconds: int = 60,
    ) -> Any:
        """Get a prompt from providers that support it."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        # Try providers in order, return first successful result
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.PROMPT_MANAGEMENT):
                continue

            try:
                result = provider.get_prompt(
                    name=name,
                    version=version,
                    label=label,
                    cache_ttl_seconds=cache_ttl_seconds,
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to get prompt: {e}",
                    exc_info=True,
                )

        return None

    def create_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a dataset in providers that support it."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
                continue

            try:
                result = provider.create_dataset(
                    name=name,
                    description=description,
                    metadata=metadata,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create dataset: {e}",
                    exc_info=True,
                )

        return results[0] if results else None

    def get_dataset(self, name: str) -> Any:
        """Get a dataset from providers that support it."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
                continue

            try:
                result = provider.get_dataset(name=name)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to get dataset: {e}",
                    exc_info=True,
                )

        return None

    def create_dataset_item(
        self,
        dataset_name: str,
        *,
        input: Any,
        expected_output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Add an item to a dataset in providers that support it."""
        providers = self._registry.get_enabled()
        if not providers:
            return None

        results = []
        for provider_name, provider in providers.items():
            if not provider.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
                continue

            try:
                result = provider.create_dataset_item(
                    dataset_name=dataset_name,
                    input=input,
                    expected_output=expected_output,
                    metadata=metadata,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed to create dataset item: {e}",
                    exc_info=True,
                )

        return results[0] if results else None


def get_provider_manager() -> ProviderManager:
    """
    Get the global provider manager instance.

    Returns:
        The global ProviderManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ProviderManager()
    return _global_manager
