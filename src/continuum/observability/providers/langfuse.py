"""
Langfuse provider implementation.

Wraps the existing LangfuseClient to implement the ObservabilityProvider interface.
"""

from __future__ import annotations

from typing import Any

from continuum.logging import get_logger
from continuum.observability.config import ObservabilityConfig
from continuum.observability.providers.base import (
    ObservabilityProvider,
    ProviderCapabilities,
)
from continuum.observability.providers.langfuse_client import LangfuseClient

logger = get_logger(__name__)


class LangfuseProvider(ObservabilityProvider):
    """
    Langfuse provider implementation.

    Wraps the existing LangfuseClient to provide Langfuse observability
    through the provider interface.

    Example:
        ```python
        from continuum.observability.providers.langfuse import LangfuseProvider
        from continuum.observability.config import ObservabilityConfig

        config = ObservabilityConfig()
        provider = LangfuseProvider("langfuse", config=config.to_dict())
        ```
    """

    def __init__(
        self,
        name: str = "langfuse",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Langfuse provider.

        Args:
            name: Provider name (default: "langfuse")
            config: Provider configuration dict or ObservabilityConfig
        """
        # Convert dict config to ObservabilityConfig if needed
        if config is None:
            obs_config = ObservabilityConfig()
        elif isinstance(config, ObservabilityConfig):
            obs_config = config
        else:
            # Try to create ObservabilityConfig from dict
            try:
                obs_config = ObservabilityConfig(**config)
            except Exception as e:
                logger.warning("Failed to create ObservabilityConfig from dict: %s", e)
                obs_config = ObservabilityConfig()

        super().__init__(name, config)
        self._obs_config = obs_config
        self._client = LangfuseClient(config=obs_config, auto_initialize=True)
        self._enabled = self._client.is_enabled

    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled."""
        return self._enabled and self._client.is_enabled

    def enable(self) -> None:
        """Enable the provider."""
        super().enable()
        if not self._client.is_enabled:
            self._client.initialize()

    def disable(self) -> None:
        """Disable the provider."""
        super().disable()

    def supports_feature(self, feature: ProviderCapabilities) -> bool:
        """Check if Langfuse supports a feature."""
        # Langfuse supports all standard features
        supported_features = {
            ProviderCapabilities.TRACE,
            ProviderCapabilities.SPAN,
            ProviderCapabilities.GENERATION,
            ProviderCapabilities.EVENT,
            ProviderCapabilities.SCORE,
            ProviderCapabilities.PROMPT_MANAGEMENT,
            ProviderCapabilities.DATASET_MANAGEMENT,
            ProviderCapabilities.ERROR_REPORTING,
            ProviderCapabilities.METRICS,
            ProviderCapabilities.BATCH_FLUSH,
        }
        return feature in supported_features

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
        """Create a new trace."""
        if not self.is_enabled:
            return None
        return self._client.trace(
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
        """Create a standalone span."""
        if not self.is_enabled:
            return None
        return self._client.span(
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
        )

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
        """Create a standalone generation."""
        if not self.is_enabled:
            return None
        return self._client.generation(
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
        """Create an event."""
        if not self.is_enabled:
            return None
        return self._client.event(
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
        )

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
        """Add a score to a trace or observation."""
        if not self.is_enabled:
            return None
        return self._client.score(
            trace_id=trace_id,
            observation_id=observation_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
        )

    def flush(self) -> None:
        """Flush all pending events."""
        if self._client:
            self._client.flush()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        if self._client:
            self._client.shutdown()

    def create_prompt(
        self,
        name: str,
        prompt: str | list[dict[str, str]],
        *,
        config: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        is_active: bool = True,
    ) -> Any:
        """Create or update a prompt."""
        if not self.is_enabled:
            return None
        return self._client.create_prompt(
            name=name,
            prompt=prompt,
            config=config,
            labels=labels,
            is_active=is_active,
        )

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        cache_ttl_seconds: int = 60,
    ) -> Any:
        """Get a prompt."""
        if not self.is_enabled:
            return None
        return self._client.get_prompt(
            name=name,
            version=version,
            label=label,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    def create_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a dataset."""
        if not self.is_enabled:
            return None
        return self._client.create_dataset(
            name=name,
            description=description,
            metadata=metadata,
        )

    def get_dataset(self, name: str) -> Any:
        """Get a dataset."""
        if not self.is_enabled:
            return None
        return self._client.get_dataset(name=name)

    def create_dataset_item(
        self,
        dataset_name: str,
        *,
        input: Any,
        expected_output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Add an item to a dataset."""
        if not self.is_enabled:
            return None
        return self._client.create_dataset_item(
            dataset_name=dataset_name,
            input=input,
            expected_output=expected_output,
            metadata=metadata,
        )

    @property
    def client(self) -> LangfuseClient:
        """Get the underlying LangfuseClient."""
        return self._client
