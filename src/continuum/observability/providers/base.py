"""
Base provider interface for observability providers.

Defines the abstract interface that all observability providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from continuum.logging import get_logger

logger = get_logger(__name__)


class ProviderCapabilities(str, Enum):
    """Capabilities that providers may support."""

    TRACE = "trace"
    SPAN = "span"
    GENERATION = "generation"
    EVENT = "event"
    SCORE = "score"
    PROMPT_MANAGEMENT = "prompt_management"
    DATASET_MANAGEMENT = "dataset_management"
    ERROR_REPORTING = "error_reporting"
    METRICS = "metrics"
    STREAMING = "streaming"
    BATCH_FLUSH = "batch_flush"


class ObservabilityProvider(ABC):
    """
    Abstract base class for observability providers.

    All observability providers must implement this interface to be compatible
    with the observability module. Providers should gracefully degrade for
    unsupported features by returning None or no-op implementations.

    Example:
        ```python
        class MyProvider(ObservabilityProvider):
            def trace(self, name: str, **kwargs):
                # Implementation
                pass
        ```
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        Args:
            name: Unique name for this provider
            config: Provider-specific configuration
        """
        self.name = name
        self.config = config or {}
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled and ready to use."""
        return self._enabled

    def enable(self) -> None:
        """Enable the provider."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the provider."""
        self._enabled = False

    @abstractmethod
    def supports_feature(self, feature: ProviderCapabilities) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: The capability to check

        Returns:
            True if the feature is supported, False otherwise
        """
        pass

    @abstractmethod
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
        Create a new trace.

        Args:
            name: Name of the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
            input: Input data
            output: Output data
            metadata: Additional metadata
            tags: Tags for filtering
            version: Version identifier
            public: Whether trace is public

        Returns:
            Provider-specific trace object or None if not supported/disabled
        """
        pass

    @abstractmethod
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
        """
        Create a standalone span.

        Args:
            trace_id: Optional trace ID to attach to
            parent_observation_id: Optional parent observation ID
            name: Name of the span
            input: Input data
            output: Output data
            metadata: Additional metadata
            level: Log level (DEFAULT, DEBUG, WARNING, ERROR)

        Returns:
            Provider-specific span object or None if not supported/disabled
        """
        pass

    @abstractmethod
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
        """
        Create a standalone generation (LLM call).

        Args:
            trace_id: Optional trace ID to attach to
            parent_observation_id: Optional parent observation ID
            name: Name of the generation
            model: Model name
            model_parameters: Model parameters (temperature, etc.)
            input: Input data
            output: Output data
            metadata: Additional metadata
            level: Log level
            usage: Token usage information

        Returns:
            Provider-specific generation object or None if not supported/disabled
        """
        pass

    @abstractmethod
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
        """
        Create an event.

        Args:
            trace_id: Optional trace ID to attach to
            parent_observation_id: Optional parent observation ID
            name: Event name
            input: Input data
            output: Output data
            metadata: Additional metadata
            level: Log level

        Returns:
            Provider-specific event object or None if not supported/disabled
        """
        pass

    @abstractmethod
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
        """
        Add a score to a trace or observation.

        Args:
            trace_id: ID of the trace to score
            observation_id: Optional specific observation to score
            name: Name of the score
            value: Score value
            comment: Optional comment
            data_type: Type of score data

        Returns:
            Provider-specific score object or None if not supported/disabled
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush all pending events to the provider."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the provider and flush pending events."""
        pass

    def create_prompt(
        self,
        name: str,
        prompt: str | list[dict[str, str]],
        *,
        config: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        is_active: bool = True,
    ) -> Any:
        """
        Create or update a prompt (if supported).

        Args:
            name: Unique name for the prompt
            prompt: Prompt text or chat messages
            config: Model configuration
            labels: Labels for the prompt
            is_active: Whether this is the active version

        Returns:
            Provider-specific prompt object or None if not supported
        """
        if not self.supports_feature(ProviderCapabilities.PROMPT_MANAGEMENT):
            logger.debug("Provider %s does not support prompt management", self.name)
            return None
        return None

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        cache_ttl_seconds: int = 60,
    ) -> Any:
        """
        Get a prompt (if supported).

        Args:
            name: Name of the prompt
            version: Specific version
            label: Specific label
            cache_ttl_seconds: Cache TTL

        Returns:
            Provider-specific prompt object or None if not supported
        """
        if not self.supports_feature(ProviderCapabilities.PROMPT_MANAGEMENT):
            logger.debug("Provider %s does not support prompt management", self.name)
            return None
        return None

    def create_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create a dataset (if supported).

        Args:
            name: Dataset name
            description: Optional description
            metadata: Optional metadata

        Returns:
            Provider-specific dataset object or None if not supported
        """
        if not self.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
            logger.debug("Provider %s does not support dataset management", self.name)
            return None
        return None

    def get_dataset(self, name: str) -> Any:
        """
        Get a dataset (if supported).

        Args:
            name: Dataset name

        Returns:
            Provider-specific dataset object or None if not supported
        """
        if not self.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
            logger.debug("Provider %s does not support dataset management", self.name)
            return None
        return None

    def create_dataset_item(
        self,
        dataset_name: str,
        *,
        input: Any,
        expected_output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Add an item to a dataset (if supported).

        Args:
            dataset_name: Name of the dataset
            input: Input data
            expected_output: Expected output
            metadata: Optional metadata

        Returns:
            Provider-specific dataset item object or None if not supported
        """
        if not self.supports_feature(ProviderCapabilities.DATASET_MANAGEMENT):
            logger.debug("Provider %s does not support dataset management", self.name)
            return None
        return None
