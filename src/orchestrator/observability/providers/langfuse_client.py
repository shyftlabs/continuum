"""
Langfuse client implementation.

Internal Langfuse client wrapper used by LangfuseProvider.
"""

from __future__ import annotations

import random
import threading
from typing import TYPE_CHECKING, Any

from orchestrator.logging import get_logger
from orchestrator.observability.config import ObservabilityConfig

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = get_logger(__name__)


class LangfuseClient:
    """
    Wrapper around the Langfuse client with SDK-specific features.

    This is an internal class used by LangfuseProvider.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize the Langfuse client wrapper.

        Args:
            config: Optional configuration. Uses global settings if not provided.
            auto_initialize: Whether to initialize the client immediately.
        """
        self._config = config or ObservabilityConfig()
        self._client: Langfuse | None = None
        self._initialized = False
        self._lock = threading.Lock()

        if auto_initialize:
            self.initialize()

    @property
    def config(self) -> ObservabilityConfig:
        """Get the current configuration."""
        return self._config

    @property
    def client(self) -> Langfuse | None:
        """Get the underlying Langfuse client."""
        return self._client

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled and initialized."""
        return self._initialized and self._client is not None

    def initialize(self) -> bool:
        """
        Initialize the Langfuse client.

        Thread-safe initialization that only runs once.

        Returns:
            True if initialization was successful, False otherwise.
        """
        with self._lock:
            if self._initialized:
                return self._client is not None

            if not self._config.is_configured():
                logger.info(
                    "Langfuse not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
                )
                self._initialized = True
                return False

            try:
                from langfuse import Langfuse

                self._client = Langfuse(**self._config.to_langfuse_kwargs())
                self._initialized = True
                logger.debug(
                    "Langfuse initialized",
                    extra={
                        "host": self._config.host,
                        "environment": self._config.environment,
                        "sample_rate": self._config.sample_rate,
                    },
                )
                return True

            except ImportError:
                logger.error("Langfuse package not installed. Run: pip install langfuse")
                self._initialized = True  # package missing won't change, no point retrying
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                # Leave _initialized = False so transient failures (network, bad creds) can be retried
                return False

    def should_sample(self) -> bool:
        """Determine if this trace should be sampled based on sample rate."""
        return random.random() < self._config.sample_rate

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

        if not self.should_sample():
            logger.debug(f"Trace '{name}' not sampled (rate: {self._config.sample_rate})")
            return None

        # Merge default tags and metadata
        all_tags = list(self._config.default_tags)
        if tags:
            all_tags.extend(tags)

        all_metadata = dict(self._config.default_metadata)
        all_metadata["environment"] = self._config.environment
        if metadata:
            all_metadata.update(metadata)

        try:
            return self._client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                input=input,
                output=output,
                metadata=all_metadata,
                tags=all_tags if all_tags else None,
                version=version,
                release=self._config.release,
                public=public,
            )
        except Exception as e:
            logger.warning(f"Failed to create trace: {e}")
            return None

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

        try:
            return self._client.span(
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                level=level,
            )
        except Exception as e:
            logger.warning(f"Failed to create span: {e}")
            return None

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
        """Create a standalone generation (LLM call)."""
        if not self.is_enabled:
            return None

        try:
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
        except Exception as e:
            logger.warning(f"Failed to create generation: {e}")
            return None

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

        try:
            return self._client.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=value,
                comment=comment,
                data_type=data_type,
            )
        except Exception as e:
            logger.warning(f"Failed to create score: {e}")
            return None

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

        try:
            return self._client.event(
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                level=level,
            )
        except Exception as e:
            logger.warning(f"Failed to create event: {e}")
            return None

    def create_prompt(
        self,
        name: str,
        prompt: str | list[dict[str, str]],
        *,
        config: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        is_active: bool = True,
    ) -> Any:
        """Create or update a prompt in Langfuse."""
        if not self.is_enabled:
            return None

        try:
            return self._client.create_prompt(
                name=name,
                prompt=prompt,
                config=config,
                labels=labels,
                is_active=is_active,
            )
        except Exception as e:
            logger.warning(f"Failed to create prompt: {e}")
            return None

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        cache_ttl_seconds: int = 60,
    ) -> Any:
        """Get a prompt from Langfuse."""
        if not self.is_enabled:
            return None

        try:
            return self._client.get_prompt(
                name=name,
                version=version,
                label=label,
                cache_ttl_seconds=cache_ttl_seconds,
            )
        except Exception as e:
            logger.warning(f"Failed to get prompt: {e}")
            return None

    def create_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a dataset for evaluation."""
        if not self.is_enabled:
            return None

        try:
            return self._client.create_dataset(
                name=name,
                description=description,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to create dataset: {e}")
            return None

    def get_dataset(self, name: str) -> Any:
        """Get a dataset by name."""
        if not self.is_enabled:
            return None

        try:
            return self._client.get_dataset(name=name)
        except Exception as e:
            logger.warning(f"Failed to get dataset: {e}")
            return None

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

        try:
            return self._client.create_dataset_item(
                dataset_name=dataset_name,
                input=input,
                expected_output=expected_output,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to create dataset item: {e}")
            return None

    def flush(self) -> None:
        """Flush all pending events to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse: {e}")

    def shutdown(self) -> None:
        """Shutdown the Langfuse client and flush pending events."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
                logger.info("Langfuse client shutdown complete")
            except Exception as e:
                logger.warning(f"Failed to shutdown Langfuse: {e}")
        self._client = None
        self._initialized = False

    def auth_check(self) -> bool:
        """Verify Langfuse authentication."""
        if not self.is_enabled:
            return False

        try:
            result = self._client.auth_check()
            return result
        except Exception as e:
            logger.warning(f"Langfuse auth check failed: {e}")
            return False
