"""
Observability configuration.

Provides configuration classes for observability and tracing settings.
"""

from typing import Any

from pydantic import BaseModel, Field

from orchestrator.config import settings


class ObservabilityConfig(BaseModel):
    """
    Configuration for observability and tracing.

    Supports multiple providers with auto-detection of Langfuse configuration
    from legacy fields when providers list is not explicitly set.
    """

    # Multi-Provider Configuration
    providers: list[str] = Field(
        default_factory=list,
        description="List of provider names to enable (e.g., ['langfuse', 'vertex'])",
    )
    provider_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configurations keyed by provider name",
    )

    # Langfuse Connection (auto-detected if providers not explicitly set)
    enabled: bool = Field(default_factory=lambda: settings.langfuse_enabled)
    public_key: str | None = Field(default_factory=lambda: settings.langfuse_public_key)
    secret_key: str | None = Field(default_factory=lambda: settings.langfuse_secret_key)
    host: str = Field(default_factory=lambda: settings.langfuse_host)

    # Tracing Configuration
    sample_rate: float = Field(
        default_factory=lambda: settings.langfuse_sample_rate,
        ge=0.0,
        le=1.0,
        description="Sampling rate for traces (0.0 to 1.0)",
    )
    flush_interval: int = Field(
        default_factory=lambda: settings.langfuse_flush_interval,
        description="Flush interval in seconds",
    )
    flush_at: int = Field(
        default_factory=lambda: settings.langfuse_flush_at,
        description="Flush when this many events are queued",
    )

    # Debug Settings
    debug: bool = Field(
        default_factory=lambda: settings.langfuse_debug,
        description="Enable debug logging for Langfuse",
    )

    # Default Metadata
    release: str | None = Field(
        default_factory=lambda: settings.langfuse_release,
        description="Release/version identifier",
    )
    environment: str = Field(
        default_factory=lambda: settings.environment,
        description="Environment name (development, staging, production)",
    )

    # Trace defaults
    default_tags: list[str] = Field(
        default_factory=list,
        description="Default tags to add to all traces",
    )
    default_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Default metadata to add to all traces",
    )

    def is_configured(self) -> bool:
        """
        Check if observability is properly configured.

        Returns True if:
        - Any providers are explicitly configured, OR
        - Langfuse is configured (auto-detected from legacy fields)
        """
        # If providers are explicitly set, check if any are configured
        if self.providers:
            return any(
                self.get_provider_config(provider_name) is not None
                for provider_name in self.providers
            )

        # Auto-detect: check Langfuse configuration
        return bool(self.enabled and self.public_key and self.secret_key)

    def get_provider_config(self, provider_name: str) -> dict[str, Any] | None:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider-specific configuration dict or None if not configured
        """
        # Check provider-specific configs first
        if provider_name in self.provider_configs:
            return self.provider_configs[provider_name]

        # Auto-detect: if provider is "langfuse" and not in provider_configs,
        # build config from Langfuse-specific fields
        if provider_name == "langfuse":
            config: dict[str, Any] = {
                "enabled": self.enabled,
                "public_key": self.public_key,
                "secret_key": self.secret_key,
                "host": self.host,
                "sample_rate": self.sample_rate,
                "flush_interval": self.flush_interval,
                "flush_at": self.flush_at,
                "debug": self.debug,
                "release": self.release,
                "environment": self.environment,
                "default_tags": self.default_tags,
                "default_metadata": self.default_metadata,
            }
            # Only return if actually configured
            if config.get("enabled") and config.get("public_key") and config.get("secret_key"):
                return config

        return None

    def to_langfuse_kwargs(self) -> dict[str, Any]:
        """
        Convert config to Langfuse client initialization kwargs.

        Used internally by LangfuseProvider for client initialization.
        """
        kwargs: dict[str, Any] = {
            "host": self.host,
            "flush_interval": self.flush_interval,
            "flush_at": self.flush_at,
            "debug": self.debug,
        }

        if self.public_key:
            kwargs["public_key"] = self.public_key
        if self.secret_key:
            kwargs["secret_key"] = self.secret_key
        if self.release:
            kwargs["release"] = self.release

        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary.

        Useful for passing to provider constructors.
        """
        return self.model_dump(exclude_none=True)
