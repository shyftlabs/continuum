"""
Session configuration.

Provides configuration classes for session management settings.
"""

from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.config import settings


class SessionConfig(BaseModel):
    """Configuration for session management.

    Supports multiple providers via the `provider` field. Each provider
    may use different configuration fields. Redis-specific fields are
    kept for backward compatibility when using the Redis provider.

    Example:
        ```python
        from orchestrator.session import SessionConfig

        # Using defaults from environment (Redis)
        config = SessionConfig()

        # Explicit configuration with Redis provider
        config = SessionConfig(
            provider="redis",
            enabled=True,
            redis_host="localhost",
            redis_port=6379,
        )

        # Future: Use different provider
        config = SessionConfig(
            provider="dynamodb",
            enabled=True,
            # DynamoDB-specific config...
        )
        ```
    """

    # Provider Selection
    provider: str = Field(
        default="redis",
        description="Session provider to use: 'redis', 'dynamodb', etc.",
    )

    # Session Enable/Disable
    enabled: bool = Field(
        default_factory=lambda: settings.session_enabled,
        description="Enable/disable session management",
    )

    # Redis Configuration (for Redis provider)
    redis_host: str = Field(
        default_factory=lambda: settings.session_redis_host,
        description="Redis host for session storage",
    )
    redis_port: int = Field(
        default_factory=lambda: settings.session_redis_port,
        description="Redis port for session storage",
    )
    redis_password: str | None = Field(
        default_factory=lambda: settings.session_redis_password,
        description="Redis password for authentication",
    )
    redis_db: int = Field(
        default_factory=lambda: settings.session_redis_db,
        description="Redis database number",
    )
    redis_ssl: bool = Field(
        default_factory=lambda: settings.session_redis_ssl,
        description="Enable SSL/TLS for Redis",
    )

    # Connection Pool Configuration
    redis_max_connections: int = Field(
        default=10,
        description="Maximum Redis connections in pool",
    )

    # Session Behavior
    ttl_seconds: int = Field(
        default_factory=lambda: settings.session_ttl_seconds,
        description="Session TTL in seconds (default: 7 days)",
    )
    max_messages: int = Field(
        default_factory=lambda: settings.session_max_messages,
        description="Maximum messages per session (for scalability)",
    )
    key_prefix: str = Field(
        default_factory=lambda: settings.session_key_prefix,
        description="Redis key prefix for sessions",
    )

    # Message Limit Behavior
    message_limit_strategy: Literal["error", "sliding_window"] = Field(
        default="sliding_window",
        description=(
            "Strategy when message limit is reached. "
            "'error' raises SessionMessageLimitError, "
            "'sliding_window' removes oldest messages to make room for new ones."
        ),
    )
    sliding_window_trim_count: int = Field(
        default=100,
        description=(
            "Number of oldest messages to remove when sliding window is triggered. "
            "Higher values reduce trim frequency but remove more history at once."
        ),
    )

    def is_configured(self) -> bool:
        """Check if session is properly configured."""
        if not self.enabled:
            return False

        # Provider-specific configuration checks
        if self.provider == "redis":
            return bool(self.redis_host)
        # Add checks for other providers as they're added
        # elif self.provider == "dynamodb":
        #     return bool(self.dynamodb_table_name)

        # Unknown provider — fail explicitly rather than silently proceeding
        # and crashing later with a confusing error.
        raise ValueError(
            f"Unknown session provider: '{self.provider}'. "
            f"Supported providers: 'redis'. "
            f"Check the SESSION_PROVIDER environment variable."
        )

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        protocol = "rediss" if self.redis_ssl else "redis"
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
