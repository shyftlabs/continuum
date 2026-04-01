"""
Global configuration for the Orchestrator SDK.

Loads configuration from environment variables using pydantic-settings.
Environment variables are loaded from .env file into os.environ first,
then pydantic-settings reads them. This ensures both our SDK and
external libraries can access the same variables.
"""

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into os.environ BEFORE creating Settings
# This ensures all libraries can read env vars via os.getenv()
load_dotenv()


class Settings(BaseSettings):
    """
    Global settings loaded from environment variables.

    Environment variables are loaded from os.environ (which is populated
    from .env file by load_dotenv() above). This ensures consistency
    between our SDK and external libraries that read from os.environ.
    """

    model_config = SettingsConfigDict(
        # Read from os.environ (already populated by load_dotenv)
        # Not reading directly from .env to avoid duplicate loading
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str | None = None
    openai_organization: str | None = None

    # -------------------------------------------------------------------------
    # Google Gemini Configuration
    # -------------------------------------------------------------------------
    gemini_api_key: str | None = None
    google_application_credentials: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None

    # -------------------------------------------------------------------------
    # Anthropic Configuration
    # -------------------------------------------------------------------------
    anthropic_api_key: str | None = None

    # -------------------------------------------------------------------------
    # Azure OpenAI Configuration
    # -------------------------------------------------------------------------
    azure_api_key: str | None = None
    azure_api_base: str | None = None
    azure_api_version: str | None = None

    # -------------------------------------------------------------------------
    # Default LLM Configuration
    # -------------------------------------------------------------------------
    default_llm_model: str = "gpt-4o-mini"
    fallback_llm_model: str = "gemini/gemini-1.5-flash"
    default_llm_temperature: float = 0.7
    default_llm_max_tokens: int = 4096
    llm_request_timeout: int = 60
    llm_max_retries: int = 3
    llm_enable_fallback: bool = True

    # -------------------------------------------------------------------------
    # Langfuse Configuration (Self-Hosted)
    # -------------------------------------------------------------------------
    langfuse_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"  # Self-hosted default

    # Tracing Configuration
    langfuse_sample_rate: float = 1.0  # 1.0 = trace everything
    langfuse_flush_interval: int = 1  # Flush interval in seconds
    langfuse_flush_at: int = 15  # Flush when this many events are queued
    langfuse_debug: bool = False  # Enable debug logging for Langfuse
    langfuse_release: str | None = None  # Release/version identifier

    # -------------------------------------------------------------------------
    # Environment Configuration
    # -------------------------------------------------------------------------
    environment: str = "development"  # development, staging, production

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # -------------------------------------------------------------------------
    # Memory Configuration (mem0 with Qdrant)
    # -------------------------------------------------------------------------
    memory_enabled: bool = True  # Enable/disable long-term memory

    # Qdrant Vector Store Configuration
    qdrant_host: str = "localhost"  # Qdrant host (use 'localhost' for local Docker)
    qdrant_port: int = 6333  # Qdrant port
    qdrant_api_key: str | None = None  # Qdrant API key (for cloud deployment)
    qdrant_collection: str = "orchestrator_memories"  # Collection name for memories

    # Memory LLM Configuration (use cheap models for memory operations)
    memory_llm_model: str = "gpt-4o-mini"  # LLM for fact extraction
    memory_llm_temperature: float = 0.1  # Lower temperature for consistent fact extraction

    # Embedder Configuration
    # Provider options supported by mem0: "openai", "azure_openai", "huggingface", "ollama",
    #                                     "gemini", "vertexai", "cohere"
    # Supported by mem0: "openai", "azure_openai", "huggingface", "ollama", "gemini", "vertexai", "cohere"
    embedder_provider: str = "openai"  # Embedding provider
    embedder_model: str = "text-embedding-3-small"  # Embedding model name
    embedding_dims: int = 1536  # Embedding dimensions (must match model output)

    # Embedder API Configuration
    # Model format varies by provider:
    #   - openai: "text-embedding-3-small", "text-embedding-3-large"
    #   - huggingface: "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"
    #   - cohere: "embed-english-v3.0", "embed-multilingual-v3.0"
    #   - ollama: "nomic-embed-text", "mxbai-embed-large"
    embedder_api_key: str | None = (
        None  # Explicit API key for embedder (falls back to provider-specific env vars)
    )
    embedder_api_base: str | None = (
        None  # Custom API base URL (for self-hosted models, Azure, etc.)
    )

    # Memory Behavior
    memory_history_db_path: str = "~/.orchestrator/memory_history.db"  # SQLite history DB
    memory_isolation: Literal["shared", "user", "agent", "run"] = "user"  # Isolation level
    memory_search_limit: int = 5  # Default number of memories to retrieve

    # -------------------------------------------------------------------------
    # Session Configuration (Redis for short-term memory)
    # -------------------------------------------------------------------------
    session_enabled: bool = True  # Enable/disable session management
    session_redis_host: str = "localhost"  # Redis host for sessions
    session_redis_port: int = 6380  # Redis port for sessions (different from Langfuse Redis)
    session_redis_password: str | None = None  # Redis password (matches docker-compose default)
    session_redis_db: int = 0  # Redis database number
    session_redis_ssl: bool = False  # Enable SSL/TLS for Redis
    session_ttl_seconds: int = 86400 * 7  # Session TTL: 7 days (configurable)
    session_max_messages: int = 1000  # Maximum messages per session (configurable, for scalability)
    session_key_prefix: str = "orchestrator:session"  # Redis key prefix for sessions

    # -------------------------------------------------------------------------
    # Context Management Configuration (Dynamic Context Compression)
    # -------------------------------------------------------------------------
    context_management_enabled: bool = True  # Enable/disable automatic context management
    context_compression_threshold: float = (
        0.8  # Compress when context reaches 80% of limit (0.0-1.0)
    )
    context_summarization_model: str = (
        "gpt-4o-mini"  # Model for summarization (cheap model recommended)
    )
    context_summarization_temperature: float = (
        0.1  # Temperature for summarization (lower = more consistent)
    )
    context_summarization_timeout: int = 30  # Timeout for summarization in seconds
    context_summarization_max_retries: int = 2  # Max retries for summarization on failure
    context_keep_recent_messages: int = 10  # Number of recent messages to keep when compressing
    context_enable_caching: bool = True  # Cache summaries to avoid re-summarizing same content
    context_cache_ttl_seconds: int = 3600  # Cache TTL for summaries (1 hour)

    # -------------------------------------------------------------------------
    # Temporal Configuration (Optional - requires `pip install shyftlabs-continuum[temporal]`)
    # -------------------------------------------------------------------------
    temporal_enabled: bool = False
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "orchestrator-agents"
    temporal_enable_human_in_loop: bool = True
    temporal_approval_timeout_seconds: int = 86400  # 24h default
    temporal_workflow_execution_timeout: int = 86400 * 7  # 7 days
    temporal_activity_start_to_close_timeout: int = 300  # 5 min per activity
    temporal_activity_retry_max_attempts: int = 3

    # -------------------------------------------------------------------------
    # Lifecycle Configuration (Shutdown Behavior)
    # -------------------------------------------------------------------------
    shared_services_enabled: bool = (
        True  # If True, Redis/Langfuse are shared services that persist after shutdown
    )
    # When True: Only flush Langfuse traces, don't shutdown client. Don't close Redis connections.
    # When False: Fully shutdown Langfuse and close Redis connections on shutdown.

    def __repr__(self) -> str:
        """Mask all secret/key/password fields in repr output."""
        from orchestrator.utils.secrets import mask_value

        parts: list[str] = []
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if any(
                s in field_name for s in ("_key", "_secret", "_password", "api_key")
            ) and isinstance(value, str):
                parts.append(f"{field_name}={mask_value(value)!r}")
            else:
                parts.append(f"{field_name}={value!r}")
        return f"Settings({', '.join(parts)})"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
