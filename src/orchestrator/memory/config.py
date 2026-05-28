"""
Memory configuration.

Provides configuration classes for long-term memory settings using mem0.
"""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from orchestrator.config import settings
from orchestrator.memory.exceptions import MemoryConfigurationError


class MemoryConfig(BaseModel):
    """
    Configuration for long-term memory.

    Supports multiple providers via the `provider` field. Each provider
    may use different configuration fields.

    Example:
        ```python
        from orchestrator.memory import MemoryConfig

        # Using defaults from environment
        config = MemoryConfig()

        # Explicit configuration with mem0 provider
        config = MemoryConfig(
            provider="mem0",
            enabled=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            memory_llm_model="gpt-4o-mini",
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            memory_isolation="user",
        )

        # Future: Use different provider
        config = MemoryConfig(
            provider="pinecone",
            enabled=True,
            ...
        )
        ```
    """

    # Provider Selection
    provider: str = Field(
        default="mem0",
        description="Memory provider to use: 'mem0', 'pinecone', etc.",
    )

    # Memory Enable/Disable
    enabled: bool = Field(
        default_factory=lambda: settings.memory_enabled,
        description="Enable/disable memory system",
    )

    # Vector Store Provider Selection
    vector_store_provider: Literal["qdrant", "milvus"] = Field(
        default_factory=lambda: settings.vector_store_provider,  # type: ignore[return-value]
        description="Vector store provider: 'qdrant' or 'milvus'",
    )

    # Qdrant Vector Store Configuration
    qdrant_host: str = Field(
        default_factory=lambda: settings.qdrant_host,
        description="Qdrant host URL",
    )
    qdrant_port: int = Field(
        default_factory=lambda: settings.qdrant_port,
        description="Qdrant port",
    )
    qdrant_api_key: str | None = Field(
        default_factory=lambda: settings.qdrant_api_key,
        description="Qdrant API key (for cloud deployment)",
    )
    qdrant_collection: str = Field(
        default_factory=lambda: settings.qdrant_collection,
        description="Qdrant collection name for storing memories",
    )

    # Milvus Vector Store Configuration
    milvus_host: str = Field(
        default_factory=lambda: settings.milvus_host,
        description="Milvus host",
    )
    milvus_port: int = Field(
        default_factory=lambda: settings.milvus_port,
        description="Milvus port",
    )
    milvus_token: str | None = Field(
        default_factory=lambda: settings.milvus_token,
        description="Milvus token (for Zilliz Cloud)",
    )
    milvus_collection: str = Field(
        default_factory=lambda: settings.milvus_collection,
        description="Milvus collection name for storing memories",
    )

    # LLM Configuration for Memory Operations
    memory_llm_model: str = Field(
        default_factory=lambda: settings.memory_llm_model,
        description="LLM model for memory operations (fact extraction, etc.)",
    )
    memory_llm_temperature: float = Field(
        default_factory=lambda: settings.memory_llm_temperature,
        description="Temperature for memory LLM operations",
    )

    # Embedder Configuration
    embedder_provider: str = Field(
        default_factory=lambda: settings.embedder_provider,
        description=(
            "Embedding provider supported by mem0: 'openai', 'azure_openai', 'huggingface', "
            "'ollama', 'gemini', 'vertexai', 'cohere'"
        ),
    )
    embedder_model: str = Field(
        default_factory=lambda: settings.embedder_model,
        description="Embedding model name",
    )
    embedding_dims: int = Field(
        default_factory=lambda: settings.embedding_dims,
        description="Embedding dimensions (must match the output of your chosen model)",
    )
    embedder_api_key: str | None = Field(
        default_factory=lambda: settings.embedder_api_key,
        description="Explicit API key for embedder (falls back to provider env vars)",
    )
    embedder_api_base: str | None = Field(
        default_factory=lambda: settings.embedder_api_base,
        description="Custom API base URL for self-hosted models or Azure",
    )

    # History Store (SQLite for local)
    history_db_path: str = Field(
        default_factory=lambda: settings.memory_history_db_path,
        description="Path to SQLite database for memory history",
    )

    # Memory Behavior
    memory_isolation: Literal["shared", "user", "agent", "conversation"] = Field(
        default_factory=lambda: settings.memory_isolation,
        description="Memory isolation level: shared (all), user, agent, or conversation",
    )

    # Search Configuration
    search_limit: int = Field(
        default_factory=lambda: settings.memory_search_limit,
        description="Default number of memories to retrieve in search",
    )

    # Reranker (disabled by default)
    reranker_enabled: bool = Field(
        default=False,
        description="Enable reranker for search results",
    )

    # Custom Configuration
    custom_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration for advanced mem0 settings",
    )

    def is_configured(self) -> bool:
        """Check if memory is properly configured with basic requirements."""
        if not (
            self.enabled
            and self.memory_llm_model
            and self.embedder_model
            and self.embedding_dims > 0
        ):
            return False
        if self.vector_store_provider == "milvus":
            return bool(self.milvus_host)
        # qdrant
        return bool(self.qdrant_host)

    def _get_embedder_api_key(self) -> str | None:
        """
        Get the API key for the configured embedder provider.

        Priority:
        1. Explicit embedder_api_key setting
        2. Provider-specific environment variable
        """
        if self.embedder_api_key:
            return self.embedder_api_key

        provider = self.embedder_provider.lower()

        # Map providers to their API key environment variables
        provider_env_map: dict[str, list[str]] = {
            "openai": ["OPENAI_API_KEY"],
            "azure_openai": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
            "azure": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
            "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "HF_API_KEY"],
            "hugging_face": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "HF_API_KEY"],
            "hf": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "HF_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "vertexai": ["GOOGLE_APPLICATION_CREDENTIALS"],
            "vertex_ai": ["GOOGLE_APPLICATION_CREDENTIALS"],
            "ollama": [],  # Local, no API key needed
        }

        for env_var in provider_env_map.get(provider, []):
            key = os.environ.get(env_var)
            if key:
                return key

        return None

    def _build_llm_config(self) -> dict[str, Any]:
        """Detect LLM provider from model name and build mem0 llm config block."""
        model = self.memory_llm_model

        if model.startswith("gemini/") or model.startswith("gemini-"):
            model_name = model.removeprefix("gemini/") if model.startswith("gemini/") else model
            return {
                "provider": "gemini",
                "config": {
                    "model": model_name,
                    "temperature": self.memory_llm_temperature,
                    "api_key": os.environ.get("GEMINI_API_KEY", ""),
                },
            }
        if model.startswith("claude"):
            return {
                "provider": "anthropic",
                "config": {
                    "model": model,
                    "temperature": self.memory_llm_temperature,
                    "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                },
            }
        # Default: OpenAI (gpt-* or any unrecognized model)
        return {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": self.memory_llm_temperature,
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        }

    def _build_embedder_config(self) -> tuple[str, dict[str, Any]]:
        """
        Build the embedder configuration for mem0.

        Returns:
            Tuple of (provider_name, config_dict).
        """
        provider = self.embedder_provider.lower()

        # Base config for all providers
        base_config: dict[str, Any] = {
            "model": self.embedder_model,
            "embedding_dims": self.embedding_dims,
        }

        api_key = self._get_embedder_api_key()

        # OpenAI
        if provider == "openai":
            config = base_config.copy()
            if api_key:
                config["api_key"] = api_key
            if self.embedder_api_base:
                config["openai_base_url"] = self.embedder_api_base
            return "openai", config

        # Azure OpenAI
        if provider in ("azure_openai", "azure"):
            config = base_config.copy()
            if api_key:
                config["api_key"] = api_key
            if self.embedder_api_base:
                config["azure_kwargs"] = {"azure_endpoint": self.embedder_api_base}
            api_version = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview")
            if "azure_kwargs" not in config:
                config["azure_kwargs"] = {}
            config["azure_kwargs"]["api_version"] = api_version
            return "azure_openai", config

        # Hugging Face
        if provider in ("huggingface", "hugging_face", "hf"):
            config = base_config.copy()
            if api_key:
                config["api_key"] = api_key
            return "huggingface", config

        # Ollama (local)
        if provider == "ollama":
            config = base_config.copy()
            ollama_base = self.embedder_api_base or os.environ.get(
                "OLLAMA_HOST", "http://localhost:11434"
            )
            config["ollama_base_url"] = ollama_base
            return "ollama", config

        # Google Gemini
        if provider == "gemini":
            config = base_config.copy()
            if api_key:
                config["api_key"] = api_key
            return "gemini", config

        # Vertex AI
        if provider in ("vertexai", "vertex_ai"):
            config = base_config.copy()
            credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials:
                config["vertex_credentials_json"] = credentials
            project = os.environ.get("VERTEX_PROJECT", os.environ.get("GOOGLE_PROJECT"))
            location = os.environ.get("VERTEX_LOCATION", "us-central1")
            if project:
                config["project_id"] = project
            config["location"] = location
            return "vertexai", config

        # Cohere
        if provider == "cohere":
            config = base_config.copy()
            if api_key:
                config["api_key"] = api_key
            return "cohere", config

        # Unsupported provider
        raise MemoryConfigurationError(
            f"Unsupported embedder provider: {self.embedder_provider}. "
            f"Supported: openai, azure_openai, huggingface, ollama, gemini, vertexai, cohere",
            config_key="embedder_provider",
        )

    def to_mem0_config(self) -> dict[str, Any]:
        """
        Convert config to mem0 Memory.from_config() format.

        Returns:
            Dictionary configuration for mem0.
        """
        # Expand history path (mem0 doesn't do this automatically)
        history_path = os.path.expanduser(self.history_db_path)
        # Container environments (e.g. Docker appuser) often have HOME=/nonexistent;
        # avoid [Errno 13] Permission denied by using /tmp when path is under /nonexistent
        if history_path.startswith("/nonexistent"):
            history_path = "/tmp/orchestrator_memory_history.db"

        # Ensure parent directory exists
        from pathlib import Path

        Path(history_path).parent.mkdir(parents=True, exist_ok=True)

        # Build embedder configuration
        embedder_provider, embedder_config = self._build_embedder_config()

        # Build vector store config based on selected provider
        if self.vector_store_provider == "milvus":
            milvus_vs_config: dict[str, Any] = {
                "collection_name": self.milvus_collection,
                "embedding_model_dims": self.embedding_dims,
                "url": f"http://{self.milvus_host}:{self.milvus_port}",
            }
            if self.milvus_token:
                milvus_vs_config["token"] = self.milvus_token
            vector_store_block: dict[str, Any] = {
                "provider": "milvus",
                "config": milvus_vs_config,
            }
        else:
            qdrant_vs_config: dict[str, Any] = {
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "collection_name": self.qdrant_collection,
                "embedding_model_dims": self.embedding_dims,
            }
            if self.qdrant_api_key:
                qdrant_vs_config["api_key"] = self.qdrant_api_key
            vector_store_block = {
                "provider": "qdrant",
                "config": qdrant_vs_config,
            }

        config: dict[str, Any] = {
            "version": "v1.1",
            "llm": self._build_llm_config(),
            "embedder": {
                "provider": embedder_provider,
                "config": embedder_config,
            },
            "vector_store": vector_store_block,
            "history_db_path": history_path,
        }

        # Reranker
        if self.reranker_enabled:
            config["reranker"] = {
                "provider": "cohere",
                "config": {},
            }

        # Merge custom configuration
        if self.custom_config:
            config.update(self.custom_config)

        return config
