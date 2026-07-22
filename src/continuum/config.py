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
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into os.environ BEFORE creating Settings
# This ensures all libraries can read env vars via os.getenv()
load_dotenv()


def _resolve_default_model(
    explicit: str | None,
    *,
    has_openai: bool,
    has_anthropic: bool,
    has_gemini: bool,
    anthropic_model: str,
    gemini_model: str,
) -> str:
    """Pick the default chat model from whatever provider is actually configured.

    An explicit ``DEFAULT_LLM_MODEL`` always wins. Otherwise the default is
    auto-detected from the configured API key, so an Anthropic-only or
    Gemini-only deployment does not silently require an OpenAI key (TL-65).

    Falls back to ``gpt-4o-mini`` (the historical default) when nothing is
    configured — this keeps imports/tests without any key working; a real call
    with no matching provider key still surfaces that provider's own auth error.
    The per-provider model ids are themselves settings, so they stay overridable
    and there is no hardcoded model *list* to maintain.
    """
    if explicit:
        return explicit
    if has_openai:
        return "gpt-4o-mini"
    if has_anthropic:
        return anthropic_model
    if has_gemini:
        return gemini_model
    return "gpt-4o-mini"


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

    # Hugging Face (tier classifier via HF router when LLM_ROUTE_TIER_CLASSIFIER=qwen)
    hf_api_key: str | None = (
        None  # HF_API_KEY — used if LLM_ROUTE_ROUTER_API_KEY / tier_router_api_key unset
    )

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
    # Per-provider default chat model, used by the provider-aware resolver when no
    # explicit DEFAULT_LLM_MODEL is set and only this provider's key is configured.
    # Overridable via ANTHROPIC_DEFAULT_MODEL / GEMINI_DEFAULT_MODEL.
    anthropic_default_model: str = "claude-haiku-4-5"
    gemini_default_model: str = "gemini/gemini-2.5-flash"
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
    # Smart Gateway integration
    # -------------------------------------------------------------------------
    smart_gateway_url: str | None = None  # SMART_GATEWAY_URL
    smart_gateway_api_key: str | None = None  # SMART_GATEWAY_API_KEY
    smart_gateway_default_mode: str = "modest"  # SMART_GATEWAY_DEFAULT_MODE

    # -------------------------------------------------------------------------
    # Smart layer (model_tier routing + tier classifiers)
    # -------------------------------------------------------------------------
    smart_layer_enabled: bool = True  # When False, RouterAgent model_tier falls back to llm routing

    # Playground / SDK env aliases (OpenAI-compatible classifier host, e.g. HF router)
    llm_route_tier_classifier: str | None = None  # LLM_ROUTE_TIER_CLASSIFIER
    llm_route_router_model: str | None = None  # LLM_ROUTE_ROUTER_MODEL → tier_classifier_llm_model
    llm_route_router_api_base: str | None = None  # LLM_ROUTE_ROUTER_API_BASE
    llm_route_router_api_key: str | None = None  # LLM_ROUTE_ROUTER_API_KEY
    llm_route_force_completion_model: str | None = None  # LLM_ROUTE_FORCE_COMPLETION_MODEL
    llm_route_local_router_api_base: str | None = (
        None  # LLM_ROUTE_LOCAL_ROUTER_API_BASE (qwen_local)
    )
    llm_route_local_router_api_key: str | None = None  # LLM_ROUTE_LOCAL_ROUTER_API_KEY
    llm_route_local_router_model: str | None = (
        None  # LLM_ROUTE_LOCAL_ROUTER_MODEL → MLX/local model id for qwen_local
    )
    # When False, skip keyword/length heuristics and always run the classifier LLM (if mode allows).
    llm_route_tier_classifier_heuristic_shortcut: bool | None = (
        None  # LLM_ROUTE_TIER_CLASSIFIER_HEURISTIC_SHORTCUT
    )

    # -------------------------------------------------------------------------
    # Memory Configuration (mem0 with pluggable vector store)
    # -------------------------------------------------------------------------
    memory_enabled: bool = True  # Enable/disable long-term memory

    # Vector Store Provider Selection
    vector_store_provider: str = "milvus"  # "qdrant" | "milvus"

    # Qdrant Vector Store Configuration
    qdrant_host: str = "localhost"  # Qdrant host (use 'localhost' for local Docker)
    qdrant_port: int = 6333  # Qdrant port
    qdrant_api_key: str | None = None  # Qdrant API key (for cloud deployment)
    qdrant_collection: str = "orchestrator_memories"  # Collection name for memories

    # Milvus Vector Store Configuration
    milvus_host: str = "localhost"  # Milvus host
    milvus_port: int = 19530  # Milvus port
    milvus_token: str | None = None  # Milvus token (for Zilliz Cloud)
    milvus_collection: str = "orchestrator_memories"  # Collection name for memories

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
    memory_isolation: Literal["shared", "user", "agent", "conversation"] = "user"  # Isolation level
    memory_search_limit: int = 5  # Default number of memories to retrieve
    memory_max_query_chars: int | None = (
        8000  # Truncate search queries to this many chars (None disables)
    )

    # -------------------------------------------------------------------------
    # Session Configuration (Redis for short-term memory)
    # -------------------------------------------------------------------------
    session_enabled: bool = True  # Enable/disable session management
    session_redis_host: str = "localhost"  # Redis host for sessions
    session_redis_port: int = 6380  # Redis port for sessions (different from Langfuse Redis)
    session_redis_password: str | None = None  # Redis password (matches docker-compose default)
    session_redis_db: int = 0  # Redis database number
    session_redis_ssl: bool = False  # Enable SSL/TLS for Redis
    # TLS certificate verification policy, passed through to redis-py's
    # SSLConnection only when set. None (default) => omit the kwarg => redis-py's
    # verifying default ('required'). Managed endpoints (ElastiCache) need
    # nothing here. Set 'none'/'optional' only for self-signed / private-CA test
    # endpoints. NOTE: passing 'none' disables verification — opt-in, never default.
    session_redis_ssl_cert_reqs: str | None = None
    # Path to a CA bundle for verifying the Redis server cert. None (default) =>
    # omit the kwarg => redis-py uses the system CA store. Set for private-CA endpoints.
    session_redis_ssl_ca_certs: str | None = None
    session_redis_max_connections: int = (
        10  # Redis pool size (configurable via env; floored at the safe minimum)
    )
    session_ttl_seconds: int = 86400 * 7  # Session TTL: 7 days (configurable)
    session_max_messages: int = 1000  # Maximum messages per session (configurable, for scalability)
    session_key_prefix: str = "orchestrator:session"  # Redis key prefix for sessions
    # Long-term memory (mem0) write timing: 'sync' (await before returning) or
    # 'background' (fire-and-forget, faster responses, eventual consistency).
    # Default 'background' — the mem0 fact-extraction (an LLM call) is kept off
    # the response path. Writes inside a Temporal activity are auto-forced to
    # 'sync' regardless of this setting. Override per deployment via
    # SESSION_MEMORY_WRITE_MODE; set 'sync' for strict read-after-write.
    session_memory_write_mode: Literal["sync", "background"] = "background"
    # What to do when Redis-backed session persistence is unavailable
    # (unconfigured / unreachable at startup, or it fails mid-session):
    #   'degrade' (default) — fall back to a non-durable in-memory store and keep
    #             serving (good DX; dev/demo). The SessionClient exposes
    #             ``persistence_degraded=True`` so it can be monitored/alerted.
    #   'fail'    — raise SessionConnectionError instead of silently degrading.
    #             Prefer in strict production where silently losing durability is
    #             worse than failing loudly.
    session_fallback_mode: Literal["degrade", "fail"] = "degrade"

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
    # Headroom Compression (Optional — sidecar or in-process library)
    # -------------------------------------------------------------------------
    # Off by default. Applies to async calls (chat/chat_stream) only. Two modes:
    #   local (default): in-process `import headroom` — no sidecar to run;
    #       requires the [headroom-local] extra. api_base/api_key are ignored.
    #       Headroom env knobs (e.g. HEADROOM_CCR_BACKEND) apply to THIS process.
    #       If the extra isn't installed: fail-open → compression silently
    #       disabled (no crash); fail-closed → error at first use.
    #   endpoint: HTTP to a running `headroom proxy` sidecar — the multi-worker
    #       production mode (fault isolation, one engine for many workers). Set
    #       HEADROOM_MODE=endpoint + HEADROOM_API_BASE to use it.
    # See gap-analysis/headroom-native-integration-plan.md.
    headroom_enabled: bool = False
    headroom_mode: Literal["endpoint", "local"] = "local"  # HEADROOM_MODE
    headroom_api_base: str = "http://127.0.0.1:8787"  # HEADROOM_API_BASE — must be loopback
    headroom_api_key: str | None = None  # HEADROOM_API_KEY — bearer token if the sidecar sets one
    headroom_fail_open: bool = True  # True: compression error → forward uncompressed (recommended)
    headroom_timeout_seconds: float = 30.0  # Compress timeout (large payloads take seconds)
    # In-process prose (Kompress ML `text`) compression — local mode only.
    # Off by default because Headroom deliberately SKIPS the ML model on the hot
    # path (loads it in a background thread, gives each call a ~25ms budget) so
    # prose otherwise never compresses in-process. When True (needs the
    # [headroom-local-ml] extra): (1) pre-warm the model at startup in a daemon
    # thread so it's ready without blocking boot, and (2) raise the per-call
    # execution budget below so a call waits for a slot instead of skipping.
    # Trade-off: prose is the ~23% floor (logs/tables/search are the big wins
    # and need none of this); enabling it costs a one-time warmup + a little
    # per-call latency under contention. Ignored in endpoint mode (the sidecar
    # owns its own Kompress config). No effect on non-prose transforms.
    headroom_kompress_local: bool = False  # HEADROOM_KOMPRESS_LOCAL
    headroom_kompress_execution_timeout_ms: int = 5000  # HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS
    # When Headroom is on, raise the summarizer's trigger so the (cache-hostile,
    # history-rewriting) summarizer fires only as a rare last resort behind
    # Headroom's cache-friendly per-turn compression. max() semantics — never
    # lowers an explicitly higher context_compression_threshold.
    headroom_context_threshold: float = 0.92

    # -------------------------------------------------------------------------
    # Untrusted tool-content hardening (security finding F2 — indirect prompt
    # injection). When True (default), the content of every ``role == "tool"``
    # message is, right before the provider call: (1) stripped of invisible /
    # control characters (Unicode-tag smuggling, zero-width, bidi overrides,
    # C0/C1 controls), and (2) wrapped in a ``<tool_result untrusted="true">``
    # envelope, with a one-line system instruction that content inside such tags
    # is data, never instructions. Defence-in-depth only — the hard boundary is
    # authorization on side-effecting tools, not this. Headroom-independent.
    # Set False for byte-identical legacy behavior.
    # -------------------------------------------------------------------------
    untrusted_tool_content_hardening: bool = True  # UNTRUSTED_TOOL_CONTENT_HARDENING

    # -------------------------------------------------------------------------
    # Temporal Configuration (Optional - requires `pip install shyftlabs-continuum[temporal]`)
    # -------------------------------------------------------------------------
    temporal_enabled: bool = False
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    # Temporal Cloud / TLS connection (local docker needs neither). When an API
    # key is set, TLS is implied. See TemporalConnector for mode inference.
    temporal_tls: bool = False  # TEMPORAL_TLS — enable TLS (managed/cloud)
    temporal_api_key: str | None = None  # TEMPORAL_API_KEY — Temporal Cloud API key
    temporal_task_queue: str = "orchestrator-agents"
    temporal_enable_human_in_loop: bool = True
    temporal_approval_timeout_seconds: int = 86400  # 24h default
    temporal_workflow_execution_timeout: int = 86400 * 7  # 7 days
    temporal_activity_start_to_close_timeout: int = 300  # 5 min per activity
    temporal_activity_retry_max_attempts: int = 3

    # -------------------------------------------------------------------------
    # Decision Trace Configuration (reasoning traceability)
    # -------------------------------------------------------------------------
    # Capture an ordered decision trace per run, attach it to the response, and
    # persist it (keyed by run_id). When disabled, no recorder is created and the
    # execution path is byte-for-byte unchanged.
    decision_trace_enabled: bool = False  # DECISION_TRACE_ENABLED
    # How much of the trace is attached to the response (the full trace is always
    # persisted): 'off' (persist only, attach nothing) or 'full' (attach the
    # complete trace to the response).
    decision_trace_detail: Literal["off", "full"] = "full"
    # Persistence backend: 'redis' (durable, reuses session Redis), 'memory'
    # (process-local), 'null' (don't persist).
    decision_trace_store: Literal["redis", "memory", "null"] = "redis"
    decision_trace_ttl_days: int = 14  # Redis TTL for persisted traces
    # Snapshot the LLM message array at each turn so a run can be forked/replayed
    # from any step (the "what-if"). Heavier (stores prompts); off by default.
    decision_trace_checkpoint: bool = False

    # -------------------------------------------------------------------------
    # Run-State Persistence Configuration
    # -------------------------------------------------------------------------
    # Write per-run state (RunState) to Redis on start/finish, intended for a
    # future pause/resume/recovery feature. Reuses the session Redis instance.
    # Off by default: nothing currently reads this data back, so enabling it only
    # adds Redis writes (and a connection attempt when Redis is unavailable).
    # Turn on via PERSIST_RUN_STATE when a consumer of run-state actually exists.
    persist_run_state: bool = False  # PERSIST_RUN_STATE

    # -------------------------------------------------------------------------
    # Lifecycle Configuration (Shutdown Behavior)
    # -------------------------------------------------------------------------
    shared_services_enabled: bool = (
        True  # If True, Redis/Langfuse are shared services that persist after shutdown
    )
    # When True: Only flush Langfuse traces, don't shutdown client. Don't close Redis connections.
    # When False: Fully shutdown Langfuse and close Redis connections on shutdown.

    @model_validator(mode="after")
    def _apply_provider_aware_defaults(self) -> "Settings":
        """Make the OpenAI-model defaults provider-aware (TL-65).

        ``default_llm_model``, ``memory_llm_model`` and
        ``context_summarization_model`` all historically defaulted to an OpenAI
        model, so an Anthropic- or Gemini-only deployment still needed an OpenAI
        key for routing, reflection, the tier classifier, memory fact-extraction
        and summarization. Resolve the chat default from whatever provider key is
        configured; memory and summarization inherit it unless set explicitly.

        Explicit values are preserved. Router routing and reflection critique need
        no change here — they already read ``default_llm_model``. The Smart Gateway
        path is unaffected (it rewrites the model to ``auto/<tier>`` downstream).
        """
        fields_set = self.model_fields_set
        self.default_llm_model = _resolve_default_model(
            self.default_llm_model if "default_llm_model" in fields_set else None,
            has_openai=bool(self.openai_api_key),
            has_anthropic=bool(self.anthropic_api_key),
            has_gemini=bool(self.gemini_api_key),
            anthropic_model=self.anthropic_default_model,
            gemini_model=self.gemini_default_model,
        )
        if "memory_llm_model" not in fields_set:
            self.memory_llm_model = self.default_llm_model
        if "context_summarization_model" not in fields_set:
            self.context_summarization_model = self.default_llm_model
        return self

    def __repr__(self) -> str:
        """Mask all secret/key/password fields in repr output."""
        from continuum.utils.secrets import mask_value

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
