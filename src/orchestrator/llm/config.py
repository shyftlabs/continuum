"""
LLM-specific configuration.

Provides configuration classes for LLM client settings.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from orchestrator.config import settings

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent


class LLMConfig(BaseModel):
    """Configuration for LLM client requests."""

    # Model Configuration
    model: str = Field(default_factory=lambda: settings.default_llm_model)
    fallback_models: list[str] = Field(
        default_factory=lambda: [settings.fallback_llm_model] if settings.fallback_llm_model else []
    )

    # Generation Parameters
    temperature: float = Field(default_factory=lambda: settings.default_llm_temperature)
    max_tokens: int | None = Field(default_factory=lambda: settings.default_llm_max_tokens)
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    seed: int | None = None

    # Request Configuration
    timeout: int = Field(default_factory=lambda: settings.llm_request_timeout)
    max_retries: int = Field(default_factory=lambda: settings.llm_max_retries)
    enable_fallback: bool = Field(default_factory=lambda: settings.llm_enable_fallback)

    # Response Format
    # Can be a dict (for json_object or json_schema), or a Pydantic model class
    response_format: dict[str, Any] | type[BaseModel] | None = None
    json_mode: bool = False

    # Metadata
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Advanced provider options
    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    custom_llm_provider: str | None = None

    # Rate limiting
    rate_limit_rpm: int | None = None

    # Caching
    cache: bool = False
    cache_ttl: int | None = None

    # Smart Gateway
    extra_body: dict[str, Any] | None = None  # passed as extra_body to the OpenAI SDK call
    gateway_router_mode: str | None = None  # value for x-portkey-router-mode header

    def to_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs for LLM completion call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
        }

        if self.enable_fallback and self.fallback_models:
            kwargs["fallbacks"] = self.fallback_models

        # Add optional parameters
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.user is not None:
            kwargs["user"] = self.user

        # Response format
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        elif self.response_format is not None:
            # Handle Pydantic models
            if isinstance(self.response_format, type) and issubclass(
                self.response_format, BaseModel
            ):
                kwargs["response_format"] = self.response_format
            elif isinstance(self.response_format, dict):
                # Already a dict format (json_object or json_schema)
                kwargs["response_format"] = self.response_format

        # Custom provider settings
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.custom_llm_provider is not None:
            kwargs["custom_llm_provider"] = self.custom_llm_provider

        # Caching
        if self.cache:
            kwargs["cache"] = {"type": "local"}
            if self.cache_ttl:
                kwargs["cache"]["ttl"] = self.cache_ttl

        # Add metadata if present
        if self.metadata:
            kwargs["metadata"] = self.metadata

        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body

        return kwargs

    def with_overrides(self, **kwargs: Any) -> "LLMConfig":
        """Create a new config with overrides applied."""
        data = self.model_dump()
        data.update(kwargs)
        return LLMConfig(**data)

    @classmethod
    def from_agent_config(cls, agent: "BaseAgent") -> "LLMConfig":
        """
        Create LLMConfig from agent configuration.

        Handles JSON mode configuration:
        - If enable_json_mode is True and json_schema is a Pydantic model: uses the model directly
        - If enable_json_mode is True and json_schema is a dict: creates json_schema format
        - If enable_json_mode is True and json_schema is None: uses simple json_object mode

        Args:
            agent: BaseAgent instance with JSON mode configuration

        Returns:
            LLMConfig with appropriate response_format set
        """
        config = cls(
            model=agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            gateway_router_mode=getattr(agent, "gateway_mode", None),
        )

        if agent.enable_json_mode:
            if agent.json_schema is not None:
                # Check if json_schema is a Pydantic model
                if isinstance(agent.json_schema, type) and issubclass(agent.json_schema, BaseModel):
                    config.response_format = agent.json_schema
                elif isinstance(agent.json_schema, dict):
                    config.response_format = {
                        "type": "json_schema",
                        "json_schema": agent.json_schema,
                        "strict": agent.json_strict,
                    }
                else:
                    # Fallback to simple JSON mode
                    config.json_mode = True
            else:
                # Simple JSON object mode
                config.json_mode = True

        return config
