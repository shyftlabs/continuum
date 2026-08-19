"""
LLM-specific configuration.

Provides configuration classes for LLM client settings.
"""

import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from continuum.config import settings
from continuum.llm.structured_output import to_openai_response_format

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent


class LLMConfig(BaseModel):
    """Configuration for LLM client requests."""

    # Model Configuration
    model: str = Field(default_factory=lambda: settings.default_llm_model)
    fallback_models: list[str] = Field(
        default_factory=lambda: [settings.fallback_llm_model] if settings.fallback_llm_model else []
    )

    # Generation Parameters
    # None omits the parameter entirely (for providers/models that reject it).
    temperature: float | None = Field(default_factory=lambda: settings.default_llm_temperature)
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
    # Accepts a dict (json_object / json_schema) or a Pydantic model class; a
    # model class is normalized to the json_schema dict on the way in (see
    # _normalize_response_format) so providers only ever see one shape.
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

    @field_validator("response_format", mode="before")
    @classmethod
    def _normalize_response_format(cls, value: Any) -> Any:
        """Turn a Pydantic model class into the json_schema dict it describes.

        The OpenAI SDK refuses a model class on ``chat.completions.create()``
        ("You must use chat.completions.parse() instead"), which surfaced as an
        LLMError on every OpenAI-wire provider — OpenAI, Gemini and the Smart
        Gateway alike. Normalizing here means the class form stays a usable way
        to ask for a schema, and every provider receives one predictable shape.
        """
        if isinstance(value, type) and issubclass(value, BaseModel):
            return to_openai_response_format(value)
        return value

    def with_overrides(self, **kwargs: Any) -> "LLMConfig":
        """Create a new config with overrides applied."""
        data = self.model_dump()
        data.update(kwargs)
        return LLMConfig(**data)

    @classmethod
    def from_agent_config(cls, agent: "BaseAgent") -> "LLMConfig":
        """
        Create LLMConfig from agent configuration.

        Handles the legacy ``enable_json_mode`` configuration:
        - json_schema is a Pydantic model: normalized to a json_schema response_format
        - json_schema is a dict: wrapped as a json_schema response_format
        - json_schema is None: simple json_object mode

        ``enable_json_mode``/``json_schema`` are superseded by ``output_schema``,
        which validates the result rather than only requesting a shape; see
        _legacy_json_response_format.

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
            # getattr rather than agent.extra_body: this classmethod is called with
            # anything BaseAgent-shaped, including test doubles and subclasses that
            # predate the field. Missing => None => the SDK call is unchanged.
            extra_body=getattr(agent, "extra_body", None),
        )

        if agent.enable_json_mode:
            response_format = cls._legacy_json_response_format(agent)
            if response_format is None:
                config.json_mode = True
            else:
                config.response_format = response_format

        return config

    @staticmethod
    def _legacy_json_response_format(agent: "BaseAgent") -> dict[str, Any] | None:
        """Build a response_format from the deprecated ``json_schema`` field.

        Returns None when the agent named no schema, meaning bare json_object
        mode is all that was asked for.

        Assignment does not run field validators (LLMConfig does not enable
        validate_assignment), so the Pydantic-class normalization is applied
        explicitly here rather than being inherited from the field validator.
        """
        schema = agent.json_schema
        if schema is None:
            return None

        warnings.warn(
            "BaseAgent.enable_json_mode/json_schema are deprecated; use "
            "output_schema=<PydanticModel> instead. output_schema validates the "
            "result into AgentResponse.structured_output and retries a failed "
            "format, whereas json_schema only requests a shape.",
            DeprecationWarning,
            stacklevel=3,
        )

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return to_openai_response_format(schema)

        if not isinstance(schema, dict):
            return None

        # Accept both spellings seen in the wild: a bare JSON Schema, or an
        # already-assembled OpenAI json_schema block. Wrapping the latter again
        # would bury the real schema one level too deep.
        block = dict(schema) if "schema" in schema else {"schema": schema}
        block.setdefault("name", schema.get("title", "response"))
        # `strict` belongs INSIDE the json_schema block; at the top level OpenAI
        # rejects the request outright.
        block.setdefault("strict", getattr(agent, "json_strict", True))
        return {"type": "json_schema", "json_schema": block}
