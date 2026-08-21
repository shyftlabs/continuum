"""Unit tests for LLM config."""

import logging
import warnings
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from continuum.llm.config import LLMConfig

logger = logging.getLogger(__name__)


def _legacy_json_agent(json_schema):
    """An agent using the deprecated enable_json_mode/json_schema pair."""
    agent = MagicMock()
    agent.model = "gpt-4"
    agent.temperature = 0.7
    agent.max_tokens = 4096
    agent.gateway_mode = None
    agent.extra_body = None
    agent.enable_json_mode = True
    agent.json_schema = json_schema
    return agent


class TestLLMConfig:
    def test_config_defaults(self):
        logger.info("LLMConfig: config defaults")
        c = LLMConfig()
        assert c.temperature == 0.7
        assert c.max_retries == 3
        assert c.json_mode is False

    def test_config_temperature_none_is_preserved(self):
        """None means "omit the parameter"; each provider's _build_kwargs drops
        it. Storing 0.0 or a default here would send it after all."""
        logger.info("LLMConfig: temperature=None is preserved as None")
        c = LLMConfig(model="gpt-4", temperature=None)
        assert c.temperature is None

    def test_config_with_fallbacks(self):
        logger.info("LLMConfig: config with fallbacks")
        c = LLMConfig(fallback_models=["gpt-3.5-turbo"], enable_fallback=True)
        assert c.fallback_models == ["gpt-3.5-turbo"]
        assert c.enable_fallback is True

    def test_config_json_mode(self):
        logger.info("LLMConfig: config json mode")
        c = LLMConfig(json_mode=True)
        assert c.json_mode is True
        assert c.response_format is None

    def test_config_response_format_dict(self):
        logger.info("LLMConfig: config response format dict")
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        c = LLMConfig(response_format=rf)
        assert c.response_format == rf

    def test_config_response_format_pydantic_is_normalized(self):
        """A model class used to be stored as-is and handed to the OpenAI SDK,
        which rejects it outright ("use chat.completions.parse() instead") — an
        LLMError on OpenAI, Gemini and the gateway alike."""
        logger.info("LLMConfig: pydantic response format is normalized to a dict")

        class MyModel(BaseModel):
            name: str

        c = LLMConfig(response_format=MyModel)
        assert c.response_format == {
            "type": "json_schema",
            "json_schema": {"name": "MyModel", "schema": MyModel.model_json_schema()},
        }

    def test_config_with_overrides(self):
        logger.info("LLMConfig: config with overrides")
        c = LLMConfig(model="gpt-4")
        c2 = c.with_overrides(model="gpt-3.5-turbo", temperature=0.1)
        assert c2.model == "gpt-3.5-turbo"
        assert c2.temperature == 0.1
        assert c.model == "gpt-4"

    def test_config_optional_params(self):
        logger.info("LLMConfig: config optional params")
        c = LLMConfig(
            top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3, stop=["END"], seed=42, user="u1"
        )
        assert c.top_p == 0.9
        assert c.frequency_penalty == 0.5
        assert c.stop == ["END"]
        assert c.seed == 42
        assert c.user == "u1"

    def test_config_custom_provider(self):
        logger.info("LLMConfig: config custom provider")
        c = LLMConfig(
            api_base="http://localhost",
            api_key="key",
            api_version="v1",
            custom_llm_provider="azure",
        )
        assert c.api_base == "http://localhost"
        assert c.api_key == "key"
        assert c.custom_llm_provider == "azure"

    def test_config_cache_settings(self):
        logger.info("LLMConfig: config cache settings")
        c = LLMConfig(cache=True, cache_ttl=3600)
        assert c.cache is True
        assert c.cache_ttl == 3600

    def test_config_from_agent_config(self):
        logger.info("LLMConfig: config from agent config")
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.3
        agent.max_tokens = 200
        agent.gateway_mode = None
        agent.extra_body = None
        agent.enable_json_mode = False
        agent.json_schema = None
        c = LLMConfig.from_agent_config(agent)
        assert c.model == "gpt-4"
        assert c.temperature == 0.3
        assert c.max_tokens == 200
        assert c.gateway_router_mode is None
        assert c.json_mode is False
        assert c.response_format is None

    def test_config_from_agent_config_json_mode(self):
        logger.info("LLMConfig: config from agent config json mode")
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.7
        agent.max_tokens = 4096
        agent.gateway_mode = None
        agent.extra_body = None
        agent.enable_json_mode = True
        agent.json_schema = None
        c = LLMConfig.from_agent_config(agent)
        assert c.json_mode is True
        assert c.response_format is None

    def test_config_from_agent_config_pydantic_schema(self):
        """The class must be converted, not stored — see
        test_config_response_format_pydantic_is_normalized."""
        logger.info("LLMConfig: config from agent config pydantic schema")

        class MyModel(BaseModel):
            name: str

        agent = _legacy_json_agent(MyModel)
        with pytest.deprecated_call():
            c = LLMConfig.from_agent_config(agent)
        assert c.response_format == {
            "type": "json_schema",
            "json_schema": {"name": "MyModel", "schema": MyModel.model_json_schema()},
        }
        assert c.json_mode is False

    def test_config_from_agent_config_dict_schema(self):
        """`strict` belongs inside the json_schema block; at the top level
        OpenAI rejects the whole request."""
        logger.info("LLMConfig: config from agent config dict schema")
        agent = _legacy_json_agent({"name": "test", "schema": {"type": "object"}})
        agent.json_strict = True
        with pytest.deprecated_call():
            c = LLMConfig.from_agent_config(agent)
        assert c.response_format == {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}, "strict": True},
        }

    def test_config_from_agent_config_bare_json_schema_is_wrapped(self):
        """A raw JSON Schema (no name/schema envelope) is the other spelling
        callers use; it has to be wrapped rather than passed through."""
        logger.info("LLMConfig: bare json schema is wrapped")
        agent = _legacy_json_agent({"title": "Review", "type": "object", "properties": {}})
        agent.json_strict = False
        with pytest.deprecated_call():
            c = LLMConfig.from_agent_config(agent)
        block = c.response_format["json_schema"]
        assert block["name"] == "Review"
        assert block["schema"]["type"] == "object"
        assert block["strict"] is False

    def test_legacy_json_mode_without_schema_does_not_warn(self):
        """Bare json_object mode has no output_schema equivalent to point at."""
        logger.info("LLMConfig: bare json mode is not deprecated")
        agent = _legacy_json_agent(None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            c = LLMConfig.from_agent_config(agent)
        assert c.json_mode is True

    def test_config_metadata(self):
        logger.info("LLMConfig: config metadata")
        c = LLMConfig(metadata={"task": "test"})
        assert c.metadata["task"] == "test"
