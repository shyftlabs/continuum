"""Unit tests for LLM config."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from orchestrator.llm.config import LLMConfig
import logging

logger = logging.getLogger(__name__)


class TestLLMConfig:
    def test_config_defaults(self):
        logger.info("LLMConfig: config defaults")
        c = LLMConfig()
        assert c.temperature == 0.7
        assert c.max_retries == 3
        assert c.json_mode is False

    def test_config_to_kwargs(self):
        logger.info("LLMConfig: config to provider kwargs")
        c = LLMConfig(model="gpt-4", temperature=0.5, max_tokens=100)
        kw = c.to_kwargs()
        assert kw["model"] == "gpt-4"
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 100

    def test_config_with_fallbacks(self):
        logger.info("LLMConfig: config with fallbacks")
        c = LLMConfig(fallback_models=["gpt-3.5-turbo"], enable_fallback=True)
        kw = c.to_kwargs()
        assert kw["fallbacks"] == ["gpt-3.5-turbo"]

    def test_config_json_mode(self):
        logger.info("LLMConfig: config json mode")
        c = LLMConfig(json_mode=True)
        kw = c.to_kwargs()
        assert kw["response_format"] == {"type": "json_object"}

    def test_config_response_format_dict(self):
        logger.info("LLMConfig: config response format dict")
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        c = LLMConfig(response_format=rf)
        kw = c.to_kwargs()
        assert kw["response_format"] == rf

    def test_config_response_format_pydantic(self):
        logger.info("LLMConfig: config response format pydantic")
        class MyModel(BaseModel):
            name: str
        c = LLMConfig(response_format=MyModel)
        kw = c.to_kwargs()
        assert kw["response_format"] is MyModel

    def test_config_with_overrides(self):
        logger.info("LLMConfig: config with overrides")
        c = LLMConfig(model="gpt-4")
        c2 = c.with_overrides(model="gpt-3.5-turbo", temperature=0.1)
        assert c2.model == "gpt-3.5-turbo"
        assert c2.temperature == 0.1
        assert c.model == "gpt-4"

    def test_config_optional_params(self):
        logger.info("LLMConfig: config optional params")
        c = LLMConfig(top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3, stop=["END"], seed=42, user="u1")
        kw = c.to_kwargs()
        assert kw["top_p"] == 0.9
        assert kw["frequency_penalty"] == 0.5
        assert kw["stop"] == ["END"]
        assert kw["seed"] == 42
        assert kw["user"] == "u1"

    def test_config_custom_provider(self):
        logger.info("LLMConfig: config custom provider")
        c = LLMConfig(api_base="http://localhost", api_key="key", api_version="v1", custom_llm_provider="azure")
        kw = c.to_kwargs()
        assert kw["api_base"] == "http://localhost"
        assert kw["api_key"] == "key"
        assert kw["custom_llm_provider"] == "azure"

    def test_config_cache_settings(self):
        logger.info("LLMConfig: config cache settings")
        c = LLMConfig(cache=True, cache_ttl=3600)
        kw = c.to_kwargs()
        assert kw["cache"]["type"] == "local"
        assert kw["cache"]["ttl"] == 3600

    def test_config_from_agent_config(self):
        logger.info("LLMConfig: config from agent config")
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.3
        agent.max_tokens = 200
        agent.enable_json_mode = False
        agent.json_schema = None
        c = LLMConfig.from_agent_config(agent)
        assert c.model == "gpt-4"
        assert c.temperature == 0.3

    def test_config_from_agent_config_json_mode(self):
        logger.info("LLMConfig: config from agent config json mode")
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.7
        agent.max_tokens = 4096
        agent.enable_json_mode = True
        agent.json_schema = None
        c = LLMConfig.from_agent_config(agent)
        assert c.json_mode is True

    def test_config_from_agent_config_pydantic_schema(self):
        logger.info("LLMConfig: config from agent config pydantic schema")
        class MyModel(BaseModel):
            name: str
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.7
        agent.max_tokens = 4096
        agent.enable_json_mode = True
        agent.json_schema = MyModel
        c = LLMConfig.from_agent_config(agent)
        assert c.response_format is MyModel

    def test_config_from_agent_config_dict_schema(self):
        logger.info("LLMConfig: config from agent config dict schema")
        agent = MagicMock()
        agent.model = "gpt-4"
        agent.temperature = 0.7
        agent.max_tokens = 4096
        agent.enable_json_mode = True
        agent.json_schema = {"name": "test", "schema": {}}
        agent.json_strict = True
        c = LLMConfig.from_agent_config(agent)
        assert c.response_format["type"] == "json_schema"

    def test_config_metadata(self):
        logger.info("LLMConfig: config metadata")
        c = LLMConfig(metadata={"task": "test"})
        kw = c.to_kwargs()
        assert kw["metadata"]["task"] == "test"
