"""``extra_body`` must reach every provider that can carry it.

``LLMConfig.extra_body`` is how an application passes provider-specific options
the SDK does not model — a safety-settings block, a routing preference, a beta
flag. ``BaseAgent`` accepts it, ``LLMConfig.from_agent`` copies it across, and
``OpenAIProvider`` forwards it to the wire call.

``GeminiProvider`` speaks the same OpenAI-compatible wire protocol and dropped
it on the floor. No error, no warning: the request simply went without the
options, and the only way to find out was to notice the upstream behaving as if
you had never configured anything. A setting that silently does nothing is worse
than one that is rejected.

Asserted on the kwargs handed to the SDK rather than on a mocked call, because
that is the boundary the option has to cross — a test that mocks the client and
checks it was called would pass on a provider that built the kwargs and threw
them away.
"""

from __future__ import annotations

import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.providers.gemini_provider import GeminiProvider
from continuum.llm.providers.openai_provider import OpenAIProvider

SAFETY = {
    "safety_settings": [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]
}


def _kwargs(provider, **config_kwargs) -> dict:
    config = LLMConfig(model="gemini-2.5-flash", **config_kwargs)
    return provider._build_kwargs(config, tools=None, tool_choice=None)


@pytest.fixture
def gemini() -> GeminiProvider:
    return GeminiProvider(api_key="sk-test")


@pytest.fixture
def openai() -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test")


class TestGeminiForwardsExtraBody:
    def test_it_reaches_the_request(self, gemini):
        assert _kwargs(gemini, extra_body=SAFETY)["extra_body"] == SAFETY

    def test_absent_when_unset(self, gemini):
        """None must leave the request exactly as it was, so enabling this
        changes nothing for anyone not using it."""
        assert "extra_body" not in _kwargs(gemini)

    def test_an_empty_dict_is_still_forwarded(self, gemini):
        """`{}` is a caller saying 'no options', which is different from not
        configuring the field. `if config.extra_body:` would swallow it."""
        assert _kwargs(gemini, extra_body={})["extra_body"] == {}

    def test_the_caller_s_dict_is_not_shared_with_the_sdk(self, gemini):
        """A config object outlives one call, and an SDK that mutates what it is
        given would change every later request made from the same agent."""
        original = {"safety_settings": [{"threshold": "BLOCK_ONLY_HIGH"}]}
        sent = _kwargs(gemini, extra_body=original)["extra_body"]

        sent["safety_settings"][0]["threshold"] = "BLOCK_NONE"
        assert original["safety_settings"][0]["threshold"] == "BLOCK_ONLY_HIGH"


class TestTheTwoProvidersAgree:
    """Both speak the OpenAI wire protocol, so a config that works against one
    must not silently do nothing against the other."""

    def test_both_forward_it(self, gemini, openai):
        assert _kwargs(gemini, extra_body=SAFETY)["extra_body"] == SAFETY
        assert _kwargs(openai, extra_body=SAFETY)["extra_body"] == SAFETY

    def test_both_omit_it_when_unset(self, gemini, openai):
        assert "extra_body" not in _kwargs(gemini)
        assert "extra_body" not in _kwargs(openai)

    def test_neither_hands_the_sdk_the_config_s_own_dict(self, gemini, openai):
        """One LLMConfig serves many calls, and pydantic's copy is shallow -- a
        nested value is shared with whatever the caller passed in. An SDK that
        mutated what it was given would change every later request from that
        agent, and the caller's dict with it."""
        for provider in (gemini, openai):
            original = {"safety_settings": [{"threshold": "BLOCK_ONLY_HIGH"}]}
            config = LLMConfig(model="gemini-2.5-flash", extra_body=original)
            sent = provider._build_kwargs(config, tools=None, tool_choice=None)["extra_body"]

            sent["safety_settings"][0]["threshold"] = "BLOCK_NONE"
            assert original["safety_settings"][0]["threshold"] == "BLOCK_ONLY_HIGH"
            assert config.extra_body["safety_settings"][0]["threshold"] == "BLOCK_ONLY_HIGH"
