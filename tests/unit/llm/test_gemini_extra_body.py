"""Provider options reach the wire without service-specific monkey patches."""
from continuum.llm.config import LLMConfig
from continuum.llm.providers.gemini_provider import GeminiProvider


def test_gemini_default_request_omits_extra_body():
    provider = object.__new__(GeminiProvider)
    result = provider._build_kwargs(LLMConfig(model="gemini/gemini-2.5-flash"), None, None)
    assert "extra_body" not in result


def test_gemini_forwards_explicit_options_without_changing_other_parameters():
    provider = object.__new__(GeminiProvider)
    cfg = LLMConfig(model="gemini/gemini-2.5-pro", max_tokens=2400, extra_body={"reasoning_effort": "low"})
    result = provider._build_kwargs(cfg, [{"type": "function"}], "auto")
    assert result["extra_body"] == {"reasoning_effort": "low"}
    assert result["model"] == "gemini-2.5-pro"
    assert result["max_tokens"] == 2400 and result["tool_choice"] == "auto"
    assert result["tools"] == [{"type": "function"}]


def test_gemini_empty_options_remain_an_explicit_empty_object():
    provider = object.__new__(GeminiProvider)
    result = provider._build_kwargs(LLMConfig(model="gemini-2.5-flash", extra_body={}), None, None)
    assert result["extra_body"] == {}
