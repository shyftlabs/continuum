"""Unit tests for LLM types."""

from unittest.mock import MagicMock

import pytest

from orchestrator.llm.types import (
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
)
import logging

logger = logging.getLogger(__name__)


class TestFunctionCall:
    def test_to_dict(self):
        logger.info("FunctionCall: to dict")
        fc = FunctionCall(name="get_weather", arguments='{"city":"NYC"}')
        d = fc.to_dict()
        assert d == {"name": "get_weather", "arguments": '{"city":"NYC"}'}


class TestToolCall:
    def test_to_dict(self):
        logger.info("ToolCall: to dict")
        tc = ToolCall(id="tc1", function=FunctionCall(name="fn", arguments="{}"))
        d = tc.to_dict()
        assert d["id"] == "tc1"
        assert d["type"] == "function"
        assert d["function"]["name"] == "fn"


class TestChatMessage:
    def test_to_dict_minimal(self):
        logger.info("ChatMessage: to dict minimal")
        m = ChatMessage(role="user", content="hello")
        d = m.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_to_dict_with_tool_calls(self):
        logger.info("ChatMessage: to dict with tool calls")
        tc = ToolCall(id="tc1", function=FunctionCall(name="fn", arguments="{}"))
        m = ChatMessage(role="assistant", content="ok", tool_calls=[tc])
        d = m.to_dict()
        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1

    def test_to_dict_with_tool_call_id(self):
        logger.info("ChatMessage: to dict with tool call id")
        m = ChatMessage(role="tool", content="result", tool_call_id="tc1")
        d = m.to_dict()
        assert d["tool_call_id"] == "tc1"

    def test_to_dict_with_name(self):
        logger.info("ChatMessage: to dict with name")
        m = ChatMessage(role="function", content="result", name="my_func")
        d = m.to_dict()
        assert d["name"] == "my_func"

    def test_to_dict_with_function_call(self):
        logger.info("ChatMessage: to dict with function call")
        fc = FunctionCall(name="fn", arguments="{}")
        m = ChatMessage(role="assistant", function_call=fc)
        d = m.to_dict()
        assert d["function_call"]["name"] == "fn"


class TestFunctionDefinition:
    def test_to_dict(self):
        logger.info("FunctionDefinition: to dict")
        fd = FunctionDefinition(name="fn", description="desc", parameters={"type": "object"})
        d = fd.to_dict()
        assert d["name"] == "fn"
        assert d["description"] == "desc"

    def test_to_dict_minimal(self):
        logger.info("FunctionDefinition: to dict minimal")
        fd = FunctionDefinition(name="fn")
        d = fd.to_dict()
        assert d == {"name": "fn"}


class TestToolDefinition:
    def test_to_dict(self):
        logger.info("ToolDefinition: to dict")
        td = ToolDefinition(function=FunctionDefinition(name="fn", description="d"))
        d = td.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "fn"


class TestUsage:
    def test_usage_model(self):
        logger.info("Usage: usage model")
        u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert u.prompt_tokens == 10
        assert u.total_tokens == 15


class TestLLMResponse:
    def test_from_openai_response(self):
        logger.info("LLMResponse: from openai response")
        mock_resp = MagicMock()
        mock_resp.id = "resp-1"
        mock_resp.model = "gpt-4"
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_choice.message.role = "assistant"
        mock_choice.message.tool_calls = None
        mock_choice.message.function_call = None
        mock_choice.finish_reason = "stop"
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}

        resp = LLMResponse.from_openai_response(mock_resp)
        assert resp.content == "Hello"
        assert resp.model == "gpt-4"
        assert resp.usage.total_tokens == 15

    def test_from_openai_response_with_tool_calls(self):
        logger.info("LLMResponse: from openai response with tool calls")
        mock_resp = MagicMock()
        mock_resp.id = "resp-2"
        mock_resp.model = "gpt-4"
        mock_tc = MagicMock()
        mock_tc.id = "tc1"
        mock_tc.type = "function"
        mock_tc.function.name = "fn"
        mock_tc.function.arguments = "{}"
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.role = "assistant"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.message.function_call = None
        mock_choice.finish_reason = "tool_calls"
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_resp.model_dump.return_value = {}

        resp = LLMResponse.from_openai_response(mock_resp)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "fn"

    def test_from_openai_response_no_choices(self):
        logger.info("LLMResponse: from openai response no choices")
        mock_resp = MagicMock()
        mock_resp.id = "resp-3"
        mock_resp.model = "gpt-4"
        mock_resp.choices = []
        mock_resp.usage = None
        mock_resp.model_dump.return_value = {}

        resp = LLMResponse.from_openai_response(mock_resp)
        assert resp.content is None


class TestStreamChunk:
    def test_from_openai_chunk(self):
        logger.info("StreamChunk: from openai chunk")
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-1"
        mock_chunk.model = "gpt-4"
        mock_delta = MagicMock()
        mock_delta.content = "Hello"
        mock_delta.role = "assistant"
        mock_delta.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        mock_chunk.choices = [mock_choice]

        chunk = StreamChunk.from_openai_chunk(mock_chunk)
        assert chunk.content == "Hello"
        assert chunk.is_finished is False

    def test_from_openai_chunk_finished(self):
        logger.info("StreamChunk: from openai chunk finished")
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-2"
        mock_chunk.model = "gpt-4"
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.role = None
        mock_delta.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"
        mock_chunk.choices = [mock_choice]

        chunk = StreamChunk.from_openai_chunk(mock_chunk)
        assert chunk.is_finished is True
        assert chunk.finish_reason == "stop"
