"""
Type definitions for the LLM module.

Provides Pydantic models for structured data handling with LLM responses.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Type alias for tool call input (can be ToolCall object or dict)
# Used throughout the SDK for flexible tool call handling
ToolCallDict = dict[str, Any]


# NOTE: FunctionCall and ToolCall must be defined BEFORE ChatMessage
# because ChatMessage references them. This ensures Pydantic can properly
# deserialize nested models when loading from dict/JSON.


class FunctionCall(BaseModel):
    """Represents a function call in a message."""

    name: str
    arguments: str  # JSON string of arguments

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"name": self.name, "arguments": self.arguments}


class ToolCall(BaseModel):
    """Represents a tool call in a message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }


# Union type for flexible tool call handling
# Accepts either a ToolCall Pydantic model or a raw dictionary
ToolCallInput = ToolCall | ToolCallDict


class ChatMessage(BaseModel):
    """Represents a chat message in a conversation."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: FunctionCall | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for LLM provider APIs."""
        result: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.function_call is not None:
            result["function_call"] = self.function_call.to_dict()
        return result


class FunctionDefinition(BaseModel):
    """Defines a function that can be called by the model."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = self.parameters
        return result


class ToolDefinition(BaseModel):
    """Defines a tool that can be used by the model."""

    type: Literal["function"] = "function"
    function: FunctionDefinition

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"type": self.type, "function": self.function.to_dict()}


class Usage(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Represents a complete response from an LLM."""

    id: str | None = None
    model: str
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None

    @classmethod
    def from_openai_response(cls, response: Any) -> "LLMResponse":
        """Create LLMResponse from an OpenAI SDK response (also used for Gemini compat)."""
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None

        tool_calls = None
        if message and message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
            ]

        function_call = None
        if message and hasattr(message, "function_call") and message.function_call:
            function_call = FunctionCall(
                name=message.function_call.name,
                arguments=message.function_call.arguments,
            )

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return cls(
            id=response.id,
            model=response.model,
            content=message.content if message else None,
            role=message.role if message else "assistant",
            tool_calls=tool_calls,
            function_call=function_call,
            usage=usage,
            finish_reason=choice.finish_reason if choice else None,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    @classmethod
    def from_anthropic_response(cls, response: Any, model: str) -> "LLMResponse":
        """Create LLMResponse from an Anthropic SDK response."""
        import json as _json

        text_content: str | None = None
        tool_calls: list[ToolCall] | None = None

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=_json.dumps(block.input),
                        ),
                    )
                )

        usage = None
        if response.usage:
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(response.stop_reason or "", response.stop_reason)

        return cls(
            id=response.id,
            model=model,
            content=text_content,
            role="assistant",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )


class StreamChunk(BaseModel):
    """Represents a single chunk from a streaming response."""

    id: str | None = None
    model: str | None = None
    content: str | None = None
    role: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    is_finished: bool = False

    @classmethod
    def from_openai_chunk(cls, chunk: Any) -> "StreamChunk":
        """Create StreamChunk from an OpenAI SDK streaming chunk (also used for Gemini compat)."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None

        tool_calls = None
        if delta and delta.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id or "",
                    type=tc.type or "function",
                    function=FunctionCall(
                        name=tc.function.name or "",
                        arguments=tc.function.arguments or "",
                    ),
                )
                for tc in delta.tool_calls
                if tc.function
            ]

        return cls(
            id=chunk.id,
            model=chunk.model,
            content=delta.content if delta else None,
            role=delta.role if delta else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=choice.finish_reason if choice else None,
            is_finished=choice.finish_reason is not None if choice else False,
        )

    @classmethod
    def from_anthropic_response(cls, response: Any, model: str) -> "StreamChunk":
        """Create a final StreamChunk from a completed Anthropic message (end of stream)."""
        import json as _json

        tool_calls: list[ToolCall] | None = None
        for block in response.content:
            if block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=_json.dumps(block.input),
                        ),
                    )
                )

        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        finish_reason = finish_reason_map.get(response.stop_reason or "", response.stop_reason)

        return cls(
            model=model,
            content=None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            is_finished=True,
        )
