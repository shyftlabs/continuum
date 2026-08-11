"""
LLM Provider Support Module.

Provides a unified interface for multi-LLM provider support via direct provider SDKs.
Supports OpenAI, Google Gemini, Anthropic, and Azure OpenAI.
"""

from continuum.llm.callbacks import (
    LangfuseTraceContext,
    flush_langfuse,
    get_langfuse_callback,
    get_langfuse_metadata,
    get_trace_context,
    set_trace_context,
    setup_langfuse,
    shutdown_langfuse,
    trace_context,
)
from continuum.llm.client import LLMClient
from continuum.llm.config import LLMConfig
from continuum.llm.context_management import (
    CompressionResult,
    CompressionStrategy,
    ContextManagementConfig,
    ProgressiveContextManager,
    get_progressive_context_manager,
)
from continuum.llm.context_window import (
    ContextWindowManager,
    ModelLimits,
    TruncationResult,
    TruncationStrategy,
    get_context_window_manager,
)
from continuum.llm.dispatcher import PriorityDispatcher, TwoLevelDispatcher
from continuum.llm.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMFallbackExhaustedError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMStreamingError,
    LLMTimeoutError,
    LLMToolCallError,
)
from continuum.llm.types import (
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolCallDict,
    ToolCallInput,
    ToolDefinition,
    Usage,
)
from continuum.llm.utils import supports_tools_with_json_mode

__all__ = [
    # Client
    "LLMClient",
    "LLMConfig",
    # Dispatchers
    "PriorityDispatcher",
    "TwoLevelDispatcher",
    # Types
    "ChatMessage",
    "LLMResponse",
    "StreamChunk",
    "Usage",
    "FunctionCall",
    "FunctionDefinition",
    "ToolCall",
    "ToolCallDict",
    "ToolCallInput",
    "ToolDefinition",
    # Context Window
    "ContextWindowManager",
    "ModelLimits",
    "TruncationStrategy",
    "TruncationResult",
    "get_context_window_manager",
    # Context Management
    "CompressionStrategy",
    "CompressionResult",
    "ContextManagementConfig",
    "ProgressiveContextManager",
    "get_progressive_context_manager",
    # Callbacks & Tracing
    "setup_langfuse",
    "get_langfuse_callback",
    "get_langfuse_metadata",
    "trace_context",
    "get_trace_context",
    "set_trace_context",
    "flush_langfuse",
    "shutdown_langfuse",
    "LangfuseTraceContext",
    # Exceptions
    "LLMError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMContextLengthError",
    "LLMInvalidRequestError",
    "LLMServiceUnavailableError",
    "LLMFallbackExhaustedError",
    "LLMToolCallError",
    "LLMStreamingError",
    "LLMContentFilterError",
    # Utils
    "supports_tools_with_json_mode",
]
