"""
LLM Provider Support Module.

Provides a unified interface for multi-LLM provider support via direct provider SDKs.
Supports OpenAI, Google Gemini, Anthropic, and Azure OpenAI.
"""

from orchestrator.llm.callbacks import (
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
from orchestrator.llm.client import LLMClient
from orchestrator.llm.config import LLMConfig
from orchestrator.llm.context_management import (
    CompressionResult,
    CompressionStrategy,
    ContextManagementConfig,
    ProgressiveContextManager,
    get_progressive_context_manager,
)
from orchestrator.llm.context_window import (
    ContextWindowManager,
    ModelLimits,
    TruncationResult,
    TruncationStrategy,
    get_context_window_manager,
)
from orchestrator.llm.exceptions import (
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
from orchestrator.llm.types import (
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
from orchestrator.llm.utils import (
    check_json_schema_support,
    check_response_format_support,
    supports_tools_with_json_mode,
    validate_json_schema_config,
)

__all__ = [
    # Client
    "LLMClient",
    "LLMConfig",
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
    "check_response_format_support",
    "check_json_schema_support",
    "supports_tools_with_json_mode",
    "validate_json_schema_config",
]
