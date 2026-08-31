"""
Session module - Short-term memory and conversation history management.

This module provides session management capabilities for agents using provider-based
storage (Redis, DynamoDB, etc.), with full integration with mem0 (long-term memory)
and Langfuse tracing.

Features:
- Session-based conversation history (short-term memory via providers)
- Provider abstraction for easy extensibility (Redis, DynamoDB, PostgreSQL, etc.)
- Integration with mem0 for long-term memory
- Standardized ID alignment (session_id maps to run_id in mem0)
- Explicit session creation via get_or_create_session() — the caller owns
  session lifecycle; runner.run() loads/saves but never creates a session
- Configurable TTL and message limits
- Full observability with Langfuse (automatic via @observe decorator)
- Complete conversation history (user, assistant, tool calls, tool results)
"""

from continuum.session.base import BaseSessionProvider
from continuum.session.client import (
    SessionClient,
    get_global_session_client,
    initialize_global_session_client,
)
from continuum.session.config import SessionConfig
from continuum.session.exceptions import (
    SessionConfigurationError,
    SessionConnectionError,
    SessionError,
    SessionMessageLimitError,
    SessionNotCreatedError,
    SessionNotEnabledError,
    SessionNotFoundError,
    SessionOwnershipError,
)
from continuum.session.providers import (
    create_provider,
    get_provider_class,
    list_providers,
    register_provider,
)
from continuum.session.types import (
    Session,
    SessionMessage,
    SessionMetadata,
    generate_session_id,
)

__all__ = [
    # Client
    "SessionClient",
    "get_global_session_client",
    "initialize_global_session_client",
    # Base Provider
    "BaseSessionProvider",
    # Provider Registry
    "create_provider",
    "get_provider_class",
    "list_providers",
    "register_provider",
    # Config
    "SessionConfig",
    # Types
    "Session",
    "SessionMetadata",
    "SessionMessage",
    "generate_session_id",
    # Exceptions
    "SessionError",
    "SessionConfigurationError",
    "SessionNotEnabledError",
    "SessionConnectionError",
    "SessionNotFoundError",
    "SessionNotCreatedError",
    "SessionOwnershipError",
    "SessionMessageLimitError",
]
