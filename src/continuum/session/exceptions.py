"""
Session exceptions.

Custom exceptions for session management operations.
"""


class SessionError(Exception):
    """Base exception for session operations."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.session_id = session_id
        self.original_error = original_error


class SessionNotEnabledError(SessionError):
    """Raised when session operations are attempted but sessions are disabled."""

    pass


class SessionConfigurationError(SessionError):
    """Raised when session configuration is invalid."""

    pass


class SessionConnectionError(SessionError):
    """Raised when connection to Redis fails."""

    pass


class SessionNotFoundError(SessionError):
    """Raised when a session is not found."""

    pass


class SessionNotCreatedError(SessionError):
    """Raised when a ``session_id`` is passed to ``runner.run()`` but no such
    session exists in the store.

    The runner never creates sessions itself — the caller owns session
    lifecycle. Create it first and pass the returned id::

        session_id = await runner.session_client.get_or_create_session(
            user_id="user-123", conversation_id="conv-456"
        )
        response = await runner.run(agent, msg, session_id=session_id, user_id="user-123")

    Only raised when strict mode is enabled (``require_session=True`` on the
    call, or ``SessionConfig.strict_sessions=True``). By default the runner
    warns instead and continues without history/persistence for that run.
    """

    pass


class SessionOwnershipError(SessionError):
    """Raised when a caller tries to use a session owned by another scope.

    A session ID is not authorization on its own. When a stored session is
    bound to a user or conversation, ``AgentRunner`` requires the caller's
    validated identifiers to match before it loads history, restores tool
    context, or writes anything back.
    """

    pass


class SessionMessageLimitError(SessionError):
    """Raised when session message limit is exceeded."""

    def __init__(
        self,
        message: str,
        session_id: str,
        current_count: int,
        max_messages: int,
        original_error: Exception | None = None,
    ):
        super().__init__(message, session_id, original_error)
        self.current_count = current_count
        self.max_messages = max_messages
