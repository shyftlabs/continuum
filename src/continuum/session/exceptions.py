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


class SessionOwnershipError(SessionError):
    """Raised when a caller touches a session owned by a different principal.

    A session id names storage; it is not authorization on its own. When a
    stored session records an owner, the ``SessionClient`` compares it against
    the principal bound for the call (see
    :func:`continuum.session.principal.bind_principal`) before reading history,
    reading metadata, or writing anything back.

    Only raised when ``SessionConfig.session_ownership='enforce'``. Under
    ``'open'`` (the default) and ``'audit'`` the same condition is reported to
    logs and metrics and the call proceeds, so a deployment can measure the
    impact before enforcing.

    The message deliberately never names the stored owner — the caller has just
    failed to prove they are that person, so disclosing it would make the
    refusal an identity oracle.
    """

    pass
