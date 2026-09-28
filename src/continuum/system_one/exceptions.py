"""Errors raised by the System One layer.

Every failure a backend can have surfaces as a :class:`SystemOneError` subclass,
so a seam can hold one rule -- "the classifier did not answer" -- and apply its
own fallback, instead of guessing which vendor exceptions mean what.
"""

from __future__ import annotations

from typing import Any

from continuum.exceptions import (
    ConfigurationError,
    ErrorCategory,
    OrchestratorError,
    PolicyDeniedError,
    ProviderError,
)


class SystemOneError(OrchestratorError):
    """Base class: the System One layer did not produce an answer."""

    default_message = "System One classification failed"
    default_error_code = "SYSTEM_ONE_ERROR"


class SystemOneNotConfiguredError(ConfigurationError, SystemOneError):
    """A seam was switched on, or a spec named, with no backend to answer it.

    Raised rather than falling back to the old path: a switch that silently does
    nothing reads as protection that is not there.
    """

    default_message = "No System One backend is configured"
    default_error_code = "SYSTEM_ONE_NOT_CONFIGURED"


class SystemOneDisabledError(SystemOneError):
    """The kill switch (``SYSTEM_ONE_DISABLED``) is on."""

    default_message = "System One classification is disabled (SYSTEM_ONE_DISABLED)"
    default_error_code = "SYSTEM_ONE_DISABLED"


class SystemOneCapabilityError(SystemOneError):
    """A question kind the backend cannot answer, directly or by filling in."""

    default_message = "The System One backend cannot answer this question kind"
    default_error_code = "SYSTEM_ONE_CAPABILITY"
    default_category = ErrorCategory.VALIDATION


class SystemOneResponseError(SystemOneError):
    """A backend answered with something that cannot be trusted as a distribution."""

    default_message = "The System One backend returned an invalid answer"
    default_error_code = "SYSTEM_ONE_INVALID_RESPONSE"
    default_category = ErrorCategory.PROVIDER


class SystemOneBackendError(ProviderError, SystemOneError):
    """The backend failed: an HTTP error, a connection failure, a model error.

    ``status`` and ``retry_after`` are kept for programmatic handling. Response
    bodies are never kept: they can echo the classified state.
    """

    default_message = "The System One backend failed"
    default_error_code = "SYSTEM_ONE_BACKEND_ERROR"

    def __init__(
        self,
        message: str | None = None,
        *,
        backend: str | None = None,
        status: int | None = None,
        retry_after: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, provider=backend, **kwargs)
        self.status = status
        self.retry_after = retry_after
        if status is not None:
            self.context["status"] = status


class SystemOneTimeoutError(SystemOneBackendError):
    """The backend did not answer within its timeout."""

    default_message = "The System One backend timed out"
    default_error_code = "SYSTEM_ONE_TIMEOUT"
    default_category = ErrorCategory.TIMEOUT


class SystemOneAccessDeniedError(SystemOneError, PolicyDeniedError):
    """The run's data-label policy denies sending its state to this backend.

    A governance outcome, not a failure (see :class:`PolicyDeniedError`): a run
    tainted with ``phi`` denied a remote backend is the policy working.
    """

    default_message = "Access denied: System One backend is blocked by policy"
    default_error_code = "SYSTEM_ONE_ACCESS_DENIED"

    def __init__(
        self,
        resource: str,
        policy_name: str | None = None,
        denial_message: str = "",
        **kwargs: Any,
    ):
        message = f"Access denied: '{resource}' is blocked by policy"
        if policy_name:
            message += f" '{policy_name}'"
        super().__init__(message, **kwargs)
        self.context["resource"] = resource
        if policy_name:
            self.context["policy_name"] = policy_name
        if denial_message:
            self.context["denial_message"] = denial_message
