"""Circuit breaker for protecting against cascading failures."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""

    def __init__(self, remaining_cooldown: float):
        self.remaining_cooldown = remaining_cooldown
        super().__init__(f"Circuit breaker is open. Retry after {remaining_cooldown:.1f}s")


# Set by hold_failures(): record_failure() appends the breaker here instead of counting.
_held_failures: ContextVar[list[CircuitBreaker] | None] = ContextVar(
    "continuum_circuit_breaker_held_failures", default=None
)


@contextmanager
def hold_failures() -> Iterator[list[CircuitBreaker]]:
    """Hold back the failures that breakers record inside this block.

    Yields a list of the breakers whose ``record_failure()`` was held. The
    caller counts them after the block (``breaker.record_failure()``) only if
    the failure should count, for example on the last attempt of a call that
    is retried, so one failing call counts once however many attempts it
    takes. A breaker that is half-open still counts its failure at once.
    """
    held: list[CircuitBreaker] = []
    token = _held_failures.set(held)
    try:
        yield held
    finally:
        _held_failures.reset(token)


class CircuitBreaker:
    """
    Simple circuit breaker that opens after N consecutive failures.

    Once open, all calls are rejected until the cooldown period elapses.
    After cooldown, a single call is allowed through (half-open state).
    If it succeeds, the breaker closes; if it fails, it re-opens.
    """

    def __init__(self, threshold: int = 5, cooldown: int = 60):
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                if time.monotonic() - self._opened_at >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    def check(self) -> None:
        """Check if a call is allowed. Raises CircuitBreakerOpen if not."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
                else:
                    remaining = self._cooldown - elapsed
                    raise CircuitBreakerOpen(max(0, remaining))
            elif self._state == CircuitState.OPEN:
                # opened_at is None but state is OPEN — should not happen; reset
                self._state = CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful call, resetting the breaker to closed."""
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call. Opens the breaker if threshold is reached.

        Inside ``hold_failures()`` the failure is held for the caller instead.
        """
        held = _held_failures.get()
        if held is not None and self.state != CircuitState.HALF_OPEN:
            held.append(self)
            return
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
