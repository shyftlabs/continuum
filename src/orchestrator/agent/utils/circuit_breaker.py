"""Circuit breaker for protecting against cascading failures."""

from __future__ import annotations

import time
import threading
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""

    def __init__(self, remaining_cooldown: float):
        self.remaining_cooldown = remaining_cooldown
        super().__init__(
            f"Circuit breaker is open. Retry after {remaining_cooldown:.1f}s"
        )


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
        """Record a failed call. Opens the breaker if threshold is reached."""
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
