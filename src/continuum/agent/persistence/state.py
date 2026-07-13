"""
Run State Manager - Redis backend for agent run state.

Manages run state persistence for pause/resume and recovery.
Uses the same Redis instance as session management.

NOTE: All Redis operations are wrapped in asyncio.to_thread() to prevent
blocking the event loop when using synchronous redis-py client.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from continuum.agent.exceptions import (
    RunStateNotFoundError,
    RunStatePersistenceError,
)
from continuum.agent.types import RunState, RunStatus
from continuum.config import settings
from continuum.logging import get_logger

logger = get_logger(__name__)

# Global state manager
_global_lock = threading.Lock()
_global_state_manager: RunStateManager | None = None
_initialized = False


class RunStateManager:
    """
    Manages agent run state persistence in Redis.

    Uses the same Redis instance as the session provider (redis-sdk).
    State is stored with a TTL for automatic cleanup.

    Example:
        ```python
        from continuum.agent.persistence import RunStateManager

        manager = RunStateManager()

        # Save state
        await manager.save(run_state)

        # Load state
        state = await manager.load(run_id)

        # Update status
        await manager.update_status(run_id, RunStatus.COMPLETED)
        ```
    """

    KEY_PREFIX = "orchestrator:run_state"

    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_password: str | None = None,
        redis_db: int | None = None,
        state_ttl: int = 3600 * 24,  # 24 hours default
        auto_initialize: bool = True,
    ):
        """
        Initialize the Run State Manager.

        Args:
            redis_host: Redis host (defaults to session Redis)
            redis_port: Redis port (defaults to session Redis)
            redis_password: Redis password
            redis_db: Redis database number
            state_ttl: TTL for state entries in seconds
            auto_initialize: Whether to initialize immediately
        """
        self._redis_host = redis_host or settings.session_redis_host
        self._redis_port = redis_port or settings.session_redis_port
        self._redis_password = redis_password or settings.session_redis_password
        self._redis_db = redis_db if redis_db is not None else settings.session_redis_db
        self._state_ttl = state_ttl

        self._redis: Any = None
        self._initialized = False
        self._lock = threading.Lock()

        if auto_initialize:
            self.initialize()

    @property
    def is_enabled(self) -> bool:
        """Check if state manager is enabled and initialized."""
        return self._initialized and self._redis is not None

    def initialize(self) -> bool:
        """
        Initialize Redis connection.

        Thread-safe initialization that only runs once.

        Returns:
            True if initialization was successful.
        """
        with self._lock:
            if self._initialized:
                return True

            try:
                import redis

                self._redis = redis.Redis(
                    host=self._redis_host,
                    port=self._redis_port,
                    password=self._redis_password,
                    db=self._redis_db,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )

                # Test connection
                self._redis.ping()

                self._initialized = True
                logger.info(
                    "Run state manager initialized",
                    extra={
                        "redis_host": self._redis_host,
                        "redis_port": self._redis_port,
                    },
                )
                return True

            except ImportError:
                logger.warning("Redis not installed, state persistence disabled")
                return False
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis for state persistence: {e}",
                    extra={"error": str(e)},
                )
                return False

    def _get_key(self, run_id: str) -> str:
        """Get Redis key for a run state."""
        return f"{self.KEY_PREFIX}:{run_id}"

    def _get_index_key(self, index_type: str, value: str) -> str:
        """Get Redis key for an index."""
        return f"{self.KEY_PREFIX}:idx:{index_type}:{value}"

    async def save(self, state: RunState) -> None:
        """
        Save run state to Redis.

        Args:
            state: RunState to save

        Raises:
            RunStatePersistenceError: If save fails
        """
        if not self.is_enabled:
            return

        try:
            state.update_timestamp()
            key = self._get_key(state.run_id)

            # Serialize state
            state_json = json.dumps(state.to_dict())

            # Define sync operation for thread execution
            def _sync_save() -> None:
                # Save with TTL
                self._redis.set(key, state_json, ex=self._state_ttl)

                # Update indexes
                if state.session_id:
                    idx_key = self._get_index_key("session", state.session_id)
                    self._redis.sadd(idx_key, state.run_id)
                    self._redis.expire(idx_key, self._state_ttl)

                if state.user_id:
                    idx_key = self._get_index_key("user", state.user_id)
                    self._redis.sadd(idx_key, state.run_id)
                    self._redis.expire(idx_key, self._state_ttl)

            # Run in thread to avoid blocking event loop
            await asyncio.to_thread(_sync_save)

            logger.debug(
                f"Saved run state: {state.run_id}",
                extra={
                    "run_id": state.run_id,
                    "status": state.status.value,
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to save run state: {e}",
                extra={
                    "run_id": state.run_id,
                    "error": str(e),
                },
            )
            raise RunStatePersistenceError(
                f"Failed to save run state: {e}",
                run_id=state.run_id,
                original_error=e,
            ) from e

    async def load(self, run_id: str) -> RunState | None:
        """
        Load run state from Redis.

        Args:
            run_id: Run ID to load

        Returns:
            RunState if found, None otherwise
        """
        if not self.is_enabled:
            return None

        try:
            key = self._get_key(run_id)

            # Define sync operation for thread execution
            def _sync_load() -> str | None:
                return self._redis.get(key)

            # Run in thread to avoid blocking event loop
            state_json = await asyncio.to_thread(_sync_load)

            if not state_json:
                return None

            state_dict = json.loads(state_json)
            return RunState.from_dict(state_dict)

        except Exception as e:
            logger.error(
                f"Failed to load run state: {e}",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                },
            )
            return None

    async def load_or_raise(self, run_id: str) -> RunState:
        """
        Load run state or raise if not found.

        Args:
            run_id: Run ID to load

        Returns:
            RunState

        Raises:
            RunStateNotFoundError: If state not found
        """
        state = await self.load(run_id)
        if state is None:
            raise RunStateNotFoundError(run_id)
        return state

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None = None,
    ) -> RunState | None:
        """
        Update run status.

        Args:
            run_id: Run ID
            status: New status
            error: Optional error message

        Returns:
            Updated RunState or None if not found
        """
        state = await self.load(run_id)
        if state is None:
            return None

        state.status = status
        if error:
            state.metadata["error"] = error

        await self.save(state)
        return state

    async def delete(self, run_id: str) -> bool:
        """
        Delete run state.

        Args:
            run_id: Run ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.is_enabled:
            return False

        try:
            key = self._get_key(run_id)

            # Define sync operation for thread execution
            def _sync_delete() -> int:
                return self._redis.delete(key)

            # Run in thread to avoid blocking event loop
            result = await asyncio.to_thread(_sync_delete)
            return result > 0

        except Exception as e:
            logger.error(
                f"Failed to delete run state: {e}",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                },
            )
            return False

    async def list_by_session(self, session_id: str) -> list[str]:
        """
        List run IDs for a session.

        Args:
            session_id: Session ID

        Returns:
            List of run IDs
        """
        if not self.is_enabled:
            return []

        try:
            idx_key = self._get_index_key("session", session_id)

            # Define sync operation for thread execution
            def _sync_list() -> set:
                return self._redis.smembers(idx_key)

            # Run in thread to avoid blocking event loop
            members = await asyncio.to_thread(_sync_list)
            return list(members)
        except Exception:
            return []

    async def list_by_user(self, user_id: str) -> list[str]:
        """
        List run IDs for a user.

        Args:
            user_id: User ID

        Returns:
            List of run IDs
        """
        if not self.is_enabled:
            return []

        try:
            idx_key = self._get_index_key("user", user_id)

            # Define sync operation for thread execution
            def _sync_list() -> set:
                return self._redis.smembers(idx_key)

            # Run in thread to avoid blocking event loop
            members = await asyncio.to_thread(_sync_list)
            return list(members)
        except Exception:
            return []

    async def list_active(self) -> list[RunState]:
        """
        List all active (running) run states.

        Returns:
            List of active RunStates
        """
        if not self.is_enabled:
            return []

        try:
            # Use pattern matching to find all run states
            pattern = f"{self.KEY_PREFIX}:run_*"

            # Define sync operation for thread execution
            def _sync_list_active() -> list[RunState]:
                keys = self._redis.keys(pattern)
                active_states = []
                for key in keys:
                    state_json = self._redis.get(key)
                    if state_json:
                        state = RunState.from_dict(json.loads(state_json))
                        if state.status == RunStatus.RUNNING:
                            active_states.append(state)
                return active_states

            # Run in thread to avoid blocking event loop
            return await asyncio.to_thread(_sync_list_active)

        except Exception as e:
            logger.error(f"Failed to list active states: {e}")
            return []

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        self._initialized = False


# =============================================================================
# Global State Manager
# =============================================================================


def initialize_global_state_manager(
    redis_host: str | None = None,
    redis_port: int | None = None,
    state_ttl: int = 3600 * 24,
) -> bool:
    """
    Initialize the global run state manager.

    Args:
        redis_host: Optional Redis host
        redis_port: Optional Redis port
        state_ttl: State TTL in seconds

    Returns:
        True if initialization was successful
    """
    global _global_state_manager, _initialized

    with _global_lock:
        if _initialized:
            return _global_state_manager is not None and _global_state_manager.is_enabled

        _global_state_manager = RunStateManager(
            redis_host=redis_host,
            redis_port=redis_port,
            state_ttl=state_ttl,
        )
        _initialized = True

        return _global_state_manager.is_enabled


def get_global_state_manager() -> RunStateManager:
    """
    Get the global run state manager.

    Auto-initializes if not already initialized.

    Returns:
        RunStateManager instance
    """
    global _global_state_manager, _initialized

    if not _initialized:
        initialize_global_state_manager()

    if _global_state_manager is None:
        with _global_lock:
            if _global_state_manager is None:
                _global_state_manager = RunStateManager(auto_initialize=True)
                _initialized = True

    return _global_state_manager
