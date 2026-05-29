"""
Temporal Worker Manager.

Manages Temporal workers for the SDK, auto-registering built-in
activities, workflows, and any user-registered custom ones.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

try:
    from temporalio.worker import Worker
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. "
        "Install it with: pip install -e '.[temporal]'"
    ) from _err

from orchestrator.logging import get_logger
from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.client import TemporalClient
from orchestrator.temporal.config import TemporalConfig
from orchestrator.temporal.registry import AgentRegistry
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow
from orchestrator.temporal.workflows.loop_workflow import LoopAgentWorkflow
from orchestrator.temporal.workflows.parallel_workflow import ParallelAgentWorkflow
from orchestrator.temporal.workflows.sequential_workflow import SequentialAgentWorkflow

logger = get_logger(__name__)


class WorkerManager:
    """Manages Temporal workers for the SDK."""

    def __init__(
        self,
        client: TemporalClient,
        registry: AgentRegistry,
        config: TemporalConfig | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._config = config or TemporalConfig.from_settings()
        self._worker: Worker | None = None
        self._worker_task: Any = None
        self._custom_workflows: list[type] = []
        self._custom_activities: list[Callable[..., Any]] = []
        self._running = False

    async def start(self, task_queue: str | None = None) -> None:
        """Start the worker.

        Auto-registers built-in activities and workflows plus any
        user-registered custom workflows and activities.
        """
        if self._running:
            logger.warning("Worker already running")
            return

        queue = task_queue or self._config.task_queue

        workflows = [
            AgentWorkflow,
            SequentialAgentWorkflow,
            ParallelAgentWorkflow,
            LoopAgentWorkflow,
            *self._custom_workflows,
        ]

        activities = [
            run_agent_activity,
            send_notification_activity,
            *self._custom_activities,
        ]

        self._worker = Worker(
            self._client.raw_client,
            task_queue=queue,
            workflows=workflows,
            activities=activities,
        )

        import asyncio

        self._worker_task = asyncio.create_task(self._worker.run())
        self._running = True
        logger.info(
            f"Temporal worker started on queue '{queue}' "
            f"({len(workflows)} workflows, {len(activities)} activities)"
        )

    async def stop(self) -> None:
        """Gracefully shutdown the worker."""
        if not self._running or self._worker is None:
            return

        await self._worker.shutdown()
        if self._worker_task:
            try:
                await self._worker_task
            except Exception:
                pass
        self._running = False
        self._worker = None
        self._worker_task = None
        logger.info("Temporal worker stopped")

    def register_workflow(self, workflow_cls: type) -> None:
        """Register a user's custom workflow class."""
        self._custom_workflows.append(workflow_cls)

    def register_activity(self, activity_fn: Callable[..., Any]) -> None:
        """Register a user's custom activity function."""
        self._custom_activities.append(activity_fn)

    @property
    def is_running(self) -> bool:
        return self._running


_global_worker_manager: WorkerManager | None = None
_worker_lock = threading.Lock()


def get_worker_manager(
    client: TemporalClient | None = None,
    registry: AgentRegistry | None = None,
) -> WorkerManager:
    """Get the global worker manager (singleton)."""
    global _global_worker_manager
    if _global_worker_manager is None:
        with _worker_lock:
            if _global_worker_manager is None:
                from orchestrator.temporal.client import get_temporal_client
                from orchestrator.temporal.registry import get_agent_registry

                _global_worker_manager = WorkerManager(
                    client=client or get_temporal_client(),
                    registry=registry or get_agent_registry(),
                )
    return _global_worker_manager


def reset_worker_manager() -> None:
    """Reset the global worker manager (for testing)."""
    global _global_worker_manager
    _global_worker_manager = None
