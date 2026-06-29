"""
Temporal Client wrapper.

Provides an ergonomic wrapper over temporalio.client.Client with
connection management and convenience methods.
"""

from __future__ import annotations

import threading
import uuid
from datetime import timedelta
from typing import Any

try:
    from temporalio.client import Client, WorkflowHandle
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

from continuum.logging import get_logger
from continuum.temporal.config import TemporalConfig
from continuum.temporal.exceptions import TemporalConnectionError

logger = get_logger(__name__)


class TemporalClient:
    """Thin ergonomic wrapper over temporalio.client.Client."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self._config = config or TemporalConfig.from_settings()
        self._client: Client | None = None

    async def connect(self, host: str | None = None, namespace: str | None = None) -> None:
        """Connect to Temporal server. Uses settings defaults if not provided.

        Delegates the actual connection (TLS / API key / data converter) to the
        TemporalConnector, the single source of truth for Temporal connections.
        """
        from continuum.connectors.temporal import TemporalConnector

        connector = TemporalConnector(self._config)
        self._client = await connector.connect(host=host, namespace=namespace)

    async def disconnect(self) -> None:
        """Disconnect from Temporal server.

        The temporalio SDK does not expose an explicit ``close()`` on its
        ``Client``.  The underlying gRPC channel lives inside a Rust bridge
        object; dropping the Python reference lets the Rust destructor
        reclaim the channel once the reference count reaches zero.

        We proactively nil the bridge reference so the channel is released
        immediately rather than waiting for Python GC.
        """
        if self._client is not None:
            try:
                service_client = getattr(self._client, "service_client", None)
                if service_client is not None:
                    bridge = getattr(service_client, "_bridge_client", None)
                    if bridge is not None:
                        service_client._bridge_client = None
            except Exception as e:
                logger.debug(f"Non-critical: could not release bridge client: {e}")
            self._client = None
        logger.info("Disconnected from Temporal")

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def raw_client(self) -> Client:
        """Access the underlying Temporal client."""
        if self._client is None:
            raise TemporalConnectionError("Not connected to Temporal server")
        return self._client

    async def start_workflow(
        self,
        workflow_fn: Any,
        arg: Any,
        *,
        id: str | None = None,
        task_queue: str | None = None,
        execution_timeout: timedelta | None = None,
        **kwargs: Any,
    ) -> WorkflowHandle:
        """Start any workflow."""
        workflow_id = id or f"workflow-{uuid.uuid4().hex[:16]}"
        queue = task_queue or self._config.task_queue
        timeout = execution_timeout or timedelta(seconds=self._config.workflow_execution_timeout)

        handle = await self.raw_client.start_workflow(
            workflow_fn,
            arg,
            id=workflow_id,
            task_queue=queue,
            execution_timeout=timeout,
            **kwargs,
        )
        logger.info(f"Started workflow {workflow_id} on queue {queue}")
        return handle

    async def run_agent_workflow(
        self,
        input: Any,
        *,
        id: str | None = None,
        task_queue: str | None = None,
    ) -> WorkflowHandle:
        """Convenience: start the generic AgentWorkflow."""
        from continuum.temporal.workflows.agent_workflow import AgentWorkflow

        return await self.start_workflow(
            AgentWorkflow.run,
            input,
            id=id,
            task_queue=task_queue,
        )

    async def signal_workflow(self, workflow_id: str, signal_name: str, arg: Any = None) -> None:
        """Send a signal to a running workflow."""
        handle = self.raw_client.get_workflow_handle(workflow_id)
        await handle.signal(signal_name, arg)

    async def query_workflow(self, workflow_id: str, query_name: str) -> Any:
        """Query a running workflow."""
        handle = self.raw_client.get_workflow_handle(workflow_id)
        return await handle.query(query_name)

    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
        handle = self.raw_client.get_workflow_handle(workflow_id)
        await handle.cancel()

    async def get_workflow_result(self, workflow_id: str, result_type: type | None = None) -> Any:
        """Get the result of a completed workflow."""
        handle = self.raw_client.get_workflow_handle(workflow_id)
        if result_type:
            return await handle.result()
        return await handle.result()

    async def get_workflow_handle(self, workflow_id: str) -> WorkflowHandle:
        """Get a handle to an existing workflow."""
        return self.raw_client.get_workflow_handle(workflow_id)


_global_client: TemporalClient | None = None
_client_lock = threading.Lock()


def get_temporal_client() -> TemporalClient:
    """Get the global Temporal client (singleton)."""
    global _global_client
    if _global_client is None:
        with _client_lock:
            if _global_client is None:
                _global_client = TemporalClient()
    return _global_client


def reset_temporal_client() -> None:
    """Reset the global client (for testing)."""
    global _global_client
    _global_client = None
