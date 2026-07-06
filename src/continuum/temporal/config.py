"""
Temporal integration configuration.

Provides TemporalConfig which reads from the global Settings
and can be overridden programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuum.config import settings


@dataclass
class TemporalConfig:
    """Configuration for the Temporal integration.

    Reads defaults from the global Settings instance but allows
    per-instance overrides.
    """

    enabled: bool = False
    host: str = "localhost:7233"
    namespace: str = "default"
    tls: bool = False
    api_key: str | None = None
    task_queue: str = "orchestrator-agents"
    enable_human_in_loop: bool = True
    approval_timeout_seconds: int = 86400
    workflow_execution_timeout: int = 86400 * 7
    activity_start_to_close_timeout: int = 300
    activity_retry_max_attempts: int = 3

    @classmethod
    def from_settings(cls) -> TemporalConfig:
        """Create config from global settings."""
        return cls(
            enabled=settings.temporal_enabled,
            host=settings.temporal_host,
            namespace=settings.temporal_namespace,
            tls=settings.temporal_tls,
            api_key=settings.temporal_api_key,
            task_queue=settings.temporal_task_queue,
            enable_human_in_loop=settings.temporal_enable_human_in_loop,
            approval_timeout_seconds=settings.temporal_approval_timeout_seconds,
            workflow_execution_timeout=settings.temporal_workflow_execution_timeout,
            activity_start_to_close_timeout=settings.temporal_activity_start_to_close_timeout,
            activity_retry_max_attempts=settings.temporal_activity_retry_max_attempts,
        )
