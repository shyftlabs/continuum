"""
Pytest fixtures and configuration for memory benchmarks.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.benchmark


@pytest.fixture
def mock_memory_config():
    """Configured in-memory provider for fast benchmark harness testing."""
    from continuum.memory.config import MemoryConfig

    return MemoryConfig(
        provider="mock",
        enabled=True,
    )
