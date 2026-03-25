"""
Root conftest.py — shared fixtures for all test levels.

All fixtures use REAL services (Redis, Qdrant, LLM APIs) configured via .env.
No mocking.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from dotenv import load_dotenv

# Load .env before any orchestrator imports
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Unique IDs for test isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def test_id() -> str:
    """Unique ID to isolate test data."""
    return f"test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def session_id(test_id: str) -> str:
    return f"sess-{test_id}"


@pytest.fixture
def user_id(test_id: str) -> str:
    return f"user-{test_id}"


# ---------------------------------------------------------------------------
# Real Redis client (requires redis-sdk on port 6380)
# ---------------------------------------------------------------------------


@pytest.fixture
async def real_redis():
    """Real async Redis client connected to the SDK Redis container."""
    import redis.asyncio as aioredis

    host = os.getenv("SESSION_REDIS_HOST", "localhost")
    port = int(os.getenv("SESSION_REDIS_PORT", "6380"))
    password = os.getenv("SESSION_REDIS_PASSWORD", "sdk123456789")

    client = aioredis.Redis(host=host, port=port, password=password or None, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available on port 6380")

    yield client

    await client.aclose()


# ---------------------------------------------------------------------------
# Real Qdrant client (requires qdrant on port 6333)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_qdrant():
    """Real Qdrant client connected to the local container."""
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    client = QdrantClient(host=host, port=port)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant not available on port 6333")

    yield client
    client.close()


# ---------------------------------------------------------------------------
# Real LLM client
# ---------------------------------------------------------------------------


@pytest.fixture
def real_llm_client():
    """Real LLMClient configured from .env."""
    from orchestrator.llm.client import LLMClient
    from orchestrator.llm.config import LLMConfig

    model = os.getenv("DEFAULT_LLM_MODEL", "gemini/gemini-2.5-flash")
    client = LLMClient(
        config=LLMConfig(model=model, temperature=0.0, max_tokens=256),
        enable_langfuse=False,
    )
    return client


# ---------------------------------------------------------------------------
# Container (DI) fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def container():
    """Get a fresh container instance."""
    from orchestrator.core.container import get_container, reset_container

    reset_container()
    c = get_container()
    yield c
    reset_container()
