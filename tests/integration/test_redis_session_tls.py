"""
Integration test for the TLS Redis session path (S-TLS regression).

Guards the bug where TLS was requested via an `ssl=True` pool kwarg, which
redis-py forwards to AbstractConnection and rejects with TypeError on the first
command — silently degrading every TLS deployment to the in-memory fallback.

Requires the opt-in TLS Redis service and generated certs:

    tests/integration/redis_tls/gen_certs.sh
    docker compose --profile tls-test up -d redis-sdk-tls
    pytest tests/integration/test_redis_session_tls.py

Skips cleanly (never fails) when the certs or the TLS service are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CA_CERT = Path(__file__).parent / "redis_tls" / "certs" / "ca.crt"
_TLS_PORT = 6381


@pytest.fixture
async def tls_session_provider():
    if not _CA_CERT.exists():
        pytest.skip(
            "TLS certs missing — run tests/integration/redis_tls/gen_certs.sh "
            "and start `docker compose --profile tls-test up redis-sdk-tls`."
        )

    from continuum.session.config import SessionConfig
    from continuum.session.providers.redis import RedisSessionProvider

    config = SessionConfig(
        enabled=True,
        redis_host="localhost",
        redis_port=_TLS_PORT,
        redis_ssl=True,
        # Verify against the local test CA (default 'required' cert_reqs is kept).
        redis_ssl_ca_certs=str(_CA_CERT),
        ttl_seconds=300,
        max_messages=100,
    )
    provider = RedisSessionProvider(config=config)
    provider.initialize()
    # Probe over TLS; skip (not fail) if the opt-in service isn't up.
    if not await provider.aping():
        await provider.close()
        pytest.skip("TLS Redis not reachable on :6381 (start the tls-test profile).")
    yield provider
    await provider.close()


class TestRedisSessionTLS:
    async def test_tls_round_trip_persists(self, tls_session_provider, test_id):
        """A full create → add → read cycle must work over TLS (no TypeError,
        no silent in-memory degrade)."""
        from continuum.session.types import ChatMessage

        sid = await tls_session_provider.get_or_create_session(
            session_id=f"tls-sess-{test_id}", user_id="tls-user"
        )
        await tls_session_provider.add_message(sid, ChatMessage(role="user", content="over TLS"))
        messages = await tls_session_provider.get_messages(sid)

        assert [m.content for m in messages] == ["over TLS"]

    async def test_pool_uses_ssl_connection(self, tls_session_provider):
        """The live client must be backed by an SSLConnection pool — proving TLS
        is actually negotiated, not silently dropped to plaintext."""
        from redis.asyncio.connection import SSLConnection

        pool = tls_session_provider._redis.connection_pool
        assert pool.connection_class is SSLConnection
