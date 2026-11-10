"""PostgresSessionProvider without a database: registration, config checks, failure mapping."""

from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")

from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionConnectionError
from continuum.session.providers import create_provider, list_providers
from continuum.session.providers.postgres import PostgresSessionProvider


def test_registered_under_postgres():
    assert "postgres" in list_providers()
    provider = create_provider("postgres", SessionConfig(provider="postgres"))
    assert isinstance(provider, PostgresSessionProvider)
    assert provider.provider_name == "postgres"


def test_is_configured_needs_a_dsn():
    assert not SessionConfig(provider="postgres", postgres_dsn=None).is_configured()
    assert SessionConfig(provider="postgres", postgres_dsn="postgresql://h/db").is_configured()


@pytest.mark.parametrize("prefix", ["Sessions", "a;drop table x", "1abc", ""])
def test_table_prefix_must_be_an_identifier(prefix):
    with pytest.raises(ValueError, match="postgres_table_prefix"):
        PostgresSessionProvider(SessionConfig(provider="postgres", postgres_table_prefix=prefix))


async def test_missing_dsn_is_a_connection_error():
    provider = PostgresSessionProvider(SessionConfig(provider="postgres", postgres_dsn=None))
    assert await provider.aping() is False
    assert "SESSION_POSTGRES_DSN" in provider.last_probe_error
    with pytest.raises(SessionConnectionError):
        await provider.get_or_create_session(user_id="u1", conversation_id="c1")


async def test_unreachable_database_is_a_connection_error():
    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_dsn="postgresql://127.0.0.1:1/none")
    )
    with pytest.raises(SessionConnectionError):
        await provider.get_or_create_session(user_id="u1", conversation_id="c1")


async def test_session_client_degrades_when_postgres_is_unreachable():
    from continuum.session.client import SessionClient

    client = SessionClient(
        SessionConfig(
            provider="postgres",
            postgres_dsn="postgresql://127.0.0.1:1/none",
            fallback_mode="degrade",
        )
    )
    sid = await client.get_or_create_session(user_id="u1", conversation_id="c1")
    assert sid
    assert client.persistence_degraded is True


async def test_given_pool_is_never_closed():
    class Pool:
        closed = False

        async def close(self):
            self.closed = True

    pool = Pool()
    provider = PostgresSessionProvider(SessionConfig(provider="postgres"), pool=pool)
    await provider.close()
    assert pool.closed is False
