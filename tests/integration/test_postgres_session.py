"""PostgresSessionProvider against a real PostgreSQL.

Runs only when CONTINUUM_TEST_POSTGRES_DSN is set, e.g.
    docker run -d --rm -p 55499:5432 -e POSTGRES_HOST_AUTH_METHOD=trust postgres:16-alpine
    CONTINUUM_TEST_POSTGRES_DSN=postgresql://postgres@localhost:55499/postgres pytest -m postgres
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

asyncpg = pytest.importorskip("asyncpg")

from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionMessageLimitError, SessionNotFoundError
from continuum.session.providers.postgres import PostgresSessionProvider
from continuum.session.types import ChatMessage

DSN = os.environ.get("CONTINUUM_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="CONTINUUM_TEST_POSTGRES_DSN not set"),
]


def _config(**overrides) -> SessionConfig:
    values = {
        "provider": "postgres",
        "enabled": True,
        "postgres_dsn": DSN,
        "postgres_table_prefix": f"t_{uuid.uuid4().hex[:8]}_session",
        "ttl_seconds": 300,
        "max_messages": 100,
    }
    values.update(overrides)
    return SessionConfig(**values)


@pytest.fixture
async def provider():
    p = PostgresSessionProvider(_config())
    yield p
    async with p._conn() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
    await p.close()


def _msg(role: str, text: str) -> ChatMessage:
    return ChatMessage(role=role, content=text)


async def test_round_trip(provider):
    assert await provider.aping()
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="c1")
    assert await provider.get_or_create_session(user_id="u1", conversation_id="c1") == sid
    await provider.add_message(sid, _msg("user", "hi"), metadata={"k": 1})
    await provider.add_message(sid, _msg("assistant", "hello"))
    assert [(m.role, m.content) for m in await provider.get_messages(sid)] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]
    meta = await provider.get_session_metadata(sid)
    assert meta.message_count == 2 and meta.user_id == "u1" and meta.conversation_id == "c1"


async def test_tool_calls_survive_storage(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="tools")
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "sql", "arguments": "{}"}}
        ],
    )
    await provider.add_message(sid, msg)
    [back] = await provider.get_messages(sid)
    assert back.tool_calls[0].function.name == "sql"


async def test_limit_counts_turns_and_starts_at_a_user_message(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="c2")
    for i in range(5):
        await provider.add_message(sid, _msg("user", f"q{i}"))
        await provider.add_message(sid, _msg("assistant", f"a{i}"))
    recent = await provider.get_messages(sid, limit=2)
    assert [m.content for m in recent] == ["q3", "a3", "q4", "a4"]


async def test_sliding_window_and_error_strategies():
    p = PostgresSessionProvider(_config(max_messages=4, sliding_window_trim_count=2))
    try:
        sid = await p.get_or_create_session(user_id="u1", conversation_id="w")
        for i in range(6):
            await p.add_message(sid, _msg("user", str(i)))
        assert [m.content for m in await p.get_messages(sid)] == ["2", "3", "4", "5"]
    finally:
        async with p._conn() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
        await p.close()

    p = PostgresSessionProvider(_config(max_messages=1, message_limit_strategy="error"))
    try:
        sid = await p.get_or_create_session(user_id="u1", conversation_id="e")
        await p.add_message(sid, _msg("user", "one"))
        with pytest.raises(SessionMessageLimitError):
            await p.add_message(sid, _msg("user", "two"))
    finally:
        async with p._conn() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
        await p.close()


async def test_clear_delete_and_missing(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="c3")
    await provider.add_message(sid, _msg("user", "x"))
    assert await provider.clear_session(sid)
    assert await provider.get_messages(sid) == []
    assert await provider.delete_session(sid)
    assert await provider.get_session_metadata(sid) is None
    with pytest.raises(SessionNotFoundError):
        await provider.get_messages(sid)
    with pytest.raises(SessionNotFoundError):
        await provider.add_message(sid, _msg("user", "late"))


async def test_expired_sessions_read_as_missing_and_purge(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="old")
    await provider.add_message(sid, _msg("user", "x"))
    async with provider._conn() as conn:
        await conn.execute(
            f"UPDATE {provider._sessions} SET expires_at = now() - interval '1 second'"
        )
    assert await provider.get_session_metadata(sid) is None
    assert await provider.purge_expired() == 1
    # Re-creating starts empty
    assert await provider.get_or_create_session(user_id="u1", conversation_id="old") == sid
    assert await provider.get_messages(sid) == []


async def test_update_metadata(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="m")
    meta = await provider.get_session_metadata(sid)
    meta.custom = {"fund": "F-12"}
    meta.agent_id = "research"
    assert await provider.update_session_metadata(sid, meta)
    again = await provider.get_session_metadata(sid)
    assert again.custom == {"fund": "F-12"} and again.agent_id == "research"
    assert not await provider.update_session_metadata("nope", meta)


async def test_concurrent_writers_keep_every_message(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="race")
    await asyncio.gather(*(provider.add_message(sid, _msg("user", str(i))) for i in range(20)))
    assert (await provider.get_session_metadata(sid)).message_count == 20


async def test_keyed_ids_migrate_plaintext_sessions():
    prefix = f"t_{uuid.uuid4().hex[:8]}_session"
    plain = PostgresSessionProvider(_config(postgres_table_prefix=prefix))
    keyed = PostgresSessionProvider(
        _config(postgres_table_prefix=prefix, hash_session_ids=True, session_id_secret="s" * 32)
    )
    try:
        old = await plain.get_or_create_session(user_id="u1", conversation_id="c1")
        await plain.add_message(old, _msg("user", "kept"))
        new = await keyed.get_or_create_session(user_id="u1", conversation_id="c1")
        assert new != old
        assert [m.content for m in await keyed.get_messages(new)] == ["kept"]
        assert await keyed.get_session_metadata(old) is None
    finally:
        async with plain._conn() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {plain._messages}, {plain._sessions}")
        await plain.close()
        await keyed.close()


async def test_shared_pool_is_left_open():
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)
    p = PostgresSessionProvider(_config(), pool=pool)
    try:
        sid = await p.get_or_create_session(user_id="u1", conversation_id="pool")
        assert sid
        await p.close()
        assert await pool.fetchval("SELECT 1") == 1
    finally:
        await pool.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
        await pool.close()
