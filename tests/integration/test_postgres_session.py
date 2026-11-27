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
    # Pin plaintext ids: SESSION_HASH_IDS=true in the environment must not change it.
    plain = PostgresSessionProvider(_config(postgres_table_prefix=prefix, hash_session_ids=False))
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


async def _drop(p: PostgresSessionProvider) -> None:
    async with p._conn() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
    await p.close()


async def test_clear_racing_a_trimming_add_does_not_deadlock():
    p = PostgresSessionProvider(_config(max_messages=4, sliding_window_trim_count=2))
    try:
        sid = await p.get_or_create_session(user_id="u1", conversation_id="dl")
        for round_ in range(30):
            while (await p.get_session_metadata(sid)).message_count < 4:
                await p.add_message(sid, _msg("user", "fill"))
            await asyncio.gather(
                p.clear_session(sid), p.add_message(sid, _msg("user", f"r{round_}"))
            )
    finally:
        await _drop(p)


async def test_clear_refreshes_expiry(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="exp")
    async with provider._conn() as conn:
        await conn.execute(
            f"UPDATE {provider._sessions} SET expires_at = now() + interval '5 seconds'"
        )
    await provider.clear_session(sid)
    async with provider._conn() as conn:
        left = await conn.fetchval(
            f"SELECT extract(epoch FROM expires_at - now()) FROM {provider._sessions}"
        )
    assert left > 200  # ttl_seconds=300 in _config


async def test_application_pool_with_jsonb_codec():
    import json

    async def init(conn):
        for name in ("json", "jsonb"):
            await conn.set_type_codec(
                name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2, init=init)
    p = PostgresSessionProvider(_config(), pool=pool)
    try:
        sid = await p.get_or_create_session(user_id="u1", conversation_id="codec")
        await p.add_message(sid, _msg("user", "hi"), metadata={"k": 1})
        assert [m.content for m in await p.get_messages(sid)] == ["hi"]
        meta = await p.get_session_metadata(sid)
        meta.custom = {"fund": "F-1"}
        assert await p.update_session_metadata(sid, meta)
        assert (await p.get_session_metadata(sid)).custom == {"fund": "F-1"}
        # Stored as an object, not as a JSON string holding an object.
        assert await pool.fetchval(f"SELECT jsonb_typeof(message) FROM {p._messages}") == "object"
    finally:
        await pool.execute(f"DROP TABLE IF EXISTS {p._messages}, {p._sessions}")
        await pool.close()


async def test_nul_characters_are_stored_without_error(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="nul")
    await provider.add_message(sid, _msg("user", "pdf\x00text"), metadata={"src": "a\x00b"})
    [back] = await provider.get_messages(sid)
    assert back.content == "pdftext"
    meta = await provider.get_session_metadata(sid)
    meta.custom = {"note": "x\x00y"}
    assert await provider.update_session_metadata(sid, meta)
    assert (await provider.get_session_metadata(sid)).custom == {"note": "xy"}


async def test_role_that_does_not_own_the_tables_can_use_them():
    from urllib.parse import urlsplit, urlunsplit

    owner = PostgresSessionProvider(_config())
    prefix = owner._sessions[:-1]
    role = f"app_{uuid.uuid4().hex[:8]}"
    parts = urlsplit(DSN)
    netloc = f"{role}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    app = PostgresSessionProvider(
        _config(
            postgres_dsn=urlunsplit(parts._replace(netloc=netloc)), postgres_table_prefix=prefix
        )
    )
    try:
        await owner.get_or_create_session(user_id="owner", conversation_id="setup")
        async with owner._conn() as conn:
            await conn.execute(f"CREATE ROLE {role} LOGIN")
            await conn.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {owner._sessions}, {owner._messages} "
                f"TO {role}"
            )
            await conn.execute(f"GRANT USAGE ON SEQUENCE {owner._messages}_id_seq TO {role}")
        assert await app.aping(), app.last_probe_error
        sid = await app.get_or_create_session(user_id="u1", conversation_id="least-privilege")
        await app.add_message(sid, _msg("user", "hi"))
        assert [m.content for m in await app.get_messages(sid)] == ["hi"]
    finally:
        await app.close()
        async with owner._conn() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {owner._messages}, {owner._sessions}")
            await conn.execute(f"DROP ROLE IF EXISTS {role}")
        await owner.close()


async def test_hung_query_times_out_as_connection_error():
    from continuum.session.exceptions import SessionConnectionError

    p = PostgresSessionProvider(_config(postgres_timeout_seconds=1))
    try:
        await p.get_or_create_session(user_id="u1", conversation_id="slow")
        with pytest.raises(SessionConnectionError, match="TimeoutError"):
            async with p._conn() as conn:
                await conn.execute("SELECT pg_sleep(5)")
    finally:
        await _drop(p)


async def _expire_all(provider) -> None:
    async with provider._conn() as conn:
        await conn.execute(
            f"UPDATE {provider._sessions} SET expires_at = now() - interval '1 second'"
        )


async def test_clear_does_not_revive_an_expired_session(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="gone")
    meta = await provider.get_session_metadata(sid)
    meta.custom = {"plan": "old"}
    assert await provider.update_session_metadata(sid, meta)
    await _expire_all(provider)
    assert await provider.clear_session(sid) is True
    assert await provider.get_session_metadata(sid) is None
    # Coming back starts a fresh session, without the old one's fields.
    assert await provider.get_or_create_session(user_id="u1", conversation_id="gone") == sid
    assert (await provider.get_session_metadata(sid)).custom == {}


async def test_purge_works_off_a_backlog_in_batches(provider):
    for i in range(7):
        sid = await provider.get_or_create_session(user_id="u1", conversation_id=f"b{i}")
        await provider.add_message(sid, _msg("user", "x"))
    await _expire_all(provider)
    live = await provider.get_or_create_session(user_id="u1", conversation_id="live")
    statements = []
    real_conn = provider._conn

    def counting_conn(**kwargs):
        statements.append(1)
        return real_conn(**kwargs)

    provider._conn = counting_conn
    assert await provider.purge_expired(batch_size=3) == 7
    assert len(statements) == 3  # 3 + 3 + 1
    provider._conn = real_conn
    async with provider._conn() as conn:
        assert await conn.fetchval(f"SELECT count(*) FROM {provider._messages}") == 0
        assert await conn.fetchval(f"SELECT count(*) FROM {provider._sessions}") == 1
    assert await provider.get_session_metadata(live) is not None
    assert await provider.purge_expired() == 0


async def test_lone_surrogates_are_stored_without_error(provider):
    sid = await provider.get_or_create_session(user_id="u1", conversation_id="emoji")
    await provider.add_message(sid, _msg("user", "cut \ud83d"), metadata={"m": "\ude00"})
    [back] = await provider.get_messages(sid)
    assert back.content == "cut \ufffd"
