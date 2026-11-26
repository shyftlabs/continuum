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


# -----------------------------------------------------------------------------
# Review fixes: settings, DDL skip, timeouts, encoding, health, messages, CI
# -----------------------------------------------------------------------------


class _Conn:
    """Records statements; fetchval answers from a queue."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.sql: list[str] = []

    async def fetchval(self, sql, *args):
        self.sql.append(sql)
        return self.answers.pop(0) if self.answers else None

    async def execute(self, sql, *args):
        self.sql.append(sql)
        return "DELETE 0"

    def transaction(self):
        return _Ctx(None)


class _Ctx:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn
        self.acquire_timeouts: list[float | None] = []

    def acquire(self, timeout=None):
        self.acquire_timeouts.append(timeout)
        return _Ctx(self.conn)

    async def close(self):
        pass


def test_session_provider_is_read_from_the_environment(monkeypatch):
    from continuum.config import Settings

    monkeypatch.setenv("SESSION_PROVIDER", "postgres")
    assert Settings(_env_file=None).session_provider == "postgres"
    monkeypatch.delenv("SESSION_PROVIDER")
    assert Settings(_env_file=None).session_provider == "redis"


def test_session_config_defaults_to_the_configured_provider(monkeypatch):
    from continuum.config import settings

    monkeypatch.setattr(settings, "session_provider", "postgres")
    assert SessionConfig().provider == "postgres"
    monkeypatch.setattr(settings, "session_provider", "redis")
    assert SessionConfig().provider == "redis"


async def test_existing_tables_skip_the_ddl():
    conn = _Conn(answers=[True])
    provider = PostgresSessionProvider(SessionConfig(provider="postgres"), pool=_Pool(conn))
    await provider._ensure_ready()
    assert any("to_regclass" in q for q in conn.sql)
    assert not any("CREATE" in q for q in conn.sql)


async def test_missing_tables_are_created():
    conn = _Conn(answers=[False])
    provider = PostgresSessionProvider(SessionConfig(provider="postgres"), pool=_Pool(conn))
    await provider._ensure_ready()
    assert any("CREATE TABLE IF NOT EXISTS" in q for q in conn.sql)


async def test_create_tables_false_runs_no_check_and_no_ddl():
    conn = _Conn()
    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_create_tables=False), pool=_Pool(conn)
    )
    await provider._ensure_ready()
    assert conn.sql == []


async def test_connect_command_and_acquire_timeouts(monkeypatch):
    import continuum.session.providers.postgres as pg

    conn = _Conn(answers=[True, 1])
    pool = _Pool(conn)
    seen: dict = {}

    async def create_pool(dsn, **kwargs):
        seen.update(kwargs)
        return pool

    monkeypatch.setattr(pg.asyncpg, "create_pool", create_pool)
    provider = PostgresSessionProvider(
        SessionConfig(
            provider="postgres", postgres_dsn="postgresql://h/db", postgres_timeout_seconds=2.5
        )
    )
    assert await provider.aping() is True
    assert seen["timeout"] == 2.5 and seen["command_timeout"] == 2.5
    assert pool.acquire_timeouts and all(t == 2.5 for t in pool.acquire_timeouts)


async def test_acquire_timeout_is_a_connection_error():
    class ExhaustedPool(_Pool):
        def acquire(self, timeout=None):
            raise TimeoutError

    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_create_tables=False),
        pool=ExhaustedPool(_Conn()),
    )
    with pytest.raises(SessionConnectionError):
        await provider.get_messages("s1")


def test_nul_characters_are_dropped_from_stored_json():
    import json

    from continuum.session.providers.postgres import _dumps

    out = _dumps({"content": "a\x00b", "custom": {"k\x00": ["x\x00", 1, None]}, "p": "\\u0000"})
    assert "\\u0000" not in out.replace("\\\\u0000", "")
    assert json.loads(out) == {"content": "ab", "custom": {"k": ["x", 1, None]}, "p": "\\u0000"}


def test_jsonb_goes_over_the_wire_as_text():
    import inspect

    import continuum.session.providers.postgres as pg

    src = inspect.getsource(pg.PostgresSessionProvider)
    assert "$2::jsonb" not in src and "$5::jsonb" not in src
    assert "$2::text::jsonb" in src and "$5::text::jsonb" in src
    assert "message::text" in src and "custom::text" in src


def _warnings(monkeypatch) -> list[str]:
    """Capture the session client's warnings without relying on caplog.

    Another test can leave the continuum logger not propagating, so caplog sees
    nothing depending on test order; patching the module logger does not.
    """
    import continuum.session.client as client_module

    seen: list[str] = []
    real = client_module.logger

    class _Logger:
        def __getattr__(self, name):
            return getattr(real, name)

        def warning(self, msg, *args, **kwargs):
            seen.append(msg % args if args else str(msg))

    monkeypatch.setattr(client_module, "logger", _Logger())
    return seen


async def test_fallback_warning_names_postgres(monkeypatch):
    from continuum.session.client import SessionClient

    seen = _warnings(monkeypatch)
    client = SessionClient(
        SessionConfig(provider="postgres", postgres_dsn=None, fallback_mode="degrade")
    )
    await client.get_or_create_session(user_id="u1", conversation_id="c1")
    text = "\n".join(seen)
    assert "PostgreSQL is not configured" in text
    assert "SESSION_POSTGRES_DSN" in text
    assert "SESSION_REDIS_HOST" not in text


async def test_unreachable_postgres_message_keeps_display_casing(monkeypatch):
    from continuum.session.client import SessionClient

    seen = _warnings(monkeypatch)
    client = SessionClient(
        SessionConfig(
            provider="postgres",
            postgres_dsn="postgresql://127.0.0.1:1/none",
            postgres_timeout_seconds=1,
            fallback_mode="degrade",
        )
    )
    await client.get_or_create_session(user_id="u1", conversation_id="c1")
    assert any("PostgreSQL probe failed" in m for m in seen)


async def test_redis_messages_keep_their_casing(monkeypatch):
    from continuum.session.client import SessionClient, _display_name

    seen = _warnings(monkeypatch)
    client = SessionClient(SessionConfig(provider="redis", redis_host="", fallback_mode="degrade"))
    await client.get_or_create_session(user_id="u1", conversation_id="c1")
    text = "\n".join(seen)
    assert "Redis is not configured" in text
    assert "configure Redis (SESSION_REDIS_HOST / SESSION_REDIS_PORT)" in text
    assert f"{_display_name('redis')} probe failed" == "Redis probe failed"


def test_health_check_probes_postgres_not_redis(monkeypatch):
    from continuum.config import settings
    from continuum.core.health import HealthCheck

    monkeypatch.setattr(settings, "session_provider", "postgres")
    checks = HealthCheck()._checks
    assert "postgres" in checks and "redis" not in checks

    monkeypatch.setattr(settings, "session_provider", "redis")
    checks = HealthCheck()._checks
    assert "redis" in checks and "postgres" not in checks


async def test_health_check_reports_unreachable_postgres(monkeypatch):
    from continuum.config import settings
    from continuum.core.health import HealthCheck, HealthStatus

    monkeypatch.setattr(settings, "session_enabled", True)
    monkeypatch.setattr(settings, "session_provider", "postgres")
    monkeypatch.setattr(settings, "session_postgres_dsn", "postgresql://127.0.0.1:1/none")
    monkeypatch.setattr(settings, "session_postgres_timeout_seconds", 1.0)
    result = await HealthCheck().check_postgres()
    assert result.name == "postgres"
    assert result.status == HealthStatus.UNHEALTHY
    assert "PostgreSQL connection failed" in result.message


async def test_connector_registry_probes_postgres_and_skips_redis(monkeypatch):
    from continuum.config import settings
    from continuum.connectors import registry
    from continuum.connectors.postgres import PostgresConnector
    from continuum.connectors.redis import RedisConnector

    monkeypatch.setattr(settings, "session_enabled", True)
    monkeypatch.setattr(settings, "session_provider", "postgres")
    monkeypatch.setattr(settings, "session_postgres_dsn", "postgresql://app@db-host:5432/app")
    monkeypatch.setattr(registry, "_REGISTRY", {})
    registry.register_connector("redis", RedisConnector())
    pg_conn = PostgresConnector()
    pings = []

    async def aping():
        pings.append(True)
        return False

    monkeypatch.setattr(pg_conn, "aping", aping)
    registry.register_connector("postgres", pg_conn)

    report = await registry.health_check_all()
    assert report["redis"]["status"] == "disabled"
    assert report["postgres"]["status"] == "unhealthy"
    assert report["postgres"]["host"] == "db-host" and report["postgres"]["database"] == "app"
    assert pings == [True]


def test_connector_describe_never_shows_credentials():
    from continuum.connectors.postgres import PostgresConnector

    secret = "s3cr3t-" + "value"
    dsn = "postgresql://app:" + secret + "@db-host:5432/app"
    d = PostgresConnector(SessionConfig(provider="postgres", postgres_dsn=dsn)).describe()
    assert secret not in str(d) and "app:" not in str(d)


def test_asyncpg_floor_has_python_313_wheels():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert "asyncpg>=0.30.0,<1" in project["optional-dependencies"]["postgres"]


def test_ci_installs_postgres_extra_and_runs_integration_tests():
    from pathlib import Path

    ci = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    assert 'pip install -e ".[dev,postgres]"' in ci
    assert '--resolution lowest-direct -e ".[dev,postgres]"' in ci
    assert "image: postgres:16" in ci
    assert "POSTGRES_HOST_AUTH_METHOD: trust" in ci
    assert "CONTINUUM_TEST_POSTGRES_DSN: postgresql://postgres@localhost:5432/postgres" in ci
    assert "tests/integration/test_postgres_session.py" in ci


# -----------------------------------------------------------------------------
# Round-2 review fixes
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Redis", "redis"),
        (" redis", "redis"),
        ("", "redis"),
        ("  ", "redis"),
        ("POSTGRES", "postgres"),
        ("memory", "memory"),
        ("postgresql", "postgres"),
        (" PostgreSQL ", "postgres"),
    ],
)
def test_session_provider_env_is_normalized(monkeypatch, raw, expected):
    from continuum.config import Settings

    monkeypatch.setenv("SESSION_PROVIDER", raw)
    assert Settings(_env_file=None).session_provider == expected


def test_session_config_normalizes_provider():
    assert SessionConfig(provider=" Redis ").provider == "redis"
    assert SessionConfig(provider="").provider == "redis"
    assert SessionConfig(provider="Postgres").provider == "postgres"
    assert SessionConfig(provider="postgresql").provider == "postgres"
    assert SessionConfig(provider="memory").is_configured() is True
    with pytest.raises(ValueError, match="'memory'"):
        SessionConfig(provider="dynamodb").is_configured()


def test_zero_timeout_from_the_environment_is_rejected(monkeypatch):
    from pydantic import ValidationError

    from continuum.config import Settings, settings

    monkeypatch.setenv("SESSION_POSTGRES_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError, match="session_postgres_timeout_seconds"):
        Settings(_env_file=None)
    monkeypatch.setattr(settings, "session_postgres_timeout_seconds", 0)
    with pytest.raises(ValidationError, match="postgres_timeout_seconds"):
        SessionConfig()


async def test_purge_expired_deletes_in_batches():
    class PurgeConn(_Conn):
        def __init__(self, counts):
            super().__init__()
            self.counts = list(counts)
            self.args: list[tuple] = []

        async def execute(self, sql, *args):
            self.sql.append(sql)
            self.args.append(args)
            return f"DELETE {self.counts.pop(0)}"

    conn = PurgeConn([2, 2, 1])
    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_create_tables=False), pool=_Pool(conn)
    )
    assert await provider.purge_expired(batch_size=2) == 5
    assert len(conn.sql) == 3
    assert all("LIMIT $1" in q for q in conn.sql) and conn.args == [(2,)] * 3
    with pytest.raises(ValueError, match="batch_size"):
        await provider.purge_expired(batch_size=0)


async def test_clear_session_only_refreshes_a_live_session():
    conn = _Conn()
    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_create_tables=False), pool=_Pool(conn)
    )
    await provider.clear_session("s1")
    [update] = [q for q in conn.sql if q.startswith("UPDATE")]
    assert "expires_at > now()" in update


async def test_timeout_reason_names_the_error_type():
    class ExhaustedPool(_Pool):
        def acquire(self, timeout=None):
            raise TimeoutError

    provider = PostgresSessionProvider(
        SessionConfig(provider="postgres", postgres_create_tables=False),
        pool=ExhaustedPool(_Conn()),
    )
    with pytest.raises(SessionConnectionError, match="unreachable: TimeoutError"):
        await provider.get_messages("s1")


def test_lone_surrogates_become_replacement_characters():
    import json

    from continuum.session.providers.postgres import _dumps

    out = _dumps({"content": "cut \ud83d", "custom": {"\ude00": ["ok 😀"]}})
    # PostgreSQL rejects a lone \ud83d escape in jsonb; a valid pair is fine.
    assert '"cut \\ufffd"' in out
    assert json.loads(out) == {"content": "cut \ufffd", "custom": {"\ufffd": ["ok 😀"]}}


def test_connector_describe_hides_a_keyword_dsn_password():
    from continuum.connectors.postgres import PostgresConnector

    secret = "s3cr3t-" + "value"
    dsn = f"host=db-host port=5433 dbname=app user=app password={secret} sslmode=require"
    d = PostgresConnector(SessionConfig(provider="postgres", postgres_dsn=dsn)).describe()
    assert secret not in str(d) and "password" not in str(d)
    assert (d["host"], d["port"], d["database"]) == ("db-host", 5433, "app")


def _probe_provider(*, reachable: bool, fail_calls: bool = False):
    """A stand-in session provider that records whether it was closed."""
    from continuum.session.base import BaseSessionProvider

    class Provider(BaseSessionProvider):
        provider_name = "postgres"
        config = None
        is_initialized = True
        last_probe_error = None if reachable else "OSError: refused"

        def __init__(self):
            self.closed = 0

        async def aping(self):
            return reachable

        async def get_or_create_session(self, *args, **kwargs):
            if fail_calls:
                raise SessionConnectionError("PostgreSQL session store unreachable: OSError: x")
            return "sid"

        async def add_message(self, *args, **kwargs): ...
        async def get_messages(self, *args, **kwargs): ...
        async def get_session_metadata(self, *args, **kwargs): ...
        async def update_session_metadata(self, *args, **kwargs): ...
        async def clear_session(self, *args, **kwargs): ...
        async def delete_session(self, *args, **kwargs): ...

        async def close(self):
            self.closed += 1

    return Provider()


@pytest.mark.parametrize("mode", ["degrade", "fail"])
async def test_failed_probe_closes_the_dropped_provider(monkeypatch, mode):
    import continuum.session.client as client_module
    from continuum.session.client import SessionClient

    dropped = _probe_provider(reachable=False)
    monkeypatch.setattr(client_module, "create_provider", lambda name, cfg: dropped)
    client = SessionClient(
        SessionConfig(provider="postgres", postgres_dsn="postgresql://h/db", fallback_mode=mode)
    )
    if mode == "fail":
        with pytest.raises(SessionConnectionError):
            await client.get_or_create_session(user_id="u1", conversation_id="c1")
    else:
        await client.get_or_create_session(user_id="u1", conversation_id="c1")
        assert client.persistence_degraded is True
    assert dropped.closed == 1


async def test_mid_session_degrade_closes_the_dropped_provider(monkeypatch):
    import continuum.session.client as client_module
    from continuum.session.client import SessionClient
    from continuum.session.providers.memory import MemorySessionProvider

    dropped = _probe_provider(reachable=True, fail_calls=True)
    monkeypatch.setattr(client_module, "create_provider", lambda name, cfg: dropped)
    client = SessionClient(
        SessionConfig(
            provider="postgres", postgres_dsn="postgresql://h/db", fallback_mode="degrade"
        )
    )
    assert await client.get_or_create_session(user_id="u1", conversation_id="c1")
    assert isinstance(client.active_provider, MemorySessionProvider)
    assert dropped.closed == 1


def _container_with(monkeypatch, *, session_config=None, client=None, provider=None):
    from continuum.core import container as container_module
    from continuum.core.container import Container, ContainerConfig

    container = Container(
        ContainerConfig(auto_initialize=False, enable_session=True, session_config=session_config)
    )
    if client is not None:
        container.set_session_client(client, provider)
    monkeypatch.setattr(container_module, "_global_container", container)
    return container


def test_health_check_follows_session_config_not_only_the_env(monkeypatch):
    from continuum.config import settings
    from continuum.core.health import HealthCheck

    monkeypatch.setattr(settings, "session_enabled", True)
    monkeypatch.setattr(settings, "session_provider", "redis")
    _container_with(monkeypatch, session_config={"provider": "postgres"})
    checks = HealthCheck()._checks
    assert "postgres" in checks and "redis" not in checks


async def test_health_check_probes_a_provider_given_a_pool(monkeypatch):
    from continuum.config import settings
    from continuum.core.health import HealthCheck, HealthStatus
    from continuum.session.client import SessionClient

    monkeypatch.setattr(settings, "session_enabled", True)
    monkeypatch.setattr(settings, "session_provider", "redis")
    monkeypatch.setattr(settings, "session_postgres_dsn", None)
    health = HealthCheck()  # built before the application sets up its store

    conn = _Conn(answers=[True, 1])
    provider = PostgresSessionProvider(SessionConfig(provider="postgres"), pool=_Pool(conn))
    client = SessionClient(SessionConfig(provider="redis"))
    _container_with(monkeypatch, client=client, provider=provider)

    health._checks = {
        name: fn for name, fn in health._checks.items() if name in ("redis", "postgres")
    }
    result = await health.check_all(timeout=5)
    [store] = result.checks
    assert store.name == "postgres" and store.status == HealthStatus.HEALTHY
    assert "SELECT 1" in conn.sql


def test_custom_store_check_is_not_replaced(monkeypatch):
    from continuum.config import settings
    from continuum.core.health import HealthCheck

    monkeypatch.setattr(settings, "session_provider", "redis")
    _container_with(monkeypatch, session_config={"provider": "postgres"})
    health = HealthCheck()

    async def mine():
        raise AssertionError

    health.register_check("postgres", mine)
    health._sync_session_store_check()
    assert health._checks["postgres"] is mine


# ---------------------------------------------------------------------------
# Round-3 review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
async def test_health_check_survives_an_unusable_session_id_secret(monkeypatch, enabled):
    # SESSION_HASH_IDS without a secret makes SessionConfig() raise; the health
    # checks must not build it, so they report instead of crashing.
    from continuum.config import settings
    from continuum.core.health import HealthCheck

    monkeypatch.setattr(settings, "session_enabled", enabled)
    monkeypatch.setattr(settings, "session_provider", "postgres")
    monkeypatch.setattr(settings, "session_hash_ids", True)
    monkeypatch.setattr(settings, "session_id_secret", None)
    with pytest.raises(Exception):  # noqa: B017 — the setting is indeed unusable
        SessionConfig()
    _container_with(monkeypatch)

    health = HealthCheck()
    health._checks = {"postgres": health._checks["postgres"]}
    result = await health.check_all(timeout=5)
    [check] = result.checks
    assert check.name == "postgres"
    if not enabled:
        assert check.details == {"enabled": False}


def test_custom_provider_keeps_its_mixed_case_name():
    from continuum.session.client import SessionClient
    from continuum.session.providers import PROVIDER_REGISTRY, register_provider
    from continuum.session.providers.memory import MemorySessionProvider

    class MyStore(MemorySessionProvider):
        pass

    register_provider("MyStore", MyStore)
    try:
        assert SessionConfig(provider="MyStore").provider == "MyStore"
        client = SessionClient(SessionConfig(provider="MyStore"))
        assert type(client.provider) is MyStore
    finally:
        PROVIDER_REGISTRY.pop("MyStore", None)


def test_settings_repr_masks_the_postgres_dsn(monkeypatch):
    from continuum.config import Settings

    secret = "s3cret" + "pw"  # built at run time: no credential-like literal in the repo
    dsn = "postgresql://app:" + secret + "@" + "db.internal:5432/app"
    monkeypatch.setenv("SESSION_POSTGRES_DSN", dsn)
    s = Settings()
    assert s.session_postgres_dsn == dsn
    assert secret not in repr(s)
    config = SessionConfig(provider="postgres", postgres_dsn="host=db password=" + secret)
    assert secret not in repr(config)


@pytest.mark.parametrize("prefix", ["continuum_session\n", "abc\n"])
def test_table_prefix_with_a_trailing_newline_is_rejected(prefix):
    with pytest.raises(ValueError, match="postgres_table_prefix"):
        PostgresSessionProvider(SessionConfig(provider="postgres", postgres_table_prefix=prefix))


# -----------------------------------------------------------------------------
# Round-4 review fixes
# -----------------------------------------------------------------------------


def test_settings_str_masks_the_postgres_dsn(monkeypatch):
    from continuum.config import Settings

    secret = "s3cret" + "pw"  # built at run time: no credential-like literal in the repo
    monkeypatch.setenv("SESSION_POSTGRES_DSN", "postgresql://app:" + secret + "@db.internal/app")
    monkeypatch.setenv("SESSION_REDIS_PASSWORD", "redis" + secret)
    s = Settings(_env_file=None)
    assert secret not in str(s)
    assert str(s) == repr(s)


@pytest.mark.parametrize("raw", ["dynamodb", "MyStore", "redis-cluster"])
def test_unknown_session_provider_does_not_fail_settings_load(monkeypatch, raw):
    from continuum.config import Settings

    monkeypatch.setenv("SESSION_PROVIDER", raw)
    assert Settings(_env_file=None).session_provider == raw


def test_import_continuum_with_an_unknown_provider_and_sessions_off():
    import os
    import subprocess
    import sys

    env = {**os.environ, "SESSION_PROVIDER": "dynamodb", "SESSION_ENABLED": "false"}
    result = subprocess.run(
        [sys.executable, "-c", "import continuum"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_custom_provider_from_the_environment_is_configured(monkeypatch):
    from continuum.config import settings
    from continuum.session.providers import PROVIDER_REGISTRY, register_provider
    from continuum.session.providers.memory import MemorySessionProvider

    class MyStore(MemorySessionProvider):
        pass

    monkeypatch.setattr(settings, "session_provider", "MyStore")
    register_provider("MyStore", MyStore)
    try:
        config = SessionConfig(enabled=True)
        assert config.provider == "MyStore"
        assert config.is_configured() is True
    finally:
        PROVIDER_REGISTRY.pop("MyStore", None)


def _session_settings(monkeypatch, **values):
    from continuum.config import settings

    monkeypatch.setattr(settings, "session_enabled", True)
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)


def _session_issues(found):
    return {e.field: e.message for e in found if e.field.startswith("SESSION_")}


def test_startup_validation_reports_an_unknown_provider(monkeypatch):
    from continuum.core.lifecycle import validate_configuration

    _session_settings(monkeypatch, session_provider="dynamodb")
    errors, _ = validate_configuration()
    message = _session_issues(errors)["SESSION_PROVIDER"]
    assert "dynamodb" in message
    assert all(f"'{name}'" in message for name in ("redis", "postgres", "memory"))

    monkeypatch.setattr("continuum.config.settings.session_enabled", False)
    errors, _ = validate_configuration()
    assert "SESSION_PROVIDER" not in _session_issues(errors)


async def test_first_session_call_reports_an_unknown_provider():
    from continuum.session.client import SessionClient

    client = SessionClient(SessionConfig(provider="dynamodb", enabled=True))
    with pytest.raises(ValueError, match="'postgres'"):
        await client.get_or_create_session(user_id="u1", conversation_id="c1")


def test_startup_validation_postgres_needs_no_redis_host(monkeypatch):
    from continuum.core.lifecycle import validate_configuration

    _session_settings(
        monkeypatch,
        session_provider="postgres",
        session_redis_host="",
        session_postgres_dsn=None,
    )
    errors, warnings = validate_configuration()
    assert _session_issues(errors) == {}
    assert "SESSION_POSTGRES_DSN" in _session_issues(warnings)

    _session_settings(monkeypatch, session_provider="redis", session_redis_host="")
    errors, _ = validate_configuration()
    assert "SESSION_REDIS_HOST" in _session_issues(errors)


def test_startup_log_names_the_postgres_store(monkeypatch):
    from continuum.core import lifecycle

    secret = "s3cret" + "pw"
    config = SessionConfig(
        provider="postgres",
        enabled=True,
        redis_host="localhost",
        postgres_dsn="postgresql://app:" + secret + "@db.internal:5432/appdb",
    )

    class Client:
        is_enabled = True

    Client.config = config
    lines: list[str] = []

    class Logger:
        def info(self, msg, *args):
            lines.append(msg % args if args else msg)

        def debug(self, *a, **k):
            pass

        warning = error = debug

    _session_settings(monkeypatch)
    monkeypatch.setattr(lifecycle, "logger", Logger())
    monkeypatch.setattr("continuum.session.get_global_session_client", lambda: Client())
    lifecycle.OrchestratorLifecycle()._log_service_configurations()
    [line] = [x for x in lines if "Session:" in x]
    assert "postgres host=db.internal database=appdb" in line
    assert "redis" not in line and secret not in line and "app:" not in line


def _unencoded_password_dsns():
    """DSNs whose password holds an unencoded "#", "?" or "/", the fragments that
    must never be shown, and whether asyncpg fails to parse the DSN. Built at run time so no literal credential is in git.
    With "/" asyncpg reads "app:<digits>" as host and port, so it fails to resolve
    rather than to parse; the password must still not be shown."""
    head, tail, digits = "Sec7" + "Frag", "tail" + "99", "8604" + "13"
    return [
        (f"postgresql://app:{head}#{tail}@127.0.0.1:1/appdb", (head, tail), True),
        (f"postgresql://app:{head}?{tail}@127.0.0.1:1/appdb", (head, tail), True),
        (f"postgresql://app:{digits}/{head}@127.0.0.1:1/appdb", (digits, head), False),
    ]


@pytest.mark.parametrize(("dsn", "fragments", "parse_error"), _unencoded_password_dsns())
async def test_unencoded_password_is_not_shown_by_the_session_store(
    monkeypatch, caplog, dsn, fragments, parse_error
):
    import logging

    from continuum.session.client import SessionClient

    caplog.set_level(logging.DEBUG)
    config = SessionConfig(provider="postgres", postgres_dsn=dsn, postgres_timeout_seconds=1.0)

    provider = PostgresSessionProvider(config)
    assert await provider.aping() is False
    with pytest.raises(SessionConnectionError) as raised:
        await provider.get_or_create_session(user_id="u1", conversation_id="c1")

    client = SessionClient(config.model_copy(update={"fallback_mode": "degrade"}))
    assert await client.get_or_create_session(user_id="u1", conversation_id="c1")
    strict = SessionClient(config.model_copy(update={"fallback_mode": "fail"}))
    with pytest.raises(SessionConnectionError) as strict_error:
        await strict.get_or_create_session(user_id="u1", conversation_id="c1")

    shown = " ".join(
        [caplog.text, str(provider.last_probe_error), str(raised.value), str(strict_error.value)]
    )
    assert ("percent-encode" in shown) is parse_error
    assert not [f for f in fragments if f in shown]


@pytest.mark.parametrize(("dsn", "fragments", "parse_error"), _unencoded_password_dsns())
async def test_unencoded_password_is_not_shown_by_health_or_describe(
    monkeypatch, caplog, dsn, fragments, parse_error
):
    import logging

    from continuum.config import settings
    from continuum.connectors.postgres import PostgresConnector
    from continuum.core.health import HealthCheck, HealthStatus

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(settings, "session_enabled", True)
    monkeypatch.setattr(settings, "session_provider", "postgres")
    monkeypatch.setattr(settings, "session_postgres_dsn", dsn)
    monkeypatch.setattr(settings, "session_postgres_timeout_seconds", 1.0)
    result = await HealthCheck().check_postgres()
    assert result.status == HealthStatus.UNHEALTHY

    connector = PostgresConnector(SessionConfig(provider="postgres", postgres_dsn=dsn))
    assert await connector.aping() is False
    described = connector.describe()
    assert (described["host"], described["port"], described["database"]) == (
        "127.0.0.1",
        1,
        "appdb",
    )

    shown = " ".join([caplog.text, result.message, str(result.details), str(described)])
    assert ("percent-encode" in shown) is parse_error
    assert not [f for f in fragments if f in shown]


@pytest.mark.parametrize(("dsn", "fragments", "parse_error"), _unencoded_password_dsns())
def test_unencoded_password_is_not_in_the_startup_log(monkeypatch, dsn, fragments, parse_error):
    from continuum.core import lifecycle

    class Client:
        is_enabled = True
        config = SessionConfig(provider="postgres", enabled=True, postgres_dsn=dsn)

    lines: list[str] = []

    class Logger:
        def info(self, msg, *args):
            lines.append(msg % args if args else msg)

        def debug(self, *a, **k):
            pass

        warning = error = debug

    _session_settings(monkeypatch)
    monkeypatch.setattr(lifecycle, "logger", Logger())
    monkeypatch.setattr("continuum.session.get_global_session_client", lambda: Client())
    lifecycle.OrchestratorLifecycle()._log_service_configurations()
    [line] = [x for x in lines if "Session:" in x]
    assert "postgres host=127.0.0.1 database=appdb" in line
    assert not [f for f in fragments if f in line]
