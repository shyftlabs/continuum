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
