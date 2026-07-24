"""Integration tests for the ``continuum`` CLI against a real ``docker compose``.

These assert that each profile resolves to exactly the intended set of services
(via ``docker compose config``, which validates the bundled file without pulling
images or binding host ports) and that the bundled compose + Temporal config are
actually present and self-consistent.

Skipped automatically when Docker is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from continuum import cli

pytestmark = pytest.mark.integration


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


requires_docker = pytest.mark.skipif(not _docker_ok(), reason="Docker not available")


@pytest.fixture(autouse=True)
def _minio_secret_for_compose(monkeypatch):
    # The bundled compose now REQUIRES MINIO_ROOT_PASSWORD (D3). These structural
    # tests don't care about its value, so provide a dummy so interpolation
    # succeeds. Tests that assert the requirement itself override this explicitly.
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "test-minio-secret-for-config")


# Expected service sets per profile — the contract minimal ⊂ standard ⊂ full.
EXPECTED = {
    "minimal": {"qdrant", "redis-sdk"},
    "standard": {
        "qdrant",
        "redis-sdk",
        "clickhouse",
        "langfuse-web",
        "langfuse-worker",
        "minio",
        "postgres",
        "redis",
    },
    "full": {
        "qdrant",
        "redis-sdk",
        "clickhouse",
        "langfuse-web",
        "langfuse-worker",
        "minio",
        "postgres",
        "redis",
        "milvus",
        "milvus-etcd",
        "postgres-temporal",
        "temporal",
        "temporal-ui",
    },
}


def _services_for(profile: str) -> set[str]:
    compose = cli.compose_file_path()
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose), "--profile", profile, "config", "--services"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


@requires_docker
@pytest.mark.parametrize("profile", ["minimal", "standard", "full"])
def test_profile_resolves_to_expected_services(profile: str) -> None:
    assert _services_for(profile) == EXPECTED[profile]


@requires_docker
def test_profiles_are_strictly_nested() -> None:
    minimal = _services_for("minimal")
    standard = _services_for("standard")
    full = _services_for("full")
    assert minimal < standard < full


@requires_docker
def test_bundled_compose_is_valid() -> None:
    # `config -q` validates the whole file (syntax, interpolation, volume refs).
    compose = cli.compose_file_path()
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose), "--profile", "full", "config", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@requires_docker
def test_published_ports_are_overridable_via_env() -> None:
    # Multi-project machines collide on default ports; the host ports must be
    # parameterized so a user can remap them without editing the bundled file.
    compose = cli.compose_file_path()
    env = {**os.environ, "QDRANT_PORT": "6399", "SESSION_REDIS_PORT": "6390"}
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose), "--profile", "minimal", "config"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    published = {line.split('"')[1] for line in result.stdout.splitlines() if "published:" in line}
    assert "6399" in published  # qdrant REST remapped
    assert "6390" in published  # session redis remapped
    assert "6333" not in published  # default no longer used


@requires_docker
def test_minio_password_required_and_single_sourced() -> None:
    # D3: MINIO_ROOT_PASSWORD is required (compose refuses without it), and the
    # one value feeds MinIO's server password AND every Langfuse S3 client key.
    compose = cli.compose_file_path()
    base = ["docker", "compose", "-f", str(compose), "--profile", "full", "config"]

    # Unset -> compose refuses (fail-closed / required).
    env_no = {k: v for k, v in os.environ.items() if k != "MINIO_ROOT_PASSWORD"}
    refused = subprocess.run([*base, "-q"], capture_output=True, text=True, env=env_no)
    assert refused.returncode != 0
    assert "MINIO_ROOT_PASSWORD" in refused.stderr

    # Set -> every Langfuse S3 secret key resolves to the single MinIO value.
    env_yes = {**os.environ, "MINIO_ROOT_PASSWORD": "single-source-check-value"}
    ok = subprocess.run(base, capture_output=True, text=True, check=True, env=env_yes)
    secret_lines = [ln for ln in ok.stdout.splitlines() if "SECRET_ACCESS_KEY:" in ln]
    assert secret_lines  # sanity: we found the client keys
    assert all("single-source-check-value" in ln for ln in secret_lines)


def test_qdrant_and_temporal_healthchecks_avoid_missing_tools() -> None:
    # Regression guard for the two healthcheck bugs we fixed: qdrant's image has no
    # curl/wget (use bash /dev/tcp), and temporal must bind 0.0.0.0 so the localhost
    # healthcheck can reach it.
    text = cli.compose_file_path().read_text()
    assert "curl" not in text.split("qdrant:")[1].split("milvus-etcd:")[0]
    assert "BIND_ON_IP=0.0.0.0" in text


def test_temporal_dynamic_config_is_bundled_alongside_compose() -> None:
    # The compose mounts ./temporal/dynamicconfig; it must ship next to the file.
    compose = cli.compose_file_path()
    cfg = compose.parent / "temporal" / "dynamicconfig" / "development-sql.yaml"
    assert cfg.is_file()
