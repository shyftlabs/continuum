"""`continuum up` MinIO secret bootstrap (D3, Option B).

ensure_minio_secret() must generate a strong MINIO_ROOT_PASSWORD when none is
set, never overwrite a user's strong value, persist it outside the managed
block, and warn (not silently fail) when a weak value is exported in the shell.
"""

from __future__ import annotations

import pytest

from continuum.cli import (
    MANAGED_BEGIN,
    MANAGED_END,
    _is_weak_minio_secret,
    ensure_minio_secret,
)


def _read_pw(env_path) -> str | None:
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("MINIO_ROOT_PASSWORD="):
            return line.split("=", 1)[1].strip()
    return None


@pytest.fixture(autouse=True)
def _no_shell_minio(monkeypatch):
    monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)


class TestIsWeakMinioSecret:
    @pytest.mark.parametrize("v", [None, "", "  ", "miniosecret", "MinioSecret", "CHANGEME_x"])
    def test_weak(self, v):
        assert _is_weak_minio_secret(v) is True

    @pytest.mark.parametrize("v", ["f3a1c0de" * 8, "a-strong-one"])
    def test_strong(self, v):
        assert _is_weak_minio_secret(v) is False


class TestEnsureMinioSecret:
    def test_generates_when_absent(self, tmp_path):
        env = tmp_path / ".env"
        msgs = ensure_minio_secret(env)
        pw = _read_pw(env)
        assert pw and len(pw) == 64 and not _is_weak_minio_secret(pw)
        assert any("Generated" in m for m in msgs)

    def test_replaces_weak_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("MINIO_ROOT_PASSWORD=miniosecret\nOTHER=keep\n")
        ensure_minio_secret(env)
        assert not _is_weak_minio_secret(_read_pw(env))
        assert "OTHER=keep" in env.read_text()  # unrelated content preserved

    def test_respects_user_strong_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("MINIO_ROOT_PASSWORD=my-own-strong-secret-123456\n")
        assert ensure_minio_secret(env) == []
        assert _read_pw(env) == "my-own-strong-secret-123456"

    def test_idempotent(self, tmp_path):
        env = tmp_path / ".env"
        ensure_minio_secret(env)
        first = _read_pw(env)
        assert ensure_minio_secret(env) == []  # second run: already strong
        assert _read_pw(env) == first

    def test_written_outside_managed_block(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(f"{MANAGED_BEGIN}\nFOO=bar\n{MANAGED_END}\n")
        ensure_minio_secret(env)
        text = env.read_text()
        # The generated line must sit outside the managed markers.
        block = text[text.index(MANAGED_BEGIN) : text.index(MANAGED_END)]
        assert "MINIO_ROOT_PASSWORD=" not in block
        assert not _is_weak_minio_secret(_read_pw(env))

    def test_shell_strong_value_left_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIO_ROOT_PASSWORD", "shell-strong-secret-abcdef123456")
        env = tmp_path / ".env"
        assert ensure_minio_secret(env) == []
        assert not env.exists()  # nothing written

    def test_shell_weak_value_warns(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIO_ROOT_PASSWORD", "miniosecret")
        env = tmp_path / ".env"
        msgs = ensure_minio_secret(env)
        assert any(m.startswith("warning:") for m in msgs)
        assert not env.exists()  # can't fix via .env when shell shadows it
