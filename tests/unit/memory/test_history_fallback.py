"""Secure fallback for the memory-history DB when HOME is unwritable (S5).

When the configured history path is under ``/nonexistent`` (a Docker appuser
container's home), the DB must land in a private, per-uid ``0700`` directory
under the system temp dir — never a fixed, world-writable ``/tmp`` path that a
co-tenant could pre-seed with a symlink (Bandit B108 / CWE-377). The directory
must also be stable across runs so history persists.
"""

from __future__ import annotations

import os
import stat

import pytest

from continuum.memory.config import _secure_fallback_history_dir
from continuum.memory.exceptions import MemoryConfigurationError


@pytest.fixture(autouse=True)
def _tmp_as_tempdir(tmp_path, monkeypatch):
    # Point the system temp dir at a pytest tmp so we never touch the real /tmp.
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    return tmp_path


class TestSecureFallbackHistoryDir:
    def test_creates_private_0700_dir_owned_by_us(self, _tmp_as_tempdir):
        d = _secure_fallback_history_dir()
        assert d == _tmp_as_tempdir / f"continuum-{os.getuid()}"
        info = d.lstat()
        assert stat.S_ISDIR(info.st_mode)
        assert not stat.S_ISLNK(info.st_mode)
        assert info.st_uid == os.getuid()
        assert stat.S_IMODE(info.st_mode) == 0o700  # no group/other access

    def test_stable_across_calls_for_persistence(self):
        # Same path each run — history is not lost on restart.
        assert _secure_fallback_history_dir() == _secure_fallback_history_dir()

    def test_rejects_symlink_at_target(self, _tmp_as_tempdir, tmp_path):
        target = _tmp_as_tempdir / f"continuum-{os.getuid()}"
        (tmp_path / "elsewhere").mkdir()
        target.symlink_to(tmp_path / "elsewhere")  # attacker-planted redirect
        with pytest.raises(MemoryConfigurationError, match="insecure"):
            _secure_fallback_history_dir()

    def test_rejects_group_or_other_accessible_dir(self, _tmp_as_tempdir):
        target = _tmp_as_tempdir / f"continuum-{os.getuid()}"
        target.mkdir(mode=0o755)  # world-readable/executable — not private
        os.chmod(target, 0o755)  # mkdir mode is umask-masked; force it
        with pytest.raises(MemoryConfigurationError, match="insecure"):
            _secure_fallback_history_dir()

    def test_rejects_dir_not_owned_by_us(self, _tmp_as_tempdir, monkeypatch):
        # Create it as the real uid, then pretend to be a different uid so the
        # ownership check fails (simulates a dir planted by another user).
        _secure_fallback_history_dir()
        real = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real + 1)
        with pytest.raises(MemoryConfigurationError, match="insecure"):
            _secure_fallback_history_dir()
