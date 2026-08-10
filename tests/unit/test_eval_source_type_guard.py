"""source_type allowlist guard for the evaluation SQL scripts (S3).

The eval/dataset-generation scripts interpolate ``source_type`` into pgvector
queries run via ``docker exec psql`` (no driver-side parameter binding). The
value must therefore be validated against a fixed allowlist so untrusted input
can never reach the SQL string — eliminating the f-string injection class by
construction (Bandit B608).
"""

from __future__ import annotations

import pytest

from continuum.evaluation import build_golden_dataset as bgd
from continuum.evaluation import generate_eval_dataset as ged


@pytest.mark.parametrize("mod", [bgd, ged], ids=["build_golden_dataset", "generate_eval_dataset"])
class TestSourceTypeGuard:
    def test_known_types_pass_through(self, mod):
        for st in mod.ALLOWED_SOURCE_TYPES:
            assert mod._require_known_source_type(st) == st

    @pytest.mark.parametrize(
        "hostile",
        [
            "irc'; DROP TABLE tax_law_chunks; --",
            "irc' OR '1'='1",
            "unknown_source",
            "",
            "IRC",  # case-sensitive: not in the allowlist
        ],
    )
    def test_hostile_or_unknown_rejected(self, mod, hostile):
        with pytest.raises(ValueError, match="Unknown source_type"):
            mod._require_known_source_type(hostile)

    def test_allowlist_is_nonempty_frozenset(self, mod):
        assert isinstance(mod.ALLOWED_SOURCE_TYPES, frozenset)
        assert mod.ALLOWED_SOURCE_TYPES  # not empty


def test_build_golden_allowlist_matches_sample_targets():
    # The build script derives its allowlist from the sampling targets, so the
    # two can never drift apart.
    assert bgd.ALLOWED_SOURCE_TYPES == frozenset(bgd.SAMPLE_TARGETS)
