"""docs/framer/ContinuumDocs.tsx is generated from docs/index.html.

Nothing enforced that. Editing the HTML and forgetting to re-run the generator
leaves the published Framer component describing an older version of the docs,
and the only way anyone found out was by noticing — which has already had to
happen several times.

The generator stamps a sha256 of the source it read into the output header:

    // source-sha256: 5bd363b4e164711f…

so the check is a hash comparison, not a rebuild. No subprocess, no temporary
directory, no moving the committed file aside and hoping to restore it if the
assertion raises. It runs in about a millisecond and cannot leave the working
tree dirty, which the rebuild-and-compare version could if it failed partway.

What it does NOT verify is that the generator's *output* is correct — only that
it was run against the current source. That is the failure mode worth catching:
nobody edits ContinuumDocs.tsx by hand (the header says not to), they forget to
regenerate it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "index.html"
COMPONENT = ROOT / "docs" / "framer" / "ContinuumDocs.tsx"
GENERATOR = ROOT / "docs" / "framer" / "build_framer.py"

STAMP = re.compile(r"^// source-sha256: ([0-9a-f]{64})$", re.MULTILINE)
REGENERATE = "run `python3 docs/framer/build_framer.py` and commit the result"

pytestmark = pytest.mark.skipif(
    not COMPONENT.exists() or not SOURCE.exists(),
    reason="docs/framer/ContinuumDocs.tsx or docs/index.html is not present",
)


def _stamped_hash() -> str | None:
    # Only the header is scanned. The body is ~288 KB of escaped HTML that could
    # contain anything, including a line that looks like the stamp.
    head = COMPONENT.read_text(encoding="utf-8")[:2000]
    m = STAMP.search(head)
    return m.group(1) if m else None


def _source_hash() -> str:
    return hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def test_the_component_carries_a_source_stamp():
    """A missing stamp reads exactly like a passing comparison.

    Without this, dropping the stamp from the generator would make the test
    below vacuous rather than failing — the same shape as the guard whose
    'found nothing' tripwire could not detect its own scan going stale.
    """
    assert _stamped_hash() is not None, (
        f"{COMPONENT.name} has no `// source-sha256:` header line. Either the "
        f"generator stopped emitting it, or the file was hand-edited — {REGENERATE}."
    )


def test_the_generator_still_emits_the_stamp():
    """The stamp is only meaningful while the generator writes it."""
    assert GENERATOR.exists(), "the generator is missing; the stamp cannot be refreshed"
    assert "source-sha256" in GENERATOR.read_text(encoding="utf-8"), (
        "build_framer.py no longer emits `source-sha256`, so the sync check above "
        "would pass forever against a stale stamp"
    )


def test_the_component_was_generated_from_the_current_html():
    stamped, actual = _stamped_hash(), _source_hash()
    assert stamped == actual, (
        f"docs/index.html has changed since ContinuumDocs.tsx was generated "
        f"(stamped {stamped[:12]}…, source is {actual[:12]}…). The published Framer "
        f"component is describing an older version of the docs — {REGENERATE}."
    )
