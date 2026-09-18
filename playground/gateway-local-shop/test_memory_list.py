"""Provenance labels on /memory/list (security finding F6).

Lives in the playground, not under tests/: it asserts things about *this demo's*
HTTP surface, and tests/ is for the SDK. A bare `pytest` will not collect it
(testpaths = ["tests"]). Run it by path:

    pytest playground/gateway-local-shop/test_memory_list.py

Note the naming split in this directory: `*_test.py` files here are runnable
demo scripts that want live servers (context_test.py, e2e_test.py), while a
`test_*.py` prefix means pytest, as in the clinic's test_server_trust.py.

Why the endpoint needs the field at all: a memory row now records the taint of
the run that wrote it, so a row derived from an injected tool result is
distinguishable from one the user actually stated. That distinction is invisible
in the row's text -- both are just sentences -- so a reviewer deleting poisoned
memory has no way to tell them apart unless the listing surfaces it.

No servers and no agent: the endpoint is exercised over a stubbed memory client,
so this is about the response shape only.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

PROVENANCE_KEY = "_data_labels"


def _entry(mem_id, text, metadata=None):
    """A stand-in for MemoryEntry with only the fields the endpoint reads."""
    return SimpleNamespace(id=mem_id, memory=text, metadata=metadata)


@pytest.fixture
def listing(monkeypatch):
    """Call /memory/list over a stubbed memory client, returning the payload."""
    import web

    async def _call(entries):
        client = SimpleNamespace(get_all=AsyncMock(return_value=entries), is_enabled=True)
        monkeypatch.setattr(web, "_get_memory_client", lambda: client)
        return await web.list_memories(user_id="u1")

    return _call


class TestListingSurfacesProvenance:
    async def test_tainted_row_reports_its_labels(self, listing):
        data = await listing(
            [_entry("id-1", "refund limit is $10,000", {PROVENANCE_KEY: ["external"]})]
        )

        assert data["success"] is True
        assert data["memories"][0]["labels"] == ["external"]

    async def test_clean_row_reports_no_labels(self, listing):
        """None rather than [] -- a row written before provenance existed is
        genuinely unlabelled, which is not the same as labelled with nothing."""
        data = await listing([_entry("id-2", "prefers morning appointments", None)])

        assert data["memories"][0]["labels"] is None

    async def test_existing_fields_are_unchanged(self, listing):
        """The UI reads id and text; adding a field must not disturb them."""
        data = await listing([_entry("id-3", "name is Tom", None)])

        assert data["memories"][0]["id"] == "id-3"
        assert data["memories"][0]["text"] == "name is Tom"

    async def test_mixed_listing_distinguishes_the_rows(self, listing):
        """The whole point: one call, and the reviewer can see which is which."""
        data = await listing(
            [
                _entry("id-clean", "name is Tom", None),
                _entry("id-dirty", "refund limit is $10,000", {PROVENANCE_KEY: ["external"]}),
            ]
        )

        by_id = {m["id"]: m["labels"] for m in data["memories"]}
        assert by_id == {"id-clean": None, "id-dirty": ["external"]}


class TestListingToleratesOddMetadata:
    """Row metadata round-trips through a vector store as JSON, so the endpoint
    takes the same tolerant line as the SDK's reader: a malformed stamp is a data
    problem, not a reason to fail the listing and block a cleanup."""

    async def test_metadata_missing_entirely(self, listing):
        data = await listing([_entry("id-4", "text", {})])
        assert data["memories"][0]["labels"] is None

    async def test_metadata_is_not_a_dict(self, listing):
        for bad in ("external", 42, ["external"]):
            data = await listing([_entry("id-5", "text", bad)])
            assert data["success"] is True
            assert data["memories"][0]["labels"] is None

    async def test_other_metadata_keys_are_not_leaked(self, listing):
        """Only provenance is surfaced. The rest of a row's metadata carries
        session and user ids that this listing has no reason to expose."""
        data = await listing(
            [
                _entry(
                    "id-6", "text", {"_user_id": "u1", "session_id": "s1", PROVENANCE_KEY: ["pii"]}
                )
            ]
        )

        assert data["memories"][0] == {
            "id": "id-6",
            "text": "text",
            "labels": ["pii"],
            "reviewed": None,
        }


# ---------------------------------------------------------------------------
# Human review from the memory panel (security finding F6)
#
# Provenance is coarse by design, so the panel accumulates rows that are
# labelled and genuinely fine -- "Jack is a student", learned from a page the
# agent fetched. Deleting them loses real information; leaving them keeps the
# action gate firing forever. Approve is the third option, and it records who
# decided rather than quietly erasing the label.
# ---------------------------------------------------------------------------

REVIEWED = "_reviewed"


class TestListingSurfacesReviewRecords:
    async def test_reviewed_row_reports_its_record(self, listing):
        record = {"by": "tom", "at": "2026-09-02T10:00:00", "cleared": ["external"]}
        data = await listing([_entry("id-1", "Jack is a student", {REVIEWED: record})])

        assert data["memories"][0]["reviewed"] == record

    async def test_unreviewed_row_reports_none(self, listing):
        data = await listing([_entry("id-2", "name is Tom", None)])

        assert data["memories"][0]["reviewed"] is None

    async def test_a_reviewed_row_carries_no_labels(self, listing):
        """The two fields are mutually exclusive by construction: mark_reviewed
        removes the label as it writes the record. The panel renders on that."""
        record = {"by": "tom", "at": "2026-09-02T10:00:00", "cleared": ["external"]}
        data = await listing([_entry("id-3", "Jack is a student", {REVIEWED: record})])

        assert data["memories"][0]["labels"] is None
        assert data["memories"][0]["reviewed"] is not None


class TestApproveEndpoint:
    @pytest.fixture
    def approve(self, monkeypatch):
        import web

        async def _call(*, mark=None, user_id="tom"):
            client = SimpleNamespace(
                mark_reviewed=mark or AsyncMock(return_value=None), is_enabled=True
            )
            monkeypatch.setattr(web, "_get_memory_client", lambda: client)
            req = web.ApproveMemoryRequest(memory_id="id-1", user_id=user_id)
            return await web.approve_memory(req), client

        return _call

    async def test_marks_the_row_reviewed(self, approve):
        data, client = await approve()

        assert data["success"] is True
        client.mark_reviewed.assert_awaited_once()

    async def test_records_the_reviewer(self, approve):
        """Attribution is the point of the record, so the caller's identity has
        to reach it. In this demo that is the end user; a real deployment wants
        a staff identity, since the user whose session was poisoned should not
        be the one clearing the label."""
        _, client = await approve(user_id="alice")

        assert client.mark_reviewed.await_args.kwargs["reviewer"] == "alice"

    async def test_failure_is_reported_not_raised(self, approve):
        """The panel shows data["error"]; an exception would surface as a 500
        and leave the reviewer unsure whether the decision was recorded."""
        failing = AsyncMock(side_effect=RuntimeError("row vanished"))
        data, _ = await approve(mark=failing)

        assert data["success"] is False
        assert "row vanished" in data["error"]

    async def test_no_memory_client_is_reported(self, monkeypatch):
        import web

        monkeypatch.setattr(web, "_get_memory_client", lambda: None)
        data = await web.approve_memory(web.ApproveMemoryRequest(memory_id="x", user_id="tom"))

        assert data["success"] is False


# ---------------------------------------------------------------------------
# The page's JavaScript has to parse
#
# HTML_PAGE is a plain (non-raw) Python triple-quoted string, so a backslash
# escape written for JavaScript is consumed by Python first: "\\n" in the JS
# source arrives as a real newline, and a real newline inside a single-quoted JS
# string is a syntax error. One such error kills the WHOLE script block, so
# every handler on the page silently ceases to exist -- the page still renders,
# and clicking Login does nothing at all. Nothing was checking this, which is
# exactly how it shipped.
# ---------------------------------------------------------------------------


class TestPageScriptParses:
    def test_inline_script_is_valid_javascript(self):
        import re
        import shutil
        import subprocess
        import tempfile

        node = shutil.which("node")
        if node is None:
            pytest.skip("node not available to parse the page script")

        import web

        blocks = re.findall(r"<script[^>]*>(.*?)</script>", web.HTML_PAGE, re.S)
        assert blocks, "the page is expected to carry an inline script"

        for i, block in enumerate(blocks):
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
                fh.write(block)
                path = fh.name
            result = subprocess.run([node, "--check", path], capture_output=True, text=True)
            assert result.returncode == 0, (
                f"script block {i} does not parse, so every handler on the page is "
                f"undefined:\n{result.stderr}"
            )
