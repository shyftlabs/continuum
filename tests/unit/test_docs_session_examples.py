"""The session docs must run.

``docs/session.md`` shipped an example calling ``get_or_create_session`` with an
``agent_id`` argument that method has never accepted, and every one of its
examples began raising the moment session ownership was enforced. Prose drifts
from code silently; these tests make it fail loudly instead.

Two kinds of check:

  - the documented flows are executed here, so an API change breaks a test
    rather than only a reader's afternoon
  - the documented *signatures* are compared against the real ones, which is
    what the ``agent_id`` example needed and did not have
"""

from __future__ import annotations

import inspect
import re
import textwrap
from pathlib import Path

import pytest

from continuum.llm.types import ChatMessage
from continuum.session import SessionClient, SessionConfig, bind_principal
from continuum.session.exceptions import SessionOwnershipError
from continuum.session.providers.memory import MemorySessionProvider

DOC = Path(__file__).resolve().parents[2] / "docs" / "session.md"


def _client(*, require_principal: bool = False) -> SessionClient:
    """A client whose ownership posture is stated, not inherited.

    ``require_principal`` used to be left unset here, so it fell through to
    ``settings.session_require_principal`` -- the environment. That made the
    refusal test below pass on a machine whose .env set it true and fail in CI,
    which ships it false. The test was asserting a property of the author's
    environment.

    Both values are exercised: the anonymous and history flows document what
    happens on the shipped default, and the refusal documents what happens once
    an operator turns it on.
    """
    cfg = SessionConfig(
        enabled=True,
        provider="memory",
        hash_session_ids=False,
        require_principal=require_principal,
    )
    client = SessionClient(session_config=cfg, memory_client=None, auto_initialize=False)
    client.set_provider(MemorySessionProvider(cfg))
    client._initialized = True
    return client


# ── the documented flows actually run ─────────────────────────────────────────


@pytest.mark.asyncio
class TestDocumentedFlows:
    async def test_quick_start(self):
        """§1. The shape a first-time reader copies."""
        client = _client()

        with bind_principal("user-123"):
            sid = await client.get_or_create_session(user_id="user-123")
            await client.add_message(
                sid, ChatMessage(role="user", content="Hello"), store_in_memory=False
            )
            await client.add_message(
                sid, ChatMessage(role="assistant", content="Hi!"), store_in_memory=False
            )
            history = await client.get_conversation_history(sid)

        assert [m.content for m in history] == ["Hello", "Hi!"]

    async def test_resume_an_existing_conversation(self):
        """§8. The same identifiers must resolve back to the same session."""
        client = _client()

        with bind_principal("u1"):
            first = await client.get_or_create_session(user_id="u1", conversation_id="c1")
            await client.add_message(
                first, ChatMessage(role="user", content="Hi"), store_in_memory=False
            )
            again = await client.get_or_create_session(user_id="u1", conversation_id="c1")
            history = await client.get_conversation_history(again, limit=20)

        assert again == first
        assert [m.content for m in history] == ["Hi"]

    async def test_manual_cleanup(self):
        """§8. Deleting is a gated operation too."""
        client = _client()

        with bind_principal("u1"):
            sid = await client.get_or_create_session(user_id="u1")
            assert await client.delete_session(sid) is True
            assert await client.get_session_metadata(sid) is None

    async def test_an_anonymous_session_needs_no_principal(self):
        """A session created without a user_id has no owner, so single-user and
        demo code keeps working exactly as the older docs described."""
        client = _client()

        sid = await client.get_or_create_session()
        await client.add_message(
            sid, ChatMessage(role="user", content="Hello"), store_in_memory=False
        )
        assert len(await client.get_conversation_history(sid)) == 1

    async def test_the_documented_refusal(self):
        """The doc has to show what going wrong looks like, so the error it
        names must be the error that is raised.

        Needs ``require_principal=True`` explicitly: the refusal being
        documented is the one an operator opts into. On shipped defaults an
        unbound caller is allowed through, so leaving this ambient asserted
        nothing on a default install.
        """
        client = _client(require_principal=True)

        with bind_principal("u1"):
            sid = await client.get_or_create_session(user_id="u1")

        with pytest.raises(SessionOwnershipError):
            await client.get_conversation_history(sid)


# ── the documented signatures match the real ones ─────────────────────────────


class TestDocumentedSignatures:
    """``docs/session.md`` documented ``get_or_create_session(..., agent_id=None)``
    for a method that takes ``conversation_id``. Nothing caught it."""

    @staticmethod
    def _doc() -> str:
        return DOC.read_text()

    @pytest.mark.parametrize(
        "method",
        [
            "get_or_create_session",
            "add_message",
            "get_conversation_history",
            "clear_session",
            "delete_session",
            "get_session_metadata",
        ],
    )
    def test_every_documented_parameter_exists(self, method):
        """Each name in the doc's API-table signature must be a real parameter."""
        doc = self._doc()
        match = re.search(rf"`{method}\((.*?)\)`", doc, re.S)
        assert match, f"{method} is not documented in the API table"

        real = set(inspect.signature(getattr(SessionClient, method)).parameters)
        documented = {
            p.split("=")[0].split(":")[0].strip().lstrip("*").strip()
            for p in match.group(1).split(",")
        }
        documented -= {"", "self"}

        assert documented <= real, (
            f"docs/session.md documents parameters {method} does not accept: "
            f"{sorted(documented - real)}"
        )


class TestTheDocCoversOwnership:
    """A reader who never reaches the ownership section will write code that
    raises on its third line."""

    @staticmethod
    def _doc() -> str:
        return DOC.read_text()

    def test_bind_principal_is_documented(self):
        assert "bind_principal" in self._doc()

    def test_the_exception_is_listed(self):
        assert "SessionOwnershipError" in self._doc()

    def test_the_settings_are_documented(self):
        doc = self._doc()
        for var in (
            "SESSION_OWNERSHIP",
            "SESSION_REQUIRE_PRINCIPAL",
            "SESSION_HASH_IDS",
            "SESSION_ID_SECRET",
        ):
            assert var in doc, f"{var} is not documented"

    def test_the_migration_path_is_documented(self):
        """The breaking change is only survivable if the way down is written
        somewhere a reader will find it."""
        doc = self._doc()
        assert "audit" in doc and "require_principal" in doc.replace("REQUIRE_PRINCIPAL", "")


# ── the landing page's sample ─────────────────────────────────────────────────

INDEX = Path(__file__).resolve().parents[2] / "docs" / "index.html"


def _index_session_sample() -> str:
    """The rendered text of index.html's Sessions code block.

    That page is hand-written HTML with syntax-highlighting spans wrapped around
    every token, so its samples are invisible to any check that reads the repo's
    markdown. Stripping the tags brings it back into reach.
    """
    import html as html_mod

    block = re.search(
        r"<pre>(<span class=\"kw\">from</span> continuum\.session <span class=\"kw\">import</span> "
        r"<span class=\"cls\">SessionClient</span>.*?)</pre>",
        INDEX.read_text(),
        re.S,
    )
    assert block, "the Sessions code block is no longer where this test expects it"
    return html_mod.unescape(re.sub(r"<[^>]+>", "", block.group(1)))


def _index_code_blocks() -> list[str]:
    """Every code sample on the landing page, with its highlighting stripped."""
    import html as html_mod

    return [
        html_mod.unescape(re.sub(r"<[^>]+>", "", b))
        for b in re.findall(r"<pre>(.*?)</pre>", INDEX.read_text(), re.S)
    ]


class TestLandingPageOwnershipSamples:
    """The page teaches the create-then-run pattern in three places. Each one
    creates a session *with* a ``user_id`` — an owned session — so each needs a
    principal or it hands the reader code that raises."""

    def test_every_owned_session_sample_binds_a_principal(self):
        offenders = [
            b.splitlines()[0][:60]
            for b in _index_code_blocks()
            if "get_or_create_session(" in b and "user_id=" in b and "bind_principal" not in b
        ]
        assert offenders == [], (
            "docs/index.html samples create an owned session without binding a "
            f"principal: {offenders}"
        )

    def test_every_sample_that_binds_one_is_valid_python(self):
        import ast

        for block in _index_code_blocks():
            if "bind_principal" in block:
                ast.parse(block)

    def test_save_turn_carries_its_ownership_warning(self):
        """``save_turn()`` writes through the same gate as any other session
        write, so a workflow agent hits it too. The warning has to sit in the
        section that teaches the pattern — the page mentions ``save_turn`` in
        three places, and a check anchored on the first one passes for the
        wrong reason."""
        page = INDEX.read_text()
        start = page.index('id="run-session-multi"')
        end = page.index('id="run-final"')
        section = page[start:end]

        assert "save_turn" in section
        assert "SessionOwnershipError" in section


class TestLandingPageSample:
    def test_it_is_valid_python(self):
        import ast

        ast.parse(_index_session_sample())

    def test_it_binds_a_principal(self):
        """Every call in that sample but the first is ownership-checked, so
        without this the page hands a reader four lines, three of which raise."""
        assert "bind_principal" in _index_session_sample()

    def test_it_documents_the_settings(self):
        page = INDEX.read_text()
        for var in ("SESSION_OWNERSHIP", "SESSION_HASH_IDS", "SESSION_ID_SECRET"):
            assert var in page, f"{var} is not mentioned on the landing page"


@pytest.mark.asyncio
class TestLandingPageSampleRuns:
    async def test_the_sample_executes(self):
        """Run the page's own code, not a paraphrase of it."""
        client = _client()
        ns = {"SessionClient": lambda: client, "bind_principal": bind_principal}

        source = _index_session_sample().replace(
            "from continuum.session import SessionClient, bind_principal", ""
        )
        exec(
            compile(f"async def _sample():\n{textwrap.indent(source, '    ')}", "<index>", "exec"),
            ns,
        )
        await ns["_sample"]()  # must not raise
