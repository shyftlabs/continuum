"""How this demo's workflows are configured.

Lives beside the demo rather than under tests/, like
gateway-local-shop/test_agent_logging.py: it asserts something about *this
playground's* code, and a bare `pytest` does not collect it (testpaths =
["tests"]). Run it by path:

    pytest playground/gateway-multi-agent-shop/test_workflows.py

The naming split in this directory: `*_test.py` files are runnable demo scripts
that want live servers (headroom_multiagent_test.py); a `test_*.py` prefix means
pytest.

Run one playground's tests at a time. This directory and gateway-local-shop both
have bare modules named config, cli, server and web, and neither can be a
package (the hyphens). pytest's default prepend import mode puts each test file's
directory at sys.path[0] at *collection* time, so a single command naming both
playgrounds' test files resolves `config` to whichever was collected last, and
the other playground's tests import the wrong one. This file's own imports are
scoped (see _this_playground) so its tests pass either way; the other
playground's may not. It cannot be fixed from inside a test file, because it
happens before any test runs.
"""

from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# Modules this playground shares a name with in gateway-local-shop, imported bare
# -- workflows.py does `from config import ...`. In one pytest process that
# collects both playgrounds' tests, a bare import resolves to whichever directory
# is first on sys.path and whichever copy is already in sys.modules, so
# workflows.py gets local-shop's config.py, or local-shop's agent.py gets this one.
_SHARED_NAMES = ("config", "workflows")


@contextlib.contextmanager
def _this_playground():
    """Resolve bare imports to this directory for the duration, then put the
    process back exactly as it was.

    Scoped deliberately. A first version purged the foreign copies and left this
    directory at the front of sys.path, which only moved the collision: every
    local-shop test that ran afterwards imported this playground's config.py and
    failed -- 29 of them.
    """
    saved_path = list(sys.path)
    saved_modules = {n: sys.modules.pop(n) for n in _SHARED_NAMES if n in sys.modules}
    sys.path.insert(0, HERE)
    try:
        yield
    finally:
        for name in _SHARED_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _debate_agent():
    """DebateShop's DebateAgent, built without connecting to anything."""
    with _this_playground():
        from config import default_config
        from workflows import DebateShop

        shop = DebateShop.__new__(DebateShop)
        shop.config = default_config
        shop._build_workflow()
    return shop._agent


class TestDebateShop:
    """A live run showed the judge receiving the first 2000 characters of each
    side -- 49% of a 4056-character argument and 58% of a 3436-character one --
    because this demo used DebateConfig's defaults: summarise_arguments=False,
    truncate_chars=2000. The cut is a plain character slice, so it drops each
    argument's end, which is where a persuasive case lands its conclusion.

    The demo exists to show a debate being judged, so it uses the mode the SDK
    provides for exactly this: each side condenses its own argument into bullet
    points before the judge sees it, keeping what matters rather than whatever
    fits in the first 2000 characters. Two extra LLM calls, run in parallel,
    and only for a side that is actually over the limit.
    """

    def test_each_side_summarises_its_own_argument_for_the_judge(self):
        assert _debate_agent().debate_config.summarise_arguments is True

    def test_the_limit_still_decides_which_sides_need_summarising(self):
        """A side within the limit reaches the judge verbatim, with no extra
        call; only a side over it is summarised. The limit is not disabled."""
        assert _debate_agent().debate_config.truncate_chars == 2000
