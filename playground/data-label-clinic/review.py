#!/usr/bin/env python3
"""
Read both MCP servers' tool catalogues before trusting them.

    python review.py                         # print both catalogues
    python review.py --write-pins            # ...and approve them (agent's path)
    python review.py --write-pins PATH       # ...and approve them elsewhere

`continuum mcp inspect` can review the *clinic*, which needs no credentials.
It cannot review the *pharmacy*: that command sends a bare URL, and the pharmacy
requires a bearer token, so it gets a 401 however correct the URL is. Try it --

    continuum mcp inspect http://localhost:8912/mcp --name pharmacy

`review_server` takes the server *object*, which already carries the header, so
headers, transport, env and cwd are whatever agent.py configured rather than
whatever you retyped. It is also the only route for a stdio server, which has
no URL at all.

Reading and approving are separate acts on purpose. This script prints and
stops; `--write-pins` exists for approving before the agent has ever run. The
ordinary path is to read this output and then:

    continuum mcp approve clinic   --pins tool-trust/tool-pins.json --all
    continuum mcp approve pharmacy --pins tool-trust/tool-pins.json --all

which works straight after a refusal, because the runtime records what it was
served on the fetch it then refused.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# The SAME factory the agent uses -- not a copy. Re-specifying the servers here
# is the failure this whole feature is about: a header omitted or a URL retyped
# and you review one server while the agent runs another, then write a pin file
# that vouches for something nobody read.
from agent import build_mcp_servers, server_address
from config import default_config

from continuum.tools import review_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """--write-pins takes an optional PATH, matching `continuum mcp inspect`.

    It used to be a bare boolean, so `--write-pins /tmp/other.json` wrote to the
    configured path and discarded the argument. Same flag name as the CLI's,
    different meaning -- anyone who learned the CLI first would type a path and
    have it dropped.
    """
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--write-pins",
        nargs="?",
        const=default_config.tool_pin_path,
        default=None,
        metavar="PATH",
        help=(
            "Record the catalogues as approved. Defaults to the path the agent "
            f"reads ({default_config.tool_pin_path}); pass one to override."
        ),
    )
    return parser.parse_args(argv)


async def main(write_pins: str | None) -> None:
    if write_pins is not None and os.path.abspath(write_pins) != os.path.abspath(
        default_config.tool_pin_path
    ):
        # Approving into a file the agent never opens approves nothing: it will
        # refuse the same servers on the next run while a pin file sits there
        # looking complete. Worth a word rather than a silent no-op.
        print(
            f"Note: writing to {write_pins}, but the agent reads "
            f"{default_config.tool_pin_path}. Change config.ClinicConfig."
            f"tool_pin_path if you meant to move it.\n"
        )
    for server in build_mcp_servers():
        try:
            await review_server(server, write_pins=write_pins)
        except Exception as e:  # noqa: BLE001 - one unreachable server must not hide the other
            print(f"\nCould not review '{server.name}' at {server_address(server)}: {e}\n")
        print()


if __name__ == "__main__":
    asyncio.run(main(parse_args().write_pins))
