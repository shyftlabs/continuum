"""``continuum`` command-line interface — one command to bring up the infra stack.

Before this existed, users had to locate ``docker-compose.yml`` in the repo and run
it by hand (and it wasn't even shipped in the wheel). Now the compose file is bundled
under ``continuum/infra/`` and this CLI resolves it, picks a profile, and keeps the
project's ``.env`` in sync so the running services match what the SDK expects.

Profiles (nested ``minimal`` ⊂ ``standard`` ⊂ ``full``):

  * ``minimal``  — redis-sdk + qdrant (2 containers): a stateful agent, nothing heavy.
  * ``standard`` — minimal + the Langfuse observability stack.
  * ``full``     — everything, incl. Temporal and Milvus.

Usage::

    continuum up [minimal|standard|full]   # default: minimal
    continuum down [-v]
    continuum status
    continuum logs [SERVICE] [-f]
    continuum config-path
    continuum mcp inspect URL [--write-pins PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

# Stable project name so containers/volumes are identical regardless of where the
# wheel is installed (otherwise compose derives it from the install directory).
PROJECT_NAME = "continuum"

# Commands that operate on an already-running stack. docker compose excludes
# profiled services from these unless their profile is active, so we activate
# every profile to avoid orphaning minimal/standard containers.
_LIFECYCLE_ALL_PROFILES = {"down", "ps", "logs"}

MANAGED_BEGIN = "# >>> continuum managed >>>"
MANAGED_END = "# <<< continuum managed <<<"

_MANAGED_BLOCK_RE = re.compile(
    rf"\n?{re.escape(MANAGED_BEGIN)}.*?{re.escape(MANAGED_END)}\n?",
    re.DOTALL,
)


@dataclass(frozen=True)
class ProfileSpec:
    """A deployment tier: which compose profiles to activate and the env it implies."""

    compose_profiles: list[str]
    # Env keys written to the managed ``.env`` block so the SDK only talks to services
    # this profile actually starts.
    env: dict[str, str] = field(default_factory=dict)


PROFILES: dict[str, ProfileSpec] = {
    "minimal": ProfileSpec(
        compose_profiles=["minimal"],
        env={
            "VECTOR_STORE_PROVIDER": "qdrant",
            "LANGFUSE_ENABLED": "false",
            "TEMPORAL_ENABLED": "false",
        },
    ),
    "standard": ProfileSpec(
        compose_profiles=["standard"],
        env={
            "VECTOR_STORE_PROVIDER": "qdrant",
            "LANGFUSE_ENABLED": "true",
            "TEMPORAL_ENABLED": "false",
        },
    ),
    "full": ProfileSpec(
        compose_profiles=["full"],
        env={
            "VECTOR_STORE_PROVIDER": "milvus",
            "LANGFUSE_ENABLED": "true",
            "TEMPORAL_ENABLED": "true",
        },
    ),
}


# ---------------------------------------------------------------------------
# Bundled compose file
# ---------------------------------------------------------------------------


def compose_file_path() -> Path:
    """Filesystem path to the bundled ``docker-compose.yml`` (inside the wheel)."""
    return Path(str(files("continuum.infra").joinpath("docker-compose.yml")))


# ---------------------------------------------------------------------------
# docker compose argv builder
# ---------------------------------------------------------------------------


def build_compose_command(
    action: str,
    profile: str | None = None,
    *,
    env_file: Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Construct the ``docker compose`` argv for *action* under *profile*.

    Global flags (``-f``/``-p``/``--env-file``/``--profile``) must precede the
    subcommand, so they are assembled first.
    """
    compose_path = compose_file_path()
    argv: list[str] = [
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "--project-directory",
        str(compose_path.parent),
        "-p",
        PROJECT_NAME,
    ]
    if env_file is not None:
        argv += ["--env-file", str(env_file)]
    if profile is not None:
        for name in PROFILES[profile].compose_profiles:
            argv += ["--profile", name]
    elif action in _LIFECYCLE_ALL_PROFILES:
        for name in PROFILES:
            argv += ["--profile", name]

    argv.append(action)
    if action == "up":
        argv.append("-d")
    if extra_args:
        argv += extra_args
    return argv


# ---------------------------------------------------------------------------
# Managed .env writer
# ---------------------------------------------------------------------------


def render_managed_block(profile: str) -> str:
    """Render the delimited ``.env`` block for *profile* (no surrounding newlines)."""
    spec = PROFILES[profile]
    lines = [
        MANAGED_BEGIN,
        f"# Managed by `continuum up {profile}` — edits inside this block are overwritten.",
    ]
    lines += [f"{key}={value}" for key, value in spec.env.items()]
    lines.append(MANAGED_END)
    return "\n".join(lines)


def _user_value(text: str, key: str) -> str | None:
    """Return the value a user assigned to *key* in *text* (outside any block), or None."""
    match = re.search(rf"^\s*{re.escape(key)}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def apply_managed_env(env_path: Path, profile: str) -> list[str]:
    """Write/refresh the managed block in *env_path* for *profile*.

    Idempotent: re-running with the same profile is a no-op; switching profiles
    replaces the block. Content outside the markers is preserved. Returns a list
    of human-readable warnings (e.g. a managed key the user pinned to a different
    value outside the block, which the SDK will read instead).
    """
    text = env_path.read_text() if env_path.exists() else ""
    outside = _MANAGED_BLOCK_RE.sub("", text)

    warnings: list[str] = []
    for key, want in PROFILES[profile].env.items():
        have = _user_value(outside, key)
        if have is not None and have != want:
            warnings.append(
                f"{key} is set to {have!r} outside the managed block; "
                f"the '{profile}' profile expects {want!r}. Your value wins — "
                f"remove it to let the profile manage it."
            )

    block = render_managed_block(profile)
    outside = outside.rstrip("\n")
    new_text = f"{outside}\n\n{block}\n" if outside else f"{block}\n"
    env_path.write_text(new_text)
    return warnings


# ---------------------------------------------------------------------------
# MinIO secret bootstrap
# ---------------------------------------------------------------------------

# Placeholder/weak MinIO passwords the auto-generator should replace.
_WEAK_MINIO_SECRETS = frozenset({"", "miniosecret"})


def _is_weak_minio_secret(value: str | None) -> bool:
    """True if *value* is missing, blank, or a known-weak MinIO placeholder."""
    if value is None:
        return True
    lowered = value.strip().lower()
    return lowered in _WEAK_MINIO_SECRETS or "changeme" in lowered


def ensure_minio_secret(env_path: Path) -> list[str]:
    """Ensure a strong ``MINIO_ROOT_PASSWORD`` exists before compose interpolates it.

    The bundled Langfuse stack makes ``MINIO_ROOT_PASSWORD`` a hard requirement,
    single-sourced across MinIO and Langfuse (D3). To keep the quick-start
    frictionless while never shipping a weak default, generate a strong value
    when the operator hasn't set one — persisted OUTSIDE the managed block so it
    survives, is user-editable, and is never clobbered by ``continuum up``.

    An explicit strong value (shell export or ``.env``) always wins and is left
    untouched. Returns human-readable messages to print (empty when nothing to do).
    """
    # A strong value exported in the shell wins and can't be improved via .env.
    shell = os.environ.get("MINIO_ROOT_PASSWORD")
    if shell is not None and not _is_weak_minio_secret(shell):
        return []
    if shell is not None:  # set but weak — compose uses the shell value over .env
        return [
            "warning: MINIO_ROOT_PASSWORD is set to a weak value in your shell "
            "environment; unset it and re-run — Docker Compose uses the shell "
            "value over .env, so it cannot be secured here."
        ]

    text = env_path.read_text() if env_path.exists() else ""
    outside = _MANAGED_BLOCK_RE.sub("", text)
    existing = _user_value(outside, "MINIO_ROOT_PASSWORD")
    if existing is not None and not _is_weak_minio_secret(existing):
        return []  # user already set a strong one — respect it.

    generated = secrets.token_hex(32)
    if existing is not None:
        # Replace the weak line in place (preserves position outside the block).
        new_text = re.sub(
            r"^\s*MINIO_ROOT_PASSWORD=.*$",
            f"MINIO_ROOT_PASSWORD={generated}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        sep = "" if (text == "" or text.endswith("\n")) else "\n"
        new_text = (
            f"{text}{sep}"
            "# Auto-generated strong MinIO root password (used by MinIO and\n"
            "# Langfuse's S3 client — single source of truth). Edit to rotate,\n"
            "# then recreate the MinIO container (continuum down && continuum up).\n"
            f"MINIO_ROOT_PASSWORD={generated}\n"
        )
    env_path.write_text(new_text)
    return [
        "Generated a strong MINIO_ROOT_PASSWORD and saved it to .env "
        "(no secure MinIO secret was set)."
    ]


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _docker_available() -> tuple[bool, str]:
    """Return (ok, message). False if docker is missing or the daemon is unreachable."""
    if shutil.which("docker") is None:
        return (
            False,
            "Docker is not installed or not on PATH. See https://docs.docker.com/get-docker/",
        )
    probe = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        return False, "Docker is installed but the daemon is not running. Start Docker and retry."
    return True, ""


def _run(argv: list[str]) -> int:
    env = {**os.environ, "COMPOSE_PROJECT_NAME": PROJECT_NAME}
    print(f"$ {' '.join(argv)}")
    return subprocess.run(argv, env=env, check=False).returncode


def _env_file() -> Path | None:
    path = Path.cwd() / ".env"
    return path if path.exists() else None


def _cmd_up(args: argparse.Namespace) -> int:
    ok, message = _docker_available()
    if not ok:
        print(f"error: {message}", file=sys.stderr)
        return 1

    env_path = Path.cwd() / ".env"
    for msg in ensure_minio_secret(env_path):
        stream = sys.stderr if msg.startswith("warning:") else sys.stdout
        print(msg, file=stream)
    warnings = apply_managed_env(env_path, args.profile)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Configured .env for '{args.profile}' profile ({env_path}).")

    return _run(build_compose_command("up", args.profile, env_file=env_path))


def _cmd_down(args: argparse.Namespace) -> int:
    extra = ["-v"] if args.volumes else []
    return _run(build_compose_command("down", env_file=_env_file(), extra_args=extra))


def _cmd_status(_: argparse.Namespace) -> int:
    return _run(build_compose_command("ps", env_file=_env_file()))


def _cmd_logs(args: argparse.Namespace) -> int:
    extra = (["-f"] if args.follow else []) + ([args.service] if args.service else [])
    return _run(build_compose_command("logs", env_file=_env_file(), extra_args=extra))


def _cmd_config_path(_: argparse.Namespace) -> int:
    print(compose_file_path())
    return 0


def _cmd_mcp_inspect(args: argparse.Namespace) -> int:
    """Print a remote MCP server's tool catalogue for human review.

    Exists because "read the descriptions before you trust a server" was advice
    nobody could act on: there was no way to see what a server ships without
    hand-writing an async script (security finding F3).

    The pin file is a *byproduct* of reviewing, written only when asked. A command
    that emitted pins without showing the contents would make pin-without-reading
    the easy path.
    """
    import asyncio
    import json

    from continuum.tools.mcp import MCPServerStreamableHttp
    from continuum.tools.pinning import format_tool_catalog, snapshot_tool_digests

    async def _run() -> int:
        server = MCPServerStreamableHttp(
            params={"url": args.url},
            name=args.name or args.url,
            cache_tools_list=False,
        )
        try:
            await server.connect()
            # Bypass list_tools() so the catalogue is shown exactly as sent --
            # unfiltered and, crucially, with invisible characters intact so they
            # can be reported rather than silently cleaned.
            assert server.session is not None
            result = await server.session.list_tools()
            tools = result.tools
        except Exception as e:  # noqa: BLE001 - surface any connection failure plainly
            print(f"Could not inspect {args.url}: {e}", file=sys.stderr)
            return 1
        finally:
            await server.cleanup()

        print(format_tool_catalog(server.name, tools))

        if args.write_pins:
            path = Path(args.write_pins)
            existing: dict[str, Any] = {}
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except ValueError:
                    print(f"Overwriting unparseable pin file {path}.", file=sys.stderr)
            existing[server.name] = snapshot_tool_digests(server.name, tools)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nPinned {len(tools)} tool(s) for '{server.name}' to {path}.")
            print("Pass tool_pin_path= to the MCPServer to have drift reported at runtime.")

        return 0

    return asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continuum",
        description="Continuum infrastructure orchestration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Start infra containers (default profile: minimal).")
    up.add_argument(
        "profile",
        nargs="?",
        default="minimal",
        choices=list(PROFILES),
        help="Deployment tier to start.",
    )
    up.set_defaults(func=_cmd_up)

    down = sub.add_parser("down", help="Stop and remove infra containers.")
    down.add_argument(
        "-v", "--volumes", action="store_true", help="Also remove named volumes (data loss)."
    )
    down.set_defaults(func=_cmd_down)

    status = sub.add_parser("status", help="Show container status (docker compose ps).")
    status.set_defaults(func=_cmd_status)

    logs = sub.add_parser("logs", help="Show container logs.")
    logs.add_argument("service", nargs="?", help="Limit to one service.")
    logs.add_argument("-f", "--follow", action="store_true", help="Follow log output.")
    logs.set_defaults(func=_cmd_logs)

    config_path = sub.add_parser(
        "config-path", help="Print the path to the bundled docker-compose.yml."
    )
    config_path.set_defaults(func=_cmd_config_path)

    mcp = sub.add_parser("mcp", help="Inspect MCP servers before trusting them.")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    inspect_cmd = mcp_sub.add_parser(
        "inspect",
        help="Print a server's tool descriptions and schemas for review.",
        description=(
            "Print every tool description and parameter description a remote MCP "
            "server reports. This text reaches the model's prompt verbatim and "
            "instructs it, so read it before connecting an agent to the server. "
            "Hidden/invisible characters are flagged rather than removed."
        ),
    )
    inspect_cmd.add_argument("url", help="Streamable-HTTP MCP endpoint, e.g. http://host:8931/mcp")
    inspect_cmd.add_argument("--name", help="Name to record for this server (defaults to the URL).")
    inspect_cmd.add_argument(
        "--write-pins",
        metavar="PATH",
        help=(
            "After printing, record each tool's digest to PATH. Pass the same path as "
            "tool_pin_path= to the MCPServer to have later changes reported."
        ),
    )
    inspect_cmd.set_defaults(func=_cmd_mcp_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
