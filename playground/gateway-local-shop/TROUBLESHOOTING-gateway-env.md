# Troubleshooting: gateway is used even though `.env` doesn't set it

**Symptom**

You run this playground and the logs show the Smart Gateway even though the
gateway is commented out in the repo-root `.env`:

```
continuum.llm.providers - Smart Gateway routing: model=... url=https://continuum.shyftops.io/v1
```

`unset SMART_GATEWAY_URL` fixes it for one terminal, but a new terminal — or a
Cursor restart — brings it back.

## Root cause (it is NOT the SDK or `.env`)

The Cursor/VS Code extension **`ms-python.vscode-python-envs`** ("Python
Environments") reads the workspace-root `.env` and injects *every* variable in
it into *every* integrated terminal, **before the shell starts**. It caches that
list per-workspace in:

```
~/Library/Application Support/Cursor/User/workspaceStorage/<workspace-id>/state.vscdb
  key: terminal.integrated.environmentVariableCollectionsV2
```

The cache goes **stale**: it was captured when the gateway lines were
*uncommented* in root `.env`. After you comment them out, the extension keeps
injecting the old `SMART_GATEWAY_URL` / `SMART_GATEWAY_API_KEY` /
`EMBEDDER_API_BASE`.

This is why the usual suspects all come up empty:

| Checked | Result |
|---|---|
| SDK code | only reads `os.environ.get(...)`, never sets the gateway |
| repo-root `.env` | gateway commented (verified: loading `config.py` from a clean env → `None`) |
| shell rc files (`.zshrc`/`.zprofile`/`.zshenv`/`.zlogin`), `/etc/zsh*` | clean |
| `launchctl getenv`, LaunchAgents | clean |
| venv `activate` scripts | clean |
| macOS Terminal.app (outside Cursor) | **clean** — proves it's Cursor-injected |
| Cursor integrated terminal | **polluted** — extension injects pre-shell |

## Fix

Cursor must be **closed** (the DB is per-workspace and rewritten on exit), so run
this in **Terminal.app**, not a Cursor terminal:

1. Quit Cursor completely (⌘Q).
2. Clear the stale cache (replace `<workspace-id>` if the path differs):

```bash
db="$HOME/Library/Application Support/Cursor/User/workspaceStorage/<workspace-id>/state.vscdb"
sqlite3 "$db"        "DELETE FROM ItemTable WHERE key='terminal.integrated.environmentVariableCollectionsV2';"
sqlite3 "$db.backup" "DELETE FROM ItemTable WHERE key='terminal.integrated.environmentVariableCollectionsV2';"
```
   To find `<workspace-id>`: it's the `workspaceStorage/*/` folder whose
   `workspace.json` points at this repo.

3. Reopen Cursor, open a terminal, verify:
```bash
echo "$SMART_GATEWAY_URL"   # should print nothing
```

Deleting the key is safe — on relaunch the extensions (debugpy, git,
python-envs) rebuild their contributions, and python-envs re-reads the
**current** (clean) `.env`, so the gateway does not come back.

## Prevention

- The extension injecting the *entire* `.env` — including secrets
  (`SMART_GATEWAY_API_KEY`, `LANGFUSE_SECRET_KEY`, `OPENAI_API_KEY`, …) — into
  every terminal is a footgun. To stop it, disable **`ms-python.vscode-python-envs`**
  (or its env-file injection) and rely on the app's own `load_dotenv()`.
- `config.py` in this playground has a guard (currently commented out) that pops
  any `SMART_GATEWAY_*` / `EMBEDDER_API_*` var **not present in the loaded
  `.env`**. Re-enabling it makes root `.env` the single source of truth even if a
  stray value is injected into the environment.
