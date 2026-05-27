---
name: continuum-claude-code
description: Set up Claude Code for a Continuum project — CLAUDE.md wiring, skill imports, project settings, and hooks. Invoke when the user asks "set up Claude Code for my project", "CLAUDE.md for Continuum", "add skills to my project", or "configure Claude Code hooks".
---

# Continuum + Claude Code Skill

---

## CLAUDE.md setup

Create a `CLAUDE.md` at the project root and import the Continuum knowledge pack:

```markdown
# My Project

@AGENTS.md

## Project-specific notes
- Entry point: `src/my_app/main.py`
- Run: `python -m my_app`
- Tests: `pytest tests/`
```

The `@AGENTS.md` import pulls in the full Continuum API reference automatically.
Claude Code merges it with your project-specific notes.

---

## Skills wiring

The 13 Continuum skills in `.claude/skills/` are invocable via `/skill-name`.
They are loaded automatically when Claude Code detects relevant queries.

To use them in a downstream project (not the framework source repo itself),
copy the `.claude/` directory into your project root:

```bash
cp -r /path/to/continuum/.claude ./
```

Or reference the skills directory in your `.claude/settings.json`:

```json
{
  "skills": [".claude/skills"]
}
```

Available skills for Continuum development:

| Skill | Trigger words |
|---|---|
| `continuum-agent` | "create an agent", "BaseAgent", "lifecycle hooks" |
| `continuum-memory` | "remember", "long-term memory", "Milvus", "Qdrant" |
| `continuum-streaming` | "stream tokens", "websocket", "live output" |
| `continuum-tools-mcp` | "connect MCP", "filesystem tool", "remote API" |
| `continuum-workflows` | "chain agents", "parallel", "router" |
| `continuum-handoffs` | "transfer to another agent", "triage" |
| `continuum-llm-providers` | "switch to Claude", "gateway_mode", "Gemini" |
| `continuum-observability` | "Langfuse", "traces", "latency" |
| `continuum-testing` | "mock the LLM", "pytest", "fakeredis" |
| `continuum-temporal` | "long-running workflow", "approval gate" |
| `continuum-quickstart` | "get started", "first agent" |
| `continuum-recipes` | "RAG", "FastAPI integration", "plan-and-execute" |
| `continuum-evaluation` | "evaluate output", "DeepEval", "RAGAS" |

---

## Project settings (.claude/settings.json)

```json
{
  "permissions": {
    "allow": [
      "Bash(python:*)",
      "Bash(pytest:*)",
      "Bash(pip:*)",
      "Bash(docker:*)"
    ]
  }
}
```

---

## Useful hooks

Auto-run tests after edits to source files:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "cd /path/to/project && pytest tests/ -x -q 2>&1 | tail -20"
          }
        ]
      }
    ]
  }
}
```

---

## Don't

- Don't duplicate content from `AGENTS.md` in `CLAUDE.md` — use `@AGENTS.md` import instead.
- Don't add skills to `.claude/settings.json` manually — place SKILL.md files in `.claude/skills/<name>/`.
- Don't put secrets in `CLAUDE.md` — it may be committed; use `.env` and `load_dotenv()`.
