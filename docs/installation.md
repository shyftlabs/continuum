# Installation & Configuration

Continuum is distributed as `shyftlabs-continuum` on PyPI and
importable as `orchestrator`. This doc covers both **library
consumers** (who `pip install` the package) and **framework
contributors** (working in this repository).

---

## 1 · Prerequisites

- **Python 3.13** — the framework requires it. Use `pyenv`, `uv`, or
  a system 3.13.
- **Docker + Docker Compose** — for Redis, Milvus (default vector store), and (optionally) Langfuse.
- **An LLM provider key** — at minimum `OPENAI_API_KEY`. mem0's default
  embedder is OpenAI's `text-embedding-3-small`, which means an
  `OPENAI_API_KEY` is required at startup *even if* you only call
  Anthropic or Gemini for chat.

---

## 2 · Install paths

### As a library (consumer)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install shyftlabs-continuum            # latest release
# or for a specific extra:
# pip install "shyftlabs-continuum[temporal,eval]"
```

### From source (contributor / development)

```bash
git clone https://github.com/bhavik-shyftlabs/continuum.git
cd continuum

cp .env.template .env                      # add OPENAI_API_KEY
docker compose up -d                       # Redis (:6380) + Milvus (:19530)

python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,temporal,eval]"     # editable install with all extras

# Smoke-test against a runnable example
python -m playground.sdk_feature_test
```

---

## 3 · The package

```bash
$ pip show shyftlabs-continuum
Name: shyftlabs-continuum
```

Importable as `orchestrator`:

```python
from orchestrator.agent import BaseAgent, AgentRunner
```

Runtime dependencies (declared in `pyproject.toml`):

- `aiohttp >= 3.13.2`
- `openai >= 1.50.0`
- `anthropic >= 0.40.0`
- `google-genai >= 1.0.0`
- `tiktoken >= 0.7.0`
- `langfuse >= 2.57.0, < 3.0.0`
- `pydantic >= 2.10.0`, `pydantic-settings >= 2.6.0`
- `python-dotenv >= 1.0.1`
- `mem0ai >= 1.0.0`
- `qdrant-client >= 1.16.0`
- `redis >= 5.0.0`
- `mcp >= 1.23.0`

Optional extras (declared but not installed by default):

| Extra | Adds | Use when |
|---|---|---|
| `[temporal]` | `temporalio >= 1.23.0` | Building durable workflows |
| `[embeddings]` | `sentence-transformers >= 2.2.0` | Local embeddings |
| `[cohere]` | `cohere >= 5.0.0` | Cohere embeddings |
| `[eval]` | `deepeval`, `ragas` | Evaluation framework |
| `[dev]` | `pytest`, `ruff`, `mypy`, `respx`, `fakeredis` | Tests & linting |

Install on demand:

```bash
pip install "shyftlabs-continuum[temporal]"
```

---

## 4 · Environment variables

The framework reads everything via pydantic-settings from
`os.environ`, which is populated from `.env` by `load_dotenv()` at
import time. **Restart your shell or re-`source` the venv after editing
`.env`.**

### LLM provider keys

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI; **required at startup** for the mem0 embedder |
| `OPENAI_ORGANIZATION` | Optional OpenAI organization id |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI |
| `VERTEX_PROJECT`, `VERTEX_LOCATION` | Vertex AI |
| `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | Azure OpenAI |

### Default LLM

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Default model |
| `FALLBACK_LLM_MODEL` | `gemini/gemini-1.5-flash` | Used when the primary fails (and `LLM_ENABLE_FALLBACK=true`) |
| `DEFAULT_LLM_TEMPERATURE` | `0.7` | |
| `DEFAULT_LLM_MAX_TOKENS` | `4096` | |
| `LLM_REQUEST_TIMEOUT` | `60` | Seconds |
| `LLM_MAX_RETRIES` | `3` | |
| `LLM_ENABLE_FALLBACK` | `true` | |

### Memory (mem0 + Qdrant / Milvus)

| Variable | Default | Description |
|---|---|---|
| `MEMORY_ENABLED` | `true` | Master switch |
| `VECTOR_STORE_PROVIDER` | `milvus` | `qdrant` or `milvus` |
| `QDRANT_HOST` | `localhost` | Used when `VECTOR_STORE_PROVIDER=qdrant` |
| `QDRANT_PORT` | `6333` | |
| `QDRANT_API_KEY` | unset | Qdrant Cloud |
| `QDRANT_COLLECTION` | `orchestrator_memories` | |
| `MILVUS_HOST` | `localhost` | Used when `VECTOR_STORE_PROVIDER=milvus` |
| `MILVUS_PORT` | `19530` | |
| `MILVUS_TOKEN` | unset | Zilliz Cloud |
| `MILVUS_COLLECTION` | `orchestrator_memories` | |
| `MEMORY_LLM_MODEL` | `gpt-4o-mini` | LLM for fact extraction |
| `MEMORY_LLM_TEMPERATURE` | `0.1` | |
| `EMBEDDER_PROVIDER` | `openai` | `openai` / `azure_openai` / `huggingface` / `ollama` / `gemini` / `vertexai` / `cohere` |
| `EMBEDDER_MODEL` | `text-embedding-3-small` | |
| `EMBEDDING_DIMS` | `1536` | Must match the embedder output |
| `EMBEDDER_API_KEY` | unset | Override env-supplied key |
| `EMBEDDER_API_BASE` | unset | Self-hosted / Azure |
| `MEMORY_HISTORY_DB_PATH` | `~/.orchestrator/memory_history.db` | SQLite history |
| `MEMORY_ISOLATION` | `user` | `shared` / `user` / `agent` / `conversation` |
| `MEMORY_SEARCH_LIMIT` | `5` | Default top-K |

### Session (Redis)

| Variable | Default | Description |
|---|---|---|
| `SESSION_ENABLED` | `true` | |
| `SESSION_REDIS_HOST` | `localhost` | |
| `SESSION_REDIS_PORT` | `6380` | **6380, not 6379** |
| `SESSION_REDIS_PASSWORD` | unset | |
| `SESSION_REDIS_DB` | `0` | |
| `SESSION_REDIS_SSL` | `false` | |
| `SESSION_TTL_SECONDS` | `604800` | 7 days |
| `SESSION_MAX_MESSAGES` | `1000` | Sliding-window trim above this |
| `SESSION_KEY_PREFIX` | `orchestrator:session` | |

### Observability (Langfuse)

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_ENABLED` | `true` | Set to `false` to disable tracing entirely |
| `LANGFUSE_PUBLIC_KEY` | unset | |
| `LANGFUSE_SECRET_KEY` | unset | |
| `LANGFUSE_HOST` | `http://localhost:3000` | |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | 0.0–1.0 |
| `LANGFUSE_FLUSH_INTERVAL` | `1` | Seconds |
| `LANGFUSE_FLUSH_AT` | `15` | Event-count threshold |
| `LANGFUSE_DEBUG` | `false` | Verbose tracing logs |
| `LANGFUSE_RELEASE` | unset | Tag for releases |
| `ENVIRONMENT` | `development` | Tagged on traces |

### Context management

| Variable | Default | Description |
|---|---|---|
| `CONTEXT_MANAGEMENT_ENABLED` | `true` | |
| `CONTEXT_COMPRESSION_THRESHOLD` | `0.8` | Compress at 80 % of model window |
| `CONTEXT_SUMMARIZATION_MODEL` | `gpt-4o-mini` | |
| `CONTEXT_SUMMARIZATION_TEMPERATURE` | `0.1` | |
| `CONTEXT_SUMMARIZATION_TIMEOUT` | `30` | Seconds |
| `CONTEXT_SUMMARIZATION_MAX_RETRIES` | `2` | |
| `CONTEXT_KEEP_RECENT_MESSAGES` | `10` | Always keep the N most recent |
| `CONTEXT_ENABLE_CACHING` | `true` | |
| `CONTEXT_CACHE_TTL_SECONDS` | `3600` | 1 hour |

### Temporal *(optional, requires `[temporal]` extra)*

| Variable | Default | Description |
|---|---|---|
| `TEMPORAL_ENABLED` | `false` | |
| `TEMPORAL_HOST` | `localhost:7233` | |
| `TEMPORAL_NAMESPACE` | `default` | |
| `TEMPORAL_TASK_QUEUE` | `orchestrator-agents` | |
| `TEMPORAL_ENABLE_HUMAN_IN_LOOP` | `true` | |
| `TEMPORAL_APPROVAL_TIMEOUT_SECONDS` | `86400` | 24 h |
| `TEMPORAL_WORKFLOW_EXECUTION_TIMEOUT` | `604800` | 7 days |
| `TEMPORAL_ACTIVITY_START_TO_CLOSE_TIMEOUT` | `300` | 5 min |
| `TEMPORAL_ACTIVITY_RETRY_MAX_ATTEMPTS` | `3` | |

### Lifecycle / logging

| Variable | Default | Description |
|---|---|---|
| `SHARED_SERVICES_ENABLED` | `true` | If `true`, `Container.shutdown()` does not close Redis or flush Langfuse |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |

**`LOG_FULL_PROMPT`** is **not** a `Settings` field — it's read directly
via `os.environ.get("LOG_FULL_PROMPT", "")` in
`agent/execution/message_builder.py`. Set it to `true` to log the
assembled prompt before each LLM call. Useful for debugging memory /
RAG / handoff flows; does not need to appear in `.env` to work.

---

## 5 · Verifying installation

```bash
python -c "import orchestrator; print('orchestrator imports OK')"
python -c "from orchestrator.agent import BaseAgent, AgentRunner; print('agent imports OK')"
python -c "from orchestrator.llm.providers import get_provider; print('providers OK')"
```

Smoke-test with infra:

```bash
docker compose up -d
python -m playground.sdk_feature_test
```

Expected: agent prints a one-line greeting. A clean `OpenAIError:
Missing credentials` means infra is fine but `OPENAI_API_KEY` isn't set
(check `.env` and re-source).

---

## 6 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: orchestrator` | venv not active | `source .venv/bin/activate` |
| `Failed to initialize mem0: Missing credentials` | mem0 needs OpenAI for embeddings | set `OPENAI_API_KEY` or `MEMORY_ENABLED=false` |
| `redis.exceptions.ConnectionError` | Redis not running / wrong port | `docker compose ps`; check `SESSION_REDIS_PORT=6380` |
| Vector store collection not found | Stale volume after schema change | `docker compose down -v && docker compose up -d` |
| `aiohttp` deprecation warnings | Older aiohttp | `pip install "aiohttp>=3.13.2" --upgrade` |
| Imports work but tools never fire | Forgot `await server.connect()` or `executor.initialize()` | Add the missing `await` |
| `add_message() got an unexpected keyword argument 'role'` | Old API | Pass a `ChatMessage` object — see [`session.md`](session.md) |
| Python version error during install | System Python too old | `python3.13 -m venv .venv` |
