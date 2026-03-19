# Continuum SDK in Tax Pilot

The **shyftlabs-continuum-customer-os** (Continuum SDK) is an agent SDK with multi-LLM support, memory (Qdrant + mem0), sessions (Redis), and observability (Langfuse). Its Docker components are wired into the main Tax Pilot stack.

## Running the stack

From the **project root** (where `docker-compose.yml` lives):

```bash
# Start everything (Tax Pilot + Continuum services)
docker compose up -d

# Or with build
docker compose up -d --build
```

## Ports (Continuum)

| Service           | Port(s)   | Purpose                    |
|-------------------|-----------|----------------------------|
| **Langfuse UI**   | 3001      | Observability dashboard    |
| **redis-sdk**     | 6380      | Continuum session store    |
| **Qdrant**        | 6333, 6334| Vector DB (long-term memory) |
| **Temporal**      | 7233      | Workflow engine            |
| **Temporal UI**    | 8233      | Workflow dashboard         |
| **Postgres (Langfuse)** | 5433 | Langfuse DB                |
| **Redis (Langfuse)**    | 6381 | Langfuse queue/cache       |
| **ClickHouse**    | 8123, 9002| Langfuse analytics         |

Tax Pilot keeps its own ports (e.g. API 8000, Web 3000, Postgres 5432, Redis 6379).

## Environment (single .env)

Continuum uses the **same env file as the API**: `apps/api/.env`. There is no separate continuum `.env`. Copy `apps/api/.env.example` to `apps/api/.env` and set API keys and Continuum vars (session Redis, Qdrant, Langfuse, etc.) in that one file.

## Using the SDK from the API

1. **Install the SDK** in the API app (e.g. in `apps/api`):

   ```bash
   pip install -e ./shyftlabs-continuum-customer-os
   ```

   Or add to `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   continuum = ["shyftlabs-continuum @ file:./shyftlabs-continuum-customer-os"]
   ```

2. **Optional env** (when running API in Docker, these reach the Continuum services):

   - `SESSION_ENABLED=true`
   - `SESSION_REDIS_HOST=redis-sdk` (and `SESSION_REDIS_PORT=6379`)
   - `QDRANT_HOST=qdrant` (and `QDRANT_PORT=6333` for REST, `6334` for gRPC)
   - `LANGFUSE_ENABLED=true` and Langfuse keys from the Langfuse UI (http://localhost:3001)

3. **Temporal** (optional workflows): point the SDK at `temporal:7233` when running inside Docker.

## Langfuse

- First run: open **http://localhost:3001** and create an account / project.
- Copy the project **Public Key** and **Secret Key** into your app env as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable tracing.

## SDK docs

- [Orchestrator SDK docs](docs/README.md)
- [Session (Redis)](docs/session.md)
- [Memory (Qdrant)](docs/memory.md)
- [Observability (Langfuse)](docs/observability.md)
- [Temporal workflows](docs/temporal/getting-started.md)
