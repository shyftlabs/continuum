# Temporal — Docker Compose Setup

The repository's `docker-compose.yml` includes Temporal services
behind the optional `temporal` profile so they don't run by default.

```bash
# Start everything (Redis + Qdrant + Langfuse + Temporal)
docker compose --profile observability --profile temporal up -d

# Just Temporal alongside core infra
docker compose --profile temporal up -d
```

---

## Services

| Service | Image | Host port | Purpose |
|---|---|---|---|
| `temporal-postgres` | `postgres:17` | (internal) | Backing store for Temporal server state |
| `temporal` | `temporalio/auto-setup:latest` | `7233` | Temporal server (gRPC) |
| `temporal-ui` | `temporalio/ui:latest` | `8080` | Web UI for browsing workflows |

The `auto-setup` image initializes the database schema on first start —
no migration step required.

---

## Configuration

`.env` keys read by the framework:

```env
TEMPORAL_ENABLED=true
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
TEMPORAL_ENABLE_HUMAN_IN_LOOP=true
TEMPORAL_APPROVAL_TIMEOUT_SECONDS=86400
TEMPORAL_WORKFLOW_EXECUTION_TIMEOUT=604800
TEMPORAL_ACTIVITY_START_TO_CLOSE_TIMEOUT=300
TEMPORAL_ACTIVITY_RETRY_MAX_ATTEMPTS=3
```

Match `TEMPORAL_HOST` to the compose service name from inside other
containers (`temporal:7233`); use `localhost:7233` from the host.

---

## Common operations

### Open the UI

http://localhost:8080 — search by workflow id, namespace, or status,
and inspect step-by-step event history.

### Tail logs

```bash
docker compose logs -f temporal
docker compose logs -f temporal-ui
```

### Reset state (development only)

```bash
docker compose --profile temporal down -v
docker compose --profile temporal up -d
```

`-v` drops the Postgres volume — every workflow you ran will be gone.
Don't do this in shared environments.

### Multi-namespace

Default namespace is `default`. To create more:

```bash
docker compose exec temporal tctl --namespace billing namespace register
```

Set `TEMPORAL_NAMESPACE=billing` in your `.env` and the framework will
use it.

---

## Health check

```python
from orchestrator.core.health import check_all_health
result = await check_all_health()
print(result.to_dict()["temporal"])
```

The built-in temporal health probe pings the gRPC endpoint and reports
`HEALTHY` / `UNHEALTHY` with latency.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `failed to connect: localhost:7233` | Temporal not running | `docker compose --profile temporal up -d` |
| Workflow IDs reused / errors about "already started" | Same `id` submitted twice | Use a unique id per submission |
| `task_queue` mismatch | Worker and caller use different queues | Align `TEMPORAL_TASK_QUEUE` |
| Slow startup | `auto-setup` schema init on first boot | Wait ~30 s after first `up`; subsequent starts are fast |
| UI shows no workflows | Pointed at wrong namespace | Toggle namespace in the UI header |
