# Configuration

All settings are read from environment variables (or a `.env` file) by
[`app/config.py`](../app/config.py). Names map case-insensitively, so
`anthropic_model` reads `ANTHROPIC_MODEL`. Unknown variables are ignored. Start
from [`.env.example`](../.env.example).

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | **yes** | — | Postgres connection string for the single app database. The `vector` extension is created automatically at startup. `postgres://` is rewritten to the asyncpg driver. |
| `ANTHROPIC_API_KEY` | **yes** | — | Claude API key — the only required cloud credential. Get one at [console.anthropic.com](https://console.anthropic.com). |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-4-6` | Chat model id. Use a larger model (e.g. an Opus id) for harder reasoning, a smaller one (e.g. a Haiku id) for lower cost/latency. |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` | fastembed model id. Runs locally on CPU; downloaded and cached on first use. |
| `EMBEDDING_DIM` | no | `384` | Vector dimension. **Must match `EMBEDDING_MODEL`'s output.** Changing it requires resetting the database, since the vector columns are sized to it. |
| `AI_API_KEY` | **yes** | — | Shared secret clients must send (as `X-API-Key` or `?api_key=`) to reach `/chat` and `/ingest`. |
| `CACHE_SIMILARITY_THRESHOLD` | no | `0.92` | Minimum cosine similarity for a semantic-cache hit. Higher = stricter (fewer, safer reuses); lower = more aggressive caching. |
| `AGENT_MAX_ITERATIONS` | no | `5` | Caps the agent's reasoning/tool loop per request (recursion limit = `2·n + 2`). |
| `MAX_CONTEXT_TOKENS` | no | `8000` | Approximate per-session history budget; older turns are trimmed before each LLM call. |
| `LOG_LEVEL` | no | `INFO` | Root log level. `DEBUG` also emits GC gen0/1 events from the observability monitor. |

## Notes

- **Single database.** There is no separate vector database and no external
  read-only database — one `DATABASE_URL` is all that's needed.
- **Embedding model ↔ dimension.** If you change `EMBEDDING_MODEL`, set
  `EMBEDDING_DIM` to its output size and recreate the schema (e.g.
  `docker compose down -v && docker compose up -d && docker compose run --rm seed`).
- **Tests** read `TEST_DATABASE_URL` if set, else `DATABASE_URL`. DB-backed tests
  are skipped when no database is reachable.
