# Finance AI Assistant

[![CI](https://github.com/Pissinatti-py/finance-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Pissinatti-py/finance-ai-assistant/actions/workflows/ci.yml)

A standalone, self-hostable **personal-finance assistant**: ask about your own
spending and income, project investments, and get plain explanations of finance
concepts — in natural language. It is a compact but real showcase of a modern
LLM-agent backend: a **LangGraph agent** with **tools**, **RAG** over a knowledge
base, and a **semantic cache**, all backed by Postgres + pgvector.

Runs end-to-end with a single cloud credential (an Anthropic API key);
embeddings run locally on CPU, so the vector side needs no account at all.

```
          ┌──────────── WebSocket /chat ────────────┐
user ──▶  │  START → check_cache ─(hit)─────────────┼──▶ answer
          │              │(miss)                    │
          │              ▼                          │
          │           agent ⇄ tools                 │
          │              │(no tool calls)           │
          │              ▼                          │
          │           persist (cache) ──────────────┘
          └─────────────────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/Pissinatti-py/finance-ai-assistant.git finance-ai-assistant
cd finance-ai-assistant
cp .env.example .env            # then set ANTHROPIC_API_KEY and AI_API_KEY

docker compose up -d            # starts Postgres+pgvector and the API
docker compose run --rm seed    # loads demo categories, transactions and KB docs

python -m client                # interactive streaming chat in your terminal
```

The API is then at `http://localhost:8000` (Swagger UI at `/docs`). The CLI
connects to the WebSocket and streams answers token by token.

> Running natively instead of Docker? Install with `uv sync`, point
> `DATABASE_URL` at a local Postgres with the `vector` extension, then
> `uv run uvicorn app.main:app --reload` and `uv run python -m scripts.seed`.

## Try it

With the seeded demo data, ask:

- *"Quanto gastei com mercado nos últimos meses?"* → uses `summarize_spending`
- *"Mostre minhas últimas transações de restaurante."* → uses `search_transactions`
- *"Quanto rende R$10.000 a 12% ao ano em 5 anos com aporte de R$500/mês?"* → uses `compound_interest`
- *"O que é uma reserva de emergência?"* → uses RAG (`retrieve_knowledge`)

Ask the same question twice and the second answer returns instantly from the
**semantic cache** (the `done` event reports `cached: true`).

## High-level features

| Feature | Where |
|---|---|
| LangGraph agent (cache → agent ⇄ tools → persist) | [`app/agent/graph.py`](app/agent/graph.py) |
| Composable **skills** (tools + prompt fragments) | [`app/agent/skills/`](app/agent/skills/) |
| Structured-lookup tools over your data | [`app/agent/tools/transactions.py`](app/agent/tools/transactions.py) |
| **RAG** ingest + retrieval (chunked, pgvector) | [`app/rag/store.py`](app/rag/store.py) |
| **Semantic cache** (cosine similarity short-circuit) | [`app/agent/cache.py`](app/agent/cache.py) |
| Generic async CRUD **`BaseManager`** | [`app/db/managers/base.py`](app/db/managers/base.py) |
| **`VectorManager`** similarity search (HNSW) | [`app/db/managers/vector.py`](app/db/managers/vector.py) |
| Local embeddings (fastembed, CPU) | [`app/agent/embeddings.py`](app/agent/embeddings.py) |
| Streaming WebSocket chat + API-key auth | [`app/api/routes.py`](app/api/routes.py) |
| Prompt-injection defences (data fencing) | [`app/agent/tools/knowledge.py`](app/agent/tools/knowledge.py) |

## Configuration

Set in `.env` (see [`.env.example`](.env.example)). Full reference:
[`docs/configuration.md`](docs/configuration.md).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | yes | — | Postgres + pgvector connection (single DB) |
| `ANTHROPIC_API_KEY` | yes | — | Claude API key (the only cloud credential) |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-4-6` | Chat model id |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` | fastembed model (local) |
| `EMBEDDING_DIM` | no | `384` | Must match the embedding model output |
| `AI_API_KEY` | yes | — | Shared secret clients send as `X-API-Key` |
| `CACHE_SIMILARITY_THRESHOLD` | no | `0.92` | Min cosine similarity for a cache hit |
| `AGENT_MAX_ITERATIONS` | no | `5` | Caps the agent's reasoning loop |

## How to extend

The core is domain-agnostic — add a skill, a tool, a model+manager, or swap the
LLM/embeddings provider with small, localized changes. Step-by-step guides in
[`docs/extending.md`](docs/extending.md). You can even replace the finance domain
wholesale; the agent, cache, RAG and managers stay as they are.

## Documentation

- [`docs/usage.md`](docs/usage.md) — endpoints, the CLI, ingesting your own docs
- [`docs/architecture.md`](docs/architecture.md) — modules, the graph, the data flow
- [`docs/extending.md`](docs/extending.md) — how to add/change capabilities
- [`docs/configuration.md`](docs/configuration.md) — every setting explained

## Tests

```bash
uv run pytest            # unit tests + DB tests (skipped if no database)
uv run pytest -m integration   # full cache+RAG flow against a real database
```

## Disclaimer

Educational project. It provides general information, not personalized financial
advice.
