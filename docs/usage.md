# Usage

## Running

See the [Quickstart](../README.md#quickstart). In short: `docker compose up -d`,
then `docker compose run --rm seed`, then `python -m client`.

To run the API natively: `uv sync`, set `DATABASE_URL` to a Postgres with the
`vector` extension, then `uv run uvicorn app.main:app --reload`.

## Authentication

Every protected endpoint requires the shared secret from `AI_API_KEY`.

- HTTP (`/ingest`): send it as the `X-API-Key` header.
- WebSocket (`/chat`): send it as the `X-API-Key` header **or** the `?api_key=`
  query parameter (handy for browsers/CLIs).

Missing key → `403`; wrong key → `401`.

## Endpoints

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

### `WebSocket /chat`

Real-time chat. After connecting, send one JSON message per turn:

```json
{ "question": "Quanto gastei com mercado?", "lang": "pt-br", "session_id": null }
```

- `question` — required, 1–4000 chars.
- `lang` — response language (`pt-br`, `pt`, `en`, `es`, …). Default `pt-br`.
- `session_id` — pass the id returned by a previous turn to keep conversation
  memory; omit/`null` to start a new thread.

You receive a stream of events:

```json
{ "type": "token", "content": "Você " }
{ "type": "token", "content": "gastou " }
{ "type": "done", "answer": "Você gastou R$...", "cached": false,
  "sources": ["..."], "session_id": "e3b0c4..." }
```

On a **cache hit** there are no `token` events — just the final `done` with
`cached: true`. Malformed payloads and runtime errors come back as
`{ "type": "error", "detail": "..." }` without closing the socket.

Example with [`wscat`](https://github.com/websockets/wscat):

```bash
wscat -c "ws://localhost:8000/chat?api_key=$AI_API_KEY"
> {"question":"O que é uma reserva de emergência?","lang":"pt-br"}
```

Or just use the bundled client: `python -m client` (flags: `--url`, `--api-key`).

### `POST /ingest`

Add your own document to the RAG knowledge base. It is split into chunks,
embedded locally, and stored for retrieval.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: $AI_API_KEY" -H "Content-Type: application/json" \
  -d '{"content":"Texto sobre diversificação de investimentos...","source":"meu-guia"}'
# -> {"chunks_stored": 3}
```

`content` is required (≤100k chars); `source` and `metadata` are optional.

## Seeding / resetting demo data

`python -m scripts.seed` creates the schema, inserts categories and ~6 months of
deterministic transactions, and ingests the Markdown files under `data/kb/`. It
is idempotent — re-running skips data that already exists. To start clean, drop
the database volume (`docker compose down -v`) and seed again.
