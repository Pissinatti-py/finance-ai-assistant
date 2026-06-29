# Architecture

## Modules

```
app/
├── api/          FastAPI routes (WebSocket /chat, /ingest, /health) + schemas
├── agent/        the LLM agent
│   ├── graph.py      LangGraph assembly (cache → agent ⇄ tools → persist)
│   ├── runner.py     stream_agent / run_agent entry points
│   ├── nodes.py      cache lookup + persistence nodes
│   ├── cache.py      semantic cache read/write
│   ├── llm.py        Claude (Anthropic API) chat model
│   ├── embeddings.py local fastembed client (async adapter)
│   ├── prompts.py    lang-aware system prompt + guardrails
│   ├── context.py    token-budget trimming of history
│   ├── skills/       composable capabilities (tools + prompt fragments)
│   └── tools/        low-level tool factories
├── rag/          chunk + embed + retrieve over the knowledge base
├── db/
│   ├── base.py       single declarative Base
│   ├── session.py    one async engine / SessionLocal
│   ├── models/       finance, cache, knowledge tables
│   └── managers/     BaseManager (CRUD), VectorManager (similarity), domain managers
└── schemas/      Pydantic create/read schemas
```

The app owns a **single Postgres + pgvector database**. There is no external or
read-only database: the finance tables, the semantic cache and the knowledge
base all live in it and are created by `Base.metadata.create_all` at startup.

## The agent graph

Built once per response language and memoized (`get_graph(lang)`), all graphs
sharing one checkpointer so a session's history is consistent.

1. **check_cache** — embed the question, run a cosine-similarity search over the
   semantic cache. Above the threshold → return the cached answer and skip the
   LLM entirely.
2. **agent** — the Claude model, bound to the active skills' tools, runs on the
   system prompt plus the token-budget-trimmed history. It either answers or
   requests tool calls.
3. **tools** — executes the requested tools (transaction lookups, compound
   interest, RAG retrieval) and loops back to the agent.
4. **persist** — once the agent answers with no further tool calls, the answer
   (with tool outputs as sources) is written to the semantic cache.

Streaming: `stream_agent` yields `token` events from the agent node as the
answer is produced, then a final `done` event. A cache hit yields only `done`.

## Semantic cache

A `SemanticCache` row stores the question text, its embedding, the answer and
its sources. `VectorManager.similarity_search` (cosine distance, ordered to use
pgvector's HNSW index) finds the closest prior question; if its similarity clears
`CACHE_SIMILARITY_THRESHOLD`, the stored answer is reused. This turns repeated or
near-duplicate questions into instant, LLM-free responses.

## RAG knowledge base

`rag/store.py` splits ingested text (`RecursiveCharacterTextSplitter`), embeds
each chunk locally, and stores it as a `KnowledgeChunk` (HNSW-indexed). The
`retrieve_knowledge` tool embeds the query, pulls the top-k chunks, and wraps
them in a data boundary that neutralizes the delimiter tag — a defence against
indirect prompt injection through ingested documents.

## Managers

`BaseManager` is a generic async CRUD layer (create/get/filter/paginate/update/
delete) parameterized by model + create/update schemas. `VectorManager` extends
it with `similarity_search` over an embedding column. Domain managers subclass
these — e.g. `TransactionManager` adds a `sum_amount` aggregate, `SemanticCache`
and `KnowledgeChunk` managers set their embedding column. New persistence needs
rarely require new query code: instantiate a `BaseManager[Model, ...]`.

## Why these choices

- **Anthropic API + local embeddings** — one cloud credential, no AWS account or
  model-access approval; embeddings cost nothing and need no network.
- **Single DB, `create_all`, no migrations** — the app owns its schema, so a
  fresh clone is one `docker compose up` away. Add Alembic when the schema starts
  evolving in production.
- **Skills over a monolithic prompt** — a capability is a small, testable unit;
  a different agent is just a different list of skills.
