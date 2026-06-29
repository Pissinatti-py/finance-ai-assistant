"""Shared pytest fixtures.

Isolation strategy: SAVEPOINT-per-test (same pattern as pissync-core).
Each test runs inside an outer transaction on a single connection. The session
uses ``join_transaction_mode="create_savepoint"`` so every commit() the
manager issues commits a SAVEPOINT — visible within the test — while the outer
transaction rolls back at teardown. The DB is clean for the next test without
DROP/CREATE cycles.

Schema is created once per session via _schema. The same database used in
development is safe to target because all writes roll back.

Set TEST_DATABASE_URL to override the target (e.g. a dedicated CI database).
DB tests are skipped when neither DATABASE_URL nor TEST_DATABASE_URL is set.
"""

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── env defaults (must be before any app import) ────────────────────────────
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://finance:finance@localhost:5432/finance")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("AI_API_KEY", "test-api-key")
os.environ.setdefault("CACHE_SIMILARITY_THRESHOLD", "0.5")

from app.config import settings
from app.db import models as _models_import  # noqa: F401 — registers models with Base
from app.db.base import Base

# ── constants ────────────────────────────────────────────────────────────────
# Tests target a dedicated "<db>_test" database (auto-created) so development
# seed data in the main database never affects assertions. Override with
# TEST_DATABASE_URL to point at your own.
_base_url = make_url(os.environ["DATABASE_URL"])
TEST_DB_URL: str = os.getenv("TEST_DATABASE_URL") or _base_url.set(
    database=f"{_base_url.database}_test"
).render_as_string(hide_password=False)
FAKE_VEC: list[float] = [0.1] * settings.embedding_dim
TEST_API_KEY = "test-api-key"


async def _ensure_database(async_url: str) -> None:
    """Create the test database if it does not exist (connects to ``postgres``)."""
    url = make_url(async_url)
    admin_dsn = url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", url.database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{url.database}"')
    finally:
        await conn.close()


# ── schema lifecycle (session-scoped) ────────────────────────────────────────
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _schema() -> AsyncGenerator[None, None]:
    """Create tables once; leave them for the session (SAVEPOINT handles isolation)."""
    try:
        await _ensure_database(TEST_DB_URL)
        engine = create_async_engine(TEST_DB_URL, echo=False)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
    except Exception as exc:
        pytest.skip(f"database unavailable — {exc}")
    yield


# ── per-test engine ──────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def engine(_schema):
    """Fresh async engine per test (avoids cross-loop asyncpg errors)."""
    eng = create_async_engine(TEST_DB_URL, echo=False)
    try:
        yield eng
    finally:
        await eng.dispose()


# ── SAVEPOINT session ────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def db(engine) -> AsyncGenerator[AsyncSession, None]:
    """Async session with SAVEPOINT isolation.

    Manager commits() are promoted to SAVEPOINT commits (still visible in the
    same test) but the outer BEGIN never commits, so teardown rollback leaves
    the DB clean.
    """
    async with engine.connect() as conn:
        outer = await conn.begin()
        Session = async_sessionmaker(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        async with Session() as session:
            yield session
        await outer.rollback()


# ── integration: session factory bound to the test connection ────────────────
@pytest_asyncio.fixture
async def session_factory(db: AsyncSession):
    """Return an async_sessionmaker bound to the test connection.

    Passing this as SessionLocal to cache/rag modules makes their internal
    sessions share the test's outer transaction, so SAVEPOINT rollback cleans
    their writes too.
    """
    conn = await db.connection()
    return async_sessionmaker(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )


# ── embeddings stub ──────────────────────────────────────────────────────────
@pytest.fixture
def fake_embeddings(monkeypatch):
    """Replace the embeddings client with a stub returning FAKE_VEC."""
    from app.agent import embeddings as emb_module

    mock = MagicMock()
    mock.aembed_query = AsyncMock(return_value=FAKE_VEC)
    mock.aembed_documents = AsyncMock(side_effect=lambda docs: [FAKE_VEC] * len(docs))
    monkeypatch.setattr(emb_module, "client", mock)
    return mock


# ── HTTP client (no lifespan) ────────────────────────────────────────────────
@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[AsyncClient, None]:
    """FastAPI test client built from a bare app (no lifespan, no embeddings/DB init).

    Routes are imported directly; run_agent / add_chunks are mocked per test.
    """
    from fastapi import FastAPI

    from app.api.routes import router

    _app = FastAPI()
    _app.include_router(router)

    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as client:
        yield client
