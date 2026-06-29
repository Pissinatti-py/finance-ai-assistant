"""Full-flow integration tests: semantic cache + RAG knowledge base.

Requires the postgres+pgvector container running and DATABASE_URL (or
TEST_DATABASE_URL) set. Embeddings are stubbed via fake_embeddings; all DB
layers are real.

Run:
    uv run pytest tests/test_integration.py -m integration

All tests share SAVEPOINT isolation: each write is rolled back after the test.
The patch_sessions fixture (autouse) redirects SessionLocal in both cache and
rag modules to the test's connection so those writes are visible within the
test and cleaned up on teardown.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def patch_sessions(session_factory, monkeypatch):
    """Wire cache + rag modules to the test's SAVEPOINT-isolated connection."""
    import app.agent.cache as cache_mod
    import app.rag.store as rag_mod

    monkeypatch.setattr(cache_mod, "SessionLocal", session_factory)
    monkeypatch.setattr(rag_mod, "SessionLocal", session_factory)


# ── semantic cache ────────────────────────────────────────────────────────────


async def test_cache_miss_on_empty_store(fake_embeddings):
    from app.agent.cache import get_cached

    vec = await fake_embeddings.aembed_query("What is diabetes?")
    assert await get_cached(vec) is None


async def test_cache_store_then_hit(fake_embeddings):
    import app.agent.cache as cache_mod

    vec = await fake_embeddings.aembed_query("What is diabetes?")
    await cache_mod.store("What is diabetes?", vec, "Diabetes is a metabolic disease.", ["art-001"])
    result = await cache_mod.get_cached(vec)

    assert result is not None
    assert result["answer"] == "Diabetes is a metabolic disease."
    assert "art-001" in result["sources"]


async def test_cache_multiple_entries_returns_closest(fake_embeddings):
    """With identical embeddings, both entries score 1.0; at least one is returned."""
    import app.agent.cache as cache_mod

    vec = await fake_embeddings.aembed_query("any question")
    await cache_mod.store("question A", vec, "answer A", [])
    await cache_mod.store("question B", vec, "answer B", [])
    result = await cache_mod.get_cached(vec)

    assert result is not None
    assert result["answer"] in ("answer A", "answer B")


async def test_cache_orthogonal_embedding_is_miss(fake_embeddings):
    """When stored and query vectors are orthogonal the threshold filters the hit."""
    import app.agent.cache as cache_mod
    from app.config import settings

    dim = settings.embedding_dim
    store_vec = [1.0] + [0.0] * (dim - 1)  # dim-0
    query_vec = [0.0, 1.0] + [0.0] * (dim - 2)  # dim-1 (orthogonal, sim≈0)

    await cache_mod.store("stored question", store_vec, "stored answer", [])
    # CACHE_SIMILARITY_THRESHOLD=0.5 in conftest, orthogonal → sim=0 → miss
    assert await cache_mod.get_cached(query_vec) is None


# ── rag knowledge base ────────────────────────────────────────────────────────


async def test_rag_empty_store_returns_empty(fake_embeddings):
    from app.rag.store import retrieve

    assert await retrieve("anything") == []


async def test_rag_ingest_creates_chunks(fake_embeddings):
    from app.rag.store import add_chunks

    content = "Hypertension is elevated blood pressure. " * 30  # long → multiple chunks
    n = await add_chunks(content, source="medical_guide")
    assert n >= 2


async def test_rag_retrieve_returns_stored_content(fake_embeddings):
    from app.rag.store import add_chunks, retrieve

    await add_chunks("Hypertension is high blood pressure.", source="test_src")
    results = await retrieve("blood pressure")

    assert len(results) > 0
    assert results[0]["source"] == "test_src"
    assert "content" in results[0]
    assert "similarity" in results[0]


async def test_rag_retrieve_limit_respected(fake_embeddings):
    from app.rag.store import add_chunks, retrieve

    for i in range(6):
        await add_chunks(f"Chunk number {i}.", source=f"src_{i}")

    results = await retrieve("query", k=3)
    assert len(results) <= 3


async def test_rag_source_preserved_in_chunks(fake_embeddings):
    from app.rag.store import add_chunks, retrieve

    await add_chunks("Medical content from chapter one.", source="chapter_1")
    results = await retrieve("medical content")

    assert all(r["source"] == "chapter_1" for r in results)
