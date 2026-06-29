"""VectorManager.similarity_search coverage.

Uses unit vectors in named dimensions so cosine similarity is mathematically
predictable:
  - same dimension → sim = 1.0
  - orthogonal dimensions → sim = 0.0
  - shared + extra dimension → sim = 1/√2 ≈ 0.707
"""

import math

import pytest

from app.config import settings
from app.db.managers.cache_manager import semantic_cache_manager as mgr
from app.schemas.cache import SemanticCacheCreate


def _unit(dim: int) -> list[float]:
    """Unit vector with weight only in `dim`."""
    v = [0.0] * settings.embedding_dim
    v[dim] = 1.0
    return v


def _mixed(*dims: int) -> list[float]:
    """Un-normalised vector with equal weight in each listed dimension."""
    v = [0.0] * settings.embedding_dim
    for d in dims:
        v[d] = 1.0
    return v


def _cache(text: str, vec: list[float]) -> SemanticCacheCreate:
    return SemanticCacheCreate(
        question_text=text,
        question_embedding=vec,
        response_text=f"ans:{text}",
        sources=[],
    )


async def test_identical_vector_returns_sim_one(db):
    await mgr.create(db, _cache("identical", _unit(0)))
    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=5)
    assert len(results) == 1
    _, sim = results[0]
    assert sim == pytest.approx(1.0, abs=1e-4)


async def test_orthogonal_vectors_have_zero_similarity(db):
    await mgr.create(db, _cache("orth", _unit(1)))
    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=5)
    assert len(results) == 1
    _, sim = results[0]
    assert sim == pytest.approx(0.0, abs=1e-4)


async def test_partial_overlap_similarity(db):
    # _mixed(0, 1) has cosine sim ≈ 1/√2 with _unit(0)
    await mgr.create(db, _cache("partial", _mixed(0, 1)))
    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=5)
    _, sim = results[0]
    assert sim == pytest.approx(1 / math.sqrt(2), abs=1e-4)


async def test_results_ordered_by_similarity_desc(db):
    await mgr.create(db, _cache("identical", _unit(0)))
    await mgr.create(db, _cache("partial", _mixed(0, 1)))
    await mgr.create(db, _cache("orthogonal", _unit(1)))

    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=10)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0][0].question_text == "identical"


async def test_threshold_filters_low_similarity(db):
    await mgr.create(db, _cache("high", _unit(0)))  # sim ≈ 1.0
    await mgr.create(db, _cache("low", _unit(1)))  # sim ≈ 0.0

    results = await mgr.similarity_search(db, _unit(0), threshold=0.9, limit=10)
    texts = [r.question_text for r, _ in results]
    assert "high" in texts
    assert "low" not in texts


async def test_limit_respected(db):
    for i in range(5):
        await mgr.create(db, _cache(f"item_{i}", _unit(0)))

    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=3)
    assert len(results) == 3


async def test_empty_store_returns_no_results(db):
    results = await mgr.similarity_search(db, _unit(0), threshold=0.0, limit=5)
    assert results == []
