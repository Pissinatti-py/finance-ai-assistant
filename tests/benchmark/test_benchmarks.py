"""Opt-in wall-clock smoke benchmarks for the vector hot path.

Run explicitly: ``uv run pytest -m benchmark`` (needs the pgvector DB).
Bounds are deliberately generous — they catch order-of-magnitude regressions,
not micro changes. ponytail: generous bound; tighten only if a real regression appears.
"""

import random
import time

import pytest

from app.config import settings
from app.db.models.cache import SemanticCache

pytestmark = pytest.mark.benchmark

N_ROWS = 1000
N_QUERIES = 50
MAX_SECONDS = 5.0  # for N_QUERIES searches over N_ROWS — HNSW should be far under this


def _rand_vec() -> list[float]:
    return [random.random() for _ in range(settings.embedding_dim)]


async def test_similarity_search_throughput(db):
    from app.db.managers.cache_manager import semantic_cache_manager as mgr

    db.add_all(
        SemanticCache(
            question_text=f"q{i}",
            question_embedding=_rand_vec(),
            response_text=f"a{i}",
            sources=[],
        )
        for i in range(N_ROWS)
    )
    await db.commit()

    start = time.perf_counter()
    for _ in range(N_QUERIES):
        await mgr.similarity_search(db, _rand_vec(), threshold=0.0, limit=4)
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_SECONDS, f"{N_QUERIES} searches over {N_ROWS} rows took {elapsed:.2f}s"
