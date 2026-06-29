from typing import ClassVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.managers.base import BaseManager, CreateSchemaType, ModelType, UpdateSchemaType


class VectorManager(BaseManager[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Extends BaseManager with cosine similarity search over a vector column."""

    # Override in subclasses to name the embedding column on the model
    _embedding_field: ClassVar[str] = "embedding"

    async def similarity_search(
        self,
        db: AsyncSession,
        embedding: list[float],
        *,
        threshold: float = 0.0,
        limit: int = 4,
    ) -> list[tuple[ModelType, float]]:
        """Return (instance, cosine_similarity) ordered by similarity desc.

        threshold=0.0 returns all results; raise it (e.g. 0.92) to filter noise.
        """
        vec_col = getattr(self.model, self._embedding_field)
        # Order by raw cosine distance asc so pgvector's HNSW index is used; the
        # similarity threshold becomes a distance bound (sim >= t ⟺ dist <= 1 - t).
        distance = vec_col.cosine_distance(embedding)
        query = (
            select(self.model, distance.label("distance"))
            .where(distance <= 1 - threshold)
            .order_by(distance.asc())
            .limit(limit)
        )
        rows = (await db.execute(query)).all()
        return [(row[0], 1.0 - float(row[1])) for row in rows]
