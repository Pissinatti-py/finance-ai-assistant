from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.db.base import Base


class SemanticCache(Base):
    """
    A cached question/answer pair with the question embedding.

    Stored in the vector DB. ``question_embedding`` is indexed with HNSW
    (cosine) so similar questions can short-circuit the agent.
    """

    __tablename__ = "semantic_cache"

    __table_args__ = (
        Index(
            "ix_semantic_cache_embedding_hnsw",
            "question_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"question_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    question_text: Mapped[str] = mapped_column(Text, nullable=False)

    question_embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim), nullable=True)

    response_text: Mapped[str] = mapped_column(Text, nullable=False)

    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
