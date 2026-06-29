from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.db.base import Base


class KnowledgeChunk(Base):
    """
    A chunk of ingested knowledge with its embedding.

    Stored in the vector DB. ``embedding`` is indexed with HNSW (cosine) for
    semantic retrieval by the ``retrieve_knowledge`` tool. The Python attribute
    ``metadata_`` maps to the ``metadata`` column (SQLAlchemy reserves
    ``metadata``).
    """

    __tablename__ = "knowledge_chunks"

    __table_args__ = (
        Index(
            "ix_knowledge_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim), nullable=True)

    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
