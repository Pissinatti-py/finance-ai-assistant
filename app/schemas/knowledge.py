from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeChunkCreate(BaseModel):
    """Input schema for storing one knowledge chunk with its embedding."""

    content: str
    embedding: list[float]
    source: str | None = None
    metadata_: dict = {}


class KnowledgeChunkRead(BaseModel):
    """Output schema for a knowledge chunk (omits the raw embedding)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
    source: str | None
    metadata_: dict
    created_at: datetime
