from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SemanticCacheCreate(BaseModel):
    """Input schema for storing a question/answer pair in the semantic cache."""

    question_text: str
    question_embedding: list[float]
    response_text: str
    sources: list[str] = []


class SemanticCacheRead(BaseModel):
    """Output schema for a semantic cache entry (omits the raw embedding)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    question_text: str
    response_text: str
    sources: list[str]
    created_at: datetime
