from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """
    Request body / message payload for a chat turn.

    Length/charset limits are a first line of defence against prompt-injection
    and abuse: an oversized ``question`` is a common jailbreak vector, and a
    constrained ``lang`` keeps the per-language graph cache bounded.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    lang: str = Field(default="pt-br", max_length=10, pattern=r"^[a-zA-Z-]+$")
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9-]+$")


class ChatResponse(BaseModel):
    """Final chat result returned to the client."""

    answer: str
    cached: bool
    sources: list[str]
    session_id: str | None = None  # echo back so the client can continue the thread


class IngestRequest(BaseModel):
    """Request body for ingesting a document into the knowledge base."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=100_000)
    source: str | None = Field(default=None, max_length=500)
    metadata: dict | None = None


class IngestResponse(BaseModel):
    """Result of an ingest operation."""

    chunks_stored: int
