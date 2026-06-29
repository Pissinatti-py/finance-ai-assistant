from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / ``.env``.

    Field names map to env vars case-insensitively (e.g. ``anthropic_model``
    reads ``ANTHROPIC_MODEL``). Unknown env vars are ignored.
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Single application database (Postgres + pgvector): owns the finance domain
    # tables, the semantic cache and the knowledge base. No external/productive DB.
    database_url: str

    # LLM: Anthropic API direct — runs with just ANTHROPIC_API_KEY, no AWS account.
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    # Embeddings: fastembed runs locally on CPU (no cloud, no credentials).
    # embedding_dim MUST match the model output (bge-small-en-v1.5 = 384).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # API key clients must send (X-API-Key) to reach the chat/ingest endpoints.
    ai_api_key: str

    cache_similarity_threshold: float = 0.92
    agent_max_iterations: int = 5  # caps the agent's reasoning loop per request
    max_context_tokens: int = 8000  # per-agent conversation history budget (approx tokens)
    log_level: str = "INFO"  # root log level (DEBUG also emits gen0/1 GC events)


settings = Settings()
