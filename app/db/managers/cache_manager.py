from app.db.managers.vector import VectorManager
from app.db.models.cache import SemanticCache
from app.schemas.cache import SemanticCacheCreate


class SemanticCacheManager(VectorManager[SemanticCache, SemanticCacheCreate, SemanticCacheCreate]):
    """Vector manager for the semantic cache, searching on ``question_embedding``."""

    _embedding_field = "question_embedding"


semantic_cache_manager = SemanticCacheManager(SemanticCache)
