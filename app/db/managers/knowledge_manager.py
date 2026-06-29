from app.db.managers.vector import VectorManager
from app.db.models.knowledge import KnowledgeChunk
from app.schemas.knowledge import KnowledgeChunkCreate


class KnowledgeManager(VectorManager[KnowledgeChunk, KnowledgeChunkCreate, KnowledgeChunkCreate]):
    """Vector manager for the knowledge base, searching on ``embedding``."""

    _embedding_field = "embedding"


knowledge_manager = KnowledgeManager(KnowledgeChunk)
