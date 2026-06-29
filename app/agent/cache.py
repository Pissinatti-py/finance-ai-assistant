import logging

from app.config import settings
from app.db.managers.cache_manager import semantic_cache_manager
from app.db.session import SessionLocal
from app.schemas.cache import SemanticCacheCreate

logger = logging.getLogger(__name__)


async def get_cached(embedding: list[float]) -> dict | None:
    """
    Look up a semantically similar cached answer for a question embedding.

    Runs a cosine similarity search above the configured threshold. The caller
    supplies the embedding so it can be computed once per turn and reused.

    :param embedding: The embedded user question to look up.
    :type embedding: list[float]
    :return: ``{"answer": str, "sources": list[str]}`` on a hit, else None.
    :rtype: dict | None
    """
    async with SessionLocal() as db:
        results = await semantic_cache_manager.similarity_search(
            db, embedding, threshold=settings.cache_similarity_threshold, limit=1
        )
    if results:
        obj, similarity = results[0]
        logger.info(f"cache hit similarity={similarity:.4f}")
        return {"answer": obj.response_text, "sources": list(obj.sources)}
    logger.info("cache miss")
    return None


async def store(question: str, embedding: list[float], answer: str, sources: list[str]) -> None:
    """
    Persist a question/answer pair (with embedding) in the semantic cache.

    :param question: The original user question.
    :type question: str
    :param embedding: The question embedding (computed once at cache lookup).
    :type embedding: list[float]
    :param answer: The agent's answer to cache.
    :type answer: str
    :param sources: The source identifiers consulted for the answer.
    :type sources: list[str]
    :return: Nothing.
    :rtype: None
    """
    obj_in = SemanticCacheCreate(
        question_text=question,
        question_embedding=embedding,
        response_text=answer,
        sources=sources,
    )
    async with SessionLocal() as db:
        await semantic_cache_manager.create(db, obj_in)
    logger.debug(f"cached answer sources={len(sources)}")
