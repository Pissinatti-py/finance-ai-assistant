import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent import embeddings as _emb
from app.db.managers.knowledge_manager import knowledge_manager
from app.db.session import SessionLocal
from app.schemas.knowledge import KnowledgeChunkCreate

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


async def add_chunks(content: str, source: str | None = None, metadata: dict | None = None) -> int:
    """
    Split text into chunks, embed them, and store them in the knowledge base.

    :param content: The full text to ingest.
    :type content: str
    :param source: An optional origin identifier stored on each chunk.
    :type source: str | None
    :param metadata: Optional free-form metadata stored on each chunk.
    :type metadata: dict | None
    :return: The number of chunks created.
    :rtype: int
    """
    chunks = _splitter.split_text(content)
    vecs = await _emb.client.aembed_documents(chunks)
    async with SessionLocal() as db:
        for chunk, vec in zip(chunks, vecs, strict=True):
            await knowledge_manager.create(
                db,
                KnowledgeChunkCreate(content=chunk, embedding=vec, source=source, metadata_=metadata or {}),
                auto_commit=False,
            )
        await db.commit()
    logger.info(f"ingested chunks={len(chunks)} source={source}")
    return len(chunks)


async def retrieve(query: str, k: int = 4) -> list[dict]:
    """
    Retrieve the most semantically similar knowledge chunks for a query.

    :param query: The text to search for.
    :type query: str
    :param k: The maximum number of chunks to return.
    :type k: int
    :return: Dicts with ``content``, ``source`` and ``similarity``, best first.
    :rtype: list[dict]
    """
    vec = await _emb.client.aembed_query(query)
    async with SessionLocal() as db:
        results = await knowledge_manager.similarity_search(db, vec, threshold=0.0, limit=k)
    logger.debug(f"rag retrieve k={k} hits={len(results)}")
    return [{"content": r.content, "source": r.source, "similarity": sim} for r, sim in results]
