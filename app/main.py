import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.agent.embeddings import init_embeddings
from app.agent.graph import get_graph
from app.api.routes import router
from app.db.base import Base
from app.db.models import Category, KnowledgeChunk, SemanticCache, Transaction  # noqa: F401 — register models
from app.db.session import engine
from app.logging_config import configure_logging
from app.observability import install_gc_monitor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook: configure logging + GC monitor, init embeddings,
    pre-build the graph, ensure schema.

    :param app: The FastAPI application instance.
    :type app: FastAPI
    """
    configure_logging()
    install_gc_monitor()
    logger.info("starting finance-ai-assistant")
    init_embeddings()
    get_graph()  # pre-build the default-language graph so import errors surface at boot
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("startup complete")
    yield
    logger.info("shutting down finance-ai-assistant")


app = FastAPI(title="Finance AI Assistant", lifespan=lifespan)

app.include_router(router)
