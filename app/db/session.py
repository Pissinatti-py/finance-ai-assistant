from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings


def _async_url(url: str) -> str:
    """
    Rewrite a ``postgres://`` URL to use the asyncpg driver.

    :param url: A standard PostgreSQL connection URL.
    :type url: str
    :return: The same URL using the ``postgresql+asyncpg`` scheme.
    :rtype: str
    """
    return url.replace("postgres://", "postgresql+asyncpg://", 1)


# Single application database (Postgres + pgvector): finance domain tables,
# semantic cache and knowledge base all live here. No external/productive DB.
engine = create_async_engine(_async_url(settings.database_url), pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
