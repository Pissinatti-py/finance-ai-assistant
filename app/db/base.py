from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Declarative base for all application models.

    Everything the app owns — the finance domain (categories, transactions),
    the semantic cache and the knowledge base — inherits from this and is
    created by ``Base.metadata.create_all`` on the single application engine.
    """

    pass
