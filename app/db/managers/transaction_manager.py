from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.managers.base import BaseManager
from app.db.models.finance import Category, Transaction
from app.schemas.finance import TransactionCreate


class TransactionManager(BaseManager[Transaction, TransactionCreate, TransactionCreate]):
    """Transaction CRUD (from BaseManager) plus an amount aggregate.

    ponytail: only a single SUM aggregate — add richer rollups (group-by
    category, monthly buckets) when a tool actually needs them.
    """

    async def sum_amount(
        self,
        db: AsyncSession,
        filters: dict[str, Any] | None = None,
        expressions: Sequence[ColumnElement] | None = None,
    ) -> float:
        """
        Sum the ``amount`` column over the rows matching the filters.

        :param db: The active async session.
        :type db: AsyncSession
        :param filters: A mapping of field name to value (or list of values).
        :type filters: dict[str, Any] | None
        :param expressions: Extra SQLAlchemy filter expressions (e.g. date ranges).
        :type expressions: Sequence[ColumnElement] | None
        :return: The total amount (0.0 when no rows match).
        :rtype: float
        """
        query = select(func.coalesce(func.sum(self.model.amount), 0))
        if expressions:
            for expr in expressions:
                query = query.where(expr)
        if filters:
            for name, value in filters.items():
                if hasattr(self.model, name):
                    col = getattr(self.model, name)
                    query = query.where(col.in_(value) if isinstance(value, list) else col == value)
        return float((await db.execute(query)).scalar())


transaction_manager = TransactionManager(Transaction)
category_manager = BaseManager[Category, Any, Any](Category)
