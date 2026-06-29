"""Finance domain models — the structured data the agent's tools query.

These are owned by the app and created by ``Base.metadata.create_all``; the
seed script populates them. A ``Category`` groups transactions (e.g. groceries,
salary); a ``Transaction`` is a single dated money movement.
"""

import datetime as dt

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Category(Base):
    """A spending/income category (e.g. ``mercado``, ``salário``)."""

    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    # "income" or "expense" — lets the agent separate earnings from spending.
    kind: Mapped[str] = mapped_column(String(16), nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="category")


class Transaction(Base):
    """A single dated money movement, optionally tagged with a category."""

    __tablename__ = "transactions"

    # Index the columns the tools filter/sort on most: date and category.
    __table_args__ = (Index("ix_transactions_date_category", "date", "category_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[dt.date] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # Numeric(12, 2): money — never float. Positive value; ``type`` gives direction.
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    # "debit" (money out) or "credit" (money in).
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(nullable=False, server_default=func.now())

    category: Mapped[Category | None] = relationship(back_populates="transactions")
