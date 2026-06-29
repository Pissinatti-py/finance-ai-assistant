"""Structured-lookup tools over the finance domain (categories + transactions).

These mirror the manual-lookup tools of the original project, but query the
app's own seedable tables via the BaseManager/TransactionManager instead of an
external database. The agent calls them to ground answers in real numbers.
"""

from datetime import date

from langchain_core.tools import tool
from sqlalchemy import select

from app.db.managers.transaction_manager import transaction_manager
from app.db.models.finance import Category, Transaction
from app.db.session import SessionLocal


def _parse_date(value: str) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` string, or None if blank/malformed."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _date_expressions(since: str, until: str) -> list:
    """Build the date-range filter expressions shared by the tools."""
    exprs = []
    if (s := _parse_date(since)) is not None:
        exprs.append(Transaction.date >= s)
    if (u := _parse_date(until)) is not None:
        exprs.append(Transaction.date <= u)
    return exprs


def transaction_tools(lang: str = "pt-br") -> list:
    """
    Build the transaction/category lookup tools.

    :param lang: Accepted for signature consistency with other skills.
    :type lang: str
    :return: The lookup tools.
    :rtype: list
    """

    @tool
    async def list_categories() -> str:
        """List the available transaction categories (id, name and kind: income/expense). Call this to map a category name the user mentions to its id."""  # noqa: E501
        async with SessionLocal() as db:
            rows = (await db.execute(select(Category).order_by(Category.name))).scalars().all()
        if not rows:
            return "Nenhuma categoria cadastrada."
        return "\n".join(f"id={c.id} | nome={c.name} | tipo={c.kind}" for c in rows)

    @tool
    async def search_transactions(query: str = "", category: str = "", since: str = "", until: str = "") -> str:
        """Search transactions, most recent first. Filter by free text in the description (query), a category name, and/or an ISO date range (since/until, YYYY-MM-DD). All filters are optional; omit to broaden. Returns up to 20 rows."""  # noqa: E501
        stmt = select(Transaction, Category.name).join(Category, isouter=True)
        for expr in _date_expressions(since, until):
            stmt = stmt.where(expr)
        if query:
            stmt = stmt.where(Transaction.description.ilike(f"%{query}%"))
        if category:
            stmt = stmt.where(Category.name.ilike(f"%{category}%"))
        stmt = stmt.order_by(Transaction.date.desc()).limit(20)
        async with SessionLocal() as db:
            rows = (await db.execute(stmt)).all()
        if not rows:
            return "Nenhuma transação encontrada para esses filtros."
        return "\n".join(
            f"{t.date} | {t.description} | R${t.amount} | {t.type} | categoria={cat or 'sem'}" for t, cat in rows
        )

    @tool
    async def summarize_spending(category: str = "", since: str = "", until: str = "") -> str:
        """Total money out (debits) and in (credits) over an optional category and ISO date range (since/until, YYYY-MM-DD). Use this for 'how much did I spend/earn' questions instead of adding rows yourself."""  # noqa: E501
        exprs = _date_expressions(since, until)
        if category:
            async with SessionLocal() as db:
                cat = (
                    await db.execute(select(Category.id).where(Category.name.ilike(f"%{category}%")))
                ).scalar_one_or_none()
            if cat is None:
                return f"Categoria '{category}' não encontrada. Use list_categories para ver as opções."
            exprs.append(Transaction.category_id == cat)
        async with SessionLocal() as db:
            spent = await transaction_manager.sum_amount(db, filters={"type": "debit"}, expressions=exprs)
            earned = await transaction_manager.sum_amount(db, filters={"type": "credit"}, expressions=exprs)
        scope = f" na categoria '{category}'" if category else ""
        return f"Total gasto{scope}: R${spent:.2f} | Total recebido{scope}: R${earned:.2f}"

    return [list_categories, search_transactions, summarize_spending]
