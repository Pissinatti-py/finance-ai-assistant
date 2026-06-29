"""TransactionManager + category lookups over the finance domain.

DB-backed (SAVEPOINT-isolated via the `db` fixture); skipped when no database
is available. Verifies the SUM aggregate and that BaseManager CRUD/filters work
on the new models.
"""

from datetime import date

from app.db.managers.transaction_manager import category_manager, transaction_manager
from app.db.models.finance import Category, Transaction
from app.schemas.finance import TransactionCreate


async def _seed(db) -> int:
    """Insert a category + a few transactions; return the category id."""
    cat = Category(name="mercado", kind="expense")
    db.add(cat)
    await db.flush()
    cid = cat.id
    rows = [
        TransactionCreate(date=date(2026, 1, 5), description="Supermercado", amount=100, type="debit", category_id=cid),
        TransactionCreate(date=date(2026, 1, 9), description="Hortifruti", amount=50, type="debit", category_id=cid),
        TransactionCreate(date=date(2026, 1, 9), description="Salário", amount=5000, type="credit", category_id=None),
    ]
    for r in rows:
        await transaction_manager.create(db, r)
    return cat.id


async def test_sum_amount_filters_by_type(db):
    await _seed(db)
    spent = await transaction_manager.sum_amount(db, filters={"type": "debit"})
    earned = await transaction_manager.sum_amount(db, filters={"type": "credit"})
    assert spent == 150.0
    assert earned == 5000.0


async def test_sum_amount_empty_is_zero(db):
    assert await transaction_manager.sum_amount(db, filters={"type": "debit"}) == 0.0


async def test_sum_amount_with_date_expression(db):
    await _seed(db)
    spent = await transaction_manager.sum_amount(
        db,
        filters={"type": "debit"},
        expressions=[Transaction.date >= date(2026, 1, 6)],
    )
    assert spent == 50.0  # only the Hortifruti debit on the 9th


async def test_category_manager_get_by_field(db):
    cat_id = await _seed(db)
    found = await category_manager.get_by_field(db, "name", "mercado")
    assert found is not None and found.id == cat_id


async def test_transaction_manager_count_filter(db):
    await _seed(db)
    n = await transaction_manager.count(db, filters={"type": "debit"})
    assert n == 2
