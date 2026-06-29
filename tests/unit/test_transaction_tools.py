"""Transaction/category lookup tools (app/agent/tools/transactions.py).

DB-backed: the tools' internal SessionLocal is patched to the test's
SAVEPOINT-isolated connection (mirrors test_integration.patch_sessions), and
rows are seeded on the same connection so the tools see them.
"""

import datetime as dt

import pytest

from app.agent.tools.transactions import _date_expressions, _parse_date, transaction_tools
from app.db.models.finance import Category, Transaction


@pytest.fixture(autouse=True)
def patch_session(session_factory, monkeypatch):
    import app.agent.tools.transactions as tx_mod

    monkeypatch.setattr(tx_mod, "SessionLocal", session_factory)


async def _seed(db):
    food = Category(name="Mercado", kind="expense")
    salary = Category(name="Salário", kind="income")
    db.add_all([food, salary])
    await db.flush()
    db.add_all(
        [
            Transaction(
                date=dt.date(2026, 1, 5),
                description="Compra supermercado",
                amount=30,
                type="debit",
                category_id=food.id,
            ),
            Transaction(date=dt.date(2026, 1, 6), description="Padaria", amount=20, type="debit", category_id=food.id),
            Transaction(
                date=dt.date(2026, 1, 31),
                description="Pagamento salário",
                amount=100,
                type="credit",
                category_id=salary.id,
            ),
        ]
    )
    await db.commit()
    return {"food": food.id, "salary": salary.id}


# ── pure date helpers ─────────────────────────────────────────────────────────


def test_parse_date():
    assert _parse_date("2026-01-05") == dt.date(2026, 1, 5)
    assert _parse_date("") is None
    assert _parse_date("not-a-date") is None


def test_date_expressions_count():
    assert len(_date_expressions("", "")) == 0
    assert len(_date_expressions("2026-01-01", "")) == 1
    assert len(_date_expressions("2026-01-01", "2026-12-31")) == 2
    assert len(_date_expressions("bad", "also-bad")) == 0  # malformed dropped


# ── list_categories ───────────────────────────────────────────────────────────


async def test_list_categories_empty(db):
    list_categories, _, _ = transaction_tools()
    assert "Nenhuma categoria" in await list_categories.ainvoke({})


async def test_list_categories_returns_rows(db):
    await _seed(db)
    list_categories, _, _ = transaction_tools()
    out = await list_categories.ainvoke({})
    assert "Mercado" in out and "Salário" in out
    assert "tipo=expense" in out


# ── search_transactions ───────────────────────────────────────────────────────


async def test_search_no_match(db):
    await _seed(db)
    _, search, _ = transaction_tools()
    assert "Nenhuma transação" in await search.ainvoke({"query": "inexistente"})


async def test_search_filters_by_text_and_category(db):
    await _seed(db)
    _, search, _ = transaction_tools()
    out = await search.ainvoke({"query": "Padaria"})
    assert "Padaria" in out and "supermercado" not in out

    out = await search.ainvoke({"category": "Salário"})
    assert "salário" in out.lower() and "Padaria" not in out


async def test_search_filters_by_date_range(db):
    await _seed(db)
    _, search, _ = transaction_tools()
    out = await search.ainvoke({"since": "2026-01-06", "until": "2026-01-06"})
    assert "Padaria" in out
    assert "supermercado" not in out and "salário" not in out.lower()


async def test_search_limited_to_20_rows(db):
    food = Category(name="Mercado", kind="expense")
    db.add(food)
    await db.flush()
    db.add_all(
        [
            Transaction(
                date=dt.date(2026, 2, i % 28 + 1), description=f"item {i}", amount=1, type="debit", category_id=food.id
            )
            for i in range(25)
        ]
    )
    await db.commit()

    _, search, _ = transaction_tools()
    out = await search.ainvoke({"query": "item"})
    assert len(out.splitlines()) == 20


# ── summarize_spending ────────────────────────────────────────────────────────


async def test_summarize_totals(db):
    await _seed(db)
    _, _, summarize = transaction_tools()
    out = await summarize.ainvoke({})
    assert "R$50.00" in out  # 30 + 20 debits
    assert "R$100.00" in out  # credit


async def test_summarize_scoped_by_category(db):
    await _seed(db)
    _, _, summarize = transaction_tools()
    out = await summarize.ainvoke({"category": "Mercado"})
    assert "Mercado" in out and "R$50.00" in out
    assert "R$100.00" not in out  # salary excluded by category scope


async def test_summarize_unknown_category(db):
    await _seed(db)
    _, _, summarize = transaction_tools()
    out = await summarize.ainvoke({"category": "Inexistente"})
    assert "não encontrada" in out
