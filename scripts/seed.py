"""Seed the database with demo data so the assistant is useful out of the box.

Run once after the database is up::

    python -m scripts.seed          # native
    docker compose run --rm seed    # docker

Creates the schema, a set of income/expense categories, ~6 months of
deterministic (seeded RNG) transactions, and ingests the Markdown knowledge
base into the RAG store. Idempotent: re-running skips data that already exists.
"""

import asyncio
import logging
import random
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import func, select, text

from app.agent.embeddings import init_embeddings
from app.db.base import Base
from app.db.models import KnowledgeChunk  # noqa: F401 — ensure models register on Base
from app.db.models.finance import Category, Transaction
from app.db.session import SessionLocal, engine
from app.rag.store import add_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("seed")

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

# kind, name. Income first so the agent has earnings to compare against.
CATEGORIES: list[tuple[str, str]] = [
    ("salário", "income"),
    ("freelance", "income"),
    ("aluguel", "expense"),
    ("mercado", "expense"),
    ("transporte", "expense"),
    ("restaurante", "expense"),
    ("lazer", "expense"),
    ("saúde", "expense"),
    ("assinaturas", "expense"),
]

# Per expense category: descriptions, (min, max) amount, count per month.
EXPENSE_PLAN: dict[str, tuple[list[str], tuple[float, float], int]] = {
    "aluguel": (["Aluguel apartamento"], (1800, 1800), 1),
    "mercado": (["Supermercado", "Hortifruti", "Mercado da esquina"], (80, 420), 6),
    "transporte": (["Combustível", "Aplicativo de transporte", "Recarga bilhete"], (15, 220), 5),
    "restaurante": (["Restaurante", "Delivery", "Padaria"], (25, 160), 6),
    "lazer": (["Cinema", "Streaming avulso", "Show", "Livraria"], (20, 300), 2),
    "saúde": (["Farmácia", "Consulta", "Academia"], (40, 350), 2),
    "assinaturas": (["Assinatura de streaming", "Plano de celular", "Software"], (20, 120), 3),
}

MONTHS_BACK = 6


async def _ensure_schema() -> None:
    """Create the pgvector extension and all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def _seed_finance() -> None:
    """Insert categories and ~6 months of deterministic transactions."""
    rng = random.Random(42)
    async with SessionLocal() as db:
        existing = (await db.execute(select(func.count(Transaction.id)))).scalar()
        if existing:
            logger.info("transactions already present (%s) — skipping finance seed", existing)
            return

        cats: dict[str, Category] = {}
        for name, kind in CATEGORIES:
            cat = (await db.execute(select(Category).where(Category.name == name))).scalar_one_or_none()
            if cat is None:
                cat = Category(name=name, kind=kind)
                db.add(cat)
                await db.flush()
            cats[name] = cat

        # First day of each of the last MONTHS_BACK months, newest first.
        month_starts: list[date] = []
        cursor = date.today().replace(day=1)
        for _ in range(MONTHS_BACK):
            month_starts.append(cursor)
            cursor = (cursor - timedelta(days=1)).replace(day=1)

        rows: list[Transaction] = []
        for month_start in month_starts:

            def day_in_month(d: int, _start: date = month_start) -> date:
                # Clamp to 27 so the offset never spills into the next month.
                return _start + timedelta(days=min(d, 27) - 1)

            # Income: salary on the 5th, occasional freelance.
            rows.append(
                Transaction(
                    date=day_in_month(5),
                    description="Salário mensal",
                    amount=round(rng.uniform(4800, 5200), 2),
                    type="credit",
                    category_id=cats["salário"].id,
                )
            )
            if rng.random() < 0.5:
                rows.append(
                    Transaction(
                        date=day_in_month(rng.randint(10, 25)),
                        description="Projeto freelance",
                        amount=round(rng.uniform(600, 2500), 2),
                        type="credit",
                        category_id=cats["freelance"].id,
                    )
                )
            # Expenses per plan.
            for cat_name, (descs, (lo, hi), count) in EXPENSE_PLAN.items():
                for _ in range(count):
                    rows.append(
                        Transaction(
                            date=day_in_month(rng.randint(1, 27)),
                            description=rng.choice(descs),
                            amount=round(rng.uniform(lo, hi), 2),
                            type="debit",
                            category_id=cats[cat_name].id,
                        )
                    )

        db.add_all(rows)
        await db.commit()
        logger.info("seeded %s categories and %s transactions", len(cats), len(rows))


async def _seed_knowledge() -> None:
    """Ingest the Markdown knowledge base into the RAG store (if empty)."""
    async with SessionLocal() as db:
        existing = (await db.execute(select(func.count(KnowledgeChunk.id)))).scalar()
    if existing:
        logger.info("knowledge base already has %s chunks — skipping ingest", existing)
        return
    init_embeddings()
    total = 0
    for path in sorted(KB_DIR.glob("*.md")):
        total += await add_chunks(path.read_text(encoding="utf-8"), source=path.name)
    logger.info("ingested %s knowledge chunks from %s", total, KB_DIR)


async def main() -> None:
    await _ensure_schema()
    await _seed_finance()
    await _seed_knowledge()
    await engine.dispose()
    logger.info("seed complete")


if __name__ == "__main__":
    asyncio.run(main())
