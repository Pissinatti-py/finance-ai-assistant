from datetime import date

from pydantic import BaseModel, ConfigDict


class CategoryCreate(BaseModel):
    """Input schema for creating a category."""

    name: str
    kind: str  # "income" or "expense"


class TransactionCreate(BaseModel):
    """Input schema for creating a transaction."""

    date: date
    description: str
    amount: float
    type: str  # "debit" or "credit"
    category_id: int | None = None


class TransactionRead(BaseModel):
    """Output schema for a transaction row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    description: str
    amount: float
    type: str
    category_id: int | None
