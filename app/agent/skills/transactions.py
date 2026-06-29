from app.agent.skills.base import Skill
from app.agent.tools.transactions import transaction_tools


class TransactionLookupSkill(Skill):
    """Structured lookups over the user's transactions and categories."""

    name = "transaction_lookup"
    description = (
        "Query the personal-finance database: list categories, search "
        "transactions by text/category/date, and total spending or income."
    )
    instructions = (
        "For any question about the user's own money (what they spent, earned, "
        "or where), call the transaction tools — list_categories to resolve a "
        "category name, summarize_spending for totals, search_transactions for "
        "individual entries. Never invent amounts."
    )

    def tools(self, lang: str = "pt-br") -> list:
        """Return the transaction/category lookup tools scoped to ``lang``."""
        return transaction_tools(lang)
