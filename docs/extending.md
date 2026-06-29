# Extending the assistant

The agent, semantic cache, RAG and managers are domain-agnostic. Most changes
are small and localized. Each guide below is copy-paste sized.

## 1. Add a skill

A *skill* bundles tools plus an optional system-prompt fragment. Template:
[`app/agent/skills/finance_calc.py`](../app/agent/skills/finance_calc.py).

```python
# app/agent/skills/fx.py
from langchain_core.tools import tool
from app.agent.skills.base import Skill


class FxSkill(Skill):
    name = "fx"
    description = "Convert between currencies."
    instructions = "Call convert_currency for any currency-conversion request."

    def tools(self, lang: str = "pt-br") -> list:
        @tool
        async def convert_currency(amount: float, rate: float) -> str:
            """Convert an amount using the given exchange rate."""
            return f"{amount * rate:.2f}"
        return [convert_currency]
```

Register it in [`app/agent/skills/__init__.py`](../app/agent/skills/__init__.py)
by adding `FxSkill()` to `DEFAULT_SKILLS` — or pass it explicitly to
`build_graph(skills=[...])` to compose a different agent. No graph changes.

## 2. Add a tool to an existing skill

Tools are built by factory functions in `app/agent/tools/`. To add one to the
transaction skill, write a new `@tool` in
[`app/agent/tools/transactions.py`](../app/agent/tools/transactions.py) inside
`transaction_tools(...)` and append it to the returned list. The skill picks it
up automatically. Keep the docstring sharp — the model reads it to decide when
to call the tool.

## 3. Add a model + manager

1. Define the model on the shared `Base`
   ([`app/db/models/finance.py`](../app/db/models/finance.py) is the template)
   and export it from `app/db/models/__init__.py` so `create_all` registers it.
2. Add a Pydantic create schema in `app/schemas/`.
3. Instantiate a manager — usually just
   `my_manager = BaseManager[MyModel, MyCreate, MyUpdate](MyModel)`. Add custom
   methods only for queries `BaseManager` doesn't cover (see `TransactionManager.
   sum_amount` in
   [`app/db/managers/transaction_manager.py`](../app/db/managers/transaction_manager.py)).

## 4. Add vector (semantic) search to a model

Give the model a `Vector(settings.embedding_dim)` column with an HNSW index
(template: [`app/db/models/cache.py`](../app/db/models/cache.py)), then back it
with a `VectorManager` whose `_embedding_field` names that column (template:
[`app/db/managers/cache_manager.py`](../app/db/managers/cache_manager.py)). You
get `similarity_search` for free.

## 5. Swap the LLM or embeddings provider

- **LLM** — edit [`app/agent/llm.py`](../app/agent/llm.py). `get_llm()` returns a
  LangChain chat model; swap `ChatAnthropic` for any `BaseChatModel`
  (`ChatBedrock`, `ChatOpenAI`, …) and adjust the settings it reads.
- **Embeddings** — edit [`app/agent/embeddings.py`](../app/agent/embeddings.py).
  The cache and RAG layers only require `client.aembed_query(str)` and
  `client.aembed_documents(list[str])`. Give `client` any object with those two
  async methods (Voyage, Bedrock, OpenAI). Update `EMBEDDING_DIM` to match the
  new model's output, and reset the database (old vectors have the old dim).

## 6. Replace the finance domain entirely

The point of the boilerplate: keep the core, change the domain.

1. Remove `app/db/models/finance.py`, `app/agent/tools/transactions.py`,
   `app/agent/skills/transactions.py`, and `app/db/managers/transaction_manager.py`.
2. Add your own models + tools + skill following guides 1–3.
3. Point `DEFAULT_SKILLS` at your skills, rewrite the system prompt in
   [`app/agent/prompts.py`](../app/agent/prompts.py), and update
   [`scripts/seed.py`](../scripts/seed.py) to seed your data and KB docs.

The agent graph, semantic cache, RAG store, managers, API and CLI are untouched.
