"""Agent graph wiring tests.

No DB or Bedrock: the LLM is faked, the cache is mocked. Verifies the two
branches of the graph — cache hit short-circuits the agent, cache miss runs the
agent and persists. Tools are never invoked (the fake LLM returns no tool calls).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import embeddings as emb_mod
from app.agent import graph as graph_mod
from app.agent import nodes as nodes_mod

CONFIG = {"configurable": {"thread_id": "test-thread"}}


class _FakeRunnable:
    async def ainvoke(self, messages):
        return AIMessage(content="resposta do agente")


class _FakeLLM:
    def bind_tools(self, tools):
        return _FakeRunnable()


@pytest.fixture
def patched(monkeypatch):
    """Fake the LLM and stub cache I/O; return the mocks for assertions."""
    monkeypatch.setattr(graph_mod, "get_llm", lambda: _FakeLLM())
    emb = MagicMock()
    emb.aembed_query = AsyncMock(return_value=[0.1] * 8)
    monkeypatch.setattr(emb_mod, "client", emb)
    get_cached = AsyncMock(return_value=None)
    store = AsyncMock()
    monkeypatch.setattr(nodes_mod, "get_cached", get_cached)
    monkeypatch.setattr(nodes_mod, "store", store)
    return get_cached, store


async def test_cache_miss_runs_agent_and_persists(patched):
    get_cached, store = patched
    graph = graph_mod.build_graph("pt-br")

    state = await graph.ainvoke(
        {"messages": [HumanMessage("o que é diabetes?")], "question": "o que é diabetes?", "lang": "pt-br"},
        config=CONFIG,
    )

    assert state["cached"] is False
    assert state["answer"] == "resposta do agente"
    get_cached.assert_awaited_once()
    store.assert_awaited_once()  # persisted to cache


async def test_cache_hit_short_circuits_agent(patched):
    get_cached, store = patched
    get_cached.return_value = {"answer": "do cache", "sources": ["art-1"]}
    graph = graph_mod.build_graph("pt-br")

    state = await graph.ainvoke(
        {"messages": [HumanMessage("q")], "question": "q", "lang": "pt-br"},
        config=CONFIG,
    )

    assert state["cached"] is True
    assert state["answer"] == "do cache"
    assert state["sources"] == ["art-1"]
    store.assert_not_awaited()  # agent never ran, nothing to persist
