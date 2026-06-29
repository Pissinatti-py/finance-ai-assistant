"""Structural performance guards (deterministic — no wall-clock timing).

These protect the cheap invariants of the hot path: the question is embedded
once per turn (not re-embedded for the cache write), and a cache hit never
reaches the LLM. Wall-clock benchmarks live in tests/benchmark/.
"""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import embeddings as emb_mod
from app.agent import graph as graph_mod
from app.agent import nodes as nodes_mod

CONFIG = {"configurable": {"thread_id": "perf-thread"}}


class _FakeRunnable:
    def __init__(self, spy):
        self._spy = spy

    async def ainvoke(self, messages):
        self._spy()
        return AIMessage(content="resposta")


class _FakeLLM:
    def __init__(self, spy):
        self._spy = spy

    def bind_tools(self, tools):
        return _FakeRunnable(self._spy)


def _wire(monkeypatch, *, cache_hit):
    calls = {"llm": 0}
    monkeypatch.setattr(graph_mod, "get_llm", lambda: _FakeLLM(lambda: calls.__setitem__("llm", calls["llm"] + 1)))
    emb = MagicMock()
    emb.aembed_query = AsyncMock(return_value=[0.1] * 8)
    monkeypatch.setattr(emb_mod, "client", emb)
    hit = {"answer": "cached", "sources": []} if cache_hit else None
    monkeypatch.setattr(nodes_mod, "get_cached", AsyncMock(return_value=hit))
    monkeypatch.setattr(nodes_mod, "store", AsyncMock())
    return emb, calls


async def test_question_embedded_once_per_turn(monkeypatch):
    emb, _ = _wire(monkeypatch, cache_hit=False)
    graph = graph_mod.build_graph("pt-br")

    await graph.ainvoke({"messages": [HumanMessage("q")], "question": "q", "lang": "pt-br"}, config=CONFIG)

    # check_cache embeds; persist reuses state["question_embedding"] — no second call.
    assert emb.aembed_query.await_count == 1


async def test_cache_hit_never_calls_llm(monkeypatch):
    _, calls = _wire(monkeypatch, cache_hit=True)
    graph = graph_mod.build_graph("pt-br")

    await graph.ainvoke({"messages": [HumanMessage("q")], "question": "q", "lang": "pt-br"}, config=CONFIG)

    assert calls["llm"] == 0  # short-circuited before the agent node
