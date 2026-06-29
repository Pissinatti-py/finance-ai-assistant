"""runner.py public entry points: stream_agent / run_agent event assembly.

stream_agent's own logic (token filtering, done assembly, session_id handling)
is tested against a fake graph whose astream yields controlled (mode, data)
items — deterministic, no langgraph token-streaming internals. The multi-turn
test uses the real graph to confirm MemorySaver retains history per thread.
"""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import embeddings as emb_mod
from app.agent import graph as graph_mod
from app.agent import nodes as nodes_mod
from app.agent import runner as runner_mod
from app.agent.runner import run_agent, stream_agent


class _Chunk:
    def __init__(self, content):
        self.content = content


class _FakeGraph:
    """astream replays a fixed list of (mode, data) items."""

    def __init__(self, items):
        self._items = items

    async def astream(self, *_args, **_kwargs):
        for item in self._items:
            yield item


def _use_graph(monkeypatch, items):
    monkeypatch.setattr(runner_mod, "get_graph", lambda lang: _FakeGraph(items))


async def test_stream_agent_emits_tokens_then_done(monkeypatch):
    _use_graph(
        monkeypatch,
        [
            ("messages", (_Chunk("Hello "), {"langgraph_node": "agent"})),
            ("messages", (_Chunk("skip"), {"langgraph_node": "tools"})),  # non-agent → dropped
            ("messages", (_Chunk("world"), {"langgraph_node": "agent"})),
            ("values", {"answer": "Hello world", "cached": False, "sources": ["s1"]}),
        ],
    )

    events = [e async for e in stream_agent("q", session_id="abc")]

    tokens = [e["content"] for e in events if e["type"] == "token"]
    done = events[-1]
    assert tokens == ["Hello ", "world"]
    assert done["type"] == "done"
    assert done["answer"] == "".join(tokens)
    assert done["cached"] is False
    assert done["sources"] == ["s1"]
    assert done["session_id"] == "abc"  # echoed back


async def test_stream_agent_generates_session_id_when_absent(monkeypatch):
    _use_graph(monkeypatch, [("values", {"answer": "a", "cached": False, "sources": []})])

    events = [e async for e in stream_agent("q", session_id=None)]

    assert events[-1]["session_id"]  # a fresh uuid, not None


async def test_cache_hit_yields_no_tokens(monkeypatch):
    _use_graph(monkeypatch, [("values", {"answer": "cached", "cached": True, "sources": []})])

    events = [e async for e in stream_agent("q")]

    assert all(e["type"] != "token" for e in events)
    assert events[-1]["cached"] is True
    assert events[-1]["answer"] == "cached"


async def test_run_agent_returns_done_payload(monkeypatch):
    _use_graph(
        monkeypatch,
        [
            ("messages", (_Chunk("hi"), {"langgraph_node": "agent"})),
            ("values", {"answer": "hi", "cached": False, "sources": []}),
        ],
    )

    result = await run_agent("q", session_id="sess")

    assert result == {"answer": "hi", "cached": False, "sources": [], "session_id": "sess"}
    assert "type" not in result


# ── real graph: multi-turn session memory ─────────────────────────────────────


class _FakeRunnable:
    async def ainvoke(self, messages):
        return AIMessage(content="resposta")


class _FakeLLM:
    def bind_tools(self, tools):
        return _FakeRunnable()


async def test_multiturn_shares_history(monkeypatch):
    monkeypatch.setattr(graph_mod, "get_llm", lambda: _FakeLLM())
    emb = MagicMock()
    emb.aembed_query = AsyncMock(return_value=[0.1] * 8)
    monkeypatch.setattr(emb_mod, "client", emb)
    monkeypatch.setattr(nodes_mod, "get_cached", AsyncMock(return_value=None))
    monkeypatch.setattr(nodes_mod, "store", AsyncMock())

    graph = graph_mod.build_graph("pt-br")
    cfg = {"configurable": {"thread_id": "mt-1"}}

    await graph.ainvoke({"messages": [HumanMessage("primeira")], "question": "primeira", "lang": "pt-br"}, config=cfg)
    s2 = await graph.ainvoke(
        {"messages": [HumanMessage("segunda")], "question": "segunda", "lang": "pt-br"}, config=cfg
    )

    humans = [m.content for m in s2["messages"] if isinstance(m, HumanMessage)]
    assert "primeira" in humans and "segunda" in humans  # ponytail: MemorySaver is per-process
