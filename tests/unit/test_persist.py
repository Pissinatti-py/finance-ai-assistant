"""persist node: source extraction and the never-cache-empty-answer guard.

No DB: nodes.store is mocked, so we assert *whether* a cache write happens and
what answer/sources are returned, not the storage itself.
"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import nodes as nodes_mod
from app.agent.nodes import _message_text, persist


@pytest.fixture
def store(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(nodes_mod, "store", mock)
    return mock


def _state(last, *, tool_outputs=()):
    messages = [HumanMessage("q")]
    messages += [ToolMessage(o, tool_call_id=str(i)) for i, o in enumerate(tool_outputs)]
    messages.append(last)
    return {"messages": messages, "question": "q", "question_embedding": [0.1] * 8}


async def test_non_empty_answer_is_cached_with_sources(store):
    out = await persist(_state(AIMessage("the answer"), tool_outputs=["src1", "src2"]))

    store.assert_awaited_once()
    assert out["answer"] == "the answer"
    assert out["sources"] == ["src1", "src2"]  # ToolMessage order preserved
    assert out["cached"] is False


async def test_empty_answer_is_not_cached(store):
    out = await persist(_state(AIMessage("   ")))

    store.assert_not_awaited()  # caching whitespace would poison future lookups
    assert out["answer"] == "   "


def test_message_text_flattens_content_blocks():
    assert _message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _message_text("plain") == "plain"
    assert _message_text(None) == ""
