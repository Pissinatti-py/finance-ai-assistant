"""Unit tests for the streaming token filter (no graph / DB / Bedrock)."""

from langchain_core.messages import AIMessageChunk

from app.agent.runner import _stream_token


def test_keeps_agent_node_text():
    chunk = AIMessageChunk(content="hello")
    assert _stream_token("messages", (chunk, {"langgraph_node": "agent"})) == "hello"


def test_skips_non_agent_node():
    chunk = AIMessageChunk(content="ignored")
    assert _stream_token("messages", (chunk, {"langgraph_node": "tools"})) is None


def test_skips_empty_content():
    # empty chunks are emitted while the model decides on tool calls
    chunk = AIMessageChunk(content="")
    assert _stream_token("messages", (chunk, {"langgraph_node": "agent"})) is None


def test_skips_values_mode():
    assert _stream_token("values", {"answer": "x"}) is None
