"""Public entry points for the agent.

stream_agent() is the streaming primitive (used by the WebSocket /chat route);
run_agent() is a non-streaming convenience wrapper that consumes it.
"""

import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage

from app.agent.graph import get_graph
from app.agent.nodes import _message_text
from app.config import settings

logger = logging.getLogger(__name__)


def _stream_token(mode: str, data) -> str | None:
    """
    Extract streamable answer text from one astream item.

    Only the agent node's text output is streamed; tool nodes and the empty
    chunks emitted while the model is deciding on tool calls are skipped.

    :param mode: The stream mode of the item ("messages" or "values").
    :type mode: str
    :param data: The item payload; for "messages" a (chunk, metadata) tuple.
    :type data: Any
    :return: The token text to stream, or None to skip this item.
    :rtype: str | None
    """
    if mode != "messages":
        return None
    chunk, meta = data
    if meta.get("langgraph_node") != "agent":
        return None
    return _message_text(chunk.content) or None


async def stream_agent(question: str, lang: str = "pt-br", session_id: str | None = None) -> AsyncIterator[dict]:
    """
    Run the agent and yield real-time events as the answer is produced.

    Emits ``{"type": "token", "content": str}`` for each answer fragment, then a
    final ``{"type": "done", "answer", "cached", "sources", "session_id"}``. A
    cache hit yields no tokens — just the final event with the cached answer.

    :param question: The user question.
    :type question: str
    :param lang: The response language code.
    :type lang: str
    :param session_id: An existing conversation thread, or None to start one.
    :type session_id: str | None
    :return: An async stream of event dicts.
    :rtype: AsyncIterator[dict]
    """
    graph = get_graph(lang)
    thread_id = session_id or str(uuid.uuid4())
    # PII-safe: log the question length, never its content (medical domain).
    logger.info(f"chat turn start session={thread_id} lang={lang} q_len={len(question)}")
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_max_iterations * 2 + 2,
    }
    final_state: dict = {}
    tokens = 0
    async for mode, data in graph.astream(
        {"messages": [HumanMessage(question)], "question": question, "lang": lang},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "values":
            final_state = data
            continue
        token = _stream_token(mode, data)
        if token:
            tokens += 1
            yield {"type": "token", "content": token}

    cached = bool(final_state.get("cached"))
    sources = final_state.get("sources") or []
    logger.info(f"chat turn done session={thread_id} cached={cached} tokens={tokens} sources={len(sources)}")
    yield {
        "type": "done",
        "answer": final_state.get("answer") or "",
        "cached": cached,
        "sources": sources,
        "session_id": thread_id,
    }


async def run_agent(question: str, lang: str = "pt-br", session_id: str | None = None) -> dict:
    """
    Answer a question and return the final result (non-streaming).

    Consumes :func:`stream_agent` and returns the payload of its ``done`` event.

    :param question: The user question.
    :type question: str
    :param lang: The response language code.
    :type lang: str
    :param session_id: An existing conversation thread, or None to start one.
    :type session_id: str | None
    :return: ``{"answer", "cached", "sources", "session_id"}``.
    :rtype: dict
    """
    result = {"answer": "", "cached": False, "sources": [], "session_id": session_id}
    async for event in stream_agent(question, lang, session_id):
        if event["type"] == "done":
            result = {k: v for k, v in event.items() if k != "type"}
    return result
