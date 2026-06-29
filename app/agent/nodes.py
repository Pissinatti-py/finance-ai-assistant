"""Lang-independent graph nodes: cache short-circuit and persistence.

The agent/tools nodes live in graph.py because they close over the per-language
tool list. These two are pure I/O around the conversation and stay here.
"""

import logging

from langchain_core.messages import ToolMessage

from app.agent import embeddings as _emb
from app.agent.cache import get_cached, store
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


async def check_cache(state: AgentState) -> dict:
    """Look up the semantic cache for the incoming question.

    Embeds the question once here and stashes the vector in state so persist can
    reuse it for the cache write without a second embedding call.
    """
    vec = await _emb.client.aembed_query(state["question"])
    hit = await get_cached(vec)
    if hit:
        return {"answer": hit["answer"], "sources": hit["sources"], "cached": True}
    return {"cached": False, "question_embedding": vec}


def route_after_cache(state: AgentState) -> str:
    """Skip the agent entirely on a cache hit."""
    return "end" if state.get("cached") else "agent"


def _message_text(content) -> str:
    """Flatten an AIMessage content (str or Anthropic content blocks) to text."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


async def persist(state: AgentState) -> dict:
    """Extract the final answer, gather tool outputs as sources, cache it.

    An empty answer (e.g. the model returned only tool calls or failed to
    produce text) is never cached — caching it would poison future lookups.
    """
    answer = _message_text(state["messages"][-1].content)
    sources = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    if answer.strip():
        await store(state["question"], state["question_embedding"], answer, sources)
    else:
        logger.warning("persist: empty answer, skipping cache write")
    return {"answer": answer, "sources": sources, "cached": False}
