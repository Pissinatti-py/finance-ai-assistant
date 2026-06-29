from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state flowing through the agent graph.

    ``messages`` accumulates across turns via the add_messages reducer, so a
    thread (session_id) keeps its conversation history through the checkpointer.
    The remaining keys are the per-request inputs and the resolved output.
    """

    messages: Annotated[list, add_messages]
    question: str  # raw user question (used for cache lookup + persist)
    question_embedding: list[float]  # embedded once in check_cache, reused in persist
    lang: str  # response language, e.g. "pt-br"
    answer: str  # final answer (set by check_cache hit OR persist)
    sources: list[str]  # tool outputs / cached sources
    cached: bool  # True when answered from the semantic cache
