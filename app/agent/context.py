"""Conversation context management.

A WebSocket /chat session has no defined end, so the checkpointed message
history grows turn after turn — eventually overflowing the model's context
window and leaking memory in the in-process checkpointer.

:func:`trim_to_budget` bounds the history to an approximate-token budget,
keeping the most recent messages. Returning ``RemoveMessage`` markers (rather
than just a filtered list) lets the caller prune the *stored* state through the
``add_messages`` reducer, so both the LLM input and process memory stay bounded.

``start_on="human"`` guarantees the kept window begins at a user turn, so an
AIMessage with tool calls is never separated from its ToolMessages.
"""

import logging

from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.config import settings

logger = logging.getLogger(__name__)


def trim_to_budget(
    messages: list[BaseMessage], max_tokens: int | None = None
) -> tuple[list[BaseMessage], list[RemoveMessage]]:
    """
    Trim conversation history to the token budget, keeping the most recent turns.

    :param messages: The current conversation messages (excluding the system prompt).
    :type messages: list[BaseMessage]
    :param max_tokens: Override for the budget; defaults to ``MAX_CONTEXT_TOKENS``.
    :type max_tokens: int | None
    :return: ``(kept, to_remove)`` — the messages to send to the model, and
        ``RemoveMessage`` markers for the ones pruned from stored state.
    :rtype: tuple[list[BaseMessage], list[RemoveMessage]]
    """
    budget = max_tokens if max_tokens is not None else settings.max_context_tokens
    kept = trim_messages(
        messages,
        max_tokens=budget,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    kept_ids = {m.id for m in kept}
    to_remove = [RemoveMessage(id=m.id) for m in messages if m.id not in kept_ids]
    if to_remove:
        logger.info(f"context trimmed messages_removed={len(to_remove)} kept={len(kept)}")
    return kept, to_remove
