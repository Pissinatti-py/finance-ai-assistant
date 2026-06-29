from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

# ponytail: in-memory, per-process. Conversation history survives within a
# worker but is lost on restart and not shared across workers. For persistent,
# multi-worker history swap for AsyncPostgresSaver(engine) and call
# `await saver.setup()` in the FastAPI lifespan.
_checkpointer: BaseCheckpointSaver | None = None


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Return the shared checkpointer, creating it on first call.

    Shared across the per-language graphs so a session's history is consistent
    regardless of which graph handled a turn.

    :return: The conversation-state checkpointer.
    :rtype: BaseCheckpointSaver
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer
