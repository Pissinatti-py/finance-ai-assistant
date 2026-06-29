from langchain_anthropic import ChatAnthropic

from app.config import settings

# ponytail: single shared chat model — created on first use, reused after.
_llm: ChatAnthropic | None = None


def get_llm() -> ChatAnthropic:
    """
    Return the shared ChatAnthropic instance, creating it on first call.

    :return: The lazily-instantiated Anthropic chat model.
    :rtype: ChatAnthropic
    """
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        )
    return _llm
