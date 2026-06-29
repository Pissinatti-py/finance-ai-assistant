"""Skill abstraction.

A *skill* is a reusable, self-contained capability: a named bundle of LangChain
tools plus an optional system-prompt fragment. An agent is composed from a list
of skills, so a new capability is a new ``Skill`` subclass — no change to the
graph — and a different agent simply picks a different subset of skills.

To add a skill:
1. Subclass :class:`Skill`, set ``name``/``description`` (and optionally
   ``instructions``), and implement :meth:`tools`.
2. Register it in ``app/agent/skills/__init__.py`` (``DEFAULT_SKILLS``), or pass
   it explicitly to ``build_graph(skills=[...])`` for another agent.
"""

from abc import ABC, abstractmethod


class Skill(ABC):
    """
    A reusable agent capability: tools plus optional prompt instructions.

    :cvar name: Stable identifier for the skill.
    :cvar description: Human-readable summary of what the skill enables.
    :cvar instructions: Optional system-prompt fragment appended to the agent's
        prompt when the skill is active (e.g. when to prefer a tool).
    """

    name: str
    description: str
    instructions: str | None = None

    @abstractmethod
    def tools(self, lang: str = "pt-br") -> list:
        """
        Return the LangChain tools this skill contributes.

        :param lang: The response language code, for tools that resolve
            translated content.
        :type lang: str
        :return: The tools to expose to the agent.
        :rtype: list
        """
        ...
