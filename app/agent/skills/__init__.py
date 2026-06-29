"""Skill registry and composition helpers.

``DEFAULT_SKILLS`` is the set the main finance agent runs with. Other agents
can build their own graph from a different subset via
``build_graph(skills=[...])``.
"""

from app.agent.skills.base import Skill
from app.agent.skills.finance_calc import FinanceCalcSkill
from app.agent.skills.knowledge_base import KnowledgeBaseSkill
from app.agent.skills.transactions import TransactionLookupSkill

DEFAULT_SKILLS: list[Skill] = [
    TransactionLookupSkill(),
    KnowledgeBaseSkill(),
    FinanceCalcSkill(),
]


def collect_tools(skills: list[Skill], lang: str = "pt-br") -> list:
    """
    Flatten the tools contributed by every skill.

    :param skills: The skills to draw tools from.
    :type skills: list[Skill]
    :param lang: The response language code passed to each skill.
    :type lang: str
    :return: All tools from all skills, in order.
    :rtype: list
    """
    tools: list = []
    for skill in skills:
        tools.extend(skill.tools(lang))
    return tools


def collect_instructions(skills: list[Skill]) -> str:
    """
    Join the prompt instructions of the skills that provide them.

    :param skills: The skills to gather instructions from.
    :type skills: list[Skill]
    :return: A bulleted block of instructions, or an empty string if none.
    :rtype: str
    """
    parts = [skill.instructions for skill in skills if skill.instructions]
    return "\n".join(f"- {p}" for p in parts)


__all__ = ["Skill", "DEFAULT_SKILLS", "collect_tools", "collect_instructions"]
