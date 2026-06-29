"""Agent graph assembly.

    START → check_cache ─(hit)─→ END
                  │(miss)
                  ▼
               agent ⇄ tools
                  │(no tool calls)
                  ▼
               persist → END

One compiled graph is built per language (tool prompts/queries are
lang-scoped) and memoized. All graphs share a single checkpointer so a
session's history is consistent.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.checkpoint import get_checkpointer
from app.agent.context import trim_to_budget
from app.agent.llm import get_llm
from app.agent.nodes import check_cache, persist, route_after_cache
from app.agent.prompts import build_system_prompt
from app.agent.skills import DEFAULT_SKILLS, Skill, collect_instructions, collect_tools
from app.agent.state import AgentState

_graphs: dict[str, CompiledStateGraph] = {}


def build_graph(lang: str = "pt-br", skills: list[Skill] | None = None) -> CompiledStateGraph:
    """
    Compile the agent graph for a single response language and skill set.

    :param lang: The response language code; scopes the tools and system prompt.
    :type lang: str
    :param skills: The skills the agent is composed of. Defaults to
        ``DEFAULT_SKILLS`` (the main finance agent); pass a different list
        to build another agent.
    :type skills: list[Skill] | None
    :return: The compiled graph, wired to the shared checkpointer.
    :rtype: CompiledStateGraph
    """
    skills = DEFAULT_SKILLS if skills is None else skills
    tools = collect_tools(skills, lang)
    system_prompt = build_system_prompt(lang)
    skill_instructions = collect_instructions(skills)
    if skill_instructions:
        system_prompt = f"{system_prompt}\n\nAdditional capabilities:\n{skill_instructions}"

    async def agent(state: AgentState) -> dict:
        """Invoke the tool-bound LLM on the system prompt plus the (budgeted) history.

        History is trimmed to the token budget before each call; the trimmed
        messages are also pruned from stored state so memory stays bounded over
        long-lived WebSocket sessions.
        """
        kept, to_remove = trim_to_budget(state["messages"])
        llm = get_llm().bind_tools(tools)
        response = await llm.ainvoke([SystemMessage(system_prompt), *kept])
        return {"messages": [*to_remove, response]}

    g = StateGraph(AgentState)
    g.add_node("check_cache", check_cache)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("persist", persist)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges("check_cache", route_after_cache, {"agent": "agent", "end": END})
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "persist"})
    g.add_edge("tools", "agent")
    g.add_edge("persist", END)

    return g.compile(checkpointer=get_checkpointer())


def get_graph(lang: str = "pt-br") -> CompiledStateGraph:
    """
    Return the compiled graph for a language, building and memoizing on first use.

    :param lang: The response language code.
    :type lang: str
    :return: The memoized compiled graph for that language.
    :rtype: CompiledStateGraph
    """
    if lang not in _graphs:
        _graphs[lang] = build_graph(lang)
    return _graphs[lang]
