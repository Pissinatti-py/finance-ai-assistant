"""Skill framework tests: composition helpers, the finance-calc skill, graph wiring."""

import pytest

from app.agent.skills import DEFAULT_SKILLS, collect_instructions, collect_tools
from app.agent.skills.base import Skill
from app.agent.skills.finance_calc import FinanceCalcSkill, future_value

# ── compound-interest pure logic ──────────────────────────────────────────────


def test_future_value_principal_only():
    # 10000 at 12% a.a. (1%/month) for 5y = 10000 * 1.01**60 ≈ 18166.97
    assert abs(future_value(10000, 0.12, 5) - 18166.97) < 0.5


def test_future_value_zero_rate_is_principal_plus_deposits():
    assert future_value(1000, 0.0, 1, 100) == 1000 + 100 * 12


def test_future_value_with_contributions_exceeds_invested():
    fv = future_value(10000, 0.12, 5, 500)
    assert fv > 10000 + 500 * 60  # interest was earned on top of deposits


def test_future_value_rejects_invalid():
    with pytest.raises(ValueError):
        future_value(-1, 0.1, 5)
    with pytest.raises(ValueError):
        future_value(1000, 0.1, 0)


# ── finance-calc skill / tool ──────────────────────────────────────────────────


async def test_compound_interest_tool_returns_future_value():
    tool = FinanceCalcSkill().tools()[0]
    out = await tool.ainvoke({"principal": 10000, "annual_rate": 0.12, "years": 5, "monthly_contribution": 500})
    assert "Valor futuro" in out and "R$" in out


async def test_compound_interest_tool_handles_invalid_input():
    tool = FinanceCalcSkill().tools()[0]
    out = await tool.ainvoke({"principal": -1, "annual_rate": 0.1, "years": 5})
    assert "inválidos" in out


# ── composition helpers ───────────────────────────────────────────────────────


def test_collect_tools_aggregates_default_skills():
    tools = collect_tools(DEFAULT_SKILLS)
    names = {t.name for t in tools}
    # 3 transaction lookup + 1 knowledge + 1 finance_calc
    assert names == {
        "list_categories",
        "search_transactions",
        "summarize_spending",
        "retrieve_knowledge",
        "compound_interest",
    }


def test_collect_instructions_bullets_and_skips_none():
    text = collect_instructions(DEFAULT_SKILLS)
    # TransactionLookupSkill and FinanceCalcSkill have instructions; KnowledgeBaseSkill doesn't.
    assert text.startswith("- ")
    assert "compound_interest" in text
    assert text.count("\n") == 1  # exactly two instruction lines


def test_collect_instructions_empty_when_none():
    class _Bare(Skill):
        name = "bare"
        description = "no tools, no instructions"

        def tools(self, lang="pt-br"):
            return []

    assert collect_instructions([_Bare()]) == ""


# ── graph wiring with a custom skill set ──────────────────────────────────────


def test_build_graph_accepts_custom_skills():
    from app.agent.graph import build_graph

    graph = build_graph(skills=[FinanceCalcSkill()])
    assert hasattr(graph, "ainvoke")
