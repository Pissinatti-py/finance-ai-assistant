"""Finance calculation skill — example of a self-contained, non-DB capability.

Demonstrates the skill pattern: a pure-Python tool plus prompt instructions,
added to the agent without touching the graph. Compound interest with monthly
contributions is a real, common personal-finance question, so this is a
genuine capability rather than a toy.
"""

from langchain_core.tools import tool

from app.agent.skills.base import Skill


def future_value(principal: float, annual_rate: float, years: float, monthly_contribution: float = 0.0) -> float:
    """
    Future value of a principal plus fixed monthly contributions, compounded monthly.

    :param principal: The starting amount (must be >= 0).
    :type principal: float
    :param annual_rate: The nominal annual rate as a fraction (0.12 = 12% a.a.).
    :type annual_rate: float
    :param years: The investment horizon in years (must be > 0).
    :type years: float
    :param monthly_contribution: Amount added at the end of each month.
    :type monthly_contribution: float
    :raises ValueError: If principal/contribution is negative or years is not positive.
    :return: The future value rounded to two decimals.
    :rtype: float
    """
    if principal < 0 or monthly_contribution < 0 or years <= 0:
        raise ValueError("principal and monthly_contribution must be >= 0, years > 0")
    months = round(years * 12)
    r = annual_rate / 12  # monthly rate
    if r == 0:
        total = principal + monthly_contribution * months
    else:
        growth = (1 + r) ** months
        # principal compounds; each monthly deposit is an ordinary annuity.
        total = principal * growth + monthly_contribution * (growth - 1) / r
    return round(total, 2)


def _finance_calc_tools(lang: str = "pt-br") -> list:
    @tool
    async def compound_interest(
        principal: float, annual_rate: float, years: float, monthly_contribution: float = 0.0
    ) -> str:
        """Compute the future value of an investment with monthly compounding, given a principal, a nominal annual rate (as a fraction, e.g. 0.12 for 12%), a horizon in years, and an optional fixed monthly contribution. Use this for any 'how much will it grow / yield' question instead of estimating."""  # noqa: E501
        try:
            fv = future_value(principal, annual_rate, years, monthly_contribution)
        except ValueError:
            return "Valores inválidos: principal e aporte devem ser >= 0 e o prazo > 0."
        invested = principal + monthly_contribution * round(years * 12)
        return f"Valor futuro: R${fv:.2f} (total investido: R${invested:.2f}, juros: R${fv - invested:.2f})."

    return [compound_interest]


class FinanceCalcSkill(Skill):
    """Compute compound-interest future value from principal, rate, term and contributions."""

    name = "finance_calc"
    description = "Compute the future value of an investment with compound interest and monthly contributions."
    instructions = (
        "When the user asks how much an investment will grow or yield, call the "
        "compound_interest tool rather than estimating. Convert percentages to "
        "fractions (12% a.a. → 0.12) before calling."
    )

    def tools(self, lang: str = "pt-br") -> list:
        """Return the compound-interest tool."""
        return _finance_calc_tools(lang)


# Runnable check for the pure logic lives in tests/unit/test_skills.py
# (test_future_value_*), which exercises the principal-only, zero-rate and
# with-contributions cases.
