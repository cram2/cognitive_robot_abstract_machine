"""
Tests for de-cluttered grouped ``set_of`` verbalization (ROUND 9):

- the literal *"sets of"* head is dropped — a set-of reads *"Find (v1, v2, …) …"*;
- a query whose chains all share one root variable pronominalises to *"its …"* / *"their …"*
  instead of restating the full root (a ``set_of`` previously had no discourse focus);
- a ``limit`` on a set-of renders as a ranking (*"Find the top three (…)"*) and suppresses the
  standalone *"ordered by …"* clause.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import a, variable, set_of
import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

from ...dataset.department_and_employee import Department, Employee


@dataclass
class Bound:
    month: int
    year: int = 0


@dataclass
class Period:
    begin: Bound
    end: Bound


@dataclass
class Money:
    amount: float


@dataclass
class Revenue:
    money: Money


@dataclass
class ProfitAndLossStatement:
    period: Period
    revenue: Revenue


# ── "sets of" is dropped ──────────────────────────────────────────────────────


def test_set_of_drops_the_sets_of_head():
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e.department, e.salary)))
    # the two attributes of one owner fold into a shared genitive, not "… and its salary"
    assert text == "Find the department and salary of an Employee"
    assert "sets of" not in text
    assert "(" not in text  # the code-like tuple parentheses are gone


def test_three_co_owned_attributes_fold_with_oxford_comma():
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e.name, e.salary, e.starting_salary)))
    assert text == "Find the name, salary, and starting_salary of an Employee"


def test_attributes_of_different_owners_do_not_fold():
    e = variable(Employee, [])
    d = variable(Department, [])
    text = verbalize_expression(a(set_of(e.salary, d.name)))
    assert text == "Find the salary of an Employee and the name of a Department"


# ── single-root pronominalisation ─────────────────────────────────────────────


def test_non_grouped_aggregation_reads_as_a_report():
    """
    An aggregation set-of with no GROUP BY is a calculation, so it opens with *"Report"*
    (not *"Find"*) and carries no *"for each"* frame.
    """
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(eql.sum(e.salary))))
    assert text == "Report the sum of salaries of Employees"
    assert "For each" not in text and "Find" not in text


def test_grouped_aggregation_reads_as_a_for_each_report():
    """
    A grouped aggregation set-of is a report, not a search: it fronts the grouping as
    *"For each <key>, report …"*, names the key bare (no *"of the Employee"*), and drops
    the redundant key column.
    """
    e = variable(Employee, [])
    text = verbalize_expression(
        a(set_of(e.department, eql.sum(e.salary)).grouped_by(e.department))
    )
    assert text == "For each department, report the sum of salaries of Employees"
    assert "grouped by" not in text  # fronted, not a trailing SQL-ish clause
    assert "of the Employee" not in text  # the key is the bare group label


def test_multi_root_set_of_does_not_pronominalise():
    """
    Two distinct root variables → no single focus → no pronoun, full chains kept.
    """
    e1 = variable(Employee, [])
    e2 = variable(Employee, [])
    text = verbalize_expression(a(set_of(e1.department, e2.salary)))
    assert "its " not in text
    assert "their " not in text


def test_unlimited_ordered_set_of_reports_pronominalised():
    """
    An unranked ordered set-of reports the plural population, pronominalised to "their".
    """
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e).ordered_by(e.salary, descending=True)))
    assert text == "Report Employees ordered by their salaries from highest to lowest"


# ── limit on a set_of → ranking, ordered-by suppressed ───────────────────────


def _pnl_query(n: int):
    pnl = variable(ProfitAndLossStatement, domain=None)
    revenue = eql.sum(pnl.revenue.money.amount)
    return a(
        set_of(pnl.period, revenue)
        .grouped_by(pnl.period)
        .where(pnl.period.begin.month == pnl.period.end.month)
        .ordered_by(revenue, descending=True)
        .limit(n)
    )


def test_set_of_limit_renders_ranking_and_suppresses_ordered_by():
    text = verbalize_expression(_pnl_query(3))
    # Ranked by an aggregate → "the three <entity> with the highest <aggregate>" (names the basis),
    # the aggregate reduced to "the sum" in the body, the such-that folded, no trailing "grouped by".
    assert text == (
        "For the three ProfitAndLossStatements "
        "with the highest sum of the amount of money of their revenue, "
        "report their period and the sum "
        "such that the begin and end of their period have the same month"
    )
    # the ranking conveys the ordering — no standalone clause
    assert "ordered by" not in text
    assert "descending" not in text
    assert "grouped by" not in text
    assert "(" not in text  # the code-like tuple parentheses are gone


def test_set_of_limit_ascending_is_bottom():
    pnl = variable(ProfitAndLossStatement, domain=None)
    revenue = eql.sum(pnl.revenue.money.amount)
    q = a(
        set_of(pnl.period, revenue).grouped_by(pnl.period).ordered_by(revenue).limit(2)
    )
    text = verbalize_expression(q)
    assert text.startswith("For the two ProfitAndLossStatements with the lowest sum")
    assert "(" not in text
    assert "ordered by" not in text
    assert "grouped by" not in text


# ── limit(1) on a grouped aggregation → superlative on the ranked aggregate ──


def _top_revenue_month_query(*, by_year: bool = False):
    pnl = variable(ProfitAndLossStatement, domain=None)
    keys = (
        [pnl.period.begin.year, pnl.period.begin.month]
        if by_year
        else [pnl.period.begin.month]
    )
    return a(
        set_of(*keys, eql.sum(pnl.revenue.money.amount))
        .grouped_by(*keys)
        .ordered_by(eql.sum(pnl.revenue.money.amount), descending=True)
        .limit(1)
    )


def test_grouped_aggregation_limit_one_ranks_the_aggregate():
    """
    A grouped aggregation taking the single highest group identifies the winner by the
    value it is ranked by — 'with the highest <aggregate>' — names the entity first so
    the body pronominalises to it, and reduces the aggregate to 'the sum' on its second
    mention.

    Never restates the grouping.
    """
    text = verbalize_expression(_top_revenue_month_query())
    assert text == (
        "For the ProfitAndLossStatement "
        "with the highest sum of the amount of money of its revenue, "
        "report the month of the begin of its period and the sum"
    )
    assert "grouped by" not in text


def test_grouped_aggregation_limit_one_folds_co_owned_keys():
    """
    Co-owned group keys (year, month of the same begin) fold into one genitive.
    """
    text = verbalize_expression(_top_revenue_month_query(by_year=True))
    assert text == (
        "For the ProfitAndLossStatement "
        "with the highest sum of the amount of money of its revenue, "
        "report the year and month of the begin of its period and the sum"
    )
    assert "grouped by" not in text


def test_grouped_aggregation_limit_one_ascending_is_lowest():
    """
    Ascending limit(1) reads as the lowest of the ranked aggregate.
    """
    pnl = variable(ProfitAndLossStatement, domain=None)
    query = a(
        set_of(pnl.period.begin.month, eql.sum(pnl.revenue.money.amount))
        .grouped_by(pnl.period.begin.month)
        .ordered_by(eql.sum(pnl.revenue.money.amount))
        .limit(1)
    )
    text = verbalize_expression(query)
    assert "the lowest sum of the amount of money of its revenue" in text
    assert "grouped by" not in text


def test_ranked_report_reduces_only_the_ranked_aggregate():
    """
    With two aggregates selected, only the one the query is ranked by is named in the
    frame and so reduces to 'the sum' in the body; the unranked aggregate keeps its full
    description.
    """
    pnl = variable(ProfitAndLossStatement, domain=None)
    ranked = eql.sum(pnl.revenue.money.amount)
    other = eql.average(pnl.revenue.money.amount)
    query = a(
        set_of(pnl.period.begin.month, ranked, other)
        .grouped_by(pnl.period.begin.month)
        .ordered_by(ranked, descending=True)
        .limit(1)
    )
    text = verbalize_expression(query)
    assert text == (
        "For the ProfitAndLossStatement "
        "with the highest sum of the amount of money of its revenue, "
        "report the month of the begin of its period, the sum, "
        "and the average of the amount of money of its revenue"
    )


def test_ranked_report_generalises_to_the_average_aggregate():
    """
    The aggregate-ranked framing is not sum-specific: ranking by an average names 'the
    highest average …' and reduces it to 'the average' in the body.
    """
    pnl = variable(ProfitAndLossStatement, domain=None)
    average = eql.average(pnl.revenue.money.amount)
    query = a(
        set_of(pnl.period.begin.month, average)
        .grouped_by(pnl.period.begin.month)
        .ordered_by(average, descending=True)
        .limit(1)
    )
    text = verbalize_expression(query)
    assert text == (
        "For the ProfitAndLossStatement "
        "with the highest average of the amount of money of its revenue, "
        "report the month of the begin of its period and the average"
    )
