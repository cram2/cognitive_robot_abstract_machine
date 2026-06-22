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
    """An aggregation set-of with no GROUP BY is a calculation, so it opens with *"Report"* (not
    *"Find"*) and carries no *"for each"* frame."""
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(eql.sum(e.salary))))
    assert text == "Report the sum of salaries of Employees"
    assert "For each" not in text and "Find" not in text


def test_grouped_aggregation_reads_as_a_for_each_report():
    """A grouped aggregation set-of is a report, not a search: it fronts the grouping as
    *"For each <key>, report …"*, names the key bare (no *"of the Employee"*), and drops the
    redundant key column."""
    e = variable(Employee, [])
    text = verbalize_expression(
        a(set_of(e.department, eql.sum(e.salary)).grouped_by(e.department))
    )
    assert text == "For each department, report the sum of salaries of Employees"
    assert "grouped by" not in text  # fronted, not a trailing SQL-ish clause
    assert "of the Employee" not in text  # the key is the bare group label


def test_multi_root_set_of_does_not_pronominalise():
    """Two distinct root variables → no single focus → no pronoun, full chains kept."""
    e1 = variable(Employee, [])
    e2 = variable(Employee, [])
    text = verbalize_expression(a(set_of(e1.department, e2.salary)))
    assert "its " not in text
    assert "their " not in text


def test_unlimited_ordered_set_of_reports_pronominalised():
    """An unranked ordered set-of reports the plural population, pronominalised to "their"."""
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e).ordered_by(e.salary, descending=True)))
    assert text == "Report Employees ordered by their salary (descending)"


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
    assert text == (
        "Find the top three (the period of a ProfitAndLossStatement, "
        "the sum of the amount of the money of its revenue) "
        "such that the month of the begin of its period is the month of the end of its period, "
        "grouped by its period"
    )
    # the ranking conveys the ordering — no standalone clause, and the root appears once
    assert "ordered by" not in text
    assert "descending" not in text
    assert text.count("ProfitAndLossStatement") == 1


def test_set_of_limit_ascending_is_bottom():
    pnl = variable(ProfitAndLossStatement, domain=None)
    revenue = eql.sum(pnl.revenue.money.amount)
    q = a(
        set_of(pnl.period, revenue).grouped_by(pnl.period).ordered_by(revenue).limit(2)
    )
    text = verbalize_expression(q)
    assert text.startswith("Find the bottom two (")
    assert "ordered by" not in text
