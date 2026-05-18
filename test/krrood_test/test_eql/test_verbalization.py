"""
Tests for EQL verbalization.

Coverage:
- Unit tests for individual node types (Variable, Literal, Attribute, Comparator,
  AND/OR/Not, ForAll/Exists, aggregators, Entity, SetOf).
- Integration tests derived from existing EQL tests:
    test_presentation_example, test_for_all, test_order_by_aggregation,
    test_complex_having_success, test_nested_rule_explanation,
    test_explanation_condition_graph_and_visualize,
    test_equivalent_to_contains_type_using_exists.
- Predicate template tests: HasType, ContainsType, custom predicates,
  and graceful fallback when no template is set.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Any

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import (
    an,
    a,
    entity,
    set_of,
    variable,
    variable_from,
    flat_variable,
    match_variable,
    inference,
    for_all,
    exists,
    not_,
    and_,
    or_,
)
from krrood.entity_query_language.predicate import HasType, Predicate, Triple
from krrood.entity_query_language.verbalization.verbalizer import (
    EQLVerbalizer,
    VerbalizationContext,
    verbalize_expression,
)
from ..dataset.department_and_employee import Department, Employee


@dataclass
class _Task:
    name: str
    completed: bool


@dataclass
class _Robot:
    name: str
    battery: int
    tasks: List[_Task]


from ..dataset.semantic_world_like_classes import (
    Apple,
    Body,
    Cabinet,
    Container,
    ContainsType,
    Drawer,
    FixedConnection,
    FruitBox,
    Handle,
    PrismaticConnection,
)


@dataclass
class AmountDetails:
    amount: float


@dataclass
class BankTransaction:
    amount_details: AmountDetails
    booking_date: datetime.datetime


# ── Unit tests: leaves ─────────────────────────────────────────────────────────


def test_verbalize_variable_first_mention():
    x = variable(int, [1, 2])
    assert verbalize_expression(x) == "an int"


def test_verbalize_variable_article_consonant():
    x = variable(Body, [])
    assert verbalize_expression(x) == "a Body"


def test_verbalize_variable_coreference():
    ctx = VerbalizationContext()
    v = EQLVerbalizer()
    x = variable(Handle, [])
    first = v.verbalize(x, ctx)
    second = v.verbalize(x, ctx)
    assert first == "a Handle"
    assert second == "the Handle"


def test_verbalize_literal_plain_value():
    lit = Literal(_value_=42)
    assert "42" in verbalize_expression(lit)


def test_verbalize_literal_type_object():
    lit = Literal(_value_=Apple)
    assert verbalize_expression(lit) == "Apple"


def test_verbalize_literal_tuple_of_types():
    lit = Literal(_value_=(Apple, Body))
    text = verbalize_expression(lit)
    assert "Apple" in text and "Body" in text


# ── Unit tests: MappedVariable chain ──────────────────────────────────────────


def test_verbalize_attribute_uses_of_form():
    emp = variable(Employee, [])
    text = verbalize_expression(emp.salary)
    assert "Employee" in text
    assert "salary" in text
    assert " of " in text


def test_verbalize_attribute_uses_of_form_all_hops():
    # single-hop also uses "of" form
    cab = variable(Cabinet, [])
    one_hop = verbalize_expression(cab.container)
    assert " of " in one_hop

    # Two-hop chain: Employee → department → name
    emp = variable(Employee, [])
    dept_name = emp.department.name
    text = verbalize_expression(dept_name)
    assert "Employee" in text
    assert "department" in text
    assert "name" in text
    assert " of " in text


def test_verbalize_index_access_merged_into_attribute():
    @dataclass
    class Robot:
        tasks: list

    r = variable(Robot, [])
    text = verbalize_expression(r.tasks[0])
    assert "Robot" in text
    assert "tasks" in text
    assert "[0]" in text


def test_verbalize_bool_attribute_predicative():
    @dataclass
    class _RobotActive:
        active: bool

    r = variable(_RobotActive, [])
    text = verbalize_expression(r.active)
    assert "_RobotActive" in text or "Robot" in text
    assert "is" in text
    assert "active" in text
    assert "of" not in text


def test_verbalize_bool_attribute_negated():
    @dataclass
    class _RobotActive:
        active: bool

    r = variable(_RobotActive, [])
    text = verbalize_expression(not_(r.active))
    assert "is not" in text
    assert "active" in text


def test_verbalize_indexed_bool_attribute_predicative():
    r = variable(_Robot, [])
    text = verbalize_expression(r.tasks[0].completed)
    assert "first" in text
    assert "tasks" in text
    assert "is" in text
    assert "completed" in text
    # must NOT be "completed of tasks[0] of …"
    assert "completed of" not in text


def test_verbalize_indexed_bool_attribute_negated():
    r = variable(_Robot, [])
    text = verbalize_expression(not_(r.tasks[0].completed))
    assert "first" in text
    assert "tasks" in text
    assert "is not" in text
    assert "completed" in text


def test_verbalize_second_index_ordinal():
    r = variable(_Robot, [])
    text = verbalize_expression(r.tasks[1].completed)
    assert "second" in text
    assert "tasks" in text
    assert "is" in text
    assert "completed" in text


def test_verbalize_non_bool_indexed_attribute_possession():
    r = variable(_Robot, [])
    text = verbalize_expression(r.tasks[0].name)
    # name is a str — should use possession/of form, NOT "is"
    assert "tasks" in text
    assert "name" in text
    assert " is " not in text


def test_verbalize_flat_variable_delegates_to_child():
    cab = variable(Cabinet, [])
    drawer_var = flat_variable(cab.drawers)
    text = verbalize_expression(drawer_var)
    assert "Cabinet" in text
    assert "drawers" in text


# ── Unit tests: comparators ────────────────────────────────────────────────────


@pytest.mark.parametrize("op,word", [
    ("__gt__", "greater than"),
    ("__lt__", "less than"),
    ("__ge__", "at least"),
    ("__le__", "at most"),
    ("__eq__", "is"),
    ("__ne__", "is not"),
])
def test_verbalize_comparator_operators(op, word):
    x = variable(int, [1])
    comp = getattr(x, op)(5)
    text = verbalize_expression(comp)
    assert word in text


def test_verbalize_comparator_greater_than():
    x = variable(int, [])
    text = verbalize_expression(x > 10)
    assert "greater than" in text
    assert "10" in text


def test_verbalize_comparator_at_least():
    x = variable(int, [])
    text = verbalize_expression(x >= 10)
    assert "at least" in text


# ── Unit tests: logical operators ─────────────────────────────────────────────


def test_verbalize_and_chain_flattening():
    x = variable(int, [])
    cond = and_(x > 1, x < 10, x != 5)
    text = verbalize_expression(cond)
    assert "greater than" in text
    assert "less than" in text
    assert "is not" in text
    assert ", and " in text


def test_verbalize_and_stops_at_or():
    x = variable(int, [])
    cond = and_(x > 1, or_(x < 10, x == 5))
    text = verbalize_expression(cond)
    assert "greater than" in text
    assert "either" in text  # the inner OR produces "either ..."


def test_verbalize_or_chain():
    x = variable(int, [])
    cond = or_(x > 10, x < 0)
    text = verbalize_expression(cond)
    assert "either" in text
    assert "greater than" in text
    assert "less than" in text


def test_verbalize_not():
    x = variable(int, [])
    text = verbalize_expression(not_(x > 5))
    assert "is not greater than" in text


def test_verbalize_not_comparator_gt():
    x = variable(int, [])
    assert "is not greater than" in verbalize_expression(not_(x > 50))
    assert "50" in verbalize_expression(not_(x > 50))


def test_verbalize_not_comparator_eq():
    x = variable(int, [])
    assert "is not" in verbalize_expression(not_(x == 5))


def test_verbalize_not_comparator_le():
    x = variable(int, [])
    assert "is not at most" in verbalize_expression(not_(x <= 100))


def test_verbalize_not_complex_fallback():
    x = variable(int, [])
    text = verbalize_expression(not_(or_(x > 50, x < 10)))
    assert text.startswith("not (")
    assert "either" in text


# ── Unit tests: aggregators ────────────────────────────────────────────────────


def test_verbalize_count():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.count(x))
    assert "number of" in text and "ints" in text


def test_verbalize_average():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.average(x))
    assert "average" in text and "ints" in text


def test_verbalize_sum():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.sum(x))
    assert "sum" in text and "ints" in text


def test_verbalize_max_min():
    x = variable(int, [1, 2])
    assert "maximum" in verbalize_expression(eql.max(x)) and "ints" in verbalize_expression(eql.max(x))
    assert "minimum" in verbalize_expression(eql.min(x)) and "ints" in verbalize_expression(eql.min(x))


# ── Integration: target test cases ────────────────────────────────────────────


def test_verbalize_presentation_example():
    robots = [
        _Robot("Robot1", 100, [_Task("Task1", True)]),
        _Robot("Robot3", 75, [_Task("Task5", False)]),
    ]
    r = variable(_Robot, robots)
    q = an(entity(r).where(r.battery > 50, not_(r.tasks[0].completed)))
    text = verbalize_expression(q)

    assert "Find" in text
    assert "Robot" in text
    assert "battery" in text
    assert "is greater than" in text
    assert "50" in text
    assert "first" in text
    assert "tasks" in text
    assert "is not completed" in text


def test_verbalize_for_all(handles_and_containers_world):
    world = handles_and_containers_world
    cabinets = variable(Cabinet, world.views)
    container_var = variable(Container, world.bodies)
    the_cabinet_container = eql.the(
        entity(container_var).where(container_var.name == "Container2")
    )
    query = an(
        entity(the_cabinet_container).where(
            for_all(cabinets.container, the_cabinet_container == cabinets.container)
        )
    )
    text = verbalize_expression(query)

    assert "for all" in text
    assert "Cabinets" in text
    assert "containers" in text
    assert "Container" in text
    assert "is" in text
    # no bare article after "for all"
    assert "for all a " not in text
    assert "for all an " not in text
    # bound variable reused in condition must use definite article
    assert "container of the Cabinet" in text


def test_verbalize_order_by_aggregation(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = flat_variable(cabinet.drawers)
    query = an(
        entity(cabinet)
        .grouped_by(cabinet)
        .ordered_by(eql.count(drawer), descending=True)
    )
    text = verbalize_expression(query)

    assert "Cabinet" in text
    assert "grouped by" in text
    assert "ordered by" in text
    assert "number" in text
    assert "Cabinets" in text
    assert "drawers" in text
    assert "descending" in text


def test_verbalize_complex_having(departments_and_employees_fixture):
    departments, employees = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    department = emp.department
    avg_salary = eql.average(emp.salary)
    query = a(
        set_of(department, avg_salary).grouped_by(department).having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    assert "Employees" in text
    assert "department" in text
    assert "average" in text
    assert "salaries" in text
    assert "grouped by" in text
    assert "having" in text
    assert "30000" in text


def test_verbalize_nested_rule(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = match_variable(FixedConnection, world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer_var = inference(Drawer)(
        container=fixed_connection.parent, handle=fixed_connection.child
    )
    # Wrap in entity() to trigger the if/then rule form
    text = verbalize_expression(entity(drawer_var))

    # if/then structure
    assert text.startswith("If "), f"Expected 'If' at start, got: {text!r}"
    assert "then" in text, f"Expected 'then' in: {text!r}"
    # IF clause introduces the FixedConnection antecedent
    assert "there's a FixedConnection" in text, f"Expected 'there's a FixedConnection' in: {text!r}"
    # IF clause describes its bindings as "whose" clauses
    assert "whose parent is the child of a PrismaticConnection" in text
    assert "whose child is a Handle" in text
    # THEN clause introduces the Drawer
    assert "there's a Drawer" in text, f"Expected 'there's a Drawer' in: {text!r}"
    # THEN clause uses "whose" for each binding
    assert "whose container is the parent of the FixedConnection" in text
    assert "whose handle is the child of the FixedConnection" in text


def test_verbalize_condition_graph_example():
    @dataclass(frozen=True)
    class Item:
        value: int

    val = variable_from([6])
    item_var = inference(Item)(value=val)
    query = entity(item_var).where(or_(and_(val > 5, val < 10), val == 11))
    text = verbalize_expression(query)

    assert "Item" in text
    assert "either" in text
    assert "greater than" in text
    assert "less than" in text
    assert "is" in text


def test_verbalize_has_type_with_exists():
    fb1 = FruitBox("FruitBox1", [Apple("apple"), Body("Body1")])
    fb2 = FruitBox("FruitBox2", [Body("Body3")])
    fb = variable(FruitBox, domain=[fb1, fb2])
    query = an(
        entity(fb).where(
            exists(fb, HasType(flat_variable(fb.fruits), Apple))
        )
    )
    text = verbalize_expression(query)

    assert "FruitBox" in text
    assert "exists" in text
    assert "Apple" in text
    assert "is of type" in text


# ── Predicate template tests ───────────────────────────────────────────────────


def test_verbalize_has_type_template():
    fruit = variable(Body, [])
    pred = HasType(fruit, Apple)
    text = verbalize_expression(pred)
    assert "Body" in text
    assert "is of type" in text
    assert "Apple" in text


def test_verbalize_has_type_tuple_of_types():
    fruit = variable(Body, [])
    pred = HasType(fruit, (Apple, Body))
    text = verbalize_expression(pred)
    assert "is of type" in text
    assert "Apple" in text
    assert "Body" in text


def test_verbalize_contains_type_template():
    fb = variable(FruitBox, [])
    pred = ContainsType(fb.fruits, Apple)
    text = verbalize_expression(pred)
    assert "contains an instance of" in text
    assert "Apple" in text
    assert "fruits" in text


def test_verbalize_custom_predicate_robotics_domain(handles_and_containers_world):
    @dataclass(eq=False)
    class IsReachable(Predicate):
        body: Any

        def __call__(self) -> bool:
            return True

        @classmethod
        def _verbalization_template_(cls) -> str:
            return "{body} is reachable"

    world = handles_and_containers_world
    handle = variable(Handle, world.bodies)
    pred = IsReachable(handle)
    text = verbalize_expression(pred)
    assert "Handle" in text
    assert "is reachable" in text


def test_verbalize_custom_predicate_employee_domain():
    @dataclass(eq=False)
    class WorksInDepartment(Predicate):
        employee: Any
        department: Any

        def __call__(self) -> bool:
            return self.employee.department == self.department

        @classmethod
        def _verbalization_template_(cls) -> str:
            return "{employee} works in {department}"

    emp = variable(Employee, [])
    dept = variable(Department, [])
    pred = WorksInDepartment(emp, dept)
    text = verbalize_expression(pred)
    assert "Employee" in text
    assert "works in" in text
    assert "Department" in text


def test_verbalize_predicate_no_template_fallback():
    @dataclass(eq=False)
    class HasHighSalary(Predicate):
        employee: Any
        threshold: float

        def __call__(self) -> bool:
            return self.employee.salary > self.threshold

    emp = variable(Employee, [])
    pred = HasHighSalary(emp, 50000.0)
    text = verbalize_expression(pred)
    assert "Employee" in text
    assert "HasHighSalary" in text
    assert "50000.0" in text


def test_verbalize_predicate_no_template_no_args_fallback():
    @dataclass(eq=False)
    class IsActive(Predicate):
        entity: Any

        def __call__(self) -> bool:
            return True

    emp = variable(Employee, [])
    pred = IsActive(emp)
    text = verbalize_expression(pred)
    assert "IsActive" in text


# ── Aggregator coreference & HAVING compact form ──────────────────────────────


def test_aggregator_coreference_second_mention_is_the(departments_and_employees_fixture):
    """Same aggregator expression in set_of and having → second mention prefixed with 'the'."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    avg_salary = eql.average(emp.salary)
    query = a(set_of(avg_salary).grouped_by(emp.department).having(avg_salary > 30000))
    text = verbalize_expression(query)

    # First mention (in set_of): "average of …"
    # Second mention (in having): "the average of …"
    assert "the average of" in text


def test_having_comparator_omits_is_copula(departments_and_employees_fixture):
    """HAVING condition should use compact comparator form without the 'is' copula."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    avg_salary = eql.average(emp.salary)
    query = a(
        set_of(emp.department, avg_salary).grouped_by(emp.department).having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    having_part = text[text.index("having"):]
    assert "is greater than" not in having_part
    assert "greater than" in having_part


def test_where_keeps_is_copula(departments_and_employees_fixture):
    """WHERE condition must still use the full 'is greater than' form."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    query = a(entity(emp).where(emp.salary > 30000))
    text = verbalize_expression(query)

    where_part = text[text.index("such that"):]
    assert "is greater than" in where_part


def test_having_compound_condition_all_compact(departments_and_employees_fixture):
    """AND/OR inside HAVING: every comparator should use compact form."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    avg_salary = eql.average(emp.salary)
    count_emp = eql.count(emp)
    query = a(
        set_of(emp.department, avg_salary)
        .grouped_by(emp.department)
        .having(and_(avg_salary > 30000, count_emp >= 2))
    )
    text = verbalize_expression(query)

    having_part = text[text.index("having"):]
    assert "is greater than" not in having_part
    assert "is at least" not in having_part
    assert "greater than" in having_part
    assert "at least" in having_part


def test_having_negated_comparator_compact(departments_and_employees_fixture):
    """NOT over a comparator inside HAVING should also omit 'is'."""
    _, _ = departments_and_employees_fixture
    emp = variable(Employee, domain=None)
    avg_salary = eql.average(emp.salary)
    query = a(
        set_of(emp.department, avg_salary)
        .grouped_by(emp.department)
        .having(not_(avg_salary > 30000))
    )
    text = verbalize_expression(query)

    having_part = text[text.index("having"):]
    assert "is not greater than" not in having_part
    assert "not greater than" in having_part


# ── Comparator "is" form ──────────────────────────────────────────────────────


def test_verbalize_comparator_eq_uses_is():
    x = variable(int, [])
    text = verbalize_expression(x == 5)
    assert "is 5" in text
    assert "equals" not in text


def test_verbalize_comparator_ne_uses_is_not():
    x = variable(int, [])
    text = verbalize_expression(x != 5)
    assert "is not" in text


def test_verbalize_not_comparator_ne():
    x = variable(int, [])
    text = verbalize_expression(not_(x != 5))
    assert "is" in text


def test_verbalize_having_compact_eq_uses_equals():
    """HAVING compact mode: == keeps 'equals' (not 'is') so the copula-less form is readable."""
    emp = variable(Employee, domain=None)
    count_emp = eql.count(emp)
    query = a(
        set_of(emp.department, count_emp)
        .grouped_by(emp.department)
        .having(count_emp == 2)
    )
    text = verbalize_expression(query)
    having_part = text[text.index("having"):]
    assert "equals" in having_part
    assert "is 2" not in having_part


# ── Non-predicate InstantiatedVariable natural-English form ───────────────────


def test_verbalize_inference_no_sub_query(doors_and_drawers_world):
    """inference(...) with plain variable bindings: 'where' clause, no 'such that'."""
    world = doors_and_drawers_world
    h_var = variable(Handle, world.bodies)
    c_var = variable(Container, world.bodies)
    drawer = inference(Drawer)(handle=h_var, container=c_var)
    text = verbalize_expression(drawer)

    assert "a Drawer" in text
    assert "where" in text
    assert "a Handle" in text
    assert "a Container" in text
    assert "such that" not in text


def test_verbalize_inference_repeated_entity_article(doors_and_drawers_world):
    """Same inner entity used for two fields: first mention 'a', second 'the'."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = match_variable(FixedConnection, world.connections)(
        parent=pc.child, child=handle
    )
    drawer = inference(Drawer)(container=fc.parent, handle=fc.child)
    text = verbalize_expression(drawer)

    assert "a FixedConnection" in text
    assert "the FixedConnection" in text
    # Indefinite article comes before definite in the binding section
    assert text.index("a FixedConnection") < text.index("the FixedConnection")


def test_verbalize_inference_literal_field(doors_and_drawers_world):
    """A child var that is a Python literal value is rendered inside the binding."""
    world = doors_and_drawers_world
    h_var = variable(Handle, world.bodies)
    c_var = variable(Container, world.bodies)
    drawer = inference(Drawer)(handle=h_var, container=c_var, correct=True)
    text = verbalize_expression(drawer)

    assert "Drawer" in text
    assert "Handle" in text
    assert "True" in text


def test_verbalize_double_nested_constraint_stack(doors_and_drawers_world):
    """
    InstantiatedVariable whose binding value is itself an InstantiatedVariable with
    Entity sub-query constraints.  The inner 'such that' must not leak into the outer
    verbalization — each level keeps its own constraint frame.
    """

    @dataclass
    class Wrapper:
        drawer: Any

    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = match_variable(FixedConnection, world.connections)(
        parent=pc.child, child=handle
    )
    drawer_var = inference(Drawer)(container=fc.parent, handle=fc.child)
    wrapper_var = inference(Wrapper)(drawer=drawer_var)
    text = verbalize_expression(wrapper_var)

    assert "a Wrapper" in text
    assert "a Drawer" in text
    # The "such that" clause belongs to the Drawer level, not leaked to Wrapper
    assert text.count("such that") == 1
    # Wrapper's binding references the full Drawer description inline
    assert "the drawer of the Wrapper is a Drawer" in text
    # Drawer's constraints are present
    assert "FixedConnection" in text
    assert "PrismaticConnection" in text


def test_verbalize_double_nested_with_outer_entity(doors_and_drawers_world):
    """
    Wrapper has both an inner InstantiatedVariable (Drawer) and its own direct
    Entity binding.  The inner and outer constraint frames must remain separate.
    """

    @dataclass
    class Wrapper:
        drawer: Any
        connection: Any

    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = match_variable(FixedConnection, world.connections)(
        parent=pc.child, child=handle
    )
    drawer_var = inference(Drawer)(container=fc.parent, handle=fc.child)

    # A second entity used directly by the outer Wrapper
    handle2 = variable(Handle, world.bodies)
    pc2 = variable(PrismaticConnection, world.connections)
    fc2 = match_variable(FixedConnection, world.connections)(
        parent=pc2.child, child=handle2
    )

    wrapper_var = inference(Wrapper)(drawer=drawer_var, connection=fc2.parent)
    text = verbalize_expression(wrapper_var)

    assert "a Wrapper" in text
    assert "a Drawer" in text
    # Both the Drawer's and Wrapper's sub-query constraints appear as "such that" clauses
    assert text.count("such that") == 2


# ── 2-argument Predicate triple form ─────────────────────────────────────────


def test_verbalize_triple():
    @dataclass(eq=False)
    class ConnectsTo(Triple):
        source: Any
        target: Any

        @property
        def subject(self) -> Any:
            return self.source

        @property
        def object(self) -> Any:
            return self.target

        def __call__(self) -> bool:
            return True

    src = variable(Body, [])
    tgt = variable(Handle, [])
    pred = ConnectsTo(src, tgt)
    text = verbalize_expression(pred)

    assert "Body" in text
    assert "connects to" in text
    assert "Handle" in text
    # Subject–predicate–object order
    assert text.index("Body") < text.index("Handle")

def test_verbalize_1arg_predicate_generic_fallback():
    """1-arg Predicate without template still uses generic constructor-like fallback."""

    @dataclass(eq=False)
    class IsActive(Predicate):
        entity: Any

        def __call__(self) -> bool:
            return True

    emp = variable(Employee, [])
    pred = IsActive(emp)
    text = verbalize_expression(pred)
    assert "IsActive" in text
    assert "Employee" in text


# ── Same-type variable disambiguation ─────────────────────────────────────────


def test_two_same_type_variables_are_disambiguated():
    """Two distinct variables of the same type must get numbered labels to tell them apart.

    Currently both variables produce 'an Employee', making the output ambiguous:
    'an Employee's salary is greater than an Employee's salary'
    Expected: 'Employee 1's salary is greater than Employee 2's salary'
    """
    emp1 = variable(Employee, [])
    emp2 = variable(Employee, [])
    cond = emp1.salary > emp2.salary
    text = verbalize_expression(cond)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"


def test_two_same_type_variables_in_query_are_disambiguated():
    """Two same-type variables inside a query must each get distinct numbered labels."""
    emp1 = variable(Employee, [])
    emp2 = variable(Employee, [])
    query = an(entity(emp1).where(emp1.salary > emp2.salary))
    text = verbalize_expression(query)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"


def test_three_same_type_variables_are_disambiguated():
    """Three variables of the same type each get distinct numbered labels."""
    emp1 = variable(Employee, [])
    emp2 = variable(Employee, [])
    emp3 = variable(Employee, [])
    query = an(entity(emp1).where(and_(emp1.salary > emp2.salary, emp2.salary > emp3.salary)))
    text = verbalize_expression(query)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"
    assert "Employee 3" in text, f"Expected 'Employee 3' in: {text!r}"


def test_single_type_variable_not_numbered():
    """A single variable of a type must keep the plain 'an Employee' form — no numbering."""
    emp = variable(Employee, [])
    text = verbalize_expression(emp)
    assert "an Employee" in text, f"Expected 'an Employee' in: {text!r}"
    assert "Employee 1" not in text, f"Did not expect numbering in: {text!r}"


def test_number_of_business_transactions():
    @dataclass
    class BankTransaction:
        amount: float
        date: str

    t = variable(BankTransaction, domain=None)
    query = an(entity(eql.count(t)))
    text = verbalize_expression(query)
    assert "BankTransaction" in text, f"Expected 'BankTransaction' in: {text!r}"
    assert "number of" in text, f"Expected 'number of' in: {text!r}"


def test_search_by_booking_date_and_aggregate():
    t = variable(BankTransaction, domain=None)
    query = an(entity(eql.sum(t.amount_details.amount)).where(t.booking_date < datetime.datetime(2024, 5, 17)))

    text = verbalize_expression(query)
    assert "BankTransaction" in text, f"Expected 'BankTransaction' in: {text!r}"
    assert "sum of" in text, f"Expected 'sum of' in: {text!r}"


def test_cabinet_rule_verbalization(handles_and_containers_world):
    """if/then form: aggregated antecedents plural, group-key binding uses 'common'."""
    drawer = variable(Drawer, handles_and_containers_world.views)
    prismatic_connection = variable(PrismaticConnection, handles_and_containers_world.connections)
    query = (
        entity(
            inference(Cabinet)(
                container=prismatic_connection.parent,
                drawers=drawer,
            )
        )
        .where(prismatic_connection.child == drawer.container)
        .grouped_by(prismatic_connection.parent)
    )
    text = verbalize_expression(query)
    assert "Cabinet" in text, f"Expected 'Cabinet' in: {text!r}"
    # if/then structure
    assert text.startswith("If "), f"Expected 'If' at start, got: {text!r}"
    assert "then" in text, f"Expected 'then' in: {text!r}"
    # IF clause introduces PrismaticConnections in plural (aggregated)
    assert "there are PrismaticConnections" in text, f"Expected 'there are PrismaticConnections' in: {text!r}"
    # IF clause contains a "whose children" constraint from the WHERE condition
    assert "whose children" in text, f"Expected 'whose children' in: {text!r}"
    # THEN clause introduces the Cabinet
    assert "there's a Cabinet" in text, f"Expected 'there's a Cabinet' in: {text!r}"
    # group-key binding uses "common"
    assert "whose container is the common parent of the PrismaticConnections" in text, \
        f"Expected 'whose container is the common parent of the PrismaticConnections' in: {text!r}"
    # aggregated plural binding
    assert "whose drawers are the Drawers" in text, f"Expected 'whose drawers are the Drawers' in: {text!r}"


def test_plural_field_binding_uses_are(handles_and_containers_world):
    """A plural field name in an InstantiatedVariable binding uses 'are <Plural>' not 'is <Article> <Singular>'."""
    drawer = variable(Drawer, handles_and_containers_world.views)
    cabinet = inference(Cabinet)(drawers=drawer)
    text = verbalize_expression(cabinet)
    assert "drawers of the Cabinet are Drawers" in text, f"Expected 'drawers of the Cabinet are Drawers' in: {text!r}"
    assert "drawers is" not in text, f"Did not expect 'drawers is' in: {text!r}"


def test_grouped_by_without_instantiated_variable(handles_and_containers_world):
    """grouped_by on a plain variable query falls back to 'grouped by X' (no aggregated subject)."""
    cabinet = variable(Cabinet, handles_and_containers_world.views)
    drawer = flat_variable(cabinet.drawers)
    query = an(
        entity(cabinet)
        .grouped_by(cabinet)
        .ordered_by(eql.count(drawer), descending=True)
    )
    text = verbalize_expression(query)
    # The selected variable IS the group key, so there are no extra aggregated nouns;
    # the sentence should still contain "grouped by" without crashing.
    assert "grouped by" in text, f"Expected 'grouped by' in: {text!r}"
    assert "Cabinet" in text, f"Expected 'Cabinet' in: {text!r}"


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def departments_and_employees_fixture():
    d1 = Department("HR")
    d2 = Department("Finance")
    e1 = Employee("John", d1, 10000)
    e2 = Employee("Anna", d2, 40000)
    return [d1, d2], [e1, e2]
