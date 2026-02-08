from collections import defaultdict
from dataclasses import dataclass, field

import pytest
from typing_extensions import List

import krrood.entity_query_language.entity_result_processors as eql
from ..dataset.example_classes import NamedNumbers
from krrood.entity_query_language.entity import (
    variable,
    variable_from,
    entity,
    set_of,
    contains,
    distinct,
)
from krrood.entity_query_language.entity_result_processors import an, a
from krrood.entity_query_language.failures import (
    NonAggregatedSelectedVariablesError,
    AggregatorInWhereConditionsError,
    NestedAggregationError,
    UnsupportedAggregationOfAGroupedByVariable,
)
from krrood.entity_query_language.symbolic import Having, GroupBy
from ..dataset.department_and_employee import Department, Employee
from ..dataset.semantic_world_like_classes import Cabinet, Body, Container, Drawer


def test_count(handles_and_containers_world):
    world = handles_and_containers_world
    body = variable(type_=Body, domain=world.bodies)
    query = eql.count(
        entity(body).where(
            contains(body.name, "Handle"),
        )
    )
    assert query.tolist()[0] == len([b for b in world.bodies if "Handle" in b.name])


def test_sum(handles_and_containers_world):
    heights = [1, 2, 3, 4, 5]
    heights_var = variable(int, domain=heights)
    query = an(entity(eql.sum(heights_var)))
    assert query.tolist()[0] == sum(heights)
    assert eql.sum(heights_var).tolist()[0] == sum(heights)


def test_aggregate_distinct(handles_and_containers_world):
    heights = [1, 2, 3, 4, 4, 5, 5]
    heights_var = variable(int, domain=heights)
    assert eql.sum(heights_var, distinct=True).tolist()[0] == sum(set(heights))
    assert eql.count(heights_var, distinct=True).tolist()[0] == len(set(heights))
    assert eql.average(heights_var, distinct=True).tolist()[0] == sum(
        set(heights)
    ) / len(set(heights))
    assert eql.max(heights_var, distinct=True).tolist()[0] == max(set(heights))


@pytest.fixture
def test_numbers():
    test_numbers = [
        NamedNumbers("A", [1, 2, 3]),
        NamedNumbers("B", [4, 2]),
        NamedNumbers("C", [5]),
    ]
    return test_numbers


def test_grouping_already_grouped_by_object_attribute(test_numbers):
    test_numbers_var = variable(NamedNumbers, domain=test_numbers)

    assert eql.sum(test_numbers_var.numbers).grouped_by(test_numbers_var).tolist() == [
        6,
        6,
        5,
    ]


def test_distinct_sum(test_numbers):
    test_numbers_var = variable(NamedNumbers, domain=test_numbers)

    assert distinct(
        eql.sum(test_numbers_var.numbers).grouped_by(test_numbers_var)
    ).tolist() == [6, 5]


def test_average(handles_and_containers_world):
    heights = [1, 2, 3, 4, 5]
    heights_var = variable(int, domain=heights)
    query = an(entity(eql.average(heights_var)))
    assert list(query.evaluate())[0] == sum(heights) / len(heights)
    assert eql.average(heights_var).tolist()[0] == sum(heights) / len(heights)


def test_sum_on_empty_list(handles_and_containers_world):
    empty_var = variable(int, domain=[])
    query = an(entity(eql.sum(empty_var)))
    assert query.tolist()[0] is None
    assert eql.sum(empty_var).tolist()[0] is None
    assert eql.sum(empty_var, default=0).tolist()[0] == 0


def test_max_on_empty_list(handles_and_containers_world):
    empty_var = variable(int, domain=[])
    query = an(entity(eql.max(empty_var)))
    assert query.tolist()[0] is None
    assert eql.max(empty_var).tolist()[0] is None
    assert eql.max(empty_var, default=0).tolist()[0] == 0


def test_non_aggregated_selectables_with_aggregated_ones(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)
    with pytest.raises(NonAggregatedSelectedVariablesError):
        query = a(
            set_of(drawer, eql.max(drawer))
            .where(drawer.handle.name.startswith("H"))
            .grouped_by(cabinet)
        )
        _ = list(query.evaluate())


def test_non_aggregated_conditions_with_aggregated_ones(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)
    query = a(
        set_of(cabinet, eql.max(drawer.handle.name))
        .where(cabinet.container.name.startswith("C"))
        .grouped_by(cabinet)
    )
    _ = list(query.evaluate())
    with pytest.raises(AggregatorInWhereConditionsError):
        query = a(
            set_of(cabinet, max_handle_name := eql.max(drawer.handle.name))
            .where(max_handle_name.startswith("H"))
            .grouped_by(cabinet)
        )
        _ = list(query.evaluate())


def test_max_grouped_by(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)

    # We want to find the drawer with the "largest" handle name (alphabetically) per cabinet.
    # drawers_by_cabinet = variable_from(cabinet.drawers).grouped_by(cabinet)
    query = a(
        set_of(
            cabinet, max_drawer := eql.max(drawer, key=lambda d: d.handle.name)
        ).grouped_by(cabinet)
    )
    results = list(query.evaluate())

    # expected: for each cabinet, one result which is the drawer with max handle name
    expected_cabinets = [c for c in world.views if isinstance(c, Cabinet)]
    assert len(results) == len(expected_cabinets)

    for res in results:
        # res should have cabinet and the drawer in bindings
        c = res[cabinet]
        d = res[max_drawer]
        assert d in c.drawers
        assert d.handle.name == max(cd.handle.name for cd in c.drawers)


def test_having_with_max(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)

    query = a(
        set_of(
            cabinet,
            drawer_count := eql.count(drawer),
            eql.max(drawer, key=lambda d: d.handle.name),
        )
        .grouped_by(cabinet)
        .having(drawer_count > 1)
    )
    query.visualize()
    results = list(query.evaluate())
    assert len(results) == 1


def test_multiple_grouped_variables(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)

    # Group by both cabinet and drawer (silly, but tests multiple variables)
    query = a(
        set_of(cabinet, count := eql.count(drawer), drawer).grouped_by(cabinet, drawer)
    )
    results = list(query.evaluate())

    # Each result should have count=1 because each (cabinet, drawer) pair is unique here
    for res in results:
        assert res[count] == 1
        assert res[cabinet] is not None
        assert res[drawer] is not None


def test_sum_grouped_by(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)
    # Give drawers a numeric property to sum. They don't have one, but we can use a key func.
    # Let's sum the length of handle names per cabinet.

    query = a(
        set_of(
            total_characters := eql.sum(drawer, key=lambda d: len(d.handle.name)),
            cabinet,
        ).grouped_by(cabinet)
    )
    results = list(query.evaluate())

    expected_cabinets = [c for c in world.views if isinstance(c, Cabinet)]
    assert len(results) == len(expected_cabinets)

    for res in results:
        c = res[cabinet]
        s = res[total_characters]
        assert s == sum(len(d.handle.name) for d in c.drawers)


def test_count_grouped_by(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    query = an(entity(eql.count(cabinet_drawers)).grouped_by(cabinet))
    result = list(query.evaluate())
    expected = [len(c.drawers) for c in world.views if isinstance(c, Cabinet)]
    assert result == expected

    # without grouped_by should be all drawers of all cabinets
    query_all = an(entity(eql.count(cabinet_drawers)))
    results = list(query_all.evaluate())
    assert len(results) == 1
    result_all = results[0]
    expected_all = sum(len(c.drawers) for c in world.views if isinstance(c, Cabinet))
    assert result_all == expected_all


def test_count_all_or_without_a_specific_child(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    query = a(set_of(count := eql.count(), cabinet).grouped_by(cabinet))
    results = list(query.evaluate())
    expected = defaultdict(lambda: 0)
    for c in world.views:
        if isinstance(c, Cabinet):
            expected[c] += 1
    for result in results:
        assert result[count] == expected[result[cabinet]]


def test_count_variable_in_grouped_by_variables(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    query = set_of(count := eql.count(cabinet), cabinet).grouped_by(cabinet)
    results = query.tolist()
    expected = defaultdict(lambda: 0)
    for c in world.views:
        if isinstance(c, Cabinet):
            expected[c] += 1
    for result in results:
        assert result[count] == expected[result[cabinet]]


def test_non_count_aggregation_of_variable_in_grouped_by_variables(
    handles_and_containers_world,
):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    query = set_of(avg := eql.average(cabinet), cabinet).grouped_by(cabinet)
    with pytest.raises(UnsupportedAggregationOfAGroupedByVariable):
        results = query.tolist()


def test_count_variable_in_grouped_by_variables_selectnig_only_the_count(
    handles_and_containers_world,
):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    query = eql.count(cabinet).grouped_by(cabinet)
    results = query.tolist()
    expected = defaultdict(lambda: 0)
    for c in world.views:
        if isinstance(c, Cabinet):
            expected[c] += 1
    assert results == list(expected.values())


def test_count_with_duplicates(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet_with_duplicate_drawers = Cabinet(
        next(b for b in world.bodies if isinstance(b, Container)),
        [
            next(d for d in world.views if isinstance(d, Drawer)),
            next(d for d in world.views if isinstance(d, Drawer)),
        ],
        world=world,
    )
    world.views.append(cabinet_with_duplicate_drawers)
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawer = variable_from(cabinet.drawers)
    query = a(
        set_of(count := eql.count(), cabinet, cabinet_drawer).grouped_by(
            cabinet, cabinet_drawer
        )
    )
    results = list(query.evaluate())
    expected = defaultdict(lambda: 0)
    for c in world.views:
        if isinstance(c, Cabinet):
            for d in c.drawers:
                expected[(c, d)] += 1
    for result in results:
        print(result)
        assert result[count] == expected[(result[cabinet], result[cabinet_drawer])]


def test_max_count_grouped_by(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    query = eql.max(entity(eql.count(cabinet_drawers)).grouped_by(cabinet))
    result = query.tolist()
    assert len(result) == 1
    result_max = result[0]
    expected = 0
    for c in world.views:
        if isinstance(c, Cabinet) and len(c.drawers) > expected:
            expected = len(c.drawers)
    assert result_max == expected


def test_max_count_grouped_by_without_explicit_entity(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    query = eql.max(eql.count(cabinet_drawers).grouped_by(cabinet))
    result = query.tolist()
    assert len(result) == 1
    result_max = result[0]
    expected = 0
    for c in world.views:
        if isinstance(c, Cabinet) and len(c.drawers) > expected:
            expected = len(c.drawers)
    assert result_max == expected


def test_max_count_grouped_by_wrong(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    with pytest.raises(NestedAggregationError):
        query = eql.max(eql.count(cabinet_drawers))


def test_max_min_no_variable():
    values = [2, 1, 3, 5, 4]
    value = variable(int, domain=values)

    max_query = eql.max(entity(value))
    assert max_query.tolist()[0] == max(values)

    min_query = eql.min(entity(value))
    assert min_query.tolist()[0] == min(values)


def test_max_min_without_entity():
    values = [2, 1, 3, 5, 4]
    value = variable(int, domain=values)

    max_query = eql.max(value)
    assert max_query.tolist()[0] == max(values)

    min_query = eql.min(value)
    assert min_query.tolist()[0] == min(values)


@pytest.fixture
def departments_and_employees():
    d1 = Department("HR")
    d2 = Department("IT")
    d3 = Department("Finance")

    e1 = Employee("John", d1, 10000)
    e2 = Employee("Anna", d1, 20000)

    e3 = Employee("Anna", d2, 20000, 20000)
    e4 = Employee("Mary", d2, 30000, 30000)

    e5 = Employee("Peter", d3, 30000)
    e6 = Employee("Paul", d3, 40000)

    departments = [d1, d2, d3]
    employees = [e1, e2, e3, e4, e5, e6]
    return departments, employees


def test_average_with_condition(departments_and_employees):
    departments, employees = departments_and_employees

    emp = variable(Employee, domain=None)

    department = emp.department
    query = a(
        set_of(department, avg_salary := eql.average(emp.salary))
        .where(department.name.startswith("F"))
        .grouped_by(department)
        .having(avg_salary > 20000)
    )
    query.visualize()
    results = list(query.evaluate())
    assert len(results) == 1
    assert results[0][department] == next(
        d for d in departments if d.name.startswith("F")
    )


def test_multiple_aggregations_per_group_on_different_variables(
    departments_and_employees,
):
    departments, employees = departments_and_employees

    emp = variable(Employee, domain=None)
    department = emp.department
    avg_salary = eql.average(emp.salary)
    avg_starting_salary = eql.average(emp.starting_salary)
    query = a(
        set_of(avg_salary, avg_starting_salary, department)
        .grouped_by(department)
        .having(avg_salary == avg_starting_salary)
    )
    results = list(query.evaluate())
    assert len(results) == 1
    assert results[0][department] == next(
        d for d in departments if d.name.startswith("I")
    )


def test_multiple_aggregations_per_group_on_same_variable(departments_and_employees):
    departments, employees = departments_and_employees

    emp = variable(Employee, domain=None)
    department = emp.department
    avg_salary = eql.average(emp.salary)
    max_salary = eql.max(emp.salary)
    query = a(
        set_of(avg_salary, max_salary, department)
        .grouped_by(department)
        .having(max_salary > 25000)
    )
    results = list(query.evaluate())
    result_tuples = []
    assert len(results) == 2
    for result in results:
        result_tuples.append(
            (result[department], result[avg_salary], result[max_salary])
        )
    salary_per_department = defaultdict(list)
    for emp in employees:
        salary_per_department[emp.department].append(emp.salary)
    expected_result_tuples = [
        (
            d,
            sum(salary_per_department[d]) / len(salary_per_department[d]),
            max(salary_per_department[d]),
        )
        for d in departments
        if max(salary_per_department[d]) > 25000
    ]
    for result_tuple, expected_result_tuple in zip(
        result_tuples, expected_result_tuples
    ):
        assert result_tuple == expected_result_tuple


def test_having_node_hierarchy(departments_and_employees):

    emp = variable(Employee, domain=None)
    department = emp.department
    avg_salary = eql.average(emp.salary)

    query = a(
        set_of(department, avg_salary).grouped_by(department).having(avg_salary > 20000)
    )

    descriptor = query._child_
    # Graph hierarchy check
    assert isinstance(descriptor._child_, Having)
    assert isinstance(descriptor._child_.group_by, GroupBy)
    assert descriptor._child_.conditions._name_ == ">"


def test_complex_having_success(departments_and_employees):
    departments, employees = departments_and_employees
    emp = variable(Employee, domain=None)
    department = emp.department
    avg_salary = eql.average(emp.salary)

    query = a(
        set_of(department, avg_salary).grouped_by(department).having(avg_salary > 30000)
    )

    results = list(query.evaluate())
    # Should only return Finance department (avg 35000)
    assert len(results) == 1
    assert results[0][department].name == "Finance"
