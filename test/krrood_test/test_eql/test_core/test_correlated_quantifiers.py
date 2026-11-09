"""
Tests pinning first-order semantics of the EQL quantifiers ``for_all`` and ``exists``
when the quantified variable's domain depends on an outer binding (a correlated
quantifier).

The pinned semantics:

* ``for_all(x, phi)`` is the universal quantification over the finite domain of ``x``,
  ``exists(x, phi)`` the existential one, and ``not_`` is negation as failure.
* A variable that occurs anywhere else in the query scope (selected, or used by another
  condition) is outer-visible, and the quantifier is evaluated once per outer binding.
* A variable that occurs only inside the quantifier is local to it.
* ``exists`` is a semi-join: it never multiplies rows, and local variables never escape.
* ``not_(exists(...))`` is an antijoin, including outer bindings whose collection is empty.
* ``for_all`` over an empty collection is vacuously true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from krrood.entity_query_language.factories import (
    alternative,
    an,
    contains,
    deduced_variable,
    entity,
    exists,
    flat_variable,
    for_all,
    inference,
    not_,
    or_,
    refinement,
    set_of,
    variable,
)
from krrood.entity_query_language.rules.conclusion import Add
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol


@dataclass(eq=False)
class Academic(Symbol):
    name: str


@dataclass(eq=False)
class QuantifiedCourse(Symbol):
    name: str
    taught_by: set[Academic] = field(default_factory=set)


@dataclass(eq=False)
class Student(Role[Academic]):
    takes_course: set[QuantifiedCourse] = field(default_factory=set)


@dataclass(eq=False)
class StudentFlag(Symbol):
    student: Student
    reason: str


@dataclass(eq=False)
class Box(Symbol):
    size: int


@dataclass(eq=False)
class SalariedWorker(Symbol):
    salary: int


@dataclass
class University:
    """
    The finite domain shared by the correlated quantifier tests.
    """

    turing: Academic
    logic: QuantifiedCourse
    artificial_intelligence: QuantifiedCourse
    databases: QuantifiedCourse
    students: list[Student]


@pytest.fixture
def university() -> University:
    turing, minsky, codd = Academic("Turing"), Academic("Minsky"), Academic("Codd")
    logic = QuantifiedCourse("Logic", {turing})
    artificial_intelligence = QuantifiedCourse("AI", {turing, minsky})
    databases = QuantifiedCourse("Databases", {codd})
    students = [
        Student(
            role_taker=Academic("Alice"),
            takes_course={logic, artificial_intelligence},
        ),
        Student(role_taker=Academic("Bob"), takes_course={artificial_intelligence}),
        Student(role_taker=Academic("Carol"), takes_course={logic, databases}),
        Student(role_taker=Academic("Dave"), takes_course=set()),
    ]
    return University(turing, logic, artificial_intelligence, databases, students)


def student_names(results) -> list[str]:
    """
    :return: The sorted names of the students in *results*, keeping duplicates.
    """
    return sorted(student.name for student in results)


def correlated(university: University):
    """
    :return: A fresh student variable and a variable ranging over its courses.
    """
    student = variable(Student, domain=university.students)
    return student, flat_variable(student.takes_course)


# %% for_all over a correlated collection


def test_for_all_is_evaluated_per_student(university):
    """
    ``for_all(c, c.name != "Databases")`` holds for exactly the students who do not take
    Databases, including the student with no courses (vacuous truth).
    """
    student, course = correlated(university)
    query = an(entity(student).where(for_all(course, course.name != "Databases")))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Dave"]


def test_for_all_over_an_empty_collection_is_vacuously_true(university):
    """
    A never-satisfiable body still holds for the student whose collection is empty, and
    only for that student.
    """
    student, course = correlated(university)
    query = an(entity(student).where(for_all(course, course.name == "Nope")))
    assert student_names(query.evaluate()) == ["Dave"]


def test_for_all_with_a_universally_true_body_keeps_every_student(university):
    """
    A body satisfied by every course keeps every student.
    """
    student, course = correlated(university)
    query = an(entity(student).where(for_all(course, course.name != "Nope")))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol", "Dave"]


# %% exists over a correlated collection


@pytest.mark.parametrize(
    "course_name, expected",
    [
        ("Logic", ["Alice", "Carol"]),
        ("AI", ["Alice", "Bob"]),
        ("Databases", ["Carol"]),
        ("Nope", []),
    ],
)
def test_exists_yields_every_student_with_a_witness(university, course_name, expected):
    """
    ``exists`` keeps each outer binding that has a witness; the first witness found must
    not short-circuit the remaining students.
    """
    student, course = correlated(university)
    query = an(entity(student).where(exists(course, course.name == course_name)))
    assert student_names(query.evaluate()) == expected


def test_exists_excludes_the_student_with_an_empty_collection(university):
    """
    A body that every course satisfies still excludes the student with no courses.
    """
    student, course = correlated(university)
    query = an(entity(student).where(exists(course, course.name != "Nope")))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol"]


def test_exists_is_a_semi_join_and_does_not_multiply_rows(university):
    """
    Alice has two witnessing courses but must appear exactly once.
    """
    student, course = correlated(university)
    query = an(entity(student).where(exists(course, course.name != "Databases")))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol"]


def test_exists_over_flat_variable_equals_membership(university):
    """
    ``exists(c, c == k)`` with an external ``k`` holds for the students taking ``k``.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(
            exists(course, course == university.artificial_intelligence)
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Bob"]


# %% negation


def test_not_exists_is_an_antijoin_including_empty_collections(university):
    """
    ``not_(exists(c, c.name == "Logic"))`` keeps students without a Logic course,
    including the student whose collection is empty.
    """
    student, course = correlated(university)
    query = an(entity(student).where(not_(exists(course, course.name == "Logic"))))
    assert student_names(query.evaluate()) == ["Bob", "Dave"]


def test_not_exists_with_no_witness_keeps_every_student(university):
    """
    When no course ever witnesses the body, the antijoin keeps all outer bindings.
    """
    student, course = correlated(university)
    query = an(entity(student).where(not_(exists(course, course.name == "Nope"))))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol", "Dave"]


def test_not_for_all_means_some_course_fails(university):
    """
    ``not_(for_all(c, phi))`` is true exactly for students with a course violating phi,
    and false for the vacuously satisfying student.
    """
    student, course = correlated(university)
    query = an(entity(student).where(not_(for_all(course, course.name != "Databases"))))
    assert student_names(query.evaluate()) == ["Carol"]


def test_double_negation_agrees_with_exists(university):
    """
    ``not_(not_(exists(...)))`` selects the same students as ``exists(...)``.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(not_(not_(exists(course, course.name == "Logic"))))
    )
    assert student_names(query.evaluate()) == ["Alice", "Carol"]


# %% disjunction


def test_exists_inside_or_keeps_witnessed_and_alternative_students(university):
    """
    ``or_(exists(...), other)`` is the union of the quantified and the other branch, one
    row per student.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(
            or_(
                exists(course, course.name == "Databases"),
                student.name == "Bob",
            )
        )
    )
    assert student_names(query.evaluate()) == ["Bob", "Carol"]


def test_for_all_inside_or_keeps_satisfied_and_alternative_students(university):
    """
    ``or_(for_all(...), other)`` is the union of the quantified and the other branch.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(
            or_(
                for_all(course, course.name != "Databases"),
                student.name == "Carol",
            )
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol", "Dave"]


# %% nested quantifiers


def test_exists_course_whose_teachers_are_all_turing(university):
    """
    Exists a course all of whose teachers are Turing, correlated through the student.
    """
    student, course = correlated(university)
    teacher = flat_variable(course.taught_by)
    query = an(
        entity(student).where(
            exists(course, for_all(teacher, teacher.name == "Turing"))
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Carol"]


def test_for_all_courses_have_some_teacher_turing(university):
    """
    Every course of the student has Turing among its teachers; the student without
    courses satisfies this vacuously.
    """
    student, course = correlated(university)
    teacher = flat_variable(course.taught_by)
    query = an(
        entity(student).where(
            for_all(course, exists(teacher, teacher.name == "Turing"))
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Dave"]


def test_exists_course_with_some_teacher_codd(university):
    """
    Nested existentials: some course of the student has Codd as a teacher.
    """
    student, course = correlated(university)
    teacher = flat_variable(course.taught_by)
    query = an(
        entity(student).where(exists(course, exists(teacher, teacher.name == "Codd")))
    )
    assert student_names(query.evaluate()) == ["Carol"]


def test_for_all_courses_have_only_turing_or_minsky_teachers(university):
    """
    Nested universals: every teacher of every course of the student is Turing or Minsky.
    """
    student, course = correlated(university)
    teacher = flat_variable(course.taught_by)
    query = an(
        entity(student).where(
            for_all(
                course,
                for_all(
                    teacher, or_(teacher.name == "Turing", teacher.name == "Minsky")
                ),
            )
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Dave"]


# %% independent quantified variables keep their behaviour


def test_for_all_over_an_independent_variable_holds_globally(university):
    """
    A quantified variable whose domain does not depend on the student is evaluated once;
    every course of the domain being taught by Turing keeps every student.
    """
    student = variable(Student, domain=university.students)
    course = variable(
        QuantifiedCourse,
        domain=[university.logic, university.artificial_intelligence],
    )
    query = an(
        entity(student).where(
            for_all(course, contains(course.taught_by, university.turing))
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol", "Dave"]


def test_for_all_over_an_independent_variable_fails_globally(university):
    """
    A body violated by one course of the independent domain removes every student.
    """
    student = variable(Student, domain=university.students)
    course = variable(
        QuantifiedCourse,
        domain=[
            university.logic,
            university.artificial_intelligence,
            university.databases,
        ],
    )
    query = an(entity(student).where(for_all(course, course.name != "AI")))
    assert student_names(query.evaluate()) == []


def test_exists_over_an_independent_variable_holds_globally(university):
    """
    A witness in the independent domain keeps every student exactly once.
    """
    student = variable(Student, domain=university.students)
    course = variable(
        QuantifiedCourse, domain=[university.logic, university.artificial_intelligence]
    )
    query = an(entity(student).where(exists(course, course.name == "AI")))
    assert student_names(query.evaluate()) == ["Alice", "Bob", "Carol", "Dave"]


def test_exists_over_an_independent_variable_without_witness_is_empty(university):
    """
    Without a witness in the independent domain no student remains.
    """
    student = variable(Student, domain=university.students)
    course = variable(QuantifiedCourse, domain=[university.logic])
    query = an(entity(student).where(exists(course, course.name == "Databases")))
    assert student_names(query.evaluate()) == []


# %% outer-visible variables that are not selected


def boxes_and_workers():
    return [Box(2), Box(5), Box(10)], [
        SalariedWorker(1),
        SalariedWorker(4),
        SalariedWorker(7),
    ]


def box_sizes(results) -> list[int]:
    return sorted(box.size for box in results)


def test_exists_over_a_variable_used_elsewhere_equals_the_plain_conjunction_last():
    """
    ``worker`` also occurs outside the quantifier, so it is outer-visible and
    ``exists(worker, worker.salary < box.size)`` agrees with the plain condition.
    """
    boxes, workers = boxes_and_workers()
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    quantified = an(
        entity(box).where(exists(worker, worker.salary < box.size), worker.salary > 3)
    )
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    plain = an(entity(box).where(worker.salary < box.size, worker.salary > 3))
    assert box_sizes(quantified.evaluate()) == box_sizes(plain.evaluate())


def test_exists_over_a_variable_used_elsewhere_equals_the_plain_conjunction_first():
    """
    The result does not depend on the quantifier being listed after the other condition.
    """
    boxes, workers = boxes_and_workers()
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    quantified = an(
        entity(box).where(worker.salary > 3, exists(worker, worker.salary < box.size))
    )
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    plain = an(entity(box).where(worker.salary > 3, worker.salary < box.size))
    assert box_sizes(quantified.evaluate()) == box_sizes(plain.evaluate())


def test_exists_over_a_course_variable_used_elsewhere_binds_it_per_row_last(university):
    """
    The course variable is constrained outside the quantifier, so the quantifier only
    sees the course already bound for that row: students taking a non-AI Logic course.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(
            exists(course, course.name == "Logic"), course.name != "AI"
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Carol"]


def test_exists_over_a_course_variable_used_elsewhere_binds_it_per_row_first(
    university,
):
    """
    Same as above with the quantifier listed after the other condition.
    """
    student, course = correlated(university)
    query = an(
        entity(student).where(
            course.name != "AI", exists(course, course.name == "Logic")
        )
    )
    assert student_names(query.evaluate()) == ["Alice", "Carol"]


def test_extending_an_evaluated_query_updates_the_outer_visible_variables():
    """
    Adding a condition after a first evaluation rebuilds the query around the same
    quantifier; the quantifier then sees the new outside occurrence of ``worker``.
    """
    boxes, workers = boxes_and_workers()
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    query = an(entity(box).where(exists(worker, worker.salary < box.size)))
    assert box_sizes(query.evaluate()) == [2, 5, 10]
    query.where(worker.salary > 3)
    assert box_sizes(query.evaluate()) == [5, 10, 10]


def test_a_quantifier_reused_in_two_queries_is_scoped_by_the_evaluated_one():
    """
    The same quantifier object is local to one query and correlated in another; each
    evaluation uses the scope of the query being evaluated.
    """
    boxes, workers = boxes_and_workers()
    box, worker = variable(Box, domain=boxes), variable(SalariedWorker, domain=workers)
    quantifier = exists(worker, worker.salary < box.size)
    uncorrelated = an(entity(box).where(quantifier))
    correlated_query = an(entity(box).where(quantifier, worker.salary > 3))
    assert box_sizes(uncorrelated.evaluate()) == [2, 5, 10]
    assert box_sizes(correlated_query.evaluate()) == [5, 10, 10]
    assert box_sizes(uncorrelated.evaluate()) == [2, 5, 10]


# %% local variables never escape


def test_exists_over_a_collection_of_a_local_variable_keeps_one_row_per_outer_binding(
    university,
):
    """
    The student occurs only inside the quantifier, so it is local to it: the selected
    course appears once, not once per student taking Logic.
    """
    student, course = correlated(university)
    other = variable(QuantifiedCourse, domain=[university.logic])
    query = an(entity(other).where(exists(course, course.name == "Logic")))
    assert list(query.evaluate()) == [university.logic]


def test_set_of_with_an_exists_does_not_expose_the_local_variable(university):
    """
    Only the selected variables appear in each result row of a set query.
    """
    student, course = correlated(university)
    other = variable(QuantifiedCourse, domain=[university.logic])
    query = set_of(student, other).where(exists(course, course.name == "Logic"))
    rows = list(query.evaluate())
    assert all(course not in row.data for row in rows)


def test_set_of_with_an_exists_yields_one_row_per_outer_binding(university):
    """
    Two selected variables and a semi-join: one row per (student, other) combination
    whose student has a witness.
    """
    student, course = correlated(university)
    other = variable(QuantifiedCourse, domain=[university.logic, university.databases])
    query = set_of(student, other).where(exists(course, course.name == "Logic"))
    assert len(list(query.evaluate())) == 4


def test_correlated_exists_with_a_selected_variable_inside_the_body(university):
    """
    A selected variable occurring only inside the quantifier body is outer-visible, so
    the result is exactly the (student, course) pairs with membership, not a cross
    product.
    """
    student, course = correlated(university)
    selected_course = variable(
        QuantifiedCourse,
        domain=[
            university.logic,
            university.artificial_intelligence,
            university.databases,
        ],
    )
    query = set_of(student, selected_course).where(
        exists(course, course == selected_course)
    )
    pairs = {(row[student].name, row[selected_course].name) for row in query.evaluate()}
    assert pairs == {
        ("Alice", "Logic"),
        ("Alice", "AI"),
        ("Bob", "AI"),
        ("Carol", "Logic"),
        ("Carol", "Databases"),
    }


# %% rule bodies


def flag_pairs(results) -> set[tuple[str, str]]:
    return {(flag.student.name, flag.reason) for flag in results}


def test_correlated_exists_in_an_inference_rule_condition(university):
    """
    An inference rule whose condition holds a correlated ``exists`` fires once for each
    student with a witness.
    """
    student, course = correlated(university)
    flags = deduced_variable(StudentFlag)
    query = an(
        entity(flags).where(
            student.name != "Nobody", exists(course, course.name == "Logic")
        )
    )
    with query:
        Add(flags, inference(StudentFlag)(student=student, reason="logic"))
    assert flag_pairs(query.evaluate()) == {("Alice", "logic"), ("Carol", "logic")}


def test_correlated_exists_in_a_refinement_condition(university):
    """
    A refinement guarded by a correlated ``exists`` overrides the base conclusion only
    for students with a witness.
    """
    student, course = correlated(university)
    flags = deduced_variable(StudentFlag)
    query = an(entity(flags).where(student.name != "Nobody"))
    with query:
        Add(flags, inference(StudentFlag)(student=student, reason="base"))
        with refinement(exists(course, course.name == "Databases")):
            Add(flags, inference(StudentFlag)(student=student, reason="refined"))
    assert flag_pairs(query.evaluate()) == {
        ("Alice", "base"),
        ("Bob", "base"),
        ("Carol", "refined"),
        ("Dave", "base"),
    }


def test_correlated_for_all_in_an_alternative_condition(university):
    """
    An alternative guarded by a correlated ``for_all`` fires for students the base
    condition rejects and who satisfy the universal.
    """
    student, course = correlated(university)
    flags = deduced_variable(StudentFlag)
    query = an(entity(flags).where(student.name == "Bob"))
    with query:
        Add(flags, inference(StudentFlag)(student=student, reason="base"))
        with alternative(for_all(course, course.name != "AI")):
            Add(flags, inference(StudentFlag)(student=student, reason="alternative"))
    assert flag_pairs(query.evaluate()) == {
        ("Bob", "base"),
        ("Carol", "alternative"),
        ("Dave", "alternative"),
    }
