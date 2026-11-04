"""
Tests pinning the semantics of ``match`` on collection-valued attributes, which follow a
triple pattern: a single value or a nested pattern given for an attribute whose declared
type is a collection is satisfied by *some element* of the collection.

The pinned semantics:

* ``attr=value`` with a single non-collection value means ``value`` is an element of the
  attribute; so does a symbolic value whose type is the element type.
* ``attr=pattern`` means some element of the attribute matches the pattern; the owner
  appears once however many elements match, and an owner with an empty collection never
  matches.
* A single-valued attribute keeps equality, a collection literal or a symbolic value of a
  collection type keeps whole-collection equality, a list of patterns keeps positional
  matching, and a ``type[X]`` attribute keeps equality with the class.

The domain classes have names no other test module uses: the class diagram resolves a
string annotation such as ``set[Member]`` by class name, so a same-named class imported
by another module could otherwise take its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Iterable, List

from krrood.entity_query_language.backends import EntityQueryLanguageGenerativeBackend
from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
    variable_from,
)
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol

# %% Domain


@dataclass(eq=False)
class CampusOrganization(Symbol):
    name: str


@dataclass(eq=False)
class CampusMember(Symbol):
    name: str
    is_dean_of: set[CampusOrganization] = field(default_factory=set)


@dataclass(eq=False)
class CampusCourse(Symbol):
    name: str
    is_taught_by: set[CampusMember] = field(default_factory=set)


@dataclass(eq=False)
class SeminarCourse(CampusCourse):
    pass


@dataclass(eq=False)
class CampusStudent(Role[CampusMember]):
    takes_course: set[CampusCourse] = field(default_factory=set)


@dataclass(eq=False)
class TaggedEntry(Symbol):
    name: str
    tags: list[str] = field(default_factory=list)
    labels: tuple[str, ...] = ()


@dataclass(eq=False)
class CoursePlaylist(Symbol):
    name: str
    courses: list[CampusCourse] = field(default_factory=list)


@dataclass(eq=False)
class CourseRegistration(Symbol):
    name: str
    kind: type[CampusCourse] = CampusCourse


@dataclass(eq=False)
class WeightedItem(Symbol):
    weight: float


@dataclass(eq=False)
class ItemBasket(Symbol):
    items: list[WeightedItem] = field(default_factory=list)
    label: str = ""


def names_of(students: Iterable[CampusStudent]) -> List[str]:
    """
    :param students: The matched students.
    :return: Their sorted names, read from their role taker.
    """
    return sorted(student.role_taker.name for student in students)


def plain_names_of(matches: Iterable[Symbol]) -> List[str]:
    """
    :param matches: The matched objects, each with a ``name``.
    :return: Their sorted names.
    """
    return sorted(match.name for match in matches)


# %% Fixtures as builders


def build_university():
    """
    :return: The courses ``(logic, ai, databases)`` and the students Alice (Logic, AI),
        Bob (AI), Carol (Logic, Databases) and Dave, who takes nothing.
    """
    logic = CampusCourse("Logic")
    ai = CampusCourse("AI")
    databases = CampusCourse("Databases")
    students = [
        CampusStudent(role_taker=CampusMember("Alice"), takes_course={logic, ai}),
        CampusStudent(role_taker=CampusMember("Bob"), takes_course={ai}),
        CampusStudent(
            role_taker=CampusMember("Carol"), takes_course={logic, databases}
        ),
        CampusStudent(role_taker=CampusMember("Dave"), takes_course=set()),
    ]
    return (logic, ai, databases), students


def build_deanery():
    """
    :return: Students over the chain student, course, teacher, organization, excluded
        at every level: Alice takes a course taught by a dean; Bob only a course whose
        teacher is no dean; Carol a course with a dean among its two teachers; Dave
        nothing; Erin only a course nobody teaches.
    """
    faculty = CampusOrganization("Faculty")
    dean = CampusMember("Dean", is_dean_of={faculty})
    lecturer = CampusMember("Lecturer")
    logic = CampusCourse("Logic", is_taught_by={dean})
    ai = CampusCourse("AI", is_taught_by={lecturer})
    databases = CampusCourse("Databases", is_taught_by={lecturer, dean})
    orphan = CampusCourse("Orphan")
    return [
        CampusStudent(role_taker=CampusMember("Alice"), takes_course={logic}),
        CampusStudent(role_taker=CampusMember("Bob"), takes_course={ai}),
        CampusStudent(role_taker=CampusMember("Carol"), takes_course={databases}),
        CampusStudent(role_taker=CampusMember("Dave")),
        CampusStudent(role_taker=CampusMember("Erin"), takes_course={orphan}),
    ]


def dean_pattern():
    """
    :return: Students taking some course taught by some dean (the OWL2Bench Q22 shape).
    """
    return a(CampusStudent)(
        takes_course=a(CampusCourse)(
            is_taught_by=a(CampusMember)(is_dean_of=a(CampusOrganization)())
        )
    )


# %% Element membership


def test_value_in_collection_matches_owners_containing_it():
    """
    ``takes_course=logic`` matches the students whose courses include Logic.
    """
    (logic, _, _), students = build_university()
    matches = a(CampusStudent)(takes_course=logic).from_(students).evaluate()
    assert names_of(matches) == ["Alice", "Carol"]


def test_membership_with_a_different_element():
    """
    ``takes_course=ai`` matches the students whose courses include AI.
    """
    (_, ai, _), students = build_university()
    matches = a(CampusStudent)(takes_course=ai).from_(students).evaluate()
    assert names_of(matches) == ["Alice", "Bob"]


def test_value_in_no_collection_matches_nothing():
    """
    A course nobody takes matches no student.
    """
    _, students = build_university()
    unused = CampusCourse("Unused")
    matches = a(CampusStudent)(takes_course=unused).from_(students).evaluate()
    assert list(matches) == []


def test_membership_in_list_attribute():
    """
    A plain value matches the owners of a ``list[str]`` attribute holding it.
    """
    entries = [
        TaggedEntry("both", tags=["red", "blue"]),
        TaggedEntry("blue", tags=["blue"]),
        TaggedEntry("none"),
    ]
    matches = a(TaggedEntry)(tags="red").from_(entries).evaluate()
    assert plain_names_of(matches) == ["both"]


def test_membership_in_tuple_attribute():
    """
    A plain value matches the owners of a ``tuple[str, ...]`` attribute holding it.
    """
    entries = [TaggedEntry("both", labels=("x", "y")), TaggedEntry("y", labels=("y",))]
    matches = a(TaggedEntry)(labels="x").from_(entries).evaluate()
    assert plain_names_of(matches) == ["both"]


def test_variable_of_the_element_type_means_membership():
    """
    A symbolic value whose type is the element type stands for one element, so it must
    be in the collection.
    """
    (logic, _, _), students = build_university()
    course = variable(CampusCourse, domain=[logic])
    matches = a(CampusStudent)(takes_course=course).from_(students).evaluate()
    assert names_of(matches) == ["Alice", "Carol"]


# %% Nested pattern over elements


def test_nested_pattern_matches_when_some_element_matches():
    """
    A nested pattern matches the students with a course satisfying it.
    """
    _, students = build_university()
    pattern = a(CampusStudent)(takes_course=a(CampusCourse)(name="Databases"))
    assert names_of(pattern.from_(students).evaluate()) == ["Carol"]


def test_unconstrained_pattern_matches_each_owner_with_an_element_once():
    """
    An unconstrained pattern matches every student with at least one course, each once
    although Alice and Carol have two, and excludes Dave, whose collection is empty.
    """
    _, students = build_university()
    pattern = a(CampusStudent)(takes_course=a(CampusCourse)())
    assert names_of(pattern.from_(students).evaluate()) == ["Alice", "Bob", "Carol"]


def test_nested_pattern_with_condition_satisfied_by_two_elements_yields_owner_once():
    """
    A student with two courses both named "Seminar" appears exactly once.
    """
    seminars = [CampusCourse("Seminar"), CampusCourse("Seminar")]
    students = [
        CampusStudent(role_taker=CampusMember("Twice"), takes_course=set(seminars)),
        CampusStudent(role_taker=CampusMember("Once"), takes_course={seminars[0]}),
        CampusStudent(
            role_taker=CampusMember("Never"), takes_course={CampusCourse("Other")}
        ),
    ]
    pattern = a(CampusStudent)(takes_course=a(CampusCourse)(name="Seminar"))
    assert names_of(pattern.from_(students).evaluate()) == ["Once", "Twice"]


def test_pattern_of_a_subtype_filters_the_elements_by_type():
    """
    A pattern of a subclass of the element type is matched only by elements of that
    subclass.
    """
    seminar = SeminarCourse("Reading group")
    students = [
        CampusStudent(role_taker=CampusMember("Seminarist"), takes_course={seminar}),
        CampusStudent(
            role_taker=CampusMember("Lecture goer"),
            takes_course={CampusCourse("Lecture")},
        ),
    ]
    pattern = a(CampusStudent)(takes_course=a(SeminarCourse)())
    assert names_of(pattern.from_(students).evaluate()) == ["Seminarist"]


# %% Two-level nesting through two collection attributes


def test_two_level_nesting_through_collection_attributes():
    """
    Exactly Alice and Carol take some course taught by some dean: Bob's teacher is no
    dean, Erin's course has no teacher and Dave takes nothing.
    """
    students = build_deanery()
    assert names_of(dean_pattern().from_(students).evaluate()) == ["Alice", "Carol"]


def test_two_level_nesting_yields_each_student_once():
    """
    Carol's course has two teachers yet she appears once; every student taking a taught
    course appears, and Dave (no course) and Erin (untaught course) do not.
    """
    students = build_deanery()
    pattern = a(CampusStudent)(
        takes_course=a(CampusCourse)(is_taught_by=a(CampusMember)())
    )
    assert names_of(pattern.from_(students).evaluate()) == ["Alice", "Bob", "Carol"]


# %% Combination with single-valued keywords


def test_nested_pattern_combined_with_single_valued_keyword_requires_both():
    """
    A course pattern and a role taker value both have to hold.
    """
    _, students = build_university()
    alice_member = students[0].role_taker
    matches = (
        a(CampusStudent)(
            role_taker=alice_member, takes_course=a(CampusCourse)(name="Logic")
        )
        .from_(students)
        .evaluate()
    )
    assert names_of(matches) == ["Alice"]


def test_nested_pattern_combined_with_failing_single_valued_keyword_matches_nothing():
    """
    Bob does not take Logic, so Bob's role taker with a Logic pattern matches nothing.
    """
    _, students = build_university()
    bob_member = students[1].role_taker
    matches = (
        a(CampusStudent)(
            role_taker=bob_member, takes_course=a(CampusCourse)(name="Logic")
        )
        .from_(students)
        .evaluate()
    )
    assert list(matches) == []


def test_value_membership_combined_with_nested_role_taker_pattern():
    """
    Membership of a course and a nested pattern on the role taker both have to hold.
    """
    (logic, _, _), students = build_university()
    matches = (
        a(CampusStudent)(takes_course=logic, role_taker=a(CampusMember)(name="Carol"))
        .from_(students)
        .evaluate()
    )
    assert names_of(matches) == ["Carol"]


# %% Regression guards: unchanged semantics


def test_collection_literal_keeps_whole_collection_equality():
    """
    A set literal matches only the students whose whole collection equals it.
    """
    (_, ai, _), students = build_university()
    matches = a(CampusStudent)(takes_course={ai}).from_(students).evaluate()
    assert names_of(matches) == ["Bob"]


def test_variable_of_a_collection_type_keeps_whole_collection_equality():
    """
    A symbolic value whose values are whole collections is compared to the whole
    collection.
    """
    (_, ai, _), students = build_university()
    collection = variable(set, domain=[{ai}])
    matches = a(CampusStudent)(takes_course=collection).from_(students).evaluate()
    assert names_of(matches) == ["Bob"]


def test_list_literal_keeps_whole_collection_equality():
    """
    A list literal equals the whole list attribute, not one of its elements.
    """
    entries = [
        TaggedEntry("pair", tags=["red", "blue"]),
        TaggedEntry("single", ["red"]),
    ]
    matches = a(TaggedEntry)(tags=["red"]).from_(entries).evaluate()
    assert plain_names_of(matches) == ["single"]


def test_single_valued_attribute_keeps_equality():
    """
    A single-valued attribute is matched by equality.
    """
    courses, _ = build_university()
    matches = a(CampusCourse)(name="AI").from_(list(courses)).evaluate()
    assert plain_names_of(matches) == ["AI"]


def test_list_of_patterns_keeps_positional_matching():
    """
    A list of patterns matches element by element, in order.
    """
    logic, ai = CampusCourse("Logic"), CampusCourse("AI")
    playlists = [
        CoursePlaylist("forward", [logic, ai]),
        CoursePlaylist("backward", [ai, logic]),
    ]
    pattern = a(CoursePlaylist)(
        courses=[a(CampusCourse)(name="Logic"), a(CampusCourse)(name="AI")]
    )
    assert plain_names_of(pattern.from_(playlists).evaluate()) == ["forward"]


def test_type_attribute_keeps_equality_with_a_class():
    """
    A ``type[CampusCourse]`` attribute is compared to the class by equality.
    """
    registrations = [
        CourseRegistration("course"),
        CourseRegistration("seminar", kind=SeminarCourse),
    ]
    pattern = a(CourseRegistration)(kind=SeminarCourse)
    assert plain_names_of(pattern.from_(registrations).evaluate()) == ["seminar"]


# %% Match used inside a larger query


def test_match_as_variable_in_a_larger_query_keeps_one_row_per_owner():
    """
    Selecting the matched students through an enclosing query yields one row per student
    although Alice and Carol have two courses satisfying the pattern.
    """
    _, students = build_university()
    student = a(CampusStudent)(takes_course=a(CampusCourse)()).from_(students)
    assert names_of(an(entity(student)).evaluate()) == ["Alice", "Bob", "Carol"]


# %% Construction from a pattern on a collection-valued attribute


def test_generative_backend_still_constructs_through_a_collection_pattern():
    """
    Generating from a pattern stated for a collection-valued attribute writes each
    generated value into that pattern instead of rejecting the write, giving one
    instance per value.

    ..note:: Construction passes the element built from the pattern to the factory as
        it is, not wrapped in a collection, so only the number of instances is pinned.
    """
    pattern = an(ItemBasket)(
        items=a(WeightedItem)(weight=variable_from([1.0, 2.0])), label="x"
    )
    baskets = list(pattern.evaluate(backend=EntityQueryLanguageGenerativeBackend()))
    assert len(baskets) == 2
    assert all(basket.label == "x" for basket in baskets)
