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
- Predicate fragment tests: HasType, ContainsType, custom predicates,
  and the required fragment (verbalizing one without it is an error).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any
from typing_extensions import List

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
    inference,
    for_all,
    exists,
    not_,
    and_,
    or_,
)
from krrood.entity_query_language.predicate import HasType, HasTypes, Predicate, Triple
from krrood.entity_query_language.verbalization.exceptions import (
    PredicateFragmentRequiredError,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
    Verb,
)
from krrood.entity_query_language.verbalization.pipeline import (
    VerbalizationPipeline,
    verbalize_expression,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    ANSIFormatter,
    HTMLFormatter,
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    HierarchicalRenderer,
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.verbalizer import (
    EQLVerbalizer,
    MicroplanningServices,
)
from ...dataset.department_and_employee import Department, Employee


@dataclass
class _Task:
    name: str
    completed: bool


@dataclass
class _Robot:
    name: str
    battery: int
    tasks: List[_Task]


from ...dataset.semantic_world_like_classes import (
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


# ── Sanity / regression tests ──────────────────────────────────────────────────


def test_all_rules_registry_is_populated():
    """RULES must be non-empty after import — otherwise every expression falls back to
    its name string and verbalization is silently broken."""
    from krrood.entity_query_language.verbalization.grammar.framework.registry import (
        RULES,
    )

    assert len(RULES) > 0, (
        "RULES is empty — the grammar was not imported, so every expression would "
        "fall back to its name string."
    )


def test_uncovered_construct_raises_instead_of_degrading():
    """A construct with no grammar rule (e.g. Concatenation) must raise, not silently
    render its bare class name."""
    from krrood.entity_query_language.factories import concatenation
    from krrood.entity_query_language.verbalization.exceptions import (
        UnverbalizableExpressionError,
    )

    expr = concatenation(variable(int, []), variable(int, []))
    with pytest.raises(UnverbalizableExpressionError):
        verbalize_expression(expr)


def test_verbalization_produces_natural_language_not_repr():
    """Smoke test: verbalizing a simple query must produce English prose,
    not fall back to the expression's repr string like '(EntityType)'."""

    @dataclass(unsafe_hash=True)
    class _Smoke:
        value: float

    v = variable(_Smoke, domain=None)
    query = an(entity(v).where(v.value > 10))
    result = verbalize_expression(query)

    assert isinstance(
        result, str
    ), f"Expected str, got {type(result).__name__}: {result!r}"
    assert result.startswith(
        "Find "
    ), f"Expected natural-language output starting with 'Find ', got: {result!r}"
    assert "_Smoke" in result
    assert "10" in result


# ── Unit tests: leaves ─────────────────────────────────────────────────────────


def test_verbalize_variable_first_mention():
    x = variable(int, [1, 2])
    assert verbalize_expression(x) == "an Integer"


def test_verbalize_variable_article_consonant():
    x = variable(Body, [])
    assert verbalize_expression(x) == "a Body"


def test_verbalize_variable_coreference():
    context = MicroplanningServices()
    pipeline = VerbalizationPipeline.plain()
    x = variable(Handle, [])
    first = pipeline.verbalize(x, context)
    second = pipeline.verbalize(x, context)
    assert first == "a Handle"
    assert second == "the Handle"


def test_verbalize_literal_plain_value():
    literal_value = Literal(_value_=42)
    assert "42" in verbalize_expression(literal_value)


def test_value_phrase_none_is_nothing():
    from krrood.entity_query_language.verbalization.value_lexicon import value_phrase

    assert value_phrase(None) == "nothing"


def test_value_phrase_enum_uses_member_name():
    import enum

    from krrood.entity_query_language.verbalization.value_lexicon import value_phrase

    class _Choice(enum.Enum):
        FIRST_OPTION = "first"

    assert value_phrase(_Choice.FIRST_OPTION) == "FIRST_OPTION"


def test_verbalize_literal_type_object():
    literal_value = Literal(_value_=Apple)
    assert verbalize_expression(literal_value) == "Apple"


def test_verbalize_literal_tuple_of_types_is_a_value_not_membership():
    """A bare tuple literal is a *value* — its classes are listed and joined, not read as *"one of"*
    (which would mean membership). Membership is the consuming predicate's call, so an equality with
    a tuple is never mis-read as membership."""
    assert verbalize_expression(Literal(_value_=(Apple, Body))) == "Apple and Body"
    assert (
        verbalize_expression(variable(_Robot, []).name == (Apple, Body))
        == "the name of a _Robot is Apple and Body"
    )


def test_verbalize_has_types_is_membership():
    """The type-membership predicate reads through the same *"is of type A or B"* surface as
    ``HasType``, its admissible types listed disjunctively."""
    subject = variable(Body, [])
    assert (
        verbalize_expression(HasTypes(subject, (Apple, Cabinet)))
        == "a Body is of type Apple or Cabinet"
    )


def test_verbalize_has_types_lists_all_admissible_types():
    """Every admissible type is listed disjunctively -- ``isinstance`` over the tuple holds for ANY of
    them -- joined by the :class:`Or` element with an oxford comma before the final *"or"*."""
    subject = variable(Body, [])
    many = (
        Apple,
        Cabinet,
        Container,
        Drawer,
        FixedConnection,
        Handle,
        PrismaticConnection,
    )
    text = verbalize_expression(HasTypes(subject, many))
    assert text == (
        "a Body is of type Apple, Cabinet, Container, Drawer, "
        "FixedConnection, Handle, or PrismaticConnection"
    )


# ── Unit tests: MappedVariable chain ──────────────────────────────────────────


def test_verbalize_attribute_uses_of_form():
    employee = variable(Employee, [])
    text = verbalize_expression(employee.salary)
    assert "Employee" in text
    assert "salary" in text
    assert " of " in text


def test_verbalize_attribute_uses_of_form_all_hops():
    # single-hop also uses "of" form
    cabinet = variable(Cabinet, [])
    one_hop = verbalize_expression(cabinet.container)
    assert " of " in one_hop

    # Two-hop chain: Employee → department → name
    employee = variable(Employee, [])
    dept_name = employee.department.name
    text = verbalize_expression(dept_name)
    assert "Employee" in text
    assert "department" in text
    assert "name" in text
    assert " of " in text


def test_verbalize_index_access_rendered_as_ordinal():
    @dataclass
    class Robot:
        tasks: list

    r = variable(Robot, [])
    text = verbalize_expression(r.tasks[0])
    assert "Robot" in text
    assert "task" in text
    # An integer index folds its ordinal into the singularized noun ("the first task"), not a raw
    # subscript leak.
    assert "first" in text
    assert "[0]" not in text


def test_verbalize_negative_index_reads_from_the_end():
    @dataclass
    class Robot:
        tasks: list

    r = variable(Robot, [])
    assert verbalize_expression(r.tasks[-1]) == "the last task of a Robot"
    assert verbalize_expression(r.tasks[-2]) == "the second to last task of a Robot"


def test_verbalize_index_then_attribute_is_ordinal_chain():
    @dataclass
    class Task:
        name: str

    @dataclass
    class Robot:
        tasks: list

    r = variable(Robot, [])
    text = verbalize_expression(r.tasks[0].name)
    assert text == "the name of the first task of a Robot"
    assert "[0]" not in text


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
    assert "task" in text
    assert "is" in text
    assert "completed" in text
    # must NOT be "completed of tasks[0] of …"
    assert "completed of" not in text


def test_verbalize_indexed_bool_attribute_negated():
    r = variable(_Robot, [])
    text = verbalize_expression(not_(r.tasks[0].completed))
    assert "first" in text
    assert "task" in text
    assert "is not" in text
    assert "completed" in text


def test_verbalize_second_index_ordinal():
    r = variable(_Robot, [])
    text = verbalize_expression(r.tasks[1].completed)
    assert "second" in text
    assert "task" in text
    assert "is" in text
    assert "completed" in text


# ── Relational navigation: a verb-named hop reads as a relative clause ────────────


@dataclass
class _NavRobot:
    operational: bool
    battery: int
    power: int


@dataclass
class _NavPerson:
    pass


@dataclass
class _NavAuthor:
    pass


@dataclass
class _NavAddress:
    pass


@dataclass
class _NavMission:
    assigned_to: _NavRobot  # past participle + preposition → relation


@dataclass
class _NavPair:
    primary: _NavMission
    secondary: _NavMission  # two missions, so two distinct assigned_to robots


@dataclass
class _NavBook:
    owned_by: _NavPerson  # agentive "by"


@dataclass
class _NavDoc:
    written_by: _NavAuthor  # irregular participle + agentive "by"


@dataclass
class _NavParcel:
    sent_to: _NavAddress  # irregular participle + goal "to"


@dataclass
class _NavPanel:
    lit: bool


@dataclass
class _NavGadget:
    color_in: _NavPanel  # "color" is not a participle → plain genitive hop


@dataclass
class _NavDept:
    name: str


@dataclass
class _NavEmp:
    department: _NavDept


def test_relational_navigation_reads_as_relative_clause():
    """A relational hop before a boolean terminal reads *"the <Type> which <owner> is <verb>"*,
    using the field's type as the head and dropping the genitive *of*."""
    m = variable(_NavMission, [])
    text = verbalize_expression(m.assigned_to.operational)
    assert text == "the _NavRobot to which a _NavMission is assigned is operational"
    assert "assigned_to" not in text and " of " not in text


def test_relational_navigation_standalone():
    m = variable(_NavMission, [])
    assert (
        verbalize_expression(m.assigned_to)
        == "the _NavRobot to which a _NavMission is assigned"
    )


def test_relational_navigation_agentive_by_reads_active():
    """An agentive *by* relation reads in the active voice with the related type as the verb's
    subject (*"the Person who owns a Book"*), including for an irregular participle
    (*"written"* → *"writes"*)."""
    assert (
        verbalize_expression(variable(_NavBook, []).owned_by)
        == "the _NavPerson who owns a _NavBook"
    )
    assert (
        verbalize_expression(variable(_NavDoc, []).written_by)
        == "the _NavAuthor who writes a _NavDoc"
    )


def test_relational_navigation_agentive_by_pronominalises_owner():
    """When the owner is the query subject, the active-voice agentive clause pronominalises it
    (*"the Person who owns it"*)."""
    book = variable(_NavBook, [])
    text = verbalize_expression(an(entity(book).where(book.owned_by.name == "Bob")))
    assert "the name of the _NavPerson who owns it is 'Bob'" in text


def test_relational_navigation_goal_relation_stays_passive():
    """A non-agentive *to* relation keeps the passive relative clause (only *by* relations go
    active)."""
    assert (
        verbalize_expression(variable(_NavParcel, []).sent_to)
        == "the _NavAddress to which a _NavParcel is sent"
    )


def test_relational_navigation_irregular_participle():
    """The participle check is morphological, so an irregular participle (*sent*) is recognised."""
    assert (
        verbalize_expression(variable(_NavParcel, []).sent_to)
        == "the _NavAddress to which a _NavParcel is sent"
    )


def test_relational_navigation_multi_hop_outer_genitive():
    """A plain hop after a relational one keeps the genitive, wrapping the relative clause."""
    m = variable(_NavMission, [])
    assert (
        verbalize_expression(m.assigned_to.battery)
        == "the battery of the _NavRobot to which a _NavMission is assigned"
    )


def test_noun_hop_ending_in_preposition_is_not_relativized():
    """A hop whose name merely ends in a preposition (*color_in*) is not a participle, so it stays
    the genitive form."""
    text = verbalize_expression(variable(_NavGadget, []).color_in.lit)
    assert text == "the color_in of a _NavGadget is lit"
    assert "which" not in text


def test_non_relational_navigation_unchanged():
    """A purely noun chain renders the familiar genitive path, unchanged."""
    assert (
        verbalize_expression(variable(_NavEmp, []).department.name)
        == "the name of the department of a _NavEmp"
    )


@dataclass
class _Cash:
    amount: float


@dataclass
class _Wallet:
    money: _Cash


@dataclass
class _Holder:
    wallet: _Wallet


def test_genitive_omits_article_before_an_uncountable_noun():
    """A mass noun in a genitive chain takes no article — *"the amount of money"*, never *"the
    amount of the money"* — while its countable neighbours keep *"the"*."""
    assert (
        verbalize_expression(variable(_Holder, []).wallet.money.amount)
        == "the amount of money of the wallet of a _Holder"
    )


# ── Relational navigation: pronominalisation of the relative-clause owner ─────────


def test_relational_navigation_pronominalises_the_subject():
    """When the chain root is the query subject, a deferred relational chain pronominalises the
    owner to the nominative *it* inside the relative clause."""
    m = variable(_NavMission, [])
    text = verbalize_expression(an(entity(m).where(m.assigned_to.battery > 5)))
    assert (
        "the battery of the _NavRobot to which it is assigned is greater than 5" in text
    )
    assert (
        "_NavMission is assigned" not in text
    )  # the owner is pronominalised, not repeated


def test_genitive_then_genitive_does_not_pronominalise_the_owner():
    """Two attributes of the relational referent in a row do *not* read *"its power"*: the first
    clause's subject is *the battery* (the head of its subject phrase), not the robot, so *"its"*
    would bind to the battery. The owner is spelled out instead — *"the power of the Robot"* (the
    robot reduced, already named)."""
    m = variable(_NavMission, [])
    text = verbalize_expression(
        an(entity(m).where(m.assigned_to.battery > 5, m.assigned_to.power > 10))
    )
    assert text == (
        "Find a _NavMission such that the battery of the _NavRobot to which it is "
        "assigned is greater than 5, and the power of the _NavRobot is greater than 10"
    )
    assert (
        "its power" not in text
    )  # the battery, not the robot, headed the first clause


def test_aggregation_where_on_measured_quantity_reduces_to_the_attribute():
    """When the WHERE filters the very attribute being aggregated, the relative clause is spelled
    out once (in the measure) and the repeat reduces to a bare *"the battery"* — not the whole
    possessive, and not *"its battery"* (which would re-introduce the owner)."""
    m = variable(_NavMission, [])
    nested = an(
        entity(eql.average(m.assigned_to.battery)).where(m.assigned_to.battery > 5)
    )
    text = verbalize_expression(nested)
    assert text == (
        "Find the average of the battery of the _NavRobot to which a _NavMission is "
        "assigned such that the battery is greater than 5"
    )
    assert text.count("to which") == 1  # spelled out once, then the bare attribute


def test_aggregation_where_on_other_attribute_spells_out_the_owner():
    """The aggregation foregrounds the measured quantity (the battery), not its owner, so a WHERE on
    a *different* attribute of that owner is **not** *"its power"* (which would misread as the
    battery's power) — it spells out *"the power of the Robot"*."""
    m = variable(_NavMission, [])
    nested = an(
        entity(eql.average(m.assigned_to.battery)).where(m.assigned_to.power > 5)
    )
    text = verbalize_expression(nested)
    assert "such that the power of the _NavRobot is greater than 5" in text
    assert "its power" not in text


def test_boolean_predicative_pronominalises_relational_navigation():
    """A boolean-terminal chain on the subject's relational navigation reads *"the <Type> to which
    it is <verb> is <attribute>"* — the navigation prefix is recursed through the standard grammar,
    so it pronominalises to the subject just like a deferred possessive chain."""
    m = variable(_NavMission, [])
    text = verbalize_expression(an(entity(m).where(m.assigned_to.operational)))
    assert "the _NavRobot to which it is assigned is operational" in text
    assert "_NavMission is assigned" not in text


def test_boolean_predicative_standalone_navigation_unchanged():
    """Outside a subject scope the same predicative keeps the full relative clause (no subject to
    pronominalise to)."""
    m = variable(_NavMission, [])
    assert (
        verbalize_expression(m.assigned_to.operational)
        == "the _NavRobot to which a _NavMission is assigned is operational"
    )


def test_attribute_through_relational_referent_pronominalises():
    """An attribute reached through the local centre reads as *"its <attribute>"* — a boolean
    predicative makes its relational referent the centre just as a possessive does."""
    m = variable(_NavMission, [])
    text = verbalize_expression(
        an(entity(m).where(m.assigned_to.operational, m.assigned_to.battery > 5))
    )
    assert text == (
        "Find a _NavMission such that the _NavRobot to which it is assigned is "
        "operational, and its battery is greater than 5"
    )


def test_its_continues_uniformly_once_the_subject_licenses_it():
    """A boolean predicative makes the robot the subject, so the attributes that follow read
    *"its battery … its power"* — uniformly pronominal. Once *"its"* refers to the robot it keeps it
    as the topic (a centering CONTINUE), so the run never mixes *"its battery"* with a re-named
    *"the power of the Robot"*."""
    m = variable(_NavMission, [])
    text = verbalize_expression(
        an(
            entity(m).where(
                m.assigned_to.operational,
                m.assigned_to.battery > 5,
                m.assigned_to.power > 1,
            )
        )
    )
    assert text == (
        "Find a _NavMission such that the _NavRobot to which it is assigned is "
        "operational, its battery is greater than 5, and its power is greater than 1"
    )


def test_two_distinct_relational_referents_are_numbered():
    """Two distinct relational referents of the same type are numbered *"Robot 1"* / *"Robot 2"*
    (bare, matching the variable convention) to tell them apart. A following attribute spells the
    numbered owner out (*"the power of Robot 1"*) rather than *"its power"* — the first clause's
    subject was the battery, not the robot."""
    p = variable(_NavPair, [])
    text = verbalize_expression(
        an(
            entity(p).where(
                p.primary.assigned_to.battery > 5,
                p.primary.assigned_to.power > 1,
                p.secondary.assigned_to.battery > 3,
            )
        )
    )
    assert text == (
        "Find a _NavPair such that the battery of _NavRobot 1, to which its primary is "
        "assigned, is greater than 5, the power of _NavRobot 1 is greater than 1, and the "
        "battery of _NavRobot 2, to which its secondary is assigned, is greater than 3"
    )


def test_single_relational_referent_is_not_numbered():
    """A lone relational referent (no same-type collision) keeps the plain definite form."""
    m = variable(_NavMission, [])
    text = verbalize_expression(an(entity(m).where(m.assigned_to.battery > 5)))
    assert "_NavRobot 1" not in text
    assert "the _NavRobot to which it is assigned" in text


def test_pronominal_relative_clause_agrees_with_subject_number():
    """The relative-clause copula agrees with the subject: *it is* (singular) / *they are*
    (plural) — exercised directly on the pronominal path."""
    from krrood.entity_query_language.verbalization.navigation_path import (
        PathStep,
        RelationStep,
    )
    from krrood.entity_query_language.verbalization.microplanning.possessive import (
        pronominal_path,
    )
    from krrood.entity_query_language.verbalization.fragments.features import (
        GrammaticalNumber,
    )
    from krrood.entity_query_language.verbalization.rendering.realization import (
        realize_subtree,
    )

    @dataclass
    class _Target:
        pass

    step = PathStep(
        "assigned_to",
        None,
        relation=RelationStep(_Target, _NavMission, "assigned", "to"),
    )
    assert (
        realize_subtree(pronominal_path([step], GrammaticalNumber.SINGULAR))
        == "the _Target to which it is assigned"
    )
    assert (
        realize_subtree(pronominal_path([step], GrammaticalNumber.PLURAL))
        == "the _Target to which they are assigned"
    )


def test_verbalize_non_bool_indexed_attribute_possession():
    r = variable(_Robot, [])
    text = verbalize_expression(r.tasks[0].name)
    # name is a str — should use possession/of form, NOT "is"
    assert "task" in text
    assert "name" in text
    assert " is " not in text


def test_verbalize_flat_variable_delegates_to_child():
    cabinet = variable(Cabinet, [])
    drawer_var = flat_variable(cabinet.drawers)
    text = verbalize_expression(drawer_var)
    assert "Cabinet" in text
    assert "drawers" in text


# ── Unit tests: comparators ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op,word",
    [
        ("__gt__", "greater than"),
        ("__lt__", "less than"),
        ("__ge__", "at least"),
        ("__le__", "at most"),
        ("__eq__", "is"),
        ("__ne__", "is not"),
    ],
)
def test_verbalize_comparator_operators(op, word):
    x = variable(int, [1])
    comparator = getattr(x, op)(5)
    text = verbalize_expression(comparator)
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
    # The flattened conjuncts on one bare variable factor into one shared-subject main clause; the
    # complementary bound pair folds to "between", and the inequality tail shares the lead copula.
    assert text == "an Integer is between 1 and 10 and not 5"


def test_verbalize_and_stops_at_or():
    x = variable(int, [])
    cond = and_(x > 1, or_(x < 10, x == 5))
    text = verbalize_expression(cond)
    assert "greater than" in text
    assert " or " in text  # the inner OR renders as an (inclusive) disjunction


def test_verbalize_or_chain():
    x = variable(int, [])
    cond = or_(x > 10, x < 0)
    text = verbalize_expression(cond)
    assert " or " in text
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
    assert " or " in text


# ── Unit tests: aggregators ────────────────────────────────────────────────────


def test_verbalize_count():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.count(x))
    assert "number of" in text and "Integers" in text


def test_verbalize_average():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.average(x))
    assert "average" in text and "Integers" in text


def test_verbalize_sum():
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.sum(x))
    assert "sum" in text and "Integers" in text


def test_verbalize_max_min():
    # MAX/MIN use SINGULAR_OF: child is verbalized via regular chain form, not plural.
    x = variable(int, [1, 2])
    max_text = verbalize_expression(eql.max(x))
    min_text = verbalize_expression(eql.min(x))
    assert "the maximum" in max_text
    assert "the minimum" in min_text
    assert "Integer" in max_text
    assert "Integer" in min_text


def test_aggregation_article_count_first_mention():
    """COUNT produces 'the number of …' even on first mention."""
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.count(x))
    assert text.startswith("the number of"), f"Got: {text!r}"


def test_aggregation_article_sum_first_mention():
    """SUM produces 'the sum of …' even on first mention."""
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.sum(x))
    assert text.startswith("the sum of"), f"Got: {text!r}"


def test_aggregation_article_average_first_mention():
    """AVERAGE produces 'the average of …' even on first mention."""
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.average(x))
    assert text.startswith("the average of"), f"Got: {text!r}"


def test_aggregation_article_max_first_mention():
    """MAX produces 'the maximum of …' even on first mention."""
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.max(x))
    assert text.startswith("the maximum of"), f"Got: {text!r}"


def test_aggregation_article_min_first_mention():
    """MIN produces 'the minimum of …' even on first mention."""
    x = variable(int, [1, 2])
    text = verbalize_expression(eql.min(x))
    assert text.startswith("the minimum of"), f"Got: {text!r}"


def test_max_single_attribute_chain():
    """MAX on a single-level attribute: 'the maximum of the <attr> of a <Type>'."""
    t = variable(BankTransaction, domain=None)
    text = verbalize_expression(an(entity(eql.max(t.amount_details))))
    assert (
        "the maximum of the amount_details of a BankTransaction" in text
    ), f"Got: {text!r}"


def test_min_single_attribute_chain():
    """MIN on a single-level attribute: 'the minimum of the <attr> of a <Type>'."""
    t = variable(BankTransaction, domain=None)
    text = verbalize_expression(an(entity(eql.min(t.amount_details))))
    assert (
        "the minimum of the amount_details of a BankTransaction" in text
    ), f"Got: {text!r}"


def test_max_multi_level_attribute_chain():
    """MAX on a multi-level chain: article before agg word, 'of' separator, no leading article on child."""
    t = variable(BankTransaction, domain=None)
    query = an(entity(eql.max(t.amount_details.amount)))
    text = verbalize_expression(query)
    assert (
        "the maximum of the amount of the amount_details of a BankTransaction" in text
    ), f"Got: {text!r}"


def test_min_multi_level_attribute_chain():
    """MIN on a multi-level chain: mirrors MAX behaviour."""
    t = variable(BankTransaction, domain=None)
    query = an(entity(eql.min(t.amount_details.amount)))
    text = verbalize_expression(query)
    assert (
        "the minimum of the amount of the amount_details of a BankTransaction" in text
    ), f"Got: {text!r}"


def test_grouped_having_on_max_fronts_onto_group_key():
    """A MAX HAVING fronts onto the group key as a determiner-less possession (*"whose maximum of …"*);
    the reported column restates it in full (*"the maximum of …"*)."""
    t = variable(BankTransaction, domain=None)
    max_amount = eql.max(t.amount_details.amount)
    query = a(set_of(max_amount).grouped_by(t.amount_details).having(max_amount > 100))
    text = verbalize_expression(query)
    assert text.startswith("For each amount_details whose maximum of")
    assert "is greater than 100, report the maximum of" in text


# ── Nested sub-queries as values (the imperative "Find" is reserved for the top level) ──


def test_nested_unconstrained_aggregation_no_second_find():
    """An aggregation sub-query used as a comparison value must not emit a second 'Find'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.max(t2.amount_details.amount)))
        )
    )
    text = verbalize_expression(query)
    assert text.count("Find") == 1, f"Expected exactly one 'Find' in: {text!r}"
    assert "maximum" in text, f"Got: {text!r}"


def test_nested_non_aggregation_entity_no_second_find():
    """A nested non-aggregation entity sub-query renders as a noun phrase, not a second 'Find'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(t1 == an(entity(t2).where(t2.amount_details.amount > 100)))
    )
    text = verbalize_expression(query)
    assert text.count("Find") == 1, f"Expected exactly one 'Find' in: {text!r}"
    assert "where" in text, f"Got: {text!r}"


def test_nested_constrained_aggregation_preserves_filter():
    """A constrained aggregation sub-query keeps its filter when verbalized."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount
            == an(
                entity(eql.max(t2.amount_details.amount)).where(
                    t2.booking_date < datetime.datetime(2024, 1, 1)
                )
            )
        )
    )
    text = verbalize_expression(query)
    assert "booking_date" in text, f"Got: {text!r}"
    assert "before" in text, f"Got: {text!r}"


def test_deeply_nested_subqueries_golden():
    """Nesting has no depth limit: three levels of constrained aggregation sub-queries render
    with exactly one top-level 'Find' and every deeper query as a noun phrase.  Each level's
    subject is differentiated: the unique top-level subject pronominalises to 'its', the
    level-2 population to 'their' (never the ambiguous singular 'the BankTransaction'), and
    the level-3 condition folds into 'whose'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    t3 = variable(BankTransaction, domain=None)
    level3 = an(
        entity(eql.min(t3.amount_details.amount)).where(
            t3.booking_date < datetime.datetime(2024, 1, 1)
        )
    )
    level2 = an(
        entity(eql.max(t2.amount_details.amount)).where(
            t2.amount_details.amount == level3
        )
    )
    query = eql.the(entity(t1).where(t1.amount_details.amount == level2))
    text = verbalize_expression(query)
    assert text == (
        "Find the unique BankTransaction such that the amount of its amount_details "
        "is equal to the maximum amount among BankTransactions such that "
        "the amount of their amount_details is equal to "
        "the minimum amount among BankTransactions whose booking_date is before "
        "January 1, 2024"
    )
    assert text.count("Find") == 1  # only the top level emits the imperative


def test_superlative_fold_max_and_min():
    """``subject.<chain> == max/min(<same-type>.<same chain>)`` folds to the superlative
    *"with the maximum/minimum <leaf>"* — meaning-preserving (the superlative *is* "equal to the
    extreme over the whole population"), not an optimisation."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    max_query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.max(t2.amount_details.amount)))
        )
    )
    assert (
        verbalize_expression(max_query)
        == "Find the unique BankTransaction with the maximum amount"
    )
    t3 = variable(BankTransaction, domain=None)
    t4 = variable(BankTransaction, domain=None)
    min_query = eql.the(
        entity(t3).where(
            t3.amount_details.amount == an(entity(eql.min(t4.amount_details.amount)))
        )
    )
    assert (
        verbalize_expression(min_query)
        == "Find the unique BankTransaction with the minimum amount"
    )


def test_superlative_fold_declines_when_aggregation_is_constrained_or_grouped():
    """The fold is strict: a *constrained* extreme sub-query is load-bearing, so it stays the
    explicit *"such that … is equal to the maximum amount among … whose …"* form."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount
            == an(
                entity(eql.max(t2.amount_details.amount)).where(
                    t2.booking_date < datetime.datetime(2024, 1, 1)
                )
            )
        )
    )
    text = verbalize_expression(query)
    assert "with the maximum amount" not in text, f"Should not fold: {text!r}"
    assert "is equal to the maximum amount among BankTransactions" in text


def test_superlative_fold_declines_for_non_extreme_aggregation():
    """Only Max/Min are superlatives — SUM stays *"is equal to the sum of amounts"*."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.sum(t2.amount_details.amount)))
        )
    )
    assert "the sum of amounts" in verbalize_expression(query)


def test_superlative_fold_declines_on_different_chain():
    """The fold requires the *same* attribute chain on both sides — comparing one attribute to
    the maximum of a *different* one is not a superlative and must not fold."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.max(t2.booking_date)))
        )
    )
    text = verbalize_expression(query)
    assert "with the maximum" not in text, f"Should not fold: {text!r}"
    assert "the maximum booking_date" in text, f"Got: {text!r}"


def test_nested_aggregation_sum_uses_plural_leaf():
    """SUM (a '… of' aggregation) collapses with a plural leaf: 'the sum of amounts'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.sum(t2.amount_details.amount)))
        )
    )
    text = verbalize_expression(query)
    assert "the sum of amounts" in text, f"Got: {text!r}"


def test_nested_constrained_aggregation_no_second_find():
    """A constrained aggregation sub-query keeps its filter via 'among … such that …', no second 'Find'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount
            == an(
                entity(eql.max(t2.amount_details.amount)).where(
                    t2.booking_date < datetime.datetime(2024, 1, 1)
                )
            )
        )
    )
    text = verbalize_expression(query)
    assert text.count("Find") == 1, f"Expected exactly one 'Find' in: {text!r}"
    assert "the maximum amount among" in text, f"Got: {text!r}"
    assert "booking_date" in text and "before" in text, f"Got: {text!r}"


def test_nested_aggregation_source_not_numbered_on_outer_subject():
    """The aggregation source is a population, not a numbered entity: the outer subject stays unnumbered."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount == an(entity(eql.max(t2.amount_details.amount)))
        )
    )
    text = verbalize_expression(query)
    assert text == "Find the unique BankTransaction with the maximum amount"
    assert "BankTransaction 1" not in text, f"Spurious numbering in: {text!r}"
    assert "BankTransaction 2" not in text, f"Spurious numbering in: {text!r}"


def test_nested_constrained_aggregation_scope_is_plural_unnumbered():
    """A constrained aggregation scope reads 'among BankTransactions', not 'among BankTransaction 2'."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(
            t1.amount_details.amount
            == an(
                entity(eql.max(t2.amount_details.amount)).where(
                    t2.booking_date < datetime.datetime(2024, 1, 1)
                )
            )
        )
    )
    text = verbalize_expression(query)
    assert "among BankTransactions" in text, f"Got: {text!r}"
    assert "BankTransaction 2" not in text, f"Spurious numbering in: {text!r}"


def test_nested_non_aggregation_entity_keeps_numbering():
    """A genuine (non-aggregation) entity sub-query is a specific entity and keeps its disambiguation number."""
    t1 = variable(BankTransaction, domain=None)
    t2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(t1).where(t1 == an(entity(t2).where(t2.amount_details.amount > 100)))
    )
    text = verbalize_expression(query)
    assert "BankTransaction 1" in text and "BankTransaction 2" in text, f"Got: {text!r}"


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
    assert "task" in text
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
    # The selection IS the group key, so the grouping is fronted as a distinct listing,
    # never a trailing "grouped by".
    assert "distinct" in text
    assert "grouped by" not in text
    assert "ordered by" in text
    assert "number" in text
    assert "Cabinets" in text
    assert "drawers" in text
    assert "from highest to lowest" in text


def test_verbalize_complex_having(departments_and_employees_fixture):
    departments, employees = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    department = employee.department
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(department, avg_salary).grouped_by(department).having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    assert "Employees" in text
    assert "department" in text
    assert "average" in text
    assert "salaries" in text
    assert (
        "For each department whose" in text
    )  # report fronted, HAVING on the group key
    assert "grouped by" not in text
    assert (
        "is greater than" in text
    )  # HAVING woven as a "whose <aggregate> is …" filter
    assert "30000" in text


def test_grouped_having_fronts_aggregate_onto_group_key():
    """A grouped HAVING fronts onto the group key as *"For each department whose <aggregate> is …"* —
    the aggregate a determiner-less possession of the group — then the report restates it in full.
    """
    employee = variable(Employee, domain=None)
    total = eql.sum(employee.salary)
    query = a(
        set_of(employee.department, total)
        .grouped_by(employee.department)
        .having(total > 30000)
    )
    text = verbalize_expression(query)
    assert text == (
        "For each department whose sum of salaries of Employees is greater than 30000, "
        "report the sum of salaries of Employees"
    )


def test_grouped_selection_equal_to_key_reports_distinct():
    """A grouped query whose selection IS the group key reports the distinct keys, fronted —
    never a trailing 'grouped by'."""
    employee = variable(Employee, domain=None)
    text = verbalize_expression(
        a(set_of(employee.department).grouped_by(employee.department))
    )
    assert text == "Report the distinct departments"
    assert "grouped by" not in text


def test_grouped_selection_other_than_key_fronts_for_each_all():
    """A grouped query with a non-key selection fronts the grouping as 'For each <key>' and lists
    the per-group population with 'all'."""
    employee = variable(Employee, domain=None)
    text = verbalize_expression(an(entity(employee).grouped_by(employee.department)))
    assert text == "For each department, report all Employees"
    assert "grouped by" not in text


def test_verbalize_nested_rule(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = an(FixedConnection).from_(world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer_var = inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )
    # Wrap in entity() to trigger the if/then rule form
    text = verbalize_expression(entity(drawer_var))

    # if/then structure
    assert text.startswith("If "), f"Expected 'If' at start, got: {text!r}"
    assert "then" in text, f"Expected 'then' in: {text!r}"
    # IF clause introduces the FixedConnection antecedent
    assert (
        "there's a FixedConnection" in text
    ), f"Expected 'there's a FixedConnection' in: {text!r}"
    # IF clause describes its conditions, each prefixed by its own "whose …, and whose …"
    assert "whose parent is the child of a PrismaticConnection" in text
    assert "and whose child is a Handle" in text
    # THEN clause introduces the Drawer
    assert "there's a Drawer" in text, f"Expected 'there's a Drawer' in: {text!r}"
    # THEN clause repeats "whose" per binding: "whose …, and whose …"
    assert "whose container is the parent of the FixedConnection" in text
    assert "and whose handle is the child of the FixedConnection" in text


def test_verbalize_inference_rule_golden(doors_and_drawers_world):
    """Exact IF/THEN surface for a sub-query inference rule (pins the whole sentence:
    antecedent intro, ``whose`` conditions, the ``, then`` join, and consequent bindings —
    including the FixedConnection reading ``a`` first then ``the`` via coreference)."""
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = an(FixedConnection).from_(world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer_var = inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )
    assert verbalize_expression(entity(drawer_var)) == (
        "If there's a FixedConnection whose parent is the child of a PrismaticConnection, "
        "and whose child is a Handle, "
        "then there's a Drawer whose container is the parent of the FixedConnection, "
        "and whose handle is the child of the FixedConnection"
    )


def test_verbalize_inference_no_sub_query_golden(doors_and_drawers_world):
    """Exact surface for a plain-binding inference rule (no antecedent sub-query): a noun
    phrase with appositive ``, where …`` bindings, no IF/THEN block, no ``such that``.
    """
    world = doors_and_drawers_world
    handle_variable = variable(Handle, world.bodies)
    container_variable = variable(Container, world.bodies)
    drawer = inference(Drawer)(handle=handle_variable, container=container_variable)
    assert verbalize_expression(drawer) == (
        "a Drawer, where the handle of the Drawer is a Handle, "
        "and the container of the Drawer is a Container"
    )


def test_verbalize_condition_graph_example():
    @dataclass(frozen=True)
    class Item:
        value: int

    value = variable_from([6])
    item_var = inference(Item)(value=value)
    query = entity(item_var).where(or_(and_(value > 5, value < 10), value == 11))
    text = verbalize_expression(query)

    assert "Item" in text
    assert " or " in text
    # The bare-variable bound pair folds to "between" inside its relative clause.
    assert "between 5 and 10" in text
    assert "is 11" in text


def test_verbalize_has_type_with_exists():
    fb1 = FruitBox("FruitBox1", [Apple("apple"), Body("Body1")])
    fb2 = FruitBox("FruitBox2", [Body("Body3")])
    fruit_box = variable(FruitBox, domain=[fb1, fb2])
    query = an(
        entity(fruit_box).where(
            exists(fruit_box, HasType(flat_variable(fruit_box.fruits), Apple))
        )
    )
    text = verbalize_expression(query)

    assert "FruitBox" in text
    assert "exists" in text
    assert "Apple" in text
    assert "is of type" in text


# ── Predicate fragment tests ────────────────────────────────────────────────────


def test_verbalize_has_type():
    fruit = variable(Body, [])
    predicate = HasType(fruit, Apple)
    text = verbalize_expression(predicate)
    assert "Body" in text
    assert "is of type" in text
    assert "Apple" in text


def test_verbalize_has_type_tuple_of_types():
    fruit = variable(Body, [])
    predicate = HasType(fruit, (Apple, Body))
    text = verbalize_expression(predicate)
    assert "is of type" in text
    assert "Apple" in text
    assert "Body" in text


def test_has_type_lists_a_tuple_of_types_disjunctively():
    """``isinstance`` over a tuple holds when the value is ANY of the types, so the listing joins
    with *"or"* — *"is of type Apple or Body"*, never the conjunctive *"and"* (a value cannot be of
    both types at once)."""
    fruit = variable(Body, [])
    assert (
        verbalize_expression(HasType(fruit, (Apple, Body)))
        == "a Body is of type Apple or Body"
    )


def test_verbalize_contains_type():
    fruit_box = variable(FruitBox, [])
    predicate = ContainsType(fruit_box.fruits, Apple)
    text = verbalize_expression(predicate)
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
        def _verbalization_fragment_(cls, fields):
            return clause(Noun(fields["body"]), Copula(), Adjective("reachable"))

    world = handles_and_containers_world
    handle = variable(Handle, world.bodies)
    predicate = IsReachable(handle)
    text = verbalize_expression(predicate)
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
        def _verbalization_fragment_(cls, fields):
            return clause(
                Noun(fields["employee"]),
                Verb("work"),
                Prepositions.IN,
                Noun(fields["department"]),
            )

    employee = variable(Employee, [])
    department = variable(Department, [])
    predicate = WorksInDepartment(employee, department)
    text = verbalize_expression(predicate)
    assert "Employee" in text
    assert "works in" in text
    assert "Department" in text


def test_verbalize_predicate_without_fragment_raises():
    """A predicate that supplies no verbalization fragment is an error — there is no name-based
    string fallback; fragments are required."""

    @dataclass(eq=False)
    class HasHighSalary(Predicate):
        employee: Any
        threshold: float

        def __call__(self) -> bool:
            return self.employee.salary > self.threshold

    employee = variable(Employee, [])
    with pytest.raises(PredicateFragmentRequiredError):
        verbalize_expression(HasHighSalary(employee, 50000.0))


def test_verbalize_predicate_without_fragment_no_args_raises():
    @dataclass(eq=False)
    class IsActive(Predicate):
        entity: Any

        def __call__(self) -> bool:
            return True

    employee = variable(Employee, [])
    with pytest.raises(PredicateFragmentRequiredError):
        verbalize_expression(IsActive(employee))


# ── Aggregator coreference & HAVING compact form ──────────────────────────────


def test_aggregator_coreference_second_mention_is_the(
    departments_and_employees_fixture,
):
    """Same aggregator expression in set_of and having → both mentions include 'the'."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(avg_salary).grouped_by(employee.department).having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    assert "the average of" in text


def test_grouped_having_fronts_onto_group_key(departments_and_employees_fixture):
    """A grouped HAVING is woven onto the group key as a *"For each department whose <aggregate> is
    <op> <value>"* filter, so the group condition is unambiguous rather than the bare *"having the sum
    greater than …"* that misparses as modifying the reported population."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(employee.department, avg_salary)
        .grouped_by(employee.department)
        .having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    assert text.startswith("For each department whose ")
    whose_part = text[text.index("whose") :]
    assert "is greater than" in whose_part


def test_where_keeps_is_copula(departments_and_employees_fixture):
    """A single-hop subject predicate groups under 'whose' but keeps the full 'is greater than' form."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    query = a(entity(employee).where(employee.salary > 30000))
    text = verbalize_expression(query)

    where_part = text[text.index("whose") :]
    assert "is greater than" in where_part


def test_having_compound_condition_renders_full_clauses(
    departments_and_employees_fixture,
):
    """AND/OR inside the fronted HAVING *whose* clause: every comparator reads as a full *"is <op> …"* clause."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    count_emp = eql.count(employee)
    query = a(
        set_of(employee.department, avg_salary)
        .grouped_by(employee.department)
        .having(and_(avg_salary > 30000, count_emp >= 2))
    )
    text = verbalize_expression(query)

    whose_part = text[text.index("whose") :]
    assert "is greater than" in whose_part
    assert "is at least" in whose_part


def test_having_negated_comparator_renders_full_clause(
    departments_and_employees_fixture,
):
    """NOT over a comparator inside the fronted HAVING *whose* clause reads as *"is not <op> …"*."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(employee.department, avg_salary)
        .grouped_by(employee.department)
        .having(not_(avg_salary > 30000))
    )
    text = verbalize_expression(query)

    whose_part = text[text.index("whose") :]
    assert "is not greater than" in whose_part


def test_set_of_grouped_by_reports_without_restating_the_key(
    departments_and_employees_fixture,
):
    """A grouped aggregation reads as a single fronted report — no double header, no trailing
    'grouped by', and the group key named once (in 'For each …'), not restated as a column.
    """
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(employee.department, avg_salary)
        .grouped_by(employee.department)
        .having(avg_salary > 30000)
    )
    text = verbalize_expression(query)

    assert text.startswith("For each department whose ")
    assert text.count("report") == 1  # no double header
    assert "Find" not in text and "grouped by" not in text
    assert (
        text.count("department") == 1
    )  # named once in "For each department", not restated
    assert "whose" in text and "30000" in text


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


def test_verbalize_having_eq_uses_is_copula():
    """The fronted HAVING *whose* clause renders ``==`` with the copula like any other clause —
    *"whose number of Employees is equal to 2"* — not the copula-less *"equals"* the old compact
    HAVING form used.
    """
    employee = variable(Employee, domain=None)
    count_emp = eql.count(employee)
    query = a(
        set_of(employee.department, count_emp)
        .grouped_by(employee.department)
        .having(count_emp == 2)
    )
    text = verbalize_expression(query)
    whose_part = text[text.index("whose") :]
    assert "is equal to 2" in whose_part


# ── Non-predicate InstantiatedVariable natural-English form ───────────────────


def test_verbalize_inference_no_sub_query(doors_and_drawers_world):
    """inference(...) with plain variable bindings: 'where' clause, no 'such that'."""
    world = doors_and_drawers_world
    handle_variable = variable(Handle, world.bodies)
    container_variable = variable(Container, world.bodies)
    drawer = inference(Drawer)(handle=handle_variable, container=container_variable)
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
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = an(FixedConnection).from_(world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer = inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )
    text = verbalize_expression(drawer)

    assert "a FixedConnection" in text
    assert "the FixedConnection" in text
    # Indefinite article comes before definite in the binding section
    assert text.index("a FixedConnection") < text.index("the FixedConnection")


def test_verbalize_inference_literal_field(doors_and_drawers_world):
    """A child var that is a Python literal value is rendered inside the binding."""
    world = doors_and_drawers_world
    handle_variable = variable(Handle, world.bodies)
    container_variable = variable(Container, world.bodies)
    drawer = inference(Drawer)(
        handle=handle_variable, container=container_variable, correct=True
    )
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
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = an(FixedConnection).from_(world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer_var = inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )
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
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = an(FixedConnection).from_(world.connections)(
        parent=prismatic_connection.child, child=handle
    )
    drawer_var = inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )

    # A second entity used directly by the outer Wrapper
    handle2 = variable(Handle, world.bodies)
    pc2 = variable(PrismaticConnection, world.connections)
    fc2 = an(FixedConnection).from_(world.connections)(parent=pc2.child, child=handle2)

    wrapper_var = inference(Wrapper)(
        drawer=drawer_var, connection=fc2.expression.parent
    )
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

    source = variable(Body, [])
    target = variable(Handle, [])
    predicate = ConnectsTo(source, target)
    text = verbalize_expression(predicate)

    assert "Body" in text
    assert "connects to" in text
    assert "Handle" in text
    # Subject–predicate–object order
    assert text.index("Body") < text.index("Handle")


def test_verbalize_1arg_predicate_without_fragment_raises():
    """A 1-arg predicate without a verbalization fragment is an error — fragments are required, with
    no generic name-based fallback."""

    @dataclass(eq=False)
    class IsActive(Predicate):
        entity: Any

        def __call__(self) -> bool:
            return True

    employee = variable(Employee, [])
    with pytest.raises(PredicateFragmentRequiredError):
        verbalize_expression(IsActive(employee))


# ── Same-type variable disambiguation ─────────────────────────────────────────


def test_two_same_type_variables_are_disambiguated():
    """Two distinct variables of the same type must get numbered labels to tell them apart.

    Currently both variables produce 'an Employee', making the output ambiguous:
    'an Employee's salary is greater than an Employee's salary'
    Expected: 'Employee 1's salary is greater than Employee 2's salary'
    """
    employee1 = variable(Employee, [])
    employee2 = variable(Employee, [])
    cond = employee1.salary > employee2.salary
    text = verbalize_expression(cond)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"


def test_two_same_type_variables_in_query_are_disambiguated():
    """Two same-type variables inside a query must each get distinct numbered labels."""
    employee1 = variable(Employee, [])
    employee2 = variable(Employee, [])
    query = an(entity(employee1).where(employee1.salary > employee2.salary))
    text = verbalize_expression(query)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"


def test_three_same_type_variables_are_disambiguated():
    """Three variables of the same type each get distinct numbered labels."""
    employee1 = variable(Employee, [])
    employee2 = variable(Employee, [])
    emp3 = variable(Employee, [])
    query = an(
        entity(employee1).where(
            and_(employee1.salary > employee2.salary, employee2.salary > emp3.salary)
        )
    )
    text = verbalize_expression(query)
    assert "Employee 1" in text, f"Expected 'Employee 1' in: {text!r}"
    assert "Employee 2" in text, f"Expected 'Employee 2' in: {text!r}"
    assert "Employee 3" in text, f"Expected 'Employee 3' in: {text!r}"


def test_single_type_variable_not_numbered():
    """A single variable of a type must keep the plain 'an Employee' form — no numbering."""
    employee = variable(Employee, [])
    text = verbalize_expression(employee)
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
    query = an(
        entity(eql.sum(t.amount_details.amount)).where(
            t.booking_date < datetime.datetime(2024, 5, 17)
        )
    )

    text = verbalize_expression(query)
    assert "BankTransaction" in text, f"Expected 'BankTransaction' in: {text!r}"
    assert "sum of" in text, f"Expected 'sum of' in: {text!r}"


def test_cabinet_rule_verbalization(handles_and_containers_world):
    """if/then form: aggregated antecedents plural, group-key binding uses 'common'."""
    drawer = variable(Drawer, handles_and_containers_world.views)
    prismatic_connection = variable(
        PrismaticConnection, handles_and_containers_world.connections
    )
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
    assert (
        "there are PrismaticConnections" in text
    ), f"Expected 'there are PrismaticConnections' in: {text!r}"
    # IF clause contains a "whose children" constraint from the WHERE condition
    assert "whose children" in text, f"Expected 'whose children' in: {text!r}"
    # THEN clause introduces the Cabinet
    assert "there's a Cabinet" in text, f"Expected 'there's a Cabinet' in: {text!r}"
    # group-key binding uses "common"
    assert (
        "whose container is the common parent of the PrismaticConnections" in text
    ), f"Expected 'whose container is the common parent of the PrismaticConnections' in: {text!r}"
    # aggregated plural binding (second clause, with its own repeated "whose")
    assert (
        "and whose drawers are the Drawers" in text
    ), f"Expected 'and whose drawers are the Drawers' in: {text!r}"


def test_inference_planner_decomposes_rule_without_rendering(
    handles_and_containers_world,
):
    """The InferencePlanner produces the RuleStructure as pure data — no fragments rendered."""
    from krrood.entity_query_language.verbalization.grammar.inference.planner import (
        AggregationStatus,
        InferencePlanner,
    )

    drawer = variable(Drawer, handles_and_containers_world.views)
    prismatic_connection = variable(
        PrismaticConnection, handles_and_containers_world.connections
    )
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
    query.build()

    assert InferencePlanner.can_handle(query) is True
    plan = InferencePlanner(query).plan()

    assert plan.consequent_type == "Cabinet"
    assert [b.field_name for b in plan.consequent_bindings] == ["container", "drawers"]
    assert plan.group_key_ids  # grouped_by → non-empty
    # "container" is the group key; "drawers" is aggregated.
    by_field = {b.field_name: b.aggregation_status for b in plan.consequent_bindings}
    assert by_field["container"] == AggregationStatus.GROUP_KEY
    assert by_field["drawers"] == AggregationStatus.AGGREGATED

    # The planner only collects each antecedent's raw conditions; choosing the surface form
    # (whose / standalone) is the condition-form registry's concern at render time.
    from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
        ConditionForm,
        Placement,
        SurfacePosition,
    )

    conditions = [
        condition
        for antecedent in plan.primary_antecedents
        for condition in antecedent.conditions
    ]
    assert conditions, "expected at least one attributed antecedent condition"
    antecedent = plan.primary_antecedents[0]
    form = ConditionForm.most_applicable(
        Placement(item=antecedent.conditions[0], subject=antecedent.variable)
    )
    assert form.position is SurfacePosition.WHOSE


def test_query_planner_collects_subject_restriction_without_placing():
    """The QueryPlanner only flattens the WHERE into conjuncts — range-folding and choosing each
    conjunct's surface form/slot are the condition-form registry's concern, not the plan's.
    """
    from krrood.entity_query_language.verbalization.grammar.query.planner import (
        QueryPlanner,
        SelectionKind,
    )
    from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
        ConditionForm,
        Placement,
        SurfacePosition,
        WhosePredicateForm,
    )

    r = variable(_Robot, [])
    query = entity(r).where(r.battery > 50)
    plan = QueryPlanner(query).plan()

    assert plan.kind is SelectionKind.SUBJECT
    assert plan.is_the is False
    assert plan.selected_type == "_Robot"
    assert plan.subject is not None
    assert plan.is_aggregation_subquery is False

    # The plan carries only the raw conjuncts — no folding, no placement decision.
    assert plan.subject_restriction is not None
    assert len(plan.subject_restriction.conditions) == 1
    conjunct = plan.subject_restriction.conditions[0]

    # "battery > 50" is a single-hop, non-boolean attribute predicate → the registry selects the
    # whose-predicate form (slot WHOSE: "whose battery is greater than 50").
    form = ConditionForm.most_applicable(Placement(item=conjunct, subject=plan.subject))
    assert form is WhosePredicateForm
    assert form.position is SurfacePosition.WHOSE


def test_instantiated_planner_decomposes_bindings_without_rendering(
    handles_and_containers_world,
):
    """The InstantiatedPlanner records type + per-field number as pure data — no fragments."""
    from krrood.entity_query_language.verbalization.grammar.instantiated.planner import (
        InstantiatedPlanner,
    )

    drawer = variable(Drawer, handles_and_containers_world.views)
    instantiated = inference(Cabinet)(drawers=drawer)
    plan = InstantiatedPlanner(instantiated).plan()

    assert plan.type_name == "Cabinet"
    # "drawers" is a plural field name → plural copula/value at realisation time.
    assert [(b.field_name, b.is_plural) for b in plan.bindings] == [("drawers", True)]


def test_grouped_by_planner_extracts_keys_without_rendering(
    handles_and_containers_world,
):
    """The GroupedByPlanner reads the GROUP BY keys as pure data — no fragments rendered."""
    from krrood.entity_query_language.verbalization.grammar.clauses.planner import (
        GroupedByPlanner,
    )

    cabinet = variable(Cabinet, handles_and_containers_world.views)
    query = entity(cabinet).grouped_by(cabinet)
    query.build()

    plan = GroupedByPlanner(query).plan()
    assert plan.has_keys
    assert len(plan.keys) == 1
    # The selected variable *is* the group key, so nothing is aggregated over it.
    assert plan.aggregated == []


def test_plural_field_binding_uses_are(handles_and_containers_world):
    """A plural field name in an InstantiatedVariable binding uses 'are <Plural>' not 'is <Article> <Singular>'."""
    drawer = variable(Drawer, handles_and_containers_world.views)
    cabinet = inference(Cabinet)(drawers=drawer)
    text = verbalize_expression(cabinet)
    assert (
        "drawers of the Cabinet are Drawers" in text
    ), f"Expected 'drawers of the Cabinet are Drawers' in: {text!r}"
    assert "drawers is" not in text, f"Did not expect 'drawers is' in: {text!r}"


def test_grouped_by_without_instantiated_variable(handles_and_containers_world):
    """grouped_by where the selection IS the group key renders a fronted distinct listing
    (never a trailing 'grouped by')."""
    cabinet = variable(Cabinet, handles_and_containers_world.views)
    drawer = flat_variable(cabinet.drawers)
    query = an(
        entity(cabinet)
        .grouped_by(cabinet)
        .ordered_by(eql.count(drawer), descending=True)
    )
    text = verbalize_expression(query)
    assert (
        text
        == "Report the distinct Cabinets ordered by the number of drawers of Cabinets from highest to lowest"
    )


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def departments_and_employees_fixture():
    d1 = Department("HR")
    d2 = Department("Finance")
    e1 = Employee("John", d1, 10000)
    e2 = Employee("Anna", d2, 40000)
    return [d1, d2], [e1, e2]


# ── Reference pronouns, condition grouping, range folding, calc-equality ────────


def test_example_sum_between_full_sentence():
    """Motivating example 1: pronoun + calc-equality + 'whose … between' in the aggregation scope."""
    bank_transaction = variable(BankTransaction, domain=None)
    bt_sum = variable(BankTransaction, domain=None)
    start_date = datetime.datetime(2026, 5, 15)
    end_date = datetime.datetime(2026, 5, 30)
    sum_val = an(
        entity(eql.sum(bt_sum.amount_details.amount)).where(
            bt_sum.booking_date >= start_date,
            bt_sum.booking_date <= end_date,
        )
    )
    query = an(
        entity(bank_transaction).where(
            bank_transaction.amount_details.amount == sum_val
        )
    )
    assert verbalize_expression(query) == (
        "Find a BankTransaction such that the amount of its amount_details is equal to "
        "the sum of amounts among BankTransactions whose booking_date is between "
        "May 15, 2026 and May 30, 2026"
    )


def test_example_max_full_sentence():
    """Motivating example 2: pronoun + calc-equality, multi-hop residual keeps 'such that'.

    Uses a *constrained* max (which does not fold to a superlative) so the calc-equality residual
    survives as the full *"such that … is equal to the maximum amount among … whose …"* sentence.
    """
    bank_transaction = variable(BankTransaction, domain=None)
    bt2 = variable(BankTransaction, domain=None)
    max_q = an(
        entity(eql.max(bt2.amount_details.amount)).where(
            bt2.booking_date < datetime.datetime(2024, 1, 1)
        )
    )
    query = eql.the(
        entity(bank_transaction).where(bank_transaction.amount_details.amount == max_q)
    )
    assert verbalize_expression(query) == (
        "Find the unique BankTransaction such that the amount of its amount_details "
        "is equal to the maximum amount among BankTransactions whose booking_date "
        "is before January 1, 2024"
    )


def test_pronoun_multi_hop_chain_elides_subject():
    """A multi-hop chain rooted at the subject becomes 'the amount of its amount_details'.

    Uses a *constrained* max so the chain stays a residual comparison (an unconstrained max would
    fold to the superlative 'with the maximum amount', eliding the chain entirely)."""
    bank_transaction = variable(BankTransaction, domain=None)
    bt2 = variable(BankTransaction, domain=None)
    query = eql.the(
        entity(bank_transaction).where(
            bank_transaction.amount_details.amount
            == an(
                entity(eql.max(bt2.amount_details.amount)).where(
                    bt2.booking_date < datetime.datetime(2024, 1, 1)
                )
            )
        )
    )
    text = verbalize_expression(query)
    assert "the amount of its amount_details" in text, f"Got: {text!r}"
    assert "of the BankTransaction" not in text, f"Subject not elided in: {text!r}"


def test_pronoun_single_hop_in_residual_or():
    """A single-hop subject chain that stays residual (inside OR) uses 'its booking_date'."""
    bank_transaction = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 1, 1)
    upper_bound = datetime.datetime(2026, 12, 31)
    query = an(
        entity(bank_transaction).where(
            or_(
                bank_transaction.booking_date < lower_bound,
                bank_transaction.booking_date > upper_bound,
            )
        )
    )
    text = verbalize_expression(query)
    assert "its booking_date" in text, f"Got: {text!r}"
    assert "of the BankTransaction" not in text, f"Got: {text!r}"


def test_pronoun_not_used_for_non_subject_variable():
    """Only the subject is pronominalised; another variable in the condition stays explicit."""
    bank_transaction = variable(BankTransaction, domain=None)
    employee = variable(Employee, domain=None)
    query = an(
        entity(bank_transaction).where(
            bank_transaction.amount_details.amount == employee.starting_salary
        )
    )
    text = verbalize_expression(query)
    assert "the amount of its amount_details" in text, f"Got: {text!r}"
    assert "the starting_salary of an Employee" in text, f"Got: {text!r}"
    # object (non-calculation) equality keeps the bare copula
    assert (
        "is the starting_salary" in text and "is equal to" not in text
    ), f"Got: {text!r}"


def test_whose_grouping_top_level_between():
    """A >=/<= pair on a single-hop subject attribute folds into 'whose … is between … and …'."""
    bank_transaction = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 5, 15)
    upper_bound = datetime.datetime(2026, 5, 30)
    query = an(
        entity(bank_transaction).where(
            bank_transaction.booking_date >= lower_bound,
            bank_transaction.booking_date <= upper_bound,
        )
    )
    assert verbalize_expression(query) == (
        "Find a BankTransaction whose booking_date is between May 15, 2026 and May 30, 2026"
    )


def test_whose_grouping_mixed_groupable_and_residual():
    """Groupable single-hop predicates go under 'whose'; multi-hop ones stay in 'such that'."""
    bank_transaction = variable(BankTransaction, domain=None)
    bt2 = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 5, 15)
    upper_bound = datetime.datetime(2026, 5, 30)
    query = an(
        entity(bank_transaction).where(
            bank_transaction.booking_date >= lower_bound,
            bank_transaction.booking_date <= upper_bound,
            bank_transaction.amount_details.amount
            == an(
                entity(eql.max(bt2.amount_details.amount)).where(
                    bt2.booking_date < datetime.datetime(2024, 1, 1)
                )
            ),
        )
    )
    text = verbalize_expression(query)
    assert (
        "whose booking_date is between May 15, 2026 and May 30, 2026" in text
    ), f"Got: {text!r}"
    assert (
        "such that the amount of its amount_details is equal to the maximum amount"
        in text
    ), f"Got: {text!r}"


def test_top_level_aggregation_max_whose():
    """A top-level max selection ends with its source variable, so its WHERE folds into 'whose'."""
    bank_transaction = variable(BankTransaction, domain=None)
    query = an(
        entity(eql.max(bank_transaction, key=lambda t: t.amount_details.amount)).where(
            bank_transaction.booking_date > datetime.datetime(2026, 5, 15)
        )
    )
    assert verbalize_expression(query) == (
        "Find the maximum of a BankTransaction whose booking_date is after May 15, 2026"
    )


def test_top_level_aggregation_average_between_whose():
    """A top-level average selection folds a >=/<= subject-attribute pair into 'whose … between'."""
    bank_transaction = variable(BankTransaction, domain=None)
    query = an(
        entity(eql.average(bank_transaction.amount_details.amount)).where(
            bank_transaction.booking_date >= datetime.datetime(2026, 5, 15),
            bank_transaction.booking_date <= datetime.datetime(2026, 5, 27),
        )
    )
    assert verbalize_expression(query) == (
        "Find the average of the amount of the amount_details of a BankTransaction "
        "whose booking_date is between May 15, 2026 and May 27, 2026"
    )


def test_top_level_aggregation_mixed_groupable_and_residual():
    """For an aggregation subject, single-hop predicates fold to 'whose'; multi-hop stay 'such that'."""
    bank_transaction = variable(BankTransaction, domain=None)
    query = an(
        entity(eql.average(bank_transaction.amount_details.amount)).where(
            bank_transaction.booking_date > datetime.datetime(2026, 5, 15),
            bank_transaction.amount_details.amount > 100,
        )
    )
    text = verbalize_expression(query)
    assert "whose booking_date is after May 15, 2026" in text, f"Got: {text!r}"
    assert "such that" in text, f"Got: {text!r}"
    assert "is greater than 100" in text, f"Got: {text!r}"


def test_second_domain_top_level_aggregation_whose():
    """Aggregation whose-grouping generalises to a second domain (Employee)."""
    employee = variable(Employee, domain=None)
    query = an(
        entity(eql.average(employee.salary)).where(employee.starting_salary > 20000)
    )
    assert verbalize_expression(query) == (
        "Find the average of salaries of Employees whose starting_salary is greater than 20000"
    )


def test_calc_equality_uses_is_equal_to():
    """== against a calculation reads 'is equal to'."""
    bank_transaction = variable(BankTransaction, domain=None)
    bt2 = variable(BankTransaction, domain=None)
    query = an(
        entity(bank_transaction).where(
            bank_transaction.amount_details.amount
            == an(entity(eql.average(bt2.amount_details.amount)))
        )
    )
    assert "is equal to the average of amounts" in verbalize_expression(query)


def test_object_equality_keeps_is():
    """== against a plain value keeps the bare 'is'."""
    bank_transaction = variable(BankTransaction, domain=None)
    query = an(
        entity(bank_transaction).where(bank_transaction.amount_details.amount == 3.5)
    )
    text = verbalize_expression(query)
    assert "the amount of its amount_details is 3.5" in text, f"Got: {text!r}"
    assert "is equal to" not in text, f"Got: {text!r}"


def test_not_calc_equality_uses_is_not_equal_to():
    """not(== calculation) reads 'is not equal to'."""
    bank_transaction = variable(BankTransaction, domain=None)
    bt2 = variable(BankTransaction, domain=None)
    query = an(
        entity(bank_transaction).where(
            not_(
                bank_transaction.amount_details.amount
                == an(entity(eql.max(bt2.amount_details.amount)))
            )
        )
    )
    assert "is not equal to the maximum amount" in verbalize_expression(query)


def test_range_fold_without_subject_has_no_whose():
    """A bare AND of bound comparisons folds to 'between' with the full chain (no subject → no 'whose')."""
    bank_transaction = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 5, 15)
    upper_bound = datetime.datetime(2026, 5, 30)
    text = verbalize_expression(
        and_(
            bank_transaction.booking_date >= lower_bound,
            bank_transaction.booking_date <= upper_bound,
        )
    )
    assert "is between May 15, 2026 and May 30, 2026" in text, f"Got: {text!r}"
    assert "whose" not in text, f"Got: {text!r}"
    assert "booking_date of" in text, f"Expected full chain in: {text!r}"


# ── Generality: a second domain (Employee) ──────────────────────────────────────


def test_second_domain_whose_single_predicate():
    employee = variable(Employee, domain=None)
    query = an(entity(employee).where(employee.salary > 50000))
    assert (
        verbalize_expression(query)
        == "Find an Employee whose salary is greater than 50000"
    )


def test_second_domain_whose_between():
    employee = variable(Employee, domain=None)
    query = an(
        entity(employee).where(employee.salary >= 30000, employee.salary <= 60000)
    )
    assert verbalize_expression(query) == (
        "Find an Employee whose salary is between 30000 and 60000"
    )


def test_second_domain_calc_equality_in_whose():
    employee = variable(Employee, domain=None)
    employee2 = variable(Employee, domain=None)
    query = an(
        entity(employee).where(
            employee.salary == an(entity(eql.average(employee2.salary)))
        )
    )
    text = verbalize_expression(query)
    assert "whose salary is equal to" in text, f"Got: {text!r}"
    assert "average of salaries" in text, f"Got: {text!r}"


# ── Declarative-rule unit tests ─────────────────────────────────────────────────


def test_is_calculation_value_predicate():
    from krrood.entity_query_language.query.aggregation_structure import (
        is_calculation_value,
    )

    bank_transaction = variable(BankTransaction, domain=None)
    assert is_calculation_value(eql.max(bank_transaction.amount_details.amount)) is True
    assert (
        is_calculation_value(
            an(entity(eql.sum(bank_transaction.amount_details.amount)))
        )
        is True
    )
    assert is_calculation_value(variable(BankTransaction, domain=None)) is False


def test_fold_range_pairs_is_position_independent():
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        fold_range_pairs,
        has_pair,
        RangeFold,
    )

    bank_transaction = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 5, 15)
    upper_bound = datetime.datetime(2026, 5, 30)
    lower = bank_transaction.booking_date >= lower_bound
    upper = bank_transaction.booking_date <= upper_bound

    folded = fold_range_pairs([lower, upper])
    assert len(folded) == 1 and isinstance(folded[0], RangeFold)
    assert has_pair([lower, upper]) is True

    # Upper bound written first: lower_bound/upper_bound are still assigned by direction, not position.
    folded_rev = fold_range_pairs([upper, lower])
    assert isinstance(folded_rev[0], RangeFold)
    assert folded_rev[0].lower_expression is lower.right
    assert folded_rev[0].upper_expression is upper.right


def test_fold_range_pairs_ignores_unrelated_bounds():
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        has_pair,
    )

    bank_transaction = variable(BankTransaction, domain=None)
    lower_bound = datetime.datetime(2026, 5, 15)
    # Two lower bounds on different chains do not form a range.
    assert (
        has_pair(
            [
                bank_transaction.booking_date >= lower_bound,
                bank_transaction.amount_details.amount > 5,
            ]
        )
        is False
    )


def test_pr_example():

    @dataclass(unsafe_hash=True)
    class BankTransaction:
        amount: float
        booking_date: datetime.datetime

    bt = variable(BankTransaction, domain=None)
    query = an(
        entity(bt).where(
            bt.amount > 1000,
            bt.booking_date >= datetime.datetime(2026, 5, 1),
            bt.booking_date <= datetime.datetime(2026, 5, 30),
        )
    )

    assert verbalize_expression(query) == (
        "Find a BankTransaction whose amount is greater than 1000, and whose booking_date is between May 1, 2026"
        " and May 30, 2026"
    )


# ── verbalize_expression renderer argument tests ──────────────────────────────


def test_verbalize_expression_no_renderer_backward_compatible():
    """Calling verbalize_expression matches pipeline plain output."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    plain = VerbalizationPipeline.plain().verbalize(q)
    assert verbalize_expression(q) == plain


def test_verbalize_expression_explicit_plain_paragraph():
    """Explicit pipeline verbalize produces same output as verbalize_expression."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    result = VerbalizationPipeline.plain().verbalize(q)
    assert "Find" in result
    assert "Robot" in result
    assert "\033" not in result
    assert "<span" not in result


def test_verbalize_expression_ansi_paragraph():
    """ANSI formatter + paragraph renderer produces escape codes, no added newlines."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    result = VerbalizationPipeline(ParagraphRenderer(ANSIFormatter())).verbalize(q)
    assert "\033[" in result
    assert "Robot" in result


def test_verbalize_expression_ansi_hierarchical():
    """ANSI formatter + hierarchical renderer produces escape codes and newlines."""
    r = variable(_Robot, [])
    q = an(entity(r).where(or_(r.battery > 50, r.battery < 10)))
    result = VerbalizationPipeline(HierarchicalRenderer(ANSIFormatter())).verbalize(q)
    assert "\033[" in result
    assert "\n" in result


def test_verbalize_expression_html_paragraph():
    """HTML formatter + paragraph renderer produces spans and the dark wrapper div."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    result = VerbalizationPipeline(ParagraphRenderer(HTMLFormatter())).verbalize(q)
    assert "<span" in result
    # HTML pipeline wraps output in a dark <div> for Jupyter rendering.
    assert "<div" in result


def test_verbalize_expression_html_hierarchical():
    """HTML formatter + hierarchical renderer produces spans and br tags."""
    r = variable(_Robot, [])
    q = an(entity(r).where(or_(r.battery > 50, r.battery < 10)))
    result = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(q)
    assert "<span" in result
    assert "<br>" in result


@pytest.mark.parametrize(
    "renderer_factory",
    [
        lambda: ParagraphRenderer(PlainFormatter()),
        lambda: ParagraphRenderer(ANSIFormatter()),
        lambda: ParagraphRenderer(HTMLFormatter()),
        lambda: HierarchicalRenderer(PlainFormatter()),
        lambda: HierarchicalRenderer(ANSIFormatter()),
        lambda: HierarchicalRenderer(HTMLFormatter()),
    ],
)
def test_verbalize_expression_all_combos_produce_non_empty(renderer_factory):
    """Every formatter × renderer combination produces a non-empty string."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    result = VerbalizationPipeline(renderer_factory()).verbalize(q)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Robot" in result


def test_verbalize_expression_equivalence_to_pipeline():
    """Every renderer in VerbalizationPipeline produces a non-empty string."""
    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))

    for renderer in [
        ParagraphRenderer(PlainFormatter()),
        ParagraphRenderer(ANSIFormatter()),
        ParagraphRenderer(HTMLFormatter()),
        HierarchicalRenderer(PlainFormatter()),
        HierarchicalRenderer(ANSIFormatter()),
        HierarchicalRenderer(HTMLFormatter()),
    ]:
        result = VerbalizationPipeline(renderer).verbalize(q)
        assert isinstance(result, str) and len(result) > 0, (
            f"Empty result for {type(renderer).__name__} / "
            f"{type(renderer.formatter).__name__}"
        )
        assert "Robot" in result


def test_verbalize_expression_html_hierarchical_handles_constrained_aggregation(
    departments_and_employees_fixture,
):
    """A constrained aggregation query works end-to-end with HTML hierarchical output."""
    _, _ = departments_and_employees_fixture
    employee = variable(Employee, domain=None)
    avg_salary = eql.average(employee.salary)
    query = a(
        set_of(employee.department, avg_salary)
        .grouped_by(employee.department)
        .having(avg_salary > 30000)
    )
    result = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(
        query
    )
    # The grouped report fronts its HAVING onto the group key, so it is a single prose line (no
    # block <br>); the HTML wrapper and coloured content still render end-to-end.
    assert "<span" in result
    assert "Employee" in result
    assert "average" in result
