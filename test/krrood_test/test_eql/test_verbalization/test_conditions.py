"""
Unit tests for the condition component (``grammar/conditions/``): the shared recognizers
and the comparator predicate surface form.

The recognizers are pure structural predicates (no rendering); the predicate form is
exercised end-to-end through the verbalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    set_of,
    variable,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    attribute_names,
    is_atomic_value,
    is_boolean_attribute_chain,
    is_concrete_object_literal,
    is_none_literal,
    references,
    relational_verb_phrase,
    single_hop_attribute,
)
from krrood.entity_query_language.verbalization.morphology import is_past_participle
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


@dataclass
class _Task:
    completed: bool


@dataclass
class _Robot:
    battery: int
    tasks: List[_Task]


def test_single_hop_attr_recognizes_subject_attribute():
    r = variable(_Robot, [])
    attr = single_hop_attribute(r.battery, r)
    assert attr is not None and attr._attribute_name_ == "battery"


def test_single_hop_attr_rejects_multi_hop_and_other_subject():
    r = variable(_Robot, [])
    other = variable(_Robot, [])
    assert single_hop_attribute(r.tasks[0].completed, r) is None  # multi-hop
    assert single_hop_attribute(r.battery, other) is None  # different subject


def test_is_concrete_object_literal_recognizer():
    """A dataclass-instance literal is a concrete object; a primitive / class / ``None`` is not."""

    @dataclass
    class _Obj:
        x: int

    assert is_concrete_object_literal(Literal(_value_=_Obj(1))) is True
    assert is_concrete_object_literal(Literal(_value_=5)) is False
    assert (
        is_concrete_object_literal(Literal(_value_=_Obj)) is False
    )  # the class itself
    assert is_concrete_object_literal(Literal(_value_=None)) is False


def test_is_atomic_value_recognizer():
    """A plain scalar literal is atomic (groupable); a concrete object / ``None`` is not."""

    @dataclass
    class _Obj:
        x: int

    assert is_atomic_value(Literal(_value_=5)) is True
    assert is_atomic_value(Literal(_value_="x")) is True
    assert is_atomic_value(Literal(_value_=_Obj(1))) is False
    assert is_atomic_value(Literal(_value_=None)) is False


def test_is_boolean_attribute_chain_only_for_boolean_terminal():
    r = variable(_Robot, [])
    assert is_boolean_attribute_chain(r.tasks[0].completed) is True
    assert is_boolean_attribute_chain(r.battery) is False


def test_attribute_names_walks_the_chain():
    r = variable(_Robot, [])
    assert attribute_names(r.battery) == ["battery"]


def test_references_detects_the_subject():
    r = variable(_Robot, [])
    assert references(r.battery, r) is True


def test_predicate_form_end_to_end():
    r = variable(_Robot, [])
    text = verbalize_expression(r.battery > 50)
    assert "battery" in text
    assert "greater than" in text
    assert "50" in text


# ── Absence: ``== None`` as a predicate, not a value ─────────────────────────


@dataclass
class _Robot2:
    pass


@dataclass
class _Address:
    pass


@dataclass
class _Author:
    pass


@dataclass
class _Mission:
    assigned_to: _Robot2  # past participle + preposition → relational


@dataclass
class _Parcel:
    shipped_to: _Address  # regular participle
    written_by: _Author  # irregular participle ("written", not "*writed")


@dataclass
class _Gadget:
    color_in: str  # noun + preposition → NOT a relation


@dataclass
class _Note:
    located_in: str  # relational, but the related type is a primitive → "anything"


def test_is_none_literal_recognizer():
    r = variable(_Robot, [])
    assert is_none_literal((r.battery == None).right) is True
    assert is_none_literal((r.battery == 5).right) is False


def test_is_past_participle_morphology():
    """The participle check is morphological (dictionary + rules), not 'ends in -ed'."""
    assert is_past_participle("assigned")  # regular
    assert is_past_participle("sent") and is_past_participle("written")  # irregular
    assert not is_past_participle("assign")  # base form
    assert not is_past_participle("color")  # noun that is also a base-form verb
    assert not is_past_participle("battery")  # plain noun
    assert not is_past_participle("sang")  # past tense ≠ participle ("sung")


def test_relational_verb_phrase_recognizer():
    """A relational field is *participle + preposition*; a noun ending in a preposition is not."""
    assert relational_verb_phrase("assigned_to") == "assigned to"
    assert relational_verb_phrase("written_by") == "written by"  # irregular participle
    assert relational_verb_phrase("cross_referenced_with") == "cross referenced with"
    assert relational_verb_phrase("color_in") is None  # 'color' is not a participle
    assert relational_verb_phrase("number_of") is None  # 'of' excluded (genitive)
    assert (
        relational_verb_phrase("depends_on") is None
    )  # present tense, not a participle
    assert relational_verb_phrase("orientation") is None  # no preposition at all


def test_noun_attribute_eq_none_reads_as_has_no():
    """A plain *noun* attribute ``== None`` reads *"<owner> has no <attribute>"*, not a value."""
    r = variable(_Robot, [])
    text = verbalize_expression(r.battery == None)
    assert "has no battery" in text
    assert "None" not in text and "nothing" not in text


def test_relational_attribute_eq_none_reads_as_passive():
    """A relational (participle + preposition) attribute ``== None`` reads as a passive verb naming
    the related type — *"<owner> has not been <verb> <prep> any <Type>"* — not *"has no <field>"*.
    """
    m = variable(_Mission, [])
    text = verbalize_expression(m.assigned_to == None)
    assert "has not been assigned to any _Robot2" in text
    assert "has no assigned" not in text and "assigned_to" not in text


def test_relational_absence_handles_irregular_participle():
    """An irregular participle (*written*) is recognised, and the related type is the field's type."""
    p = variable(_Parcel, [])
    assert "has not been shipped to any _Address" in verbalize_expression(
        p.shipped_to == None
    )
    assert "has not been written by any _Author" in verbalize_expression(
        p.written_by == None
    )


def test_noun_with_preposition_suffix_is_not_passivised():
    """A noun that merely ends in a preposition (*color_in*) keeps *"has no …"* — its leading token
    is not a participle."""
    g = variable(_Gadget, [])
    text = verbalize_expression(g.color_in == None)
    assert "has no color_in" in text
    assert "has not been" not in text


def test_relational_absence_with_primitive_type_says_anything():
    """When the related type is not a nameable class (here a primitive), the object is *"anything"*."""
    n = variable(_Note, [])
    text = verbalize_expression(n.located_in == None)
    assert "has not been located in anything" in text


def test_bare_variable_eq_none_reads_as_does_not_exist():
    """A bare variable ``== None`` (no attribute to name) reads *"<subject> does not exist"*."""
    r = variable(_Robot, [])
    text = verbalize_expression(r == None)
    assert "does not exist" in text


def test_subject_where_relational_eq_none_is_standalone_not_whose():
    """A subject ``where … == None`` is said as its own clause, never folded into *"whose"*."""
    m = variable(_Mission, [])
    text = verbalize_expression(an(entity(m).where(m.assigned_to == None)))
    assert "has not been assigned to any _Robot2" in text
    assert "whose" not in text


# ── Domain-constrained value variables: "one of …" in value position ─────────


def test_primitive_domain_variable_lists_in_value_position():
    """A primitive value-type variable in value position lists its candidates; a subject does not."""
    r = variable(_Robot, [])
    text = verbalize_expression(r.battery == variable(int, [1, 2, 3]))
    assert "one of 1, 2, or 3" in text
    # The same variable as a bare subject keeps its type-name noun.
    assert verbalize_expression(variable(int, [1, 2])) == "an int"


def test_entity_domain_variable_not_listed():
    """An entity-type variable's domain is the population and is never listed."""
    assert verbalize_expression(variable(_Robot, [])) == "a _Robot"


def test_two_value_domain_drops_the_serial_comma():
    """A two-candidate domain reads *"one of X or Y"* (no comma); three keep the serial comma."""
    r = variable(_Robot, [])
    assert "one of 1 or 2" in verbalize_expression(r.battery == variable(int, [1, 2]))
    assert "one of 1, 2, or 3" in verbalize_expression(
        r.battery == variable(int, [1, 2, 3])
    )


# ── Boolean attribute compared to a boolean: polarity, never "is True" ───────


@dataclass
class _Coffee:
    decaf: bool


def test_boolean_attribute_eq_true_is_predicative():
    """``bool_attr == True`` folds into the predicate — *"is decaf"*, not *"is decaf is True"*."""
    c = variable(_Coffee, [])
    text = verbalize_expression(c.decaf == True)
    assert "is decaf" in text
    assert "True" not in text
    assert "is decaf is" not in text  # no double copula


def test_boolean_attribute_eq_false_is_negated_predicative():
    """``bool_attr == False`` (and ``!= True``) read *"is not decaf"*."""
    c = variable(_Coffee, [])
    assert "is not decaf" in verbalize_expression(c.decaf == False)
    assert "is not decaf" in verbalize_expression(c.decaf != True)
    assert "False" not in verbalize_expression(c.decaf == False)


def test_boolean_attribute_open_domain_is_either_or_not():
    """A bool attribute constrained to a domain holding both values leaves the value open."""
    c = variable(_Coffee, [])
    text = verbalize_expression(c.decaf == variable(bool, [True, False]))
    assert "is either decaf or not" in text
    assert "True" not in text and "False" not in text


# ── Co-indexed comparator factoring ──────────────────────────────────────────


@dataclass
class _Date:
    month: int
    year: int
    day: int


@dataclass
class _Period:
    begin: _Date
    end: _Date


@dataclass
class _Inner:
    z: _Date


@dataclass
class _Outer:
    x: _Date
    y: _Inner


@dataclass
class _Statement:
    period: _Period
    revenue: float


def test_coindexed_equality_factors_to_have_the_same():
    """The motivating example: two ``begin.X == end.X`` conditions fold to one natural clause."""
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month, eql.average(p.revenue))
        .where(
            p.period.begin.month == p.period.end.month,
            p.period.begin.year == p.period.end.year,
        )
        .grouped_by(p.period.begin.month)
    )
    text = verbalize_expression(query)
    # The two prefixes coordinate and the shared period factors out (pronominalised by the
    # coreference pass — "its"/"their" depending on the subject's number).
    assert "the begin and end of" in text
    assert "period have the same month and year" in text
    # The shared structure is said once — no repeated per-attribute equality.
    assert "is the month of the end" not in text


def test_coindexed_equality_coordinates_three_terminals():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(
            p.period.begin.month == p.period.end.month,
            p.period.begin.year == p.period.end.year,
            p.period.begin.day == p.period.end.day,
        )
        .grouped_by(p.period.begin.month)
    )
    assert "have the same month, year, and day" in verbalize_expression(query)


def test_coindexed_non_equality_uses_faithful_those_of_form():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(
            p.period.begin.month > p.period.end.month,
            p.period.begin.year > p.period.end.year,
        )
        .grouped_by(p.period.begin.month)
    )
    text = verbalize_expression(query)
    assert (
        "the month and year of the begin of its period are greater than "
        "those of the end of its period" in text
    )


def test_coindexed_partial_group_keeps_unrelated_condition_in_order():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(
            p.period.begin.month == p.period.end.month,
            p.period.begin.year == p.period.end.year,
            p.revenue > 0,
        )
        .grouped_by(p.period.begin.month)
    )
    text = verbalize_expression(query)
    factored = text.index("have the same month and year")
    unrelated = text.index("revenue")
    assert factored < unrelated  # the factored clause precedes the unrelated one


def test_coindexed_does_not_fold_inequality_operator():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(
            p.period.begin.month != p.period.end.month,
            p.period.begin.year != p.period.end.year,
        )
        .grouped_by(p.period.begin.month)
    )
    text = verbalize_expression(query)
    assert "have the same" not in text
    assert "is not the month of the end" in text


def test_coindexed_single_condition_is_not_factored():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(p.period.begin.month == p.period.end.month)
        .grouped_by(p.period.begin.month)
    )
    text = verbalize_expression(query)
    assert "have the same" not in text
    assert "is the month of the end of its period" in text


def test_coindexed_mixed_operators_are_not_folded_together():
    p = variable(_Statement, domain=None)
    query = a(
        set_of(p.period.begin.month)
        .where(
            p.period.begin.month == p.period.end.month,
            p.period.begin.year > p.period.end.year,
        )
        .grouped_by(p.period.begin.month)
    )
    assert "have the same" not in verbalize_expression(query)


def test_coindexed_non_sibling_prefixes_use_faithful_form():
    o = variable(_Outer, domain=None)
    query = a(
        set_of(o.x.month)
        .where(o.x.month == o.y.z.month, o.x.year == o.y.z.year)
        .grouped_by(o.x.month)
    )
    text = verbalize_expression(query)
    assert "are equal to those of" in text
    assert "have the same" not in text


def test_coindexed_factoring_in_subject_whose_path():
    p = variable(_Period, domain=None)
    query = an(
        entity(p).where(
            p.begin.month == p.end.month,
            p.begin.year == p.end.year,
        )
    )
    assert "the begin and end of the _Period have the same month and year" in (
        verbalize_expression(query)
    )
