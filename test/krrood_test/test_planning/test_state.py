"""Tests for evaluating queries with a :class:`~krrood.planning.state.State`.

A query reads the values the state's facts give ground terms - attributes, indexed elements,
calls, symbolic functions and predicates - and live values for every other ground term.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import (
    a,
    flat_variable,
    literal,
    not_,
    the,
    variable,
)
from krrood.entity_query_language.predicate import symbolic_function
from krrood.planning.exceptions import (
    AssignmentBypassesSetter,
    ComparisonIsNotAnEquality,
    ConditionIsNotAFact,
    FactValueIsNotConstant,
    MissingGroundTerm,
)
from krrood.planning.state import State

from ..dataset.setting_with_side_effects import (
    AttributeReadThroughProperty,
    AttributeSetThroughProperty,
    AttributesSetThroughSetattr,
    ElementsSetThroughSetitem,
    FrozenAttributes,
    SlottedAttributes,
)
from ..dataset.blocks_world import (
    Block,
    IsOn,
    Shelf,
    SupportName,
    Table,
    is_table,
    tallest_tower,
    tower_height,
)

# %% fixtures


@dataclass
class TwoTowers:
    """A blocksworld with block ``c`` on block ``a``, blocks ``a`` and ``b`` on
    the table, and a shelf holding ``a`` and ``b`` in its slots."""

    table: Table
    """
    The table both towers stand on.
    """

    a: Block
    """The bottom block of the first tower."""

    b: Block
    """
    The only block of the second tower.
    """

    c: Block
    """The top block of the first tower."""

    shelf: Shelf
    """
    The shelf whose slots hold ``a`` and ``b``.
    """

    @property
    def blocks(self) -> list[Block]:
        """
        :return: Every block of the world.
        """
        return [self.a, self.b, self.c]

    @property
    def b_on_c(self) -> State:
        """
        :return: The state that puts ``b`` onto ``c``.
        """
        return State().with_facts(literal(self.b).support == self.c)


@pytest.fixture
def two_towers() -> TwoTowers:
    """
    :return: A fresh :class:`TwoTowers` world for each test.
    """
    table = Table("table")
    a = Block("a", table)
    b = Block("b", table)
    return TwoTowers(
        table=table, a=a, b=b, c=Block("c", a), shelf=Shelf("shelf", [a, b])
    )


def blocks_on(
    blocks: list[Block], support: Block | Table, state: State | None = None
) -> list[Block]:
    """
    :param blocks: The blocks to query over.
    :param support: The block or table the selected blocks rest on.
    :param state: The state to evaluate in, or ``None`` for the live values.
    :return: The blocks resting on *support*.
    """
    return a(Block)(support=support).from_(blocks).tolist(state=state)


# %% reading assigned and unassigned attributes


def test_query_under_state_reads_assigned_value(two_towers):
    result = blocks_on(two_towers.blocks, two_towers.c, two_towers.b_on_c)

    assert result == [two_towers.b]


def test_unassigned_ground_terms_keep_their_live_values(two_towers):
    result = blocks_on(two_towers.blocks, two_towers.table, two_towers.b_on_c)

    assert result == [two_towers.a]


def test_state_never_changes_live_objects(two_towers):
    blocks_on(two_towers.blocks, two_towers.c, two_towers.b_on_c)

    assert two_towers.b.support is two_towers.table


def test_query_outside_state_reads_live_values(two_towers):
    two_towers.b_on_c

    assert blocks_on(two_towers.blocks, two_towers.c) == []


# %% ground terms of every kind


def test_query_reads_assigned_indexed_element(two_towers):
    state = State().with_facts(literal(two_towers.shelf).slots[0] == two_towers.c)

    result = a(Shelf).from_([two_towers.shelf]).slots[0].tolist(state=state)

    assert result == [two_towers.c]


def test_index_by_expression_reads_the_same_ground_term_as_index_by_value(two_towers):
    state = State().with_facts(literal(two_towers.shelf).slots[0] == two_towers.c)
    slot = variable(int, [0, 1])

    result = a(Shelf).from_([two_towers.shelf]).slots[slot].tolist(state=state)

    assert result == [two_towers.c, two_towers.b]


def test_query_reads_assigned_method_call(two_towers):
    state = State().with_facts(
        literal(two_towers.a).rests_on(two_towers.table) == False
    )
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(block.rests_on(two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_symbolic_function(two_towers):
    state = State().with_facts(tower_height(literal(two_towers.b)) == 3)
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(tower_height(block) == 3)
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_predicate(two_towers):
    state = State().with_facts(
        not_(IsOn(literal(two_towers.a), literal(two_towers.table)))
    )
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(IsOn(block, two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_symbolic_function_class(two_towers):
    state = State().with_facts(SupportName(literal(two_towers.b)) == "a")
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(SupportName(block) == "a")
        .tolist(state=state)
    )

    assert result == [two_towers.b, two_towers.c]


# %% facts about every binding of a variable


def test_fact_about_a_variable_assigns_the_ground_term_of_each_of_its_values(
    two_towers,
):
    state = State().with_facts(
        variable(Block, [two_towers.a, two_towers.b]).support == two_towers.c
    )

    result = blocks_on(two_towers.blocks, two_towers.c, state)

    assert result == [two_towers.a, two_towers.b]


def test_fact_about_a_match_assigns_only_the_ground_terms_of_its_matches(two_towers):
    state = State().with_facts(
        a(Block)(support=two_towers.table).from_(two_towers.blocks).support
        == two_towers.c
    )

    result = blocks_on(two_towers.blocks, two_towers.c, state)

    assert result == [two_towers.a, two_towers.b]


def test_fact_with_an_expression_index_assigns_every_element_it_reaches(two_towers):
    state = State().with_facts(
        literal(two_towers.shelf).slots[variable(int, [0, 1])] == two_towers.c
    )

    slot = variable(int, [0, 1])

    result = a(Shelf).from_([two_towers.shelf]).slots[slot].tolist(state=state)

    assert result == [two_towers.c, two_towers.c]


# %% conditions that are not facts


def test_assigning_an_attribute_the_instance_lacks_is_rejected(two_towers):
    with pytest.raises(MissingGroundTerm):
        State().with_facts(literal(two_towers.table).support == two_towers.a)


def test_assigning_an_element_the_container_lacks_is_rejected(two_towers):
    with pytest.raises(MissingGroundTerm):
        State().with_facts(literal(two_towers.shelf).slots[5] == two_towers.c)


@pytest.mark.parametrize(
    "ground_term",
    [
        lambda: literal(AttributeSetThroughProperty()).is_lit,
        lambda: literal(AttributesSetThroughSetattr()).status,
        lambda: literal(ElementsSetThroughSetitem(lit=False))["lit"],
    ],
    ids=["property setter", "__setattr__", "__setitem__"],
)
def test_assigning_a_ground_term_whose_setting_runs_code_is_rejected(ground_term):
    with pytest.raises(AssignmentBypassesSetter):
        State().with_facts(ground_term() == True)


@pytest.mark.parametrize(
    "ground_term",
    [
        lambda: literal(SlottedAttributes(1.0)).coordinate,
        lambda: literal(FrozenAttributes(1.0)).coordinate,
        lambda: literal(AttributeReadThroughProperty(1.0)).diameter,
    ],
    ids=["slot", "frozen dataclass", "read-only property"],
)
def test_assigning_a_ground_term_that_is_only_stored_is_accepted(ground_term):
    expression = ground_term()
    state = State().with_facts(expression == 5.0)

    assert expression.tolist(state=state) == [5.0]


def test_a_comparison_other_than_equality_is_not_a_fact(two_towers):
    with pytest.raises(ComparisonIsNotAnEquality):
        State().with_facts(tower_height(literal(two_towers.b)) > 3)


def test_negating_a_fact_that_is_not_a_predicate_is_not_a_fact(two_towers):
    with pytest.raises(ConditionIsNotAFact):
        State().with_facts(not_(literal(two_towers.b).support == two_towers.c))


def test_a_fact_cannot_take_its_value_from_an_expression(two_towers):
    with pytest.raises(FactValueIsNotConstant):
        State().with_facts(
            literal(two_towers.b).support == literal(two_towers.a).support
        )


# %% one ground term, written in several ways


def test_call_names_one_ground_term_however_its_arguments_are_passed(two_towers):
    block = literal(two_towers.a)
    positional = State().with_facts(block.rests_on(two_towers.table) == False)
    by_name = State().with_facts(block.rests_on(support=two_towers.table) == False)
    with_default = State().with_facts(
        block.rests_on(two_towers.table, directly=True) == False
    )

    assert positional == by_name == with_default


def test_different_arguments_name_different_ground_terms(two_towers):
    block = literal(two_towers.a)

    assert State().with_facts(
        block.rests_on(two_towers.table) == False
    ) != State().with_facts(block.rests_on(two_towers.table, directly=False) == False)


def test_assignment_by_keyword_is_read_by_a_positional_call(two_towers):
    state = State().with_facts(
        literal(two_towers.a).rests_on(support=two_towers.table) == False
    )
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(block.rests_on(two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_ground_term_reached_through_another_object_is_the_same_ground_term(two_towers):
    direct = State().with_facts(
        literal(two_towers.a).rests_on(two_towers.table) == False
    )
    through_literal = State().with_facts(
        literal(two_towers).a.rests_on(two_towers.table) == False
    )
    through_match = State().with_facts(
        the(TwoTowers).from_([two_towers]).a.rests_on(two_towers.table) == False
    )

    assert direct == through_literal == through_match


# %% how arguments are compared


def test_ground_terms_identify_unhashable_instances_by_identity(two_towers):
    """Two blocks that compare equal are still different objects, so assigning
    a ground term of one never assigns the other."""
    first = Block("d", two_towers.table)
    second = Block("d", two_towers.table)
    state = State().with_facts(literal(first).support == two_towers.c)

    result = blocks_on([first, second], two_towers.table, state)

    assert result == [second]


def test_ground_terms_identify_list_arguments_by_identity(two_towers):
    """Two lists with the same elements are still different objects, so
    assigning a ground term called with one never assigns the ground term
    called with the other."""
    first = [two_towers.b]
    second = [two_towers.b]
    state = State().with_facts(tallest_tower(literal(first)) == 7)

    heights = [
        tallest_tower(literal(tops)).tolist(state=state) for tops in (first, second)
    ]

    assert heights == [[7], [tallest_tower(second)]]


def test_ground_terms_compare_hashable_arguments_by_value(two_towers):
    key = 1
    equal_key = int("1")
    slots = literal(two_towers.shelf).slots

    assert State().with_facts(slots[key] == two_towers.c) == State().with_facts(
        slots[equal_key] == two_towers.c
    )


# %% ground terms read through other ground terms


def test_attribute_of_an_assigned_attribute_reads_the_new_value(two_towers):
    result = (
        a(Block).from_([two_towers.b]).support.support.tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a]


def test_symbolic_function_receives_assigned_attribute_as_argument(two_towers):
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(is_table(block.support))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a]


def test_python_body_reads_live_values_it_does_not_receive_as_arguments(two_towers):
    """:class:`IsOn` reads ``self.block.support`` in Python, so it sees ``b``
    on the table although the state puts ``b`` on ``c``."""
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(IsOn(block, two_towers.c))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == []


def test_flattening_an_assigned_collection_ranges_over_its_new_elements(two_towers):
    state = State().with_facts(literal(two_towers.shelf).slots == [two_towers.c])

    result = flat_variable(a(Shelf).from_([two_towers.shelf]).slots).tolist(state=state)

    assert result == [two_towers.c]


def test_index_into_an_assigned_collection_reads_its_new_elements(two_towers):
    state = State().with_facts(literal(two_towers.shelf).slots == [two_towers.c])

    result = a(Shelf).from_([two_towers.shelf]).slots[0].tolist(state=state)

    assert result == [two_towers.c]


# %% assigned derived ground terms


def test_assigned_predicate_overrides_what_it_derives(two_towers):
    """An assigned ground term is not recomputed, even where it is derived from
    other ground terms:

    ``c`` rests on ``a``, yet the state says it rests on the table.
    """
    state = State().with_facts(IsOn(literal(two_towers.c), literal(two_towers.table)))
    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(IsOn(block, two_towers.table))
        .tolist(state=state)
    )

    assert result == two_towers.blocks


def test_assigned_call_is_never_computed(two_towers):
    computed_for = []

    @symbolic_function
    def height_counting_calls(block: Block) -> int:
        computed_for.append(block)
        return tower_height(block)

    state = State().with_facts(height_counting_calls(literal(two_towers.b)) == 5)
    (block := a(Block).from_(two_towers.blocks)).where(
        height_counting_calls(block) > 0
    ).tolist(state=state)

    assert two_towers.b not in computed_for


# %% evaluations under a state


def test_results_read_the_state_wherever_they_are_consumed(two_towers):
    results = (
        a(Block)(support=two_towers.c)
        .from_(two_towers.blocks)
        .evaluate(state=two_towers.b_on_c)
    )

    assert list(results) == [two_towers.b]


def test_state_reaches_only_the_evaluation_it_is_given_to(two_towers):
    """A query evaluated while an evaluation under a state is part way through
    reads the live values."""
    supports = (
        a(Block)
        .from_([two_towers.b, two_towers.c])
        .support.evaluate(state=two_towers.b_on_c)
    )

    first_support = next(supports)
    live_blocks_on_c = blocks_on(two_towers.blocks, two_towers.c)
    second_support = next(supports)

    assert (first_support, live_blocks_on_c, second_support) == (
        two_towers.c,
        [],
        two_towers.a,
    )


def test_query_started_during_another_evaluation_reads_state(two_towers):
    """A query started from inside a running evaluation, here by a symbolic
    function, reads the same state as the evaluation that started it."""

    @symbolic_function
    def is_covered(candidate: Block) -> bool:
        return blocks_on(two_towers.blocks, candidate) != []

    result = (
        (block := a(Block).from_(two_towers.blocks))
        .where(is_covered(block))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a, two_towers.c]


# %% building states


def test_with_facts_leaves_the_original_state_unchanged(two_towers):
    state = State()

    state.with_facts(literal(two_towers.b).support == two_towers.c)

    assert state.assignments == {}


def test_later_fact_replaces_an_earlier_one(two_towers):
    state = two_towers.b_on_c.with_facts(literal(two_towers.b).support == two_towers.a)

    assert state == State().with_facts(literal(two_towers.b).support == two_towers.a)


def test_later_fact_replaces_one_written_another_way(two_towers):
    block = literal(two_towers.a)
    state = (
        State()
        .with_facts(block.rests_on(two_towers.table) == False)
        .with_facts(block.rests_on(support=two_towers.table, directly=True) == True)
    )

    assert list(state.assignments.values()) == [True]


# %% comparing states


def test_states_with_the_same_facts_are_equal_and_hash_alike(two_towers):
    first = two_towers.b_on_c
    second = State().with_facts(literal(two_towers.b).support == two_towers.c)

    assert (first, hash(first)) == (second, hash(second))


def test_states_with_different_facts_differ(two_towers):
    assert two_towers.b_on_c != State().with_facts(
        literal(two_towers.b).support == two_towers.a
    )
