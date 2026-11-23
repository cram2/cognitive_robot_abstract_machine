"""
Tests for evaluating queries with a :class:`~krrood.planning.state.State`.

A query reads the values the state's facts give ground terms - attributes, indexed elements,
calls, symbolic functions and predicates - and live values for every other ground term.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy.orm import sessionmaker
from typing_extensions import Callable

from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    QueryBackend,
)
from krrood.ormatic.eql_interface import SQLAlchemyBackend
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.entity_query_language.exceptions import BackendCannotReadState
from krrood.entity_query_language.factories import (
    a,
    flat_variable,
    literal,
    not_,
    the,
    variable,
    variable_from,
)
from krrood.entity_query_language.predicate import symbolic_function
from krrood.planning.exceptions import (
    AssignedValueHasWrongType,
    AssignmentBypassesSetter,
    ComparisonIsNotAnEquality,
    ConditionIsNotAFact,
    MissingGroundTerm,
)
from krrood.planning.state import State

from ..dataset.setting_with_side_effects import (
    AttributeReadThroughProperty,
    AttributeSetThroughProperty,
    AttributesSetThroughSetattr,
    ElementsInAnImmutableContainer,
    ElementsSetThroughSetitem,
    FrozenAttributes,
    SlottedAttributes,
)
from ..dataset.objects_compared_by_value import PointComparedByValue
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


@dataclass(kw_only=True)
class TwoTowers:
    """
    A blocksworld with block ``c`` on block ``a``, blocks ``a`` and ``b`` on the table,
    and a shelf holding ``a`` and ``b`` in its slots.
    """

    table: Table
    """
    The table both towers stand on.
    """

    a: Block
    """
    The bottom block of the first tower.
    """

    b: Block
    """
    The only block of the second tower.
    """

    c: Block
    """
    The top block of the first tower.
    """

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
        return State().add_facts(literal(self.b).support == self.c)


@pytest.fixture
def two_towers() -> TwoTowers:
    """
    :return: A fresh :class:`TwoTowers` world for each test.
    """
    table = Table(name="table")
    a = Block(name="a", support=table)
    b = Block(name="b", support=table)
    return TwoTowers(
        table=table,
        a=a,
        b=b,
        c=Block(name="c", support=a),
        shelf=Shelf(name="shelf", slots=[a, b]),
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
    return a(Block)(support=support).from_(domain=blocks).tolist(state=state)


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
    """
    An element a fact assigns in a container is read in place of the live one.
    """
    state = State().add_facts(literal(two_towers.shelf).slots[0] == two_towers.c)

    result = a(Shelf).from_(domain=[two_towers.shelf]).slots[0].tolist(state=state)

    assert result == [two_towers.c]


def test_index_by_expression_reads_the_same_ground_term_as_index_by_value(two_towers):
    """
    An index computed by an expression names the element a constant index names.
    """
    state = State().add_facts(literal(two_towers.shelf).slots[0] == two_towers.c)
    slot = variable(int, [0, 1])

    result = a(Shelf).from_(domain=[two_towers.shelf]).slots[slot].tolist(state=state)

    assert result == [two_towers.c, two_towers.b]


def test_query_reads_assigned_method_call(two_towers):
    """
    A call a fact assigns returns the assigned value instead of running.
    """
    state = State().add_facts(
        literal(two_towers.a).rests_on(support=two_towers.table) == False
    )
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(block.rests_on(support=two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_symbolic_function(two_towers):
    """
    A symbolic function a fact assigns returns the assigned value.
    """
    state = State().add_facts(tower_height(literal(two_towers.b)) == 3)
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(tower_height(block) == 3)
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_predicate(two_towers):
    """
    A predicate stated as a fact holds, and one negated as a fact does not.
    """
    state = State().add_facts(
        not_(IsOn(block=literal(two_towers.a), support=literal(two_towers.table)))
    )
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(IsOn(block=block, support=two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_query_reads_assigned_symbolic_function_class(two_towers):
    """
    A symbolic function class a fact assigns returns the assigned value.
    """
    state = State().add_facts(SupportName(block=literal(two_towers.b)) == "a")
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(SupportName(block=block) == "a")
        .tolist(state=state)
    )

    assert result == [two_towers.b, two_towers.c]


# %% facts about every binding of a variable


def test_fact_about_a_variable_assigns_the_ground_term_of_each_of_its_values(
    two_towers,
):
    """
    A fact rooted at a variable states the value for every value of the variable.
    """
    state = State().add_facts(
        variable(Block, [two_towers.a, two_towers.b]).support == two_towers.c
    )

    result = blocks_on(two_towers.blocks, two_towers.c, state)

    assert result == [two_towers.a, two_towers.b]


def test_fact_about_a_match_assigns_only_the_ground_terms_of_its_matches(two_towers):
    """
    A fact rooted at a match leaves the objects the match excludes unassigned.
    """
    state = State().add_facts(
        a(Block)(support=two_towers.table).from_(domain=two_towers.blocks).support
        == two_towers.c
    )

    result = blocks_on(two_towers.blocks, two_towers.c, state)

    assert result == [two_towers.a, two_towers.b]


def test_fact_with_an_expression_index_assigns_every_element_it_reaches(two_towers):
    """
    Each index value the expression takes names an element the fact assigns.
    """
    state = State().add_facts(
        literal(two_towers.shelf).slots[variable(int, [0, 1])] == two_towers.c
    )

    slot = variable(int, [0, 1])

    result = a(Shelf).from_(domain=[two_towers.shelf]).slots[slot].tolist(state=state)

    assert result == [two_towers.c, two_towers.c]


# %% conditions that are not facts


def test_assigning_an_attribute_the_instance_lacks_is_rejected(two_towers):
    """
    A fact never adds an attribute to an instance.
    """
    with pytest.raises(MissingGroundTerm):
        State().add_facts(literal(two_towers.table).support == two_towers.a)


def test_assigning_an_element_the_container_lacks_is_rejected(two_towers):
    """
    A fact never adds a key to a container.
    """
    with pytest.raises(MissingGroundTerm):
        State().add_facts(literal(two_towers.shelf).slots[5] == two_towers.c)


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
    """
    A state never runs the code a setter would, so it refuses the ground term.
    """
    with pytest.raises(AssignmentBypassesSetter):
        State().add_facts(ground_term() == True)


@pytest.mark.parametrize(
    "ground_term",
    [
        lambda: literal(SlottedAttributes(coordinate=1.0)).coordinate,
        lambda: literal(FrozenAttributes(coordinate=1.0)).coordinate,
        lambda: literal(AttributeReadThroughProperty(radius=1.0)).diameter,
        lambda: literal(
            ElementsInAnImmutableContainer(end_points=(0.0, 1.0))
        ).end_points[0],
    ],
    ids=["slot", "frozen dataclass", "read-only property", "immutable container"],
)
def test_assigning_a_ground_term_whose_setting_runs_no_code_is_accepted(ground_term):
    """
    Slotted and frozen fields store their value without running code of the class.
    """
    expression = ground_term()
    state = State().add_facts(expression == 5.0)

    assert expression.tolist(state=state) == [5.0]


def test_a_comparison_other_than_equality_is_not_a_fact(two_towers):
    """
    An inequality constrains a value without fixing it.
    """
    with pytest.raises(ComparisonIsNotAnEquality):
        State().add_facts(tower_height(literal(two_towers.b)) > 3)


def test_negating_a_fact_that_is_not_a_predicate_is_not_a_fact(two_towers):
    """
    Only a predicate can be stated not to hold.
    """
    with pytest.raises(ConditionIsNotAFact):
        State().add_facts(not_(literal(two_towers.b).support == two_towers.c))


def test_assigning_a_value_of_the_wrong_type_is_rejected(two_towers):
    """
    A support is a block or a table, never a string.
    """
    with pytest.raises(AssignedValueHasWrongType):
        State().add_facts(literal(two_towers.b).support == "the floor")


# %% facts whose value is an expression


def test_fact_takes_its_value_from_an_expression(two_towers):
    """
    Block ``c`` is put on whatever ``b`` rests on.
    """
    state = State().add_facts(
        literal(two_towers.c).support == literal(two_towers.b).support
    )
    assert literal(two_towers.c).support.tolist(state=state) == [two_towers.table]


def test_fact_about_a_variable_takes_the_value_of_each_binding(two_towers):
    """
    Each block takes the name of its own support.
    """
    block = variable(Block, two_towers.blocks)
    state = State().add_facts(block.name == block.support.name)
    assert a(Block).from_(domain=two_towers.blocks).name.tolist(state=state) == [
        "table",
        "table",
        "a",
    ]


def test_fact_reads_its_value_in_the_state_it_extends(two_towers):
    """
    Block ``a`` is put on what ``b`` rests on in the state, ``c``, not on the live
    table.
    """
    state = two_towers.b_on_c.add_facts(
        literal(two_towers.a).support == literal(two_towers.b).support
    )
    assert literal(two_towers.a).support.tolist(state=state) == [two_towers.c]


# %% one ground term, written in several ways


def test_call_names_one_ground_term_however_its_arguments_are_passed(two_towers):
    """
    Positional, keyword and defaulted arguments name one call.
    """
    block = literal(two_towers.a)
    by_name = State().add_facts(block.rests_on(support=two_towers.table) == False)
    with_default = State().add_facts(
        block.rests_on(support=two_towers.table, directly=True) == False
    )

    assert by_name == with_default


def test_different_arguments_name_different_ground_terms(two_towers):
    """
    Assigning a call for one argument leaves the call for another live.
    """
    block = literal(two_towers.a)

    assert State().add_facts(
        block.rests_on(support=two_towers.table) == False
    ) != State().add_facts(
        block.rests_on(support=two_towers.table, directly=False) == False
    )


def test_assignment_by_keyword_is_read_by_a_positional_call(two_towers):
    """
    A call assigned with keyword arguments is read by one written positionally.
    """
    state = State().add_facts(
        literal(two_towers.a).rests_on(support=two_towers.table) == False
    )
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(block.rests_on(support=two_towers.table))
        .tolist(state=state)
    )

    assert result == [two_towers.b]


def test_ground_term_reached_through_another_object_is_the_same_ground_term(two_towers):
    """
    An attribute reached through another object's attribute is the attribute itself.
    """
    direct = State().add_facts(
        literal(two_towers.a).rests_on(support=two_towers.table) == False
    )
    through_literal = State().add_facts(
        literal(two_towers).a.rests_on(support=two_towers.table) == False
    )
    through_match = State().add_facts(
        the(TwoTowers).from_(domain=[two_towers]).a.rests_on(support=two_towers.table)
        == False
    )

    assert direct == through_literal == through_match


# %% how arguments are compared


def test_ground_terms_identify_unhashable_instances_by_identity(two_towers):
    """
    Two blocks that compare equal are still different objects, so assigning a ground
    term of one never assigns the other.
    """
    first = Block(name="d", support=two_towers.table)
    second = Block(name="d", support=two_towers.table)
    state = State().add_facts(literal(first).support == two_towers.c)

    result = blocks_on([first, second], two_towers.table, state)

    assert result == [second]


def test_ground_terms_identify_list_arguments_by_identity(two_towers):
    """
    Two lists with the same elements are still different objects, so assigning a ground
    term called with one never assigns the ground term called with the other.
    """
    first = [two_towers.b]
    second = [two_towers.b]
    state = State().add_facts(tallest_tower(literal(first)) == 7)

    heights = [
        tallest_tower(literal(tops)).tolist(state=state) for tops in (first, second)
    ]

    assert heights == [[7], [tallest_tower(second)]]


def test_ground_terms_compare_hashable_arguments_by_value(two_towers):
    """
    Equal hashable arguments name the same call.
    """
    key = 1
    equal_key = int("1")
    slots = literal(two_towers.shelf).slots

    assert State().add_facts(slots[key] == two_towers.c) == State().add_facts(
        slots[equal_key] == two_towers.c
    )


@pytest.mark.parametrize(
    "ground_term_of",
    [
        lambda point: literal(point).coordinate,
        lambda point: literal(point).shifted_by(distance=1.0),
    ],
    ids=["attribute", "method call"],
)
def test_equal_objects_never_share_their_ground_terms(ground_term_of):
    """
    The object a ground term belongs to is compared by identity, so assigning a ground
    term of one object never assigns that of an equal but distinct object.
    """
    first = PointComparedByValue(coordinate=1.0)
    second = PointComparedByValue(coordinate=1.0)
    state = State().add_facts(ground_term_of(first) == 7.0)

    assert ground_term_of(second).tolist(state=state) == [
        ground_term_of(second).tolist()[0]
    ]


def test_equal_containers_never_share_their_elements():
    """
    An element assigned in one container is not read from an equal one.
    """
    first = ElementsInAnImmutableContainer(end_points=tuple([0.0, 1.0]))
    second = ElementsInAnImmutableContainer(end_points=tuple([0.0, 1.0]))
    state = State().add_facts(literal(first).end_points[0] == 7.0)

    assert literal(second).end_points[0].tolist(state=state) == [second.end_points[0]]


# %% ground terms read through other ground terms


def test_attribute_of_an_assigned_attribute_reads_the_new_value(two_towers):
    result = (
        a(Block)
        .from_(domain=[two_towers.b])
        .support.support.tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a]


def test_symbolic_function_receives_assigned_attribute_as_argument(two_towers):
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(is_table(block.support))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a]


def test_python_body_reads_live_values_it_does_not_receive_as_arguments(two_towers):
    """
    :class:`IsOn` reads ``self.block.support`` in Python, so it sees ``b`` on the table
    although the state puts ``b`` on ``c``.
    """
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(IsOn(block=block, support=two_towers.c))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == []


def test_flattening_an_assigned_collection_ranges_over_its_new_elements(two_towers):
    """
    A variable over an assigned collection ranges over its assigned elements.
    """
    state = State().add_facts(literal(two_towers.shelf).slots == [two_towers.c])

    result = flat_variable(a(Shelf).from_(domain=[two_towers.shelf]).slots).tolist(
        state=state
    )

    assert result == [two_towers.c]


def test_index_into_an_assigned_collection_reads_its_new_elements(two_towers):
    """
    Indexing an assigned collection reads the assigned elements.
    """
    state = State().add_facts(literal(two_towers.shelf).slots == [two_towers.c])

    result = a(Shelf).from_(domain=[two_towers.shelf]).slots[0].tolist(state=state)

    assert result == [two_towers.c]


# %% assigned derived ground terms


def test_assigned_predicate_overrides_what_it_derives(two_towers):
    """An assigned ground term is not recomputed, even where it is derived from
    other ground terms:

    ``c`` rests on ``a``, yet the state says it rests on the table.
    """
    state = State().add_facts(
        IsOn(block=literal(two_towers.c), support=literal(two_towers.table))
    )
    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(IsOn(block=block, support=two_towers.table))
        .tolist(state=state)
    )

    assert result == two_towers.blocks


def test_assigned_call_is_never_computed(two_towers):
    """
    The body of an assigned call never runs.
    """
    computed_for = []

    @symbolic_function
    def height_counting_calls(block: Block) -> int:
        computed_for.append(block)
        return tower_height(block)

    state = State().add_facts(height_counting_calls(literal(two_towers.b)) == 5)
    (block := a(Block).from_(domain=two_towers.blocks)).where(
        height_counting_calls(block) > 0
    ).tolist(state=state)

    assert two_towers.b not in computed_for


# %% evaluations under a state


def test_results_read_the_state_wherever_they_are_consumed(two_towers):
    results = (
        a(Block)(support=two_towers.c)
        .from_(domain=two_towers.blocks)
        .evaluate(state=two_towers.b_on_c)
    )

    assert list(results) == [two_towers.b]


def test_state_reaches_only_the_evaluation_it_is_given_to(two_towers):
    """
    A query evaluated while an evaluation under a state is part way through reads the
    live values.
    """
    supports = (
        a(Block)
        .from_(domain=[two_towers.b, two_towers.c])
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
    """
    A query started from inside a running evaluation, here by a symbolic function, reads
    the same state as the evaluation that started it.
    """

    @symbolic_function
    def is_covered(candidate: Block) -> bool:
        return blocks_on(two_towers.blocks, candidate) != []

    result = (
        (block := a(Block).from_(domain=two_towers.blocks))
        .where(is_covered(block))
        .tolist(state=two_towers.b_on_c)
    )

    assert result == [two_towers.a, two_towers.c]


# %% backends and states


def test_generative_backend_reads_state(two_towers):
    """
    The generative backend evaluates queries under a state as the default backend does.
    """
    state = State().add_facts(literal(two_towers.c).support == two_towers.table)
    generated = a(Block)(name=variable_from(["d", "e"]), support=two_towers.table)
    generated.where(literal(two_towers.c).support == two_towers.table)

    result = generated.tolist(
        backend=EntityQueryLanguageGenerativeBackend(), state=state
    )

    assert [block.name for block in result] == ["d", "e"]


@pytest.mark.parametrize(
    "create_backend",
    [
        lambda: SQLAlchemyBackend(session_maker=sessionmaker()),
        lambda: ProbabilisticBackend(),
    ],
    ids=["sql", "probabilistic"],
)
def test_backend_that_cannot_read_a_state_refuses_one(
    two_towers, create_backend: Callable[[], QueryBackend]
):
    query = a(Block).from_(domain=two_towers.blocks)

    with pytest.raises(BackendCannotReadState):
        query.evaluate(backend=create_backend(), state=State())


# %% building states


def test_add_facts_leaves_the_original_state_unchanged(two_towers):
    """
    Adding facts returns a new state.
    """
    state = State()

    state.add_facts(literal(two_towers.b).support == two_towers.c)

    assert state.assignments == {}


def test_later_fact_replaces_an_earlier_one(two_towers):
    """
    Of two facts on one ground term, the later one holds.
    """
    state = two_towers.b_on_c.add_facts(literal(two_towers.b).support == two_towers.a)

    assert state == State().add_facts(literal(two_towers.b).support == two_towers.a)


def test_later_fact_replaces_one_written_another_way(two_towers):
    """
    A later fact replaces an earlier one on the same ground term written differently.
    """
    block = literal(two_towers.a)
    state = (
        State()
        .add_facts(block.rests_on(support=two_towers.table) == False)
        .add_facts(block.rests_on(support=two_towers.table, directly=True) == True)
    )

    assert list(state.assignments.values()) == [True]


# %% comparing states


def test_states_with_the_same_facts_are_equal_and_hash_alike(two_towers):
    """
    States are compared by what they assign.
    """
    first = two_towers.b_on_c
    second = State().add_facts(literal(two_towers.b).support == two_towers.c)

    assert (first, hash(first)) == (second, hash(second))


def test_states_with_different_facts_differ(two_towers):
    """
    States that assign different values are different.
    """
    assert two_towers.b_on_c != State().add_facts(
        literal(two_towers.b).support == two_towers.a
    )
