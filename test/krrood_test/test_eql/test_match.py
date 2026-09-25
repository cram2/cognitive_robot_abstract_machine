from dataclasses import dataclass

import pytest
from typing_extensions import Tuple

from krrood.entity_query_language.factories import (
    entity,
    flat_variable,
    set_of,
    variable,
    the,
    an,
    a,
)
from krrood.entity_query_language.exceptions import (
    CalledMatchAfterResolution,
    CalledMatchMultipleTimes,
    PositionalArgumentsInMatchPattern,
    MatchTypeCannotBeDetermined,
    ReadOnlyMapping,
    SymbolicDunderAccessError,
)
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    HasSymbolicOperations,
)
from krrood.entity_query_language.operators.arithmetic import ArithmeticOperation
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.match import AttributeMatch, Match
from krrood.entity_query_language.query.query_modifiers import HasQueryModifiers
from krrood.entity_query_language.core.base_expressions import UnificationDict
from ..dataset.example_classes import KRROODPositions, KRROODPosition
from ..dataset.semantic_world_like_classes import (
    Cabinet,
    Drawer,
    FixedConnection,
    Container,
    Handle,
)


def test_doc_match():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    robots = [Robot("R2D2", 100), Robot("C3PO", 0)]
    query = a(Robot)(name="R2D2", battery=100).from_(robots)
    assert query.tolist()[0].name == "R2D2"


def test_match(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"),
        child=a(Handle)(name="Handle1"),
    ).from_(world.connections)
    fixed_connection_query = the(fixed_connection)

    fc = variable(FixedConnection, domain=None)
    fixed_connection_query_manual = the(
        entity(fc).where(
            HasType(fc.parent, Container),
            HasType(fc.child, Handle),
            fc.parent.name == "Container1",
            fc.child.name == "Handle1",
        )
    )

    fixed_connection_match_result = fixed_connection_query.tolist()[0]
    fixed_connection_manual_result = fixed_connection_query_manual.tolist()[0]
    assert fixed_connection_match_result == fixed_connection_manual_result
    assert fixed_connection.first() == fixed_connection_manual_result
    assert isinstance(fixed_connection_match_result, FixedConnection)
    assert fixed_connection_match_result.parent.name == "Container1"
    assert isinstance(fixed_connection_match_result.child, Handle)
    assert fixed_connection_match_result.child.name == "Handle1"


def test_select(handles_and_containers_world):
    world = handles_and_containers_world

    # Method 1
    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"), child=a(Handle)(name="Handle1")
    ).from_(world.connections)
    container_and_handle = the(
        set_of(
            container := fixed_connection.parent,
            handle := fixed_connection.child,
        )
    )

    # Method 2
    fixed_connection_2 = variable(FixedConnection, domain=world.connections)
    container_and_handle_2 = the(
        set_of(
            container_2 := fixed_connection_2.parent,
            handle_2 := fixed_connection_2.child,
        ).where(
            HasType(container_2, Container),
            HasType(handle_2, Handle),
            container_2.name == "Container1",
            handle_2.name == "Handle1",
        )
    )

    assert set(container_and_handle_2.tolist()[0].values()) == set(
        container_and_handle.tolist()[0].values()
    )

    answers = container_and_handle.tolist()[0]
    assert isinstance(answers, UnificationDict)
    assert answers[container].name == "Container1"
    assert answers[handle].name == "Handle1"


def test_select_where(handles_and_containers_world):
    world = handles_and_containers_world

    # Method 1
    fixed_connection = a(FixedConnection)(
        parent=a(Container),
        child=a(Handle),
    ).from_(world.connections)
    container_and_handle = a(
        set_of(
            container := fixed_connection.parent,
            handle := fixed_connection.child,
        ).where(container.size > 1)
    )
    # Method 2
    fixed_connection_2 = variable(FixedConnection, domain=world.connections)
    container_and_handle_2 = the(
        set_of(
            container_2 := fixed_connection_2.parent,
            handle_2 := fixed_connection_2.child,
        ).where(
            HasType(container_2, Container),
            HasType(handle_2, Handle),
            container_2.size > 1,
        )
    )

    assert set(
        map(lambda x: tuple(x.values()), container_and_handle_2.tolist())
    ) == set(map(lambda x: tuple(x.values()), container_and_handle.tolist()))

    answers = container_and_handle.tolist()
    assert len(answers) == 1
    assert isinstance(answers[0], UnificationDict)
    assert answers[0][container].name == "Container3"
    assert answers[0][handle].name == "Handle3"


def test_domain_carrying_an_is_a_select():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    robots = [Robot("R2D2", 100), Robot("C3PO", 0)]
    # an(...)(kwargs).from_(domain) stays one Match carrying the domain; the backend decides
    # whether to select over it or generate from it, so from_ does not collapse it to an Entity.
    query = a(Robot)(name="R2D2", battery=100).from_(robots)
    assert isinstance(query, Match)
    assert query._domain_ is robots
    assert query.tolist()[0].name == "R2D2"


# %% direct symbolic attribute access on a match


def test_attribute_access_is_an_attribute_of_the_query_the_match_stands_for():
    match = a(KRROODPosition)(x=1.0, y=2.0, z=3.0)
    assert match.x is match._symbolic_expression_.x


def test_attribute_access_works_for_names_previously_shadowed_by_match_internals(
    handles_and_containers_world,
):
    # `parent`, `child` and `name` are fields of the matched classes here; they must
    # resolve symbolically instead of hitting internals of the match machinery.
    world = handles_and_containers_world
    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"), child=a(Handle)(name="Handle1")
    ).from_(world.connections)
    container_and_handle = the(
        set_of(
            container := fixed_connection.parent,
            handle := fixed_connection.child,
        )
    )
    answers = container_and_handle.tolist()[0]
    assert answers[container].name == "Container1"
    assert answers[handle].name == "Handle1"


def test_name_attribute_of_matched_class_is_symbolic():
    match = a(Handle)
    assert match.name is match._symbolic_expression_.name


def test_calling_match_after_resolution_raises():
    match = a(KRROODPosition)
    _ = match.x
    with pytest.raises(CalledMatchAfterResolution):
        match(x=1.0)


def test_dunder_attribute_access_is_not_symbolic():
    match = a(KRROODPosition)
    with pytest.raises(SymbolicDunderAccessError):
        _ = match.__missing_dunder__


def test_underscore_attribute_access_is_not_symbolic():
    match = a(KRROODPosition)
    with pytest.raises(AttributeError):
        _ = match._missing_internal_


def test_from_restricts_the_search():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str

    subset = [Robot("R2D2"), Robot("C3PO")]
    # from_ selects only over the given domain.
    selected = a(Robot).from_(subset).tolist()
    assert {robot.name for robot in selected} == {"R2D2", "C3PO"}


def test_from_after_where_still_restricts_the_search():
    """
    ``from_`` must still scope the search to its domain even when ``where`` (which
    triggers resolution) is called first: the expression built while resolving ``where``
    must not stay permanently cached against the domain-less subject ``__call__``
    eagerly created.
    """

    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    subset = [Robot("R2D2", 100), Robot("C3PO", 0)]
    query = a(Robot)()
    query.where(query.battery >= 0)
    query.from_(subset)
    selected = query.tolist()
    assert {robot.name for robot in selected} == {"R2D2", "C3PO"}


def test_from_without_kwargs_selects_all(handles_and_containers_world):
    world = handles_and_containers_world
    # an(Type).from_(X) with no kwargs is a valid "any Type in X" select.
    selected = a(FixedConnection).from_(world.connections).tolist()
    assert selected
    assert all(isinstance(connection, FixedConnection) for connection in selected)


def test_match_without_domain_selects_from_symbol_graph():
    """
    A domain-less match evaluated standalone (default selective backend) *selects* from
    the SymbolGraph for ``Symbol`` types: it returns the existing registered instance
    rather than constructing a new one.

    Generation requires an explicit generative backend.
    """
    existing = KRROODPosition(1.0, 2.0, 3.0)
    result = a(KRROODPosition)(x=1.0, y=2.0, z=3.0).tolist()
    # the existing object itself is returned (selection), not a freshly-built equal one
    assert any(r is existing for r in result)
    assert all(isinstance(r, KRROODPosition) and r == existing for r in result)


def test_match_with_list():
    domain = [
        KRROODPositions([KRROODPosition(1, 2, 3), KRROODPosition(1, 2, 3)], ["a", "b"]),
        KRROODPositions([KRROODPosition(1, 2, 3)], ["a"]),
    ]

    q = a(KRROODPositions)(
        positions=[
            a(KRROODPosition)(
                x=1,
                y=2,
            ),
            KRROODPosition(1, 2, 3),
        ],
        some_strings=["a", "b"],
    ).from_(domain)

    r = q.tolist()
    assert r == [domain[0]]


# %% an()/the() with a callable factory: target_type inference and default


def test_an_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = an(make_position)
    assert match._type_ is KRROODPosition


def test_the_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = the(make_position)
    assert match._type_ is KRROODPosition


def test_an_uses_explicit_target_type_for_unannotated_callable():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    match = an(make_position, target_type=KRROODPosition)
    assert match._type_ is KRROODPosition


def test_an_raises_when_callable_type_cannot_be_determined():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    with pytest.raises(MatchTypeCannotBeDetermined):
        an(make_position)


def test_a_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = a(make_position)
    assert match._type_ is KRROODPosition


def test_a_uses_explicit_target_type_for_unannotated_callable():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    match = a(make_position, target_type=KRROODPosition)
    assert match._type_ is KRROODPosition


# %% Match._has_ellipsis_attributes_


def test_has_ellipsis_attributes_true_for_direct_ellipsis():
    match = a(KRROODPosition)(x=..., y=2, z=3)
    assert match._has_ellipsis_attributes_ is True


def test_has_ellipsis_attributes_false_without_ellipsis():
    match = a(KRROODPosition)(x=1, y=2, z=3)
    assert match._has_ellipsis_attributes_ is False


def test_has_ellipsis_attributes_true_for_nested_ellipsis():
    match = a(KRROODPositions)(
        positions=[a(KRROODPosition)(x=..., y=2, z=3)],
        some_strings=["a"],
    )
    assert match._has_ellipsis_attributes_ is True


def test_has_ellipsis_attributes_true_for_ellipsis_element_in_plain_list():
    """
    An ``...`` element sitting inside an otherwise-concrete list (no nested ``Match``
    elements) is still resolved as one literal-valued attribute match, but is just as
    underspecified as a direct ``x=...`` assignment.
    """
    match = a(KRROODPositions)(
        positions=[KRROODPosition(1, 2, 3)],
        some_strings=["a", ..., "c"],
    )
    assert match._has_ellipsis_attributes_ is True


def test_has_ellipsis_attributes_true_for_ellipsis_mixed_with_nested_match_in_list():
    match = a(KRROODPositions)(
        positions=[a(KRROODPosition)(x=1, y=2, z=3), ...],
        some_strings=["a", "b"],
    )
    assert match._has_ellipsis_attributes_ is True


def test_has_ellipsis_attributes_true_for_ellipsis_element_in_plain_set():
    match = a(KRROODPositions)(
        positions=[KRROODPosition(1, 2, 3)],
        some_strings={"a", ..., "c"},
    )
    assert match._has_ellipsis_attributes_ is True


# %% a match reads like a query


@dataclass(unsafe_hash=True)
class FieldNamedLikeAQueryMethod:
    """
    A matched class with a field whose name a :class:`Query` member also uses.
    """

    name: str
    build: int


def test_where_through_a_forwarded_attribute_filters():
    """
    The condition the forwarding makes natural must actually filter, which it only does
    once a condition rooted at the query correlates with that query's own bindings.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    match = a(KRROODPosition).from_(positions)
    assert [position.x for position in match.where(match.x >= 5).tolist()] == [5.0]


def test_limit_bounds_the_results_and_keeps_the_chain_on_the_match():
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    match = a(KRROODPosition).from_(positions)
    assert match.limit(1) is match
    assert len(match.tolist()) == 1


def test_ordered_by_a_forwarded_attribute_orders_the_results():
    positions = [
        KRROODPosition(1.0, 0.0, 0.0),
        KRROODPosition(5.0, 0.0, 0.0),
        KRROODPosition(3.0, 0.0, 0.0),
    ]
    match = a(KRROODPosition).from_(positions)
    ordered = match.ordered_by(match.x, descending=True)
    assert [position.x for position in ordered.tolist()] == [5.0, 3.0, 1.0]


def test_every_query_modifier_returns_the_match():
    """
    A chain stays on the match rather than silently continuing on the lowered query.

    Each modifier gets its own match: the modifiers compose only in the combinations the
    lowered query itself supports, which is not what this test is about.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0)]
    assert (match := a(KRROODPosition).from_(positions)).where(match.x >= 0) is match
    assert (match := a(KRROODPosition).from_(positions)).having(match.x >= 0) is match
    assert (match := a(KRROODPosition).from_(positions)).ordered_by(match.x) is match
    assert (match := a(KRROODPosition).from_(positions)).grouped_by(match.y) is match
    assert (match := a(KRROODPosition).from_(positions)).distinct() is match
    assert (match := a(KRROODPosition).from_(positions)).limit(1) is match


def test_a_matched_class_field_is_not_shadowed_by_a_query_method():
    """
    Forwarding reaches the lowered query's symbolic attributes rather than its
    namespace, so a matched class may name a field after any :class:`Query` member.
    """
    releases = [
        FieldNamedLikeAQueryMethod("stable", 7),
        FieldNamedLikeAQueryMethod("nightly", 9),
    ]
    match = a(FieldNamedLikeAQueryMethod).from_(releases)
    # Reading the same name off the lowered query gives Query.build, which is what
    # forwarding by name would have returned.
    assert isinstance(match.build, Attribute)
    assert [release.name for release in match.where(match.build > 7).tolist()] == [
        "nightly"
    ]


def test_match_and_query_share_the_query_modifier_interface():
    assert isinstance(a(KRROODPosition), HasQueryModifiers)
    assert isinstance(entity(variable(KRROODPosition, [])), HasQueryModifiers)


# %% a match given where an expression is expected


def test_the_quantifies_a_match_itself():
    """
    Quantifying a match reads the query it stands for, so the match needs no handle of
    its own to be quantified through.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    assert the(a(KRROODPosition)(x=5.0).from_(positions)).tolist() == [positions[1]]


def test_an_quantifies_a_match_itself():
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    assert an(a(KRROODPosition)(x=5.0).from_(positions)).tolist() == [positions[1]]


def test_entity_selects_a_match():
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    assert entity(a(KRROODPosition)(x=5.0).from_(positions)).tolist() == [positions[1]]


def test_set_of_selects_a_match_beside_one_of_its_attributes():
    """
    A match selected beside one of its own attributes must bind to the same row, which
    it only does once the selection reads the query the match stands for.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    match = a(KRROODPosition)(x=5.0).from_(positions)
    (row,) = set_of(match, match.y).tolist()
    assert row[match] == positions[1]
    assert row[match.y] == 0.0


def test_selecting_a_match_carries_its_pattern():
    """
    The match's own conditions travel with it, which selecting the bare variable it
    created would drop.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    match = a(KRROODPosition)(x=5.0).from_(positions)
    assert entity(match).tolist() == [positions[1]]


def test_match_and_variable_share_the_symbolic_operations_interface():
    assert isinstance(a(KRROODPosition), HasSymbolicOperations)
    assert isinstance(variable(KRROODPosition, []), HasSymbolicOperations)


def test_names_that_became_match_internals_are_symbolic_attributes():
    """
    A rename missed by the migration would come back as a symbolic attribute named after
    the internal - truthy, chainable and wrong - so every renamed public name is pinned
    as the matched class's attribute instead.
    """
    match = a(KRROODPosition)(x=1.0, y=2.0, z=3.0)
    assert match.parent is match._symbolic_expression_.parent
    assert match.children is match._symbolic_expression_.children
    assert match.type is match._symbolic_expression_.type
    assert match.conditions is match._symbolic_expression_.conditions
    assert match.resolved is match._symbolic_expression_.resolved
    assert match.id is match._symbolic_expression_.id
    assert match.root is match._symbolic_expression_.root
    assert match.descendants is match._symbolic_expression_.descendants
    assert match.domain is match._symbolic_expression_.domain
    assert match.factory is match._symbolic_expression_.factory
    assert match.kwargs is match._symbolic_expression_.kwargs
    assert (
        match.has_ellipsis_attributes
        is match._symbolic_expression_.has_ellipsis_attributes
    )
    assert match.variable is match._symbolic_expression_.variable
    assert match.expression is match._symbolic_expression_.expression
    assert (
        match.matches_with_variables
        is match._symbolic_expression_.matches_with_variables
    )
    assert match.update_fields is match._symbolic_expression_.update_fields
    assert (
        match.create_or_update_variable
        is match._symbolic_expression_.create_or_update_variable
    )
    assert (
        match.has_cause_attributes is match._symbolic_expression_.has_cause_attributes
    )


# %% a match reads like an instance in every symbolic operation


@dataclass(unsafe_hash=True)
class SubscriptableRow:
    """
    A matched class whose instances support indexing, for ``match[key]``.
    """

    name: str
    cells: Tuple[int, ...]

    def __getitem__(self, index: int) -> int:
        return self.cells[index]


@dataclass(unsafe_hash=True)
class CallableAdder:
    """
    A matched class whose instances are callable, for the second parentheses.
    """

    name: str
    offset: int

    def __call__(self, value: int) -> int:
        return self.offset + value


def test_indexing_a_match_indexes_the_matched_instance():
    rows = [SubscriptableRow("first", (7, 8)), SubscriptableRow("second", (9, 10))]
    match = a(SubscriptableRow).from_(rows)
    assert match[0] is match._symbolic_expression_[0]
    assert [row.name for row in match.where(match[0] == 7).tolist()] == ["first"]


def test_comparing_a_match_builds_a_condition_rather_than_answering_identity():
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    match = a(KRROODPosition).from_(positions)
    assert isinstance(match == positions[1], Comparator)
    assert [position.x for position in match.where(match == positions[1]).tolist()] == [
        5.0
    ]


def test_a_match_on_the_other_side_of_a_comparison_also_filters():
    """
    A match is not part of the expression graph, so where an operand is expected it
    contributes the expression it stands for - otherwise it would be read as a literal
    value, and the condition would pass every row.
    """
    positions = [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(5.0, 0.0, 0.0)]
    subject = variable(KRROODPosition, positions)
    match = a(KRROODPosition)(x=5.0).from_(positions)
    found = entity(subject).where(subject == match).tolist()
    assert [position.x for position in found] == [5.0]


def test_arithmetic_on_a_match_operates_on_the_lowered_query():
    """
    Attaching an expression to a query copies the query node while preserving its
    identifier, so the operand is matched by identifier rather than by object identity.
    """
    match = a(KRROODPosition)(x=1.0, y=2.0, z=3.0)
    doubled = match * 2
    assert isinstance(doubled, ArithmeticOperation)
    assert doubled.left._id_ == match._symbolic_expression_._id_


def test_a_match_cannot_be_iterated():
    """
    Indexing a match must not hand Python its legacy sequence protocol, which would
    iterate a match endlessly, every index being a valid expression.
    """
    match = a(SubscriptableRow)
    with pytest.raises(TypeError):
        iter(match)


def test_the_second_parentheses_call_the_matched_instance():
    adders = [CallableAdder("one", 1), CallableAdder("two", 2)]
    match = a(CallableAdder)().from_(adders)
    assert [adder.name for adder in match.where(match(10) == 12).tolist()] == ["two"]


def test_an_empty_pattern_still_takes_its_own_parentheses():
    """
    The first parentheses are the pattern even when it is empty, so a match over every
    instance of a callable class is called through a second pair.
    """
    adders = [CallableAdder("one", 1), CallableAdder("two", 2)]
    match = a(CallableAdder)().from_(adders)
    assert [adder.name for adder in match.where(match(10) == 11).tolist()] == ["one"]


def test_the_pattern_parentheses_reject_positional_arguments():
    with pytest.raises(PositionalArgumentsInMatchPattern):
        a(CallableAdder)("one")


def test_a_second_pattern_still_raises_when_the_matched_class_is_not_callable():
    match = a(KRROODPosition)(x=1.0)
    with pytest.raises(CalledMatchMultipleTimes):
        match(y=2.0)


# %% writing an attribute value back into a match


def test_writing_through_a_flattened_attribute_of_a_match_is_rejected():
    """
    A flattening reaches every element of a collection without naming one, so a match
    whose attribute is reached through one has no single place to write the value.
    """
    drawer = Drawer(handle=Handle(name="Handle1"), container=Container(name="Drawer1"))
    match = a(Cabinet)(container=Container(name="Container1"), drawers=[drawer])
    attribute_match = AttributeMatch(
        _parent_=match,
        attribute_name="handle",
        assigned_value=Handle(name="Handle9"),
        _variable_=flat_variable(match._variable_.drawers).handle,
    )

    with pytest.raises(ReadOnlyMapping):
        attribute_match._update_kwargs_from(match)
