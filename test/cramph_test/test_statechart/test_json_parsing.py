import json

import pytest
from typing_extensions import Callable

from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import (
    LifeCyclePredicate,
    StatechartJSONKey,
    ObservationPredicate,
    TransitionConditionJSONKey,
    TransitionKind,
)
from cramph.exceptions import (
    NodeNotFoundError,
    NodeStateVariableNotSerializableError,
    UnknownConditionVariableError,
    UnsupportedConditionSyntaxError,
)
from cramph.composites import Parallel, Sequence
from cramph.node import (
    CancelStatechart,
    DeserializedNodeTracker,
    StructureCopyCompositeNode,
    StructureCopyNode,
    TransitionCondition,
)
from cramph.node import EndStatechart
from cramph.state import LifeCycleState, ObservationState
from cramph.statechart import Statechart
from cramph.nodes_for_testing import (
    ConstTrueNode,
    CompositeNodeWithNestedCompositeChild,
    NodeObservingAPredicate,
    NodeReferringToAWorldEntity,
    NodeWithOwnStructureCopy,
    SequenceCreatingItsSteps,
    SpecializedNodeWithOwnStructureCopy,
)
from krrood.adapters.json_serializer import to_json
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    logic_and,
    logic_or,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityReferenceWriter,
    WorldEntityWithID,
)


def test_trinary_transition() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    node3 = ConstTrueNode()
    node4 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)
    msc.add_node(node3)
    msc.add_node(node4)

    node1.start_condition = logic_and(
        node2.observes_true,
        logic_or(node3.observes_true, node4.observes_false),
    )
    condition = node1._start_condition
    json_data = condition.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    condition_copy = TransitionCondition.from_json(
        new_json_data,
        **DeserializedNodeTracker.from_statechart(msc).create_kwargs(),
    )
    assert condition_copy == condition


@pytest.mark.parametrize("transition_kind", TransitionKind.ending_kinds())
def test_ending_condition_round_trip(transition_kind: TransitionKind) -> None:
    """
    Every condition that ends a node survives serialization, including the predicate it
    reads, and keeps the kind that decides the outcome it yields.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([first := ConstTrueNode(), second := ConstTrueNode()])
    second.set_condition(
        transition_kind,
        logic_and(first.is_succeeded, second.observes_true),
    )
    [condition] = [
        condition
        for condition in second.conditions
        if condition.kind is transition_kind
    ]

    condition_copy = TransitionCondition.from_json(
        json.loads(json.dumps(condition.to_json())),
        **DeserializedNodeTracker.from_statechart(msc).create_kwargs(),
    )

    assert condition_copy == condition
    assert condition_copy.kind is transition_kind


def test_nested_goals(tmp_path) -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(
        sequence := Sequence(
            nodes=[
                ConstTrueNode(),
                CompositeNodeWithNestedCompositeChild(),
            ]
        )
    )
    msc.add_node(EndStatechart.when_true(sequence))
    json_data = msc.create_structure_copy().to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )
    msc_copy.compiler.add_transitions()
    msc.draw(str(tmp_path / "statechart.pdf"))

    for node in msc.nodes:
        node_copy = msc_copy.get_node_by_index(node.index)
        assert node.index == node_copy.index
        if node.parent_node_index is not None:
            assert node.parent_node.unique_name == node_copy.parent_node.unique_name
        else:
            assert node_copy.parent_node_index is None


def test_collapsed_goal_survives_json_round_trip() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(goal := CompositeNodeWithNestedCompositeChild())
    goal.plot_specifications.collapse_children = True
    msc.add_node(EndStatechart.when_true(goal))
    json_data = msc.create_structure_copy().to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )

    assert msc_copy.get_node_by_index(goal.index).plot_specifications.collapse_children


def test_structure_copy_keeps_every_condition() -> None:
    """
    A structural copy stands in for the chart it was made from, so every transition
    condition of a node comes along with it.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
    for transition_kind in TransitionKind:
        node.set_condition(transition_kind, trigger.observes_true)

    node_copy = msc.create_structure_copy().get_node_by_index(node.index)

    assert [str(condition) for condition in node_copy.conditions] == [
        str(condition) for condition in node.conditions
    ]


def test_structure_copy_uses_the_kind_a_node_declares() -> None:
    """
    Node kinds declared outside the statechart's own node classes decide their structure
    copy themselves, so a plain statechart copies them without knowing them.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(
        node := SpecializedNodeWithOwnStructureCopy(name="specialized", detail=3)
    )

    node_copy = msc.create_structure_copy().get_node_by_index(node.index)

    assert type(node_copy) is NodeWithOwnStructureCopy
    assert node_copy.name == node.name


def test_structure_copy_keeps_the_base_kinds_of_the_statechart_nodes() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes(
        [
            trigger := ConstTrueNode(),
            goal := CompositeNodeWithNestedCompositeChild(),
            end := EndStatechart.when_true(trigger),
            cancel := CancelStatechart.when_true(
                trigger, exception=NodeNotFoundError(name="muh")
            ),
        ]
    )

    msc_copy = msc.create_structure_copy()

    assert type(msc_copy.get_node_by_index(trigger.index)) is StructureCopyNode
    assert type(msc_copy.get_node_by_index(goal.index)) is StructureCopyCompositeNode
    assert type(msc_copy.get_node_by_index(end.index)) is EndStatechart
    cancel_copy = msc_copy.get_node_by_index(cancel.index)
    assert type(cancel_copy) is CancelStatechart
    assert cancel_copy.exception is cancel.exception


def test_structure_copy_conditions_read_the_copied_nodes() -> None:
    """
    The conditions of a structural copy read the nodes of the copy, not the nodes of the
    chart it was made from.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
    node.start_condition = logic_and(trigger.observes_true, trigger.is_succeeded)
    node.success_condition = node.observes_true

    msc_copy = msc.create_structure_copy()

    assert [
        variable.statechart_node
        for copied_node in msc_copy.nodes
        for condition in copied_node.conditions
        for variable in condition.variables
    ] == [
        msc_copy.get_node_by_index(variable.statechart_node.index)
        for original_node in msc.nodes
        for condition in original_node.conditions
        for variable in condition.variables
    ]


def test_start_condition(statechart_executor: StatechartExecutor) -> None:
    msc = Statechart(context=statechart_executor.context)
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = ConstTrueNode()
    msc.add_node(end)

    node1.success_condition = node1.observes_true
    node2.start_condition = node1.observes_true
    node2.pause_condition = node3.observes_true
    end.start_condition = logic_and(node2.observes_true, node3.observes_true)

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    kin_sim = StatechartExecutor(context=StatechartContext(world=World()))
    msc_copy = Statechart.from_json(new_json_data, context=kin_sim.context)

    statechart_executor.compile(statechart=msc)
    kin_sim.compile(statechart=msc_copy)
    for index, node in enumerate(msc.nodes):
        assert node.name == msc_copy.nodes[index].name
    assert len(msc.edges) == len(msc_copy.edges)
    for index, edge in enumerate(msc.edges):
        assert edge == msc_copy.edges[index]


def test_cancel_statechart() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(node := ConstTrueNode())
    msc.add_node(
        CancelStatechart.when_true(node, exception=NodeNotFoundError(name="muh"))
    )

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    kin_sim = StatechartExecutor(
        context=StatechartContext(world=World()),
    )
    msc_copy = Statechart.from_json(new_json_data, context=kin_sim.context)

    kin_sim.compile(statechart=msc_copy)

    with pytest.raises(Exception):
        kin_sim.tick_until_end()


def test_to_json_does_not_accumulate_edges() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)
    node2.start_condition = node1.observes_true

    first = msc.to_json()
    edges_after_first = len(msc.edges)
    second = msc.to_json()
    edges_after_second = len(msc.edges)

    assert edges_after_first == edges_after_second
    assert first[StatechartJSONKey.CONDITIONS] == second[StatechartJSONKey.CONDITIONS]


def test_duplicate_condition() -> None:
    """
    Tests if two condition with the same name and type will be preserved.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes(
        [
            node1 := ConstTrueNode(),
            node2 := ConstTrueNode(),
            node3 := ConstTrueNode(),
            end := EndStatechart(),
        ]
    )
    node2.start_condition = node1.observes_true
    node3.start_condition = node1.observes_true
    end.start_condition = logic_and(node2.observes_true, node3.observes_true)

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )
    msc_copy.compiler.add_transitions()
    assert len(msc_copy.unique_edges) == 3


def test_child_added_to_goal_is_its_child_once_after_json_round_trip() -> None:
    """
    A child added to a goal before compilation is a child of the deserialized goal once.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(sequence := Sequence())
    sequence.add_node(child := ConstTrueNode())
    msc.add_node(EndStatechart.when_true(sequence))

    new_json_data = json.loads(json.dumps(msc.to_json()))

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )
    sequence_copy = msc_copy.get_node_by_index(sequence.index)
    assert [node.name for node in sequence_copy.nodes] == [child.name]


def test_children_of_compiled_goal_are_its_children_once_after_json_round_trip(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Compiling adds the children of a goal to the statechart while the goal keeps them in
    its own node list, and each is still a child of the deserialized goal once.
    """
    msc = Statechart(context=statechart_executor.context)
    msc.add_node(
        sequence := Sequence(nodes=[ConstTrueNode(name="a"), ConstTrueNode(name="b")])
    )
    msc.add_node(EndStatechart.when_true(sequence))
    statechart_executor.compile(statechart=msc)

    new_json_data = json.loads(json.dumps(msc.to_json()))

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )
    sequence_copy = msc_copy.get_node_by_index(sequence.index)
    assert sequence_copy.nodes == [
        msc_copy.get_node_by_index(node.index) for node in sequence.nodes
    ]


def assert_children_keep_their_order_after_json_round_trip(
    msc: Statechart, sequence: Sequence
) -> None:
    """
    Asserts that the deserialized `sequence` holds the copies of its children in the
    order `sequence` holds them.
    """
    msc_copy = Statechart.from_json(
        json.loads(json.dumps(msc.to_json())), context=StatechartContext(world=World())
    )
    sequence_copy = msc_copy.get_node_by_index(sequence.index)
    assert sequence_copy.nodes == [
        msc_copy.get_node_by_index(node.index) for node in sequence.nodes
    ]


def test_a_replaced_step_keeps_its_position_after_json_round_trip() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(sequence := SequenceCreatingItsSteps())
    msc.replace(sequence.nodes[1], ConstTrueNode())

    assert_children_keep_their_order_after_json_round_trip(msc, sequence)


def test_an_inserted_step_keeps_its_position_after_json_round_trip() -> None:
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(sequence := SequenceCreatingItsSteps())
    sequence.insert_before(sequence.nodes[1], ConstTrueNode())

    assert_children_keep_their_order_after_json_round_trip(msc, sequence)


def test_node_referenced_by_another_node_is_one_instance_after_json_round_trip() -> (
    None
):
    """
    A node that another node refers to is deserialized as the node of the statechart,
    not as a detached copy.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(watched := ConstTrueNode())
    msc.add_node(watching := NodeObservingAPredicate(watched_node=watched))
    msc.add_node(EndStatechart.when_true(watched))

    new_json_data = json.loads(json.dumps(msc.to_json()))

    msc_copy = Statechart.from_json(
        new_json_data, context=StatechartContext(world=World())
    )
    watching_copy = msc_copy.get_node_by_index(watching.index)
    assert watching_copy.watched_node is msc_copy.get_node_by_index(watched.index)


# %% world entities


def _root_body(world: World) -> WorldEntityWithID:
    return world.get_kinematic_structure_entity_by_name("root")


def _connection(world: World) -> WorldEntityWithID:
    return world.get_connection_by_name("root_T_tip")


@pytest.mark.parametrize(
    "entity_of", [_root_body, _connection], ids=["a body", "a connection"]
)
def test_a_statechart_refers_to_world_entities_by_reference(
    mini_world: World, entity_of: Callable[[World], WorldEntityWithID]
) -> None:
    """
    Whoever reads a statechart has the world it was built for, so its nodes point at the
    entities of that world instead of carrying copies of them.
    """
    world_entity = entity_of(mini_world)
    msc = Statechart(context=StatechartContext(world=mini_world))
    msc.add_node(node := NodeReferringToAWorldEntity(world_entity=world_entity))

    json_data = json.loads(json.dumps(msc.to_json()))

    assert json_data[StatechartJSONKey.NODES][node.index][
        "world_entity"
    ] == WorldEntityReferenceWriter().write_reference(world_entity)

    tracker = WorldEntityWithIDKwargsTracker.from_world(mini_world)
    msc_copy = Statechart.from_json(
        json_data,
        context=StatechartContext(world=mini_world),
        **tracker.create_kwargs(),
    )
    assert msc_copy.get_node_by_index(node.index).world_entity is world_entity


# %% conditions of a chart that is not compiled yet


def assert_conditions_survive_json_round_trip(msc: Statechart) -> None:
    """
    Serializes `msc` before it is compiled, then compiles it and its copy and checks
    that every node of both ends up with the same conditions.
    """
    msc_copy = Statechart.from_json(
        json.loads(json.dumps(msc.to_json())), context=StatechartContext(world=World())
    )

    msc.compiler.compile()
    msc_copy.compiler.compile()

    assert [
        [str(condition) for condition in node.conditions] for node in msc_copy.nodes
    ] == [[str(condition) for condition in node.conditions] for node in msc.nodes]


def test_conditions_of_goal_children_survive_json_round_trip() -> None:
    """
    The children of a goal join the chart only when it is compiled, and the conditions
    set on them before that come along with them.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    child = ConstTrueNode()
    sibling = ConstTrueNode()
    msc.add_node(Parallel(nodes=[child, sibling]))
    child.success_condition = child.observes_true
    sibling.start_condition = child.observes_true

    assert_conditions_survive_json_round_trip(msc)


def test_goal_reading_its_child_survives_json_round_trip() -> None:
    """
    A goal may read its child before the child has joined the chart.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    child = ConstTrueNode()
    msc.add_node(parallel := Parallel(nodes=[child]))
    parallel.success_condition = child.is_succeeded

    assert_conditions_survive_json_round_trip(msc)


def test_constant_condition_survives_json_round_trip() -> None:
    """
    A condition reading no node is serialized like any other.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_node(node := ConstTrueNode())
    node.start_condition = Scalar.const_false()

    assert_conditions_survive_json_round_trip(msc)


@pytest.mark.parametrize("predicate", [*LifeCyclePredicate, *ObservationPredicate])
def test_every_predicate_survives_json_round_trip(
    predicate: LifeCyclePredicate | ObservationPredicate,
) -> None:
    """
    Every test a condition can read about a node reads back as the same test.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([watched := ConstTrueNode(), reader := ConstTrueNode()])
    match predicate:
        case LifeCyclePredicate():
            reader.start_condition = watched._life_cycle_predicate(predicate)
        case ObservationPredicate():
            reader.start_condition = watched._observation_predicate(predicate)

    assert_conditions_survive_json_round_trip(msc)


def test_condition_naming_an_unknown_variable_is_rejected() -> None:
    """
    A document may name a predicate nodes no longer offer, for example one that was
    removed after the document was written.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([watched := ConstTrueNode(), reader := ConstTrueNode()])
    reader.start_condition = watched.is_succeeded
    document = json.dumps(msc.to_json()).replace(
        f".{LifeCyclePredicate.IS_SUCCEEDED.attribute_name}", ".has_succeeded"
    )

    with pytest.raises(UnknownConditionVariableError):
        Statechart.from_json(
            json.loads(document), context=StatechartContext(world=World())
        )


def _negated_arithmetically(expression: str) -> str:
    return f"-{expression}"


def _compared_with_itself(expression: str) -> str:
    return f"{expression} == {expression}"


@pytest.mark.parametrize(
    "rewrite_condition",
    [_negated_arithmetically, _compared_with_itself],
    ids=["a unary operator other than not", "a comparison"],
)
def test_condition_using_unsupported_syntax_is_rejected(
    rewrite_condition: Callable[[str], str],
) -> None:
    """
    A rendered condition is Python syntax, so a document may hold syntax that has no
    meaning as a condition.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([watched := ConstTrueNode(), reader := ConstTrueNode()])
    reader.start_condition = watched.is_succeeded
    document = msc.to_json()
    [start_condition] = [
        condition
        for condition in document[StatechartJSONKey.CONDITIONS]
        if condition[TransitionConditionJSONKey.OWNER] == reader._node_id
        and condition[TransitionConditionJSONKey.KIND] == TransitionKind.START.name
    ]
    start_condition[TransitionConditionJSONKey.EXPRESSION] = rewrite_condition(
        start_condition[TransitionConditionJSONKey.EXPRESSION]
    )

    with pytest.raises(UnsupportedConditionSyntaxError):
        Statechart.from_json(
            json.loads(json.dumps(document)), context=StatechartContext(world=World())
        )


# %% node state variables


def test_node_state_variable_is_not_json_serializable() -> None:
    variable = ConstTrueNode().observation_variable

    with pytest.raises(NodeStateVariableNotSerializableError) as error:
        to_json(variable)

    assert error.value.variable is variable


def test_written_conditions_of_a_compiled_statechart_survive_json_round_trip(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Compiling a statechart parsed from a compiled one leaves the conditions as they were
    written on the original.
    """
    msc = Statechart(context=statechart_executor.context)
    msc.add_node(sequence := Sequence(nodes=[ConstTrueNode()]))
    written = [str(condition) for condition in sequence.conditions]
    statechart_executor.compile(statechart=msc)

    msc_copy = Statechart.from_json(
        json.loads(json.dumps(msc.to_json())), context=StatechartContext(world=World())
    )
    StatechartExecutor(msc_copy.context).compile(statechart=msc_copy)

    sequence_copy = msc_copy.get_node_by_index(sequence.index)
    assert [str(condition) for condition in sequence_copy.conditions] == written


# %% states of a ticked statechart


@pytest.mark.parametrize("state_type", [LifeCycleState, ObservationState])
def test_state_of_an_executed_json_parsed_statechart_survives_json_round_trip(
    state_type: type[LifeCycleState] | type[ObservationState],
) -> None:
    """
    The states a statechart parsed from JSON reached by ticking read back as they were.
    """
    msc = Statechart(context=StatechartContext(world=World()))
    msc.add_nodes([first := ConstTrueNode(), second := ConstTrueNode()])
    second.start_condition = first.observes_true
    msc.add_node(EndStatechart.when_true(second))
    executor = StatechartExecutor(context=StatechartContext(world=World()))
    msc_copy = Statechart.from_json(
        json.loads(json.dumps(msc.to_json())), context=executor.context
    )
    executor.compile(statechart=msc_copy)
    executor.tick_until_end()
    [state] = [
        state
        for state in (msc_copy.life_cycle_state, msc_copy.observation_state)
        if type(state) is state_type
    ]

    state_copy = state_type.from_json(
        json.loads(json.dumps(state.to_json())), statechart=msc_copy
    )

    assert state_copy == state
