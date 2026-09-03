import pydot
import pytest
from typing_extensions import List, Optional, Set, Union

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCyclePredicate,
    LifeCycleValues,
    ObservationStateValues,
    TransitionKind,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    EndMotion,
    Goal,
    GoalReachedVariable,
    MotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
    TestGoal,
    TestNestedGoal,
)
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from giskardpy.motion_statechart.plotters.styles import (
    MINIMUM_RANK_DISTANCES,
    OBSERVATION_DRAWING_STYLES,
    DisabledConditionColor,
    LineWidth,
    TaskShape,
    TaskStyle,
)

# %% helpers


def expand(motion_statechart: MotionStatechart) -> MotionStatechart:
    """
    Expands the goals of `motion_statechart` and rebuilds its transitions, which is as
    far as a statechart has to be built to be drawn.
    """
    motion_statechart._expand_goals(MotionStatechartContext.empty())
    motion_statechart._add_transitions()
    return motion_statechart


def build_motion_statechart(goal: Goal) -> MotionStatechart:
    """
    Creates a motion statechart holding `goal` and an end motion, expanded far enough to
    be drawn.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    return expand(motion_statechart)


def draw(motion_statechart: MotionStatechart) -> pydot.Graph:
    """
    :return: The dot graph of `motion_statechart`.
    """
    return MotionStatechartGraphviz(motion_statechart).to_dot_graph()


def direct_node_names(graph: Union[pydot.Graph, pydot.Cluster]) -> Set[str]:
    """
    :return: The names of the nodes declared on `graph` itself, ignoring its subgraphs.
    """
    return {node.get_name().strip('"') for node in graph.get_nodes()}


def all_node_names(graph: Union[pydot.Graph, pydot.Cluster]) -> Set[str]:
    """
    :return: The names of the nodes declared anywhere in `graph`, subgraphs included.
    """
    names = direct_node_names(graph)
    for subgraph in graph.get_subgraphs():
        names |= all_node_names(subgraph)
    return names


def all_edge_endpoints(graph: Union[pydot.Graph, pydot.Cluster]) -> Set[str]:
    """
    :return: The source and destination names of every edge in `graph`, subgraphs included.
    """
    endpoints = set()
    for edge in graph.get_edges():
        endpoints.add(edge.get_source().strip('"'))
        endpoints.add(edge.get_destination().strip('"'))
    for subgraph in graph.get_subgraphs():
        endpoints |= all_edge_endpoints(subgraph)
    return endpoints


def find_cluster_of(
    graph: pydot.Graph, node: MotionStatechartNode
) -> Optional[pydot.Cluster]:
    """
    :return: The cluster `node` owns as a goal, or None if it has none.
    """
    for subgraph in graph.get_subgraphs():
        if subgraph.get_name().strip('"') == f"cluster_{node.unique_name}":
            return subgraph
    return None


def find_node(
    graph: Union[pydot.Graph, pydot.Cluster], node: MotionStatechartNode
) -> pydot.Node:
    """
    :return: The pydot node drawn for `node`.
    """
    for candidate in graph.get_nodes():
        if candidate.get_name().strip('"') == node.unique_name:
            return candidate
    for subgraph in graph.get_subgraphs():
        try:
            return find_node(subgraph, node)
        except AssertionError:
            continue
    raise AssertionError(f"{node.unique_name} was not drawn in {graph.get_name()}")


def find_edges(
    graph: Union[pydot.Graph, pydot.Cluster],
    source: MotionStatechartNode,
    destination: MotionStatechartNode,
) -> List[pydot.Edge]:
    """
    :return: Every edge drawn from `source` to `destination`, searching subgraphs too,
        because an edge between two children of a goal is declared on that goal's cluster.
    """
    edges = [
        edge
        for edge in graph.get_edges()
        if edge.get_source().strip('"') == source.unique_name
        and edge.get_destination().strip('"') == destination.unique_name
    ]
    for subgraph in graph.get_subgraphs():
        edges += find_edges(subgraph, source, destination)
    return edges


def find_edge(
    graph: pydot.Graph,
    source: MotionStatechartNode,
    destination: MotionStatechartNode,
) -> pydot.Edge:
    """
    :return: The single edge drawn from `source` to `destination`.
    """
    edges = find_edges(graph, source, destination)
    assert len(edges) == 1
    return edges[0]


def build_dependency_statechart(
    observed: MotionStatechartNode, owner: MotionStatechartNode
) -> MotionStatechart:
    """
    Creates a statechart holding both nodes, so that `owner` may read `observed` in its
    conditions.

    The caller wires those conditions before drawing.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(observed)
    motion_statechart.add_node(owner)
    return motion_statechart


# %% expanded goals


def test_expanded_goal_draws_children_in_its_cluster():
    goal = TestGoal(name="goal")
    motion_statechart = build_motion_statechart(goal)

    cluster = find_cluster_of(draw(motion_statechart), goal)

    assert direct_node_names(cluster) == {
        goal.unique_name,
        goal.sub_node1.unique_name,
        goal.sub_node2.unique_name,
    }


def test_expanded_goal_node_is_declared_only_in_its_cluster():
    goal = TestGoal(name="goal")
    motion_statechart = build_motion_statechart(goal)

    graph = draw(motion_statechart)

    assert goal.unique_name not in direct_node_names(graph)


# %% collapsed goals


def test_collapsed_goal_hides_children_and_their_edges():
    goal = TestGoal(name="goal")
    goal.plot_specifications.collapse_children = True
    motion_statechart = build_motion_statechart(goal)

    graph = draw(motion_statechart)

    assert find_cluster_of(graph, goal) is None
    assert goal.unique_name in direct_node_names(graph)
    hidden_names = {goal.sub_node1.unique_name, goal.sub_node2.unique_name}
    assert all_node_names(graph) & hidden_names == set()
    assert all_edge_endpoints(graph) & hidden_names == set()


def test_collapsed_goal_reports_hidden_node_count():
    goal = TestGoal(name="goal")
    goal.plot_specifications.collapse_children = True
    motion_statechart = build_motion_statechart(goal)

    label = find_node(draw(motion_statechart), goal).get_label()

    assert "2 nodes hidden" in label


def test_collapsed_goal_counts_hidden_nodes_of_nested_goals():
    goal = TestNestedGoal(name="goal")
    goal.plot_specifications.collapse_children = True
    motion_statechart = build_motion_statechart(goal)

    label = find_node(draw(motion_statechart), goal).get_label()

    # the inner goal plus the two nodes it expands into
    assert "3 nodes hidden" in label


def test_expanded_goal_reports_no_hidden_node_count():
    goal = TestGoal(name="goal")
    motion_statechart = build_motion_statechart(goal)

    cluster = find_cluster_of(draw(motion_statechart), goal)

    assert "hidden" not in find_node(cluster, goal).get_label()


# %% collision avoidance defaults


def test_collision_avoidance_goals_collapse_their_children():
    assert ExternalCollisionAvoidance().plot_specifications.collapse_children
    assert SelfCollisionAvoidance().plot_specifications.collapse_children


# %% structure copies


def test_structure_copy_keeps_plot_specs():
    goal = TestGoal(name="goal")
    goal.plot_specifications.collapse_children = True
    motion_statechart = build_motion_statechart(goal)

    goal_copy = motion_statechart.create_structure_copy().get_node_by_index(goal.index)

    assert goal_copy.plot_specifications.collapse_children
    assert goal_copy.plot_specifications is not goal.plot_specifications


# %% how far apart an arrow pushes its endpoints


@pytest.mark.parametrize("transition_kind", list(TransitionKind))
def test_every_transition_kind_has_a_minimum_rank_distance(transition_kind):
    """
    The lookup is total, so a kind missing from it only shows up as a KeyError while
    drawing an arrow.
    """
    assert transition_kind in MINIMUM_RANK_DISTANCES


def test_pause_dependency_may_sit_beside_the_node_reading_it():
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.pause_condition = observed.goal_reached
    expand(motion_statechart)

    edge = find_edge(draw(motion_statechart), observed, owner)

    assert int(edge.get("minlen")) == MINIMUM_RANK_DISTANCES[TransitionKind.PAUSE]


def test_start_dependency_is_drawn_a_row_above_the_node_reading_it():
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.goal_reached
    expand(motion_statechart)

    edge = find_edge(draw(motion_statechart), observed, owner)

    assert int(edge.get("minlen")) == MINIMUM_RANK_DISTANCES[TransitionKind.START]


def test_merged_arrow_keeps_the_largest_distance_its_conditions_ask_for():
    """
    A rank distance is a lower bound, so the two arrows this used to draw already left
    graphviz solving for the larger of the two.

    The merged arrow has to state that.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.goal_reached
    owner.reset_condition = observed.goal_reached
    expand(motion_statechart)

    edge = find_edge(draw(motion_statechart), observed, owner)

    assert int(edge.get("minlen")) == max(
        MINIMUM_RANK_DISTANCES[TransitionKind.START],
        MINIMUM_RANK_DISTANCES[TransitionKind.RESET],
    )


def test_reset_dependency_points_at_the_node_reading_it():
    """
    A reset arrow used to be emitted backwards and turned around with `arrowtail`; it
    now runs the same way as every other arrow.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.reset_condition = observed.goal_reached
    expand(motion_statechart)

    graph = draw(motion_statechart)

    assert len(find_edges(graph, observed, owner)) == 1
    assert find_edges(graph, owner, observed) == []


# %% every observation has to be drawable


@pytest.mark.parametrize("observation_state", list(ObservationStateValues))
def test_every_observation_state_has_a_drawing_style(observation_state):
    """
    The lookup is total, so a state missing from it only shows up as a KeyError while
    drawing an edge.
    """
    assert observation_state in OBSERVATION_DRAWING_STYLES


# %% arrows carry the observation of the node they leave


def test_arrow_takes_the_observation_color_of_the_node_it_leaves():
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.goal_reached
    expand(motion_statechart)
    motion_statechart.observation_state[observed] = ObservationStateValues.FALSE
    motion_statechart.observation_state[owner] = ObservationStateValues.TRUE

    edge = find_edge(draw(motion_statechart), observed, owner)

    expected = OBSERVATION_DRAWING_STYLES[ObservationStateValues.FALSE]
    assert edge.get("color") == expected.color.to_hex()


def test_unknown_arrow_is_thinner_than_a_decided_arrow():
    unknown = OBSERVATION_DRAWING_STYLES[ObservationStateValues.UNKNOWN]

    assert (
        unknown.line_width
        < OBSERVATION_DRAWING_STYLES[ObservationStateValues.TRUE].line_width
    )
    assert (
        unknown.line_width
        < OBSERVATION_DRAWING_STYLES[ObservationStateValues.FALSE].line_width
    )


def test_arrow_takes_the_line_width_of_the_observation_it_carries():
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.goal_reached
    expand(motion_statechart)
    motion_statechart.observation_state[observed] = ObservationStateValues.UNKNOWN

    edge = find_edge(draw(motion_statechart), observed, owner)

    expected = OBSERVATION_DRAWING_STYLES[ObservationStateValues.UNKNOWN]
    assert float(edge.get("penwidth")) == expected.line_width


def test_one_arrow_is_drawn_per_dependency():
    """
    A node read by two of another node's conditions is reached once, because an arrow no
    longer says which condition reads it.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.end_condition = observed.goal_reached
    owner.reset_condition = observed.goal_reached
    expand(motion_statechart)

    assert len(find_edges(draw(motion_statechart), observed, owner)) == 1


# %% condition terms carry the value of the term


def test_condition_term_is_colored_by_the_value_of_the_term():
    """
    A term is inked by what it evaluates to, not by the observation of the node it
    names:

    `is_succeeded` has no answer until a node is judged, however decisive that node's
    observation already is.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.is_succeeded
    expand(motion_statechart)
    motion_statechart.observation_state[observed] = ObservationStateValues.TRUE
    motion_statechart.life_cycle_state[observed] = LifeCycleValues.RUNNING

    label = find_node(draw(motion_statechart), owner).get_label()

    term_value = LifeCyclePredicate.IS_SUCCEEDED.value.truth_value(
        LifeCycleValues.RUNNING
    )
    term_color = OBSERVATION_DRAWING_STYLES[term_value].color.to_hex()
    term = f"{observed.unique_name}.{LifeCyclePredicate.IS_SUCCEEDED.attribute_name}"
    assert f'<FONT COLOR="{term_color}">"{term}"</FONT>' in label


def test_condition_term_is_colored_when_its_node_name_reads_as_an_operator():
    """
    A term is found by what it is, not by what its text looks like: a node named after a
    logical operator still gets its own color.
    """
    observed = ConstTrueNode(name="open and close")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.goal_reached
    expand(motion_statechart)
    motion_statechart.observation_state[observed] = ObservationStateValues.TRUE
    motion_statechart.life_cycle_state[observed] = LifeCycleValues.RUNNING

    label = find_node(draw(motion_statechart), owner).get_label()

    term_color = OBSERVATION_DRAWING_STYLES[observed.goal_reached_state].color.to_hex()
    term = f"{observed.unique_name}.{GoalReachedVariable.attribute_name}"
    assert f'<FONT COLOR="{term_color}">"{term}"</FONT>' in label


def test_condition_without_terms_is_spelled_out():
    """
    A constant condition has nothing to color, and still has to render.
    """
    owner = ConstTrueNode(name="Owner")
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(owner)
    expand(motion_statechart)

    label = find_node(draw(motion_statechart), owner).get_label()

    assert "start:True" in label
    assert "pause:False" in label


def test_condition_term_and_the_arrow_it_feeds_can_differ():
    """
    The same statechart as above seen from the arrow: it reports what the node observes,
    while the term reports what the condition gets to read.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    motion_statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.is_succeeded
    expand(motion_statechart)
    motion_statechart.observation_state[observed] = ObservationStateValues.TRUE
    motion_statechart.life_cycle_state[observed] = LifeCycleValues.RUNNING

    edge = find_edge(draw(motion_statechart), observed, owner)

    assert (
        edge.get("color")
        == OBSERVATION_DRAWING_STYLES[ObservationStateValues.TRUE].color.to_hex()
    )


# %% extra border styles for terminal nodes


def test_terminal_nodes_and_tasks_have_correct_plot_specifications():
    assert EndMotion().plot_specifications.extra_border_styles == ["rounded"]
    assert CancelMotion(
        exception=Exception()
    ).plot_specifications.extra_border_styles == ["dashed, rounded"]
    assert Task().plot_specifications.style == TaskStyle
    assert Task().plot_specifications.shape == TaskShape


def test_end_motion_renders_with_rounded_outer_cluster():
    end = EndMotion()
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(end)
    expand(motion_statechart)

    graph = draw(motion_statechart)
    cluster = find_cluster_of(graph, end)
    assert cluster is not None
    assert cluster.get("style").strip('"') == "rounded"
    assert cluster.get("color").strip('"') == "black"
    assert float(cluster.get("penwidth")) == LineWidth
    assert end.unique_name in direct_node_names(cluster)
    assert end.unique_name not in direct_node_names(graph)


def test_cancel_motion_renders_with_dashed_outer_cluster():
    cancel = CancelMotion(exception=Exception("fail"))
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(cancel)
    expand(motion_statechart)

    graph = draw(motion_statechart)
    cluster = find_cluster_of(graph, cancel)
    assert cluster is not None
    assert cluster.get("style").strip('"') == "dashed, rounded"
    assert cluster.get("color").strip('"') == "black"
    assert float(cluster.get("penwidth")) == LineWidth
    assert cancel.unique_name in direct_node_names(cluster)
    assert cancel.unique_name not in direct_node_names(graph)


def test_edge_targeting_terminal_node_clips_to_outer_cluster():
    observed = ConstTrueNode(name="Observed")
    end = EndMotion()
    cancel = CancelMotion(exception=Exception("fail"))
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([observed, end, cancel])
    end.start_condition = observed.goal_reached
    cancel.start_condition = observed.goal_reached
    expand(motion_statechart)

    graph = draw(motion_statechart)
    end_edge = find_edge(graph, observed, end)
    cancel_edge = find_edge(graph, observed, cancel)

    assert end_edge.get("lhead").strip('"') == f"cluster_{end.unique_name}"
    assert cancel_edge.get("lhead").strip('"') == f"cluster_{cancel.unique_name}"


# %% condition graying based on lifecycle state


@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_conditions_are_grayed_out_when_not_triggerable(
    life_cycle_state: LifeCycleValues,
):
    """
    Conditions that cannot legally trigger a transition from the node's current
    lifecycle state are rendered in disabled gray font without individual term color
    tags.
    """
    node = ConstTrueNode(name="TestNode")
    observed = ConstTrueNode(name="ObservedNode")
    motion_statechart = build_dependency_statechart(observed, node)
    node.start_condition = observed.goal_reached
    node.pause_condition = observed.is_succeeded
    node.end_condition = observed.is_failed
    node.reset_condition = observed.is_running
    expand(motion_statechart)
    motion_statechart.life_cycle_state[node] = life_cycle_state
    motion_statechart.observation_state[observed] = ObservationStateValues.TRUE

    label = find_node(draw(motion_statechart), node).get_label()
    disabled_color = DisabledConditionColor.to_hex()

    for prefix, condition, transition_kind, term in [
        (
            "start",
            node._start_condition,
            TransitionKind.START,
            f"{observed.unique_name}.{GoalReachedVariable.attribute_name}",
        ),
        (
            "pause",
            node._pause_condition,
            TransitionKind.PAUSE,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_SUCCEEDED.attribute_name}",
        ),
        (
            "end  ",
            node._end_condition,
            TransitionKind.END,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_FAILED.attribute_name}",
        ),
        (
            "reset",
            node._reset_condition,
            TransitionKind.RESET,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_RUNNING.attribute_name}",
        ),
    ]:
        can_trigger = transition_kind.can_trigger_from(life_cycle_state)
        if can_trigger:
            assert f'<FONT FACE="monospace">{prefix}:' in label
            assert (
                f'<FONT FACE="monospace" COLOR="{disabled_color}">{prefix}:'
                not in label
            )
            variable = condition.variables[0]
            resolved_color = OBSERVATION_DRAWING_STYLES[
                variable.resolve()
            ].color.to_hex()
            assert f'<FONT COLOR="{resolved_color}">"{term}"</FONT>' in label
        else:
            assert f'<FONT FACE="monospace" COLOR="{disabled_color}">{prefix}:' in label
            assert (
                f'<FONT FACE="monospace" COLOR="{disabled_color}">{prefix}:"{term}"</FONT>'
                in label
            )


@pytest.mark.parametrize(
    "node_factory",
    [
        lambda: EndMotion(),
        lambda: CancelMotion(exception=Exception("test exception")),
    ],
)
@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_terminal_node_condition_graying(
    node_factory, life_cycle_state: LifeCycleValues
):
    """
    Terminal nodes render only the start condition, grayed out when not in NOT_STARTED.
    """
    terminal_node = node_factory()
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(terminal_node)
    expand(motion_statechart)
    motion_statechart.life_cycle_state[terminal_node] = life_cycle_state

    label = find_node(draw(motion_statechart), terminal_node).get_label()
    disabled_color = DisabledConditionColor.to_hex()

    can_trigger = TransitionKind.START.can_trigger_from(life_cycle_state)
    if can_trigger:
        assert '<FONT FACE="monospace">start:' in label
        assert f'<FONT FACE="monospace" COLOR="{disabled_color}">start:' not in label
    else:
        assert f'<FONT FACE="monospace" COLOR="{disabled_color}">start:' in label
