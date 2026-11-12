import pydot
import pytest

from cramph.context import StatechartContext
from cramph.data_types import (
    LifeCyclePredicate,
    LifeCycleValues,
    ObservationPredicate,
    ObservationStateValues,
    TransitionKind,
)
from cramph.node import CancelStatechart, CompositeNode
from cramph.node import EndStatechart, StatechartNode
from cramph.statechart import Statechart
from cramph.nodes_for_testing import (
    ConstTrueNode,
    CompositeNodeWithChainedChildren,
    CompositeNodeWithNestedCompositeChild,
)
from cramph.plotters.graphviz import StatechartGraphviz
from cramph.plotters.styles import (
    DrawingMetrics,
    Font,
    ObservationDrawingStyle,
)
from semantic_digital_twin.world import World

# %% helpers


def expand(statechart: Statechart) -> Statechart:
    """
    Rebuilds the transitions of `statechart`, whose goals expanded when they joined it,
    which is as far as a statechart has to be built to be drawn.
    """
    statechart.compiler.add_transitions()
    return statechart


def build_statechart(goal: CompositeNode) -> Statechart:
    """
    Creates a statechart holding `goal` and an EndStatechart, expanded far enough to be
    drawn.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(goal)
    statechart.add_node(EndStatechart.when_true(goal))
    return expand(statechart)


def draw(statechart: Statechart) -> pydot.Graph:
    """
    :return: The dot graph of `statechart`.
    """
    return StatechartGraphviz(statechart).to_dot_graph()


def direct_node_names(graph: pydot.Graph | pydot.Cluster) -> set[str]:
    """
    :return: The names of the nodes declared on `graph` itself, ignoring its subgraphs.
    """
    return {node.get_name().strip('"') for node in graph.get_nodes()}


def all_node_names(graph: pydot.Graph | pydot.Cluster) -> set[str]:
    """
    :return: The names of the nodes declared anywhere in `graph`, subgraphs included.
    """
    names = direct_node_names(graph)
    for subgraph in graph.get_subgraphs():
        names |= all_node_names(subgraph)
    return names


def all_edge_endpoints(graph: pydot.Graph | pydot.Cluster) -> set[str]:
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


def find_cluster_of(graph: pydot.Graph, node: StatechartNode) -> pydot.Cluster | None:
    """
    :return: The cluster `node` owns as a goal, or None if it has none.
    """
    for subgraph in graph.get_subgraphs():
        if subgraph.get_name().strip('"') == f"cluster_{node.unique_name}":
            return subgraph
    return None


def find_node(graph: pydot.Graph | pydot.Cluster, node: StatechartNode) -> pydot.Node:
    """
    :return: The pydot node drawn for `node`.
    """
    drawn = _drawn_node_or_none(graph, node)
    assert drawn is not None, f"{node.unique_name} was not drawn in {graph.get_name()}"
    return drawn


def _drawn_node_or_none(
    graph: pydot.Graph | pydot.Cluster, node: StatechartNode
) -> pydot.Node | None:
    """
    :return: The pydot node drawn for `node` in `graph` or any of its subgraphs, or None
        if it was not drawn there.
    """
    for candidate in graph.get_nodes():
        if candidate.get_name().strip('"') == node.unique_name:
            return candidate
    for subgraph in graph.get_subgraphs():
        drawn = _drawn_node_or_none(subgraph, node)
        if drawn is not None:
            return drawn
    return None


def find_edges(
    graph: pydot.Graph | pydot.Cluster,
    source: StatechartNode,
    destination: StatechartNode,
) -> list[pydot.Edge]:
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
    source: StatechartNode,
    destination: StatechartNode,
) -> pydot.Edge:
    """
    :return: The single edge drawn from `source` to `destination`.
    """
    edges = find_edges(graph, source, destination)
    assert len(edges) == 1
    return edges[0]


def build_dependency_statechart(
    observed: StatechartNode, owner: StatechartNode
) -> Statechart:
    """
    Creates a statechart holding both nodes, so that `owner` may read `observed` in its
    conditions.

    The caller wires those conditions before drawing.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(observed)
    statechart.add_node(owner)
    return statechart


# %% expanded goals


def test_expanded_goal_draws_children_in_its_cluster() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    statechart = build_statechart(goal)

    cluster = find_cluster_of(draw(statechart), goal)

    assert direct_node_names(cluster) == {
        goal.unique_name,
        goal.sub_node1.unique_name,
        goal.sub_node2.unique_name,
    }


def test_expanded_goal_node_is_declared_only_in_its_cluster() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    statechart = build_statechart(goal)

    graph = draw(statechart)

    assert goal.unique_name not in direct_node_names(graph)


# %% collapsed goals


def test_collapsed_goal_hides_children_and_their_edges() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    goal.plot_specifications.collapse_children = True
    statechart = build_statechart(goal)

    graph = draw(statechart)

    assert find_cluster_of(graph, goal) is None
    assert goal.unique_name in direct_node_names(graph)
    hidden_names = {goal.sub_node1.unique_name, goal.sub_node2.unique_name}
    assert all_node_names(graph) & hidden_names == set()
    assert all_edge_endpoints(graph) & hidden_names == set()


def test_collapsed_goal_reports_hidden_node_count() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    goal.plot_specifications.collapse_children = True
    statechart = build_statechart(goal)

    label = find_node(draw(statechart), goal).get_label()

    assert "2 nodes hidden" in label


def test_collapsed_goal_counts_hidden_nodes_of_nested_goals() -> None:
    goal = CompositeNodeWithNestedCompositeChild(name="goal")
    goal.plot_specifications.collapse_children = True
    statechart = build_statechart(goal)

    label = find_node(draw(statechart), goal).get_label()

    # the inner goal plus the two nodes it expands into
    assert "3 nodes hidden" in label


def test_expanded_goal_reports_no_hidden_node_count() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    statechart = build_statechart(goal)

    cluster = find_cluster_of(draw(statechart), goal)

    assert "hidden" not in find_node(cluster, goal).get_label()


# %% collision avoidance defaults


# %% structure copies


def test_structure_copy_keeps_plot_specs() -> None:
    goal = CompositeNodeWithChainedChildren(name="goal")
    goal.plot_specifications.collapse_children = True
    statechart = build_statechart(goal)

    goal_copy = statechart.create_structure_copy().get_node_by_index(goal.index)

    assert goal_copy.plot_specifications.collapse_children
    assert goal_copy.plot_specifications is not goal.plot_specifications


# %% how far apart an arrow pushes its endpoints


def test_pause_dependency_may_sit_beside_the_node_reading_it() -> None:
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.pause_condition = observed.last_observed_true
    expand(statechart)

    edge = find_edge(draw(statechart), observed, owner)

    assert int(edge.get("minlen")) == DrawingMetrics().minimum_rank_distance(
        TransitionKind.PAUSE
    )


def test_start_dependency_is_drawn_a_row_above_the_node_reading_it() -> None:
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.last_observed_true
    expand(statechart)

    edge = find_edge(draw(statechart), observed, owner)

    assert int(edge.get("minlen")) == DrawingMetrics().minimum_rank_distance(
        TransitionKind.START
    )


def test_merged_arrow_keeps_the_largest_distance_its_conditions_ask_for() -> None:
    """
    A rank distance is a lower bound, so the two arrows this used to draw already left
    graphviz solving for the larger of the two.

    The merged arrow has to state that.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.last_observed_true
    owner.reset_condition = observed.last_observed_true
    expand(statechart)

    edge = find_edge(draw(statechart), observed, owner)

    assert int(edge.get("minlen")) == max(
        DrawingMetrics().minimum_rank_distance(TransitionKind.START),
        DrawingMetrics().minimum_rank_distance(TransitionKind.RESET),
    )


def test_reset_dependency_points_at_the_node_reading_it() -> None:
    """
    A reset arrow runs from the observed node to the node whose reset condition reads
    it, the same way as every other arrow.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.reset_condition = observed.last_observed_true
    expand(statechart)

    graph = draw(statechart)

    assert len(find_edges(graph, observed, owner)) == 1
    assert find_edges(graph, owner, observed) == []


# %% every observation has to be drawable


@pytest.mark.parametrize("observation_state", list(ObservationStateValues))
def test_every_observation_state_has_a_drawing_style(observation_state) -> None:
    """
    The lookup is total, so a state missing from it only shows up as a KeyError while
    drawing an edge.
    """
    assert (
        ObservationDrawingStyle.of_observation(observation_state).name
        == observation_state.name
    )


# %% arrows carry the observation of the node they leave


def test_unknown_arrow_is_thinner_than_a_decided_arrow() -> None:
    unknown = ObservationDrawingStyle.UNKNOWN

    assert unknown.line_width_factor < ObservationDrawingStyle.TRUE.line_width_factor
    assert unknown.line_width_factor < ObservationDrawingStyle.FALSE.line_width_factor


def test_arrow_takes_the_line_width_of_the_observation_it_carries() -> None:
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.last_observed_true
    expand(statechart)
    statechart.observation_state[observed] = ObservationStateValues.UNKNOWN

    edge = find_edge(draw(statechart), observed, owner)

    expected = ObservationDrawingStyle.of_observation(ObservationStateValues.UNKNOWN)
    assert float(edge.get("penwidth")) == expected.line_width(DrawingMetrics())


def test_one_arrow_is_drawn_per_dependency() -> None:
    """
    A node read by two of another node's conditions is reached once, because an arrow no
    longer says which condition reads it.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.success_condition = observed.last_observed_true
    owner.reset_condition = observed.last_observed_true
    expand(statechart)

    assert len(find_edges(draw(statechart), observed, owner)) == 1


# %% condition terms carry the value of the term


def test_condition_term_is_colored_by_the_value_of_the_term() -> None:
    """
    A term is inked by what it evaluates to, not by the observation of the node it
    names:

    `is_succeeded` has no answer until a node is judged, however decisive that node's
    observation already is.
    """
    observed = ConstTrueNode(name="Observed")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.is_succeeded
    expand(statechart)
    statechart.observation_state[observed] = ObservationStateValues.TRUE
    statechart.life_cycle_state[observed] = LifeCycleValues.RUNNING

    label = find_node(draw(statechart), owner).get_label()

    term_value = LifeCyclePredicate.IS_SUCCEEDED.truth_value(LifeCycleValues.RUNNING)
    term_color = ObservationDrawingStyle.of_observation(term_value).color.to_hex()
    term = f"{observed.unique_name}.{LifeCyclePredicate.IS_SUCCEEDED.attribute_name}"
    assert f'<FONT COLOR="{term_color}">"{term}"</FONT>' in label


def test_condition_term_is_colored_when_its_node_name_reads_as_an_operator() -> None:
    """
    A term is found by what it is, not by what its text looks like: a node named after a
    logical operator still gets its own color.
    """
    observed = ConstTrueNode(name="open and close")
    owner = ConstTrueNode(name="Owner")
    statechart = build_dependency_statechart(observed, owner)
    owner.start_condition = observed.last_observed_true
    expand(statechart)
    statechart.observation_state[observed] = ObservationStateValues.TRUE
    statechart.life_cycle_state[observed] = LifeCycleValues.RUNNING

    label = find_node(draw(statechart), owner).get_label()

    term_value = ObservationPredicate.LAST_OBSERVED_TRUE.truth_value(
        observed.last_observation_state
    )
    term_color = ObservationDrawingStyle.of_observation(term_value).color.to_hex()
    term = f"{observed.unique_name}.{ObservationPredicate.LAST_OBSERVED_TRUE.attribute_name}"
    assert f'<FONT COLOR="{term_color}">"{term}"</FONT>' in label


def test_condition_without_terms_is_spelled_out() -> None:
    """
    A constant condition colors True green and False red when active, and grays them out
    when inactive.
    """
    owner = ConstTrueNode(name="Owner")
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(owner)
    expand(statechart)

    true_color = ObservationDrawingStyle.of_observation(
        ObservationStateValues.TRUE
    ).color.to_hex()
    disabled_color = StatechartGraphviz(statechart).disabled_condition_color.to_hex()

    label = find_node(draw(statechart), owner).get_label()

    assert f'start:<FONT COLOR="{true_color}">True</FONT>' in label
    assert (
        f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">pause:False</FONT>'
        in label
    )


def test_condition_boolean_constants_coloring_in_running_state() -> None:
    """
    In RUNNING state, pause:False is active and rendered with False colored red, while
    start:True is inactive and rendered grayed out.
    """
    owner = ConstTrueNode(name="Owner")
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(owner)
    expand(statechart)
    statechart.life_cycle_state[owner] = LifeCycleValues.RUNNING

    false_color = ObservationDrawingStyle.of_observation(
        ObservationStateValues.FALSE
    ).color.to_hex()
    disabled_color = StatechartGraphviz(statechart).disabled_condition_color.to_hex()

    label = find_node(draw(statechart), owner).get_label()

    assert (
        f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">start:True</FONT>'
        in label
    )
    assert f'pause:<FONT COLOR="{false_color}">False</FONT>' in label


# %% extra border styles for terminal nodes


def test_end_statechart_renders_with_rounded_outer_cluster() -> None:
    end = EndStatechart()
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(end)
    expand(statechart)

    graph = draw(statechart)
    cluster = find_cluster_of(graph, end)
    assert cluster is not None
    assert cluster.get("style").strip('"') == "rounded"
    assert cluster.get("color").strip('"') == "black"
    assert float(cluster.get("penwidth")) == DrawingMetrics().line_width
    assert end.unique_name in direct_node_names(cluster)
    assert end.unique_name not in direct_node_names(graph)


def test_cancel_statechart_renders_with_dashed_outer_cluster() -> None:
    cancel = CancelStatechart(exception=Exception("fail"))
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(cancel)
    expand(statechart)

    graph = draw(statechart)
    cluster = find_cluster_of(graph, cancel)
    assert cluster is not None
    assert cluster.get("style").strip('"') == "dashed, rounded"
    assert cluster.get("color").strip('"') == "black"
    assert float(cluster.get("penwidth")) == DrawingMetrics().line_width
    assert cancel.unique_name in direct_node_names(cluster)
    assert cancel.unique_name not in direct_node_names(graph)


def test_edge_targeting_terminal_node_clips_to_outer_cluster() -> None:
    observed = ConstTrueNode(name="Observed")
    end = EndStatechart()
    cancel = CancelStatechart(exception=Exception("fail"))
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_nodes([observed, end, cancel])
    end.start_condition = observed.last_observed_true
    cancel.start_condition = observed.last_observed_true
    expand(statechart)

    graph = draw(statechart)
    end_edge = find_edge(graph, observed, end)
    cancel_edge = find_edge(graph, observed, cancel)

    assert end_edge.get("lhead").strip('"') == f"cluster_{end.unique_name}"
    assert cancel_edge.get("lhead").strip('"') == f"cluster_{cancel.unique_name}"


# %% condition graying based on lifecycle state


@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_conditions_are_grayed_out_when_not_triggerable(
    life_cycle_state: LifeCycleValues,
) -> None:
    """
    Conditions that cannot legally trigger a transition from the node's current
    lifecycle state are rendered in disabled gray font without individual term color
    tags.
    """
    node = ConstTrueNode(name="TestNode")
    observed = ConstTrueNode(name="ObservedNode")
    statechart = build_dependency_statechart(observed, node)
    node.start_condition = observed.last_observed_true
    node.pause_condition = observed.is_succeeded
    node.success_condition = observed.is_failed
    node.fail_condition = observed.is_interrupted
    node.interrupt_condition = observed.is_paused
    node.reset_condition = observed.is_running
    expand(statechart)
    statechart.life_cycle_state[node] = life_cycle_state
    statechart.observation_state[observed] = ObservationStateValues.TRUE

    label = find_node(draw(statechart), node).get_label()
    disabled_color = StatechartGraphviz(statechart).disabled_condition_color.to_hex()

    for condition, term in [
        (
            node._start_condition,
            f"{observed.unique_name}.{ObservationPredicate.LAST_OBSERVED_TRUE.attribute_name}",
        ),
        (
            node._pause_condition,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_SUCCEEDED.attribute_name}",
        ),
        (
            node._success_condition,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_FAILED.attribute_name}",
        ),
        (
            node._fail_condition,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_INTERRUPTED.attribute_name}",
        ),
        (
            node._interrupt_condition,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_PAUSED.attribute_name}",
        ),
        (
            node._reset_condition,
            f"{observed.unique_name}.{LifeCyclePredicate.IS_RUNNING.attribute_name}",
        ),
    ]:
        prefix = StatechartGraphviz.condition_prefix(condition.kind)
        can_trigger = condition.kind.can_trigger_from(life_cycle_state)
        if can_trigger:
            assert f'<FONT FACE="{Font.MONOSPACE}">{prefix}:' in label
            assert (
                f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">{prefix}:'
                not in label
            )
            variable = condition.variables[0]
            resolved_color = ObservationDrawingStyle.of_observation(
                variable.resolve()
            ).color.to_hex()
            assert f'<FONT COLOR="{resolved_color}">"{term}"</FONT>' in label
        else:
            assert (
                f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">{prefix}:'
                in label
            )
            assert (
                f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">{prefix}:"{term}"</FONT>'
                in label
            )


@pytest.mark.parametrize(
    "node_factory",
    [
        lambda: EndStatechart(),
        lambda: CancelStatechart(exception=Exception("test exception")),
    ],
)
@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_terminal_node_condition_graying(
    node_factory, life_cycle_state: LifeCycleValues
) -> None:
    """
    Terminal nodes render only the start condition, grayed out when not in NOT_STARTED.
    """
    terminal_node = node_factory()
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(terminal_node)
    expand(statechart)
    statechart.life_cycle_state[terminal_node] = life_cycle_state

    label = find_node(draw(statechart), terminal_node).get_label()
    disabled_color = StatechartGraphviz(statechart).disabled_condition_color.to_hex()

    can_trigger = TransitionKind.START.can_trigger_from(life_cycle_state)
    if can_trigger:
        assert f'<FONT FACE="{Font.MONOSPACE}">start:' in label
        assert (
            f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">start:'
            not in label
        )
    else:
        assert f'<FONT FACE="{Font.MONOSPACE}" COLOR="{disabled_color}">start:' in label
