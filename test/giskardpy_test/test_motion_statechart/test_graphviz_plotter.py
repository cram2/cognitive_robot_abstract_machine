import pydot
from typing_extensions import Optional, Set, Union

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    Goal,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    TestGoal,
    TestNestedGoal,
)
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz

# %% helpers


def build_motion_statechart(goal: Goal) -> MotionStatechart:
    """
    Creates a motion statechart holding `goal` and an end motion, expanded far enough to
    be drawn.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    motion_statechart._expand_goals(MotionStatechartContext.empty())
    motion_statechart._add_transitions()
    return motion_statechart


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


def find_node(graph: pydot.Graph, node: MotionStatechartNode) -> pydot.Node:
    """
    :return: The pydot node drawn for `node`.
    """
    for candidate in graph.get_nodes():
        if candidate.get_name().strip('"') == node.unique_name:
            return candidate
    raise AssertionError(f"{node.unique_name} was not drawn in {graph.get_name()}")


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
    assert SelfCollisionAvoidance().plot_specs.collapse_children


# %% structure copies


def test_structure_copy_keeps_plot_specs():
    goal = TestGoal(name="goal")
    goal.plot_specifications.collapse_children = True
    motion_statechart = build_motion_statechart(goal)

    goal_copy = motion_statechart.create_structure_copy().get_node_by_index(goal.index)

    assert goal_copy.plot_specifications.collapse_children
    assert goal_copy.plot_specifications is not goal.plot_specifications
