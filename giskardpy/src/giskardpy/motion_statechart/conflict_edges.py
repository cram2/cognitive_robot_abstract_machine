"""
Causal edges between runs of motion tasks of different goals that drive a common degree
of freedom at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from cramph.causal_edges import CausalEdgeFinder, CausalEdge, WorldAccessOfNodes
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.goal_concept import HasGoal
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% two tasks driving one degree of freedom at once


@dataclass(frozen=True, kw_only=True)
class ConflictEdge(CausalEdge):
    """
    The cause drove degrees of freedom the effect drives, while both ran.

    It is symmetric and concurrent, so it is kept beside the acyclic causal graph rather
    than in it.
    """

    degrees_of_freedom: frozenset[DegreeOfFreedom]
    """
    The degrees of freedom both drove at the same time.
    """

    def describe_relation(self) -> str:
        return (
            f"drove ({self.join_sorted_names(entities=self.degrees_of_freedom)}) at "
            f"the same time as"
        )


@dataclass(kw_only=True)
class ConflictEdges(CausalEdgeFinder):
    """
    Two runs of tasks of different goals that drive a common degree of freedom while
    both run act on each other.
    """

    def find_edges(
        self, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> list[CausalEdge]:
        drivers = [
            run
            for run in runs.all_runs
            if access[run.node].moved_or_held_degrees_of_freedom
        ]
        edges = []
        for first, second in combinations(drivers, 2):
            if self.outermost_goal_node_of(
                node=first.node
            ) is self.outermost_goal_node_of(node=second.node) or not first.overlaps(
                other=second
            ):
                continue
            shared = (
                access[first.node].moved_or_held_degrees_of_freedom
                & access[second.node].moved_or_held_degrees_of_freedom
            )
            if not shared:
                continue
            edges.append(
                ConflictEdge(cause=first, effect=second, degrees_of_freedom=shared)
            )
            edges.append(
                ConflictEdge(cause=second, effect=first, degrees_of_freedom=shared)
            )
        return edges

    def outermost_goal_node_of(self, *, node: StatechartNode) -> StatechartNode:
        """
        :param node: A node.
        :return: The outermost node above `node` that :meth:`belongs_to_a_goal`, the
            node itself if there is none: the goal its task works towards together with
            the other tasks of that goal.
        """
        goal = node
        for ancestor in node.path:
            if not self.belongs_to_a_goal(node=ancestor):
                break
            goal = ancestor
        return goal

    def belongs_to_a_goal(self, *, node: StatechartNode) -> bool:
        """
        :param node: A node above a task.
        :return: Whether `node` is a motion node or takes a goal, and so belongs to the
            goal of the tasks below it.
        """
        return isinstance(node, (MotionStatechartNode, HasGoal))
