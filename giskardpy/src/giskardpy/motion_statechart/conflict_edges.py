"""
Causal edges between runs of motion tasks of different goals that drive a common degree
of freedom at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from cramph.causal_edges import CausalEdgeSource, RunEdge, WorldAccessIndex
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.mixins import (
    HasGoalOrientationParameter,
    HasGoalPointParameter,
    HasGoalPointsParameter,
    HasGoalPoseParameter,
    HasGoalStateParameter,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% two tasks driving one degree of freedom at once


@dataclass(frozen=True, kw_only=True)
class ConflictEdge(RunEdge):
    """
    The cause drove degrees of freedom the effect drives, while both ran.

    It is symmetric and concurrent, so it is kept beside the acyclic causal graph rather
    than in it.
    """

    degrees_of_freedom: frozenset[DegreeOfFreedom]
    """
    The degrees of freedom both drove at the same time.
    """

    def describe(self) -> str:
        return (
            f"drove ({self.names_of(entities=self.degrees_of_freedom)}) at the same "
            f"time as"
        )


@dataclass(kw_only=True)
class ConflictEdges(CausalEdgeSource):
    """
    Two runs of tasks of different goals that drive a common degree of freedom while
    both run act on each other.
    """

    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        drivers = [run for run in runs.all_runs if access[run.node].driven]
        edges = []
        for first, second in combinations(drivers, 2):
            if self.goal_of(node=first.node) is self.goal_of(
                node=second.node
            ) or not first.overlaps(other=second):
                continue
            shared = access[first.node].driven & access[second.node].driven
            if not shared:
                continue
            edges.append(
                ConflictEdge(cause=first, effect=second, degrees_of_freedom=shared)
            )
            edges.append(
                ConflictEdge(cause=second, effect=first, degrees_of_freedom=shared)
            )
        return edges

    def goal_of(self, *, node: StatechartNode) -> StatechartNode:
        """
        :param node: A node.
        :return: The outermost node above `node` that :meth:`is_part_of_a_goal`, the
            node itself if there is none: the goal its task works towards together with
            the other tasks of that goal.
        """
        goal = node
        for ancestor in node.path:
            if not self.is_part_of_a_goal(node=ancestor):
                break
            goal = ancestor
        return goal

    def is_part_of_a_goal(self, *, node: StatechartNode) -> bool:
        """
        :param node: A node above a task.
        :return: Whether `node` is a motion node or takes a goal, and so belongs to the
            goal of the tasks below it.
        """
        return isinstance(
            node,
            (
                MotionStatechartNode,
                HasGoalPoseParameter,
                HasGoalPointParameter,
                HasGoalPointsParameter,
                HasGoalOrientationParameter,
                HasGoalStateParameter,
            ),
        )
