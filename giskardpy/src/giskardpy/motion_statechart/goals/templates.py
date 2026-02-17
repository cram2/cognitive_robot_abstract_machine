from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from krrood.symbolic_math.symbolic_math import trinary_logic_and, sum


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Takes a list of nodes and wires their start/end conditions such that they are executed in order.
    Its observation is the observation of the last node in the sequence.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: BuildContext) -> None:
        last_node: Optional[MotionStatechartNode] = None
        for i, node in enumerate(self.nodes):
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.observation_variable
            node.end_condition = node.observation_variable
            last_node = node

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.nodes[-1].observation_variable)


@dataclass(repr=False, eq=False)
class Parallel(Goal):
    """
    Takes a list of nodes and executes them in parallel.
    This nodes' observation state turns True when up to `minimum_success` nodes are True.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    Defines the minimum number of nodes that must be True for the goal to be achieved.
    Defaults to None, which means that all nodes must be True.
    """

    def expand(self, context: BuildContext) -> None:
        for node in self.nodes:
            self.add_node(node)

    def build(self, context: BuildContext) -> NodeArtifacts:
        true_observation_variables = [x.observation_variable == True for x in self.nodes]
        minimum_success = self.minimum_success if self.minimum_success is not None else len(self.nodes)

        return NodeArtifacts(
            observation=minimum_success <= sum(*true_observation_variables)
        )
