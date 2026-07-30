from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import trinary_logic_or, trinary_logic_not
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)


@dataclass(repr=False, eq=False)
class TryAll(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    Its observation turns True as soon as any node is True and turns False only when all
    nodes are False, i.e. it only fails if every node fails.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes executed in parallel.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add all child nodes to this goal so they run in parallel.
        """
        for node in self.nodes:
            self.add_node(node)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node is True.
        """
        observations = [node.observation_variable for node in self.nodes]
        observation = (
            observations[0]
            if len(observations) == 1
            else trinary_logic_or(*observations)
        )
        return NodeArtifacts(observation=observation)


@dataclass(repr=False, eq=False)
class TryInOrder(Goal):
    """
    Takes a list of nodes and tries them one after another, short-circuiting on the
    first success.

    The next node only starts once the previous node failed; as soon as a node succeeds
    the remaining nodes are skipped. Its observation turns True if any node is True and
    turns False only when all nodes are False, i.e. it only fails if every node fails.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the child nodes and wire them so each one starts only after the previous one
        failed, short-circuiting on the first success.
        """
        last_node: Optional[MotionStatechartNode] = None
        for node in self.nodes:
            self.add_node(node)
            if last_node is not None:
                # Start the next node only if the previous one failed (short-circuit on success).
                node.start_condition = trinary_logic_not(last_node.observation_variable)
            # End this node as soon as it is decided (True or False) so the chain can advance/finish.
            node.end_condition = trinary_logic_or(
                node.observation_variable, trinary_logic_not(node.observation_variable)
            )
            last_node = node

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node is True.
        """
        observations = [node.observation_variable for node in self.nodes]
        observation = (
            observations[0]
            if len(observations) == 1
            else trinary_logic_or(*observations)
        )
        return NodeArtifacts(observation=observation)
