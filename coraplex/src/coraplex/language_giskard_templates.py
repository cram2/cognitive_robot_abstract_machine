from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import (
    Scalar,
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.progress_monitors import (
    DEFAULT_STALL_TIMEOUT,
    StillProgressing,
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
        self._check_has_children()
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node reached its goal.

        This goal ends none of its children, so a child that keeps running is judged by
        what it observes now rather than by a verdict it never reaches.
        """
        return NodeArtifacts(
            observation=trinary_logic_or(*[node.goal_reached for node in self.nodes]),
        )


@dataclass(repr=False, eq=False)
class TryInOrder(Goal):
    """
    Takes a list of nodes and tries them one after another, short-circuiting on the
    first success.

    The next alternative only starts once the previous one has ended without reaching
    its goal, not merely while it is still short of it. Its observation turns True as
    soon as an alternative succeeds and False only once every one of them is over, so it
    stays unknown while any of them is still being tried.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.
    """

    _alternatives: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """
    The nodes that were passed in, captured before :meth:`expand` adds a progress
    monitor per alternative to :attr:`nodes` alongside them.
    """

    give_up_after: float = field(default=DEFAULT_STALL_TIMEOUT, kw_only=True)
    """
    Seconds of simulated time an alternative may make no progress before it is abandoned
    and the next one is tried.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the child nodes and wire them so each one starts only after the previous one
        ended without reaching its goal, short-circuiting on the first success.

        An alternative is ended once it reaches its goal or once it stops making
        progress; which of the two happened is decided by what the alternative observes
        as it ends, not here. An observation that is merely still false means the
        alternative has not arrived yet, and is no reason to abandon it.

        A progress monitor ends with the alternative it watches, so it does not go on
        measuring progress against a node that has already been decided.
        """
        self._check_has_children()
        self._alternatives = list(self.nodes)
        previous_node: Optional[MotionStatechartNode] = None
        for node in self._alternatives:
            self.add_node(node)
            if previous_node is not None:
                node.start_condition = self._ended_without_succeeding(previous_node)
            still_progressing = StillProgressing(
                name=f"{self.name}/progress_of_{node.name}",
                monitored_node=node,
                timeout=self.give_up_after,
            )
            self.add_node(still_progressing)
            still_progressing.start_condition = node.is_running
            still_progressing.end_condition = node.is_terminated
            node.end_condition = trinary_logic_or(
                node.observation_variable,
                trinary_logic_not(still_progressing.observation_variable),
            )
            previous_node = node

    @staticmethod
    def _ended_without_succeeding(node: MotionStatechartNode) -> Scalar:
        """
        An alternative abandoned while it observed nothing decisive is of no more use
        than one that failed outright, so both count as ended without success.

        :param node: The alternative to judge.
        :return: True once that alternative ended anywhere but at its goal.
        """
        return trinary_logic_or(node.is_failed, node.is_interrupted)

    @staticmethod
    def _reached_its_goal(node: MotionStatechartNode) -> Scalar:
        """
        Whether an alternative is at its goal, answering false rather than unknown for
        one that was abandoned before it ever got there.

        .. note:: An observation may not read a life cycle predicate, so being abandoned
            is read off the life cycle state itself.

        :param node: The alternative to read.
        :return: What that alternative observes while it runs, and whether it reached
            its goal once it has ended.
        """
        was_abandoned = Scalar(
            node.life_cycle_variable == int(LifeCycleValues.INTERRUPTED)
        )
        return trinary_logic_and(node.goal_reached, trinary_logic_not(was_abandoned))

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any alternative succeeded, and
        False only once every one of them is over without one having succeeded.
        """
        return NodeArtifacts(
            observation=trinary_logic_or(
                *[self._reached_its_goal(node) for node in self._alternatives]
            ),
        )
