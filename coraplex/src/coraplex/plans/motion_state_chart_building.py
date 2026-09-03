from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from itertools import groupby

from typing_extensions import TYPE_CHECKING, List

from coraplex.plans.executables import Executable, GiskardExecutable
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Goal, MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from coraplex.plans.plan_node import PlanNode


@dataclass(eq=False)
class BuildsMotionStateChart:
    """
    Mixin class for the plan nodes that own a goal in a giskard motion state chart.

    ..note:: Mix this in as the *last* base. ORMatic resolves a data access object's
        parent by walking the method resolution order and taking the first mapped class,
        so a holder that lists this mixin first drops its plan node ancestry, and the
        columns that come with it, from the generated inheritance chain.
    """

    # %% what the plan node provides, mostly here for typing

    @property
    @abstractmethod
    def context(self) -> Context:
        """
        :return: The context the executables built here are executed in.
        """

    @property
    @abstractmethod
    def has_motions(self) -> bool:
        """
        :return: Whether this contributes any node to a motion state chart.
        """

    @property
    @abstractmethod
    def contains_execution_boundary(self) -> bool:
        """
        :return: Whether this splits the plan into separate motion state charts.
        """

    @abstractmethod
    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> MotionStatechartNode:
        """
        Add this node's giskard representation to `parent_goal`, extending the motion
        state chart the goal belongs to.

        :param parent_goal: The goal this node's representation becomes a child of.
        :param executable: The executable whose motion mappings record the added nodes.
        :return: The node that was added.
        """

    # %% building the chart

    def create_goal(self) -> Goal:
        """
        :return: An empty goal describing how the children are executed.
        """
        return Sequence(name=type(self).__name__)

    def add_children_to_motion_state_chart(
        self,
        goal: Goal,
        children: List[BuildsMotionStateChart],
        executable: GiskardExecutable,
    ) -> None:
        """
        Add every child that contributes motions below `goal`, one node at a time.

        Children without motions are skipped, so they cannot leave an empty goal behind.
        """
        for child in children:
            if not child.has_motions:
                continue
            child.add_to_motion_state_chart(goal, executable)

    def parse_children(self, children: List[PlanNode]) -> Executable:
        """
        Parse `children` into one executable, building one motion state chart per run of
        children that no execution boundary separates.

        A child whose subtree contains an execution boundary is parsed on its own, so
        every executable preceding it has run, and mutated the world, before it is
        reached. Only the children of a run that no boundary separates build a chart,
        which is why they alone are handed to :meth:`create_giskard_executable`.
        """
        if not self.contains_execution_boundary:
            return self.create_giskard_executable(children)

        execution_list = []
        for separated, group in groupby(
            children, key=lambda child: child.contains_execution_boundary
        ):
            if separated:
                execution_list.extend(child.parse() for child in group)
            else:
                execution_list.append(self.create_giskard_executable(list(group)))
        if len(execution_list) == 1:
            # a single group needs no wrapper around it
            return execution_list[0]
        return Executable(execution_list=execution_list, context=self.context)

    def create_giskard_executable(
        self, nodes: List[BuildsMotionStateChart]
    ) -> GiskardExecutable:
        """
        Create an executable holding a single motion state chart, populated by adding
        the giskard representation of every node in `nodes` one at a time.

        The chart is created once here and only ever extended afterwards, because a
        compiled chart can no longer grow.

        :param nodes: The nodes whose motions form one motion state chart.
        """
        motion_state_chart = MotionStatechart()
        root_goal = self.create_goal()
        motion_state_chart.add_node(root_goal)
        executable = GiskardExecutable(
            motion_state_chart=motion_state_chart,
            root_node=root_goal,
            context=self.context,
        )
        self.add_children_to_motion_state_chart(root_goal, nodes, executable)
        return executable
