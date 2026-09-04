"""
Tests for :class:`~coraplex.plans.motion_state_chart_building.BuildsMotionStateChart`,
the mixin the goal-owning plan nodes use to build a giskard motion state chart.

The builder below fills the hooks a plan node supplies, so a chart can be built without
standing up a plan tree around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.language import LanguageNode
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.motion_state_chart_building import BuildsMotionStateChart
from coraplex.plans.plan_node import (
    ActionNode,
    MotionNode,
    PlanNode,
)
from coraplex.plans.underspecified import UnderspecifiedNode
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Goal

# %% a chart builder that is not a plan node


@dataclass
class ChartBuilderWithoutPlan(BuildsMotionStateChart):
    """
    Builds a motion state chart while living outside any plan, contributing a goal of
    its own whenever it is added to one.
    """

    _context: Context
    """
    The context handed to the executables this builder creates.
    """

    contributes_motions: bool = True
    """
    Whether this builder claims to contribute motions.
    """

    added_goal: Optional[Goal] = field(default=None, init=False)
    """
    The goal this builder added to a chart, None until it was added to one.
    """

    @property
    def context(self) -> Context:
        return self._context

    @property
    def has_motions(self) -> bool:
        return self.contributes_motions

    @property
    def contains_execution_boundary(self) -> bool:
        return False

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Goal:
        self.added_goal = self.create_goal()
        parent_goal.add_node(self.added_goal)
        return self.added_goal


# %% which nodes build charts


def test_only_goal_owning_nodes_build_charts():
    """
    Chart building sits on the nodes that own a goal, so the rest of the plan tree does
    not carry methods it never calls.
    """
    assert issubclass(ActionNode, BuildsMotionStateChart)
    assert issubclass(MotionNode, BuildsMotionStateChart)
    assert issubclass(LanguageNode, BuildsMotionStateChart)

    assert not issubclass(PlanNode, BuildsMotionStateChart)
    assert not issubclass(ConditionNode, BuildsMotionStateChart)
    assert not issubclass(UnderspecifiedNode, BuildsMotionStateChart)
    assert not issubclass(ReAttachNode, BuildsMotionStateChart)


# %% building a chart outside a plan


def test_executable_is_built_without_a_plan(immutable_simple_pr2_world):
    """
    The mixin builds a complete executable from the hooks alone, without reaching into a
    plan tree.
    """
    world, robot_view, context = immutable_simple_pr2_world
    builder = ChartBuilderWithoutPlan(context)
    child = ChartBuilderWithoutPlan(context)

    executable = builder.create_giskard_executable([child])

    assert type(executable) is GiskardExecutable
    assert executable.context is context
    assert type(executable.root_node) is Sequence
    assert executable.root_node.nodes == [child.added_goal]
    assert executable.motion_state_chart.nodes == [
        executable.root_node,
        child.added_goal,
    ]


# %% children that contribute nothing


def test_children_without_motions_are_left_out_of_the_chart(immutable_simple_pr2_world):
    """
    A child contributing no motions is skipped, so it cannot leave an empty goal behind.
    """
    world, robot_view, context = immutable_simple_pr2_world
    builder = ChartBuilderWithoutPlan(context)
    without_motions = ChartBuilderWithoutPlan(context, contributes_motions=False)
    with_motions = ChartBuilderWithoutPlan(context)

    executable = builder.create_giskard_executable([without_motions, with_motions])

    assert without_motions.added_goal is None
    assert executable.root_node.nodes == [with_motions.added_goal]
