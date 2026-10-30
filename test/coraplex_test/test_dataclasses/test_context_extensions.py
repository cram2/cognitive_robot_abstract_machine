from dataclasses import dataclass

import pytest

from coraplex.exceptions import DebuggingWithoutRosNode
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.plan_transformation import PlanRewriting, PlanTransformation

from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

from ...plan_running import simulated_executor, statechart_of
from ...sampling import SAMPLING_SEED

# %% debug validation


def test_debug_requires_a_ros_node(pr2_apartment_context) -> None:
    """
    Debug output is visualized over ROS, so an executor created in debug mode without a
    node is rejected at creation rather than failing later during execution.
    """
    world, _, extensions = pr2_apartment_context

    with pytest.raises(DebuggingWithoutRosNode):
        KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(world, extensions), debug=True
        )


# %% the root Cartesian goals are expressed in


def test_controlled_root_is_the_robot_when_the_base_stands_still(
    pr2_apartment_context,
) -> None:
    """
    A robot that keeps its base still can only move what hangs below its own root, so
    that is what a Cartesian goal is expressed relative to.
    """
    world, robot, _ = pr2_apartment_context
    robot.mobile_base.full_body_controlled = False

    assert RobotAccess(robot).controlled_root is robot.root


def test_controlled_root_is_the_world_when_the_base_may_drive(
    pr2_apartment_context,
) -> None:
    """
    A robot that drives its base while it manipulates moves relative to the world, so
    the world root is what a Cartesian goal is expressed relative to.
    """
    world, robot, _ = pr2_apartment_context
    previous = robot.mobile_base.full_body_controlled
    robot.mobile_base.full_body_controlled = True

    try:
        assert RobotAccess(robot).controlled_root is world.root
    finally:
        robot.mobile_base.full_body_controlled = previous


# %% carrying the extensions over to a copy of the world


def test_the_robot_of_a_copied_world_is_its_own_copy(pr2_apartment_context) -> None:
    """
    A trial runs a candidate in a copy of the world, where the robot is the copy's.
    """
    world, robot, _ = pr2_apartment_context
    copy = world.__deepcopy__({})

    copied_access = copy.rebind_world_entities(RobotAccess(robot))

    assert copied_access.robot is copy.get_semantic_annotation_by_id(robot.id)


@dataclass
class TransformationCountingItsApplications(PlanTransformation[MoveTorsoAction]):
    """
    Rewrites nothing, and counts how often it was applied.
    """

    applications: int = 0
    """
    How often it was applied.
    """

    def is_applicable(self, plan_node: MoveTorsoAction) -> bool:
        return True

    def apply(self, plan_node: MoveTorsoAction) -> None:
        self.applications += 1


def test_a_copied_plan_rewriting_offers_the_nodes_offered_already_again(
    pr2_apartment_context,
) -> None:
    """
    A trial rewrites its candidate in a statechart of its own, so what the plan's
    rewriting offered already must not keep the copy from offering it again.
    """
    world, _, extensions = pr2_apartment_context
    counting = TransformationCountingItsApplications()
    rewriting = PlanRewriting(transformations=[counting])
    torso = MoveTorsoAction(torso_state=TorsoState.HIGH)
    statechart_of(simulated_executor(extensions), torso)
    rewriting.rewrite(torso)

    world.rebind_world_entities(rewriting).rewrite(torso)

    assert counting.applications == 2


# %% repeatable location samples

WORLD_FIXTURES_WITH_EXTENSIONS = [
    "pr2_apartment_context",
    "simple_pr2_context",
    "stretch_apartment_context",
    "apartment_world_pr2_copy_with_context",
]
"""
The shared fixtures that hand a test the context extensions to run its actions with.
"""


@pytest.mark.parametrize("world_fixture", WORLD_FIXTURES_WITH_EXTENSIONS)
def test_a_shared_fixture_fixes_the_samples_its_plans_make(
    world_fixture, request
) -> None:
    """
    A location samples its candidates from a costmap rather than ranking it, so a test
    handed an unseeded plan would stand somewhere else every run.
    """
    _, _, extensions = request.getfixturevalue(world_fixture)

    [grounding] = [
        extension
        for extension in extensions
        if isinstance(extension, StatementGrounding)
    ]
    assert grounding.sampling_seed == SAMPLING_SEED
