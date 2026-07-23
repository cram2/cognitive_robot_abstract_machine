"""
Underspecified wiping demo: a PR2 wipes a patch of the apartment kitchen counter with a
sponge mounted on its right gripper.

Unlike the simple demo, no concrete values are given for the base pose, the wiping patch
or the wiping technique: the poses are sampled from regions described by where-
conditions and the technique is left as an ellipsis.
"""

# Importing the ORM interface registers the data access objects the probabilistic
# backend needs to condition on the given literal values.
import coraplex.orm.ormatic_interface
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a

from experiments.tool_based_actions.experiment.scene import SpawnRegion
from experiments.tool_based_actions.simple_demo.demo_world import (
    TARGET_POSITION_XYZ,
    attach_sponge,
)
from experiments.tool_based_actions.underspecified_demo.demo_setup import (
    build_underspecified_navigation,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world, start_visualization


def main() -> None:
    """
    Build the demo world and run the underspecified plan on the simulated robot.
    """
    world = setup_world()
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.query_backend = ProbabilisticBackend()

    sponge_body = attach_sponge(world, pr2, Arms.RIGHT)

    sponge = Sponge(root=sponge_body)
    with world.modify_world():
        world.add_semantic_annotations([sponge])

    context.evaluate_conditions = False

    navigate = build_underspecified_navigation(world)

    wipe_patch_region = SpawnRegion(
        minimum_x=2.3, maximum_x=2.5, minimum_y=2.1, maximum_y=2.3, height=0.0
    )
    counter_height = TARGET_POSITION_XYZ[2]
    wiping = a(WipingAction)(
        arm=Arms.RIGHT,
        tool=sponge,
        technique=...,
        target_pose=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=counter_height,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=world.root,
        ),
    )
    wiping.where(
        wiping.variable.target_pose.x > wipe_patch_region.minimum_x,
        wiping.variable.target_pose.x < wipe_patch_region.maximum_x,
        wiping.variable.target_pose.y > wipe_patch_region.minimum_y,
        wiping.variable.target_pose.y < wipe_patch_region.maximum_y,
    )

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            navigate,
            wiping,
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
