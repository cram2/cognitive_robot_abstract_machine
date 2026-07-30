"""
Wiping demo: a PR2 wipes a patch of the apartment kitchen counter with a sponge mounted
on its right gripper.
"""

from experiments.tool_based_actions.simple_demo.demo_world import (
    BASE_POSITION_XYZ,
    TARGET_POSITION_XYZ,
    attach_sponge,
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
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world, start_visualization


def main() -> None:
    """
    Build the demo world and run the plan on the simulated robot.
    """
    world = setup_world()
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)

    sponge_body = attach_sponge(world, pr2, Arms.RIGHT)

    sponge = Sponge(root=sponge_body)
    with world.modify_world():
        world.add_semantic_annotations([sponge])

    context.evaluate_conditions = False

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(
                Pose.from_xyz_rpy(*BASE_POSITION_XYZ, reference_frame=world.root)
            ),
            WipingAction(
                arm=Arms.RIGHT,
                tool=sponge,
                target_pose=Pose.from_xyz_rpy(
                    *TARGET_POSITION_XYZ, reference_frame=world.root
                ),
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
