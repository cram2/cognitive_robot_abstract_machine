"""
Mixing demo: a PR2 mixes the contents of a bowl on the apartment kitchen counter with a
whisk mounted on its right gripper.
"""

from experiments.tool_based_actions.simple_demo.demo_world import (
    BASE_POSITION_XYZ,
    BOWL_COLOR,
    MIX_MOUNT,
    TARGET_POSITION_XYZ,
    parse_object,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import MixingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, start_visualization


def main() -> None:
    """
    Build the demo world and run the plan on the simulated robot.
    """
    world = setup_world()

    bowl_world = parse_object("bowl.stl", color=BOWL_COLOR)
    with world.modify_world():
        world.merge_world_at_pose(
            bowl_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                *TARGET_POSITION_XYZ, reference_frame=world.root
            ),
        )
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)

    whisk_body = attach_tool(
        world, pr2, Arms.RIGHT, parse_object("whisk.stl"), MIX_MOUNT
    )
    bowl_body = world.get_body_by_name("bowl.stl")

    whisk = Whisk(root=whisk_body)
    with world.modify_world():
        world.add_semantic_annotations([Bowl(root=bowl_body), whisk])

    context.evaluate_conditions = False

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(
                Pose.from_xyz_rpy(*BASE_POSITION_XYZ, reference_frame=world.root)
            ),
            MixingAction(container=bowl_body, arm=Arms.RIGHT, tool=whisk),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
