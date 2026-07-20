"""
Underspecified cutting demo: a PR2 slices a bread on the apartment kitchen counter with
a knife mounted on its right gripper.

Unlike the simple demo, no concrete values are given for the base pose or the cutting
parameters: the base pose is sampled from a region in front of the counter, the cutting
technique is left as an ellipsis, and slice thickness and cut count are omitted entirely
so the action derives them from the bread's size.
"""

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a

from experiments.tool_based_actions.simple_demo.demo_world import (
    BREAD_COLOR,
    CUT_MOUNT,
    TARGET_POSITION_XYZ,
    parse_object,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    CuttingKnife,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import CuttingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, start_visualization


def main() -> None:
    """
    Build the demo world and run the underspecified plan on the simulated robot.
    """
    # Importing the ORM interface registers the data access objects the
    # probabilistic backend needs to condition on the given literal values.
    import coraplex.orm.ormatic_interface

    world = setup_world()

    bread_world = parse_object("bread.stl", color=BREAD_COLOR)
    with world.modify_world():
        world.merge_world_at_pose(
            bread_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                *TARGET_POSITION_XYZ, reference_frame=world.root
            ),
        )
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.query_backend = ProbabilisticBackend()

    knife_body = attach_tool(
        world, pr2, Arms.RIGHT, parse_object("big-knife.stl"), CUT_MOUNT
    )
    bread_body = world.get_body_by_name("bread.stl")

    knife = CuttingKnife(root=knife_body)
    with world.modify_world():
        world.add_semantic_annotations([Bread(root=bread_body), knife])

    context.evaluate_conditions = False

    navigate = a(NavigateAction)(
        target_location=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=world.root,
        ),
        keep_joint_states=...,
    )
    navigate.where(
        navigate.variable.target_location.x > 1.7,
        navigate.variable.target_location.x < 1.95,
        navigate.variable.target_location.y > 2.1,
        navigate.variable.target_location.y < 2.35,
    )

    cutting = a(CuttingAction)(
        object_to_cut=bread_body,
        arm=Arms.RIGHT,
        tool=knife,
        technique=...,
    )

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            navigate,
            cutting,
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
