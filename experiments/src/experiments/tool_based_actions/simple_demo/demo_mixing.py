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

from coraplex.robot_plans.actions.composite.tool_based import MixingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, selected_visualizations
from coraplex.plans.context_extensions import RobotAccess
from coraplex.demonstrations import RobotDemonstrationRosSession
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.statechart import Statechart
from cramph.composites import Sequence


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
    pr2 = PR2.from_world(world)

    whisk_body = attach_tool(world, pr2.right_arm, parse_object("whisk.stl"), MIX_MOUNT)
    bowl_body = world.get_body_by_name("bowl.stl")

    whisk = Whisk(root=whisk_body)
    with world.modify_world():
        world.add_semantic_annotations([Bowl(root=bowl_body), whisk])

    plan = Sequence(
        nodes=[
            SetGripperAction(
                configuration=pr2.right_arm.end_effector.default_configuration(
                    GripperState.CLOSE
                )
            ),
            ParkArmsAction(arms=pr2.all_arms),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            NavigateAction(
                target_location=Pose.from_xyz_rpy(
                    *BASE_POSITION_XYZ, reference_frame=world.root
                )
            ),
            MixingAction(container=bowl_body, arm=pr2.right_arm, tool=whisk),
        ]
    )

    with RobotDemonstrationRosSession.start("mixing_demo") as ros_session:
        executor = KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(world, [RobotAccess(pr2)]),
            ros_node=ros_session.node,
            extensions=selected_visualizations(ros_session.node),
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(plan)
        executor.compile(statechart)
        executor.execute()


if __name__ == "__main__":
    main()
