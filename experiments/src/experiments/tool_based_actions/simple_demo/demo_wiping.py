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

from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world, selected_visualizations
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
    pr2 = PR2.from_world(world)

    sponge_body = attach_sponge(world, pr2.right_arm)

    sponge = Sponge(root=sponge_body)
    with world.modify_world():
        world.add_semantic_annotations([sponge])

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
            WipingAction(
                arm=pr2.right_arm,
                tool=sponge,
                target_pose=Pose.from_xyz_rpy(
                    *TARGET_POSITION_XYZ, reference_frame=world.root
                ),
            ),
        ]
    )

    with RobotDemonstrationRosSession.start("wiping_demo") as ros_session:
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
