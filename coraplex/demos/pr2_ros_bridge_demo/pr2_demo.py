"""
PR2 manipulation with CoraPlex and an optional ROS 1/2 bridge.
"""

from __future__ import annotations

# %% imports

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing_extensions import TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World

# %% choices


class DemonstrationTask(StrEnum):
    """
    The manipulation sequence to perform.
    """

    TORSO = "torso"
    ARMS = "arms"
    PICK_AND_PLACE = "pick-place"


class ExecutionMode(StrEnum):
    """
    Whether commands stay in the digital twin or reach the controller.
    """

    SIMULATION = "sim"
    HARDWARE = "real"


# %% demonstration


@dataclass
class PR2Demonstration(RobotDemonstration):
    """
    Perform a PR2 manipulation sequence against a local or synchronized world.
    """

    used_robot: type[AbstractRobot] = PR2
    """
    Robot whose semantic annotations supply the arms and torso configurations.
    """

    ros_node_name: str = "pr2_coraplex_demo"
    """
    Node name passed to the ROS session in both execution modes.
    """

    task: DemonstrationTask = DemonstrationTask.TORSO
    """
    Sequence selected for this run.
    """

    cereal_body_name: str = "breakfast_cereal.stl"
    """
    Existing cereal body to manipulate in the selected world.
    """

    placement_offset: float = 0.20
    """
    Displacement of the cereal along the world's y axis, in metres.
    """

    simulated_base_distance: float = 0.65
    """
    Initial separation of the simulated base from the cereal along x, in metres.
    """

    pick_and_place_torso_state: TorsoState = TorsoState.MID
    """
    Torso configuration for approaching and placing the cereal.
    """

    def build_simulated_world(self) -> World:
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return URDFParser.from_file(PR2.get_ros_file_path()).parse()
        world = setup_world()
        robot = PR2.from_world(world)
        cereal_position = world.get_body_by_name(
            self.cereal_body_name
        ).global_pose.position
        with world.modify_world():
            robot.root.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    float(cereal_position.x) - self.simulated_base_distance,
                    float(cereal_position.y),
                    reference_frame=world.root,
                )
            )
        return world

    def is_scene_populated(self, world: World) -> bool:
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return True
        cereal_body = world.get_body_by_name(self.cereal_body_name)
        return any(
            annotation.root is cereal_body
            for annotation in world.get_semantic_annotations_by_type(Cereal)
        )

    def populate_scene(self, world: World) -> None:
        """
        Annotate the existing cereal body without changing its pose or geometry.
        """
        if self.task is DemonstrationTask.PICK_AND_PLACE:
            with world.modify_world():
                world.add_semantic_annotation(
                    Cereal(root=world.get_body_by_name(self.cereal_body_name))
                )

    def build_context(self, world: World) -> Context:
        robots = world.get_semantic_annotations_by_type(PR2)
        return Context(
            world=world,
            robot=robots[0] if robots else PR2.from_world(world),
            ros_node=self.ros_node,
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=self.debug,
        )

    def build_plan(self, context: Context) -> PlanNode:
        robot = context.robot
        if self.task is DemonstrationTask.TORSO:
            return sequential(
                [MoveTorsoAction(TorsoState.HIGH), MoveTorsoAction(TorsoState.LOW)],
                context=context,
            )
        if self.task is DemonstrationTask.ARMS:
            return sequential(
                [ParkArmsAction(arms=list(robot.all_arms))], context=context
            )

        cereal_body = context.world.get_body_by_name(self.cereal_body_name)
        cereal = next(
            annotation
            for annotation in context.world.get_semantic_annotations_by_type(Cereal)
            if annotation.root is cereal_body
        )
        cereal_pose = cereal_body.global_pose
        place_pose = Pose(
            position=Point3(
                float(cereal_pose.position.x),
                float(cereal_pose.position.y) + self.placement_offset,
                float(cereal_pose.position.z),
            ),
            orientation=cereal_pose.orientation,
            reference_frame=context.world.root,
        )
        return sequential(
            [
                ParkArmsAction(arms=list(robot.all_arms)),
                MoveTorsoAction(self.pick_and_place_torso_state),
                PickUpAction(
                    grasp=cereal.grasp_candidates()[0],
                    arm=robot.get_right_arm_if_specified(),
                ),
                PlaceAction(object_designator=cereal, target_location=place_pose),
                ParkArmsAction(arms=list(robot.all_arms)),
            ],
            context=context,
        )


# %% command line


def main() -> None:
    """
    Run the selected CoraPlex plan using the shared demonstration lifecycle.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=DemonstrationTask,
        choices=list(DemonstrationTask),
        default=DemonstrationTask.TORSO,
    )
    parser.add_argument(
        "--mode",
        type=ExecutionMode,
        choices=list(ExecutionMode),
        default=ExecutionMode.SIMULATION,
    )
    parser.add_argument(
        "--collision-avoidance",
        action="store_true",
        help="Enable collision avoidance for motion planning",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Starting PR2 %s demonstration (%s)", arguments.task, arguments.mode
    )
    demonstration = PR2Demonstration(
        task=arguments.task,
        execution_type=(
            ExecutionType.REAL
            if arguments.mode is ExecutionMode.HARDWARE
            else ExecutionType.KINEMATICALLY_SIMULATED
        ),
        collision_avoidance=arguments.collision_avoidance,
    )
    demonstration.run()


if __name__ == "__main__":
    main()
