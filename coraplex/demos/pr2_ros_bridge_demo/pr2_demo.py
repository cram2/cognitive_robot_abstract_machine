"""
PR2 manipulation with CoraPlex and an optional ROS 1/2 bridge.
"""

from __future__ import annotations

# %% imports

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from enum import StrEnum
from typing_extensions import TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal, Milk
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)

from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification

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


class DemonstrationResource(PathEnumeration):
    """
    Scene files relative to the CoraPlex resource directory.
    """

    DIRECTORY = "resources"
    APARTMENT = "worlds/apartment.urdf"
    CEREAL = "objects/breakfast_cereal.stl"
    MILK = "objects/milk.stl"


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

    cereal_body_name: str = DemonstrationResource.CEREAL.name
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

    resource_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / DemonstrationResource.DIRECTORY
    )
    """
    Directory holding the demo's apartment and object meshes.
    """

    cereal_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.0487
        )
    )
    """
    Initial cereal pose relative to the apartment root, in metres.
    """

    milk_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.0345
        )
    )
    """
    Initial milk pose relative to the apartment root, in metres.
    """

    def scene_specification(self) -> WorldSpecification:
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return WorldSpecification()
        return WorldSpecification.from_urdf(
            str(self.resource_directory / DemonstrationResource.APARTMENT),
            objects=[
                SemanticAnnotationWithRootSpecification(
                    name=self.cereal_body_name,
                    semantic_annotation_type=Cereal,
                    root_specification=BodySpecification.mesh(
                        self.cereal_body_name,
                        str(self.resource_directory / DemonstrationResource.CEREAL),
                        parent_T_self=self.cereal_start,
                    ),
                ),
                SemanticAnnotationWithRootSpecification(
                    name=DemonstrationResource.MILK.name,
                    semantic_annotation_type=Milk,
                    root_specification=BodySpecification.mesh(
                        DemonstrationResource.MILK.name,
                        str(self.resource_directory / DemonstrationResource.MILK),
                        parent_T_self=self.milk_start,
                    ),
                ),
            ],
        )

    def robot_specification(self) -> RobotSpecification:
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return RobotSpecification(self.used_robot)
        return RobotSpecification(
            self.used_robot,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                float(self.cereal_start.position.x) - self.simulated_base_distance,
                float(self.cereal_start.position.y),
            ),
        )

    def build_simulated_world(self) -> World:
        specification = self.scene_specification()
        specification.robots.append(self.robot_specification())
        return specification.to_domain_object()

    def acquire_world(self) -> World:
        """
        Require the measured manipulation object before returning a controller world.

        See :meth:`RobotDemonstration.acquire_world`.
        """
        world = super().acquire_world()
        if self.execution_type is ExecutionType.REAL:
            self.is_scene_populated(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        """
        Require the measured cereal body for real manipulation; missing bodies raise the
        world's lookup error instead of requesting a synthetic controller scene.

        See :meth:`RobotDemonstration.is_scene_populated`.
        """
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return True
        if self.execution_type is ExecutionType.REAL:
            world.get_body_by_name(self.cereal_body_name)
            return True
        return any(body.name.name == self.cereal_body_name for body in world.bodies)

    def populate_scene(self, world: World) -> None:
        """
        Annotate the existing cereal body without changing its pose or geometry.
        """
        if self.task is not DemonstrationTask.PICK_AND_PLACE:
            return
        cereal_body = world.get_body_by_name(self.cereal_body_name)
        if any(
            annotation.root is cereal_body
            for annotation in world.get_semantic_annotations_by_type(Cereal)
        ):
            return
        with world.modify_world():
            world.add_semantic_annotation(Cereal(root=cereal_body))

    def build_context(self, world: World) -> Context:
        self.populate_scene(world)
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
