"""
The PR2 lays a place setting: it carries the milk, a bowl and a spoon onto the table.

Runs in simulation against the apartment, so nothing on the network is needed. The
scaffolding in :mod:`coraplex.demonstrations` owns the ROS session and publishes the
world to Rviz, so the run can be watched while it happens.

Every transport leaves the grasp and the standing pose open: each pair is tried out
before it is executed, and the first that succeeds is taken. The bowl shows why, as only
some of the grasps around its rim can be reached from where the robot may stand.

While the plan runs, SegMind segments it into the grasps, pick-ups, placings and
containments of the place setting, and reports them once the plan is done.
"""

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Optional, Tuple, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforeMoveAndPickUp
from krrood.entity_query_language.factories import (
    an,
    entity,
    variable,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import ContainmentDetector
from segmind.event_segmentation import Segmind
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from, under the coraplex resources.
    """

    APARTMENT = os.path.join("worlds", "apartment.urdf")
    MILK = os.path.join("objects", "milk.stl")
    BOWL = os.path.join("objects", "bowl.stl")
    SPOON = os.path.join("objects", "spoon.stl")

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", self.value
        )


class ApartmentBody(StrEnum):
    """
    The apartment's own bodies the demonstration acts on.
    """

    SPOON_DRAWER = "cabinet10_drawer_top"


@dataclass
class PlaceSettingObject:
    """
    One object of the place setting: what it is, where it starts, and where it is laid.
    """

    semantic_annotation_type: Type[HasRootKinematicStructureEntity]
    """
    What the object is.

    Its body is named after it, so the plan can ask for it by type.
    """

    mesh: SceneFile
    """
    The mesh the object is shaped by.
    """

    start: HomogeneousTransformationMatrix
    """
    Where the object stands before the plan begins, in :attr:`starts_on`'s frame.
    """

    along_table: float
    """
    Where along the table the object is laid, in meters.
    """

    height: float
    """
    How high the object's origin is laid so that it rests on the table surface
    (z=0.723).
    """

    starts_on: Optional[ApartmentBody] = None
    """
    The apartment body the object starts on.

    ``None`` starts it at the world root.
    """

    across_table: float = 3.3
    """
    Where the object is laid across the table, along the axis the robot faces.

    Shared by the setting unless an object needs to sit nearer the robot's side than the
    rest of it.
    """

    facing_yaw: float = 1.57
    """
    Which way the object faces once it has been laid down.
    """

    def spawn(self, world: World) -> None:
        """
        Put the object where it starts and name it among the world's annotations.

        :param world: The world to spawn into.
        """
        name = self.semantic_annotation_type.__name__
        SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=self.semantic_annotation_type,
            root_specification=BodySpecification.mesh(name, self.mesh.path),
        ).spawn(
            world,
            parent=(
                None
                if self.starts_on is None
                else world.get_body_by_name(self.starts_on)
            ),
            parent_T_self=self.start,
        )

    def annotation_in(self, world: World) -> HasRootKinematicStructureEntity:
        """
        :param world: The world the object was spawned into.
        :return: The annotation naming the object, as the plan refers to it.
        """
        return next(
            an(
                entity(
                    variable(
                        self.semantic_annotation_type,
                        domain=world.semantic_annotations,
                    )
                )
            ).evaluate()
        )

    def target_location(self, world: World) -> Pose:
        """
        :param world: The world the table stands in.
        :return: Where on the table the object is laid.
        """
        return Pose.from_xyz_rpy(
            self.along_table,
            self.across_table,
            self.height,
            yaw=self.facing_yaw,
            reference_frame=world.root,
        )


# %% the demonstration


@dataclass
class BulletWorldDemonstration(RobotDemonstration):
    """
    The PR2 transports the milk, a bowl and a spoon onto the table in the apartment.
    """

    ros_node_name: str = "bullet_world_demo_node"

    robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            1.1, 2.5, 0
        )
    )
    """
    Where the PR2 stands before the plan begins: far enough from the counter that its
    parked grippers clear the cabinets, and not so far that the torso meets the doors
    behind it.
    """

    milk: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Milk,
            SceneFile.MILK,
            HomogeneousTransformationMatrix.from_xyz_rpy(2.37, 2, 1.0345),
            along_table=4.8,
            height=0.81,
        )
    )
    """
    The milk, which starts on the counter.
    """

    bowl: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Bowl,
            SceneFile.BOWL,
            HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.2, 0.98),
            along_table=5.0,
            height=0.76,
        )
    )
    """
    The bowl, which starts on the counter.
    """

    spoon: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Spoon,
            SceneFile.SPOON,
            HomogeneousTransformationMatrix.from_xyz_rpy(-0.05, -0.05, -0.01),
            along_table=5.2,
            height=0.73,
            across_table=3.25,
            starts_on=ApartmentBody.SPOON_DRAWER,
        )
    )
    """
    The spoon, which starts inside the drawer it is fetched from.
    """

    @property
    def place_setting(self) -> Tuple[PlaceSettingObject, ...]:
        """
        :return: The objects carried onto the table, in the order they are carried. They
            are laid 20 cm apart, so placing one keeps the gripper clear of the last.
        """
        return self.milk, self.bowl, self.spoon

    def build_simulated_world(self) -> World:
        """
        The apartment with the PR2 in it, placed at :attr:`robot_start` through its
        odom.
        """
        return WorldSpecification.from_urdf(
            SceneFile.APARTMENT.path,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=self.robot_start,
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(Bowl.__name__)

    def populate_scene(self, world: World) -> None:
        """
        Put the place setting where it starts out, and let the world reasoner annotate
        the apartment's furniture, including the drawer the spoon lies in.
        """
        for placed_object in self.place_setting:
            placed_object.spawn(world)

        with world.modify_world():
            WorldReasoner(world).reason()

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            sampling_seed=0,
            alternative_motion_mappings=self.alternative_motion_mappings,
            plan_transformations=[OpenDrawerBeforeMoveAndPickUp()],
            _debug=self.debug,
        )

    def segment_events(self, world: World) -> AbstractContextManager:
        """
        Detect the place setting being grasped, picked up, placed and contained.
        """
        return Segmind.create_for_semantic_annotation_types(
            world,
            tuple(
                placed_object.semantic_annotation_type
                for placed_object in self.place_setting
            ),
            detector_types=(
                PickUpDetector,
                PlacingDetector,
                ContainmentDetector,
                GraspDetector,
            ),
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry each object to its place on the table.
        """
        world = context.world
        left_arm = context.robot.left_arm
        return sequential(
            [
                ParkArmsAction(context.robot.all_arms),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction.from_graspable_by_closest_grasps(
                    self.milk.annotation_in(world),
                    self.milk.target_location(world),
                    left_arm,
                    context,
                ),
                TransportAction.from_graspable_by_closest_grasps(
                    self.bowl.annotation_in(world),
                    self.bowl.target_location(world),
                    left_arm,
                    context,
                ),
                TransportAction.from_graspable_by_closest_grasps(
                    self.spoon.annotation_in(world),
                    self.spoon.target_location(world),
                    left_arm,
                    context,
                ),
            ],
            context=context,
        ).plan


def main(
    execution_type: ExecutionType = ExecutionType.SIMULATED,
    collision_avoidance: bool = True,
    event_segmentation: bool = True,
    debug: bool = False,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param collision_avoidance: Whether every motion state chart avoids collisions.
    :param event_segmentation: Whether SegMind segments the run into events.
    :param debug: Whether to run in debug mode, publishing every copy of the world a
        candidate is tried in.
    """
    BulletWorldDemonstration(
        used_robot=PR2,
        execution_type=execution_type,
        collision_avoidance=collision_avoidance,
        event_segmentation=event_segmentation,
        debug=debug,
    ).run()


if __name__ == "__main__":
    main()
