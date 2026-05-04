from __future__ import annotations

from dataclasses import field
from functools import cached_property

import numpy as np

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import *
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.plans.plan import Plan
from pycram.robot_plans.actions.composite.sage10k_actions import Sage10kOpenDoor
from pycram.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
)
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.processing import (
    create_hsrb_in_world,
)
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kNonShittyScenes,
    NaturalLanguageDescriptionWithTypeDescription,
    RoomWithWallsAndDoors,
    DoorWithType,
)
from semantic_digital_twin.reasoning.predicates import (
    compute_euclidean_planar_distance,
    is_supported_by,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Point3, Pose, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class Sage10kAbstractDemo:
    """
    Configuration for the Sage10k non-shitty scenes demo.
    """

    scene_url: ClassVar[str]
    """
    The URL of the scene to use for the demo.
    """

    world: Optional[World] = field(init=False, default=None)
    """
    The world to execute the demo in. Only available after """

    def create_world(self):
        loader = Sage10kDatasetLoader()
        self.world = loader.create_scene(scene_url=self.scene_url).create_world()
        create_hsrb_in_world(self.world)
        self.preprocess_world()

    def preprocess_world(self):
        self.robot.root.parent_connection.origin = self.robot_starting_pose

    @cached_property
    def robot(self) -> HSRB:
        return self.world.get_semantic_annotations_by_type(HSRB)[0]

    def remove_rooted_annotations(self, semantic_annotations: Iterable[HasRootBody]):
        with self.world.modify_world():
            for annotation in semantic_annotations:
                self.world.remove_kinematic_structure_entity(annotation.root)
                self.world.remove_semantic_annotation(annotation)

    @property
    def robot_starting_pose(self) -> Pose:
        raise NotImplementedError

    @property
    def plan(self) -> Plan:
        pass


@dataclass
class Sage10kGymDemo(Sage10kAbstractDemo):
    scene_url: ClassVar[str] = Sage10kNonShittyScenes.GYM

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(x=3, y=-7, z=0, reference_frame=self.world.root)

    @property
    def world_P_object_of_interest(self) -> Point3:
        return Point3(1.03, -0.716, 0.203, reference_frame=self.world.root)

    @property
    def pickup_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            0.94, 0.2, 0, yaw=-np.pi / 2, reference_frame=self.world.root
        )

    @property
    def place_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            -0.15, 4.55, 0.865, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @property
    def place_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            -0.12, 4, 0, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @cached_property
    def main_entrance(self):
        room = variable(RoomWithWallsAndDoors, self.world.semantic_annotations)
        gym = an(entity(room).where(contains(room.room_type, "gym"))).first()
        main_entrance: DoorWithType = an(
            entity(v := variable_from(gym.doors)).where(
                contains(v.type_description, "main")
            )
        ).first()
        return main_entrance

    def preprocess_world(self):
        super().preprocess_world()
        v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        obstacles_of_main_entrance = an(
            entity(v).where(
                compute_euclidean_planar_distance(
                    v.root, self.main_entrance.root, Vector3.Z()
                )
                < 0.9
            )
        )

        self.remove_rooted_annotations(obstacles_of_main_entrance.evaluate())

    @property
    def plan(self):
        arm = Arms.RIGHT
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            self.robot.arm.manipulator,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)

        [body] = self.world.get_bodies_by_global_position(
            self.world_P_object_of_interest, 0.1
        )

        plan = sequential(
            [
                open_door,
                ParkArmsAction(Arms.BOTH),
                MoveAndPickUpAction(
                    object_designator=body,
                    standing_position=self.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveAndPlaceAction(
                    object_designator=body,
                    standing_position=self.place_navigation_pose,
                    arm=arm,
                    target_location=self.place_pose,
                ),
            ],
            context=context,
        ).plan
        return plan


@dataclass
class Sage10kTVStudioDemo(Sage10kAbstractDemo):
    """
    Get the book from the table and present it to the audience.
    """

    scene_url: ClassVar[str] = Sage10kNonShittyScenes.TV_STUDIO

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(x=12.5, y=3, z=0, reference_frame=self.world.root)

    @property
    def pickup_navigation_pose(self):
        return Pose.from_xyz_rpy(
            x=6.83,
            y=5.38,
            z=self.robot.root.global_pose.z,
            yaw=1.78,
            reference_frame=self.world.root,
        )

    @cached_property
    def main_entrance(self):
        door_v = variable(DoorWithType, self.world.semantic_annotations)
        main_entrance: DoorWithType = an(entity(door_v)).first()
        return main_entrance

    @property
    def book_to_pick(self) -> Body:
        @symbolic_function
        def closes_to_border(target) -> float:
            return self.world.transform(target.global_pose, couch_table.root).y

        v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        couch_table = an(
            entity(v).where(contains(v.description, "coffee table"))
        ).first()

        v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        target = an(
            entity(v)
            .where(
                contains(v.type_description, "book"),
                is_supported_by(v.root, couch_table.root),
            )
            .ordered_by(
                v,
                key=lambda x: closes_to_border(
                    x.root,
                ),
                descending=False,
            )
        )
        book = target.first()
        return book.root

    @property
    def plan(self) -> Plan:
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            manipulator=context.robot.arm.manipulator,
            rotate_gripper=True,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)
        mpa = MoveAndPickUpAction(
            standing_position=self.pickup_navigation_pose,
            object_designator=self.book_to_pick,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        )
        present_book = NavigateAction(target_location=self.robot_starting_pose)

        return sequential([open_door, mpa, present_book], context=context).plan
