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
from pycram.robot_plans.actions.core.misc import MoveToReach
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
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
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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

    @cached_property
    def main_entrance(self):
        door_v = variable(DoorWithType, self.world.semantic_annotations)
        main_entrance: DoorWithType = an(entity(door_v)).first()
        return main_entrance

    @property
    def plan(self) -> Plan:
        pass


@dataclass
class Sage10kGymDemo(Sage10kAbstractDemo):
    """
    Collect a bottle from the bench and put it in the bin.
    """

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


@dataclass
class Sage10kCraftsmanLobbyDemo(Sage10kAbstractDemo):
    """
    Put something to read next to the clean couch.
    """

    scene_url: ClassVar[str] = Sage10kNonShittyScenes.CRAFTSMAN_LOBBY

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(x=14, y=6, z=0, reference_frame=self.world.root)

    @property
    def pickup_navigation_pose(self):
        return Pose.from_xyz_rpy(
            x=6.83,
            y=5.38,
            z=self.robot.root.global_pose.z,
            yaw=1.78,
            reference_frame=self.world.root,
        )

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
    def plan(self):
        target_pose = Pose.from_xyz_rpy(
            x=5.48, y=7.46, z=0.8, reference_frame=self.world.root
        )
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        open_door = Sage10kOpenDoor(self.main_entrance)
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.BACK,
            vertical_alignment=VerticalAlignment.NoAlignment,
            manipulator=context.robot.arm.manipulator,
            rotate_gripper=True,
        )
        mpu = MoveAndPickUpAction(
            standing_position=self.pickup_navigation_pose,
            object_designator=self.book_to_pick,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        )
        mpp = MoveAndPlaceAction(
            standing_position=Pose.from_xyz_rpy(
                x=5.48, y=6.96, yaw=np.pi / 2, reference_frame=self.world.root
            ),
            object_designator=self.book_to_pick,
            target_location=target_pose,
            arm=Arms.LEFT,
        )

        return sequential([open_door, mpu, mpp], context=context).plan


@dataclass
class Sage10kTropicalWarehouse(Sage10kAbstractDemo):
    """
    Fetch me a cup from the warehouse.
    """

    scene_url: ClassVar[str] = Sage10kNonShittyScenes.TROPICAL_WAREHOUSE

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(x=14, y=4.5, z=0, reference_frame=self.world.root)

    @property
    def pickup_navigation_pose(self):
        return Pose.from_xyz_rpy(
            x=2.45265, y=7.28, yaw=2.16, reference_frame=self.world.root
        )

    @property
    def target_to_pick(self) -> Body:

        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        point_guess = Pose.from_xyz_rpy(
            x=2.19, y=7.64, z=0.35, reference_frame=self.world.root
        ).position
        target_v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        target = (
            an(entity(target_v).where(target_v.type_description == "cup"))
            .ordered_by(
                variable=target_v,
                key=lambda x: planar_distance(x.root.global_pose.position, point_guess),
                descending=False,
            )
            .first()
        )
        return target.root

    @property
    def plan(self) -> Plan:
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.RIGHT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            manipulator=context.robot.arm.manipulator,
            rotate_gripper=False,
        )
        mpu = MoveAndPickUpAction(
            standing_position=self.pickup_navigation_pose,
            object_designator=self.target_to_pick,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        )

        open_door = Sage10kOpenDoor(self.main_entrance)
        park_arms = ParkArmsAction(arm=Arms.LEFT)
        present = NavigateAction(target_location=self.robot_starting_pose)

        return sequential([open_door, park_arms, mpu, present], context=context).plan


@dataclass
class Sage10kVaporwave(Sage10kAbstractDemo):
    """
    Fetch me a cup from the warehouse.
    """

    scene_url: ClassVar[str] = Sage10kNonShittyScenes.VAPORWAVE

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(x=4.4, y=10, z=0, reference_frame=self.world.root)

    @property
    def pickup_navigation_pose(self):
        return Pose.from_xyz_rpy(
            x=0.93235, y=4.74108, yaw=2.90782, reference_frame=self.world.root
        )

    @property
    def target_to_pick(self) -> Body:

        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        point_guess = Pose.from_xyz_rpy(
            x=0.468, y=4.87, z=0.528, reference_frame=self.world.root
        ).position
        target_v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        target = (
            an(entity(target_v.root))
            .ordered_by(
                variable=target_v,
                key=lambda x: planar_distance(x.root.global_pose.position, point_guess),
                descending=False,
            )
            .first()
        )
        return target

    @property
    def plan(self) -> Plan:
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.TOP,
            manipulator=context.robot.arm.manipulator,
            rotate_gripper=True,
        )
        mpu = MoveAndPickUpAction(
            standing_position=self.pickup_navigation_pose,
            object_designator=self.target_to_pick,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        )

        open_door = Sage10kOpenDoor(self.main_entrance)
        park_arms = ParkArmsAction(arm=Arms.LEFT)
        place_target_pose = Pose.from_xyz_rpy(
            x=0.605, y=1.615, z=0.66, reference_frame=self.world.root
        )
        mpp = MoveAndPlaceAction(
            standing_position=Pose.from_xyz_rpy(
                x=0.605, y=2.115, yaw=-1.5708, reference_frame=self.world.root
            ),
            target_location=place_target_pose,
            object_designator=self.target_to_pick,
            arm=Arms.LEFT,
        )

        return sequential(
            [open_door, park_arms, mpu, ParkArmsAction(arm=Arms.LEFT), mpp],
            context=context,
        ).plan


@dataclass
class Sage10kEclecticResidence(Sage10kAbstractDemo):
    """
    Fetch me a cup from the warehouse.
    """

    scene_url: ClassVar[str] = Sage10kNonShittyScenes.ECLECTIC_RESIDENCE

    @property
    def robot_starting_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            x=1.5, y=8, z=0, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @property
    def pickup_navigation_pose(self):
        return Pose.from_xyz_rpy(
            x=2.7337, y=4.60152, yaw=-1.79685, reference_frame=self.world.root
        )

    @property
    def target_to_pick(self) -> Body:

        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        point_guess = Pose.from_xyz_rpy(
            x=2.66, y=4.35, z=0.442, reference_frame=self.world.root
        ).position
        target_v = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        target = (
            an(entity(target_v.root))
            .ordered_by(
                variable=target_v,
                key=lambda x: planar_distance(x.root.global_pose.position, point_guess),
                descending=False,
            )
            .first()
        )
        return target

    @property
    def plan(self) -> Plan:
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.RIGHT,
            vertical_alignment=VerticalAlignment.TOP,
            manipulator=context.robot.arm.manipulator,
            rotate_gripper=True,
        )
        mpu = MoveAndPickUpAction(
            standing_position=self.pickup_navigation_pose,
            object_designator=self.target_to_pick,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        )

        open_door = Sage10kOpenDoor(self.main_entrance)
        park_arms = ParkArmsAction(arm=Arms.LEFT)
        present = NavigateAction(target_location=self.robot_starting_pose)

        return sequential(
            [open_door, park_arms, mpu, ParkArmsAction(arm=Arms.LEFT), present],
            context=context,
        ).plan


@dataclass
class Sage10kSouthwesternStoreDemo(Sage10kAbstractDemo):
    scene_url = Sage10kNonShittyScenes.SOUTHWESTERN_STORE

    @property
    def plan(self):
        arm = Arms.RIGHT
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.NoAlignment,
            self.robot.arm.manipulator,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)

        plan = sequential(
            [
                open_door,
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=0.81, y=4.81, reference_frame=self.world.root
                    )
                ),
                MoveAndPickUpAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=0.81, y=4.81, reference_frame=self.world.root
                    )
                ),
                MoveAndPlaceAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.place_navigation_pose,
                    arm=arm,
                    target_location=self.place_pose,
                ),
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=0.48, y=4.81, reference_frame=self.world.root
                    )
                ),
            ],
            context=context,
        ).plan
        return plan

    @property
    def world_P_object_of_interest(self):
        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        near_pose = Pose.from_xyz_rpy(
            x=1.7, y=0.49, z=1.17, reference_frame=self.world.root
        )
        v_final = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        bottle = (
            an(entity(v_final)).where(
                contains(v_final.type_description, "toyshelf"),
                planar_distance(v_final.root.global_pose.position, near_pose.position)
                < 0.9,
            )
        ).first()

        return bottle.root

    @property
    def robot_starting_pose(self):
        return Pose.from_xyz_rpy(3.8, 6.5)

    @property
    def pickup_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            0.63, 0.70, 0, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @property
    def place_pose(self) -> Pose:
        return Pose.from_xyz_rpy(4.41, 4.46, z=0.368, reference_frame=self.world.root)

    @property
    def place_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(3.99, 4.66, 0, reference_frame=self.world.root)

    @cached_property
    def main_entrance(self):
        room = variable(RoomWithWallsAndDoors, self.world.semantic_annotations)
        store = an(entity(room).where(contains(room.room_type, "toy store"))).first()
        main_entrance: DoorWithType = an(
            entity(v := variable_from(store.doors)).where(
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
        benches = an(entity(v).where(contains(v.type_description, "bench")))
        self.remove_rooted_annotations(obstacles_of_main_entrance.evaluate())
        self.remove_rooted_annotations(benches.evaluate())


@dataclass
class Sage10kBrutalistStoreDemo(Sage10kAbstractDemo):
    scene_url = Sage10kNonShittyScenes.BRUTALIST_STORE

    @property
    def plan(self):
        arm = Arms.RIGHT
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.NoAlignment,
            self.robot.arm.manipulator,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)

        plan = sequential(
            [
                open_door,
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=12, y=8.13, reference_frame=self.world.root
                    )
                ),
                MoveAndPickUpAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveAndPlaceAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.place_navigation_pose,
                    arm=arm,
                    target_location=self.place_pose,
                ),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=12, y=8.13, reference_frame=self.world.root
                    )
                ),
            ],
            context=context,
        ).plan
        return plan

    @property
    def world_P_object_of_interest(self):
        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        near_pose = Pose.from_xyz_rpy(
            x=8.28, y=0.35, z=0.69, reference_frame=self.world.root
        )
        v_final = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        bottle = (
            an(entity(v_final)).where(
                contains(v_final.type_description, "bottle"),
                planar_distance(v_final.root.global_pose.position, near_pose.position)
                < 0.9,
            )
        ).first()

        return bottle.root

    @property
    def robot_starting_pose(self):
        return Pose.from_xyz_rpy(18.5, 8)

    @property
    def pickup_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            8.31, 0.82, 0, yaw=np.pi, reference_frame=self.world.root
        )

    @property
    def place_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            0.32, 5.81, 0.588, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @property
    def place_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            0.66, 5.81, 0, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @cached_property
    def main_entrance(self):
        room = variable(RoomWithWallsAndDoors, self.world.semantic_annotations)
        store = an(
            entity(room).where(contains(room.room_type, "grocery_store"))
        ).first()
        main_entrance: DoorWithType = an(
            entity(v := variable_from(store.doors)).where(
                contains(v.type_description, "main")
            )
        ).first()
        return main_entrance


@dataclass
class Sage10kAmericanBuffetDemo(Sage10kAbstractDemo):
    scene_url = Sage10kNonShittyScenes.AMERICAN_BUFFET_RESTAURANT

    @property
    def plan(self):
        arm = Arms.RIGHT
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            ApproachDirection.LEFT,
            VerticalAlignment.NoAlignment,
            self.robot.arm.manipulator,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)
        navigate = Pose.from_xyz_rpy(x=5.14, y=2.85, reference_frame=self.world.root)
        # self.robot.root. = self.robot_starting_pose

        plan = sequential(
            [
                open_door,
                ParkArmsAction(Arms.BOTH),
                MoveAndPickUpAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                NavigateAction(target_location=navigate),
                ParkArmsAction(Arms.BOTH),
                MoveAndPlaceAction(
                    object_designator=self.world_P_object_of_interest,
                    standing_position=self.place_navigation_pose,
                    arm=arm,
                    target_location=self.place_pose,
                ),
            ],
            context=context,
        ).plan
        return plan

    @property
    def robot_starting_pose(self):
        return Pose.from_xyz_rpy(3.00, 15.00, reference_frame=self.world.root)

    @property
    def world_P_object_of_interest(self) -> Body:
        @symbolic_function
        def planar_distance(point1: Point3, point2: Point3):
            return point1.euclidean_distance(point2)

        pose = Pose.from_xyz_rpy(x=4.06, y=8.64, reference_frame=self.world.root)
        v_table = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        v_cup = variable(
            NaturalLanguageDescriptionWithTypeDescription,
            self.world.semantic_annotations,
        )
        table = an(entity(v_table)).where(
            v_table.type_description == "table",
            planar_distance(v_table.root.global_pose.position, pose.position) < 0.9,
        )

        cup = (
            an(entity(v_cup)).where(
                contains(v_cup.type_description, "cup"),
                is_supported_by(v_cup.root, table.root, 0.05),
            )
        ).first()
        return cup.root

    @property
    def pickup_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(4.66, 8.62, 0, yaw=0, reference_frame=self.world.root)

    @property
    def place_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            7.61, 0.997, 0.6, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @property
    def place_navigation_pose(self) -> Pose:
        return Pose.from_xyz_rpy(
            7.23, 1.16, 0, yaw=np.pi / 2, reference_frame=self.world.root
        )

    @cached_property
    def main_entrance(self):
        room = variable(RoomWithWallsAndDoors, self.world.semantic_annotations)
        gym = an(
            entity(room).where(contains(room.room_type, "buffet restaurant"))
        ).first()
        main_entrance: DoorWithType = an(
            entity(v := variable_from(gym.doors)).where(
                contains(v.type_description, "main")
            )
        ).first()
        return main_entrance

    def preprocess_world(self):
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
