from __future__ import annotations

from functools import cached_property

import numpy as np

from giskardpy.motion_statechart.goals.open_close import Open
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import *
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.plans.plan import Plan
from pycram.robot_plans import BaseMotion
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.composite.sage10k_actions import Sage10kOpenDoor
from pycram.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
)
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kNonShittyScenes,
    NaturalLanguageDescriptionWithTypeDescription,
    RoomWithWallsAndDoors,
    DoorWithType,
)
from semantic_digital_twin.reasoning.predicates import compute_euclidean_planar_distance
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import Point3, Pose, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenWithHandleMotion(BaseMotion):
    """
    Designator for opening container
    """

    handle: Body

    manipulator: Manipulator

    @property
    def _motion_chart(self):
        return Open(tip_link=self.manipulator.tool_frame, environment_link=self.handle)


@dataclass
class OpenWithHandleAction(ActionDescription):
    """
    Opens a container like object
    """

    handle: Handle
    manipulator: Manipulator

    def execute(self) -> None: ...


@dataclass
class Sage10kNonShittyScenesDemoConfig:
    """
    Configuration for the Sage10k non-shitty scenes demo.
    Is a Tuple of (scene_url, world_P_object_of_interest, pickup_navigation_pose, place_pose, place_navigation_pose)
    """

    scene_url: str
    """
    The URL of the scene to use for the demo.
    """

    world_P_object_of_interest: Point3
    """
    Approximate position of the object we want to pick up. Must be within 10cm euclidian distance of the actual object,
    and no other object is allowed to be within that radius. If thats a problem to do, chose another object.
    Use the "publish point" functionality in RVIZ to to get the coordinates: click the button, hover over the center of 
    the object you want to grasp, and read the coordinates from the bottom left of RViz, right next to the reset button. 
    """

    pickup_navigation_pose: Pose
    """
    Nav pose from where we want to pick up
    """

    place_pose: Pose
    """
    Pose where we want to place the object. also get this from RViz using publish point on the surface.
    Do not add any object height here, it will be added automatically.
    """

    place_navigation_pose: Pose
    """
    Nav pose from where we want to place the object
    """

    world: Optional[World] = None

    @classmethod
    def GYM(cls):

        return cls(
            scene_url=Sage10kNonShittyScenes.GYM,
            world_P_object_of_interest=Point3(1.03, -0.716, 0.203),
            pickup_navigation_pose=Pose.from_xyz_rpy(0.94, 0.2, 0, yaw=-np.pi / 2),
            place_pose=Pose.from_xyz_rpy(-0.15, 4.55, 0.865, yaw=np.pi / 2),
            place_navigation_pose=Pose.from_xyz_rpy(-0.12, 4, 0, yaw=np.pi / 2),
        )

    def load_world(self):
        loader = Sage10kDatasetLoader()
        world = loader.create_scene(scene_url=self.scene_url).create_world()

        # TODO apply preprocessing

        self.world = world

    @cached_property
    def robot(self) -> HSRB:
        return self.world.get_semantic_annotations_by_type(HSRB)[0]

    def plan(self) -> Plan:
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        [body] = self.world.get_bodies_by_global_position(
            self.world_P_object_of_interest, 0.1
        )
        arm = Arms.RIGHT
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            self.robot.arm.manipulator,
        )

        plan = sequential(
            [
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
class Sage10kGymDemo:
    scene_url: str = Sage10kNonShittyScenes.GYM
    _world: Optional[World] = None

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

    @cached_property
    def world(self) -> World:
        loader = Sage10kDatasetLoader()
        world = loader.create_scene(scene_url=self.scene_url).create_world()

        v = variable(
            NaturalLanguageDescriptionWithTypeDescription, world.semantic_annotations
        )
        self._world = world
        return world

    @cached_property
    def robot(self) -> HSRB:
        arm = Arms.RIGHT
        return world.get_semantic_annotations_by_type(HSRB)[0]

    def preprocess_world(self):
        v = variable(
            NaturalLanguageDescriptionWithTypeDescription, world.semantic_annotations
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

    def remove_rooted_annotations(self, semantic_annotations: Iterable[HasRootBody]):
        with self.world.modify_world():
            for annotation in semantic_annotations:
                world.remove_kinematic_structure_entity(annotation.root)
                world.remove_semantic_annotation(annotation)

    def plan(self):
        self.preprocess_world()
        context = Context.from_world(self.world, query_backend=ProbabilisticBackend())
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot.arm.manipulator,
        )
        open_door = Sage10kOpenDoor(self.main_entrance)

        [body] = world.get_bodies_by_global_position(
            config.world_P_object_of_interest, 0.1
        )

        plan = sequential(
            [
                open_door,
                ParkArmsAction(Arms.BOTH),
                MoveAndPickUpAction(
                    object_designator=body,
                    standing_position=config.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveAndPlaceAction(
                    object_designator=body,
                    standing_position=config.place_navigation_pose,
                    arm=arm,
                    target_location=config.place_pose,
                ),
            ],
            context=context,
        ).plan
