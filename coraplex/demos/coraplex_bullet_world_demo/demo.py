"""
The PR2 lays a place setting: it carries the milk, a bowl and a spoon onto the table.

Runs in simulation against the apartment, so nothing on the network is needed. The
scaffolding in :mod:`coraplex.demonstrations` owns the ROS session and publishes the
world to Rviz, so the run can be watched while it happens.

The bowl is the interesting one. It offers a grasp all around its rim and only some of
them can be reached from anywhere the robot may stand, so the plan names its grasp as a
variable rather than letting the action take the first one the bowl generates. The
domain is worked out when that transport grounds -- by which time the milk has been put
down and the robot has moved -- rather than when the plan is built.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Optional, Tuple, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.factories import ReachableGrasps
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from krrood.entity_query_language.factories import a, an, entity, variable
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Handle,
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
    SPOON_DRAWER_HANDLE = "handle_cab10_t"


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
    How high the object's origin is laid, which rests it on the table surface at z=0.723
    rather than in it.
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
    Where the PR2 stands before the plan begins.

    Far enough back that its parked grippers clear the counter: a meter and a half in,
    they sit inside cabinet9 and cabinet10, which a run with collision avoidance refuses
    to start from. Behind 1.0 m the torso meets the cabinet doors on the other side
    instead.
    """

    milk: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Milk,
            SceneFile.MILK,
            HomogeneousTransformationMatrix.from_xyz_rpy(2.37, 2, 1.05),
            along_table=4.8,
            height=0.82,
        )
    )
    """
    The milk, which starts on the counter.
    """

    bowl: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Bowl,
            SceneFile.BOWL,
            HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.2, 1),
            along_table=5.0,
            height=0.76,
        )
    )
    """
    The bowl, which starts on the counter and is the one whose grasp the plan chooses.
    """

    spoon: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Spoon,
            SceneFile.SPOON,
            HomogeneousTransformationMatrix.from_xyz_rpy(-0.05, -0.05, 0),
            along_table=5.2,
            height=0.74,
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
        :return: The objects carried onto the table, in the order they are carried.

        They are laid 20 cm apart along the table, which clears the widest footprint of
        the set -- the bowl's 14 cm -- plus the clearance the arm keeps while it reaches
        between them. Closer together, placing one object drives the gripper into the
        buffer zone around the one already standing there.
        """
        return self.milk, self.bowl, self.spoon

    def build_simulated_world(self) -> World:
        """
        The apartment with the PR2 in it, stood back from the counter.

        The robot is placed by its localization frame rather than by editing its parent
        connection afterwards, and comes out of :meth:`RobotSpecification.spawn` already
        registered among the world's annotations.
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
        Put the place setting where it starts out, and name what the plan acts on.

        The drawer and its handle are bodies the apartment already brought, so they are
        annotated where they stand instead of being spawned.
        """
        for placed_object in self.place_setting:
            placed_object.spawn(world)

        with world.modify_world():
            WorldReasoner(world).reason()
            world.add_semantic_annotation_recursively(
                Drawer(
                    root=world.get_body_by_name(ApartmentBody.SPOON_DRAWER),
                    handle=Handle(
                        root=world.get_body_by_name(ApartmentBody.SPOON_DRAWER_HANDLE)
                    ),
                )
            )

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            _debug=True,
            sampling_seed=0,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry each object to its place on the table.
        """
        world = context.world
        bowl = self.bowl.annotation_in(world)
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction(
                    self.milk.annotation_in(world),
                    Arms.LEFT,
                    target_location=self.milk.target_location(world),
                ),
                a(TransportAction)(
                    object_designator=bowl,
                    target_location=self.bowl.target_location(world),
                    arm=Arms.LEFT,
                    grasp_pose=variable(
                        Pose, domain=ReachableGrasps(bowl, context, Arms.LEFT)
                    ),
                ),
                TransportAction(
                    self.spoon.annotation_in(world),
                    Arms.LEFT,
                    target_location=self.spoon.target_location(world),
                ),
            ],
            context=context,
        ).plan


def main(
    execution_type: ExecutionType = ExecutionType.SIMULATED,
    collision_avoidance: bool = True,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param collision_avoidance: Whether every motion state chart avoids collisions.
    """
    BulletWorldDemonstration(
        used_robot=PR2,
        execution_type=execution_type,
        collision_avoidance=collision_avoidance,
    ).run()


if __name__ == "__main__":
    main()
