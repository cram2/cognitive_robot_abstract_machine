"""
GARMI transports a bowl and a spoon across the apartment.

The bowl starts on the kitchen counter and the spoon inside a drawer, and both are carried
to the table. Running with
:class:`~coraplex.garmi_franka_grippers.RealGarmiPlanExecutor` drives the
actual robot, its grippers through their Franka gripper driver, and takes the world from
the running world server, merging the apartment into it unless the server holds it
already. The default runs the whole plan in simulation
against a world built from the apartment's MuJoCo scene and GARMI's URDF, so nothing on
the network is needed.

Needs the ``iai_garmi_apartment`` and ``garmi_description`` packages built in the
workspace, since the scene and the robot description are read from their share
directories.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, StrEnum

from ament_index_python.packages import get_package_share_directory

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import PlanExecutor, KinematicallySimulatedPlanExecutor
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforeMoveAndPickUp
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from.
    """

    APARTMENT = os.path.join("mjcf", "scene-bodies.xml")
    """
    The apartment scene GARMI acts in, in the ``iai_garmi_apartment`` package.
    """

    BOWL = os.path.join("objects", "bowl.stl")
    """
    The bowl's mesh, under the coraplex resources.
    """

    SPOON = os.path.join("objects", "spoon.stl")
    """
    The spoon's mesh, under the coraplex resources.
    """

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        if self is SceneFile.APARTMENT:
            return os.path.join(
                get_package_share_directory("iai_garmi_apartment"), self.value
            )
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "resources",
            self.value,
        )


class SceneBody(StrEnum):
    """
    The bodies of the scene the demonstration acts on.
    """

    BOWL = "bowl"
    """
    The transported bowl.
    """

    SPOON = "spoon"
    """
    The transported spoon.
    """

    SPOON_DRAWER = "drawer_1"
    """
    The drawer the spoon starts in.
    """

    APARTMENT = "apartment"
    """
    The apartment's origin, posed in the occupancy map frame the robot localizes in,
    which also marks whether the scene was already populated.
    """


# %% where things stand


@dataclass(frozen=True)
class ScenePlacement:
    """
    A position and a heading about the vertical.
    """

    x: float
    y: float
    z: float
    yaw: float = 0.0
    """
    The heading about the vertical, in radian.
    """

    @property
    def transform(self) -> HomogeneousTransformationMatrix:
        """
        :return: The placement as a transform.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw
        )

    @property
    def position(self) -> Point3:
        """
        :return: Where the placement is, without its heading.
        """
        return Point3.from_iterable([self.x, self.y, self.z])


class ScenePose(Enum):
    """
    Where GARMI and the transported objects start, and where the objects are carried to.
    """

    GARMI_START = ScenePlacement(-4.8, 1.5, 0, yaw=math.pi)
    """
    Where GARMI starts, in its ``odom`` frame: facing the kitchen from the room.

    The simulated world's ``odom`` frame is the occupancy map frame the apartment is posed
    in, not the apartment's own frame the other placements are relative to.
    """

    BOWL_START = ScenePlacement(0.0, 7.2, 0.973)
    """
    Where the bowl starts, on the kitchen counter, relative to the apartment.
    """

    BOWL_TARGET = ScenePlacement(1.6, 5.2, 0.8)
    """
    Where the bowl is carried to, relative to the apartment.
    """

    SPOON_START = ScenePlacement(-0.265, 0.0, -0.033)
    """
    Where the spoon starts, relative to its drawer: lying on the drawer's bottom plate,
    clear of its walls.

    The spoon only becomes a body collisions are checked against once it is grasped, so a
    placement that reaches through a wall goes unnoticed until the pick-up aborts on it.
    """

    SPOON_TARGET = ScenePlacement(1.6, 5.4, 0.8)
    """
    Where the spoon is carried to, relative to the apartment.
    """


# %% how the run is set up


@dataclass(frozen=True)
class DriveVelocityLimits:
    """
    How fast a base may move.
    """

    translation: float
    """
    In meter per second.
    """

    rotation: float
    """
    In radian per second.
    """


class GarmiDrive(Enum):
    """
    How fast GARMI's base moves in this demonstration.
    """

    DEMONSTRATION = DriveVelocityLimits(translation=0.1, rotation=0.1)


# %% the demonstration


@dataclass
class GarmiApartmentDemonstration(RobotDemonstration):
    """
    GARMI carries a bowl off the kitchen counter and a spoon out of a drawer, and places
    both on the table.
    """

    ros_node_name: str = "garmi_demo_node"

    def scene_specification(self) -> WorldSpecification:
        """
        The scene keeps its collision geometry in a file that is not loaded, so every
        geom has to stand in for it.

        :return: The apartment's MuJoCo scene.
        """
        return WorldSpecification.from_mjcf(
            SceneFile.APARTMENT.path, use_visual_as_collision_backup=True
        )

    def robot_specification(self) -> RobotSpecification:
        """
        :return: GARMI at its start, driving at the demonstration's speed.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=ScenePose.GARMI_START.value.transform,
            drive_translation_velocity_limits=GarmiDrive.DEMONSTRATION.value.translation,
            drive_rotation_velocity_limits=GarmiDrive.DEMONSTRATION.value.rotation,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneBody.APARTMENT)

    def populate_scene(self, world: World) -> None:
        """
        Annotate the apartment's furniture, then add the bowl and the spoon.

        The furniture is annotated first, so the reasoner describes the apartment the
        plan navigates rather than the two objects the plan already knows.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        # %% bowl

        # Both objects are picked up and carried, and a pose can only be written to a
        # connection that has the degrees of freedom to carry it, so they hang off 6DoF
        # connections rather than the default fixed one.
        Bowl.get_annotation_specification(
            SceneBody.BOWL,
            BodySpecification.mesh(
                SceneBody.BOWL,
                SceneFile.BOWL.path,
                parent_T_self=ScenePose.BOWL_START.value.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.get_body_by_name(SceneBody.APARTMENT))

        # %% spoon

        # Hanging off the drawer rather than the world root, so it travels with the drawer
        # when the plan opens it.
        Spoon.get_annotation_specification(
            SceneBody.SPOON,
            BodySpecification.mesh(
                SceneBody.SPOON,
                SceneFile.SPOON.path,
                parent_T_self=ScenePose.SPOON_START.value.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.get_body_by_name(SceneBody.SPOON_DRAWER))

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
            StatementGrounding(),
            PlanRewriting(transformations=[OpenDrawerBeforeMoveAndPickUp()]),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Carry the bowl and then the spoon to the table.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        right_arm = robot.get_right_arm_if_specified()
        apartment = world.get_body_by_name(SceneBody.APARTMENT)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    TransportAction.from_graspable_by_closest_grasps(
                        world.get_semantic_annotations_by_type(Bowl)[0],
                        Pose(
                            position=ScenePose.BOWL_TARGET.value.position,
                            reference_frame=apartment,
                        ),
                        right_arm,
                        context,
                    ),
                    TransportAction.from_graspable_by_closest_grasps(
                        world.get_semantic_annotations_by_type(Spoon)[0],
                        Pose(
                            position=ScenePose.SPOON_TARGET.value.position,
                            reference_frame=apartment,
                        ),
                        right_arm,
                        context,
                    ),
                ]
            )
        )
        return statechart


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it
        drives the real robot or simulates it.
    """
    GarmiApartmentDemonstration(
        used_robot=Garmi, executor_type=executor_type, collision_avoidance=True
    ).run()


if __name__ == "__main__":
    main()
