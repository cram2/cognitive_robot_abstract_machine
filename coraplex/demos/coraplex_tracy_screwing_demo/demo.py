"""
Tracy joint-driven screwing demo for the Coraplex statechart API.

Provides two scenes without ROS dependencies so they can run headless in tests or be
driven by the RViz runner: :class:`NutScene` (hand screwing) and :class:`AirplaneScene`
(pick, insert, and tighten a bolt with a :class:`Screwdriver`).
"""

from __future__ import annotations

import argparse
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.screwing import (
    InsertScrewAction,
    ScrewThreadAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.executor import NoPacing
from cramph.statechart import Statechart
from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bolt,
    Screwdriver,
)
from semantic_digital_twin.specifications.connections import (
    ScrewConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color, Cylinder, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene configuration


class ScrewingSceneKind(StrEnum):
    """
    The scenes available in the Tracy screwing demo.
    """

    NUT = "nut"
    """
    Single bolt threaded onto a stationary nut plate and driven by hand.
    """

    AIRPLANE = "airplane"
    """
    Toy airplane assembly where Tracy inserts a bolt and tightens it with a screwdriver.
    """


@dataclass(kw_only=True, frozen=True)
class MeshPlacement:
    """
    Mesh file name together with its local position and yaw-pitch-roll orientation.
    """

    mesh_filename: str
    """
    Relative path of the STL mesh inside the demo resources directory.
    """

    position: tuple[float, float, float]
    """
    Local ``(x, y, z)`` offset in metres.
    """

    rotation: tuple[float, float, float]
    """
    Local ``(yaw, pitch, roll)`` orientation in radians.
    """


# %% scene builders


@dataclass(kw_only=True)
class ScrewingScene(ABC):
    """
    Base scene builder for the Tracy screwing demo.
    """

    screw_pitch: float = 0.002
    """
    Distance in metres the bolt advances along its thread per full revolution.
    """

    screw_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """
    Unit axis of the screw joint pointing from counterpart toward the bolt head.
    """

    maximum_joint_speed: float = 1.0
    """
    Velocity limit in rad/s assigned to spawned screw joints.
    """

    bolt_mesh: str = "Bolt.stl"
    """
    Relative path of the bolt mesh inside :attr:`resources_directory`.
    """

    @classproperty
    def resources_directory(cls) -> Path:
        """
        :return: Directory containing the STL meshes used by this demo.
        """
        return Path(__file__).resolve().parent / "resources"

    def build_world(self) -> tuple[World, Tracy]:
        """
        Load Tracy from its URDF and park both arms in their starting joint state.

        :return: The initialized world and Tracy robot annotation.
        """
        world = URDFParser.from_file(Tracy.get_ros_file_path()).create_world()
        tracy = Tracy.from_world(world)
        for arm in tracy.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        return world, tracy

    def spawn_mesh_body(
        self,
        *,
        world: World,
        mesh_filename: str,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        parent: Body | None = None,
        name: str | None = None,
    ) -> HasGraspCandidates:
        """
        Spawn a fixed mesh body into ``world`` with a graspable semantic annotation.

        :param world: The world to add the body to.
        :param mesh_filename: Relative mesh path inside :attr:`resources_directory`.
        :param position: ``(x, y, z)`` offset in metres relative to ``parent``.
        :param rotation: ``(yaw, pitch, roll)`` orientation in radians.
        :param parent: Parent body; defaults to ``world.root``.
        :param name: Optional body name; defaults to the mesh stem.
        :return: The :class:`HasGraspCandidates` annotation for the spawned body.
        """
        parent_body = world.root if parent is None else parent
        mesh_path = self.resources_directory / mesh_filename
        with world.modify_world():
            mesh_shape = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(mesh_path),
            )
            shapes = ShapeCollection(shapes=[mesh_shape])
            body = Body(
                name=PrefixedName(name=name or mesh_path.stem, prefix="objects"),
                collision=shapes,
                visual=shapes,
            )
            world.add_kinematic_structure_entity(body)
            world.add_connection(
                FixedConnection(
                    parent=parent_body,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=position[0],
                        y=position[1],
                        z=position[2],
                        yaw=rotation[0],
                        pitch=rotation[1],
                        roll=rotation[2],
                    ),
                )
            )
            graspable = HasGraspCandidates(root=body)
            world.add_semantic_annotations([graspable])
        return graspable

    def spawn_screw_body(
        self,
        *,
        world: World,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        parent: Body,
        turns: float,
        name: str = "Bolt",
    ) -> Bolt:
        """
        Spawn a :class:`Bolt` connected to ``parent`` by a single-DoF screw joint.

        :param world: The world to add the bolt to.
        :param position: ``(x, y, z)`` offset in metres in ``parent``'s frame.
        :param rotation: ``(yaw, pitch, roll)`` orientation in radians.
        :param parent: Counterpart body the bolt is threaded into.
        :param turns: Maximum inward revolutions defining the joint's lower limit.
        :param name: Name assigned to the bolt body.
        :return: The spawned :class:`Bolt` semantic annotation.
        """
        mesh_path = self.resources_directory / self.bolt_mesh
        dof_limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=-2.0 * math.pi * turns,
            upper_position=0.0,
            maximum_speed=self.maximum_joint_speed,
        )
        with world.modify_world():
            mesh_shape = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(mesh_path),
            )
            shapes = ShapeCollection(shapes=[mesh_shape])
            body = Body(
                name=PrefixedName(name=name, prefix="objects"),
                collision=shapes,
                visual=shapes,
            )
            world.add_kinematic_structure_entity(body)
            ScrewConnectionSpecification(
                axis=Vector3(*self.screw_axis),
                screw_pitch=self.screw_pitch,
                dof_limits=dof_limits,
            ).connect(
                world,
                child=body,
                parent=parent,
                parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                    yaw=rotation[0],
                    pitch=rotation[1],
                    roll=rotation[2],
                ),
            )
            bolt = Bolt(root=body)
            world.add_semantic_annotations([bolt])
        return bolt

    def spawn_screwdriver(
        self,
        *,
        world: World,
        position: tuple[float, float, float],
        length: float,
        diameter: float,
        name: str = "screwdriver",
    ) -> Screwdriver:
        """
        Spawn a cylinder-handled :class:`Screwdriver` with a dedicated tip frame.

        :param world: The world to add the screwdriver to.
        :param position: ``(x, y, z)`` starting coordinates in world frame.
        :param length: Total screwdriver length along local Z in metres.
        :param diameter: Handle cylinder diameter in metres.
        :param name: Base name for the screwdriver bodies and connections.
        :return: The spawned :class:`Screwdriver` semantic annotation.
        """
        with world.modify_world():
            cylinder = Cylinder(
                origin=HomogeneousTransformationMatrix(),
                width=diameter,
                height=length,
                color=Color.from_rgba(red=0.2, green=0.5, blue=0.8, opacity=1.0),
            )
            shapes = ShapeCollection(shapes=[cylinder])
            handle_body = Body(
                name=PrefixedName(name=name, prefix="objects"),
                collision=shapes,
                visual=shapes,
            )
            world.add_kinematic_structure_entity(handle_body)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=world.root,
                    child=handle_body,
                    name=PrefixedName(name=f"{name}_connection", prefix="connections"),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=position[0],
                        y=position[1],
                        z=position[2],
                    ),
                )
            )
            tip_body = Body(
                name=PrefixedName(name=f"{name}_end_effector", prefix="objects")
            )
            world.add_kinematic_structure_entity(tip_body)
            world.add_connection(
                FixedConnection(
                    parent=handle_body,
                    child=tip_body,
                    name=PrefixedName(
                        name=f"{name}_tip_connection", prefix="connections"
                    ),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.0,
                        y=0.0,
                        z=-length / 2.0,
                        pitch=math.pi,
                        reference_frame=handle_body,
                    ),
                )
            )
            screwdriver = Screwdriver(root=handle_body, grasp_height=length / 4.0)
            world.add_semantic_annotations([screwdriver])
        return screwdriver

    @abstractmethod
    def build_statechart(
        self, *, context: StatechartContext, tracy: Tracy
    ) -> Statechart:
        """
        Populate ``context``'s world and construct the screwing statechart.

        :param context: Statechart execution context holding the world.
        :param tracy: The Tracy robot performing the plan.
        :return: The constructed statechart.
        """


@dataclass(kw_only=True)
class NutScene(ScrewingScene):
    """
    Minimal scene where Tracy drives a bolt directly into a stationary nut plate.
    """

    screw_pose: tuple[float, float, float] = (0.55, 0.0, 1.0)
    """
    World position of the bolt joint in metres.
    """

    plate_mesh: str = "Nut.stl"
    """
    Mesh file for the stationary nut plate.
    """

    bolt_in_plate: tuple[float, float, float] = (0.022, 0.0, 0.0)
    """
    Bolt origin offset relative to the nut plate in metres.
    """

    grasp_offset: float = 0.030
    """
    Grasp offset along the bolt's local +X axis in metres.
    """

    turns: float = 0.5
    """
    Revolutions to screw the bolt into the nut plate.
    """

    def build_statechart(
        self, *, context: StatechartContext, tracy: Tracy
    ) -> Statechart:
        world = context.world
        plate = self.spawn_mesh_body(
            world=world,
            mesh_filename=self.plate_mesh,
            position=(
                self.screw_pose[0] - self.bolt_in_plate[0],
                self.screw_pose[1] - self.bolt_in_plate[1],
                self.screw_pose[2] - self.bolt_in_plate[2],
            ),
            rotation=(0.0, 0.0, 0.0),
        )
        bolt = self.spawn_screw_body(
            world=world,
            position=self.bolt_in_plate,
            rotation=(0.0, 0.0, 0.0),
            parent=plate.root,
            turns=self.turns,
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    ParkArmsAction(arms=tracy.all_arms),
                    ScrewThreadAction(
                        arm=tracy.right_arm,
                        screw_body=bolt.root,
                        turns=self.turns,
                        grasp_offset=self.grasp_offset,
                    ),
                    ParkArmsAction(arms=tracy.all_arms),
                ]
            )
        )
        return statechart


@dataclass(kw_only=True)
class AirplaneScene(ScrewingScene):
    """
    Assembly scene where Tracy picks a bolt, inserts it into the wing, and tightens it
    with a screwdriver.
    """

    parts: tuple[MeshPlacement, ...] = (
        MeshPlacement(
            mesh_filename="HolderUnderBody.stl",
            position=(0.0, 0.0, -0.12),
            rotation=(0.0, 0.0, 0.0),
        ),
        MeshPlacement(
            mesh_filename="plane/UnderBody.stl",
            position=(0.0, -0.12, -0.07),
            rotation=(0.0, 3.0, 0.0),
        ),
        MeshPlacement(
            mesh_filename="plane/UpperBody.stl",
            position=(0.0, -0.08, -0.05),
            rotation=(-1.5708, 0.0, 0.0),
        ),
        MeshPlacement(
            mesh_filename="plane/RearWing.stl",
            position=(0.0, 0.0, -0.12),
            rotation=(0.0, 0.0, 0.0),
        ),
    )
    """
    Fixed parts forming the airplane assembly and its cradle.
    """

    assembly_position: tuple[float, float, float] = (0.55, 0.0, 1.0)
    """
    World position of the airplane assembly origin in metres.
    """

    assembly_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    World ``(yaw, pitch, roll)`` orientation of the airplane assembly in radians.
    """

    screw_parent_name: str = "RearWing"
    """
    Name of the assembly part body the bolt is inserted into.
    """

    screw_position: tuple[float, float, float] = (0.0, -0.032, 0.06)
    """
    Target bolt position in :attr:`screw_parent_name`'s frame in metres.
    """

    screw_rotation: tuple[float, float, float] = (0.0, -math.pi / 2.0, 0.0)
    """
    Target bolt ``(yaw, pitch, roll)`` orientation relative to the parent part.
    """

    turns: float = 0.5
    """
    Total revolutions the bolt stands inwards after driving with the screwdriver.
    """

    holder_position: tuple[float, float, float] = (0.55, -0.25, 0.88)
    """
    World position of the bolt holder on the right side of the table in metres.
    """

    holder_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    World ``(yaw, pitch, roll)`` orientation of the bolt holder in radians.
    """

    holder_mesh: str = "BoltHolder.stl"
    """
    Mesh file for the bolt holder.
    """

    bolt_in_holder_position: tuple[float, float, float] = (-0.025, 0.0, 0.025)
    """
    Initial bolt position inside the bolt holder in metres.
    """

    bolt_in_holder_rotation: tuple[float, float, float] = (0.0, -math.pi / 2.0, 0.0)
    """
    Initial bolt ``(yaw, pitch, roll)`` orientation inside the bolt holder.
    """

    screwdriver_length: float = 0.10
    """
    Length of the screwdriver handle cylinder in metres.
    """

    screwdriver_diameter: float = 0.03
    """
    Diameter of the screwdriver handle cylinder in metres.
    """

    screwdriver_offset: tuple[float, float, float] = (0.025, 0.0, 0.025)
    """
    Offset of the screwdriver's lower tip from :attr:`holder_position` in metres.
    """

    turn_by_hand: bool = True
    """
    Whether Tracy turns the bolt slightly by hand right after inserting it.
    """

    hand_turns: float = 0.1
    """
    Revolutions turned by hand during :class:`InsertScrewAction`.
    """

    def build_statechart(
        self, *, context: StatechartContext, tracy: Tracy
    ) -> Statechart:
        world = context.world
        parts: dict[str, HasGraspCandidates] = {}
        for placement in self.parts:
            part = self.spawn_mesh_body(
                world=world,
                mesh_filename=placement.mesh_filename,
                position=(
                    self.assembly_position[0] + placement.position[0],
                    self.assembly_position[1] + placement.position[1],
                    self.assembly_position[2] + placement.position[2],
                ),
                rotation=(
                    self.assembly_rotation[0] + placement.rotation[0],
                    self.assembly_rotation[1] + placement.rotation[1],
                    self.assembly_rotation[2] + placement.rotation[2],
                ),
            )
            parts[Path(placement.mesh_filename).stem] = part

        plate = parts[self.screw_parent_name]
        holder = self.spawn_mesh_body(
            world=world,
            mesh_filename=self.holder_mesh,
            position=self.holder_position,
            rotation=self.holder_rotation,
        )
        bolt = self.spawn_screw_body(
            world=world,
            position=self.bolt_in_holder_position,
            rotation=self.bolt_in_holder_rotation,
            parent=holder.root,
            turns=self.turns,
        )
        screwdriver = self.spawn_screwdriver(
            world=world,
            position=(
                self.holder_position[0] + self.screwdriver_offset[0],
                self.holder_position[1] + self.screwdriver_offset[1],
                self.holder_position[2]
                + self.screwdriver_offset[2]
                + self.screwdriver_length / 2.0,
            ),
            length=self.screwdriver_length,
            diameter=self.screwdriver_diameter,
        )
        target_location = Pose.from_xyz_rpy(
            self.screw_position[0],
            self.screw_position[1],
            self.screw_position[2],
            yaw=self.screw_rotation[0],
            pitch=self.screw_rotation[1],
            roll=self.screw_rotation[2],
            reference_frame=plate.root,
        )
        bolt_grasp = bolt.grasp_candidates()[0]
        screwdriver_grasp = screwdriver.grasp_candidates()[0]

        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    ParkArmsAction(arms=tracy.all_arms),
                    PickUpAction(grasp=bolt_grasp, arm=tracy.right_arm),
                    InsertScrewAction(
                        arm=tracy.right_arm,
                        grasp=bolt_grasp,
                        counterpart=plate.root,
                        target_location=target_location,
                        turn_by_hand=self.turn_by_hand,
                        hand_turns=self.hand_turns,
                    ),
                    PickUpAction(grasp=screwdriver_grasp, arm=tracy.right_arm),
                    ScrewThreadAction(
                        arm=tracy.right_arm,
                        screw_body=bolt.root,
                        tool_grasp=screwdriver_grasp,
                        turns=self.turns,
                    ),
                    ParkArmsAction(arms=tracy.all_arms),
                ]
            )
        )
        return statechart


# %% runners


@dataclass(kw_only=True)
class ScrewingDemoRunner(ABC):
    """
    Base runner executing a :class:`ScrewingScene` plan.
    """

    scene: ScrewingScene = field(default_factory=AirplaneScene)
    """
    The scene to build and run.
    """

    simulation_time_limit: timedelta = timedelta(minutes=10)
    """
    Maximum simulated time budget per action before aborting.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Build the scene, compile the statechart, and run the screwing plan.
        """


@dataclass(kw_only=True)
class HeadlessRunner(ScrewingDemoRunner):
    """
    Executes the screwing demo kinematically without ROS visualization.
    """

    def run(self) -> None:
        world, tracy = self.scene.build_world()
        context = StatechartContext.with_extensions(
            world=world, extensions=[RobotAccess(robot=tracy)]
        )
        statechart = self.scene.build_statechart(context=context, tracy=tracy)
        executor = KinematicallySimulatedPlanExecutor(
            context=context,
            simulation_time_per_action=self.simulation_time_limit,
            pacer=NoPacing(),
        )
        try:
            executor.compile(statechart=statechart)
            executor.execute()
        finally:
            executor.close()


def build_scene(
    *, scene_kind: ScrewingSceneKind, turn_by_hand: bool = True
) -> ScrewingScene:
    """
    Create the requested screwing demo scene.

    :param scene_kind: Which scene to construct.
    :param turn_by_hand: Whether :class:`AirplaneScene` gives an initial hand turn.
    :return: Configured scene instance.
    """
    if scene_kind is ScrewingSceneKind.NUT:
        return NutScene()
    return AirplaneScene(turn_by_hand=turn_by_hand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=tuple(ScrewingSceneKind),
        default=ScrewingSceneKind.AIRPLANE.value,
        help="scene to run: nut or airplane (default: airplane)",
    )
    parser.add_argument(
        "--turn-by-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to turn the bolt by hand during insertion (default: True)",
    )
    arguments = parser.parse_args()
    selected_scene = build_scene(
        scene_kind=ScrewingSceneKind(arguments.scene),
        turn_by_hand=arguments.turn_by_hand,
    )
    HeadlessRunner(scene=selected_scene).run()
