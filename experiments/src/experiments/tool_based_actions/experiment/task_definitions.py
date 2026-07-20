"""
Per-task scene and action construction for the tool-based action experiment.

Every :class:`ToolTaskDefinition` knows how to attach its tool to the robot, how to
spawn one target at a sampled placement, and how to build the action acting on that
target. The trial runner stays task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Bread,
    CuttingKnife,
    PouringCup,
    Sponge,
    Tool,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Optional

from coraplex.datastructures.enums import Arms, CuttingTechnique
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.testing import attach_tool

from experiments.tool_based_actions.experiment.configuration import ToolBasedTask
from experiments.tool_based_actions.experiment.scene import TargetPlacement
from experiments.tool_based_actions.simple_demo.demo_world import (
    BOWL_COLOR,
    BREAD_COLOR,
    CUP_COLOR,
    CUT_MOUNT,
    MIX_MOUNT,
    POUR_MOUNT,
    attach_sponge,
    parse_object,
)


@dataclass(frozen=True)
class ExperimentTarget:
    """
    One spawned target of a trial, addressable by the tool action.
    """

    placement: TargetPlacement
    """
    The sampled placement the target was spawned at.
    """

    pose: Pose
    """
    Pose of the target in the world frame.
    """

    body: Optional[Body] = None
    """
    The spawned body, or None for targets that are pure poses (e.g. wiping patches).
    """


@dataclass
class ToolTaskDefinition(ABC):
    """
    Scene and action construction of one tool-based task.
    """

    arm: Arms = Arms.RIGHT
    """
    The arm the tool is mounted on.
    """

    @abstractmethod
    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        """
        Attach the task's tool to the robot.

        :param world: The world the robot lives in.
        :param robot: The robot performing the task.
        :return: The attached tool annotation.
        """

    @abstractmethod
    def spawn_target(
        self, world: World, placement: TargetPlacement
    ) -> ExperimentTarget:
        """
        Spawn one target at the given placement.

        :param world: The world to spawn into.
        :param placement: The sampled placement of the target.
        :return: The spawned target.
        """

    @abstractmethod
    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        """
        :param target: The target the action acts on.
        :param tool: The tool attached by :meth:`attach_tool`.
        :return: The tool action acting on the target.
        """

    def _spawn_mesh_body(
        self,
        world: World,
        placement: TargetPlacement,
        mesh_file_name: str,
        color,
    ) -> Body:
        """
        Spawn a mesh object at the placement under the placement's unique name.

        :param world: The world to spawn into.
        :param placement: The sampled placement of the object.
        :param mesh_file_name: Mesh file in the demo resources.
        :param color: Color the mesh's visual shapes are dyed with.
        :return: The spawned body inside ``world``.
        """
        object_world = parse_object(mesh_file_name, color=color)
        object_world.root.name = PrefixedName(placement.name)
        with world.modify_world():
            world.merge_world_at_pose(
                object_world,
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    placement.x,
                    placement.y,
                    placement.z,
                    yaw=placement.yaw,
                    reference_frame=world.root,
                ),
            )
        return world.get_body_by_name(placement.name)

    def _placement_pose(self, world: World, placement: TargetPlacement) -> Pose:
        """
        :param world: The world the pose is expressed in.
        :param placement: The sampled placement.
        :return: The placement as a pose in the world frame.
        """
        return Pose.from_xyz_rpy(
            placement.x,
            placement.y,
            placement.z,
            yaw=placement.yaw,
            reference_frame=world.root,
        )


@dataclass
class CuttingTaskDefinition(ToolTaskDefinition):
    """
    Cut a spawned bread with a knife.
    """

    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        knife_body = attach_tool(
            world, robot, self.arm, parse_object("big-knife.stl"), CUT_MOUNT
        )
        knife = CuttingKnife(root=knife_body)
        with world.modify_world():
            world.add_semantic_annotations([knife])
        return knife

    def spawn_target(
        self, world: World, placement: TargetPlacement
    ) -> ExperimentTarget:
        body = self._spawn_mesh_body(world, placement, "bread.stl", BREAD_COLOR)
        with world.modify_world():
            world.add_semantic_annotations([Bread(root=body)])
        return ExperimentTarget(
            placement=placement, pose=self._placement_pose(world, placement), body=body
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return CuttingAction(
            object_to_cut=target.body,
            arm=self.arm,
            tool=tool,
            technique=CuttingTechnique.SLICE,
            number_of_cuts_on_local_x_axis=3,
            slice_thickness=0.03,
        )


@dataclass
class MixingTaskDefinition(ToolTaskDefinition):
    """
    Mix the contents of a spawned bowl with a whisk.
    """

    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        whisk_body = attach_tool(
            world, robot, self.arm, parse_object("whisk.stl"), MIX_MOUNT
        )
        whisk = Whisk(root=whisk_body)
        with world.modify_world():
            world.add_semantic_annotations([whisk])
        return whisk

    def spawn_target(
        self, world: World, placement: TargetPlacement
    ) -> ExperimentTarget:
        body = self._spawn_mesh_body(world, placement, "bowl.stl", BOWL_COLOR)
        with world.modify_world():
            world.add_semantic_annotations([Bowl(root=body)])
        return ExperimentTarget(
            placement=placement, pose=self._placement_pose(world, placement), body=body
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return MixingAction(container=target.body, arm=self.arm, tool=tool)


@dataclass
class PouringTaskDefinition(ToolTaskDefinition):
    """
    Pour from a held cup into a spawned bowl.
    """

    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        cup_body = attach_tool(
            world,
            robot,
            self.arm,
            parse_object("jeroen_cup.stl", color=CUP_COLOR),
            POUR_MOUNT,
        )
        cup = PouringCup(root=cup_body)
        with world.modify_world():
            world.add_semantic_annotations([cup])
        return cup

    def spawn_target(
        self, world: World, placement: TargetPlacement
    ) -> ExperimentTarget:
        body = self._spawn_mesh_body(world, placement, "bowl.stl", BOWL_COLOR)
        with world.modify_world():
            world.add_semantic_annotations([Bowl(root=body)])
        return ExperimentTarget(
            placement=placement, pose=self._placement_pose(world, placement), body=body
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return PouringAction(
            target_container=target.body, source_container=tool, arm=self.arm
        )


@dataclass
class WipingTaskDefinition(ToolTaskDefinition):
    """
    Wipe a patch of the counter around a sampled pose with a sponge.
    """

    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        sponge_body = attach_sponge(world, robot, self.arm)
        sponge = Sponge(root=sponge_body)
        with world.modify_world():
            world.add_semantic_annotations([sponge])
        return sponge

    def spawn_target(
        self, world: World, placement: TargetPlacement
    ) -> ExperimentTarget:
        return ExperimentTarget(
            placement=placement, pose=self._placement_pose(world, placement)
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return WipingAction(arm=self.arm, tool=tool, target_pose=target.pose)


def definition_for_task(task: ToolBasedTask) -> ToolTaskDefinition:
    """
    :param task: The task to run.
    :return: The definition constructing scenes and actions for the task.
    """
    definitions: Dict[ToolBasedTask, ToolTaskDefinition] = {
        ToolBasedTask.CUTTING: CuttingTaskDefinition(),
        ToolBasedTask.MIXING: MixingTaskDefinition(),
        ToolBasedTask.POURING: PouringTaskDefinition(),
        ToolBasedTask.WIPING: WipingTaskDefinition(),
    }
    return definitions[task]
