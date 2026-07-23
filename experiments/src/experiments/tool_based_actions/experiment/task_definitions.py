"""
Per-task scene and action construction for the tool-based action experiment.

Every :class:`ToolTaskDefinition` knows how to attach its tool to the robot, how to
create its target objects, and how to build the action acting on a target. The trial
runner and the scene spawner stay task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.utils import recursive_subclasses
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
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
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from typing_extensions import Dict, Optional, Type

from coraplex.datastructures.enums import Arms, CuttingTechnique
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.testing import attach_tool

from experiments.tool_based_actions.experiment.scene import ExperimentTarget
from experiments.tool_based_actions.simple_demo.demo_world import (
    BOWL_COLOR,
    BREAD_COLOR,
    CUP_COLOR,
    CUT_MOUNT,
    MIX_MOUNT,
    POUR_MOUNT,
    attach_sponge,
    parse_object,
    spawn_mesh_body,
)


@dataclass
class ToolTaskDefinition(ABC):
    """
    Scene and action construction of one tool-based task.
    """

    arm: Arms = Arms.RIGHT
    """
    The arm the tool is mounted on.
    """

    pointer_stride: int = 10
    """
    Keep every Nth sampled tool path waypoint for execution.
    """

    @classmethod
    def task_name(cls) -> str:
        """
        :return: A compact, human-readable name of the task, used in trial identifiers
            and on the command line.
        """
        return cls.__name__.removesuffix("TaskDefinition").lower()

    @abstractmethod
    def attach_tool(self, world: World, robot: AbstractRobot) -> Tool:
        """
        Attach the task's tool to the robot.

        :param world: The world the robot lives in.
        :param robot: The robot performing the task.
        :return: The attached tool annotation.
        """

    @abstractmethod
    def create_target(
        self, world: World, name: str, scale: float
    ) -> Optional[HasRootBody]:
        """
        Create one target object of this task, ready to be placed by the scene spawner.

        :param world: The world to spawn into.
        :param name: Unique name of the target within its trial.
        :param scale: Uniform scale factor the target is sized with.
        :return: The created target annotation, or None for targets that are pure poses
            (e.g. wiping patches).
        """

    @abstractmethod
    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        """
        :param target: The target the action acts on.
        :param tool: The tool attached by :meth:`attach_tool`.
        :return: The tool action acting on the target.
        """

    def _create_mesh_target(
        self,
        world: World,
        name: str,
        scale: float,
        mesh_file_name: str,
        color: Color,
        annotation_type: Type[HasRootBody],
    ) -> HasRootBody:
        """
        Spawn a mesh object at the world origin and annotate it, ready to be placed.

        :param world: The world to spawn into.
        :param name: Unique name of the target.
        :param scale: Uniform scale factor the mesh is sized with.
        :param mesh_file_name: Mesh file in the demo resources.
        :param color: Color the mesh's visual shapes are dyed with.
        :param annotation_type: The semantic annotation the target body is wrapped in.
        :return: The created target annotation.
        """
        body = spawn_mesh_body(
            world,
            mesh_file_name,
            HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=world.root),
            color=color,
            name=name,
            scale=Scale(scale, scale, scale),
        )
        annotation = annotation_type(root=body)
        with world.modify_world():
            world.add_semantic_annotations([annotation])
        return annotation


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

    def create_target(
        self, world: World, name: str, scale: float
    ) -> Optional[HasRootBody]:
        return self._create_mesh_target(
            world, name, scale, "bread.stl", BREAD_COLOR, Bread
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return CuttingAction(
            object_to_cut=target.body,
            arm=self.arm,
            tool=tool,
            technique=CuttingTechnique.SLICE,
            number_of_cuts_on_local_x_axis=5,
            slice_thickness=0.07,
            pointer_stride=self.pointer_stride,
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

    def create_target(
        self, world: World, name: str, scale: float
    ) -> Optional[HasRootBody]:
        return self._create_mesh_target(
            world, name, scale, "bowl.stl", BOWL_COLOR, Bowl
        )

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return MixingAction(
            container=target.body,
            arm=self.arm,
            tool=tool,
            pointer_stride=self.pointer_stride,
        )


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

    def create_target(
        self, world: World, name: str, scale: float
    ) -> Optional[HasRootBody]:
        return self._create_mesh_target(
            world, name, scale, "bowl.stl", BOWL_COLOR, Bowl
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

    def create_target(
        self, world: World, name: str, scale: float
    ) -> Optional[HasRootBody]:
        return None

    def build_action(self, target: ExperimentTarget, tool: Tool) -> ActionDescription:
        return WipingAction(
            arm=self.arm,
            tool=tool,
            target_pose=target.pose,
            pointer_stride=self.pointer_stride,
        )


def tasks_by_name() -> Dict[str, Type[ToolTaskDefinition]]:
    """
    :return: Every runnable tool-based task, keyed by its command line name.
    """
    return {
        definition.task_name(): definition
        for definition in recursive_subclasses(ToolTaskDefinition)
    }
