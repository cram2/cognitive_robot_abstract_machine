"""
Tasks turning a plane of a link onto a goal plane.
"""

from dataclasses import dataclass, field

from giskardpy.motion_statechart.binding_policy import ForwardKinematicsBinding
from giskardpy.motion_statechart.cruise_velocities import DefaultCruiseVelocity
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% aligning a plane of a link


@dataclass(eq=False, repr=False, kw_only=True)
class AlignPlanes(ConvergingTask):
    """
    Will orient the tip plane to align with the goal plane.

    The planes are represented as normal vectors.
    """

    root_link: KinematicStructureEntity
    """
    Root link of the kinematic chain.
    """

    tip_link: KinematicStructureEntity
    """
    Tip link of the kinematic chain.
    """

    goal_normal: Vector3
    """
    Normal vector of the goal plane.
    """

    tip_normal: Vector3
    """
    Normal vector of the tip plane.
    """

    threshold: float = 0.01
    """
    Angle in radians below which the planes are considered aligned.
    """

    cruise_velocity: float = DefaultCruiseVelocity.ROTATION
    """
    Sets how fast the goal orientation is approached, in rad/s; not strictly enforced.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Priority weight relative to other tasks.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that rotate the tip plane onto the goal plane.

        :return: The artifacts of this task, whose error is the angle between the two
            plane normals.
        """
        artifacts = MotionNodeArtifacts()
        tip_V_tip_normal = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        tip_V_tip_normal.scale(1)
        root_V_root_normal = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).rotation_matrix
        root_V_tip_normal = root_R_tip @ tip_V_tip_normal
        root_V_tip_normal.scale(1)

        root_V_tip_normal.visualisation_frame = self.tip_link
        current_normal_debug_expression = DebugExpression(
            name=f"{self.name}/current_normal",
            expression=root_V_tip_normal,
            color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
        )
        # a copy, since transforming returns the goal normal itself if it is in the root
        root_V_goal_normal_at_tip = Vector3.from_iterable(root_V_root_normal)
        root_V_goal_normal_at_tip.visualisation_frame = self.tip_link
        goal_normal_debug_expression = DebugExpression(
            name=f"{self.name}/goal_normal",
            expression=root_V_goal_normal_at_tip,
            color=Color.from_rgba(red=0, green=0, blue=1, opacity=1),
        )
        artifacts.debug_expressions.append(current_normal_debug_expression)
        artifacts.debug_expressions.append(goal_normal_debug_expression)

        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_root_normal,
            cruise_velocity=self.cruise_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = root_V_tip_normal.angle_between(root_V_root_normal)
        return artifacts


# %% keeping what a gripper holds upright


@dataclass(eq=False, repr=False, kw_only=True)
class KeepHeldBodyUpright(AlignPlanes):
    """
    Turns the body an end effector holds when this task starts until its z-axis points
    along the z-axis of :attr:`root_link`.

    An end effector with no body below its tool frame, as when contact alone carries the
    body, keeps the direction in its tool frame that points up when this task starts.
    """

    end_effector: EndEffector
    """
    The end effector holding the body.
    """

    tip_link: KinematicStructureEntity = field(init=False)
    """
    The tool frame of :attr:`end_effector`.
    """

    goal_normal: Vector3 = field(init=False)
    """
    The z-axis of :attr:`root_link`.
    """

    tip_normal: Vector3 = field(init=False)
    """
    The z-axis of the held body in the tool frame, as bound when this task starts.
    """

    _tool_T_held_body: ForwardKinematicsBinding = field(init=False, repr=False)
    """
    The pose of the held body in the tool frame, bound when this task starts.
    """

    def set_up(self) -> None:
        """
        Expresses the plane to align over the pose this task binds when it starts.
        """
        super().set_up()
        self.tip_link = self.end_effector.tool_frame
        self.goal_normal = Vector3.Z(reference_frame=self.root_link)
        self._tool_T_held_body = ForwardKinematicsBinding(
            name=PrefixedName("tool_T_held_body", str(self.name)),
            root=self.tip_link,
            tip=self.tip_link,
            float_variable_data=self.context.float_variable_data,
        )
        self._tool_T_held_body.bind(self.context.world)
        self.tip_normal = self._tool_T_held_body.root_T_tip @ Vector3.Z(
            reference_frame=self.tip_link
        )

    def on_start(self) -> None:
        """
        Binds the pose of the body held now, or of :attr:`root_link` if none is.
        """
        super().on_start()
        held_body = self.end_effector.held_body
        self._tool_T_held_body.change_tip(
            self.root_link if held_body is None else held_body, self.context.world
        )
