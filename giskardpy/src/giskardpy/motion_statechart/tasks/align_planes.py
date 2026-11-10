"""
Aligning a plane of a tip link with a goal plane.
"""

from dataclasses import dataclass

from giskardpy.motion_statechart.cruise_velocities import DefaultCruiseVelocity
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


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
