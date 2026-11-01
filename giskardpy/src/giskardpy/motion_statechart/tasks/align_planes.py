from dataclasses import dataclass

from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from semantic_digital_twin.spatial_types import Vector3
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

    reference_velocity: float = 0.5
    """
    Reference angular velocity for normalization in rad/s.
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
        current_dbg = DebugExpression(
            name=f"{self.name}/current_normal",
            expression=root_V_tip_normal,
            color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
        )
        goal_dbg = DebugExpression(
            name=f"{self.name}/goal_normal",
            expression=root_V_root_normal,
            color=Color.from_rgba(red=0, green=0, blue=1, opacity=1),
        )
        artifacts.debug_expressions.append(current_dbg)
        artifacts.debug_expressions.append(goal_dbg)

        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_root_normal,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = root_V_tip_normal.angle_between(root_V_root_normal)
        return artifacts
