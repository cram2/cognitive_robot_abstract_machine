"""
Tasks that control a geometric relation between a feature on the tip of a kinematic
chain and one on its root, such as a height, a distance or an angle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.weights import VelocityScale
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from giskardpy.motion_statechart.mixins import (
    HasCruiseAngularVelocityParameter,
    HasCruiseVelocityParameter,
    HasKinematicChainParameter,
    HasLowerAngleParameter,
    HasLowerLimitParameter,
    HasReferenceNormalParameter,
    HasReferencePointParameter,
    HasReferenceVectorParameter,
    HasTipNormalParameter,
    HasTipPointParameter,
    HasTipVectorParameter,
    HasUpperAngleParameter,
    HasUpperLimitParameter,
)

# %% goals on geometric features


@dataclass(eq=False, repr=False, kw_only=True)
class FeatureFunctionGoal(ConvergingTask, HasKinematicChainParameter, ABC):
    """
    Base for feature tasks operating on geometric features.

    :attr:`controlled_feature` is fixed on the tip of :attr:`kinematic_chain` and
    :attr:`reference_feature` on its root. Both are transformed into a common frame, and
    debug visualizations are registered for them.
    """

    controlled_feature: Point3 | Vector3 = field(init=False)
    """
    The geometric feature (point or vector) that is being controlled, expressed in the
    tip link frame.
    """

    reference_feature: Point3 | Vector3 = field(init=False)
    """
    The geometric feature (point or vector) that serves as reference, expressed in the
    root link frame.
    """

    @abstractmethod
    def get_controlled_and_reference_features(
        self,
    ) -> tuple[Point3 | Vector3, Point3 | Vector3]:
        """
        Return the controlled and reference features.

        :return: Tuple (controlled_feature, reference_feature), each a Point3 or
            Vector3.
        """
        raise NotImplementedError

    def build(self) -> MotionNodeArtifacts:
        """
        Differs from the inherited ``build`` in also expressing both features relative
        to the root of :attr:`kinematic_chain` and drawing them as debug expressions.
        """
        self.controlled_feature, self.reference_feature = (
            self.get_controlled_and_reference_features()
        )
        feature_debug_expressions: list[DebugExpression] = []
        root_reference_feature = self.context.world.transform(
            target_frame=self.kinematic_chain.root,
            spatial_object=self.reference_feature,
        )
        tip_controlled_feature = self.context.world.transform(
            target_frame=self.kinematic_chain.tip,
            spatial_object=self.controlled_feature,
        )

        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        )
        if isinstance(self.controlled_feature, Point3):
            self.root_P_controlled_feature = root_T_tip @ tip_controlled_feature
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_P_controlled_feature",
                    expression=self.root_P_controlled_feature,
                    color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
                )
            )
        elif isinstance(self.controlled_feature, Vector3):
            self.root_V_controlled_feature = root_T_tip @ tip_controlled_feature
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_V_controlled_feature",
                    expression=self._vector_to_draw(
                        root_V_feature=self.root_V_controlled_feature,
                        feature=self.controlled_feature,
                    ),
                    color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
                )
            )

        if isinstance(self.reference_feature, Point3):
            self.root_P_reference_feature = root_reference_feature
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_P_reference_feature",
                    expression=self.root_P_reference_feature,
                    color=Color.from_rgba(red=0, green=1, blue=0, opacity=1),
                )
            )
        elif isinstance(self.reference_feature, Vector3):
            self.root_V_reference_feature = root_reference_feature
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_V_reference_feature",
                    expression=self._vector_to_draw(
                        root_V_feature=self.root_V_reference_feature,
                        feature=self.reference_feature,
                    ),
                    color=Color.from_rgba(red=0, green=1, blue=0, opacity=1),
                )
            )

        artifacts = super().build()
        artifacts.debug_expressions.extend(feature_debug_expressions)
        return artifacts

    def _vector_to_draw(self, *, root_V_feature: Vector3, feature: Vector3) -> Vector3:
        """
        :param root_V_feature: `feature` expressed in the root of :attr:`kinematic_chain`.
        :param feature: The vector feature as the task was given it.
        :return: A copy of `root_V_feature` drawn from the visualisation frame of
            `feature`, or from the tip of :attr:`kinematic_chain` if it has none.
        """
        vector_to_draw = Vector3.from_iterable(root_V_feature)
        vector_to_draw.visualisation_frame = (
            feature.visualisation_frame
            if feature.visualisation_frame is not None
            else self.kinematic_chain.tip
        )
        return vector_to_draw


@dataclass(eq=False, repr=False, kw_only=True)
class AlignPerpendicular(
    FeatureFunctionGoal,
    HasTipNormalParameter,
    HasReferenceNormalParameter,
    HasCruiseAngularVelocityParameter,
):
    """
    Align two normal vectors to be perpendicular.

    The goal drives the dot product of :attr:`tip_normal` and :attr:`reference_normal`
    to zero, and its error is the absolute dot product.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_normal, self.reference_normal

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that drives the two normals perpendicular.

        :return: The artifacts of this task, whose error is how far the dot product of
            the two normals is from zero.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature @ self.root_V_controlled_feature

        artifacts.constraints.add_equality_constraint(
            cruise_velocity=self.cruise_angular_velocity,
            velocity_scale=VelocityScale.ROTATION,
            equality_bound=0 - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )
        artifacts.error = sm.abs(expr)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class HeightGoal(
    FeatureFunctionGoal,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasLowerLimitParameter,
    HasUpperLimitParameter,
    HasCruiseVelocityParameter,
):
    """
    Moves :attr:`tip_point` to be the specified distance away from
    :attr:`reference_point` along the z-axis of the map frame.
    """

    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_point, self.reference_point

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the height difference within the limits.

        :return: The artifacts of this task, whose error is how far the height
            difference lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        expr = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3.Z()

        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=VelocityScale.TRANSLATION,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.error = sm.max(self.lower_limit - expr, expr - self.upper_limit)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceGoal(
    FeatureFunctionGoal,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasLowerLimitParameter,
    HasUpperLimitParameter,
    HasCruiseVelocityParameter,
):
    """
    Moves :attr:`tip_point` to be the specified distance away from
    :attr:`reference_point` measured in the x-y-plane of the map frame.
    """

    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_point, self.reference_point

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the planar distance within the limits.

        :return: The artifacts of this task, whose error is how far the planar distance
            lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        expr = root_V_diff.norm()

        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=VelocityScale.TRANSLATION,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        # An extra constraint that makes the execution more stable
        for i, axis_name in enumerate(["x", "y", "z"]):
            artifacts.constraints.add_inequality_constraint(
                cruise_velocity=self.cruise_velocity,
                velocity_scale=VelocityScale.TRANSLATION,
                lower_error=0,
                upper_error=0,
                quadratic_weight=self.weight,
                task_expression=root_V_diff[i],
                name=f"{self.name}_extra_{axis_name}",
            )

        artifacts.error = sm.max(self.lower_limit - expr, expr - self.upper_limit)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class ReachPoint(
    FeatureFunctionGoal,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasCruiseVelocityParameter,
):
    """
    Drives a controlled point on the tip onto a reference point, with an error in
    meters.
    """

    def get_controlled_and_reference_features(self) -> tuple[Point3, Point3]:
        return self.tip_point, self.reference_point

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that moves the controlled point onto the reference point.

        :return: The artifacts of this task, whose error is the distance between the two
            points.
        """
        artifacts = MotionNodeArtifacts()
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=self.root_P_controlled_feature,
            frame_P_goal=self.root_P_reference_feature,
            cruise_velocity=self.cruise_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = self.root_P_controlled_feature.euclidean_distance(
            self.root_P_reference_feature
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class AngleGoal(
    FeatureFunctionGoal,
    HasTipVectorParameter,
    HasReferenceVectorParameter,
    HasLowerAngleParameter,
    HasUpperAngleParameter,
    HasCruiseAngularVelocityParameter,
):
    """
    Controls the angle between :attr:`tip_vector` and :attr:`reference_vector` to be
    between `lower_angle` and `upper_angle`.
    """

    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_vector, self.reference_vector

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the angle between the vectors within the limits.

        :return: The artifacts of this task, whose error is how far the angle lies
            outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )

        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_angular_velocity,
            velocity_scale=VelocityScale.ROTATION,
            upper_error=self.upper_angle - expr,
            lower_error=self.lower_angle - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.error = sm.max(self.lower_angle - expr, expr - self.upper_angle)
        return artifacts
