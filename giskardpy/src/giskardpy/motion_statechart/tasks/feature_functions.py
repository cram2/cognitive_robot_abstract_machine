from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.cruise_velocities import DefaultCruiseVelocity
from giskardpy.motion_statechart.weights import VelocityScale
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from giskardpy.motion_statechart.mixins import (
    FeatureType,
    HasFeaturePairParameters,
    HasCruiseVelocityParameter,
    HasRootLinkParameter,
    HasTipLinkParameter,
)


@dataclass(eq=False, repr=False, kw_only=True)
class FeatureFunctionGoal(
    ConvergingTask,
    HasRootLinkParameter,
    HasTipLinkParameter,
    HasFeaturePairParameters[FeatureType],
    HasCruiseVelocityParameter,
    ABC,
):
    """
    Base for feature tasks operating on geometric features.

    :attr:`controlled_feature` is fixed on :attr:`tip_link` and :attr:`reference_feature`
    on :attr:`root_link`. Both are transformed into a common frame, and debug
    visualizations are registered for them.
    """

    def build(self) -> MotionNodeArtifacts:
        feature_debug_expressions: list[DebugExpression] = []
        root_reference_feature = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.reference_feature
        )
        tip_controlled_feature = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.controlled_feature
        )

        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
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
            self.root_V_controlled_feature.visualisation_frame = (
                self.controlled_feature.visualisation_frame
            )
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_V_controlled_feature",
                    expression=self.root_V_controlled_feature,
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
            self.root_V_reference_feature.visualisation_frame = (
                self.reference_feature.visualisation_frame
            )
            feature_debug_expressions.append(
                DebugExpression(
                    name="root_V_reference_feature",
                    expression=self.root_V_reference_feature,
                    color=Color.from_rgba(red=0, green=1, blue=0, opacity=1),
                )
            )

        artifacts = super().build()
        artifacts.debug_expressions.extend(feature_debug_expressions)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class AlignPerpendicular(FeatureFunctionGoal[Vector3]):
    """
    Align two normal vectors to be perpendicular.

    The goal drives the dot product of :attr:`controlled_feature` and
    :attr:`reference_feature` to zero, and its error is the absolute dot product.
    """

    cruise_velocity: float = DefaultCruiseVelocity.ROTATION
    """
    How fast the normals are turned perpendicular, in rad/s; not strictly enforced.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that drives the two normals perpendicular.

        :return: The artifacts of this task, whose error is how far the dot product of the two normals is from zero.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature @ self.root_V_controlled_feature

        artifacts.constraints.add_equality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=VelocityScale.ROTATION,
            equality_bound=0 - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )
        artifacts.error = sm.abs(expr)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class HeightGoal(FeatureFunctionGoal[Point3]):
    """
    Moves :attr:`controlled_feature` to be the specified distance away from
    :attr:`reference_feature`
    along the z-axis of the map frame.
    """

    lower_limit: float
    """
    Lower limit to control the distance away from :attr:`reference_feature`.
    """
    upper_limit: float
    """
    Upper limit to control the distance away from :attr:`reference_feature`.
    """
    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the height difference within the limits.

        :return: The artifacts of this task, whose error is how far the height difference lies outside the limits.
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
class DistanceGoal(FeatureFunctionGoal[Point3]):
    """
    Moves :attr:`controlled_feature` to be the specified distance away from
    :attr:`reference_feature`
    measured in the x-y-plane of the map frame.
    """

    lower_limit: float
    """
    Lower limit to control the distance away from :attr:`reference_feature`.
    """
    upper_limit: float
    """
    Upper limit to control the distance away from :attr:`reference_feature`.
    """
    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the planar distance within the limits.

        :return: The artifacts of this task, whose error is how far the planar distance lies outside the limits.
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
class ReachPoint(FeatureFunctionGoal[Point3]):
    """
    Drives a controlled point on the tip onto a reference point, with an error in
    meters.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that moves the controlled point onto the reference point.

        :return: The artifacts of this task, whose error is the distance between the two points.
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
class AngleGoal(FeatureFunctionGoal[Vector3]):
    """
    Controls the angle between :attr:`controlled_feature` and :attr:`reference_feature` to be between
    `lower_angle` and `upper_angle`.
    """

    lower_angle: float
    """
    Lower limit to control the angle between the two features.
    """
    upper_angle: float
    """
    Upper limit to control the angle between the two features.
    """

    cruise_velocity: float = DefaultCruiseVelocity.ROTATION
    """
    How fast the angle goal is approached, in rad/s; not strictly enforced.
    """
    threshold: float = 0.0
    """
    How far outside the limits still counts as achieved.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that keeps the angle between the vectors within the limits.

        :return: The artifacts of this task, whose error is how far the angle lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )

        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=VelocityScale.ROTATION,
            upper_error=self.upper_angle - expr,
            lower_error=self.lower_angle - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.error = sm.max(self.lower_angle - expr, expr - self.upper_angle)
        return artifacts
