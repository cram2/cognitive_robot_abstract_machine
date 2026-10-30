from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field, dataclass

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)


@dataclass(eq=False, repr=False, kw_only=True)
class FeatureFunctionGoal(ConvergingTask, ABC):
    """
    Base for feature tasks operating on geometric features.

    Transforms the controlled feature (from `tip_link`) and the reference feature
    (from `root_link`) into a common frame and registers debug visualizations.
    """

    tip_link: KinematicStructureEntity
    """
    The link where the controlled feature is attached. Defines the moving frame of reference.
    """
    root_link: KinematicStructureEntity
    """
    The static reference link. Defines the fixed frame of reference.
    """
    controlled_feature: Point3 | Vector3 = field(init=False)
    """
    The geometric feature (point or vector) that is being controlled, expressed in the tip link frame.
    """
    reference_feature: Point3 | Vector3 = field(init=False)
    """
    The geometric feature (point or vector) that serves as reference, expressed in the root link frame.
    """

    @abstractmethod
    def get_controlled_and_reference_features(
        self,
    ) -> tuple[Point3 | Vector3, Point3 | Vector3]:
        """
        Return the controlled and reference features.

        :return: Tuple (controlled_feature, reference_feature), each a Point3 or Vector3.
        """
        raise NotImplementedError

    def build(self) -> MotionNodeArtifacts:
        self.controlled_feature, self.reference_feature = (
            self.get_controlled_and_reference_features()
        )
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
class AlignPerpendicular(FeatureFunctionGoal):
    """
    Align two normal vectors to be perpendicular.

    The goal drives the angle between `tip_normal` and `reference_normal` to π/2.
    Completion occurs when |current_angle - π/2| < `threshold`.
    """

    tip_normal: Vector3
    """
    The normal vector to be controlled, defined in the tip link frame.
    """
    reference_normal: Vector3
    """
    The reference normal vector to align against, defined in the root link frame.
    """
    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Priority weight for the alignment constraint in the optimization problem.
    """
    maximum_velocity: float = 0.2
    """
    Maximum allowed angular velocity for the alignment motion in radians per second.
    """
    threshold: float = 0.01
    """
    Tolerance threshold in radians. The goal is considered achieved when the absolute
    difference between the current angle and 90 degrees is below this value.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_normal, self.reference_normal

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that drives the two normals perpendicular.

        :return: The artifacts of this task, whose error is how far the dot product of the two normals is from zero.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature @ self.root_V_controlled_feature

        artifacts.constraints.add_equality_constraint(
            reference_velocity=self.maximum_velocity,
            equality_bound=0 - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )
        artifacts.error = sm.abs(expr)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class HeightGoal(FeatureFunctionGoal):
    """
    Moves the `tip_point` to be the specified distance away from the `reference_point`
    along the z-axis of the map frame.
    """

    tip_point: Point3
    """
    Tip point to be controlled.
    """
    reference_point: Point3
    """
    Reference point to measure the distance against.
    """
    lower_limit: float
    """
    Lower limit to control the distance away from the `reference_point`.
    """
    upper_limit: float
    """
    Upper limit to control the distance away from the `reference_point`.
    """
    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Priority weight for the height constraint in the optimization problem.
    """
    maximum_velocity: float = 0.2
    """
    Maximum allowed velocity for the height motion in meters per second.
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

        :return: The artifacts of this task, whose error is how far the height difference lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        expr = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3.Z()

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.error = sm.max(self.lower_limit - expr, expr - self.upper_limit)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceGoal(FeatureFunctionGoal):
    """
    Moves the `tip_point` to be the specified distance away from the `reference_point`
    measured in the x-y-plane of the map frame.
    """

    tip_point: Point3
    """
    Tip point to be controlled.
    """
    reference_point: Point3
    """
    Reference point to measure the distance against.
    """
    lower_limit: float
    """
    Lower limit to control the distance away from the `reference_point`.
    """
    upper_limit: float
    """
    Upper limit to control the distance away from the `reference_point`.
    """
    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Priority weight for the distance constraint in the optimization problem.
    """
    maximum_velocity: float = 0.2
    """
    Maximum allowed velocity for the distance motion in meters per second.
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

        :return: The artifacts of this task, whose error is how far the planar distance lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        expr = root_V_diff.norm()

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        # An extra constraint that makes the execution more stable
        for i, axis_name in enumerate(["x", "y", "z"]):
            artifacts.constraints.add_inequality_constraint(
                reference_velocity=self.maximum_velocity,
                lower_error=0,
                upper_error=0,
                quadratic_weight=self.weight,
                task_expression=root_V_diff[i],
                name=f"{self.name}_extra_{axis_name}",
            )

        artifacts.error = sm.max(self.lower_limit - expr, expr - self.upper_limit)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class ReachPoint(FeatureFunctionGoal):
    """
    Drives a controlled point on the tip onto a reference point.
    """

    tip_point: Point3
    """
    The point to be controlled, defined in the tip link frame.
    """
    reference_point: Point3
    """
    The target point to reach, defined in the root link frame.
    """
    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Priority weight for the reach constraint in the optimization problem.
    """
    maximum_velocity: float = 0.2
    """
    Maximum allowed velocity for the reaching motion in meters per second.
    """
    threshold: float = 0.01
    """
    Distance at or below which the goal is considered achieved, in meters.
    """

    def get_controlled_and_reference_features(self) -> tuple[Point3, Point3]:
        return self.tip_point, self.reference_point

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that moves the controlled point onto the reference point.

        :return: The artifacts of this task, whose error is the distance between the two points.
        """
        artifacts = MotionNodeArtifacts()
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=self.root_P_controlled_feature,
            frame_P_goal=self.root_P_reference_feature,
            reference_velocity=self.maximum_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = self.root_P_controlled_feature.euclidean_distance(
            self.root_P_reference_feature
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class AngleGoal(FeatureFunctionGoal):
    """
    Controls the angle between the `tip_vector` and the `reference_vector` to be between
    `lower_angle` and `upper_angle`.
    """

    tip_vector: Vector3
    """
    Tip vector to be controlled.
    """
    reference_vector: Vector3
    """
    Reference vector to measure the angle against.
    """
    lower_angle: float
    """
    Lower limit to control the angle between the `tip_vector` and the `reference_vector`.
    """
    upper_angle: float
    """
    Upper limit to control the angle between the `tip_vector` and the `reference_vector`.
    """
    weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Priority weight for the angle constraint in the optimization problem.
    """
    maximum_velocity: float = 0.2
    """
    Maximum allowed angular velocity for the angle motion in radians per second.
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

        :return: The artifacts of this task, whose error is how far the angle lies outside the limits.
        """
        artifacts = MotionNodeArtifacts()
        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_angle - expr,
            lower_error=self.lower_angle - expr,
            quadratic_weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.error = sm.max(self.lower_angle - expr, expr - self.upper_angle)
        return artifacts
