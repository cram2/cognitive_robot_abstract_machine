from __future__ import division

from dataclasses import field, dataclass
from typing import Union
from abc import ABC, abstractmethod

import krrood.symbolic_math.symbolic_math as sm

from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task, NodeArtifacts, DebugExpression
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class FeatureFunctionGoal(Task, ABC):
    """
    Base for feature tasks operating on geometric features.

    Transforms the controlled feature (from `tip_link`) and the reference feature
    (from `root_link`) into a common frame and registers debug visualizations.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    The link where the controlled feature is attached. Defines the moving frame of reference.
    """
    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    The static reference link. Defines the fixed frame of reference.
    """
    controlled_feature: Union[Point3, Vector3] = field(init=False)
    """
    The geometric feature (point or vector) that is being controlled, expressed in the tip link frame.
    """
    reference_feature: Union[Point3, Vector3] = field(init=False)
    """
    The geometric feature (point or vector) that serves as reference, expressed in the root link frame.
    """

    @abstractmethod
    def get_controlled_and_reference_features(
        self,
    ) -> tuple[Union[Point3, Vector3], Union[Point3, Vector3]]:
        """
        Return the controlled and reference features.

        :return: Tuple (controlled_feature, reference_feature), each a Point3 or Vector3.
        """
        raise NotImplementedError

    def build(self, context: BuildContext) -> NodeArtifacts:
        self.controlled_feature, self.reference_feature = (
            self.get_controlled_and_reference_features()
        )
        artifacts = NodeArtifacts()
        root_reference_feature = context.world.transform(
            target_frame=self.root_link, spatial_object=self.reference_feature
        )
        tip_controlled_feature = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.controlled_feature
        )

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        if isinstance(self.controlled_feature, Point3):
            self.root_P_controlled_feature = root_T_tip @ tip_controlled_feature
            dbg = DebugExpression(
                name="root_P_controlled_feature",
                expression=self.root_P_controlled_feature,
                color=Color(1, 0, 0, 1),
            )
            artifacts.debug_expressions.append(dbg)
        elif isinstance(self.controlled_feature, Vector3):
            self.root_V_controlled_feature = root_T_tip @ tip_controlled_feature
            self.root_V_controlled_feature.vis_frame = (
                self.controlled_feature.visualisation_frame
            )
            dbg = DebugExpression(
                name="root_V_controlled_feature",
                expression=self.root_V_controlled_feature,
                color=Color(1, 0, 0, 1),
            )
            artifacts.debug_expressions.append(dbg)

        if isinstance(self.reference_feature, Point3):
            self.root_P_reference_feature = root_reference_feature
            dbg = DebugExpression(
                name="root_P_reference_feature",
                expression=self.root_P_reference_feature,
                color=Color(0, 1, 0, 1),
            )
            artifacts.debug_expressions.append(dbg)
        elif isinstance(self.reference_feature, Vector3):
            self.root_V_reference_feature = root_reference_feature
            self.root_V_reference_feature.vis_frame = (
                self.reference_feature.visualisation_frame
            )
            dbg = DebugExpression(
                name="root_V_reference_feature",
                expression=self.root_V_reference_feature,
                color=Color(0, 1, 0, 1),
            )
            artifacts.debug_expressions.append(dbg)

        return artifacts


@dataclass(eq=False, repr=False)
class AlignPerpendicular(FeatureFunctionGoal):
    """
    Align two normal vectors to be perpendicular.

    The goal drives the angle between `tip_normal` and `reference_normal` to π/2.
    Completion occurs when |current_angle - π/2| < `threshold`.
    """

    tip_normal: Vector3 = field(kw_only=True)
    """
    The normal vector to be controlled, defined in the tip link frame.
    """
    reference_normal: Vector3 = field(kw_only=True)
    """
    The reference normal vector to align against, defined in the root link frame.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the alignment constraint in the optimization problem.
    """
    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed angular velocity for the alignment motion in radians per second.
    """
    threshold: float = field(default=0.01, kw_only=True)
    """
    Tolerance threshold in radians. The goal is considered achieved when the absolute
    difference between the current angle and 90 degrees is below this value.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_normal, self.reference_normal

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = super().build(context)

        expr = self.root_V_reference_feature @ self.root_V_controlled_feature

        artifacts.constraints.add_equality_constraint(
            reference_velocity=self.maximum_velocity,
            equality_bound=0 - expr,
            weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )
        artifacts.observation = sm.abs(0 - expr) < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class HeightGoal(FeatureFunctionGoal):
    """
    Moves the `tip_point` to be the specified distance away from the `reference_point`
    along the z-axis of the map frame.
    """

    tip_point: Point3 = field(kw_only=True)
    """
    Tip point to be controlled.
    """
    reference_point: Point3 = field(kw_only=True)
    """
    Reference point to measure the distance against.
    """
    lower_limit: float = field(kw_only=True)
    """
    Lower limit to control the distance away from the `reference_point`.
    """
    upper_limit: float = field(kw_only=True)
    """
    Upper limit to control the distance away from the `reference_point`.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the height constraint in the optimization problem.
    """
    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed velocity for the height motion in meters per second.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_point, self.reference_point

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = super().build(context)

        expr = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3.Z()

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.observation = sm.logic_and(
            sm.if_less_eq(expr, self.upper_limit, 1, 0),
            sm.if_greater_eq(expr, self.lower_limit, 1, 0),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class DistanceGoal(FeatureFunctionGoal):
    """
    Moves the `tip_point` to be the specified distance away from the `reference_point`
    measured in the x-y-plane of the map frame.
    """

    tip_point: Point3 = field(kw_only=True)
    """
    Tip point to be controlled.
    """
    reference_point: Point3 = field(kw_only=True)
    """
    Reference point to measure the distance against.
    """
    lower_limit: float = field(kw_only=True)
    """
    Lower limit to control the distance away from the `reference_point`.
    """
    upper_limit: float = field(kw_only=True)
    """
    Upper limit to control the distance away from the `reference_point`.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the distance constraint in the optimization problem.
    """
    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed velocity for the distance motion in meters per second.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_point, self.reference_point

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = super().build(context)

        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        expr = root_V_diff.norm()

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_limit - expr,
            lower_error=self.lower_limit - expr,
            weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        # An extra constraint that makes the execution more stable
        for i, axis_name in enumerate(["x", "y", "z"]):
            artifacts.constraints.add_inequality_constraint(
                reference_velocity=self.maximum_velocity,
                lower_error=0,
                upper_error=0,
                weight=self.weight,
                task_expression=root_V_diff[i],
                name=f"{self.name}_extra_{axis_name}",
            )

        artifacts.observation = sm.logic_and(
            sm.if_less_eq(expr, self.upper_limit, sm.Scalar(1), sm.Scalar(0)),
            sm.if_greater_eq(expr, self.lower_limit, sm.Scalar(1), sm.Scalar(0)),
        )

        return artifacts


@dataclass(eq=False, repr=False)
class AngleGoal(FeatureFunctionGoal):
    """
    Controls the angle between the `tip_vector` and the `reference_vector` to be between
    `lower_angle` and `upper_angle`.
    """

    tip_vector: Vector3 = field(kw_only=True)
    """
    Tip vector to be controlled.
    """
    reference_vector: Vector3 = field(kw_only=True)
    """
    Reference vector to measure the angle against.
    """
    lower_angle: float = field(kw_only=True)
    """
    Lower limit to control the angle between the `tip_vector` and the `reference_vector`.
    """
    upper_angle: float = field(kw_only=True)
    """
    Upper limit to control the angle between the `tip_vector` and the `reference_vector`.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the angle constraint in the optimization problem.
    """
    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed angular velocity for the angle motion in radians per second.
    """

    def get_controlled_and_reference_features(self):
        return self.tip_vector, self.reference_vector

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = super().build(context)

        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            upper_error=self.upper_angle - expr,
            lower_error=self.lower_angle - expr,
            weight=self.weight,
            task_expression=expr,
            name=f"{self.name}_constraint",
        )

        artifacts.observation = sm.logic_and(
            sm.if_less_eq(expr, self.upper_angle, 1, 0),
            sm.if_greater_eq(expr, self.lower_angle, 1, 0),
        )

        return artifacts
