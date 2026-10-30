from __future__ import annotations

from dataclasses import field, dataclass

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False, kw_only=True)
class FeatureFunctionMonitor(EndedByOwner, MotionStatechartNode):
    """
    Base for monitors that compare a controlled feature (attached to ``tip_link``) with a
    reference feature (attached to ``root_link``) expressed in the root link frame.
    """

    tip_link: Body
    """Link the controlled feature is attached to."""
    root_link: Body
    """Link the reference feature is attached to."""
    reference_feature: Point3 | Vector3 = field(init=False)
    """Reference feature, typically the target for `controlled_feature`."""
    controlled_feature: Point3 | Vector3 = field(init=False)
    """Controlled feature, typically moved towards `reference_feature`."""

    def build_artifacts(self) -> NodeArtifacts:
        """
        Transform the controlled and reference features into the root link frame and store them on
        the node for use by subclasses.
        """
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
        elif isinstance(self.controlled_feature, Vector3):
            self.root_V_controlled_feature = root_T_tip @ tip_controlled_feature

        if isinstance(self.reference_feature, Point3):
            self.root_P_reference_feature = root_reference_feature
        if isinstance(self.reference_feature, Vector3):
            self.root_V_reference_feature = root_reference_feature

        return NodeArtifacts()


@dataclass(eq=False, repr=False, kw_only=True)
class HeightMonitor(FeatureFunctionMonitor):
    """
    Observes ``True`` while the height difference between the controlled and reference point along
    the root z-axis is within ``[lower_limit, upper_limit]``.
    """

    reference_point: Point3
    """Reference point to measure the height against."""
    tip_point: Point3
    """Controlled point whose height is checked."""
    lower_limit: float
    """Lower bound for the height difference in meters."""
    upper_limit: float
    """Upper bound for the height difference in meters."""

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        artifacts = super().build_artifacts()

        distance = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3.Z()
        artifacts.observation = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class PerpendicularMonitor(FeatureFunctionMonitor):
    """
    Observes ``True`` while the controlled normal is perpendicular to the reference normal within
    ``threshold``.
    """

    reference_normal: Vector3
    """Reference normal to check perpendicularity against."""
    tip_normal: Vector3
    """Controlled normal."""
    threshold: float = 0.01
    """Threshold on the dot product of the two normals."""

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_normal
        self.controlled_feature = self.tip_normal
        artifacts = super().build_artifacts()

        expr = self.root_V_reference_feature[:3] @ self.root_V_controlled_feature[:3]
        artifacts.observation = sm.abs(expr) <= self.threshold
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceMonitor(FeatureFunctionMonitor):
    """
    Observes ``True`` while the planar distance (ignoring the z-axis) between the controlled and
    reference point is within ``[lower_limit, upper_limit]``.
    """

    reference_point: Point3
    """Reference point to measure the distance against."""
    tip_point: Point3
    """Controlled point whose distance is checked."""
    lower_limit: float
    """Lower bound for the distance in meters."""
    upper_limit: float
    """Upper bound for the distance in meters."""

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        artifacts = super().build_artifacts()

        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        distance = root_V_diff.norm()
        artifacts.observation = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class AngleMonitor(FeatureFunctionMonitor):
    """
    Observes ``True`` while the angle between the controlled and reference vector is within
    ``[lower_angle, upper_angle]``.
    """

    reference_vector: Vector3
    """Reference vector to measure the angle against."""
    tip_vector: Vector3
    """Controlled vector."""
    lower_angle: float
    """Lower bound for the angle in radians."""
    upper_angle: float
    """Upper bound for the angle in radians."""

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_vector
        self.controlled_feature = self.tip_vector
        artifacts = super().build_artifacts()

        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )
        artifacts.observation = sm.logic_and(
            expr > self.lower_angle, expr < self.upper_angle
        )
        return artifacts
